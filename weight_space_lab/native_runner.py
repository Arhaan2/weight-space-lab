"""Bounded native generation with an immutable, hash-chained attempt journal.

The journal is authoritative. ``records.json`` is a recoverable Phase 1 view;
its absence never causes a completed generation to be repeated. No grades or
scientific scores are computed here. Heavy inference packages load only in a
separate, killable worker process.
"""

from __future__ import annotations

import copy
import fcntl
import json
import math
import multiprocessing
import os
import re
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .inventory import read_json
from .protocol import canonical_hash, validate_benchmark_schedule
from .records import validate_run_record, validate_run_records

_OUTCOME = {"status", "raw_output", "finish_reason", "token_counts", "duration_seconds", "error"}
_LIMITS = {"request_timeout_seconds", "startup_timeout_seconds", "max_duration_seconds", "max_sequences"}


def _number(value, *, positive=False):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def validate_fixture_spec(spec):
    """Validate the synthetic test backend; it never loads or scores a model."""
    if (not isinstance(spec, dict) or set(spec) != {"name", "behavior", "delay_seconds"}
            or spec["name"] != "fixture"
            or spec["behavior"] not in {"success", "error", "error_once", "timeout", "crash"}
            or not _number(spec["delay_seconds"])):
        raise ValueError("invalid synthetic fixture backend specification")
    return copy.deepcopy(spec)


def _failure(code, *, timeout=False, duration=None, retryable=True):
    return {"status": "timeout" if timeout else "error", "raw_output": None,
            "finish_reason": "timeout" if timeout else "error",
            "token_counts": dict.fromkeys(("input", "output", "total")),
            "duration_seconds": duration, "error": {"code": code, "retryable": retryable}}


def _missing():
    return {"status": "missing", "raw_output": None, "finish_reason": "not_started",
            "token_counts": dict.fromkeys(("input", "output", "total")),
            "duration_seconds": None, "error": None}


def _eligible(record):
    if not record["attempts"]:
        return True
    policy = record["retry_policy"]
    return (policy["mode"] == "retry_errors" and len(record["attempts"]) < policy["max_attempts"]
            and record["status"] in {"error", "timeout"} and record["error"]["retryable"])


def _inputs(configuration, tasks, schedule):
    validate_benchmark_schedule(schedule)
    if not isinstance(configuration, dict):
        raise ValueError("execution configuration must be an object")
    required = {"run_id", "evidence_kind", "profile", "stage", "checkpoint", "retry_policy", "execution_limits", "backend"}
    if not required <= configuration.keys():
        raise ValueError("execution configuration is incomplete")
    limits = configuration["execution_limits"]
    if not isinstance(limits, dict) or set(limits) != _LIMITS:
        raise ValueError("execution limits must be explicit")
    if type(limits["max_sequences"]) is not int or limits["max_sequences"] < 1:
        raise ValueError("max_sequences must be a positive integer")
    if any(not _number(limits[key], positive=True) for key in _LIMITS - {"max_sequences"}):
        raise ValueError("execution time limits must be finite and positive")
    backend = configuration["backend"]
    if not isinstance(backend, dict) or backend.get("name") not in {"fixture", "transformers", "vllm"}:
        raise ValueError("unsupported backend")
    expected_kind = "fixture" if backend["name"] == "fixture" else "real_model"
    if configuration["evidence_kind"] != expected_kind:
        raise ValueError("backend and evidence kind disagree")
    if expected_kind == "fixture":
        validate_fixture_spec(backend)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty array")
    indexed = {}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"benchmark", "item_id", "split", "dataset_revision", "messages"}:
            raise ValueError("task has missing or unknown fields")
        identity = task["benchmark"], task["item_id"]
        if identity in indexed:
            raise ValueError("duplicate task identity")
        indexed[identity] = task
        messages = task["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("task messages must be a nonempty array")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise ValueError("only explicit text chat messages are supported")
    if set(indexed) != {(row["benchmark"], row["item_id"]) for row in schedule}:
        raise ValueError("tasks must match the predetermined schedule exactly")
    frozen = {"configuration": configuration, "tasks": tasks, "schedule": schedule}
    digest = canonical_hash(frozen)  # Includes execution order, unlike the analysis schedule digest.
    manifest = {"schema_version": 1, "kind": "native_schedule", **copy.deepcopy(frozen), "sha256": digest}
    records = []
    for row in schedule:
        task = indexed[row["benchmark"], row["item_id"]]
        record = {"schema_version": 1, "record_type": "run",
                  **{key: copy.deepcopy(configuration[key]) for key in
                     ("run_id", "evidence_kind", "profile", "stage", "checkpoint", "retry_policy")},
                  **row, "dataset_revision": task["dataset_revision"], "split": task["split"],
                  "configuration_sha256": digest, "attempts": [], **_missing()}
        validate_run_record(record)
        records.append(record)
    return manifest, records, indexed


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value, *, immutable=False):
    """Commit complete bytes, then durably commit the directory entry."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # Atomic create-if-absent: a journal event is never replaced.
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(output):
    if output.is_symlink():
        raise ValueError("run output must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("run is already locked by another writer") from None
        yield
    finally:
        os.close(descriptor)


class _Journal:
    def __init__(self, output, manifest, records, *, resume):
        self.output = output
        self.manifest = manifest
        self.records = records
        self.events = []
        self.pending = None
        self.consumed_seconds = 0.0
        self.attempt_count = 0
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("run already exists; explicit resume is required")
            if manifest_path.is_symlink() or read_json(manifest_path) != manifest:
                raise ValueError("incompatible resume: frozen configuration, tasks or schedule changed")
        else:
            if resume:
                raise ValueError("cannot resume a run without a frozen manifest")
            if set(p.name for p in output.iterdir()) - {".lock"}:
                raise ValueError("new run output contains unexpected files")
            _write_json(manifest_path, manifest, immutable=True)
        event_dir = output / "events"
        if event_dir.is_symlink():
            raise ValueError("event journal must be a regular directory")
        event_dir.mkdir(exist_ok=True)
        _sync_directory(output)
        paths = sorted(event_dir.iterdir())
        for path in paths:
            # A crash before atomic publication can leave only an incomplete temp file.
            if path.name.startswith(".pending-") and path.is_file() and not path.is_symlink():
                continue
            if path.name != f"{len(self.events):08d}.json" or path.is_symlink():
                raise ValueError("journal contains missing or unexpected event files")
            envelope = read_json(path)
            if (not isinstance(envelope, dict) or set(envelope) != {"event", "sha256"}
                    or canonical_hash(envelope["event"]) != envelope["sha256"]):
                raise ValueError("journal event hash mismatch")
            self._apply(envelope)
        progress_path = output / "progress.json"
        if progress_path.exists():
            if progress_path.is_symlink():
                raise ValueError("progress must be a regular file")
            progress = read_json(progress_path)
            if not isinstance(progress, dict) or set(progress) != {"manifest_sha256", "event_count", "event_sha256"}:
                raise ValueError("invalid progress anchor")
            count = progress["event_count"]
            if type(count) is not int or not 0 <= count <= len(self.events):
                raise ValueError("journal lost committed progress")
            expected_hash = self.events[count - 1]["sha256"] if count else manifest["sha256"]
            if progress != {"manifest_sha256": manifest["sha256"], "event_count": count, "event_sha256": expected_hash}:
                raise ValueError("journal disagrees with committed progress")
        self.export()

    def _apply(self, envelope):
        event = envelope["event"]
        header = {"sequence", "previous_sha256", "kind", "schedule_index", "attempt_index"}
        if not isinstance(event, dict) or not header <= event.keys():
            raise ValueError("invalid journal event")
        previous = self.events[-1]["sha256"] if self.events else self.manifest["sha256"]
        if type(event["sequence"]) is not int or event["sequence"] != len(self.events) or event["previous_sha256"] != previous:
            raise ValueError("journal event chain is inconsistent")
        index, attempt = event["schedule_index"], event["attempt_index"]
        if type(index) is not int or not 0 <= index < len(self.records) or type(attempt) is not int:
            raise ValueError("journal sample identity is invalid")
        record = self.records[index]
        if event["kind"] == "attempt_started":
            if set(event) != header or self.pending is not None or attempt != len(record["attempts"]) or not _eligible(record):
                raise ValueError("invalid attempt start or retry")
            previous_indices = [entry["event"]["schedule_index"] for entry in self.events]
            if previous_indices and index < previous_indices[-1]:
                raise ValueError("journal execution order changed")
            self.pending = index, attempt
            self.attempt_count += 1
        elif event["kind"] == "attempt_finished":
            if set(event) != header | {"outcome", "metadata", "elapsed_seconds"} or self.pending != (index, attempt):
                raise ValueError("completion has no matching attempt start")
            if (not isinstance(event["outcome"], dict) or set(event["outcome"]) != _OUTCOME
                    or not isinstance(event["metadata"], dict) or not _number(event["elapsed_seconds"])):
                raise ValueError("invalid attempt evidence")
            record.update(copy.deepcopy(event["outcome"]))
            record["attempts"].append({"attempt_index": attempt, **copy.deepcopy(event["outcome"])})
            validate_run_record(record)
            self.consumed_seconds += event["elapsed_seconds"]
            self.pending = None
        else:
            raise ValueError("unknown journal event kind")
        self.events.append(envelope)

    def append(self, kind, index, attempt, **payload):
        event = {"sequence": len(self.events),
                 "previous_sha256": self.events[-1]["sha256"] if self.events else self.manifest["sha256"],
                 "kind": kind, "schedule_index": index, "attempt_index": attempt, **payload}
        envelope = {"event": event, "sha256": canonical_hash(event)}
        # Validate before publishing, without mutating the in-memory state on failure.
        trial = copy.copy(self)
        trial.records = copy.deepcopy(self.records)
        trial.events = list(self.events)
        trial._apply(envelope)
        _write_json(self.output / "events" / f"{len(self.events):08d}.json", envelope, immutable=True)
        self._apply(envelope)
        self.export()

    def export(self):
        validate_run_records(self.records)
        _write_json(self.output / "progress.json", {
            "manifest_sha256": self.manifest["sha256"], "event_count": len(self.events),
            "event_sha256": self.events[-1]["sha256"] if self.events else self.manifest["sha256"]})
        _write_json(self.output / "records.json", self.records)


class _FixtureBackend:
    def __init__(self, spec):
        self.spec = validate_fixture_spec(spec)
        self.calls = 0
        self.metadata = {"backend": "fixture", "synthetic": True,
                         "scientific_scores": False, "load_duration_seconds": 0.0,
                         "measurement_scope": "test process only; tokens and text are synthetic"}

    def generate(self, messages, seed):
        self.calls += 1
        started = time.monotonic()
        behavior = self.spec["behavior"]
        if behavior == "crash":
            os._exit(73)
        time.sleep(self.spec["delay_seconds"])
        if behavior == "error" or behavior == "error_once" and self.calls == 1:
            outcome = _failure("synthetic_failure", duration=time.monotonic() - started)
        else:
            outcome = {"status": "ok", "raw_output": "SYNTHETIC FIXTURE OUTPUT",
                       "finish_reason": "eos", "token_counts": {"input": 2, "output": 3, "total": 5},
                       "duration_seconds": time.monotonic() - started, "error": None}
        return {"outcome": outcome, "metadata": {"synthetic": True, "scientific_scores": False}}


def _safe_error_code(error, default):
    code = getattr(error, "code", None)
    return code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else default


def _worker(connection, spec, model_path, parent_pid):
    # A dedicated process group also contains runtime-spawned descendants.
    os.setsid()
    def parent_watchdog():
        while True:
            time.sleep(0.2)
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGKILL)
    threading.Thread(target=parent_watchdog, daemon=True).start()
    try:
        if spec["name"] == "fixture":
            backend = _FixtureBackend(spec)
        else:
            from .native_backend import create_backend
            backend = create_backend(spec, model_path)
        connection.send(("ready", backend.metadata))
    except Exception as error:
        connection.send(("failure", _safe_error_code(error, "backend_initialization_failed")))
        connection.close()
        return
    try:
        while True:
            command = connection.recv()
            if command[0] == "stop":
                return
            try:
                result = backend.generate(command[1], command[2])
                connection.send(("result", result))
            except Exception as error:
                connection.send(("failure", _safe_error_code(error, "generation_failed")))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class _WorkerFailure(Exception):
    def __init__(self, code, *, timeout=False):
        self.code, self.timeout = code, timeout


class _Process:
    def __init__(self, spec, model_path):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, spec, model_path, os.getpid()))
        self.process.start()
        child.close()
        self.metadata = {}
        self.closed = False

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _WorkerFailure("request_timeout", timeout=True)
            if self.connection.poll(min(0.05, remaining)):
                try:
                    kind, value = self.connection.recv()
                except (EOFError, OSError):
                    raise _WorkerFailure("worker_exited") from None
                if kind == "failure":
                    raise _WorkerFailure(value)
                return kind, value
            if not self.process.is_alive():
                raise _WorkerFailure("worker_exited")

    def ready(self, timeout):
        try:
            kind, value = self.receive(timeout)
        except _WorkerFailure as error:
            if error.timeout:
                raise _WorkerFailure("startup_timeout", timeout=True) from None
            raise
        if kind != "ready" or not isinstance(value, dict):
            raise _WorkerFailure("invalid_worker_message")
        self.metadata = value

    def generate(self, messages, seed, timeout):
        try:
            self.connection.send(("generate", messages, seed))
        except (BrokenPipeError, OSError):
            raise _WorkerFailure("worker_exited") from None
        kind, value = self.receive(timeout)
        if kind != "result" or not isinstance(value, dict) or set(value) != {"outcome", "metadata"}:
            raise _WorkerFailure("invalid_worker_message")
        return value

    def close(self):
        if self.closed:
            return
        if self.process.pid is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                if self.process.is_alive():
                    self.process.terminate()
            self.process.join(timeout=0.25)
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                if self.process.is_alive():
                    self.process.kill()
            self.process.join(timeout=1.0)
        self.connection.close()
        if self.process.is_alive():
            raise RuntimeError("worker could not be stopped")
        self.process.close()
        self.closed = True


def run_schedule(configuration, tasks, schedule, output, *, model_path=None, resume=False):
    """Execute a predetermined schedule, keeping every attempt and missing row.

    Native configuration and model assets are reverified on every invocation,
    including resume. Local model paths
    are operational arguments, never part of public evidence. Retry and sequence
    limits include failed attempts. Wall-time limits include backend startup.
    Unmeasurable interrupted attempts conservatively reserve both configured
    timeouts, preventing repeated crashes from resetting the resource budget.
    """
    manifest, records, indexed = _inputs(configuration, tasks, schedule)
    if configuration["evidence_kind"] == "real_model" and model_path is None:
        raise ValueError("real execution requires a verified local checkpoint")
    if configuration["evidence_kind"] == "real_model":
        from .native_assets import validate_execution_config, verify_execution_assets
        validate_execution_config(configuration, tasks, schedule)
        verify_execution_assets(configuration, model_path)
    output = Path(output)
    limits = configuration["execution_limits"]
    worker = None
    with _lock(output):
        journal = _Journal(output, manifest, records, resume=resume)
        if journal.pending is not None:
            index, attempt = journal.pending
            journal.append("attempt_finished", index, attempt,
                           outcome=_failure("interrupted"),
                           metadata={"duration_unknown": True, "budget_basis": "reserved_startup_plus_request_timeout"},
                           elapsed_seconds=limits["startup_timeout_seconds"] + limits["request_timeout_seconds"])
        try:
            for index, record in enumerate(journal.records):
                while _eligible(record):
                    remaining = limits["max_duration_seconds"] - journal.consumed_seconds
                    if journal.attempt_count >= limits["max_sequences"] or remaining <= 0:
                        break
                    attempt = len(record["attempts"])
                    journal.append("attempt_started", index, attempt)
                    started = time.monotonic()
                    metadata = {}
                    try:
                        if worker is None:
                            worker = _Process(configuration["backend"], str(model_path) if model_path is not None else None)
                            worker.ready(min(limits["startup_timeout_seconds"], remaining))
                        metadata["worker"] = copy.deepcopy(worker.metadata)
                        request_remaining = remaining - (time.monotonic() - started)
                        if request_remaining <= 0:
                            raise _WorkerFailure("run_budget_exhausted", timeout=True)
                        task = indexed[record["benchmark"], record["item_id"]]
                        result = worker.generate(task["messages"], record["seed"],
                                                 min(limits["request_timeout_seconds"], request_remaining))
                        outcome = result["outcome"]
                        metadata["generation"] = result["metadata"]
                        checked = copy.deepcopy(record)
                        checked.update(outcome)
                        checked["attempts"].append({"attempt_index": attempt, **outcome})
                        validate_run_record(checked)
                        canonical_hash(metadata)
                    except _WorkerFailure as error:
                        outcome = _failure(error.code, timeout=error.timeout, duration=time.monotonic() - started,
                                           retryable=error.code != "run_budget_exhausted")
                        if worker is not None:
                            worker.close()
                            worker = None
                    except (ValueError, TypeError, KeyError):
                        outcome = _failure("invalid_backend_result", duration=time.monotonic() - started, retryable=False)
                        metadata = {}
                        if worker is not None:
                            worker.close()
                            worker = None
                    except BaseException:
                        # The durable start is deliberately left unfinished. Resume
                        # records an interrupted attempt before considering a retry.
                        raise
                    journal.append("attempt_finished", index, attempt, outcome=outcome,
                                   metadata=metadata, elapsed_seconds=time.monotonic() - started)
        finally:
            if worker is not None:
                worker.close()
        statuses = {status: sum(record["status"] == status for record in journal.records)
                    for status in ("ok", "truncated", "error", "timeout", "missing")}
        return {"schema_version": 1, "kind": "native_execution_summary",
                "configuration_sha256": manifest["sha256"], "evidence_kind": configuration["evidence_kind"],
                "scheduled_sequences": len(journal.records), "attempted_sequences": journal.attempt_count,
                "statuses": statuses, "accounted_wall_seconds": journal.consumed_seconds,
                "scientific_scores": False, "experiment_executed": False}
