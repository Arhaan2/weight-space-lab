"""Development-only editor broker. The model receives data, never host tools.

The local operator/broker is trusted. Untrusted editor output is a single
strict JSON action. Offline imports are not authenticated model executions.
No live provider adapter is registered until identity, isolation, metering,
and cancellation can all be enforced by that provider's execution interface.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import time
import uuid

from .checkpoint import CheckpointEditor
from .editor_budget import BudgetCancelled, BudgetError, BudgetExhausted, BudgetLedger
from .editor_contract import read_json_bounded, validate_development, validate_request, validate_response
from .inventory import file_sha256
from .native_runner import _write_json
from .protocol import canonical_hash

TOOLS = ["model_info", "development", "inspect", "propose", "probe", "stop"]
_ID = re.compile(r"[0-9a-f]{32}\Z")


def _identity(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("request/candidate ID must be 32 hexadecimal characters")
    return value


def _implementation():
    # Include transitive validators/loaders too: changing an inventory or numeric
    # helper changes the experiment even if the controller file is unchanged.
    return {"files": {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            "python": platform.python_version(), "numpy": importlib.metadata.version("numpy")}


def _positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be a positive bounded integer")


class EditorController:
    """Persistent trusted broker; construct only from operator-controlled storage.

    One directory is one budget identity. Reopening requires its existing ledger
    and exact frozen implementation/assets. A new directory is a new study, not
    a resume. Editor processes cannot read or write any of these host files.
    """

    session_kind = "editor_integration_session"

    @classmethod
    def create(cls, root, *, model, policy, development, limits, interface_observation,
               probe=None, run_id=None):
        root, model = Path(root), Path(model).resolve()
        if root.exists() or root.is_symlink():
            raise ValueError("session creation requires a new output directory")
        if model == root.resolve() or model in root.resolve().parents:
            raise ValueError("session must be outside the immutable original")
        development = validate_development(development)
        if not isinstance(interface_observation, dict):
            raise ValueError("record the observed interface provenance")
        probe = cls._probe_spec(probe)
        root.mkdir(parents=True, mode=0o700)
        # An incomplete initialization remains fail-closed; never auto-recreate it.
        for name in ("requests", "responses", "feedback", "actions", "candidates"):
            (root / name).mkdir(mode=0o700)
        editor = CheckpointEditor(model, policy, exposure_log=root / "exposures.jsonl")
        info = {"checkpoint_sha256": editor.original_sha256,
                "architectures": editor.inventory["model_config"].get("architectures", []),
                "stored_parameter_count": editor.inventory["parameter_count"],
                "allowed_tensors": policy["allowed_tensors"]}
        manifest = {"schema_version": 1, "kind": "editor_integration_session",
                    "run_id": run_id or uuid.uuid4().hex,
                    "mode": "offline_integration", "model_path": str(model),
                    "model_info": info, "policy": copy.deepcopy(policy),
                    "development": development, "limits": copy.deepcopy(limits),
                    "probe": probe, "interface_observation": copy.deepcopy(interface_observation),
                    "implementation": _implementation(), "scientific_result": False,
                    "experiment_executed": False, "live_editor_status": "pending_enforced_adapter"}
        _write_json(root / "session.json", manifest, immutable=True)
        ledger = BudgetLedger(root / "budget.sqlite3", run_id=manifest["run_id"],
                              limits=limits, binding={"session_sha256": canonical_hash(manifest)}, create=True)
        ledger.close()
        # A durable ready marker distinguishes creation failures from a resumable run.
        _write_json(root / "ready.json", {"session_sha256": canonical_hash(manifest)}, immutable=True)
        return cls(root)

    @staticmethod
    def _probe_spec(probe):
        if probe is None:
            return None
        if not isinstance(probe, dict) or set(probe) != {"backend", "seeds", "max_new_tokens", "timeout_seconds"}:
            raise ValueError("probe configuration must freeze backend, seeds, token and wall-time limits")
        seeds = probe["seeds"]
        if (not isinstance(seeds, list) or not seeds or len(seeds) > 8
                or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("probe seeds must be an explicit unique bounded list")
        _positive(probe["max_new_tokens"], "probe token limit", 256)
        _positive(probe["timeout_seconds"], "probe timeout", 600)
        backend = probe["backend"]
        if isinstance(backend, dict) and backend.get("name") == "fixture":
            from .native_runner import validate_fixture_spec
            validate_fixture_spec(backend)
        else:
            from .native_backend import validate_backend_spec
            validate_backend_spec(backend)
            # This broker exposes only the existing checkpoint-preserving CPU inference path.
            if backend["name"] != "transformers" or backend["sampling"]["max_new_tokens"] != probe["max_new_tokens"]:
                raise ValueError("probe requires frozen transformers forward-only limits")
        return copy.deepcopy(probe)

    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("session directory cannot be a symlink")
        self.manifest = read_json_bounded(self.root / "session.json", max_bytes=1_048_576)
        if self.manifest.get("kind") != self.session_kind:
            raise ValueError("session protocol requires its own controller; modes cannot be mixed")
        self.digest = canonical_hash(self.manifest)
        if read_json_bounded(self.root / "ready.json") != {"session_sha256": self.digest}:
            raise ValueError("session initialization is incomplete or changed")
        if self.manifest["implementation"] != _implementation():
            raise ValueError("frozen implementation changed; do not resume with different code")
        self.ledger = BudgetLedger(self.root / "budget.sqlite3", run_id=self.manifest["run_id"],
                                  limits=self.manifest["limits"], binding={"session_sha256": self.digest})
        self._editor = None

    @property
    def editor(self):
        """Trusted inspection convenience; dispatch work uses bounded child workers."""
        if self._editor is None:
            self._editor = CheckpointEditor(self.manifest["model_path"], self.manifest["policy"],
                                            exposure_log=self.root / "exposures.jsonl")
            if self._editor.original_sha256 != self.manifest["model_info"]["checkpoint_sha256"]:
                raise ValueError("immutable checkpoint changed")
        return self._editor

    @contextmanager
    def _lock(self):
        fd = os.open(self.root / ".controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("controller already executing; cancellation uses its separate ledger") from None
            yield
        finally:
            os.close(fd)

    def status(self):
        return {"mode": "offline_integration", "scientific_result": False, "experiment_executed": False,
                "live_editor_status": self.manifest["live_editor_status"], "session_sha256": self.digest,
                "budget": self.ledger.summary(), "attempts": self.ledger.attempts()}

    def cancel(self):
        self.ledger.cancel()
        return self.status()

    def recover(self):
        with self._lock():
            # A crashed controller's sandbox guardian kills its process group. Native
            # workers independently watch their parent. Recovery never replays actions.
            recovered = self.ledger.recover_interrupted(executor_stopped=True)
            return {"recovered": recovered, **self.status()}

    def export_request(self, instruction, *, max_output_tokens=512, timeout_seconds=60, retry_of=None,
                       feedback_id=None):
        with self._lock():
            return self._export(instruction, max_output_tokens, timeout_seconds, retry_of, feedback_id)

    def _export(self, instruction, max_output_tokens, timeout_seconds, retry_of, feedback_id):
        request_id = uuid.uuid4().hex
        feedback = None
        if feedback_id is not None:
            feedback = read_json_bounded(self.root / "feedback" / f"{_identity(feedback_id)}.json")
        request = validate_request({"schema_version": 1, "kind": "editor_request", "request_id": request_id,
            "session_sha256": self.digest, "mode": "offline_integration", "instruction": instruction,
            "approved_context": {"model_info": self.manifest["model_info"],
                                 "development": self.manifest["development"],
                                 "policy": self.manifest["policy"], "feedback": feedback},
            "limits": {"max_output_tokens": max_output_tokens, "timeout_seconds": timeout_seconds},
            "allowed_tools": TOOLS})
        # Reserve durably BEFORE publishing a request to any transport.
        self.ledger.reserve(request_id, kind="editor", sequences=1, max_generated_tokens=max_output_tokens,
                            timeout_seconds=timeout_seconds, retry_of=retry_of,
                            max_billed_units=None if self.manifest["limits"]["billed_units"] is None else 0)
        _write_json(self.root / "requests" / f"{request_id}.json", request, immutable=True)
        self.ledger.mark_dispatched(request_id)
        return request

    def import_response(self, response):
        with self._lock():
            return self._accept(response, simulated=False)

    def _accept(self, response, *, simulated, started=None):
        request_id = _identity(response.get("request_id") if isinstance(response, dict) else None)
        request = read_json_bounded(self.root / "requests" / f"{request_id}.json")
        try:
            row = self.ledger.check_active(request_id)
            response = validate_response(response, request)
            if not simulated and response["evidence_kind"] != "unverified_file_handoff":
                raise ValueError("file imports must be labeled unverified_file_handoff")
            _write_json(self.root / "responses" / f"{request_id}.json", response, immutable=True)
        except (ValueError, OSError, TypeError):
            self._fail_pending(request_id)
            raise
        # Unauthenticated usage cannot release reservations. Fixture execution has
        # no generated model tokens and incurs no provider bill. External claims
        # stay in the raw response, separate from trusted metering.
        try:
            feedback = self._execute_action(request_id, response)
            failure = self._inactive_status(request_id) if response["action"]["tool"] != "stop" else None
            self.ledger.finish(request_id, status=failure or "completed",
                               completed_sequences=0 if failure else 1, failed_sequences=1 if failure else 0,
                               generated_tokens=0 if simulated else None,
                               elapsed_seconds=time.monotonic() - started if started is not None else None,
                               billed_units=0 if simulated else None)
            return feedback
        except BaseException:
            self._fail_pending(request_id)
            raise

    def _fail_pending(self, request_id):
        # Terminalize only after a worker has stopped, even when check_active
        # correctly rejects an expired/cancelled reservation. Never overwrite a result.
        row = next((r for r in self.ledger.attempts() if r["request_id"] == request_id), None)
        if row is not None and row["status"] in {"reserved", "dispatched"}:
            self.ledger.finish(request_id, status=self._inactive_status(request_id) or "failed", completed_sequences=0,
                               failed_sequences=row["requested_sequences"])

    def simulate(self, *, tool="model_info", arguments=None, timeout_seconds=10, max_output_tokens=512):
        """Exercise the enforced process boundary; not a language-model call."""
        from .editor_sandbox import SandboxRunner
        with self._lock():
            started = time.monotonic()
            request = self._export("Non-scientific integration fixture: return the declared action.",
                                   max_output_tokens, timeout_seconds, None, None)
            request_id = request["request_id"]
            payload = {"request": request, "request_sha256": canonical_hash(request),
                       "action": {"tool": tool, "arguments": arguments or {}}}
            try:
                response = SandboxRunner().run(_SIMULATOR, payload, timeout_seconds=timeout_seconds,
                    cancelled=lambda: self._cancelled(request_id))
                return self._accept(response, simulated=True, started=started)
            except BaseException:
                # Preserve unknown usage at the full reservation even for a crash
                # between receiving output and committing the terminal event.
                self._fail_pending(request_id)
                raise

    def _cancelled(self, request_id):
        return self._inactive_status(request_id) is not None

    def _inactive_status(self, request_id):
        try:
            self.ledger.check_active(request_id)
            return None
        except BudgetCancelled:
            return "cancelled"
        except BudgetExhausted:
            return "timed_out"
        except BudgetError:
            return "failed"

    def _execute_action(self, request_id, response):
        action = response["action"]
        # At-most-once action intent: never replay after interrupted publication.
        _write_json(self.root / "actions" / f"{request_id}.json",
                    {"response_sha256": canonical_hash(response), "action": action}, immutable=True)
        feedback = {"request_id": request_id, "scientific_result": False,
                    "experiment_executed": False, "editor_identity_authenticated": False,
                    "evidence_kind": response["evidence_kind"], "tool": action["tool"]}
        try:
            row = self.ledger.check_active(request_id)
            tool, args = action["tool"], action["arguments"]
            if tool == "model_info":
                result = self.manifest["model_info"]
            elif tool == "development":
                result = self.manifest["development"]
            elif tool in {"inspect", "propose"}:
                from .editor_tools import run_checkpoint_action
                if tool == "propose" and args["proposal"].get("provenance") != "editor_proposal":
                    raise ValueError("proposal provenance must be editor_proposal; response identity remains unverified")
                result = run_checkpoint_action(model=self.manifest["model_path"], policy=self.manifest["policy"],
                    exposure_log=self.root / "exposures.jsonl", output=self.root / "candidates" / request_id,
                    tool=tool, arguments=args, timeout_seconds=max(0.001, row["deadline_at"] - time.time()),
                    check_active=lambda: self.ledger.check_active(request_id),
                    expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"])
                if tool == "propose":
                    result["candidate_id"] = request_id
            elif tool == "probe":
                result = self._probe(request_id, args)
            else:  # strict contract permits only stop here
                self.ledger.cancel()
                result = {"stopped": True}
            feedback.update(status="ok", result=result)
        except (ValueError, OSError, RuntimeError) as error:
            # No traceback, paths, raw provider exception, or evaluator output crosses the broker.
            feedback.update(status="rejected", error="action_rejected", error_type=type(error).__name__)
        _write_json(self.root / "feedback" / f"{request_id}.json", feedback, immutable=True)
        return feedback

    def _probe(self, editor_request_id, arguments):
        from .native_runner import _Process, _WorkerFailure
        spec = self.manifest["probe"]
        if spec is None:
            raise ValueError("forward-pass probes are not enabled in this session")
        indexed = {row["item_id"]: row for row in self.manifest["development"]}
        if any(item not in indexed for item in arguments["task_ids"]):
            raise ValueError("only approved development IDs can be probed")
        candidate = arguments["candidate_id"]
        if candidate == "base":
            model = Path(self.manifest["model_path"])
            bundle = model
        else:
            candidate = _identity(candidate)
            proposal_response = read_json_bounded(self.root / "responses" / f"{candidate}.json")
            candidate_status = next((r["status"] for r in self.ledger.attempts() if r["request_id"] == candidate), None)
            accepted = read_json_bounded(self.root / "feedback" / f"{candidate}.json")
            if (proposal_response["action"]["tool"] != "propose" or candidate_status != "completed"
                    or accepted["status"] != "ok"):
                raise ValueError("candidate was not proposed in this session")
            bundle = self.root / "candidates" / candidate
            model = bundle / "model"
        plan = [(item, seed) for item in arguments["task_ids"] for seed in spec["seeds"]]
        probe_id = uuid.uuid4().hex
        self.ledger.reserve(probe_id, kind="probe", sequences=len(plan),
            max_generated_tokens=len(plan) * spec["max_new_tokens"], timeout_seconds=spec["timeout_seconds"],
            max_billed_units=None if self.manifest["limits"]["billed_units"] is None else 0)
        self.ledger.mark_dispatched(probe_id)
        start, completed, tokens, outputs = time.monotonic(), 0, 0, []
        worker = None
        status = "failed"
        try:
            from .editor_tools import run_checkpoint_action
            def active():
                self.ledger.check_active(editor_request_id)
                self.ledger.check_active(probe_id)
            run_checkpoint_action(model=self.manifest["model_path"], policy=self.manifest["policy"],
                exposure_log=self.root / "exposures.jsonl", output=bundle, tool="verify",
                arguments={"candidate": candidate != "base"}, timeout_seconds=spec["timeout_seconds"],
                check_active=active, expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"])
            active()
            worker = _Process(spec["backend"], model)

            def receive(expected):
                while True:
                    self.ledger.check_active(editor_request_id)
                    self.ledger.check_active(probe_id)
                    if time.monotonic() - start >= spec["timeout_seconds"]:
                        raise ValueError("probe deadline exceeded")
                    if worker.connection.poll(0.025):
                        kind, value = worker.connection.recv()
                        if kind != expected:
                            raise ValueError("probe worker failed")
                        return value
                    if not worker.process.is_alive():
                        raise ValueError("probe worker exited")

            metadata = receive("ready")
            for item, seed in plan:
                self.ledger.check_active(probe_id)
                worker.connection.send(("generate", indexed[item]["messages"], seed))
                generated = receive("result")["outcome"]
                count = generated["token_counts"]["output"]
                if type(count) is not int or not 0 <= count <= spec["max_new_tokens"]:
                    raise ValueError("probe exceeded frozen token bound")
                if generated["status"] not in {"ok", "truncated"}:
                    raise ValueError("probe generation failed")
                tokens += count
                completed += 1
                outputs.append({"item_id": item, "seed": seed, "status": generated["status"],
                    "raw_output": generated["raw_output"], "generated_tokens": count})
            status = "completed"
        except (ValueError, OSError, EOFError, RuntimeError, _WorkerFailure):
            status = self._inactive_status(probe_id) or self._inactive_status(editor_request_id) or "failed"
        finally:
            if worker is not None:
                worker.close()
            elapsed = time.monotonic() - start
            self.ledger.finish(probe_id, status=status, completed_sequences=completed,
                failed_sequences=len(plan) - completed, generated_tokens=tokens if status == "completed" else None,
                elapsed_seconds=elapsed, billed_units=0)
        result = {"probe_id": probe_id, "editor_request_id": editor_request_id,
                  "status": status, "split": "development", "candidate_id": candidate,
                  "evidence_kind": "fixture" if spec["backend"]["name"] == "fixture" else "real_model_development",
                  "outputs": outputs, "scientific_result": False, "experiment_executed": False}
        _write_json(self.root / "feedback" / f"{probe_id}.json", result, immutable=True)
        return result

    def dispatch_live(self, *args, **kwargs):
        from .editor_live import require_live_ready
        # Use the same evidence-bearing gate as the operator command. Neither
        # arbitrary kwargs nor a response's claimed model name bypass it.
        require_live_ready(root=getattr(self, "root", None))


_SIMULATOR = '''import json, sys
p = json.load(sys.stdin)
r = {"schema_version": 1, "kind": "editor_response", "request_id": p["request"]["request_id"],
     "request_sha256": p["request_sha256"], "evidence_kind": "simulated",
     "model_identity": {"provider": None, "model": None, "version": None, "interface": "isolated_python_fixture"},
     "action": p["action"], "usage": {"generated_tokens": 0, "billed_units": 0}, "scientific_result": False}
print(json.dumps(r))
'''
