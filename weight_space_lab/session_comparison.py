"""One frozen, development-only baseline/candidate comparison.

This trusted controller reuses the checkpoint verifier and forward-only native
worker. It offers no search, grading, training, arbitrary paths, or tools to an
editor. Its output contains private development text and must not be published.
The caller must keep this output and the ledger outside the editor's boundary.
"""

from __future__ import annotations

import copy
from pathlib import Path
import re
import time
import uuid

from .editor_budget import BudgetCancelled, BudgetError, BudgetExhausted, BudgetLedger
from .editor_contract import validate_development
from .editor_controller import EditorController
from .editor_tools import run_checkpoint_action
from .native_runner import _Process, _WorkerFailure, _sync_directory, _write_json
from .protocol import canonical_hash


def _receive(worker, expected, active):
    while True:
        active()
        if worker.connection.poll(0.025):
            kind, value = worker.connection.recv()
            active()
            if kind != expected or not isinstance(value, dict):
                raise ValueError("comparison worker failed or returned invalid output")
            return value
        if not worker.process.is_alive():
            raise ValueError("comparison worker exited")


def _inactive_status(ledger, request_id, check_active):
    try:
        if check_active is not None:
            check_active()
        ledger.check_active(request_id)
    except BudgetCancelled:
        return "cancelled"
    except BudgetExhausted:
        return "timed_out"
    except BudgetError:
        return "failed"
    return "failed"


def run_development_comparison(*, original_model, candidate, policy, development,
                               probe, ledger, output, expected_original_sha256,
                               check_active=None):
    """Run one sequence per arm, using one item, seed, backend and token ceiling.

    Both probe reservations are durable before any worker starts. The supplied
    ``timeout_seconds`` is a shared comparison deadline, not a fresh allowance
    for each arm. Each reservation must fit that full bound, so the ledger needs
    twice that elapsed capacity; measured elapsed is reconciled per arm.
    ``check_active`` may additionally enforce the caller's persisted whole-attempt
    deadline; it is checked before dispatch and throughout all worker waits.

    An existing output is never resumed or replaced, even after interruption.
    Unknown usage retains the reservation; recovery uses BudgetLedger's existing
    stopped-executor protocol and never reruns this comparison. There are no
    retries. The caller freezes an attempt ledger with two probes/two sequences.

    This function measures local target-model inference only. It does not call a
    frontier model, enforce provider generation caps, or infer provider billing.
    """
    if not isinstance(ledger, BudgetLedger):
        raise ValueError("comparison requires the existing durable budget ledger")
    if check_active is not None and not callable(check_active):
        raise ValueError("comparison activity check must be callable")
    development = validate_development(development)
    spec = EditorController._probe_spec(probe)
    if len(development) != 1 or spec is None or len(spec["seeds"]) != 1:
        raise ValueError("comparison requires exactly one development item and seed")
    if (type(expected_original_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_original_sha256) is None):
        raise ValueError("comparison requires a frozen original checkpoint SHA256")
    # Canonical JSON validation also prevents mutable caller references entering
    # the worker after the manifest was frozen.
    policy = copy.deepcopy(policy)
    canonical_hash(policy)
    original_model, candidate, output = map(Path, (original_model, candidate, output))
    if (output.exists() or output.is_symlink()
            or any(path.is_symlink() for path in output.parents)):
        raise ValueError("comparison output must be a new regular private directory; no rerun")
    resolved_output = output.resolve()
    if any(resolved_output == model.resolve() or model.resolve() in resolved_output.parents
           for model in (original_model, candidate)):
        raise ValueError("comparison evidence may not modify checkpoint directories")
    if not original_model.is_dir() or not candidate.is_dir():
        raise ValueError("comparison requires existing original and candidate checkpoints")
    ledger.check_active()
    if check_active is not None:
        check_active()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _sync_directory(output.parent)
    arms = [(name, uuid.uuid4().hex) for name in ("baseline", "candidate")]
    manifest = {
        "schema_version": 1, "kind": "session_development_comparison",
        "ledger_configuration_sha256": ledger.configuration_sha256,
        "original_sha256": expected_original_sha256, "policy": policy,
        "development": development, "probe": spec,
        "sequence_counts": {"baseline": 1, "candidate": 1},
        "max_total_generated_tokens": 2 * spec["max_new_tokens"],
        "shared_timeout_seconds": spec["timeout_seconds"],
        "retry_policy": "never", "arms": dict(arms),
        "scientific_result": False, "private_development_evidence": True,
    }
    manifest_digest = canonical_hash(manifest)
    _write_json(output / "manifest.json", manifest, immutable=True)
    reserved, reports = {}, []
    shared_deadline = None
    started = time.monotonic()

    def active():
        if check_active is not None:
            check_active()
        ledger.check_active()
        for request_id in reserved:
            ledger.check_active(request_id)
        if shared_deadline is not None and time.monotonic() >= shared_deadline:
            raise BudgetExhausted("shared development comparison deadline expired")

    def terminal(name, request_id, *, status, dispatched, tokens, elapsed,
                 outcome=None, metadata=None, reason=None):
        record = {"arm": name, "request_id": request_id, "status": status,
                  "dispatched": dispatched, "split": "development",
                  "item_id": development[0]["item_id"], "seed": spec["seeds"][0],
                  "generated_tokens": tokens, "elapsed_seconds": elapsed,
                  "outcome": outcome, "backend_metadata": metadata, "reason": reason}
        # Publish evidence first. A crash before ledger reconciliation retains
        # the full reservation; it cannot authorize duplicate generation.
        _write_json(output / f"{name}.json", record, immutable=True)
        ledger.finish(request_id, status=status,
                      completed_sequences=int(status == "completed"),
                      failed_sequences=int(status != "completed"),
                      generated_tokens=tokens, elapsed_seconds=elapsed, billed_units=0)
        del reserved[request_id]
        reports.append(record)

    try:
        for name, request_id in arms:
            reserved[request_id] = ledger.reserve(
                request_id, kind="probe", sequences=1,
                max_generated_tokens=spec["max_new_tokens"],
                timeout_seconds=spec["timeout_seconds"], max_billed_units=0)
        shared_deadline = min(row["deadline_monotonic"] for row in reserved.values())
        _write_json(output / "reservations.json", list(reserved.values()), immutable=True)
        # Reject an invalid candidate before either target-model generation.
        for is_candidate in (False, True):
            verified = run_checkpoint_action(
                model=original_model, policy=policy,
                exposure_log=output / "verification-exposures.jsonl", output=candidate,
                tool="verify", arguments={"candidate": is_candidate},
                timeout_seconds=max(0.001, shared_deadline - time.monotonic()),
                check_active=active, expected_original_sha256=expected_original_sha256)
            if verified != {"verified": True, "original_sha256": expected_original_sha256}:
                raise ValueError("checkpoint verification returned unexpected identity")
        _write_json(output / "verification.json", {
            "original_verified": True, "candidate_verified": True,
            "original_sha256": expected_original_sha256}, immutable=True)
        preparation_elapsed = time.monotonic() - started
        for name, request_id in arms:
            worker, outcome, metadata, tokens = None, None, None, None
            arm_started = time.monotonic()
            status, dispatched = "failed", False
            try:
                active()
                ledger.mark_dispatched(request_id)
                dispatched = True
                worker = _Process(spec["backend"], original_model if name == "baseline" else candidate / "model")
                metadata = {"startup": _receive(worker, "ready", active)}
                active()
                worker.connection.send(("generate", development[0]["messages"], spec["seeds"][0]))
                result = _receive(worker, "result", active)
                if set(result) != {"outcome", "metadata"} or type(result["outcome"]) is not dict:
                    raise ValueError("comparison worker result has invalid shape")
                outcome, metadata["generation"] = result["outcome"], result["metadata"]
                count = outcome.get("token_counts", {}).get("output")
                if type(count) is int and count >= 0:
                    tokens = count  # Retain overage as measured evidence; never clip it.
                if tokens is None or tokens > spec["max_new_tokens"]:
                    raise ValueError("target generation exceeded or omitted its frozen token bound")
                if outcome.get("status") not in {"ok", "truncated"} or type(outcome.get("raw_output")) is not str:
                    raise ValueError("target generation failed")
                active()
                status = "completed"
            except (ValueError, OSError, EOFError, RuntimeError, _WorkerFailure):
                status = _inactive_status(ledger, request_id, check_active)
            finally:
                # Do not terminalize if executor shutdown cannot be confirmed.
                # Parent-death cleanup and ledger recovery remain conservative.
                if worker is not None:
                    worker.close()
            terminal(name, request_id, status=status, dispatched=dispatched,
                     tokens=tokens, elapsed=time.monotonic() - arm_started + (
                         preparation_elapsed if name == "baseline" else 0),
                     outcome=outcome, metadata=metadata,
                     reason=None if status == "completed" else "forward_probe_failed")
            if status != "completed":
                break
    except (ValueError, OSError, EOFError, RuntimeError, _WorkerFailure):
        # No automatic retries, including failures before both reservations fit.
        # A dispatched arm left pending means shutdown was not established.
        if any(row["status"] == "dispatched" and row["request_id"] in reserved
               for row in ledger.attempts()):
            raise
    for name, request_id in arms:
        if request_id in reserved:
            terminal(name, request_id, status=_inactive_status(ledger, request_id, check_active),
                     dispatched=False, tokens=0, elapsed=(
                         time.monotonic() - started if not reports else 0),
                     reason="comparison_stopped_before_dispatch")
    report = {"schema_version": 1, "kind": "session_development_comparison_result",
              "manifest_sha256": manifest_digest,
              "status": "completed" if len(reports) == 2 and all(
                  row["status"] == "completed" for row in reports) else "failed",
              "evidence_kind": "fixture" if spec["backend"]["name"] == "fixture" else "real_model_development",
              "scientific_result": False, "private_development_evidence": True,
              "arms": reports, "ledger": ledger.summary()}
    _write_json(output / "result.json", report, immutable=True)
    return report
