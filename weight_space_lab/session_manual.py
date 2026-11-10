"""Explicit exploratory manual intake using an existing unspent session budget.

This runner never starts, authenticates, supervises or cancels a frontier session.
It accepts one externally supplied data object. Isolation remains unverified.
The original manifest/ledger binding survive an append-only protocol amendment.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import time

from .editor_budget import BudgetLedger
from .editor_contract import read_json_bounded, read_text_bounded
from .editor_controller import _implementation
from .editor_tools import run_checkpoint_action
from .native_runner import _write_json
from .protocol import canonical_hash
from .session_controller import SessionController, session_readiness
from .session_protocol import make_session_request, validate_session_protocol, validate_session_response


def _utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z")


class ManualSessionController(SessionController):
    """Manual exploratory extension of authenticated_session_v1, not research."""

    def _open_original(self, root):
        self.root = Path(root)
        self.manifest = read_json_bounded(self.root / "session.json", max_bytes=1_048_576)
        self.digest = canonical_hash(self.manifest)
        if (self.manifest.get("kind") != self.session_kind
                or read_json_bounded(self.root / "ready.json") != {"session_sha256": self.digest}):
            raise ValueError("original session identity changed or is incomplete")
        self.ledger = BudgetLedger(self.root / "budget.sqlite3", run_id=self.manifest["run_id"],
            limits=self.manifest["limits"], binding={"session_sha256": self.digest})
        self._editor = None
        self.manual_root = self.root / "manual"

    @classmethod
    def enable(cls, root, *, operator_model="Astra", operator_mode="Ultra", submission_hours=48):
        """Append an explicit amendment; never recreate or rebind the ledger."""
        if (operator_model, operator_mode, submission_hours) != ("Astra", "Ultra", 48):
            raise ValueError("this authorized manual attempt selects Astra / Ultra with a 48-hour submission window")
        self = cls.__new__(cls)
        self._open_original(root)
        try:
            with self._lock():
                self.ledger.check_active()
                if self.ledger.attempts() or any((self.root / "requests").iterdir()):
                    raise ValueError("manual amendment requires the existing unspent, unexported attempt")
                protocol = copy.deepcopy(self.manifest["protocol"])
                if protocol["limits"] != {"target_max_new_tokens": 128, "target_timeout_seconds": 300,
                        "local_elapsed_seconds": 900, "frontier_provider_token_cap": None,
                        "frontier_provider_token_cap_status": "unavailable_subscription_route"}:
                    raise ValueError("manual amendment must preserve the authorized 128-token development limits")
                protocol["editor"]["requested_effort"] = "ultra"
                validate_session_protocol(protocol)
                # Exclusive directory creation prevents renewal, replacement or
                # a second packet after partial preparation or expiry.
                self.manual_root.mkdir(mode=0o700)
                inspections = []
                for name, info in self.manifest["model_info"]["allowed_tensors"].items():
                    args = {"op": "statistics", "tensor": name,
                            "slices": [[0, min(32, size)] for size in info["shape"]]}
                    result = run_checkpoint_action(model=self.manifest["model_path"], policy=self.manifest["policy"],
                        exposure_log=self.root / "exposures.jsonl", output=self.manual_root / "inspection",
                        tool="inspect", arguments=args, timeout_seconds=60,
                        check_active=self.ledger.check_active,
                        expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"])
                    inspections.append({"request": args, "result": result})
                prepared = read_json_bounded(self.root / "prepared-request.json")
                now = time.time()
                amendment = {
                    "schema_version": 1, "extension": "manual_exploratory_v1",
                    "parent_variant": "authenticated_session_v1",
                    "scope": "exploratory_workflow_validation", "editor_isolation": "unverified",
                    "controlled_research_run": False, "frontier_launched_by_runner": False,
                    "source_session_sha256": self.digest,
                    "source_prepared_request_sha256": canonical_hash(prepared),
                    "ledger_configuration_sha256": self.ledger.configuration_sha256,
                    "implementation": _implementation(), "protocol": protocol,
                    "created_at": now, "submission_expires_at": now + 48 * 3600,
                    "submission_expires_at_utc": _utc(now + 48 * 3600),
                    "operator_selection": {"model": operator_model, "mode": operator_mode,
                        "requested_selector": "gpt-6-astra", "requested_effort": "ultra",
                        "source": "operator_instruction"},
                    "runtime_provenance": {"accepted_selector": None, "model": None, "effort": None,
                        "immutable_model_version": None, "usage": None},
                    "accounting": "single_external_submission_reserved_at_import_before_parsing_or_local_execution",
                    "submission_expiry_cancels_remote_generation": False,
                    "inspections": inspections,
                }
                _write_json(self.manual_root / "amendment.json", amendment, immutable=True)
                request = make_session_request(request_id=prepared["request_id"],
                    session_sha256=canonical_hash(amendment), protocol=protocol,
                    instruction=("Exploratory manual workflow validation; editor isolation is unverified. "
                        "Propose one bounded weight edit using only this approved packet. No training, "
                        "harness/grader modification, held-out evaluation, extra candidate or retry. "
                        "Inspection evidence is bound by the manual session digest. Return only the exact JSON response."),
                    approved_context={"model_info": self.manifest["model_info"], "policy": self.manifest["policy"],
                        "development": self.manifest["development"], "feedback": None})
                _write_json(self.manual_root / "request.json", request, immutable=True)
                from .session_packet import build_manual_packet
                packet = build_manual_packet(request=request, protocol=protocol, inspections=inspections,
                    operator_model=operator_model, operator_mode=operator_mode,
                    submission_expires_at_utc=amendment["submission_expires_at_utc"])
                # Atomic private publication, using the existing durable writer.
                _write_json(self.manual_root / "packet.json", {"text": packet}, immutable=True)
                packet_path = self.manual_root / "editor-packet.txt"
                with packet_path.open("x", encoding="utf-8") as stream:
                    import os
                    stream.write(packet)
                    stream.flush()
                    os.fsync(stream.fileno())
                packet_path.chmod(0o600)
                _write_json(self.manual_root / "ready.json", {"amendment_sha256": canonical_hash(amendment),
                    "request_sha256": canonical_hash(request), "packet_sha256": canonical_hash(packet)}, immutable=True)
        finally:
            self.ledger.close()
        return cls(root)

    def __init__(self, root):
        self._open_original(root)
        try:
            self.amendment = read_json_bounded(self.manual_root / "amendment.json", max_bytes=1_048_576)
            self.request = read_json_bounded(self.manual_root / "request.json")
            packet = read_json_bounded(self.manual_root / "packet.json")["text"]
            if read_text_bounded(self.manual_root / "editor-packet.txt") != packet:
                raise ValueError("delivered manual packet differs from its frozen content")
            ready = read_json_bounded(self.manual_root / "ready.json")
            if ready != {"amendment_sha256": canonical_hash(self.amendment),
                         "request_sha256": canonical_hash(self.request), "packet_sha256": canonical_hash(packet)}:
                raise ValueError("manual handoff changed or preparation is incomplete")
            if (self.amendment["source_session_sha256"] != self.digest
                    or self.amendment["ledger_configuration_sha256"] != self.ledger.configuration_sha256
                    or self.amendment["implementation"] != _implementation()
                    or self.request["session_sha256"] != canonical_hash(self.amendment)
                    or self.amendment["source_prepared_request_sha256"] != canonical_hash(
                        read_json_bounded(self.root / "prepared-request.json"))):
                raise ValueError("manual amendment differs from its frozen implementation or original budget")
        except BaseException:
            self.ledger.close()
            raise

    def status(self):
        return {"mode": "authenticated_session_v1", "extension": "manual_exploratory_v1",
                "scope": "exploratory_workflow_validation", "editor_isolation": "unverified",
                "controlled_research_run": False, "scientific_result": False,
                "operator_selection": self.amendment["operator_selection"],
                "runtime_provenance": self.amendment["runtime_provenance"],
                "submission_expires_at_utc": self.amendment["submission_expires_at_utc"],
                "budget_scope": self.manifest["budget_scope"], "budget": self.ledger.summary(),
                "attempts": self.ledger.attempts(), "strict_session_readiness": session_readiness()}

    def import_response_file(self, path):
        """Consume the sole external submission, including invalid/expired data."""
        if not Path(path).exists():
            raise ValueError("response file is missing; no submission accepted")
        with self._lock():
            request_id = self.request["request_id"]
            started = time.monotonic()
            row = self.ledger.reserve(request_id, kind="editor", sequences=1, max_generated_tokens=0,
                timeout_seconds=self.amendment["protocol"]["limits"]["local_elapsed_seconds"])
            _write_json(self.manual_root / "submission-intent.json", {
                "request_sha256": canonical_hash(self.request), "reserved_at": row["reserved_at"],
                "accounting_scope": "external_proposal_ingestion_and_local_validation",
                "remote_dispatch": False}, immutable=True)
            # Dispatched here means local ingestion started, never frontier dispatch.
            self.ledger.mark_dispatched(request_id)
            result = {"status": "invalid", "scope": "exploratory_workflow_validation",
                      "editor_isolation": "unverified", "controlled_research_run": False,
                      "candidate_validated": False, "scientific_result": False}
            stage = "submission_validation"
            try:
                if not self.amendment["created_at"] <= time.time() < self.amendment["submission_expires_at"]:
                    result["status"] = "expired"
                    raise ValueError("manual submission expired; this does not cancel remote generation")
                response = read_json_bounded(path)
                response = validate_session_response(response, self.request, self.amendment["protocol"])
                if any(value is not None for value in response["usage"].values()):
                    raise ValueError("manual model-authored usage must be null; runtime usage is unavailable")
                from .checkpoint import _proposal
                _proposal(response["action"]["arguments"]["proposal"], self.manifest["policy"],
                          {"tensors": self.manifest["model_info"]["allowed_tensors"]},
                          self.manifest["model_info"]["checkpoint_sha256"])
                _write_json(self.manual_root / "response.json", response, immutable=True)
                stage = "checkpoint_validation_and_execution"
                audit = run_checkpoint_action(model=self.manifest["model_path"], policy=self.manifest["policy"],
                    exposure_log=self.root / "exposures.jsonl", output=self.root / "candidates" / request_id,
                    tool="propose", arguments=response["action"]["arguments"],
                    timeout_seconds=max(0.001, row["deadline_at"] - time.time()),
                    check_active=lambda: self.ledger.check_active(request_id),
                    expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"])
                self.ledger.check_active(request_id)
                _write_json(self.manual_root / "candidate-audit.json", audit, immutable=True)
                result.update(status="accepted", candidate_validated=True,
                    result_sha256=audit["result_sha256"], changed_stored_entries=audit["changed_stored_entries"],
                    zero_effective_change=audit["changed_stored_entries"] == 0)
            except (ValueError, OSError, RuntimeError, OverflowError) as error:
                result.update(error_type=type(error).__name__, error=str(error), failure_stage=stage)
                if stage == "checkpoint_validation_and_execution":
                    result["status"] = "failed"
                if result["status"] != "expired" and self._inactive_status(request_id):
                    result["status"] = self._inactive_status(request_id)
            accepted = result["status"] == "accepted"
            self.ledger.finish(request_id, status="completed" if accepted else (
                result["status"] if result["status"] in {"cancelled", "timed_out"} else "failed"),
                completed_sequences=int(accepted), failed_sequences=int(not accepted),
                generated_tokens=0, elapsed_seconds=time.monotonic() - started, billed_units=None)
            _write_json(self.manual_root / "import-result.json", result, immutable=True)
            return {**result, "budget": self.ledger.summary()}

    def compare(self):
        """Exactly one paired development run; all raw feedback stays private."""
        with self._lock():
            accepted = read_json_bounded(self.manual_root / "import-result.json")
            row = next(r for r in self.ledger.attempts() if r["request_id"] == self.request["request_id"])
            if accepted["status"] != "accepted" or row["status"] != "completed":
                raise ValueError("only the accepted single manual candidate can be compared")
            self._attempt_active(row)
            from .session_comparison import run_development_comparison
            comparison = run_development_comparison(original_model=self.manifest["model_path"],
                candidate=self.root / "candidates" / self.request["request_id"], policy=self.manifest["policy"],
                development=self.manifest["development"], probe=self.manifest["probe"], ledger=self.ledger,
                output=self.manual_root / "comparison",
                expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"],
                check_active=lambda: self._attempt_active(row))
            result = {**accepted, "status": comparison["status"],
                "comparison_evidence_kind": comparison["evidence_kind"],
                "operator_selection": self.amendment["operator_selection"],
                "runtime_provenance": self.amendment["runtime_provenance"],
                "arms": comparison["arms"], "budget": self.ledger.summary(),
                "final_evaluation_executed": False, "benchmark_improvement_established": False}
            _write_json(self.manual_root / "result.json", result, immutable=True)
            return result

    def cancel(self):
        self.ledger.cancel()
        return {**self.status(), "remote_cancellation_attempted": False}

    def recover(self):
        with self._lock():
            # No remote executor belongs to this runner. Unknown local work keeps
            # full reservations until existing durable deadlines expire.
            recovered = self.ledger.recover_interrupted(executor_stopped=False)
            return {**self.status(), "recovered": recovered, "remote_cancellation_attempted": False}
