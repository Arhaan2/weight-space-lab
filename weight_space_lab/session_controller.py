"""Single-candidate subscription-session broker; no nested provider calls.

The production session transport is deliberately unregistered: the inspected
desktop interface cannot bind an editor launch to a verified capability boundary.
Files and model-authored assertions cannot register or authenticate a transport.
The complete proposal/checkpoint/comparison path is exercised with synthetic
transport fixtures in tests, without advertising those fixtures as Astra.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time
import uuid

from .checkpoint import CheckpointEditor
from .editor_budget import BudgetExhausted, BudgetLedger
from .editor_contract import read_json_bounded, validate_development
from .editor_controller import EditorController, _identity, _implementation
from .editor_tools import run_checkpoint_action
from .native_runner import _write_json
from .protocol import canonical_hash
from .session_protocol import (SESSION_VARIANT, make_session_request, public_session_summary,
                               validate_session_protocol, validate_session_provenance,
                               validate_session_response)


class SessionBoundaryUnavailable(ValueError):
    pass


def session_readiness():
    """Capability evidence, not a configurable operator assertion of safety."""
    return {
        "variant": SESSION_VARIANT, "ready": False,
        "requested_selector": "gpt-6-astra", "requested_effort": "high",
        "accepted_selector": None, "observed_model": None, "observed_effort": None,
        "immutable_model_version": None, "experimental_editor_launched": False,
        "separate_api_key_required": False, "nested_frontier_call": False,
        "frontier_provider_token_cap": None,
        "frontier_cap_status": "explicitly_unavailable_in_this_protocol_variant",
        "blockers": [
            {"condition": "actual_session_capability_boundary_verified",
             "category": "unavailable_interface_capability",
             "evidence": "desktop create_thread exposes no permission or tool allowlist; native UI access denied",
             "needed": "host launch binding plus actual denial checks for held-out files, tools, network and harness writes"},
            {"condition": "actual_session_cancellation_and_no_ambiguous_retry_verified",
             "category": "missing_live_verification",
             "evidence": "desktop launch exposes no retry control; app-server fields are schema evidence only",
             "needed": "same-session interrupt and no-retry behavior verified without a model generation"},
            {"condition": "session_deadline_supervisor_armed_before_handoff",
             "category": "missing_implementation",
             "evidence": "file export cannot monitor or stop an external session; late-response rejection is not execution-time cancellation",
             "needed": "trusted transport supervisor enforcing the local deadline and cancellation on the actual session"},
        ],
        "strict_automated_mode_changed": False,
    }


def _session_transport():
    """Only a tested, trusted host adapter may replace this implementation.

    Its verify_boundary(manifest) must establish denial before a request is
    exported. verify_response(request, response, launch) must bind trusted
    interface provenance/usage to that exact session and handoff, not trust a
    response JSON claim. stop_and_confirm(launch) must stop the actual executor.
    A production adapter must also arm a deadline/cancellation supervisor before
    publishing the request. Export/import checks alone cannot supervise a chat.
    No module name, command, approval JSON or environment flag enables an adapter.
    """
    raise SessionBoundaryUnavailable(
        "authenticated Astra session pending: actual launch isolation, cancellation and no-retry controls are unverified")


class SessionController(EditorController):
    session_kind = "authenticated_editor_session"

    @classmethod
    def create(cls, root, *, model, policy, development, protocol, probe, interface_observation):
        root, model = Path(root), Path(model).resolve()
        protocol = validate_session_protocol(protocol)
        development = validate_development(development)
        probe = cls._probe_spec(probe)
        if len(development) != 1 or probe is None or len(probe["seeds"]) != 1:
            raise ValueError("session attempt requires one development item and one seed")
        if (probe["max_new_tokens"] != protocol["limits"]["target_max_new_tokens"]
                or probe["timeout_seconds"] != protocol["limits"]["target_timeout_seconds"]):
            raise ValueError("target probe limits differ from the explicit session protocol")
        if root.exists() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
            raise ValueError("new session requires a new private directory without symlinks")
        if root.resolve() == model or model in root.resolve().parents:
            raise ValueError("session must be outside the immutable original")
        if not isinstance(interface_observation, dict):
            raise ValueError("session interface observation must be an object")
        root.mkdir(mode=0o700, parents=True)
        for name in ("requests", "responses", "feedback", "actions", "candidates"):
            (root / name).mkdir(mode=0o700)
        editor = CheckpointEditor(model, policy, exposure_log=root / "exposures.jsonl")
        # Only approved model information crosses the boundary; inventory paths,
        # other tensors, tokenizer text, original config and host files do not.
        tensors = editor.inventory["tensors"]
        info = {"checkpoint_sha256": editor.original_sha256,
                "architectures": editor.inventory["model_config"].get("architectures", []),
                "stored_parameter_count": editor.inventory["parameter_count"],
                "allowed_tensors": {name: {key: tensors[name][key] for key in
                    ("shape", "dtype", "parameter_count")} for name in policy["allowed_tensors"]}}
        # This ledger's generated_tokens dimension is TARGET MODEL ONLY.
        # Frontier usage is recorded separately from trusted session receipts.
        limits = {"editor_requests": 1, "model_probes": 2, "requested_sequences": 3,
                  "generated_tokens": 2 * probe["max_new_tokens"],
                  "elapsed_seconds": protocol["limits"]["local_elapsed_seconds"] + 2 * probe["timeout_seconds"],
                  "billed_units": None}
        manifest = {"schema_version": 1, "kind": cls.session_kind,
                    "mode": SESSION_VARIANT, "run_id": uuid.uuid4().hex,
                    "model_path": str(model), "model_info": info,
                    "policy": copy.deepcopy(editor.policy), "development": development,
                    "protocol": protocol, "probe": probe, "limits": limits,
                    "budget_scope": {"generated_tokens": "target_model_only",
                                     "elapsed_seconds": "sum_of_conservative_local_reservations",
                                     "frontier_usage": "separate_receipt_observation_no_hard_cap"},
                    "interface_observation": copy.deepcopy(interface_observation),
                    "implementation": _implementation()}
        _write_json(root / "session.json", manifest, immutable=True)
        ledger = BudgetLedger(root / "budget.sqlite3", run_id=manifest["run_id"], limits=limits,
                              binding={"session_sha256": canonical_hash(manifest)}, create=True)
        ledger.close()
        _write_json(root / "ready.json", {"session_sha256": canonical_hash(manifest)}, immutable=True)
        return cls(root)

    def status(self):
        summary = public_session_summary(self.manifest["protocol"])
        summary.update(session_sha256=self.digest, readiness=session_readiness(),
                       budget_scope=self.manifest["budget_scope"],
                       budget=self.ledger.summary(), attempts=self.ledger.attempts(),
                       experiment_executed=False)
        return summary

    def prepare_request(self, instruction):
        """Private, immutable preview. This is not exported to an editor."""
        with self._lock():
            request = make_session_request(request_id=uuid.uuid4().hex, session_sha256=self.digest,
                instruction=instruction, protocol=self.manifest["protocol"], approved_context={
                    "model_info": self.manifest["model_info"], "policy": self.manifest["policy"],
                    "development": self.manifest["development"], "feedback": None})
            _write_json(self.root / "prepared-request.json", request, immutable=True)
            return {"request_id": request["request_id"], "request_sha256": canonical_hash(request),
                    "exported": False, "editor_dispatched": False}

    def export_request(self):
        with self._lock():
            transport = _session_transport()
            launch = transport.verify_boundary(self.manifest)
            request = read_json_bounded(self.root / "prepared-request.json")
            from .session_protocol import validate_session_request
            validate_session_request(request, self.manifest["protocol"])
            request_id = request["request_id"]
            # A single durable editor reservation precedes the only handoff.
            # Zero target tokens says nothing about unbounded frontier generation.
            self.ledger.reserve(request_id, kind="editor", sequences=1, max_generated_tokens=0,
                timeout_seconds=self.manifest["protocol"]["limits"]["local_elapsed_seconds"])
            _write_json(self.root / "launch.json", launch, immutable=True)
            _write_json(self.root / "requests" / f"{request_id}.json", request, immutable=True)
            self.ledger.mark_dispatched(request_id)
            return request

    def _attempt_active(self, row):
        self.ledger.check_active()
        if (time.time() < row["reserved_at"] or time.monotonic() < row["reserved_monotonic"]
                or time.time() >= row["deadline_at"] or time.monotonic() >= row["deadline_monotonic"]):
            raise BudgetExhausted("whole development attempt local deadline expired")

    def import_response(self, response):
        """Validate a real receipt, build one candidate, then compare both arms.

        A model's response is data. It never receives checkpoint paths, a Python
        interpreter, harness edits, grader edits or arbitrary execution tools.
        """
        with self._lock():
            transport = _session_transport()
            request_id = _identity(response.get("request_id") if isinstance(response, dict) else None)
            request = read_json_bounded(self.root / "requests" / f"{request_id}.json")
            row = self.ledger.check_active(request_id)
            launch = read_json_bounded(self.root / "launch.json")
            try:
                evidence = transport.verify_response(request, response, launch)
                provenance = validate_session_provenance(evidence["provenance"], self.manifest["protocol"])
                # A trusted fixture exercises mechanics, never real provenance.
                if evidence["evidence_kind"] not in {"synthetic_fixture", "authenticated_session"}:
                    raise ValueError("unsupported trusted receipt evidence")
                if evidence["evidence_kind"] == "authenticated_session" and (
                        provenance["source"] != "transport_metadata"
                        or provenance["accepted_selector"] != self.manifest["protocol"]["editor"]["requested_selector"]
                        or provenance["observed_model"] not in (None, self.manifest["protocol"]["editor"]["requested_selector"])
                        or provenance["observed_effort"] not in (None, self.manifest["protocol"]["editor"]["requested_effort"])):
                    raise ValueError("session receipt does not establish the requested Astra execution")
                public_session_summary(self.manifest["protocol"], provenance, evidence["usage"])
                _write_json(self.root / "session-receipt.json", evidence, immutable=True)
                response = validate_session_response(response, request, self.manifest["protocol"])
                _write_json(self.root / "responses" / f"{request_id}.json", response, immutable=True)
                _write_json(self.root / "actions" / f"{request_id}.json",
                            {"response_sha256": canonical_hash(response), "action": response["action"]}, immutable=True)
                candidate = self.root / "candidates" / request_id
                audit = run_checkpoint_action(model=self.manifest["model_path"], policy=self.manifest["policy"],
                    exposure_log=self.root / "exposures.jsonl", output=candidate,
                    tool="propose", arguments=response["action"]["arguments"],
                    timeout_seconds=max(0.001, row["deadline_at"] - time.time()),
                    check_active=lambda: self.ledger.check_active(request_id),
                    expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"])
                self.ledger.check_active(request_id)
                _write_json(self.root / "feedback" / "candidate.json", audit, immutable=True)
                self.ledger.finish(request_id, status="completed", completed_sequences=1, failed_sequences=0,
                                   generated_tokens=0, billed_units=None)
                from .session_comparison import run_development_comparison
                result = run_development_comparison(original_model=self.manifest["model_path"], candidate=candidate,
                    policy=self.manifest["policy"], development=self.manifest["development"],
                    probe=self.manifest["probe"], ledger=self.ledger, output=self.root / "comparison",
                    expected_original_sha256=self.manifest["model_info"]["checkpoint_sha256"],
                    check_active=lambda: self._attempt_active(row))
                summary = public_session_summary(self.manifest["protocol"], provenance, evidence["usage"])
                summary.update(status=result["status"], evidence_kind=evidence["evidence_kind"],
                    editor_identity_authenticated=evidence["evidence_kind"] == "authenticated_session",
                    candidate_validated=True,
                    comparison_dispatch_attempted=any(a["dispatched"] for a in result["arms"]),
                    development_comparison_executed=result["status"] == "completed",
                    completed_target_sequences={a["arm"]: int(a["status"] == "completed") for a in result["arms"]},
                    target_evidence_kind=result["evidence_kind"],
                    generated_tokens={a["arm"]: a["generated_tokens"] for a in result["arms"]},
                    experiment_executed=False, final_evaluation_executed=False)
                _write_json(self.root / "public-summary.json", summary, immutable=True)
                return summary
            except BaseException:
                # Unknown remote outcomes never authorize retry. The receipt
                # verifier must confirm a terminal response before reconciliation.
                if (self.root / "session-receipt.json").exists():
                    self._fail_pending(request_id)
                raise

    def cancel(self):
        self.ledger.cancel()
        if any(r["status"] in {"reserved", "dispatched"} for r in self.ledger.attempts()):
            transport = _session_transport()
            if not transport.stop_and_confirm(read_json_bounded(self.root / "launch.json")):
                raise SessionBoundaryUnavailable("local cancellation recorded; remote termination remains unconfirmed")
        return self.status()

    def recover(self):
        with self._lock():
            pending = [r for r in self.ledger.attempts() if r["status"] in {"reserved", "dispatched"}]
            if any(r["kind"] == "probe" for r in pending):
                raise SessionBoundaryUnavailable(
                    "local probe termination unconfirmed; retain reservations; remote session stop cannot confirm local worker shutdown")
            if pending:
                transport = _session_transport()
                if not transport.stop_and_confirm(read_json_bounded(self.root / "launch.json")):
                    raise SessionBoundaryUnavailable("remote outcome ambiguous; reservation retained and retry prohibited")
            recovered = self.ledger.recover_interrupted(executor_stopped=True)
            return {"recovered": recovered, **self.status()}

    def simulate(self, **kwargs):
        raise ValueError("session mode has no simulated editor command; use offline integration tests")
