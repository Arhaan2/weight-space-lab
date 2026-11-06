"""Read-only live readiness and an inert, zero-action connection-test plan.

Neither an observation JSON file nor a user-supplied 'verified' flag can register
a provider. There is deliberately no remote execution implementation here.
The diagnostic preserves the distinction between local enforcement, provider
controls, and metadata that only a real response could establish.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import subprocess
from urllib.parse import quote

from .editor_contract import read_json_bounded
from .protocol import canonical_hash, validate_protocol


class LiveDispatchBlocked(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("live Astra access pending: " + ", ".join(report["dispatch_blockers"]))


def installed_cli():
    """Inspect version/help only. No account reads, session creation or model call."""
    executable = shutil.which("codex")
    if executable is None:
        return {"available": False, "version": None, "binary_sha256": None}
    try:
        version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                 timeout=5, check=True).stdout.strip()
        help_text = subprocess.run([executable, "exec", "--help"], capture_output=True, text=True,
                                   timeout=5, check=True).stdout
        return {"available": True, "version": version[:128],
                "binary_sha256": hashlib.sha256(Path(executable).resolve().read_bytes()).hexdigest(),
                "exec_help_sha256": hashlib.sha256(help_text.encode()).hexdigest(),
                "inspection_scope": "version/help only; accepted configuration is not applied provider enforcement"}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "version": None, "binary_sha256": None,
                "error": "local_cli_inspection_failed"}


def connection_test_plan():
    """Prepared intent only: no reservation, launch, credentials, or API ID."""
    return {
        "schema_version": 1, "kind": "phase7_connection_test_plan", "status": "prepared_not_dispatched",
        "requested_selector": "gpt-6-astra", "provider_api_model_identifier": None,
        "purpose": "one_development_only_transport_acknowledgement",
        "request": {"messages": [{"role": "user", "content": 'Return exactly {"status":"acknowledged"}.'}],
                    "tools": [], "response_schema": {"type": "object", "additionalProperties": False,
                        "properties": {"status": {"type": "string", "const": "acknowledged"}}, "required": ["status"]}},
        "bounds": {"editor_requests": 1, "candidate_proposals": 0, "target_model_probes": 0,
                   "target_model_generated_tokens": 0, "requested_sequences": 1,
                   "local_elapsed_seconds": 30, "frontier_generated_tokens_including_reasoning": 64,
                   "provider_usage_bound": None},
        "automatic_retries": 0, "ambiguous_remote_outcome": "record_indeterminate_and_do_not_retry",
        "actions_dispatched": False, "weight_changes_allowed": False,
        "held_out_data_included": False, "candidate_search_allowed": False,
        "identity_evidence_source": "execution_interface_metadata_only_not_response_content",
        "reservation_status": "not_reserved_requires_existing_authorized_live_budget",
        "scientific_result": False, "experiment_executed": False,
    }


def validate_connection_response(value):
    """A zero-action response cannot enter the offline tool dispatcher."""
    if type(value) is not dict or value != {"status": "acknowledged"}:
        raise ValueError("connection response must be the exact acknowledgement without tools or metadata claims")
    return dict(value)


def _budget_snapshot(root):
    """Inspect the existing ledger read-only, even if its code freeze is older.

    This cannot resume or reserve against that ledger. Do not create a fresh
    database or migrate an exhausted/offline run to obtain connection budget.
    """
    if root is None:
        return {"available": False, "reason": "no_existing_session_selected"}
    root = Path(root)
    try:
        manifest = read_json_bounded(root / "session.json", max_bytes=1_048_576)
        digest = canonical_hash(manifest)
        if read_json_bounded(root / "ready.json") != {"session_sha256": digest}:
            raise ValueError("session identity changed")
        path = root / "budget.sqlite3"
        if path.is_symlink() or not path.is_file():
            raise ValueError("ledger missing or symlinked")
        uri = "file:" + quote(str(path.absolute())) + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=1)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            frozen, cancelled = db.execute("SELECT frozen_json,cancelled FROM metadata WHERE singleton=1").fetchone()
            frozen = json.loads(frozen)
            if (frozen["run_id"] != manifest["run_id"] or frozen["limits"] != manifest["limits"]
                    or frozen["binding"].get("session_sha256") != digest):
                raise ValueError("ledger binding differs from manifest")
            used = db.execute("""SELECT COALESCE(SUM(kind='editor'),0),
                COALESCE(SUM(requested_sequences),0),COALESCE(SUM(charged_generated_tokens),0),
                COALESCE(SUM(charged_elapsed_seconds),0) FROM attempts""").fetchone()
        keys = ("editor_requests", "requested_sequences", "generated_tokens", "elapsed_seconds")
        from .editor_controller import _implementation
        return {"available": True, "mode": manifest["mode"], "session_sha256": digest,
                "implementation_matches_current": manifest["implementation"] == _implementation(),
                "cancelled": bool(cancelled), "used": dict(zip(keys, used)),
                "configured_billed_units_limit": manifest["limits"]["billed_units"],
                "remaining": {key: max(0, manifest["limits"][key] - amount) for key, amount in zip(keys, used)},
                "can_authorize_live_execution": False,
                "reason": "existing_session_is_offline_and_cannot_be_promoted_or_reset",
                "read_only": True}
    except (ValueError, OSError, sqlite3.Error, TypeError, KeyError):
        return {"available": False, "reason": "existing_session_or_ledger_could_not_be_verified", "read_only": True}


def _search_conditions(pending):
    """Explain every legacy missing pin without filling or launching the pilot."""
    rows = []
    for issue in pending:
        field = issue.split(": ", 1)[1]
        category = "configuration"
        available = "The intentionally unexecuted fresh_editor_search template has null here. No pilot configuration was changed."
        expected = "A verified, frozen " + field + " for the later candidate-search pilot."
        step = "Bind the verified pilot evidence only when that later phase is authorized; this field is not needed for a zero-probe connection check."
        if field == "editor_model_version":
            category = "unavailable_interface_capability"
            expected = "An immutable editor build actually exposed by the execution interface."
            available = "No authenticated immutable build is exposed; CLI 0.163.0 and opaque comp_hash 3000 cannot fill this pin."
            step = "Obtain a genuine immutable version or approve an explicit provenance-schema correction; do not silently relax the pilot."
        elif field.startswith("editor_provenance."):
            category = "missing_live_verification"
            expected = "Actual editor execution-interface provenance, separate from request text and model self-description."
            available = "Requested selector and installed CLI evidence exist; no live editor execution metadata exists."
            step = "Record interface acceptance and response metadata in the gated connection check before using it in a future pilot."
        rows.append({"condition": issue, "file": "configs/profiles/fresh_editor_search.json",
                     "function": "weight_space_lab.protocol.validate_protocol", "field": field,
                     "expected_evidence": expected, "available_evidence": available,
                     "classification": category, "smallest_next_step": step,
                     "blocks_connection_test": False, "scope": "later_search_pilot"})
    return rows


def live_readiness(*, root=None, inspect_cli=True):
    """Return evidence-bearing conditions; diagnostics never dispatch anything."""
    package_root = Path(__file__).resolve().parent.parent
    cli = installed_cli() if inspect_cli else {"available": None, "inspection_scope": "not_requested"}
    observation = read_json_bounded(package_root / "configs/editor_interface_observation.json")
    controls_path = package_root / "configs/phase7_cli_control_evidence.json"
    controls = read_json_bounded(controls_path, max_bytes=1_048_576) if controls_path.exists() else None
    budget = _budget_snapshot(root)
    plan = connection_test_plan()
    conditions = []

    def add(identifier, file, function, condition, expected, available, category, next_step,
            *, stage="pre_dispatch", status="blocked", mandatory=True):
        conditions.append({"id": identifier, "file": file, "function": function,
            "condition": condition, "expected_evidence": expected, "available_evidence": available,
            "classification": category, "smallest_next_step": next_step,
            "stage": stage, "status": status, "mandatory": mandatory})

    add("live_adapter_missing", "weight_space_lab/editor_controller.py", "EditorController.dispatch_live",
        "No supported live provider adapter is registered.",
        "Executable adapter selecting the requested model, binding the narrow contract, controls and returned provenance.",
        "Dispatch is fail-closed; simulate() calls a Python fixture only. File imports remain unverified.",
        "missing_implementation", "Implement one adapter only after the selected interface demonstrates the mandatory controls.")
    add("actual_launch_isolation_unverified", "weight_space_lab/editor_sandbox.py", "SandboxRunner.run / _profile",
        "The tested stdlib Python profile has never launched a live editor or its complete tool surface.",
        "Denial tests on the exact editor launch: final tasks/answers/results, host files, descriptors, plugins and network routes.",
        "Host Python fixture denials passed; CLI executable/provider network are not permitted by this profile.",
        "missing_live_verification", "Build a narrow provider broker and verify the exact launch/tools; do not widen the Python profile or infer equivalence.")
    add("frontier_generation_cap_not_demonstrated", "weight_space_lab/editor_controller.py", "_export / dispatch_live",
        "No demonstrated frontier-provider enforcement of the 64-token total output bound, including reasoning.",
        "Supported dispatch control observed on the actual provider request and documented/enforced at the provider.",
        "max_output_tokens is an offline envelope/reservation; local deadlines and output usage events are not remote generation caps.",
        "unavailable_interface_capability", "Use an explicitly supported provider-enforced output limit for this exact selector, or propose a protocol change; do not treat CLI goals as a cap.")
    money_cap = budget.get("configured_billed_units_limit")
    add("frontier_billed_cap_not_demonstrated", "weight_space_lab/editor_budget.py", "BudgetLedger.reserve / finish",
        "No provider-side billed-usage cap is demonstrated; the selected offline budget has no monetary ceiling configured.",
        "If a billed ceiling is configured: establish its unit and enforceable conservative bound before dispatch. Otherwise record available usage without claiming a monetary cap.",
        {"configured_billed_units_limit": money_cap, "actual_frontier_billed_usage": None,
         "local_fixture_zero_is_frontier_usage": False},
        "unavailable_interface_capability", "Do not invent a mandatory money cap from billed_units:null. Any future configured cap requires demonstrated enforcement.",
        stage="pre_dispatch" if money_cap is not None else "disclosure",
        status="blocked" if money_cap is not None else "not_configured", mandatory=money_cap is not None)
    add("no_substitution_control_unverified", "configs/editor_interface_observation.json", "interfaces / model_selector_sources",
        "A catalog selector and schema fallback flag do not demonstrate no substitution on a real dispatch.",
        "Explicit requested selector, no fallback behavior, and execution-interface model/reroute metadata.",
        "gpt-6-astra is exposed; allowProviderModelFallback exists in experimental schema. No real dispatch acceptance or response was observed.",
        "missing_live_verification", "Wire and test no-fallback behavior before launch; reject reroute/mismatch using interface events.")
    add("authorized_live_budget_unavailable", "weight_space_lab/editor_live.py", "_budget_snapshot",
        "No existing authorized live session can reserve this one request.",
        "A non-cancelled authorized live ledger with one request/sequence, 64 frontier tokens and 30 seconds available, with complete billing semantics.",
        budget, "configuration", "Preserve spent offline budgets; explicitly authorize/freeze a live budget only after capability gates pass. Never reset a run.")
    if budget.get("available") and budget.get("implementation_matches_current") is False:
        add("existing_session_code_freeze_changed", "weight_space_lab/editor_controller.py", "EditorController.__init__ / _implementation",
            "The selected historical session is bound to the original package hashes, which differ from this readiness implementation.",
            "Exact original package/runtime hashes for any resume, or an explicitly authorized new protocol with budget provenance.",
            "Read-only diagnostic access works; ordinary controller resume correctly refuses changed implementation.",
            "configuration", "Keep the historical session immutable. Do not relax the source freeze or recreate its ledger to obtain live budget.")
    add("zero_action_contract_prepared", "weight_space_lab/editor_live.py", "connection_test_plan / validate_connection_response",
        "Connection diagnostic must expose no proposal/probe/action execution.",
        "One structured acknowledgement request, empty tools, exact response validator, no calls to _execute_action.",
        "Separate inert zero-action plan and exact acknowledgement validator are implemented; no transport exists.",
        "missing_implementation", "Bind this narrow contract in the future verified adapter; do not reuse the six-tool offline action contract.", status="prepared")
    add("remote_cancellation_and_retry_semantics_unverified", "weight_space_lab/editor_live.py", "connection_test_plan",
        "No live transport demonstrates zero automatic retries and remote cancellation/outcome semantics.",
        "One provider request, bounded local supervisor, no retry after an ambiguous remote outcome; explicit remote cancellation capability or limitation.",
        "Local worker cancellation is tested. It cannot prove a remote provider stops generation when a local client is killed.",
        "missing_live_verification", "Verify transport retry/cancel behavior and charge ambiguous outcome conservatively; retain provider hard cap independently.")
    add("accepted_selector_not_observed", "configs/editor_interface_observation.json", "actual_returned_model_identity",
        "The requested selector has not been accepted by an actual live execution interface.",
        "Interface-produced session/dispatch acceptance for gpt-6-astra, separate from configured request and model text.",
        None, "missing_live_verification", "Record live session/dispatch metadata during the gated one-request check.", stage="post_response", status="not_observed")
    add("response_model_metadata_not_observed", "configs/editor_interface_observation.json", "actual_returned_model_identity",
        "No actual session/response model metadata has been observed.",
        "Interface response model/provider/reroute metadata, never the model's self-description.",
        None, "missing_live_verification", "Capture authenticated interface events; reject mismatches and retain unknown fields as null.", stage="post_response", status="not_observed")
    add("frontier_usage_not_observed", "weight_space_lab/editor_budget.py", "BudgetLedger.finish",
        "No frontier usage has been observed because no frontier request was sent.",
        "Record interface token usage and billed usage when exposed; preserve unknowns and charge ambiguous outcomes conservatively.",
        "CLI schema exposes token usage events. Actual provider usage and billing are null, not fixture zero.",
        "missing_live_verification", "Capture metering during the gated connection check; do not describe it as an execution cap.", stage="post_response", status="not_observed")
    add("immutable_model_version_not_exposed", "weight_space_lab/protocol.py", "validate_protocol / _PROFILE_PINS",
        "fresh_editor_search requires editor_model_version but the inspected catalog does not expose an authenticated immutable build.",
        "An immutable build value actually supplied by the interface, or an explicitly approved protocol revision.",
        {"immutable_model_version": None, "opaque_catalog_comp_hash": observation.get("catalog_comp_hash"),
         "cli_version_is_model_version": False}, "unavailable_interface_capability",
        "Propose explicit unavailable-version provenance for a connection diagnostic; keep the existing search protocol unchanged pending approval.",
        stage="protocol_proposal", status="not_exposed", mandatory=False)
    pilot = validate_protocol(read_json_bounded(package_root / "configs/profiles/fresh_editor_search.json"), require_ready=True)
    blockers = [r["id"] for r in conditions if r["stage"] == "pre_dispatch" and r["status"] != "verified" and r["mandatory"]]
    return {"schema_version": 1, "kind": "phase7_live_readiness", "ready": False,
            "status": "blocked", "dispatch_blockers": blockers, "conditions": conditions,
            "requested_selector": "gpt-6-astra", "selector_accepted_by_live_interface": None,
            "observed_session_response_model_metadata": None, "immutable_model_version": None,
            "model_self_description_used": False, "installed_cli": cli,
            "selector_accepted_by_execution_interface": {
                "local_cli_synthetic_configuration": None if controls is None else controls.get("locally_serialized_selector"),
                "live_provider_acceptance": None,
                "scope": "Local parser/serialization acceptance is separate from live execution acceptance."},
            "cli_behavior_evidence": controls,
            "cli_behavior_evidence_matches_installed_binary": None if controls is None else
                controls.get("binary_sha256") == cli.get("binary_sha256"),
            "existing_budget": budget, "connection_plan_sha256": canonical_hash(plan),
            "legacy_search_check": {"profile": pilot["profile"], "run_configuration_complete": pilot["run_configuration_complete"],
                                    "pending_run_requirements": pilot["pending_run_requirements"],
                                    "conditions": _search_conditions(pilot["pending_run_requirements"]),
                                    "scope": "later search pilot; not the Phase 7 connection diagnostic"},
            "real_editor_request_sent": False, "scientific_result": False, "experiment_executed": False}


def require_live_ready(*, root=None):
    report = live_readiness(root=root)
    # No report flag or config entry can turn this into a provider dispatcher.
    raise LiveDispatchBlocked(report)
