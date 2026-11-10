#!/usr/bin/env python3
"""Independently verify a stopped candidate stage without executing model work.

Reads immutable evidence using stdlib hashes, byte comparisons, and read-only
SQLite joins. This verifies the recorded partial result, not candidate runtime
success. Only a new, exclusively created verification report is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys
from urllib.parse import quote


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def ledger(path):
    db = sqlite3.connect("file:" + quote(str(Path(path).resolve())) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        integrity = db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        meta = dict(db.execute("SELECT * FROM metadata WHERE singleton=1").fetchone())
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY reserved_at,request_id")]
        events = [dict(row) for row in db.execute("SELECT * FROM events ORDER BY event_index")]
        return integrity, meta, attempts, events
    finally:
        db.close()


def preservation(snapshot, repo):
    counts = Counter(matched=0, changed=0, unavailable=0)
    for name, pin in snapshot["files"].items():
        path = repo / name
        if not path.is_file() or getattr(path.stat(), "st_flags", 0) & 0x40000000:
            counts["unavailable"] += 1
        else:
            expected = pin["sha256"] if isinstance(pin, dict) else pin
            size_ok = not isinstance(pin, dict) or path.stat().st_size == pin["bytes"]
            counts["matched" if size_ok and file_digest(path) == expected else "changed"] += 1
    return dict(counts)


def source_changes(original, candidate, audit, proposal):
    """Recount stored and numerical changes in the two permitted BF16 matrices."""
    totals = Counter(targeted_coordinates=0, changed_stored_entries=0, changed_numerical_entries=0)
    valid = True
    for op in proposal["operations"]:
        name = op["tensor"]
        tensor = audit["model_inventory"]["tensors"][name]
        row = audit["per_tensor"][name]
        first, last = tensor["data_offsets"]
        offset = tensor["data_start"] + first
        chunks = []
        for folder in (original, candidate):
            with (folder / tensor["shard"]).open("rb") as stream:
                stream.seek(offset)
                chunks.append(stream.read(last - first))
        old, new = chunks
        rows, width = tensor["shape"]
        valid &= (tensor["dtype"] == "BF16" and tensor["shape"] == row["shape"]
                  and len(old) == len(new) == rows * width * 2
                  and hashlib.sha256(old).hexdigest() == row["base_sha256"]
                  and hashlib.sha256(new).hexdigest() == row["result_sha256"] == tensor["sha256"])
        columns = set(op["columns"])
        stored = numerical = 0
        square_sum = max_change = 0.0
        for index, (a, b) in enumerate(zip(struct.iter_unpack("<H", old), struct.iter_unpack("<H", new))):
            if a == b:
                continue
            valid &= index % width in columns
            stored += 1
            x = struct.unpack("<f", struct.pack("<I", a[0] << 16))[0]
            y = struct.unpack("<f", struct.pack("<I", b[0] << 16))[0]
            numerical += int(x != y)
            square_sum += (y - x) ** 2
            max_change = max(max_change, abs(y - x))
        targeted = rows * len(columns)
        valid &= (row["targeted_coordinates"] == targeted and row["changed_stored_entries"] == stored
                  and row["changed_numerical_entries"] == numerical and row["max_abs_change"] == max_change
                  and math.isclose(row["l2_norm"], math.sqrt(square_sum), rel_tol=1e-10, abs_tol=1e-12))
        totals.update(targeted_coordinates=targeted, changed_stored_entries=stored,
                      changed_numerical_entries=numerical)
    valid &= all(audit[key] == value for key, value in totals.items())
    return bool(valid), dict(totals)


def verify(root, repo, failure_path, user_request):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    manifest, config = read(root / "manifest.json"), read(root / "baseline-config.json")
    request, response = read(root / "handoff/request.json"), read(root / "response.json")
    ready, imported = read(root / "handoff/ready.json"), read(root / "handoff/import-result.json")
    source = root / "candidate/source"
    audit, proposal, policy = (read(source / (name + ".json")) for name in ("audit", "proposal", "policy"))
    provenance = read(root / "proposal-provenance.json")
    original = Path(read(root / "weight-information/host.json")["model_path"])
    current_auth = root / "candidate-authorization.txt"
    auth_bytes = current_auth.read_bytes()
    exact_request = (root / "candidate-stage-request.txt").read_bytes()
    check("current_explicit_authorization_preserved", auth_bytes == exact_request == Path(user_request).read_bytes()
          and auth_bytes != (root / "authorization-source.txt").read_bytes()
          and file_digest(current_auth) == provenance["authorization_sha256"])
    exact_response = json.loads(exact_request.decode().split("EXACT PROPOSAL JSON", 1)[1])
    check("exact_original_response_preserved", response == exact_response
          and response == read(root / "handoff/supplied-response.json") == read(root / "handoff/accepted-response.json")
          and response["proposal"] == proposal
          and file_digest(root / "response.json") == provenance["response_file_sha256"])
    check("request_and_frozen_handoff_binding", response["request_id"] == request["request_id"]
          == provenance["request_id"]
          and response["request_sha256"] == digest(request) == ready["request_sha256"]
          == provenance["request_sha256"]
          and file_digest(root / "handoff/editor-packet.txt") == ready["packet_sha256"]
          and file_digest(root / "handoff/next-commands.sh") == ready["next_commands_sha256"]
          and all(file_digest(root / name) == pin for name, pin in request["evidence_files_sha256"].items()))
    check("proposal_unchanged_and_provenance_separate", proposal["provenance"] == "editor_proposal"
          and proposal == exact_response["proposal"]
          and len({op["tensor"] for op in proposal["operations"]}) == len(proposal["operations"])
          and all(op["op"] == "scale_columns" and op["tensor"] in policy["allowed_tensors"]
                  for op in proposal["operations"])
          and provenance["editor_isolation"] == "unverified"
          and provenance["original_request_preserved"] is True
          and provenance["selected_columns_causal_explanation_verified"] is False
          and "file-reading and JSON-parsing tools" in provenance["statement"])
    check("saved_checkpoint_audit_binding", imported["accepted"] is True
          and audit["complete"] is True and audit["artifact_verification"] == "passed"
          and audit["inference_executed"] is False and audit["experiment_executed"] is False
          and proposal["base_sha256"] == audit["base_sha256"] == imported["audit"]["original_sha256"]
          == manifest["checkpoint"]["sha256"]
          and audit["proposal_sha256"] == digest(proposal) and audit["policy_sha256"] == digest(policy)
          and policy == read(root / "weight-information/policy.json")
          and audit["exposure_sha256"] == digest(read(source / "exposure.json"))
          and all(imported["audit"][key] == audit[key] for key in
                  ("result_sha256", "targeted_coordinates", "changed_stored_entries", "changed_numerical_entries")))
    candidate_files = audit["model_inventory"]["files"]
    observed_files = {name: {"sha256": file_digest(source / "model" / name),
                            "size_bytes": (source / "model" / name).stat().st_size} for name in candidate_files}
    original_files = {name: {"sha256": file_digest(original / name),
                           "size_bytes": (original / name).stat().st_size} for name in candidate_files}
    check("candidate_checkpoint_actual_file_hashes", observed_files == candidate_files
          and {p.name for p in (source / "model").iterdir()} == set(candidate_files)
          and digest(observed_files) == audit["result_sha256"])
    check("pristine_checkpoint_actual_file_hashes", digest(original_files) == audit["base_sha256"])
    source_valid, stored = source_changes(original, source / "model", audit, proposal)
    check("independent_targeted_BF16_change_counts", source_valid)
    check("frozen_baseline_identity", read(root / "ready.json") == {"manifest_sha256": digest(manifest)}
          and all(file_digest(name) == pin for name, pin in manifest["files"].items())
          and file_digest(root / "authorization-source.txt") == manifest["authorization_source_sha256"])
    plan = read(root / "candidate-comparison-plan.json")
    check("frozen_candidate_plan_order_and_limits", read(root / "candidate-comparison-plan-ready.json")
          == {"plan_sha256": digest(plan)} and plan["baseline_manifest_sha256"] == digest(manifest)
          and plan["baseline_config_file_sha256"] == file_digest(root / "baseline-config.json")
          and plan["task_items_sha256"] == digest(config["items"])
          and plan["task_order"] == [item["item_id"] for item in config["items"]]
          and plan["settings"] == config["settings"] and plan["warmup_prompt"] == config["warmupPrompt"]
          and plan["grading_policy"] == manifest["policy"] and plan["grader"] == manifest["grader"]
          and plan["candidate_budget"] == {"engine_load_attempts": 1, "warmup_dispatches": 1,
              "scored_dispatches": 32, "generation_dispatches": 33, "generated_tokens": 32784,
              "request_seconds": 120, "wall_seconds": 5400, "automatic_retries": 0})
    good, meta, attempts, events = ledger(root / "handoff/import-ledger.sqlite3")
    frozen = json.loads(meta["frozen_json"])
    check("single_completed_import_ledger", good and len(attempts) == 1
          and frozen["binding"]["request_sha256"] == digest(request)
          and frozen["limits"] == {"editor_requests": 1, "model_probes": 0, "requested_sequences": 1,
              "generated_tokens": 0, "elapsed_seconds": 900.0, "billed_units": None}
          and all(a["request_id"] == request["request_id"] and a["status"] == "completed"
                  and a["retry_of"] is None and a["observed_generated_tokens"] == 0
                  and a["timeout_seconds"] == 900 and 0 <= a["observed_elapsed_seconds"] <= 900
                  and a["conservative"] == a["limit_violation"] == 0 for a in attempts)
          and [e["kind"] for e in events if e["request_id"] == request["request_id"]]
              == ["reserved", "dispatched", "completed"])
    baseline = read(root / "baseline/graded.json")
    good, base_meta, base_attempts, _ = ledger(root / "generation-budget.sqlite3")
    check("baseline_ledger_complete_and_unreopened", good and base_meta["cancelled"] == 1
          and len(base_attempts) == 33 and all(a["status"] == "completed" and a["retry_of"] is None for a in base_attempts)
          and json.loads(base_meta["frozen_json"])["binding"]["manifest_sha256"] == digest(manifest))
    items = read(root / "data/development-items.json")
    check("baseline_raw_grade_bindings_preserved", baseline["summary"]["technically_valid"] is True
          and len(items) == len(baseline["records"]) == len(baseline["grades"]) == 32
          and baseline["records"] == read(root / "baseline/records.json")
          and baseline["grades"] == read(root / "baseline/grades.json")
          and all(g["run_record_sha256"] == digest(r) and g["item_sha256"] == digest(i)
                  and g["item_id"] == r["item_id"] == i["item_id"]
                  and g["policy_sha256"] == digest(manifest["policy"]) and g["grader"] == manifest["grader"]
                  for i, r, g in zip(items, baseline["records"], baseline["grades"])))
    prohibited = ["candidate/converted", "candidate/converted-conversion-start.json",
                  "candidate/converted-source-binding.json", "candidate/converted-convert.stdout.txt",
                  "candidate/converted-convert.stderr.txt", "candidate/deployed", "candidate-comparison"]
    check("conversion_and_runtime_unexecuted", all(not (root / name).exists() for name in prohibited)
          and sorted(p.name for p in (root / "execution").iterdir()) == ["baseline-1"]
          and imported["frontier_requests"] == imported["target_generations"] == 0)
    heldout = read(root / "data/evaluation-items.json")
    check("heldout_examples_unexecuted_in_study", len(heldout) == 128
          and not {i["item_id"] for i in heldout} & {r["item_id"] for r in baseline["records"]}
          and plan["heldout_generations"] == 0 and not (root / "candidate-comparison").exists())
    historical = read(root / "preservation-before.json")
    old_counts = preservation(historical, repo)
    current = read(root / "candidate-preservation-before.json")
    current_counts = preservation(current, repo)
    check("historical_available_evidence_preserved", old_counts["matched"] == len(historical["files"])
          and old_counts["changed"] == old_counts["unavailable"] == 0
          and historical["historical_capacity_transferred"] is False)
    check("baseline_and_public_evidence_preserved", current_counts["matched"] == len(current["files"])
          and current_counts["changed"] == current_counts["unavailable"] == 0)
    check("preservation_snapshots_agree_on_overlapping_pins", all(
          current["files"][name] == pin["sha256"] for name, pin in historical["files"].items()
          if name in current["files"]))
    failure = read(failure_path)
    check("failure_record_and_evidence_bindings", failure["status"] == "stopped"
          and failure["stage"] == "candidate_conversion_package_precheck"
          and failure["command_exit_code"] == failure["command_file_invocations"] == 1
          and failure["conversion_stage_attempts"] == 1 and failure["import_accepted"] is True
          and failure["conversion_subprocess_started"] is False and failure["packaging_started"] is False
          and failure["frozen_implementation_changed"] is False
          and all(file_digest(root / name) == pin for name, pin in failure["evidence_file_sha256"].items())
          and all(not (root / name).exists() for name in failure["absent_stage_artifacts"])
          and all(failure[key] == 0 for key in ("automatic_retries", "baseline_reruns", "candidate_generated_tokens",
                  "candidate_generation_dispatches", "candidate_load_attempts", "candidate_scored_dispatches",
                  "candidate_warmup_dispatches", "experimental_frontier_requests", "heldout_generations", "new_weight_proposals"))
          and failure["selection"]["status"] == "not_evaluable" and failure["selection"]["selected_arm"] is None)
    conversion = read(repo / "artifacts/browser-integration-v1/conversion/artifacts.json")
    configured, resolved = (failure["diagnostics"][name] for name in ("configured_venv", "resolved_executable"))
    check("read_only_converter_diagnosis_matches_pins", configured["returncode"] == 0 and resolved["returncode"] == 1
          and json.loads(configured["stdout"])["packages"] == failure["expected_packages"] == conversion["converter_packages"]
          and json.loads(configured["stdout"])["venv_active"] is True
          and "PackageNotFoundError" in resolved["stderr"] and "apache-tvm-ffi" in resolved["stderr"]
          and Path(configured["executable"]).resolve() == Path(resolved["executable"])
          and failure["existing_conversion_subprocess_limit_seconds"] == 1800
          and failure["package_precheck_limit_seconds"] == 30)
    cleanup = read(root / "candidate-cleanup.json")
    check("recorded_cleanup_consistent_with_unstarted_runtime", cleanup["pipeline_exit_code"] == 1
          and cleanup["matching_candidate_stage_processes"] == [] and cleanup["no_matching_stage_processes"] is True
          and cleanup["import_worker_completed"] is True and cleanup["browser_execution_started"] is False
          and cleanup["conversion_subprocess_started"] is False
          and cleanup["browser_cleanup"] == "not_applicable_no_browser_execution_started")
    stderr = (root / "candidate-stage.stderr.txt").read_text()
    check("recorded_failure_is_converter_environment_precheck", "apache-tvm-ffi" in stderr
          and "importlib.metadata" in stderr and "CalledProcessError" in stderr
          and "PackageNotFoundError" in resolved["stderr"])
    return {"schema_version": 1, "kind": "independent_ifeval_candidate_partial_verification",
        "author": "Arhaan2", "passed": all(checks.values()), "checks": checks,
        "candidate_stage_status": "stopped_before_conversion_subprocess",
        "candidate_execution_success": False, "paired_comparison_available": False,
        "measurements": {"completed_imports": len(attempts),
            "import_seconds": sum(a["observed_elapsed_seconds"] for a in attempts), **stored,
            "candidate_engine_loads": 0, "candidate_generation_dispatches": 0,
            "candidate_generated_tokens": 0, "heldout_generations": 0, "experimental_frontier_requests": 0,
            "historical_preservation": old_counts, "candidate_preservation": current_counts,
            "historical_unavailable_files_not_verified": len(historical["unavailable_cloud_placeholders"])},
        "identities": {"verifier_sha256": file_digest(__file__), "baseline_manifest_sha256": digest(manifest),
            "candidate_checkpoint_sha256": audit["result_sha256"], "failure_record_sha256": file_digest(failure_path),
            "cleanup_record_sha256": file_digest(root / "candidate-cleanup.json"),
            "candidate_preservation_sha256": file_digest(root / "candidate-preservation-before.json")},
        "scope": "Verification of recorded partial-stage evidence and study directories, not successful candidate execution; historical unavailable files and unrelated external sessions are excluded.",
        "target_forward_passes": 0, "engine_loads_by_verifier": 0, "input_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/ifeval-browser-v1")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--failure", required=True)
    parser.add_argument("--user-request", required=True, help="Original user-supplied request file; location is not published")
    parser.add_argument("--out-name", choices=("verification.json", "verification-corrected.json", "verification-final.json"),
                        default="verification.json")
    args = parser.parse_args()
    out = Path(args.repo) / "reports/ifeval-browser-v1-candidate" / args.out_name
    if out.exists():
        raise SystemExit("Verification output exists; preserve it.")
    try:
        result = verify(args.root, args.repo, args.failure, args.user_request)
    except Exception as error:
        result = {"schema_version": 1, "kind": "independent_ifeval_candidate_partial_verification",
            "passed": False, "error_category": type(error).__name__, "complete_verification": False,
            "candidate_execution_success": False, "target_forward_passes": 0,
            "engine_loads_by_verifier": 0, "input_mutations": 0}
    if args.out_name in ("verification-corrected.json", "verification-final.json"):
        previous = out.with_name("verification.json")
        result["preserved_prior_verification_sha256"] = file_digest(previous)
        result["verifier_correction"] = ("The initial checker expected nested PackageNotFoundError text in the outer "
            "pipeline traceback. That traceback preserves CalledProcessError; the separately captured read-only "
            "diagnostic preserves PackageNotFoundError. This version checks each in its actual evidence location. "
            "The first report and all experimental evidence remain unchanged; no pipeline action was repeated.")
    if args.out_name == "verification-final.json":
        result["preserved_prior_verifications_sha256"] = {
            name: file_digest(out.with_name(name))
            for name in ("verification.json", "verification-corrected.json")}
        result["source_privacy_refinement"] = ("Redundant private request-envelope identifiers and concrete proposal "
            "parameters were removed from the verifier source. Exact binding derives from the original user attachment, "
            "its preserved local copy, and the frozen request. Tensor dimensions derive from the saved inventory. "
            "No frozen implementation or experimental evidence was changed.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
