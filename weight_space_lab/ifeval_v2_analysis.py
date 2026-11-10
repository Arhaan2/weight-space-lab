"""Private final analysis with frozen aliases and existing paired estimators.

No target generation, grading, or candidate selection occurs here. Arhaan2.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .analysis_pipeline import build_analysis_manifest
from .ifeval_baseline import save
from .ifeval_heldout_analysis import ANALYSIS_SPEC, compare_evidence
from .ifeval_v2_dataset import LABEL, verify_frozen
from .inventory import file_sha256, read_json
from .protocol import canonical_hash
from .records import run_record_sha256


def _plan(study, selection):
    dataset = verify_frozen(study["evaluator_root"])
    items = read_json(Path(study["evaluator_root"]) / "final-items.json")
    if dataset["sha256"] != study["config"]["final_set"]["manifest_sha256"]:
        raise ValueError("Final dataset differs from the frozen study")
    counts = {"prompts": len(items), "instructions": sum(len(i["target"]["instruction_id_list"]) for i in items)}
    if not 0 <= len(items) <= 128 or any(dataset["counts"][key] != count for key, count in counts.items()):
        raise ValueError("Final counts differ from the frozen official subset")
    plan = {
        "schema_version": 1, "kind": "ifeval_browser_v2_final_analysis_plan",
        "label": LABEL, "selection_sha256": canonical_hash(selection),
        "final_manifest_sha256": dataset["sha256"], "ordered_items_sha256": canonical_hash(items),
        "named_artifacts": selection["named_arms"], "analysis_spec": deepcopy(ANALYSIS_SPEC),
        "counts": counts,
        "policy": deepcopy(study["config"]["grading_policy"]),
        "split_manifest": {"schema_version": 1, "kind": "benchmark_split", "split": "test",
            "source_manifest_sha256": dataset["sha256"], "items": [
                {"benchmark": i["benchmark"], "item_id": i["item_id"],
                 "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in items]},
        "contrasts": {"selected_astra_minus_pristine": "primary"} if items and len(set(selection["named_arms"].values())) > 1 else {},
        "duplicate_rule": "pristine fallback skips final and resource comparisons; no pristine-versus-pristine dispatch or contrast",
    }
    return plan


def freeze_analysis_plan(root, study, selection):
    """Seal estimators/denominators/identities before any final arm opens."""
    root = Path(root)
    if read_json(root / "state.json").get("final"):
        raise ValueError("Final analysis plan must precede every final dispatch")
    plan = _plan(study, selection)
    path = root / "final-analysis-plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Existing final analysis plan differs from frozen inputs")
    else:
        save(path, plan)
    return plan


def _comparison_view(identity, evidence, *, alias):
    """In-memory analytical alias; immutable raw records remain untouched.

The existing paired-manifest validator requires distinct run IDs, including
when an arm falls back to pristine. Rebind only the comparison view's run ID
and its grade hash, explicitly disclose reuse, and never write it as raw data.
"""
    identity, evidence = deepcopy(identity), deepcopy(evidence)
    identity["run_id"] = identity["run_id"] + ":analysis-alias:" + alias
    hashes = {}
    for record in evidence["records"]:
        old_hash = run_record_sha256(record)
        record["run_id"] = identity["run_id"]
        hashes[old_hash] = run_record_sha256(record)
    for grade in evidence["grades"]:
        if grade["run_record_sha256"] not in hashes:
            raise ValueError("Reused grade is not bound to its original run record")
        grade["run_record_sha256"] = hashes[grade["run_record_sha256"]]
    return identity, evidence


def compare_selected(items, plan, baseline, candidate, *, alias, evidence_kind="real_model"):
    """Use the same 10,000 seed-0 prompt-cluster estimators for every contrast."""
    base_identity, base_evidence = baseline
    candidate_identity, candidate_evidence = candidate
    reused = base_identity["run_id"] == candidate_identity["run_id"]
    if reused:
        candidate_identity, candidate_evidence = _comparison_view(candidate_identity, candidate_evidence, alias=alias)
    manifests = [build_analysis_manifest(plan["split_manifest"], items, benchmark="ifeval", seeds=[42],
        policy=plan["policy"], arms={"baseline": base_identity, "candidate": candidate_identity},
        stage="full", metric="prompt_level_" + mode + "_accuracy", evidence_kind=evidence_kind)
        for mode in ("strict", "loose")]
    dispatched = ({"baseline": base_evidence["_v2_dispatched_item_ids"],
                   "candidate": candidate_evidence["_v2_dispatched_item_ids"]}
                  if "_v2_dispatched_item_ids" in base_evidence and "_v2_dispatched_item_ids" in candidate_evidence
                  else None)
    result = compare_evidence(*manifests, base_evidence, candidate_evidence, dispatched_items=dispatched)
    result.update(kind="ifeval_browser_v2_final_contrast", label=LABEL,
        analysis_plan_sha256=canonical_hash(plan), reused_identical_artifact_evidence=reused,
        analytical_alias_only=reused, added_generation_dispatches=0,
        evidence_binding_manifests_derived_from_predeclared_plan=True)
    return result


def _evidence(root, study, row, items):
    session = Path(root) / "sessions" / ("final-" + row["dispatch_arm"])
    manifest_path = session / "manifest.json"
    if not manifest_path.exists():
        source = (read_json(row["artifact_path"])["source_checkpoint_sha256"] if row["artifact_path"]
                  else study["baseline"]["source_sha256"])
        checkpoint = deepcopy(read_json(Path(study["baseline"]["baseline_root"]) / "manifest.json")["checkpoint"])
        checkpoint["sha256"] = source
        identity = {"run_id": "unexecuted-v2-final-" + row["dispatch_arm"], "profile": "browser_evaluation",
            "configuration_sha256": canonical_hash({"unexecuted": True, "frozen_artifact": row["artifact_sha256"]}),
            "checkpoint": checkpoint}
        return identity, {"records": [], "grades": [], "summary": {"technically_valid": False}, "_v2_dispatched_item_ids": []}
    manifest = read_json(manifest_path)
    config = read_json(session / "baseline-config.json")
    if (manifest["artifact_sha256"] != row["artifact_sha256"]
            or manifest["items_sha256"] != canonical_hash(items)
            or manifest["study_manifest_sha256"] != canonical_hash(study)
            or manifest["policy"] != study["config"]["grading_policy"]
            or config["configurationSha256"] != manifest["configuration_sha256"]
            or config["configurationSha256"] != canonical_hash({k: v for k, v in config.items() if k != "configurationSha256"})
            or read_json(session / "ready.json") != {"manifest_sha256": canonical_hash(manifest)}):
        raise ValueError("Final evidence identity differs from frozen selected artifact or dataset")
    for name, digest in manifest["files"].items():
        if file_sha256(name) != digest:
            raise ValueError("Frozen final session file changed")
    identity = {"run_id": config["runId"], "profile": "browser_evaluation",
        "configuration_sha256": config["configurationSha256"], "checkpoint": manifest["checkpoint"]}
    grade_path = session / manifest["arm"] / "graded.json"
    supervisor_path = session / "execution" / (manifest["arm"] + "-1") / "supervisor-result.json"
    supervisor = read_json(supervisor_path) if supervisor_path.exists() else None
    if grade_path.exists():
        evidence = read_json(grade_path)
        if evidence["manifest_sha256"] != canonical_hash(manifest):
            raise ValueError("Final grades are not bound to their session")
    else:
        if supervisor is not None and supervisor.get("outputs"):
            raise ValueError("Observed final outputs require preserved independent grades before analysis")
        evidence = {"records": [], "grades": [], "summary": {"technically_valid": False}}
    if supervisor is not None:
        evidence["_ambiguous_dispatch_item_ids"] = [r["item_id"] for r in supervisor.get("outputs", [])
            if r.get("dispatch_outcome") == "acknowledged_or_ambiguous"]
    count = supervisor["scored_dispatches"] if supervisor is not None else 0
    if type(count) is not int or not 0 <= count <= len(items):
        raise ValueError("Final dispatch count exceeds the frozen schedule")
    evidence["_v2_dispatched_item_ids"] = [item["item_id"] for item in items[:count]]
    return identity, evidence


def analyze(root):
    """Write evaluator-only final results; editor packet exporters never read them."""
    from .ifeval_v2_study import load
    root, study, state = load(root)
    selection = state["selection"]
    if selection is None or read_json(root / "final-selection.json") != selection:
        raise ValueError("Final selected artifact identities are not frozen")
    plan = read_json(root / "final-analysis-plan.json")
    dataset = verify_frozen(study["evaluator_root"])
    items = read_json(Path(study["evaluator_root"]) / "final-items.json")
    if (plan != _plan(study, selection) or plan["selection_sha256"] != canonical_hash(selection)
            or plan["final_manifest_sha256"] != dataset["sha256"]
            or plan["ordered_items_sha256"] != canonical_hash(items)
            or plan["analysis_spec"] != ANALYSIS_SPEC):
        raise ValueError("Frozen final analysis inputs changed")
    by_identity = {row["artifact_sha256"]: _evidence(root, study, row, items)
                   for row in selection["distinct_evaluations"]}
    if plan["contrasts"]:
        named = {name: by_identity[identity] for name, identity in selection["named_arms"].items()}
        contrasts = {"selected_astra_minus_pristine": compare_selected(items, plan, named["pristine_baseline"],
            named["selected_astra"], alias="selected_astra_minus_pristine")}
    else:
        contrasts = {}
    development = []
    for round_number in range(1, 5):
        slot = state["slots"][f"r{round_number}-astra"]
        development.append({"round": round_number,
            "status": slot["status"], "strict_passes": slot.get("strict_passes"),
            "scored_output_tokens": slot.get("scored_output_tokens"),
            "complete": slot.get("completed") is True and slot.get("technically_valid") is True})
    result = {
        "schema_version": 1, "kind": "ifeval_browser_v2_final_analysis", "label": LABEL,
        "selection_sha256": canonical_hash(selection), "analysis_plan_sha256": canonical_hash(plan),
        "named_artifacts": selection["named_arms"], "distinct_final_artifacts": len(by_identity),
        "scheduled_prompts_per_named_arm": len(items) if plan["contrasts"] else 0,
        "scheduled_instructions_per_named_arm": plan["counts"]["instructions"] if plan["contrasts"] else 0,
        "frozen_final_counts": plan["counts"], "contrasts": contrasts, "development_rounds": development,
        "status": "unavailable_no_eligible_official_final_items" if not items else
                  "skipped_pristine_retained" if not plan["contrasts"] else
                  "complete" if all(c["status"] == "complete" for c in contrasts.values()) else "incomplete",
        "design": "one adaptive search trajectory with at most four manual submissions; no independent editor replications",
        "primary_contrast": "selected_astra_minus_pristine", "multiple_comparison_claim": False,
        "final_outcomes_returned_to_editor": False, "new_generations": 0,
    }
    save(root / "final-analysis.json", result)
    return result
