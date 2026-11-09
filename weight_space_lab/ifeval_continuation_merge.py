"""Provenance-aware join of preserved candidate outputs and an authorized tail.

Preparation creates no ledger and dispatches nothing. A future merge validates
each source under its own run/configuration identity, then joins only compact
projections. Original records and grades are never rewritten or regraded.
Arhaan2.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .analysis_pipeline import build_analysis_manifest, project_arm
from .evaluation import compare_paired
from .ifeval_heldout_analysis import ANALYSIS_SPEC, instruction_count_bootstrap
from .inventory import file_sha256, read_json
from .protocol import canonical_hash
from .records import run_record_sha256


PREFIX_COUNT = 94
TAIL_COUNT = 34


def _identity(manifest, config):
    if manifest["configuration_sha256"] != config["configurationSha256"]:
        raise ValueError("source manifest and runtime configuration identities differ")
    return {"run_id": config["runId"], "profile": "browser_evaluation",
        "configuration_sha256": config["configurationSha256"],
        "checkpoint": copy.deepcopy(manifest["checkpoint"])}


def _source_manifest(full, items, *, candidate_identity=None, mode="strict"):
    """Bind one disjoint source segment without altering a raw-record identity."""
    split = {"schema_version": 1, "kind": "benchmark_split", "split": "test",
        "source_manifest_sha256": full["source_manifest_sha256"],
        "items": [{"benchmark": i["benchmark"], "item_id": i["item_id"],
            "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in items]}
    arms = copy.deepcopy(full["arms"])
    if candidate_identity is not None:
        arms["candidate"] = candidate_identity
    return build_analysis_manifest(split, items, benchmark="ifeval", seeds=[42],
        policy=full["policy"], arms=arms, stage="full", evidence_kind=full["evidence_kind"],
        metric="prompt_level_" + mode + "_accuracy")


def _read_evidence(directory, arm):
    manifest = read_json(directory / "manifest.json")
    config = read_json(directory / "baseline-config.json")
    _identity(manifest, config)
    evidence = read_json(directory / arm / "graded.json")
    if evidence["manifest_sha256"] != canonical_hash(manifest):
        raise ValueError("graded evidence no longer binds its original source manifest")
    for key in ("records", "grades"):
        if evidence[key] != read_json(directory / arm / (key + ".json")):
            raise ValueError("preserved graded bundle differs from its original separate evidence file")
    return manifest, config, evidence


def _select(evidence, ids):
    """Return references to unchanged objects, not rewritten raw records."""
    wanted = set(ids)
    return {"records": [r for r in evidence["records"] if r["item_id"] in wanted],
        "grades": [g for g in evidence["grades"] if g["item_id"] in wanted]}


def _evidence_refs(evidence, ids):
    records = {r["item_id"]: r for r in evidence["records"]}
    grades = {g["item_id"]: g for g in evidence["grades"]}
    return [{"item_id": item_id, "run_id": records[item_id]["run_id"],
        "configuration_sha256": records[item_id]["configuration_sha256"],
        "run_record_sha256": run_record_sha256(records[item_id]),
        "grade_record_sha256": canonical_hash(grades[item_id])} for item_id in ids]


def _old_sources(root):
    items = read_json(root / "evaluation-items.json")
    full = read_json(root / "paired-analysis-strict.json")
    bm, bc, baseline = _read_evidence(root / "baseline", "baseline")
    cm, cc, candidate = _read_evidence(root / "candidate", "candidate")
    ids = [i["item_id"] for i in items]
    references = {i["item_id"]: i for i in full["items"]}
    if (len(ids) != 128 or len(set(ids)) != 128
            or sum(len(i["target"]["instruction_id_list"]) for i in items) != 173
            or sum(len(i["target"]["instruction_id_list"]) for i in items[PREFIX_COUNT:]) != 45
            or set(ids) != set(references)
            or any(canonical_hash(i) != references[i["item_id"]]["item_sha256"] for i in items)):
        raise ValueError("original 128-prompt/173-instruction schedule or 34-prompt/45-instruction tail changed")
    if (_identity(bm, bc) != full["arms"]["baseline"]
            or _identity(cm, cc) != full["arms"]["candidate"]):
        raise ValueError("original arm source identity differs from frozen analysis")
    for arm, evidence in (("baseline", baseline), ("candidate", candidate)):
        if any([row["item_id"] for row in evidence[key]] != ids for key in ("records", "grades")):
            raise ValueError("original evidence does not preserve the full frozen order")
        project_arm(evidence["records"], evidence["grades"], full, arm=arm)
    if (not baseline["summary"]["technically_valid"]
            or any(r["status"] not in {"ok", "truncated"} for r in baseline["records"])
            or any(g["status"] != "graded" for g in baseline["grades"])):
        raise ValueError("original baseline must already be complete and graded")
    if (any(r["status"] not in {"ok", "truncated"} for r in candidate["records"][:PREFIX_COUNT])
            or any(g["status"] != "graded" for g in candidate["grades"][:PREFIX_COUNT])
            or any(r["status"] != "missing" or r["attempts"] for r in candidate["records"][PREFIX_COUNT:])
            or any(g["status"] != "not_applicable" for g in candidate["grades"][PREFIX_COUNT:])):
        raise ValueError("original candidate is not exactly 94 completed grades and 34 unexecuted slots")
    return items, full, (bm, bc, baseline), (cm, cc, candidate)


def prepare_merge_plan(old_root, continuation_manifest, output):
    """Create a fresh source-bound join declaration; no authorization or inference."""
    old_root, continuation_manifest, output = map(lambda p: Path(p).resolve(),
                                                  (old_root, continuation_manifest, output))
    if output.exists():
        raise FileExistsError("merge preparation must not overwrite an existing plan")
    items, full, (_, _, baseline), (old_manifest, old_config, candidate) = _old_sources(old_root)
    tail_manifest = read_json(continuation_manifest)
    tail_config = read_json(continuation_manifest.parent / "baseline-config.json")
    tail_identity = _identity(tail_manifest, tail_config)
    if (Path(tail_manifest["original_root"]).resolve() != old_root
            or tail_manifest["original_freeze_sha256"] != canonical_hash(read_json(old_root / "evaluation-freeze.json"))
            or tail_manifest["original_candidate_manifest_sha256"] != canonical_hash(old_manifest)
            or tail_manifest["original_candidate_config_sha256"] != file_sha256(old_root / "candidate/baseline-config.json")):
        raise ValueError("continuation does not bind the original frozen experiment")
    if (read_json(tail_manifest["items_path"]) != items[PREFIX_COUNT:]
            or tail_config["items"] != old_config["items"][PREFIX_COUNT:]
            or tail_config["runId"] != "ifeval-browser-v1-heldout-continuation-candidate"
            or tail_identity["configuration_sha256"] == full["arms"]["candidate"]["configuration_sha256"]):
        raise ValueError("continuation must bind only the unchanged 34-item tail under a new run/config identity")
    for key in ("settings", "checkpoint", "artifact_sha256", "policy", "grader"):
        if tail_manifest[key] != old_manifest[key]:
            raise ValueError("continuation changed a frozen model, decoding, or grading property")
    for key in ("settings", "warmupPrompt", "sourceCheckpointSha256", "convertedArtifactSha256"):
        if tail_config[key] != old_config[key]:
            raise ValueError("continuation runtime settings or artifact differ from the original candidate")
    pin_paths = [old_root / name for name in ("evaluation-freeze.json", "evaluation-ready.json",
        "evaluation-items.json", "paired-analysis-strict.json", "paired-analysis-loose.json",
        "analysis.json", "run-result.json")]
    for arm in ("baseline", "candidate"):
        pin_paths.extend(old_root / arm / name for name in ("manifest.json", "baseline-config.json",
            arm + "/graded.json", arm + "/records.json", arm + "/grades.json",
            "execution/" + arm + "-1/supervisor-result.json"))
    pin_paths.extend((continuation_manifest, continuation_manifest.parent / "baseline-config.json",
                      Path(tail_manifest["items_path"])))
    plan = {"schema_version": 1, "kind": "ifeval_provenance_aware_continuation_merge_plan",
        "author": "Arhaan2", "inference_authorized": False, "old_root": str(old_root),
        "continuation_manifest_path": str(continuation_manifest),
        "preserved_file_sha256": {str(p): file_sha256(p) for p in pin_paths},
        "original_analysis_spec_sha256": canonical_hash(ANALYSIS_SPEC),
        "scheduled_prompts": 128, "scheduled_instructions": 173,
        "prefix_count": PREFIX_COUNT, "tail_count": TAIL_COUNT, "tail_instructions": 45,
        "execution_order": [i["item_id"] for i in items],
        "candidate_prefix_item_ids": [i["item_id"] for i in items[:PREFIX_COUNT]],
        "candidate_tail_item_ids": [i["item_id"] for i in items[PREFIX_COUNT:]],
        "original_baseline_record_grade_references": _evidence_refs(baseline, [i["item_id"] for i in items]),
        "original_candidate_record_grade_references": _evidence_refs(candidate, [i["item_id"] for i in items[:PREFIX_COUNT]]),
        "tail_identity": tail_identity,
        "source_projection_manifests": {mode: {
            "baseline": read_json(old_root / ("paired-analysis-" + mode + ".json")),
            "candidate_prefix": _source_manifest(full, items[:PREFIX_COUNT], mode=mode),
            "candidate_tail": _source_manifest(full, items[PREFIX_COUNT:], candidate_identity=tail_identity, mode=mode)}
            for mode in ("strict", "loose")},
        "merge_policy": "validate each source under its original run/config identity; join only disjoint compact projections; retain all 128 scheduled prompts and 173 instructions; never rewrite or regrade original records",
        "limitations": "A resumed two-session candidate is a manually continued evaluation, not the originally planned one-session comparison. Preserve the original incomplete report and disclose the interruption and extra warmup."}
    plan["plan_sha256"] = canonical_hash(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return plan


def _join_projections(baseline, prefix, tail, full_manifest):
    """Join only already validated compact sources, enforcing disjoint coverage."""
    expected = {(s["item_id"], s["seed"]) for s in full_manifest["schedule"]}
    rows = prefix["records"] + tail["records"]
    keys = [(r["item_id"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("prefix and tail projections must be disjoint and cover the entire frozen schedule")
    result = compare_paired(baseline["records"], rows,
        schedule=[{"item_id": s["item_id"], "seed": s["seed"]} for s in full_manifest["schedule"]],
        truncation_policy=full_manifest["policy"]["truncation_policy"],
        bootstrap_samples=ANALYSIS_SPEC["bootstrap"]["samples"], seed=ANALYSIS_SPEC["bootstrap"]["seed"])
    official = {}
    for metric in baseline["official_metrics"]:
        left, right = prefix["official_metrics"][metric], tail["official_metrics"][metric]
        denominator = left["denominator"] + right["denominator"]
        numerator = left["numerator"] + right["numerator"]
        if denominator != baseline["official_metrics"][metric]["denominator"]:
            raise ValueError("joined official metric denominator differs from the full schedule")
        official[metric] = {"baseline": baseline["official_metrics"][metric],
            "candidate": {"numerator": numerator, "denominator": denominator, "value": numerator / denominator}}
    return result, official


def merge_evidence(old_root, tail_root, plan):
    """Future read-only join after explicit authorization, execution and grading.

    ``tail_root`` is the directory containing the continuation manifest and
    ``candidate/graded.json``. This function never creates missing generations.
    """
    old_root, tail_root = Path(old_root).resolve(), Path(tail_root).resolve()
    plan = read_json(plan) if isinstance(plan, (str, Path)) else copy.deepcopy(plan)
    if (plan["plan_sha256"] != canonical_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan["original_analysis_spec_sha256"] != canonical_hash(ANALYSIS_SPEC)
            or Path(plan["old_root"]) != old_root
            or Path(plan["continuation_manifest_path"]).parent != tail_root):
        raise ValueError("merge plan identity changed")
    for path, digest in plan["preserved_file_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("a preserved source file changed after merge preparation")
    items, full, (_, _, baseline), (_, _, candidate) = _old_sources(old_root)
    _, _, tail = _read_evidence(tail_root, "candidate")
    tail_ids = plan["candidate_tail_item_ids"]
    if any([row["item_id"] for row in tail[key]] != tail_ids for key in ("records", "grades")):
        raise ValueError("continuation evidence must retain exactly the 34 frozen tail slots in order")
    if (_evidence_refs(baseline, plan["execution_order"]) != plan["original_baseline_record_grade_references"]
            or _evidence_refs(candidate, plan["candidate_prefix_item_ids"]) != plan["original_candidate_record_grade_references"]):
        raise ValueError("an original record or grade identity changed")
    prefix = _select(candidate, plan["candidate_prefix_item_ids"])
    source_evidence = {"baseline": baseline, "candidate_prefix": prefix, "candidate_tail": tail}
    modes, instruction, paired_outcomes = {}, {}, {}
    for mode in ("strict", "loose"):
        manifests = plan["source_projection_manifests"][mode]
        projections = {name: project_arm(source_evidence[name]["records"], source_evidence[name]["grades"],
            manifest, arm="baseline" if name == "baseline" else "candidate") for name, manifest in manifests.items()}
        comparison, official = _join_projections(projections["baseline"], projections["candidate_prefix"],
                                                projections["candidate_tail"], manifests["baseline"])
        modes[mode] = {"comparison": comparison, "official_metrics": official,
            "source_projection_manifest_sha256": {k: v["manifest_sha256"] for k, v in manifests.items()}}
        grade_maps = {"baseline": {g["item_id"]: g for g in baseline["grades"]},
            "candidate": {g["item_id"]: g for g in prefix["grades"] + tail["grades"]}}
        record_maps = {"baseline": {r["item_id"]: r for r in baseline["records"]},
            "candidate": {r["item_id"]: r for r in prefix["records"] + tail["records"]}}
        ordered = sorted(items, key=lambda i: i["item_id"])
        values = {}
        for arm in ("baseline", "candidate"):
            values[arm] = []
            for item in ordered:
                grade, record = grade_maps[arm][item["item_id"]], record_maps[arm][item["item_id"]]
                credit = grade["status"] == "graded" and record["status"] in {"ok", "truncated"}
                credit = credit and not (record["status"] == "truncated" and full["policy"]["truncation_policy"] == "fail")
                values[arm].append(grade["metrics"]["instruction_" + mode + "_correct"] if credit else 0)
        instruction[mode] = instruction_count_bootstrap([len(i["target"]["instruction_id_list"]) for i in ordered],
                                                       values["baseline"], values["candidate"])
        for arm in ("baseline", "candidate"):
            if instruction[mode][arm] != official["instruction_level_" + mode + "_accuracy"][arm]:
                raise ValueError("joined instruction bootstrap count differs from bound official aggregation")
        outcomes = dict.fromkeys(("improved", "regressed", "tied", "unresolved"), 0)
        for item in ordered:
            item_id = item["item_id"]
            if any(grade_maps[arm][item_id]["status"] != "graded"
                   or record_maps[arm][item_id]["status"] not in {"ok", "truncated"}
                   for arm in ("baseline", "candidate")):
                outcomes["unresolved"] += 1
                continue
            values = {arm: int(grade_maps[arm][item_id]["metrics"]["prompt_level_" + mode + "_accuracy"])
                if not (record_maps[arm][item_id]["status"] == "truncated"
                        and full["policy"]["truncation_policy"] == "fail") else 0
                for arm in ("baseline", "candidate")}
            outcomes["improved" if values["candidate"] > values["baseline"] else
                     "regressed" if values["candidate"] < values["baseline"] else "tied"] += 1
        paired_outcomes[mode] = outcomes
    semantic_complete = not any(g["status"] in {"grader_error", "grader_timeout"} for g in tail["grades"])
    complete = (tail["summary"].get("technically_valid") is True
        and all(r["status"] in {"ok", "truncated"} for r in tail["records"])
        and all(g["status"] == "graded" for g in tail["grades"]))
    return {"schema_version": 1, "kind": "ifeval_provenance_aware_continued_comparison",
        "author": "Arhaan2", "status": "complete_manually_continued" if complete else "incomplete",
        "label": "128-example IFEval on-device subset", "merge_plan_sha256": plan["plan_sha256"],
        "scheduled_prompts": 128, "scheduled_instructions": 173,
        "original_records_rewritten": 0, "original_outputs_regraded": 0,
        "candidate_source_runs": [full["arms"]["candidate"], plan["tail_identity"]],
        "prompt_comparisons": modes, "instruction_comparisons": instruction,
        "paired_outcomes": paired_outcomes, "semantic_grading_complete": semantic_complete,
        "interpretation": ("complete manually continued paired compliance comparison; disclose two candidate sessions" if complete
            else "incomplete continued comparison; all scheduled denominators retained; no completed held-out effect claim"),
        "grader_error_interpretation": None if semantic_complete else "unresolved evaluator failures produce zero-credit arm bounds, not demonstrated model failures or an identified semantic difference",
        "candidate_source_record_grade_references": plan["original_candidate_record_grade_references"] + _evidence_refs(tail, tail_ids),
        "tail_unexecuted_slots": sum(not r["attempts"] for r in tail["records"]),
        "tail_grader_errors": sum(g["status"] in {"grader_error", "grader_timeout"} for g in tail["grades"]),
        "exact_paired_significance_test": copy.deepcopy(ANALYSIS_SPEC["exact_paired_significance_test"]),
        "limitations": plan["limitations"]}
