"""Frozen final-subset analysis; no inference, grading, selection, or retries.

The existing paired pipeline supplies prompt estimators. Instruction intervals
resample prompts as clusters and recompute the official count ratio, rather
than averaging per-prompt percentages. Authorship: Arhaan2.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import numpy as np

from .analysis_pipeline import compare_graded_runs, project_arm, validate_analysis_manifest
from .inventory import read_json
from .protocol import canonical_hash


ANALYSIS_SPEC = {
    "schema_version": 1,
    "primary_metric": "prompt_level_strict_accuracy",
    "secondary_metrics": ["prompt_level_loose_accuracy",
        "instruction_level_strict_accuracy", "instruction_level_loose_accuracy"],
    "prompt_estimator": "existing compare_graded_runs; one seed per prompt",
    "instruction_estimator": "sum compliant instructions / sum scheduled instructions",
    "bootstrap": {"samples": 10000, "seed": 0, "confidence": 0.95,
        "method": "paired percentile bootstrap", "unit": "prompt cluster",
        "order": "lexicographic item_id, matching the existing paired pipeline",
        "instruction_resample": "resample 128 paired prompts with replacement; sum compliant instruction counts and divide by sum instruction counts in each resample",
        "quantile": "numpy.quantile default linear interpolation"},
    "missing_output_policy": "retain all scheduled prompts and instructions with zero credit; distinguish unexecuted slots from attempted failures and dispatched outputs not observed",
    "grader_error_policy": "retain zero-credit arm bounds; unresolved grader errors are not demonstrated model failures and the paired difference is not an identified semantic effect",
    "truncation_policy": "grade returned text under the unchanged frozen policy",
    "exact_paired_significance_test": {"available": False,
        "reason": "The existing analysis implementation has no exact paired significance test; none is added after observation."},
    "completion_rule": "complete paired technical execution and resolved grading; significance is not required",
    "selection": "candidate already frozen on development; no reselection or further edit",
}


def instruction_count_bootstrap(counts, baseline, candidate, *, samples=10000, seed=0):
    """A paired prompt-cluster interval for the instruction-weighted ratio."""
    counts, baseline, candidate = (np.asarray(v, dtype=np.int64) for v in (counts, baseline, candidate))
    if (counts.ndim != 1 or len(counts) == 0 or baseline.shape != counts.shape
            or candidate.shape != counts.shape or np.any(counts <= 0)
            or np.any(baseline < 0) or np.any(candidate < 0)
            or np.any(baseline > counts) or np.any(candidate > counts)):
        raise ValueError("instruction counts must be aligned valid prompt clusters")
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(samples, dtype=np.float64)
    batch_size = max(1, min(1024, 1_000_000 // len(counts)))
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        indexes = rng.integers(0, len(counts), size=(size, len(counts)))
        bootstrapped[start:start + size] = (candidate[indexes].sum(axis=1)
            - baseline[indexes].sum(axis=1)) / counts[indexes].sum(axis=1)
    low, high = np.quantile(bootstrapped, [0.025, 0.975])
    denominator = int(counts.sum())
    delta = float((candidate.sum() - baseline.sum()) / denominator)
    return {"baseline": {"numerator": int(baseline.sum()), "denominator": denominator,
                         "value": float(baseline.sum() / denominator)},
        "candidate": {"numerator": int(candidate.sum()), "denominator": denominator,
                      "value": float(candidate.sum() / denominator)},
        "delta": delta, "difference_percentage_points": 100 * delta,
        "confidence_interval": {"low": float(low), "high": float(high), "level": 0.95,
            "low_percentage_points": 100 * float(low), "high_percentage_points": 100 * float(high),
            "method": "paired percentile bootstrap of instruction-count ratio clustered by prompt",
            "resamples": samples, "seed": seed, "informative": len(counts) > 1},
        "weighting": "compliant instruction counts / all scheduled instruction counts"}


def _validate_pair(strict, loose):
    strict, loose = validate_analysis_manifest(strict), validate_analysis_manifest(loose)
    if strict["benchmark"] != "ifeval" or strict["metric"] != "prompt_level_strict_accuracy" or loose["metric"] != "prompt_level_loose_accuracy":
        raise ValueError("strict and loose IFEval prompt manifests are required")
    omit = {"metric", "manifest_sha256"}
    if {k: v for k, v in strict.items() if k not in omit} != {k: v for k, v in loose.items() if k not in omit}:
        raise ValueError("strict and loose manifests must bind identical evidence and schedule")
    if len({r["item_id"] for r in strict["schedule"]}) != len(strict["schedule"]):
        raise ValueError("final subset analysis requires exactly one scheduled seed per prompt")
    return strict, loose


def _ambiguous_items(evidence):
    return set(evidence.get("_ambiguous_dispatch_item_ids", [])) | {
        row["item_id"] for row in evidence.get("item_metrics", [])
        if (row.get("browser_observation") or {}).get("dispatch_outcome") == "acknowledged_or_ambiguous"}


def _arm_accounting(evidence, manifest, *, dispatched_items=None):
    records, grades = evidence["records"], evidence["grades"]
    expected = {row["item_id"] for row in manifest["schedule"]}
    indexed = {r["item_id"]: r for r in records}
    observed_attempts = {r["item_id"] for r in records if r["attempts"]}
    dispatched = observed_attempts if dispatched_items is None else set(dispatched_items)
    if dispatched - expected or observed_attempts - dispatched:
        raise ValueError("dispatch accounting differs from scheduled observed attempts")
    unexecuted = expected - dispatched
    no_output = {i for i in dispatched if i not in indexed or indexed[i]["raw_output"] is None}
    ambiguous = _ambiguous_items(evidence)
    if ambiguous - dispatched or any(indexed[i]["raw_output"] is not None for i in ambiguous if i in indexed):
        raise ValueError("ambiguous-dispatch accounting contradicts observed evidence")
    status = Counter(r["status"] for r in records if r["attempts"])
    grader_status = Counter(g["status"] for g in grades)
    attempted = [r for r in records if r["attempts"]]
    return {"scheduled_prompts": len(expected), "generation_dispatches": len(dispatched),
        "unexecuted_slots": len(unexecuted), "dispatched_without_observed_output": len(no_output),
        "ambiguous_dispatch_outcomes": len(ambiguous),
        "attempted_model_failures": sum(r["status"] in {"error", "timeout"}
            and r["item_id"] not in ambiguous for r in records if r["attempts"]),
        "natural_completions": status["ok"], "truncations": status["truncated"],
        "missing_outputs": len(unexecuted) + len(no_output),
        "grader_errors": grader_status["grader_error"], "grader_timeouts": grader_status["grader_timeout"],
        "graded_outputs": grader_status["graded"],
        "known_generated_tokens": sum(r["token_counts"]["output"] or 0 for r in attempted),
        "known_input_tokens": sum(r["token_counts"]["input"] or 0 for r in attempted),
        "attempted_outputs_with_unknown_tokens": sum(r["token_counts"]["output"] is None for r in attempted),
        "known_response_seconds": sum(r["duration_seconds"] or 0 for r in attempted),
        "attempted_outputs_with_unknown_duration": sum(r["duration_seconds"] is None for r in attempted),
        "technical_execution_complete": evidence.get("summary", {}).get("technically_valid") is True
            and len(attempted) == len(expected) and not (no_output or status["error"] or status["timeout"]),
        "unexecuted_item_ids": sorted(unexecuted), "unobserved_dispatched_item_ids": sorted(no_output)}


def compare_evidence(strict_manifest, loose_manifest, baseline, candidate, *, dispatched_items=None):
    """Analyze independently graded evidence, retaining absent arms and slots."""
    strict, loose = _validate_pair(strict_manifest, loose_manifest)
    arms = {"baseline": baseline, "candidate": candidate}
    # The browser adapter preserves unexecuted slots as synthetic status=missing
    # records with no attempts. Represent these as absent records in the existing
    # projection so its model_failures category never describes an unexecuted
    # request. The complete frozen manifest still retains every denominator.
    projected = {}
    for arm, evidence in arms.items():
        # Validate preserved placeholders and their grades before selecting the
        # compact input; filtering must not hide stale or off-schedule evidence.
        project_arm(evidence["records"], evidence["grades"], strict, arm=arm)
        ambiguous = _ambiguous_items(evidence)
        absent = {(r["item_id"], r["seed"]) for r in evidence["records"]
                  if (r["status"] == "missing" and not r["attempts"]) or r["item_id"] in ambiguous}
        projected[arm] = {
            "records": [r for r in evidence["records"] if (r["item_id"], r["seed"]) not in absent],
            "grades": [g for g in evidence["grades"] if (g["item_id"], g["seed"]) not in absent],
        }
    comparisons = {mode: compare_graded_runs(projected["baseline"]["records"], projected["candidate"]["records"],
        projected["baseline"]["grades"], projected["candidate"]["grades"], manifest,
        bootstrap_samples=ANALYSIS_SPEC["bootstrap"]["samples"], seed=ANALYSIS_SPEC["bootstrap"]["seed"])
        for mode, manifest in (("strict", strict), ("loose", loose))}
    accounting = {arm: _arm_accounting(evidence, strict,
        dispatched_items=None if dispatched_items is None else dispatched_items[arm])
        for arm, evidence in arms.items()}
    by_record = {arm: {r["item_id"]: r for r in evidence["records"]} for arm, evidence in arms.items()}
    by_grade = {arm: {g["item_id"]: g for g in evidence["grades"]} for arm, evidence in arms.items()}
    ordered = sorted(strict["items"], key=lambda row: row["item_id"])
    counts = [row["instruction_count"] for row in ordered]
    instruction = {}
    per_item, paired = [], {}
    for mode in ("strict", "loose"):
        values = {arm: [] for arm in arms}
        changes = Counter()
        for source in ordered:
            item_id = source["item_id"]
            resolved, flags = True, {}
            for arm in arms:
                rec, grade = by_record[arm].get(item_id), by_grade[arm].get(item_id)
                valid = rec is not None and rec["status"] in {"ok", "truncated"} and grade is not None and grade["status"] == "graded"
                credit = valid and not (rec["status"] == "truncated" and strict["policy"]["truncation_policy"] == "fail")
                values[arm].append(grade["metrics"]["instruction_" + mode + "_correct"] if credit else 0)
                flags[arm] = bool(credit and grade["metrics"]["prompt_level_" + mode + "_accuracy"])
                resolved = resolved and valid
            outcome = ("unresolved" if not resolved else "improved" if flags["candidate"] > flags["baseline"]
                       else "regressed" if flags["candidate"] < flags["baseline"] else "tied")
            changes[outcome] += 1
            per_item.append({"item_id": item_id, "mode": mode, "outcome": outcome,
                "baseline": flags["baseline"] if resolved else None,
                "candidate": flags["candidate"] if resolved else None})
        instruction[mode] = instruction_count_bootstrap(counts, values["baseline"], values["candidate"])
        for arm in arms:
            official = comparisons[mode][arm + "_projection"]["official_metrics"]["instruction_level_" + mode + "_accuracy"]
            if instruction[mode][arm] != official:
                raise ValueError("cluster-count aggregate differs from independently bound official metric")
        paired[mode] = {key: changes[key] for key in ("improved", "regressed", "tied", "unresolved")}
    semantic_complete = all(c["semantic_grading_complete"] for c in comparisons.values())
    complete = semantic_complete and all(a["technical_execution_complete"] for a in accounting.values())
    return {"schema_version": 1, "kind": "ifeval_heldout_subset_analysis", "author": "Arhaan2",
        "label": "128-example IFEval on-device subset", "analysis_spec": copy.deepcopy(ANALYSIS_SPEC),
        "analysis_spec_sha256": canonical_hash(ANALYSIS_SPEC),
        "status": "complete" if complete else "incomplete", "complete_paired_evaluation": complete,
        "semantic_grading_complete": semantic_complete,
        "interpretation": ("complete paired descriptive instruction-compliance comparison" if complete
            else "incomplete paired comparison; retain every scheduled slot; no completed held-out effect claim"),
        "grader_error_interpretation": None if semantic_complete else "unresolved evaluator failures produce zero-credit arm bounds, not demonstrated model failures or an identified semantic difference",
        "scheduled_prompts": len(ordered), "scheduled_instructions": sum(counts),
        "arm_accounting": accounting, "prompt_comparisons": comparisons,
        "instruction_comparisons": instruction, "paired_outcomes": paired, "per_item_outcomes": per_item,
        "exact_paired_significance_test": copy.deepcopy(ANALYSIS_SPEC["exact_paired_significance_test"])}


def analyze(root):
    """Read frozen manifests and completed/partial evidence; do not write it.

    Required files: evaluation-items.json; paired-analysis-{strict,loose}.json;
    {baseline,candidate}/manifest.json. Graded arm files are optional only when
    no outputs exist. Execution supervisors supply consumed dispatch slots.
    """
    root = Path(root)
    strict, loose = (read_json(root / ("paired-analysis-" + mode + ".json")) for mode in ("strict", "loose"))
    strict, loose = _validate_pair(strict, loose)
    items = read_json(root / "evaluation-items.json")
    refs = {i["item_id"]: i for i in strict["items"]}
    if (strict["split"] != "test" or strict["stage"] != "full" or len(items) != 128
            or sum(len(i["target"]["instruction_id_list"]) for i in items) != 173
            or len({i["item_id"] for i in items}) != len(items)
            or {i["item_id"] for i in items} != set(refs)
            or any(canonical_hash(i) != refs[i["item_id"]]["item_sha256"] for i in items)):
        raise ValueError("held-out analysis does not bind the frozen 128-prompt / 173-instruction test subset")
    evidence, dispatches = {}, {}
    for arm in ("baseline", "candidate"):
        arm_root = root / arm
        manifest = read_json(arm_root / "manifest.json")
        grade_path = arm_root / arm / "graded.json"
        supervisor_path = arm_root / "execution" / (arm + "-1") / "supervisor-result.json"
        supervisor = read_json(supervisor_path) if supervisor_path.exists() else None
        if grade_path.exists():
            evidence[arm] = read_json(grade_path)
            if evidence[arm].get("manifest_sha256") != canonical_hash(manifest):
                raise ValueError("graded evidence differs from its frozen arm manifest")
        else:
            if supervisor is not None and supervisor.get("outputs"):
                raise ValueError("observed outputs require explicit independent grading before analysis")
            evidence[arm] = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        if supervisor is not None:
            evidence[arm]["_ambiguous_dispatch_item_ids"] = [o["item_id"] for o in supervisor.get("outputs", [])
                if o.get("dispatch_outcome") == "acknowledged_or_ambiguous"]
        count = supervisor["scored_dispatches"] if supervisor is not None else 0
        if type(count) is not int or not 0 <= count <= 128:
            raise ValueError("supervisor scored-dispatch count exceeds the frozen schedule")
        dispatches[arm] = [i["item_id"] for i in items[:count]]
    return compare_evidence(strict, loose, evidence["baseline"], evidence["candidate"], dispatched_items=dispatches)
