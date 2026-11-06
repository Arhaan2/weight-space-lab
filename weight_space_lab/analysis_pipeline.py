"""Frozen, denominator-preserving projection of independent grading evidence.

Raw generations and grader diagnostics never enter the compact paired schema.
A hash binds declarations; it is not proof of isolation or pre-registration.
"""
from __future__ import annotations

import copy
from collections import Counter

from .evaluation import compare_paired, validate_records
from .protocol import build_benchmark_schedule, canonical_hash, validate_benchmark_schedule
from .records import (_choice, _digest, _integer, _object, _text, run_record_sha256,
                      validate_run_records, RUN_PROFILES)

_IFEVAL_METRICS = {
    "prompt_level_strict_accuracy", "prompt_level_loose_accuracy",
    "instruction_level_strict_accuracy", "instruction_level_loose_accuracy",
}
_MANIFEST_FIELDS = {
    "schema_version", "kind", "frozen", "benchmark", "split", "stage",
    "evidence_kind", "split_manifest_sha256", "source_manifest_sha256", "items",
    "schedule", "schedule_sha256", "policy", "policy_sha256", "grader", "metric",
    "arms", "manifest_sha256",
}


def _manifest_hash(manifest):
    return canonical_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def _validate_arms(arms, stage, split):
    _object(arms, {"baseline", "candidate"}, "analysis arms")
    for arm in arms.values():
        _object(arm, {"run_id", "profile", "configuration_sha256", "checkpoint"}, "analysis arm")
        _text(arm["run_id"], "arm run_id")
        _choice(arm["profile"], RUN_PROFILES, "arm profile")
        _digest(arm["configuration_sha256"], "arm configuration_sha256")
        _object(arm["checkpoint"], {"model_id", "revision", "sha256"}, "arm checkpoint")
        _text(arm["checkpoint"]["model_id"], "checkpoint model_id")
        _digest(arm["checkpoint"]["revision"], "checkpoint revision", revision=True)
        _digest(arm["checkpoint"]["sha256"], "checkpoint sha256")
        if (stage == "smoke") != (arm["profile"] == "native_development_smoke"):
            raise ValueError("smoke analysis requires the native development smoke profile exclusively")
        if arm["profile"] == "fresh_editor_search" and split == "test":
            raise ValueError("fresh editor search cannot analyze final test items")
        if arm["profile"] == "frozen_final_evaluation" and (stage != "full" or split != "test"):
            raise ValueError("frozen final analysis requires full stage and test split")
    if arms["baseline"]["run_id"] == arms["candidate"]["run_id"]:
        raise ValueError("paired arms require distinct run identities")


def build_analysis_manifest(split_manifest, items, *, benchmark, seeds, policy, arms,
                            stage, metric="accuracy", evidence_kind="real_model"):
    """Freeze every item of one benchmark in an explicit dataset split.

    ``items`` contains all canonical items selected from ``split_manifest``.
    No observed run record participates in constructing the denominator.
    Persist this manifest before generation and preserve it with both arms.
    """
    from .benchmark_data import validate_item
    from .grading import grader_identity, validate_grading_policy

    _object(split_manifest, {"schema_version", "kind", "split", "items", "source_manifest_sha256"}, "split manifest")
    if type(split_manifest["schema_version"]) is not int or split_manifest["schema_version"] != 1 or split_manifest["kind"] != "benchmark_split":
        raise ValueError("unsupported benchmark split manifest")
    if not isinstance(split_manifest["items"], list) or not split_manifest["items"]:
        raise ValueError("split manifest items must be a nonempty list")
    _digest(split_manifest["source_manifest_sha256"], "source_manifest_sha256")
    refs = {}
    for ref in split_manifest["items"]:
        _object(ref, {"benchmark", "item_id", "dataset_revision", "item_sha256"}, "split item reference")
        _choice(ref["benchmark"], {"math", "gsm8k", "ifeval"}, "split benchmark")
        _text(ref["item_id"], "split item identity")
        _digest(ref["dataset_revision"], "split dataset revision", revision=True)
        _digest(ref["item_sha256"], "split item hash")
        key = ref["benchmark"], ref["item_id"]
        if key in refs:
            raise ValueError("duplicate item in split manifest")
        refs[key] = ref
    if not isinstance(items, list):
        raise ValueError("canonical items must be a list")
    selected = {}
    for source_item in items:
        item = validate_item(source_item)
        key = item["benchmark"], item["item_id"]
        if key in selected or key not in refs:
            raise ValueError("canonical item is duplicate or outside split manifest")
        if item["split"] != split_manifest["split"] or item["dataset_revision"] != refs[key]["dataset_revision"] or canonical_hash(item) != refs[key]["item_sha256"]:
            raise ValueError("canonical item disagrees with frozen split reference")
        selected[key] = item
    expected = {key for key in refs if key[0] == benchmark}
    if not expected or {key for key in selected if key[0] == benchmark} != expected:
        raise ValueError("all selected benchmark items in the split manifest are required")
    item_refs = []
    for key in sorted(expected):
        item = selected[key]
        instruction_count = len(item["target"]["instruction_id_list"]) if benchmark == "ifeval" else None
        item_refs.append({**copy.deepcopy(refs[key]), "instruction_count": instruction_count})
    schedule = build_benchmark_schedule(benchmark, [ref["item_id"] for ref in item_refs], seeds)
    checked_policy = validate_grading_policy(policy)
    # Validators may validate in place or return a copy; persist the actual declaration.
    if checked_policy is None:
        checked_policy = policy
    manifest = {
        "schema_version": 1, "kind": "paired_analysis_manifest", "frozen": True,
        "benchmark": benchmark, "split": split_manifest["split"], "stage": stage,
        "evidence_kind": evidence_kind, "split_manifest_sha256": canonical_hash(split_manifest),
        "source_manifest_sha256": split_manifest["source_manifest_sha256"], "items": item_refs,
        "schedule": schedule, "schedule_sha256": validate_benchmark_schedule(schedule)["schedule_sha256"],
        "policy": copy.deepcopy(checked_policy), "policy_sha256": canonical_hash(checked_policy),
        "grader": grader_identity(benchmark), "metric": metric, "arms": copy.deepcopy(arms),
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return validate_analysis_manifest(manifest)


def validate_analysis_manifest(manifest):
    """Check internal consistency of a previously frozen analysis manifest."""
    from .grading import grader_identity, validate_grading_policy

    _object(manifest, _MANIFEST_FIELDS, "analysis manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["kind"] != "paired_analysis_manifest" or manifest["frozen"] is not True:
        raise ValueError("analysis requires a frozen version 1 paired analysis manifest")
    _choice(manifest["benchmark"], {"math", "gsm8k", "ifeval"}, "benchmark")
    _choice(manifest["split"], {"development", "validation", "test"}, "split")
    _choice(manifest["stage"], {"smoke", "pilot", "full"}, "stage")
    _choice(manifest["evidence_kind"], {"fixture", "real_model"}, "evidence_kind")
    if manifest["stage"] == "smoke" and manifest["split"] != "development":
        raise ValueError("smoke analysis may use development items only")
    if manifest["split"] == "test" and manifest["stage"] != "full":
        raise ValueError("final-test analysis requires full stage")
    _validate_arms(manifest["arms"], manifest["stage"], manifest["split"])
    for name in ("split_manifest_sha256", "source_manifest_sha256", "schedule_sha256", "policy_sha256", "manifest_sha256"):
        _digest(manifest[name], name)
    if _manifest_hash(manifest) != manifest["manifest_sha256"]:
        raise ValueError("analysis manifest digest mismatch")
    validate_grading_policy(manifest["policy"])
    if canonical_hash(manifest["policy"]) != manifest["policy_sha256"]:
        raise ValueError("grading policy digest mismatch")
    if manifest["grader"] != grader_identity(manifest["benchmark"]):
        raise ValueError("manifest grader differs from this pinned evaluator implementation")
    metrics = _IFEVAL_METRICS if manifest["benchmark"] == "ifeval" else {"accuracy"}
    _choice(manifest["metric"], metrics, "declared benchmark metric")
    summary = validate_benchmark_schedule(manifest["schedule"])
    if set(summary["benchmarks"]) != {manifest["benchmark"]} or summary["schedule_sha256"] != manifest["schedule_sha256"]:
        raise ValueError("analysis schedule identity mismatch")
    if not isinstance(manifest["items"], list) or not manifest["items"]:
        raise ValueError("analysis items must be a nonempty list")
    seen = set()
    for item in manifest["items"]:
        _object(item, {"benchmark", "item_id", "dataset_revision", "item_sha256", "instruction_count"}, "analysis item")
        _text(item["item_id"], "item identity")
        if item["benchmark"] != manifest["benchmark"] or item["item_id"] in seen:
            raise ValueError("analysis item benchmark mismatch or duplicate identity")
        seen.add(item["item_id"])
        _digest(item["dataset_revision"], "dataset_revision", revision=True)
        _digest(item["item_sha256"], "item_sha256")
        if manifest["benchmark"] == "ifeval":
            _integer(item["instruction_count"], "instruction_count", positive=True)
        elif item["instruction_count"] is not None:
            raise ValueError("instruction count applies only to IFEval")
    if {row["item_id"] for row in manifest["schedule"]} != seen:
        raise ValueError("schedule must contain every frozen item, with no extra items")
    return copy.deepcopy(dict(manifest))


def project_arm(records, grades, manifest, *, arm):
    """Produce compact records and separate observed failure accounting.

    Missing grades for completed runs are incomplete grading work and fail
    closed. Explicit evaluator errors remain in the denominator as zero credit,
    with a bound-only interpretation until evaluator failures are resolved.
    """
    from .grading import validate_grade_record

    manifest = validate_analysis_manifest(manifest)
    _choice(arm, {"baseline", "candidate"}, "arm")
    expected_arm = manifest["arms"][arm]
    raw = validate_run_records(records)
    items = {row["item_id"]: row for row in manifest["items"]}
    scheduled = {(row["item_id"], row["seed"]): row for row in manifest["schedule"]}
    indexed = {}
    for row in raw:
        key = row["item_id"], row["seed"]
        if key not in scheduled or row["sample_index"] != scheduled[key]["sample_index"]:
            raise ValueError("raw record is outside the complete scheduled item/sample manifest")
        for name in ("benchmark", "split", "stage", "evidence_kind"):
            if row[name] != manifest[name]:
                raise ValueError(f"raw record {name} differs from the frozen manifest")
        for name in ("run_id", "profile", "configuration_sha256"):
            if row[name] != expected_arm[name]:
                raise ValueError(f"raw record {name} differs from the frozen arm")
        if row["retry_policy"] != manifest["policy"]["generation_retry"]:
            raise ValueError("generation retry policy differs from the frozen policy")
        if row["dataset_revision"] not in (None, items[row["item_id"]]["dataset_revision"]):
            raise ValueError("raw dataset revision differs from the frozen item")
        for name, expected in expected_arm["checkpoint"].items():
            if row["checkpoint"][name] not in (None, expected):
                raise ValueError("raw checkpoint differs from the frozen arm")
        if key in indexed:
            raise ValueError("duplicate raw scheduled result")
        indexed[key] = row
    if not isinstance(grades, list):
        raise ValueError("grader diagnostics must be a list")
    graded = {}
    for source_grade in grades:
        grade = validate_grade_record(source_grade)
        if grade is None:
            grade = source_grade
        key = grade["item_id"], grade["seed"]
        if key in graded or key not in indexed:
            raise ValueError("duplicate grade or grade without a raw record")
        row, item = indexed[key], items[grade["item_id"]]
        expected = {
            "benchmark": manifest["benchmark"], "sample_index": row["sample_index"],
            "evidence_kind": manifest["evidence_kind"], "run_record_sha256": run_record_sha256(row),
            "item_sha256": item["item_sha256"], "policy_sha256": manifest["policy_sha256"],
            "grader": manifest["grader"],
        }
        if any(grade[name] != value for name, value in expected.items()):
            raise ValueError("grader diagnostic does not bind the raw record, item, policy and pinned grader")
        if row["status"] in {"error", "timeout", "missing"}:
            if grade["status"] != "not_applicable":
                raise ValueError("failed model execution cannot have a semantic grade")
        elif grade["status"] == "not_applicable":
            raise ValueError("completed model execution requires a grading outcome")
        if grade["status"] == "graded" and manifest["benchmark"] == "ifeval" and grade["metrics"]["instruction_count"] != item["instruction_count"]:
            raise ValueError("grader instruction count differs from the frozen source item")
        graded[key] = grade
    compact, diagnostics = [], []
    counts, generation_counts = Counter(), Counter()
    official = {key: 0 for key in _IFEVAL_METRICS}
    instruction_denominator = 0
    for key, scheduled_row in sorted(scheduled.items()):
        row, grade = indexed.get(key), graded.get(key)
        generation_status = "missing" if row is None else row["status"]
        generation_counts[generation_status] += 1
        result = {"item_id": key[0], "seed": key[1], "status": generation_status, "correct": False}
        kind = None
        if row is None:
            counts["absent_raw_records"] += 1
            kind = "missing_generation"
        elif generation_status in {"error", "timeout", "missing"}:
            counts["model_failures"] += 1
            kind = "model_" + generation_status
        else:
            if grade is None:
                raise ValueError("completed raw output lacks an explicit independent grader diagnostic")
            if grade["status"] in {"grader_error", "grader_timeout"}:
                counts[grade["status"]] += 1
                kind = grade["status"]
                result["status"] = "error"
            else:
                counts["graded"] += 1
                credit = not (generation_status == "truncated" and manifest["policy"]["truncation_policy"] == "fail")
                if manifest["benchmark"] == "ifeval":
                    value = grade["metrics"][manifest["metric"]] if credit else 0
                    result["correct"] = bool(value == 1)
                    if manifest["metric"].startswith("instruction_"):
                        result["score"] = float(value)
                    if credit:
                        for name in _IFEVAL_METRICS:
                            if name.startswith("prompt_"):
                                official[name] += int(grade["metrics"][name])
                            else:
                                count_key = "instruction_strict_correct" if "strict" in name else "instruction_loose_correct"
                                official[name] += grade["metrics"][count_key]
                else:
                    result["correct"] = bool(grade["correct"] and credit)
                if not credit:
                    counts["truncation_policy_failures"] += 1
                    kind = "truncation_policy"
                elif not result["correct"]:
                    counts["graded_incorrect"] += 1
        if manifest["benchmark"] == "ifeval":
            instruction_denominator += items[key[0]]["instruction_count"]
        compact.append(result)
        diagnostics.append({"item_id": key[0], "sample_index": scheduled_row["sample_index"],
                            "seed": key[1], "generation_status": generation_status,
                            "grader_status": None if grade is None else grade["status"], "failure_kind": kind})
    unresolved = counts["grader_error"] + counts["grader_timeout"]
    report = {
        "schema_version": 1, "kind": "analysis_projection", "arm": arm,
        "evidence_kind": manifest["evidence_kind"], "scientific_result": False,
        "manifest_sha256": manifest["manifest_sha256"], "records": validate_records(compact),
        "accounting": {name: counts[name] for name in ("absent_raw_records", "model_failures", "graded", "graded_incorrect", "grader_error", "grader_timeout", "truncation_policy_failures")},
        "generation_status_counts": {name: generation_counts[name] for name in ("ok", "truncated", "error", "timeout", "missing")},
        "diagnostics": diagnostics, "semantic_grading_complete": unresolved == 0,
        "score_interpretation": "zero-credit lower bound; unresolved evaluator failures" if unresolved else "declared metric over every scheduled case",
    }
    if manifest["benchmark"] == "ifeval":
        report["official_metrics"] = {
            name: {"numerator": count, "denominator": len(scheduled) if name.startswith("prompt_") else instruction_denominator,
                   "value": count / (len(scheduled) if name.startswith("prompt_") else instruction_denominator)}
            for name, count in sorted(official.items())
        }
    return report


def compare_graded_runs(baseline_records, candidate_records, baseline_grades, candidate_grades,
                        manifest, *, bootstrap_samples=10000, seed=0):
    """Compare only projections made against the same complete frozen manifest."""
    manifest = validate_analysis_manifest(manifest)
    baseline = project_arm(baseline_records, baseline_grades, manifest, arm="baseline")
    candidate = project_arm(candidate_records, candidate_grades, manifest, arm="candidate")
    instruction_metric = manifest["metric"].startswith("instruction_")
    schedule = [{"item_id": row["item_id"], "seed": row["seed"]} for row in manifest["schedule"]]
    comparison = compare_paired(baseline["records"], candidate["records"], schedule=schedule,
                                metric="score" if instruction_metric else "accuracy",
                                truncation_policy=manifest["policy"]["truncation_policy"],
                                bootstrap_samples=bootstrap_samples, seed=seed)
    complete = baseline["semantic_grading_complete"] and candidate["semantic_grading_complete"]
    return {
        "schema_version": 1, "kind": "paired_graded_analysis", "evidence_kind": manifest["evidence_kind"],
        "stage": manifest["stage"], "scientific_result": False,
        "manifest_sha256": manifest["manifest_sha256"], "benchmark": manifest["benchmark"],
        "declared_metric": manifest["metric"],
        "paired_estimand": "equal prompt mean of instruction compliance; not official instruction-weighted accuracy" if instruction_metric else "equal item mean of scheduled sample correctness",
        "semantic_grading_complete": complete,
        "interpretation": "evaluator failures unresolved; zero-credit arm bounds, not an identified semantic effect" if not complete else "paired descriptive comparison; experimental claims require the declared controls and independent provenance",
        "comparison": comparison, "baseline_projection": baseline, "candidate_projection": candidate,
    }
