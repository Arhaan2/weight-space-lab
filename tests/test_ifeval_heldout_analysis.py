"""Independent synthetic counts; no target inference or real held-out reads."""
import copy
import unittest

import numpy as np

from weight_space_lab.analysis_pipeline import build_analysis_manifest
from weight_space_lab.grading import grader_identity, ifeval_metrics
from weight_space_lab.ifeval_heldout_analysis import (ANALYSIS_SPEC, compare_evidence,
                                                    instruction_count_bootstrap)
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.records import run_record_sha256


def fixtures():
    items = [{"benchmark": "ifeval", "item_id": str(index), "dataset_revision": "a" * 40,
        "split": "test", "task_text": "Synthetic instruction-count fixture " + str(index),
        "target": {"instruction_id_list": ["keywords:existence"] * count,
                   "kwargs": [{"keywords": ["hello"]}] * count},
        "source": {"dataset_id": "hand-authored-fixture", "revision": "a" * 40,
                   "split": "official", "row_id": str(index)}, "metadata": {}}
        for index, count in enumerate((1, 3))]
    split = {"schema_version": 1, "kind": "benchmark_split", "split": "test",
        "source_manifest_sha256": canonical_hash("synthetic source"),
        "items": [{"benchmark": i["benchmark"], "item_id": i["item_id"],
            "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in items]}
    arms = {arm: {"run_id": "fixture-" + arm, "profile": "browser_evaluation",
        "configuration_sha256": canonical_hash(arm), "checkpoint": {"model_id": "synthetic-array",
        "revision": "b" * 40, "sha256": canonical_hash(arm + " checkpoint")}}
        for arm in ("baseline", "candidate")}
    policy = {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
        "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
        "grader_timeout_seconds": 20, "grader_retry": {"mode": "none", "max_attempts": 1}}
    manifests = [build_analysis_manifest(split, items, benchmark="ifeval", seeds=[42],
        policy=policy, arms=arms, stage="full", metric="prompt_level_" + mode + "_accuracy",
        evidence_kind="fixture") for mode in ("strict", "loose")]
    return items, manifests


def evidence(items, manifest, arm, vectors, *, statuses=None):
    records, grades = [], []
    statuses = statuses or ["ok"] * len(items)
    for source, vector, status in zip(items, vectors, statuses, strict=True):
        good = status in {"ok", "truncated"}
        outcome = {"status": status, "raw_output": "Synthetic answer" if good else None,
            "finish_reason": {"ok": "stop", "truncated": "length", "error": "error", "missing": "not_started"}[status],
            "token_counts": {"input": 1, "output": 2, "total": 3} if good else dict.fromkeys(("input", "output", "total")),
            "duration_seconds": .1 if good else None,
            "error": {"code": "synthetic_failure", "retryable": False} if status == "error" else None}
        record = {"schema_version": 1, "record_type": "run", **copy.deepcopy(manifest["arms"][arm]),
            "evidence_kind": "fixture", "stage": "full", "benchmark": "ifeval", "split": "test",
            "dataset_revision": source["dataset_revision"], "item_id": source["item_id"],
            "sample_index": 0, "seed": 42, "retry_policy": copy.deepcopy(manifest["policy"]["generation_retry"]),
            "attempts": [] if status == "missing" else [{"attempt_index": 0, **copy.deepcopy(outcome)}], **outcome}
        grade = {"schema_version": 1, "record_type": "grade", "benchmark": "ifeval",
            "item_id": source["item_id"], "seed": 42, "sample_index": 0, "evidence_kind": "fixture",
            "run_record_sha256": run_record_sha256(record), "item_sha256": canonical_hash(source),
            "policy_sha256": manifest["policy_sha256"], "grader": grader_identity("ifeval"),
            "status": "graded" if good else "not_applicable", "correct": all(vector) if good else None,
            "metrics": ifeval_metrics(vector, vector) if good else {},
            "diagnostics": {"code": "synthetic_grade", "extracted_answer": None,
                "output_sha256": canonical_hash("Synthetic answer") if good else None, "reasoning_sha256": None}}
        records.append(record)
        grades.append(grade)
    return {"records": records, "grades": grades, "summary": {"technically_valid": True}}


class HeldoutAnalysisTests(unittest.TestCase):
    def test_count_ratio_not_mean_prompt_percentages(self):
        result = instruction_count_bootstrap([1, 3], [1, 0], [0, 3])
        self.assertEqual(result["baseline"]["value"], .25)
        self.assertEqual(result["candidate"]["value"], .75)
        self.assertEqual(result["difference_percentage_points"], 50)
        # Exhaustive possible resamples: AA=-1, AB=+0.5, BB=+1.
        self.assertEqual(result["confidence_interval"]["low"], -1)
        self.assertEqual(result["confidence_interval"]["high"], 1)

    def test_bootstrap_matches_independent_prompt_resampling(self):
        counts = np.array([1, 2, 4, 1, 3])
        baseline = np.array([1, 1, 0, 0, 2])
        candidate = np.array([0, 2, 3, 0, 1])
        sample = np.random.default_rng(0).integers(0, 5, size=(10000, 5))
        expected = (candidate[sample].sum(1) - baseline[sample].sum(1)) / counts[sample].sum(1)
        result = instruction_count_bootstrap(counts, baseline, candidate)
        self.assertEqual([result["confidence_interval"][k] for k in ("low", "high")],
                         list(np.quantile(expected, [.025, .975])))

    def test_complete_comparison_preserves_official_counts_and_truncation(self):
        items, (strict, loose) = fixtures()
        baseline = evidence(items, strict, "baseline", [[True], [False] * 3])
        candidate = evidence(items, strict, "candidate", [[False], [True] * 3], statuses=["ok", "truncated"])
        result = compare_evidence(strict, loose, baseline, candidate)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["instruction_comparisons"]["strict"]["delta"], .5)
        self.assertEqual(result["prompt_comparisons"]["strict"]["comparison"]["delta"], 0)
        self.assertEqual(result["paired_outcomes"]["strict"], {"improved": 1, "regressed": 1, "tied": 0, "unresolved": 0})
        self.assertEqual(result["arm_accounting"]["candidate"]["truncations"], 1)
        self.assertFalse(result["exact_paired_significance_test"]["available"])

    def test_absent_arm_and_synthetic_missing_are_unexecuted_not_model_failures(self):
        items, (strict, loose) = fixtures()
        baseline = evidence(items, strict, "baseline", [[False], [False] * 3], statuses=["error", "missing"])
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        result = compare_evidence(strict, loose, baseline, absent)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["arm_accounting"]["baseline"]["attempted_model_failures"], 1)
        self.assertEqual(result["arm_accounting"]["baseline"]["unexecuted_slots"], 1)
        self.assertEqual(result["arm_accounting"]["candidate"]["attempted_model_failures"], 0)
        self.assertEqual(result["arm_accounting"]["candidate"]["unexecuted_slots"], 2)
        self.assertEqual(result["prompt_comparisons"]["strict"]["baseline_projection"]["accounting"]["model_failures"], 1)
        self.assertEqual(result["prompt_comparisons"]["strict"]["baseline_projection"]["accounting"]["absent_raw_records"], 1)
        self.assertEqual(result["instruction_comparisons"]["strict"]["candidate"]["denominator"], 4)
        self.assertEqual(result["paired_outcomes"]["strict"]["unresolved"], 2)

    def test_dispatched_but_unobserved_is_distinct(self):
        items, (strict, loose) = fixtures()
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        result = compare_evidence(strict, loose, absent, absent,
            dispatched_items={"baseline": ["0"], "candidate": []})
        accounting = result["arm_accounting"]["baseline"]
        self.assertEqual(accounting["dispatched_without_observed_output"], 1)
        self.assertEqual(accounting["unexecuted_slots"], 1)
        self.assertEqual(accounting["attempted_model_failures"], 0)

    def test_unexecuted_placeholder_identity_is_validated_before_projection(self):
        items, (strict, loose) = fixtures()
        baseline = evidence(items, strict, "baseline", [[False], [False] * 3], statuses=["missing", "missing"])
        baseline["records"][0]["run_id"] = "wrong-fixture-run"
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        with self.assertRaisesRegex(ValueError, "run_id"):
            compare_evidence(strict, loose, baseline, absent)

    def test_ambiguous_dispatch_without_response_is_not_demonstrated_failure(self):
        items, (strict, loose) = fixtures()
        baseline = evidence(items, strict, "baseline", [[False], [False] * 3], statuses=["error", "missing"])
        baseline["item_metrics"] = [{"item_id": "0", "browser_observation": {
            "dispatch_outcome": "acknowledged_or_ambiguous"}}]
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        result = compare_evidence(strict, loose, baseline, absent)
        accounting = result["arm_accounting"]["baseline"]
        self.assertEqual(accounting["ambiguous_dispatch_outcomes"], 1)
        self.assertEqual(accounting["dispatched_without_observed_output"], 1)
        self.assertEqual(accounting["attempted_model_failures"], 0)
        self.assertEqual(accounting["unexecuted_slots"], 1)
        self.assertEqual(result["prompt_comparisons"]["strict"]["baseline_projection"]["accounting"]["model_failures"], 0)

    def test_grader_error_is_bound_not_semantic_model_failure(self):
        items, (strict, loose) = fixtures()
        baseline = evidence(items, strict, "baseline", [[True], [False] * 3])
        candidate = evidence(items, strict, "candidate", [[True], [True] * 3])
        candidate["grades"][1].update(status="grader_error", correct=None, metrics={})
        result = compare_evidence(strict, loose, baseline, candidate)
        self.assertFalse(result["semantic_grading_complete"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["arm_accounting"]["candidate"]["attempted_model_failures"], 0)
        self.assertEqual(result["paired_outcomes"]["strict"]["unresolved"], 1)
        self.assertIn("not demonstrated model failures", result["grader_error_interpretation"])

    def test_completed_output_without_grade_fails_closed(self):
        items, (strict, loose) = fixtures()
        baseline = evidence(items, strict, "baseline", [[True], [False] * 3])
        candidate = evidence(items, strict, "candidate", [[True], [True] * 3])
        candidate["grades"].pop()
        with self.assertRaisesRegex(ValueError, "lacks an explicit"):
            compare_evidence(strict, loose, baseline, candidate)

    def test_manifest_pair_must_bind_same_policy(self):
        items, (strict, loose) = fixtures()
        loose["policy"]["grader_timeout_seconds"] = 19
        loose["policy_sha256"] = canonical_hash(loose["policy"])
        loose["manifest_sha256"] = canonical_hash({k: v for k, v in loose.items() if k != "manifest_sha256"})
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        with self.assertRaisesRegex(ValueError, "identical evidence"):
            compare_evidence(strict, loose, absent, absent)


if __name__ == "__main__":
    unittest.main()
