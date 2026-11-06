"""Synthetic known-answer contracts, never model or experiment measurements."""
import copy
import hashlib
import unittest

from weight_space_lab.analysis_pipeline import (build_analysis_manifest, compare_graded_runs,
                                                project_arm, validate_analysis_manifest)
from weight_space_lab.grading import grade_record, grader_identity
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.records import run_record_sha256


def digest(value):
    return canonical_hash(value)


def policy(truncation="grade"):
    return {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
            "truncation_policy": truncation, "generation_retry": {"mode": "none", "max_attempts": 1},
            "grader_timeout_seconds": 10, "grader_retry": {"mode": "none", "max_attempts": 1}}


def item(name="train:000000", *, benchmark="gsm8k", instructions=1):
    target = {"answer": "2"} if benchmark != "ifeval" else {
        "instruction_id_list": ["keywords:existence"] * instructions,
        "kwargs": [{"keywords": ["hello"]}] * instructions}
    return {"benchmark": benchmark, "item_id": name, "dataset_revision": hashlib.sha1(b"synthetic fixture source").hexdigest(),
            "split": "development", "task_text": "Independent synthetic question " + name,
            "target": target, "source": {"dataset_id": "synthetic-known-answers", "revision": hashlib.sha1(b"synthetic fixture source").hexdigest(),
                                        "split": "train", "row_id": name}, "metadata": {}}


def make_manifest(items=None, *, metric="accuracy", truncation="grade"):
    items = items or [item(), item("train:000001")]
    split = {"schema_version": 1, "kind": "benchmark_split", "split": "development",
             "source_manifest_sha256": digest("synthetic source manifest"),
             "items": [{"benchmark": row["benchmark"], "item_id": row["item_id"],
                        "dataset_revision": row["dataset_revision"], "item_sha256": canonical_hash(row)} for row in items]}
    arms = {name: {"run_id": "fixture-" + name, "profile": "native_development_smoke",
                   "configuration_sha256": digest("fixture configuration " + name),
                   "checkpoint": {"model_id": "synthetic-array", "revision": digest("fixture revision"),
                                  "sha256": digest("fixture checkpoint " + name)}} for name in ("baseline", "candidate")}
    return build_analysis_manifest(split, items, benchmark=items[0]["benchmark"], seeds=[7, 11],
                                   policy=policy(truncation), arms=arms, stage="smoke", metric=metric,
                                   evidence_kind="fixture")


def raw(manifest, *, arm="baseline", item_index=0, sample_index=0, status="ok", output="#### 2"):
    success = status in {"ok", "truncated"}
    source = manifest["items"][item_index]
    outcome = {"status": status, "raw_output": output if success else None,
               "finish_reason": {"ok": "eos", "truncated": "length", "error": "error", "timeout": "timeout", "missing": "not_started"}[status],
               "token_counts": {"input": 1, "output": 2, "total": 3} if success else dict.fromkeys(("input", "output", "total")),
               "duration_seconds": 0.1 if success else None,
               "error": {"code": "fixture_failure", "retryable": True} if status in {"error", "timeout"} else None}
    return {"schema_version": 1, "record_type": "run", **copy.deepcopy(manifest["arms"][arm]),
            "evidence_kind": "fixture", "stage": manifest["stage"], "benchmark": manifest["benchmark"],
            "split": manifest["split"], "dataset_revision": source["dataset_revision"],
            "item_id": source["item_id"], "sample_index": sample_index, "seed": [7, 11][sample_index],
            "retry_policy": copy.deepcopy(manifest["policy"]["generation_retry"]),
            "attempts": [] if status == "missing" else [{"attempt_index": 0, **copy.deepcopy(outcome)}], **outcome}


def hand_grade(record, source, manifest, *, strict=None, loose=None):
    """Hand-authored instruction results test aggregation without evaluator calls."""
    strict = strict or [True]
    loose = loose or strict
    return {"schema_version": 1, "record_type": "grade", "benchmark": record["benchmark"],
            "item_id": record["item_id"], "seed": record["seed"], "sample_index": record["sample_index"],
            "evidence_kind": "fixture", "run_record_sha256": run_record_sha256(record),
            "item_sha256": canonical_hash(source), "policy_sha256": manifest["policy_sha256"],
            "grader": grader_identity(record["benchmark"]), "status": "graded", "correct": all(strict),
            "metrics": {"prompt_level_strict_accuracy": all(strict), "prompt_level_loose_accuracy": all(loose),
                        "instruction_level_strict_accuracy": sum(strict) / len(strict),
                        "instruction_level_loose_accuracy": sum(loose) / len(loose),
                        "instruction_count": len(strict), "instruction_strict_correct": sum(strict),
                        "instruction_loose_correct": sum(loose), "instruction_strict": strict, "instruction_loose": loose},
            "diagnostics": {"code": "graded", "extracted_answer": None,
                            "output_sha256": digest(record["raw_output"]), "reasoning_sha256": None}}


class AnalysisPipelineTests(unittest.TestCase):
    def test_mutually_absent_cases_remain_in_complete_denominator(self):
        manifest = make_manifest()
        baseline = raw(manifest)
        grade = grade_record(baseline, item(), manifest["policy"])
        report = compare_graded_runs([baseline], [], [grade], [], manifest, bootstrap_samples=100)
        self.assertEqual(report["comparison"]["scheduled_cases"], 4)
        self.assertEqual(report["comparison"]["baseline"]["mean"], 0.25)
        self.assertEqual(report["comparison"]["candidate"]["mean"], 0)
        self.assertEqual(report["candidate_projection"]["accounting"]["absent_raw_records"], 4)
        self.assertFalse(report["scientific_result"])
        self.assertEqual(report["evidence_kind"], "fixture")
        self.assertNotIn("raw_output", report["baseline_projection"]["records"][0])
        self.assertNotIn("diagnostics", report["baseline_projection"]["records"][0])

    def test_model_and_grader_failures_have_separate_diagnostics(self):
        manifest = make_manifest()
        records = [raw(manifest, status="timeout"), raw(manifest, sample_index=1),
                   raw(manifest, item_index=1, status="error"), raw(manifest, item_index=1, sample_index=1, status="missing")]
        failed = grade_record(records[1], item(), manifest["policy"])
        failed.update(status="grader_error", correct=None, metrics={})
        failed["diagnostics"]["code"] = "fixture_evaluator_failure"
        projection = project_arm(records, [failed], manifest, arm="baseline")
        self.assertEqual(projection["accounting"]["model_failures"], 3)
        self.assertEqual(projection["accounting"]["grader_error"], 1)
        self.assertFalse(projection["semantic_grading_complete"])
        self.assertEqual(len(projection["records"]), 4)
        self.assertTrue(all(not row["correct"] for row in projection["records"]))
        self.assertEqual(projection["generation_status_counts"]["ok"], 1)
        self.assertEqual(projection["generation_status_counts"]["error"], 1)

    def test_missing_grade_is_not_fabricated_as_incorrect(self):
        manifest = make_manifest()
        with self.assertRaisesRegex(ValueError, "lacks an explicit"):
            project_arm([raw(manifest)], [], manifest, arm="baseline")

    def test_off_schedule_extra_duplicate_stale_and_wrong_context_fail_closed(self):
        manifest = make_manifest()
        record = raw(manifest)
        grade = grade_record(record, item(), manifest["policy"])
        for field, value in [("seed", 99), ("sample_index", 1), ("run_id", "other-run"),
                             ("configuration_sha256", digest("different config")), ("stage", "pilot"),
                             ("evidence_kind", "real_model"), ("dataset_revision", digest("different dataset"))]:
            changed = copy.deepcopy(record)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                project_arm([changed], [grade], manifest, arm="baseline")
        for field in ("run_record_sha256", "item_sha256", "policy_sha256"):
            changed = copy.deepcopy(grade)
            changed[field] = digest("stale")
            with self.subTest(field=field), self.assertRaises(ValueError):
                project_arm([record], [changed], manifest, arm="baseline")
        with self.assertRaises(ValueError):
            project_arm([record, record], [grade], manifest, arm="baseline")
        with self.assertRaises(ValueError):
            project_arm([record], [grade, grade], manifest, arm="baseline")
        with self.assertRaises(ValueError):
            project_arm([], [grade], manifest, arm="baseline")

    def test_frozen_retry_and_checkpoint_are_enforced(self):
        manifest = make_manifest()
        record = raw(manifest)
        grade = grade_record(record, item(), manifest["policy"])
        for change in ("retry", "checkpoint"):
            changed = copy.deepcopy(record)
            if change == "retry":
                changed["retry_policy"] = {"mode": "retry_errors", "max_attempts": 2}
            else:
                changed["checkpoint"]["sha256"] = digest("different synthetic checkpoint")
            with self.subTest(change=change), self.assertRaises(ValueError):
                project_arm([changed], [grade], manifest, arm="baseline")

    def test_explicit_grader_timeout_retains_case_and_bounds_interpretation(self):
        manifest = make_manifest()
        record = raw(manifest)
        grade = grade_record(record, item(), manifest["policy"])
        grade.update(status="grader_timeout", correct=None, metrics={})
        grade["diagnostics"]["code"] = "evaluator_timeout"
        result = compare_graded_runs([record], [], [grade], [], manifest, bootstrap_samples=20)
        self.assertEqual(result["comparison"]["scheduled_cases"], 4)
        self.assertEqual(result["baseline_projection"]["accounting"]["grader_timeout"], 1)
        self.assertFalse(result["semantic_grading_complete"])
        self.assertIn("not an identified", result["interpretation"])

    def test_unknown_failure_identity_does_not_discard_scheduled_failure(self):
        manifest = make_manifest()
        record = raw(manifest, status="timeout")
        record["checkpoint"] = dict.fromkeys(("model_id", "revision", "sha256"))
        record["dataset_revision"] = None
        result = project_arm([record], [], manifest, arm="baseline")
        self.assertEqual(result["accounting"]["model_failures"], 1)
        self.assertEqual(len(result["records"]), 4)

    def test_truncation_is_predeclared_and_cannot_be_changed_after_grading(self):
        for truncation, correct in (("grade", True), ("fail", False)):
            manifest = make_manifest(truncation=truncation)
            record = raw(manifest, status="truncated")
            grade = grade_record(record, item(), manifest["policy"])
            projected = project_arm([record], [grade], manifest, arm="baseline")
            self.assertEqual(projected["records"][0]["correct"], correct)
            self.assertEqual(projected["records"][0]["status"], "truncated")
        modified = copy.deepcopy(manifest)
        modified["policy"]["truncation_policy"] = "grade"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_analysis_manifest(modified)

    def test_every_manifest_item_must_be_scheduled(self):
        manifest = make_manifest()
        manifest["schedule"] = manifest["schedule"][:2]
        # Rehashing a malformed declaration does not make it complete.
        from weight_space_lab.protocol import validate_benchmark_schedule
        manifest["schedule_sha256"] = validate_benchmark_schedule(manifest["schedule"])["schedule_sha256"]
        manifest["manifest_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "every frozen item"):
            validate_analysis_manifest(manifest)

    def test_production_manifest_cannot_admit_fixture_evidence(self):
        manifest = make_manifest()
        record = raw(manifest)
        manifest["evidence_kind"] = "real_model"
        manifest["manifest_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "evidence_kind"):
            project_arm([record], [], manifest, arm="baseline")

    def test_ifeval_official_instruction_weighting_is_labeled_separately(self):
        items = [item("1", benchmark="ifeval", instructions=1), item("2", benchmark="ifeval", instructions=3)]
        manifest = make_manifest(items, metric="instruction_level_strict_accuracy")
        records, grades = [], []
        for index, source in enumerate(items):
            for sample in (0, 1):
                record = raw(manifest, item_index=index, sample_index=sample, output="hello")
                records.append(record)
                vector = [True] if index == 0 else [False, False, False]
                grades.append(hand_grade(record, source, manifest, strict=vector))
        result = compare_graded_runs(records, [], grades, [], manifest, bootstrap_samples=100)
        self.assertEqual(result["comparison"]["baseline"]["mean"], 0.5)
        self.assertEqual(result["baseline_projection"]["official_metrics"]["instruction_level_strict_accuracy"]["value"], 0.25)
        self.assertEqual(result["candidate_projection"]["official_metrics"]["instruction_level_strict_accuracy"]["denominator"], 8)
        self.assertIn("not official", result["paired_estimand"])


if __name__ == "__main__":
    unittest.main()
