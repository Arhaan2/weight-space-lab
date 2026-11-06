"""Offline report checks use synthetic outputs, never benchmark execution."""
import copy
import unittest

from test_analysis_pipeline import digest, grade_record, hand_grade, item, make_manifest, raw
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.replay_report import (build_replay_report, estimate_workload,
                                          render_replay_markdown, summarize_comparison)


def summarize(manifest=None, baseline=None, candidate=None, baseline_grades=None, candidate_grades=None, **options):
    return summarize_comparison(manifest or make_manifest(), baseline or [], candidate or [],
                                baseline_grades or [], candidate_grades or [],
                                comparison_id=options.pop("comparison_id", "primary-gsm8k"),
                                comparison_role=options.pop("comparison_role", "primary"),
                                bootstrap_samples=20, seed=42, **options)


def measured_smoke():
    return {"kind": "real_model_report", "arms": [
        {"output_tokens": 10, "generation_seconds": 5, "load_seconds": 2},
        {"output_tokens": 10, "generation_seconds": 10, "load_seconds": 3},
    ]}


class ReplayReportTests(unittest.TestCase):
    def test_pending_preserves_denominator_without_claiming_zero_accuracy(self):
        comparison = summarize()
        self.assertEqual(comparison["measurement_status"], "pending")
        self.assertIsNone(comparison["measured_delta"])
        self.assertIsNone(comparison["paired_item_confidence_interval"])
        for arm in comparison["arms"].values():
            self.assertEqual(arm["scheduled_generations"], 4)
            self.assertEqual(arm["generation_status_counts"]["missing"], 4)
            self.assertEqual(arm["operational_zero_credit_rate"], 0)
            self.assertIsNone(arm["measured_average_per_generation"])
        report = build_replay_report([comparison], stage="smoke", freeze_sha256=digest("freeze"))
        self.assertTrue(report["measured_result_pending"])
        self.assertFalse(report["experiment_executed"])
        self.assertIn("pending", render_replay_markdown(report))

    def test_average_generation_correctness_not_best_of_repeats(self):
        manifest = make_manifest()
        records, grades = {}, {}
        for arm in ("baseline", "candidate"):
            records[arm], grades[arm] = [], []
            for index in (0, 1):
                for sample in (0, 1):
                    # Every baseline item has one success out of two: average
                    # correctness = 0.5 while best-of-two would be 1.0.
                    correct = sample == 0 if arm == "baseline" else index == 0
                    record = raw(manifest, arm=arm, item_index=index, sample_index=sample,
                                 output="#### 2" if correct else "#### 9")
                    records[arm].append(record)
                    grades[arm].append(grade_record(record, item() if index == 0 else item("train:000001"), manifest["policy"]))
        result = summarize(manifest, records["baseline"], records["candidate"], grades["baseline"], grades["candidate"])
        self.assertEqual(result["measurement_status"], "complete")
        self.assertEqual(result["arms"]["baseline"]["measured_average_per_generation"], 0.5)
        self.assertEqual(result["arms"]["baseline"]["operational_zero_credit_numerator"], 2)
        self.assertFalse(result["arms"]["baseline"]["best_of_n_computed"])
        self.assertEqual(len(result["item_changes"]["improved"]), 1)
        self.assertEqual(len(result["item_changes"]["regressed"]), 1)
        self.assertEqual(result["paired_item_confidence_interval"]["method"], "paired percentile bootstrap clustered by item")
        self.assertEqual(result["paired_item_confidence_interval"]["seed"], 42)

    def test_all_execution_failures_leave_measured_result_pending(self):
        manifest = make_manifest()
        records = {arm: [raw(manifest, arm=arm, item_index=index, sample_index=sample,
                             status="error" if sample == 0 else "timeout")
                         for index in (0, 1) for sample in (0, 1)] for arm in ("baseline", "candidate")}
        result = summarize(manifest, records["baseline"], records["candidate"])
        self.assertEqual(result["measurement_status"], "execution_failed")
        self.assertIsNone(result["measured_delta"])
        self.assertIsNone(result["paired_item_confidence_interval"])
        for arm in result["arms"].values():
            self.assertEqual(arm["completed_generations"], 0)
            self.assertIsNone(arm["measured_average_per_generation"])
            self.assertIsNone(arm["measured_equal_item_mean"])
            self.assertEqual(arm["operational_zero_credit_rate"], 0)
            self.assertEqual(arm["operational_zero_credit_denominator"], 4)
            self.assertEqual(arm["grading_accounting"]["model_failures"], 4)
            self.assertEqual(arm["resources"]["attempts"], 4)
            self.assertIn("not measured model correctness", arm["score_interpretation"])
        report = build_replay_report([result], stage="smoke", freeze_sha256=digest("freeze"))
        self.assertTrue(report["measured_result_pending"])
        self.assertNotEqual(report["measurement_status"], "complete")
        self.assertIn("lacks completed generations", report["pending_reason"])

    def test_one_arm_without_generated_output_does_not_measure_paired_effect(self):
        manifest = make_manifest()
        baseline, grades = [], []
        candidate = []
        for index in (0, 1):
            for sample in (0, 1):
                record = raw(manifest, item_index=index, sample_index=sample, status="truncated")
                baseline.append(record)
                grades.append(grade_record(record, item() if index == 0 else item("train:000001"), manifest["policy"]))
                candidate.append(raw(manifest, arm="candidate", item_index=index, sample_index=sample, status="error"))
        result = summarize(manifest, baseline, candidate, grades)
        self.assertEqual(result["measurement_status"], "execution_failed")
        self.assertEqual(result["arms"]["baseline"]["measured_average_per_generation"], 1)
        self.assertIsNone(result["arms"]["candidate"]["measured_average_per_generation"])
        self.assertIsNone(result["measured_delta"])
        self.assertIsNone(result["paired_item_confidence_interval"])
        self.assertIn("operational", result["item_changes_basis"])

    def test_resources_count_retries_once_and_retain_unknown_measurements(self):
        manifest = make_manifest()
        manifest["policy"]["generation_retry"] = {"mode": "retry_errors", "max_attempts": 2}
        manifest["policy_sha256"] = canonical_hash(manifest["policy"])
        manifest["manifest_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        record = raw(manifest)
        failure = raw(manifest, status="timeout")["attempts"][0]
        record["attempts"][0]["attempt_index"] = 1
        record["attempts"].insert(0, failure)
        grade = grade_record(record, item(), manifest["policy"])
        result = summarize(manifest, [record], baseline_grades=[grade],
                           baseline_execution={"accounted_wall_seconds": 4, "attempted_sequences": 2, "scheduled_sequences": 4})
        resource = result["arms"]["baseline"]["resources"]
        self.assertEqual(resource["attempts"], 2)
        self.assertEqual(resource["retry_attempts"], 1)
        self.assertEqual(resource["token_use_all_attempts"]["output"], {"known_total": 2, "unknown_attempts": 1})
        self.assertEqual(resource["attempt_duration_seconds"], {"known_total": 0.1, "unknown_attempts": 1})
        self.assertEqual(resource["accounted_wall_seconds"], 4)
        self.assertEqual(resource["unattempted_scheduled_generations"], 3)

    def test_model_grader_truncation_and_missing_counts_remain_separate(self):
        manifest = make_manifest()
        records = [raw(manifest, status="truncated"), raw(manifest, sample_index=1, status="error"),
                   raw(manifest, item_index=1), raw(manifest, item_index=1, sample_index=1, status="timeout")]
        grades = [grade_record(records[0], item(), manifest["policy"]),
                  grade_record(records[2], item("train:000001"), manifest["policy"])]
        grades[1].update(status="grader_timeout", correct=None, metrics={})
        grades[1]["diagnostics"]["code"] = "evaluator_timeout"
        result = summarize(manifest, records, baseline_grades=grades)
        arm = result["arms"]["baseline"]
        self.assertEqual(arm["generation_status_counts"], {"ok": 1, "truncated": 1, "error": 1, "timeout": 1, "missing": 0})
        self.assertEqual(arm["grading_accounting"]["grader_timeout"], 1)
        self.assertEqual(arm["grading_accounting"]["model_failures"], 2)
        self.assertEqual(result["measurement_status"], "grading_unresolved")
        self.assertIn("unresolved grader", result["uncertainty_interpretation"])

    def test_ifeval_official_weighting_is_separate(self):
        items = [item("1", benchmark="ifeval", instructions=1), item("2", benchmark="ifeval", instructions=3)]
        manifest = make_manifest(items, metric="instruction_level_strict_accuracy")
        records, grades = [], []
        for index, source in enumerate(items):
            for sample in (0, 1):
                record = raw(manifest, item_index=index, sample_index=sample, output="hello")
                records.append(record)
                grades.append(hand_grade(record, source, manifest, strict=[True] if index == 0 else [False] * 3))
        result = summarize(manifest, records, baseline_grades=grades)
        arm = result["arms"]["baseline"]
        self.assertEqual(arm["measured_average_per_generation"], 0.5)
        self.assertEqual(arm["official_metrics"]["instruction_level_strict_accuracy"]["value"], 0.25)
        self.assertIn("prompt instruction fraction", arm["estimand"])

    def test_pending_ifeval_masks_measured_values_and_retains_operational_counts(self):
        items = [item("1", benchmark="ifeval", instructions=1), item("2", benchmark="ifeval", instructions=3)]
        manifest = make_manifest(items, metric="prompt_level_strict_accuracy")
        failed = [raw(manifest, item_index=index, sample_index=sample, status="error")
                  for index in (0, 1) for sample in (0, 1)]
        result = summarize(manifest, failed)
        for arm in result["arms"].values():
            for metric, values in arm["official_metrics"].items():
                self.assertIsNone(values["value"])
                self.assertTrue(values["measurement_pending"])
                self.assertEqual(values["operational_zero_credit_value"], 0)
                self.assertEqual(values["numerator"], 0)
                self.assertEqual(values["denominator"], 4 if metric.startswith("prompt_") else 8)
        projection = result["paired_analysis"]["baseline_projection"]
        self.assertEqual(projection["official_metrics"]["instruction_level_strict_accuracy"]["value"], 0)

    def test_missing_grade_and_out_of_schedule_evidence_fail_closed(self):
        manifest = make_manifest()
        with self.assertRaisesRegex(ValueError, "explicit independent grader"):
            summarize(manifest, [raw(manifest)])
        record = raw(manifest)
        record["seed"] = 99
        with self.assertRaises(ValueError):
            summarize(manifest, [record])

    def test_report_keeps_controls_and_discrepancies_without_changing_label(self):
        primary = summarize()
        alternative = summarize(comparison_id="greedy-gsm8k", comparison_role="alternative_decoding")
        report = build_replay_report([primary, alternative], stage="smoke", freeze_sha256=digest("freeze"),
                                    discrepancies=["Published runtime version was not pinned."],
                                    required_comparison_ids=["primary-gsm8k", "greedy-gsm8k"])
        self.assertEqual(report["study_label"], "published-patch replay")
        self.assertEqual(report["comparisons"][1]["comparison_role"], "alternative_decoding")
        self.assertIn("Published runtime version was not pinned.", render_replay_markdown(report))
        with self.assertRaisesRegex(ValueError, "every frozen comparison"):
            build_replay_report([primary], stage="smoke", freeze_sha256=digest("freeze"),
                                required_comparison_ids=["primary-gsm8k", "greedy-gsm8k"])
        duplicate_primary = copy.deepcopy(primary)
        duplicate_primary["comparison_id"] = "new-primary"
        with self.assertRaisesRegex(ValueError, "one primary"):
            build_replay_report([primary, duplicate_primary], stage="smoke", freeze_sha256=digest("freeze"))

    def test_workload_uses_actual_schedule_and_measured_rates(self):
        workload = [{"benchmark": "math", "arm": arm, "scheduled_generations": 2000, "max_new_tokens": 32768}
                    for arm in ("original", "reconstructed")]
        estimate = estimate_workload(measured_smoke(), workload, target_backend="transformers", hardware_runtime_matched=True)
        self.assertEqual(estimate["scheduled_generations"], 4000)
        self.assertEqual(estimate["maximum_scheduled_output_tokens"], 4000 * 32768)
        self.assertEqual(estimate["measured_rate_range_tokens_per_second"], [1, 2])
        self.assertEqual(estimate["target_token_cap_projection"]["generation_seconds_range"], [2000 * 32768, 4000 * 32768])
        self.assertFalse(estimate["long_context_memory_fit_verified"])
        gpu = estimate_workload(measured_smoke(), workload, target_backend="vllm")
        self.assertFalse(gpu["target_estimate_available"])
        self.assertIsNone(gpu["target_token_cap_projection"])
        self.assertIn("not its ETA", gpu["target_estimate_pending_reason"])

    def test_same_backend_requires_verified_hardware_and_runtime_for_target_eta(self):
        workload = [{"benchmark": "math", "arm": "original", "scheduled_generations": 1, "max_new_tokens": 16}]
        historical = estimate_workload(measured_smoke(), workload, target_backend="transformers")
        self.assertFalse(historical["target_estimate_available"])
        self.assertFalse(historical["hardware_runtime_match_verified"])
        self.assertIsNone(historical["target_token_cap_projection"])
        self.assertIsNotNone(historical["measured_backend_token_cap_reference"])
        report = build_replay_report([], stage="full", freeze_sha256=digest("freeze"), workload_estimate=historical)
        self.assertIn("Historical measured-backend reference only", render_replay_markdown(report))
        verified = estimate_workload(measured_smoke(), workload, target_backend="transformers", hardware_runtime_matched=True)
        report["workload_estimate"] = verified
        self.assertIn("Target planning estimate", render_replay_markdown(report))
        other_backend = estimate_workload(measured_smoke(), workload, target_backend="vllm", hardware_runtime_matched=True)
        self.assertFalse(other_backend["target_estimate_available"])
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            estimate_workload(measured_smoke(), workload, target_backend="transformers", hardware_runtime_matched=1)

    def test_estimate_rejects_unmeasured_or_invalid_inputs(self):
        workload = [{"benchmark": "math", "arm": "original", "scheduled_generations": 1, "max_new_tokens": 16}]
        for smoke in ({"kind": "fixture", "arms": []}, {"kind": "real_model_report", "arms": []},
                      {"kind": "real_model_report", "arms": [{"output_tokens": 1, "generation_seconds": 0}]}):
            with self.subTest(smoke=smoke), self.assertRaises(ValueError):
                estimate_workload(smoke, workload, target_backend="vllm")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            estimate_workload(measured_smoke(), workload * 2, target_backend="transformers")


if __name__ == "__main__":
    unittest.main()
