"""End-to-end report plumbing with hand-authored journals and mocked evaluators.

The native-shaped records in this suite are schema fixtures in temporary
directories. They never load a checkpoint or establish measured model results.
"""
import copy
import json
import os
import unittest
from unittest.mock import patch

import test_published_study as fixtures
from weight_space_lab import published_study as study
from weight_space_lab.grading import grade_record as actual_grade_record
from weight_space_lab.inventory import file_sha256
from weight_space_lab.native_runner import _inputs, _Journal, _lock


class ReplayReportingTests(unittest.TestCase):
    # Share the tiny metadata fixture without inheriting its independent tests.
    setUp = fixtures.ReplayFreezeTests.setUp
    freeze = fixtures.ReplayFreezeTests.freeze

    def journal(self, name, *, unfinished=False, token_ids=True, fixture_backend=False):
        frozen = study.load_study(self.root, "smoke")
        entry = frozen["jobs"][name]
        inputs = frozen["benchmarks"][entry["benchmark"]]
        config = copy.deepcopy(entry["config"])
        if fixture_backend:
            config["evidence_kind"] = "fixture"
            config["backend"] = {"name": "fixture", "behavior": "success", "delay_seconds": 0}
        manifest, records, _ = _inputs(config, inputs["tasks"], inputs["schedule"])
        output = self.root / "smoke/runs" / name
        with _lock(output):
            journal = _Journal(output, manifest, records, resume=False)
            for index in range(len(records)):
                journal.append("attempt_started", index, 0)
                if unfinished:
                    break
                # An unclosed thinking block exercises actual truncation and
                # extraction policy without invoking optional evaluators.
                outcome = {"status": "truncated", "raw_output": "SYNTHETIC unfinished reasoning",
                           "finish_reason": "length", "token_counts": {"input": 3, "output": 16, "total": 19},
                           "duration_seconds": 0.25, "error": None}
                metadata = {"generation": {"output_token_ids": [9] * 16}} if token_ids else {}
                journal.append("attempt_finished", index, 0, outcome=outcome,
                               metadata=metadata, elapsed_seconds=0.5)
        return output

    def populate(self, *, missing_token_job=None):
        frozen = study.load_study(self.root, "smoke")
        return {name: self.journal(name, token_ids=name != missing_token_job) for name in frozen["jobs"]}

    def report(self, report_id, **options):
        with patch("weight_space_lab.grading.run_known_answer_checks",
                   return_value={"all_passed": True, "kind": "mocked_fixture_checks", "checks": []}):
            return study.report_study(self.root, "smoke", report_id=report_id, **options)

    def test_unattempted_report_is_pending_and_cannot_unlock_pilot_or_full(self):
        self.freeze()
        with patch("weight_space_lab.grading.grade_record", side_effect=AssertionError("no output to grade")) as grader:
            result = self.report("pending")
            grader.assert_not_called()
        self.assertEqual(result["measurement_status"], "pending")
        self.assertFalse(result["real_model_generation_attempted"])
        self.assertFalse(result["development_validation"]["passed"])
        self.assertEqual(len(result["comparisons"]), 12)
        for comparison in result["comparisons"]:
            self.assertIsNone(comparison["measured_delta"])
            for arm in comparison["arms"].values():
                self.assertEqual(arm["scheduled_generations"], 1)
                self.assertEqual(arm["generation_status_counts"]["missing"], 1)
                self.assertIsNone(arm["measured_average_per_generation"])
        certificate = self.root / "smoke/reports/pending/report.json"
        for stage in ("pilot", "full"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "validation"):
                self.freeze(stage=stage, development_report=certificate)
            self.assertFalse((self.root / stage / "inputs").exists())

    def test_complete_schema_journals_recover_absent_raw_exports_without_generation(self):
        self.freeze()
        outputs = self.populate()
        selected = outputs["gsm8k-original"]
        digest = file_sha256(selected / "records.json")
        (selected / "records.json").unlink()
        with patch("weight_space_lab.native_runner.run_schedule", side_effect=AssertionError("report repeated generation")) as runner:
            result = self.report("recovered")
            runner.assert_not_called()
        self.assertEqual(file_sha256(selected / "records.json"), digest)
        self.assertTrue(result["development_validation"]["passed"])
        self.assertTrue(all(check == {"checked": True, "token_ids_equal": True}
                            for check in result["identity_token_checks"].values()))
        self.assertEqual(result["reporting_runtime"]["new_grades"], 15)
        self.assertFalse(result["scientific_result"])
        self.assertFalse(result["experiment_executed"])
        self.assertEqual(sum(value["accounted_wall_seconds"] for value in
                             result["distinct_job_execution_accounting"].values()), 7.5)
        for comparison in result["comparisons"]:
            for arm in comparison["arms"].values():
                self.assertEqual(arm["generation_status_counts"]["truncated"], 1)
                self.assertEqual(arm["measured_average_per_generation"], 0)
                self.assertFalse(arm["best_of_n_computed"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.report("recovered")
        self.assertFalse(self.editor.exists())

    def test_report_reuses_cached_grader_errors_instead_of_retrying_or_selecting(self):
        self.freeze()
        self.populate()

        def grader_failure(record, item, policy, **kwargs):
            grade = actual_grade_record(record, item, policy, **kwargs)
            if record["benchmark"] == "math":
                grade.update(status="grader_error", correct=None, metrics={})
                grade["diagnostics"]["code"] = "fixture_evaluator_failure"
            return grade

        with patch("weight_space_lab.grading.grade_record", side_effect=grader_failure) as grader:
            first = self.report("failed-grader")
            self.assertEqual(grader.call_count, 15)
        before = {p.relative_to(self.root): file_sha256(p) for p in (self.root / "smoke/grade-cache").rglob("*.json")}
        with patch("weight_space_lab.grading.grade_record", side_effect=AssertionError("cached grade retried")) as grader:
            second = self.report("cached-grader")
            grader.assert_not_called()
        after = {p.relative_to(self.root): file_sha256(p) for p in (self.root / "smoke/grade-cache").rglob("*.json")}
        self.assertEqual(before, after)
        self.assertEqual(first["evidence_hashes"], second["evidence_hashes"])
        self.assertEqual(second["reporting_runtime"]["cached_grades"], 15)
        self.assertEqual(second["reporting_runtime"]["new_grades"], 0)
        self.assertFalse(second["development_validation"]["passed"])
        primary = next(row for row in second["comparisons"] if row["comparison_id"] == "math-primary")
        self.assertEqual(primary["measurement_status"], "grading_unresolved")
        self.assertEqual(primary["arms"]["baseline"]["grading_accounting"]["grader_error"], 1)

    def test_interrupted_cache_publication_leaves_no_partial_grade_and_can_resume(self):
        self.freeze()
        self.journal("gsm8k-original")
        actual_link = os.link

        def fail_cache_publication(source, destination, *args, **kwargs):
            if "grade-cache" in str(destination):
                raise OSError("synthetic interruption before atomic cache publication")
            return actual_link(source, destination, *args, **kwargs)

        with patch("weight_space_lab.native_runner.os.link", side_effect=fail_cache_publication):
            with self.assertRaisesRegex(OSError, "synthetic interruption"):
                self.report("interrupted")
        cache = self.root / "smoke/grade-cache"
        self.assertEqual(list(cache.rglob("*.json")), [])
        self.assertEqual(list(cache.rglob(".pending-*")), [])
        self.assertFalse((self.root / "smoke/reports/interrupted").exists())
        result = self.report("restarted")
        self.assertEqual(result["reporting_runtime"]["new_grades"], 1)
        self.assertEqual(len(list(cache.rglob("*.json"))), 1)
        with patch("weight_space_lab.grading.grade_record", side_effect=AssertionError("published grade retried")):
            repeated = self.report("reused")
        self.assertEqual(repeated["reporting_runtime"]["cached_grades"], 1)

    def test_unfinished_native_attempt_is_not_silently_reported_as_unattempted(self):
        self.freeze()
        self.journal("gsm8k-identity", unfinished=True)
        with self.assertRaisesRegex(ValueError, "unfinished native attempt"):
            self.report("unfinished")
        self.assertFalse((self.root / "smoke/reports/unfinished").exists())

    def test_corrupt_event_is_rejected_instead_of_trusting_records_export(self):
        self.freeze()
        output = self.journal("gsm8k-identity")
        path = output / "events/00000001.json"
        event = json.loads(path.read_text())
        event["event"]["outcome"]["raw_output"] = "changed after execution"
        fixtures.write_json(path, event)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.report("corrupt")
        self.assertFalse((self.root / "smoke/reports/corrupt").exists())

    def test_identity_requires_token_evidence_for_every_scheduled_generation(self):
        self.freeze()
        self.populate(missing_token_job="math-identity")
        result = self.report("missing-token-evidence")
        self.assertTrue(result["development_validation"]["generation_paths_complete"])
        self.assertFalse(result["development_validation"]["passed"])
        self.assertEqual(result["identity_token_checks"]["math"], {"checked": False, "token_ids_equal": False})
        self.assertTrue(result["identity_token_checks"]["gsm8k"]["checked"])
        certificate = self.root / "smoke/reports/missing-token-evidence/report.json"
        with self.assertRaisesRegex(ValueError, "validation"):
            self.freeze(stage="pilot", development_report=certificate)

    def test_fixture_journal_cannot_be_promoted_into_native_replay_evidence(self):
        self.freeze()
        self.journal("gsm8k-identity", fixture_backend=True)
        with self.assertRaisesRegex(ValueError, "incompatible resume"):
            self.report("fixture-promotion")
        self.assertFalse((self.root / "smoke/reports/fixture-promotion").exists())


if __name__ == "__main__":
    unittest.main()
