import copy
import hashlib
import unittest

from weight_space_lab.evaluation import validate_records
from weight_space_lab.records import (
    empty_status_record, run_record_sha256, to_analysis_records,
    validate_run_record, validate_run_records, validate_status_record,
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture_record(status="ok"):
    """Synthetic evidence only; hashes identify test descriptions, not a model."""
    success = status in {"ok", "truncated"}
    outcome = {
        "status": status,
        "raw_output": "fixture output" if success else None,
        "finish_reason": {"ok": "eos", "truncated": "length", "error": "error",
                          "timeout": "timeout", "missing": "not_started"}[status],
        "token_counts": {"input": 2, "output": 3, "total": 5} if success else dict.fromkeys(("input", "output", "total")),
        "duration_seconds": 0.25 if success else None,
        "error": {"code": "fixture_failure", "retryable": True} if status in {"error", "timeout"} else None,
    }
    return {
        "schema_version": 1, "record_type": "run", "run_id": "fixture-run",
        "evidence_kind": "fixture", "profile": "native_development_smoke", "stage": "smoke",
        "benchmark": "synthetic", "dataset_revision": digest("fixture dataset"),
        "item_id": "fixture-a", "split": "development", "sample_index": 0, "seed": 17,
        "checkpoint": {"model_id": "synthetic-array", "revision": digest("fixture revision"),
                       "sha256": digest("fixture checkpoint")},
        "configuration_sha256": digest("fixture configuration"),
        "retry_policy": {"mode": "none", "max_attempts": 1},
        "attempts": [] if status == "missing" else [dict(copy.deepcopy(outcome), attempt_index=0)],
        **copy.deepcopy(outcome),
    }


def grade(record, correct=True):
    return {"item_id": record["item_id"], "seed": record["seed"], "correct": correct,
            "grader_revision": digest("fixture independent grader"),
            "run_record_sha256": run_record_sha256(record)}


class RunRecordTests(unittest.TestCase):
    def test_round_trip_and_compact_schema_remain_distinct(self):
        record = fixture_record()
        checked = validate_run_record(record)
        self.assertEqual(record, checked)
        checked["attempts"][0]["raw_output"] = "changed"
        self.assertEqual(record["attempts"][0]["raw_output"], "fixture output")
        with self.assertRaises(ValueError):
            validate_records([record])
        compact = to_analysis_records([record], [grade(record)], benchmark="synthetic", run_id="fixture-run")
        self.assertEqual(compact, [{"item_id": "fixture-a", "seed": 17, "status": "ok", "correct": True}])

    def test_no_success_with_missing_identity_or_output(self):
        for key in ("dataset_revision", "raw_output", "duration_seconds"):
            record = fixture_record()
            record[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_run_record(record)
        for key in ("revision", "sha256", "model_id"):
            record = fixture_record()
            record["checkpoint"][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_run_record(record)

    def test_missing_results_preserve_unknowns_without_zero_measurements(self):
        record = fixture_record("missing")
        record["dataset_revision"] = None
        record["checkpoint"] = dict.fromkeys(("model_id", "revision", "sha256"))
        self.assertEqual(validate_run_record(record)["attempts"], [])
        record["token_counts"]["output"] = 0
        with self.assertRaises(ValueError):
            validate_run_record(record)

    def test_unknown_token_measurements_are_explicit(self):
        record = fixture_record()
        record["token_counts"] = dict.fromkeys(("input", "output", "total"))
        record["attempts"][0]["token_counts"] = copy.deepcopy(record["token_counts"])
        self.assertIsNone(validate_run_record(record)["token_counts"]["output"])

    def test_strict_types_versions_nonfinite_and_unknown_fields(self):
        changes = [("schema_version", True), ("schema_version", 2), ("sample_index", True),
                   ("seed", -1), ("duration_seconds", float("nan")),
                   ("duration_seconds", float("inf")), ("duration_seconds", True),
                   ("duration_seconds", 10 ** 1000),
                   ("raw_output", {}), ("status", []), ("profile", []),
                   ("configuration_sha256", "main"), ("dataset_revision", "todo"),
                   ("surprise", 1)]
        for key, value in changes:
            record = fixture_record()
            record[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_run_record(record)
        for value in (True, -1, 1.5, float("nan")):
            record = fixture_record()
            record["token_counts"]["output"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_run_record(record)
        record = fixture_record()
        record["token_counts"]["total"] = 99
        with self.assertRaises(ValueError):
            validate_run_record(record)
        record["token_counts"] = {"input": None, "output": 10, "total": 1}
        with self.assertRaises(ValueError):
            validate_run_record(record)

    def test_error_codes_exclude_raw_provider_text(self):
        record = fixture_record("error")
        record["error"]["code"] = "Failed to open private file: secret"
        with self.assertRaises(ValueError):
            validate_run_record(record)

    def test_retry_history_preserves_failure_and_final_attempt(self):
        record = fixture_record()
        failure = fixture_record("timeout")["attempts"][0]
        record["attempts"][0]["attempt_index"] = 1
        record["attempts"].insert(0, failure)
        record["retry_policy"] = {"mode": "retry_errors", "max_attempts": 2}
        checked = validate_run_record(record)
        self.assertEqual([a["status"] for a in checked["attempts"]], ["timeout", "ok"])
        for change in ("budget", "index", "not_retryable", "success_before_success", "mismatch"):
            modified = copy.deepcopy(record)
            if change == "budget":
                modified["retry_policy"]["max_attempts"] = 1
            elif change == "index":
                modified["attempts"][1]["attempt_index"] = 3
            elif change == "not_retryable":
                modified["attempts"][0]["error"]["retryable"] = False
            elif change == "success_before_success":
                modified["attempts"][0] = fixture_record()["attempts"][0]
            else:
                modified["raw_output"] = "selected favorable result"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_run_record(modified)

    def test_attempt_presence_and_finish_reason_consistency(self):
        for status in ("ok", "truncated", "error", "timeout", "missing"):
            record = fixture_record(status)
            validate_run_record(record)
            record["finish_reason"] = "unrecognized"
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate_run_record(record)
        record = fixture_record()
        record["attempts"] = []
        with self.assertRaises(ValueError):
            validate_run_record(record)

    def test_duplicates_by_sample_or_seed_are_rejected(self):
        record = fixture_record()
        for key in ("sample_index", "seed"):
            other = copy.deepcopy(record)
            other[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_run_records([record, other])
        other = copy.deepcopy(record)
        other["benchmark"] = "other-synthetic"
        self.assertEqual(len(validate_run_records([record, other])), 2)

    def test_stage_and_split_separation(self):
        record = fixture_record()
        record["split"] = "test"
        with self.assertRaises(ValueError):
            validate_run_record(record)
        record.update(profile="fresh_editor_search", stage="pilot")
        with self.assertRaises(ValueError):
            validate_run_record(record)
        record.update(profile="frozen_final_evaluation", stage="full")
        validate_run_record(record)
        for profile in ("fixed_patch_evaluation", "fresh_editor_search", "browser_evaluation"):
            record = fixture_record()
            record["profile"] = profile
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_run_record(record)

    def test_independent_grade_required_and_bound_to_exact_evidence(self):
        record = fixture_record()
        for grades in ([], [dict(grade(record), run_record_sha256=digest("different evidence"))],
                       [dict(grade(record), correct=1)], [grade(record), grade(record)]):
            with self.subTest(grades=grades), self.assertRaises(ValueError):
                to_analysis_records([record], grades, benchmark="synthetic", run_id="fixture-run")
        compact = to_analysis_records([fixture_record("timeout")], [], benchmark="synthetic", run_id="fixture-run")
        self.assertFalse(compact[0]["correct"])
        with self.assertRaises(ValueError):
            to_analysis_records([fixture_record("timeout")], [grade(record)], benchmark="synthetic", run_id="fixture-run")

    def test_conversion_rejects_mixed_contexts(self):
        first, second = fixture_record(), fixture_record()
        second["item_id"] = "fixture-b"
        second["configuration_sha256"] = digest("different configuration")
        with self.assertRaises(ValueError):
            to_analysis_records([first, second], [grade(first), grade(second)], benchmark="synthetic", run_id="fixture-run")

    def test_conversion_rejects_mixed_grader_revisions(self):
        first, second = fixture_record(), fixture_record()
        second["item_id"] = "fixture-b"
        other_grade = grade(second)
        other_grade["grader_revision"] = digest("another grader")
        with self.assertRaises(ValueError):
            to_analysis_records([first, second], [grade(first), other_grade], benchmark="synthetic", run_id="fixture-run")

    def test_unknown_failure_identities_project_without_dropping_failures(self):
        first = fixture_record()
        for status in ("missing", "error", "timeout"):
            second = fixture_record(status)
            second["item_id"] = "fixture-b"
            second["dataset_revision"] = None
            second["checkpoint"] = dict.fromkeys(("model_id", "revision", "sha256"))
            original = copy.deepcopy(second)
            with self.subTest(status=status):
                compact = to_analysis_records([first, second], [grade(first)], benchmark="synthetic", run_id="fixture-run")
                self.assertEqual(len(compact), 2)
                self.assertEqual(compact[1], {"item_id": "fixture-b", "seed": 17, "status": status, "correct": False})
                self.assertEqual(second, original)

    def test_conflicting_known_failure_identities_are_not_unified(self):
        first, second = fixture_record(), fixture_record("missing")
        second["item_id"] = "fixture-b"
        for key in ("model_id", "revision", "sha256"):
            changed = copy.deepcopy(second)
            changed["checkpoint"][key] = "different-fixture-model" if key == "model_id" else digest("different identity")
            with self.subTest(key=key), self.assertRaises(ValueError):
                to_analysis_records([first, changed], [grade(first)], benchmark="synthetic", run_id="fixture-run")
        second["dataset_revision"] = digest("different fixture dataset")
        with self.assertRaises(ValueError):
            to_analysis_records([first, second], [grade(first)], benchmark="synthetic", run_id="fixture-run")

    def test_projection_rejects_conflicting_observed_sample_seed_mapping(self):
        first, second = fixture_record(), fixture_record("missing")
        second["item_id"] = "fixture-b"
        for key in ("sample_index", "seed"):
            changed = copy.deepcopy(second)
            changed[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                to_analysis_records([first, changed], [grade(first)], benchmark="synthetic", run_id="fixture-run")


class StatusRecordTests(unittest.TestCase):
    def test_fixture_success_never_promotes_real_status(self):
        status = empty_status_record()
        status["fixture_validation"].update(status="passed", evidence=[{"kind": "fixture_report", "sha256": digest("fixture report")}])
        checked = validate_status_record(status)
        self.assertEqual(checked["real_model_validation"]["status"], "not_started")
        self.assertEqual(checked["experiment"]["status"], "not_started")
        checked["fixture_validation"]["evidence"].clear()
        self.assertTrue(status["fixture_validation"]["evidence"])

    def test_pass_requires_evidence_of_correct_kind(self):
        status = empty_status_record()
        status["real_model_validation"]["status"] = "passed"
        with self.assertRaises(ValueError):
            validate_status_record(status)
        status["real_model_validation"]["evidence"] = [{"kind": "fixture_report", "sha256": digest("fixture report")}]
        with self.assertRaises(ValueError):
            validate_status_record(status)

    def test_negative_result_is_completed_experiment(self):
        status = empty_status_record()
        status["experiment"].update(status="completed", outcome="negative", evidence=[{"kind": "experiment_report", "sha256": digest("negative report")}])
        self.assertEqual(validate_status_record(status)["experiment"]["outcome"], "negative")
        status["experiment"]["outcome"] = None
        with self.assertRaises(ValueError):
            validate_status_record(status)

    def test_blocked_requires_reason_and_does_not_invent_evidence(self):
        status = empty_status_record()
        status["real_model_validation"]["status"] = "blocked"
        with self.assertRaises(ValueError):
            validate_status_record(status)
        status["real_model_validation"]["reason"] = "Checkpoint has not been obtained."
        self.assertFalse(validate_status_record(status)["real_model_validation"]["evidence"])


if __name__ == "__main__":
    unittest.main()
