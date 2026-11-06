"""Independent hand-authored answer and failure cases; no benchmark scores."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from weight_space_lab.grading import (
    _known_answer_item, _known_answer_record, _worker, extract_gsm8k_answer,
    extract_last_boxed, grade_record, grader_identity, ifeval_metrics,
    separate_output, validate_grade_record, validate_grading_policy,
    verify_ifeval_assets,
)
from weight_space_lab.records import run_record_sha256
from weight_space_lab.protocol import canonical_hash


POLICY = {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
          "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
          "grader_timeout_seconds": 10, "grader_retry": {"mode": "none", "max_attempts": 1}}


class ExtractionTests(unittest.TestCase):
    def test_box_selection_is_last_not_most_favorable(self):
        cases = [(r"\boxed{1} then \boxed{2}", "2", "answer_extracted"),
                 (r"\boxed{\frac{2}{4}}", r"\frac{2}{4}", "answer_extracted"),
                 (r"\boxed{\{1,2\}}", r"\{1,2\}", "answer_extracted"),
                 (r"\boxed{1} then \boxed{", None, "malformed_final_answer"),
                 (r"\boxed{\boxed{2}+1}", None, "malformed_final_answer"),
                 (r"\boxed{unclosed then \boxed{2}", None, "malformed_final_answer"),
                 (r"\boxed{}", None, "empty_final_answer"),
                 ('{"answer":2}', None, "missing_final_answer")]
        for text, answer, code in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_last_boxed(text), (answer, code))

    def test_gsm_numeric_policy(self):
        for text, expected in [("#### 1,200.00", "1200.00"), ("#### -0.50", "-0.50"),
                               (r"\boxed{2.5}", "2.5"), ("#### 2\n#### 3", "3")]:
            self.assertEqual(extract_gsm8k_answer(text)[0], expected)
        for text in ("2", "It is 2", "#### 2 then 3", "#### 2e0", "#### NaN", "#### 12,34", "#### 1/2"):
            with self.subTest(text=text):
                self.assertIsNone(extract_gsm8k_answer(text)[0])

    def test_consistent_reasoning_output_separation(self):
        policy = dict(POLICY, reasoning_policy="think_tags")
        self.assertEqual(separate_output("<think>#### 9</think>\n#### 2", policy),
                         ("#### 9", "\n#### 2", "output_ready"))
        self.assertEqual(separate_output("No tags", policy), ("", "No tags", "output_ready"))
        for text in ("<think>#### 9", "</think>#### 2", "text<think>x</think>2", "<think>x</think><think>2"):
            self.assertEqual(separate_output(text, policy)[2], "malformed_reasoning_boundary")
        policy = dict(POLICY, reasoning_policy="qwen_thinking", output_eos_tokens=["<|im_end|>"])
        self.assertEqual(separate_output("work</think>\n#### 2<|im_end|>", policy),
                         ("work", "\n#### 2", "output_ready"))
        self.assertEqual(separate_output("#### 2", policy)[2], "malformed_reasoning_boundary")
        self.assertEqual(separate_output("#### 2<|im_end|>", POLICY)[1], "#### 2<|im_end|>")


class GradingTests(unittest.TestCase):
    def grade(self, text, *, benchmark="gsm8k", target=None, status="ok", policy=None):
        item = _known_answer_item(benchmark, target or {"answer": "2"})
        raw = _known_answer_record(item, text, status)
        return grade_record(raw, item, policy or POLICY), raw, item

    def test_semantic_grade_is_not_usability_or_json(self):
        for text, expected in [("#### 2", True), ("#### 3", False), ('{"answer":2}', False), ("", False)]:
            result, raw, item = self.grade(text)
            self.assertIs(result["correct"], expected)
            self.assertEqual(result["run_record_sha256"], run_record_sha256(raw))
            self.assertEqual(result["item_sha256"], canonical_hash(item))
            self.assertEqual(result["policy_sha256"], canonical_hash(POLICY))
            self.assertNotIn("raw_output", result)

    def test_truncation_and_reasoning_policy(self):
        grade, _, _ = self.grade("#### 2", status="truncated")
        self.assertTrue(grade["correct"])
        grade, _, _ = self.grade("#### 2", status="truncated", policy=dict(POLICY, truncation_policy="fail"))
        self.assertFalse(grade["correct"])
        self.assertEqual(grade["diagnostics"]["code"], "truncation_policy_failure")
        grade, _, _ = self.grade("<think>#### 2</think>Not finished", policy=dict(POLICY, reasoning_policy="think_tags"))
        self.assertFalse(grade["correct"])

    def test_model_failure_and_grader_failure_remain_different(self):
        item = _known_answer_item("gsm8k", {"answer": "2"})
        raw = _known_answer_record(item, "")
        outcome = {"status": "timeout", "raw_output": None, "finish_reason": "timeout",
                   "token_counts": {"input": None, "output": None, "total": None}, "duration_seconds": 1.0,
                   "error": {"code": "generation_timeout", "retryable": False}}
        raw.update(outcome)
        raw["attempts"] = [dict(outcome, attempt_index=0)]
        result = grade_record(raw, item, POLICY)
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["correct"])
        failure = {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_dependency_missing"}
        with patch("weight_space_lab.grading._worker", return_value=failure):
            result, _, _ = self.grade(r"\boxed{2}", benchmark="math")
        self.assertEqual(result["status"], "grader_error")
        self.assertIsNone(result["correct"])

    def test_hard_grader_timeout_and_invalid_process_response(self):
        with patch("weight_space_lab.grading.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1)):
            self.assertEqual(_worker({}, 1)["status"], "grader_timeout")
        with patch("weight_space_lab.grading.subprocess.run", side_effect=OSError("private process error")):
            result, _, _ = self.grade(r"\boxed{2}", benchmark="math")
            self.assertEqual(result["status"], "grader_error")
            self.assertEqual(result["diagnostics"]["code"], "evaluator_process_start_failed")
            self.assertNotIn("private process error", json.dumps(result))
        for output in ("not-json", "[]", "{}", '{"status":"graded","correct":1,"metrics":{},"code":"ok"}'):
            with patch("weight_space_lab.grading.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "private error")):
                result = _worker({}, 1)
                self.assertEqual(result["status"], "grader_error")
                self.assertNotIn("private error", json.dumps(result))
        with patch("weight_space_lab.grading._worker", return_value={"status": "graded", "correct": True,
                   "metrics": {"invalid": "usable JSON is insufficient"}, "code": "completed"}):
            result, _, _ = self.grade(r"\boxed{2}", benchmark="math")
            self.assertEqual(result["status"], "grader_error")
            self.assertEqual(result["diagnostics"]["code"], "evaluator_invalid_response")

    def test_retry_policy_and_item_binding_fail_closed(self):
        item = _known_answer_item("gsm8k", {"answer": "2"})
        record = _known_answer_record(item, "#### 2")
        with self.assertRaises(ValueError):
            grade_record(record, item, dict(POLICY, generation_retry={"mode": "retry_errors", "max_attempts": 2}))
        for key, changed in (("item_id", "other"), ("dataset_revision", "0" * 40), ("split", "validation")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                grade_record(dict(record, **{key: changed}), item, POLICY)

    def test_policy_requires_explicit_fields_and_bounded_no_grader_retry(self):
        validate_grading_policy(POLICY)
        for field in POLICY:
            value = dict(POLICY)
            del value[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_grading_policy(value)
        for changes in ({"grader_timeout_seconds": True}, {"grader_timeout_seconds": 121},
                        {"grader_retry": {"mode": "retry_errors", "max_attempts": 2}},
                        {"generation_retry": {"mode": "none", "max_attempts": 2}}):
            with self.assertRaises(ValueError):
                validate_grading_policy(dict(POLICY, **changes))

    def test_official_metric_labels_and_strict_types(self):
        metrics = ifeval_metrics([True, False], [True, True])
        self.assertFalse(metrics["prompt_level_strict_accuracy"])
        self.assertTrue(metrics["prompt_level_loose_accuracy"])
        self.assertEqual(metrics["instruction_level_strict_accuracy"], 0.5)
        target = {"instruction_id_list": ["fixture:a", "fixture:b"], "kwargs": [{}, {}]}
        with patch("weight_space_lab.grading._worker", return_value={"status": "graded", "correct": False,
                   "metrics": metrics, "code": "fixture_evaluator"}):
            result, _, _ = self.grade("text", benchmark="ifeval", target=target)
        for key, value in (("instruction_count", 2.0), ("instruction_strict_correct", True),
                           ("prompt_level_strict_accuracy", 0), ("instruction_level_loose_accuracy", True)):
            broken = copy.deepcopy(result)
            broken["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_grade_record(broken)
        with self.assertRaises(ValueError):
            ifeval_metrics([], [])

    def test_asset_pins_fail_closed_offline(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                verify_ifeval_assets(root)

    def test_identity_is_explicit_and_different_per_benchmark(self):
        identities = [grader_identity(name) for name in ("math", "gsm8k", "ifeval")]
        self.assertEqual(len({d["revision"] for d in identities}), 3)


if __name__ == "__main__":
    unittest.main()
