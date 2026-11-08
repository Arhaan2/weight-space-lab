"""Browser correctness records retain the schedule, failures and native grader."""
import copy
import unittest
from unittest.mock import patch

from weight_space_lab.browser_tasks import (adapt_browser_output, browser_grading_policy,
                                           grade_browser_arm, run_browser_grader_fixtures)
from weight_space_lab.grading import _known_answer_item
from weight_space_lab.protocol import canonical_hash


class BrowserTaskTests(unittest.TestCase):
    def setUp(self):
        self.item = _known_answer_item()
        self.kwargs = {"run_id": "browser-unit-fixture", "evidence_kind": "fixture",
            "checkpoint": {"model_id": "no-model-fixture", "revision": "a" * 40, "sha256": "b" * 64},
            "configuration_sha256": canonical_hash({"fixture": True})}

    def output(self, **changes):
        return {"item_id": self.item["item_id"], "raw_output": r"\boxed{1/2}",
                "finish_reason": "stop", "generated_tokens": 8, "input_tokens": 5,
                "response_latency_ms": 125, "loading_time_ms": 500, **changes}

    def test_actual_pinned_grader_known_answers_without_inference(self):
        result = run_browser_grader_fixtures()
        self.assertTrue(result["all_passed"], result)
        self.assertEqual(result["target_inference_sequences"], 0)
        self.assertEqual(len(result["checks"]), 6)

    def test_browser_policy_is_explicit_about_thinking(self):
        self.assertEqual(browser_grading_policy()["reasoning_policy"], "think_tags")
        self.assertEqual(browser_grading_policy(thinking=True)["reasoning_policy"], "qwen_thinking")

    def test_missing_and_cancelled_are_retained_in_denominator(self):
        items = [copy.deepcopy(self.item) for _ in range(3)]
        for n, item in enumerate(items):
            item["item_id"] = f"fixture-{n}"
            item["source"]["row_id"] = item["item_id"]
        outputs = [self.output(item_id=items[0]["item_id"], raw_output=r"\boxed{"),
                   self.output(item_id=items[1]["item_id"], finish_reason="cancelled", cancelled=True,
                               raw_output="partial", generated_tokens=None, execution_error={"code": "user_cancel"})]
        result = grade_browser_arm(items, outputs, **self.kwargs)
        self.assertEqual(result["summary"]["scheduled"], 3)
        self.assertEqual(result["summary"]["attempted"], 2)
        self.assertEqual(result["summary"]["evaluated"], 1)
        self.assertEqual(result["summary"]["interrupted"], 1)
        self.assertEqual(result["summary"]["unrun"], 1)
        self.assertEqual(result["summary"]["all_scheduled_accuracy_lower_bound"], 0)
        self.assertFalse(result["summary"]["complete_for_selection"])
        self.assertEqual(result["records"][1]["raw_output"], "partial")
        self.assertEqual(result["records"][1]["attempts"][0]["error"]["retryable"], False)
        self.assertEqual(result["records"][2]["attempts"], [])
        self.assertIsNone(result["item_metrics"][2]["generated_tokens"])

    def test_timing_loading_resources_and_cleanup_have_separate_scopes(self):
        resources = {"gpu_buffer_bytes": {"value": 50, "scope": "worker allocated WebGPU buffers; not resident VRAM"}}
        observation = self.output(resources=resources, cleanup={"status": "unloaded"})
        record, metric = adapt_browser_output(self.item, observation, **self.kwargs)
        self.assertEqual(record["duration_seconds"], .125)
        self.assertEqual(metric["loading_time_ms"], 500)
        self.assertEqual(metric["resources"], resources)
        self.assertEqual(metric["cleanup"]["status"], "unloaded")
        self.assertTrue(metric["valid_completion"])

    def test_truncation_is_not_valid_completion(self):
        _, metric = adapt_browser_output(self.item, self.output(finish_reason="length"), **self.kwargs)
        self.assertTrue(metric["truncated"])
        self.assertFalse(metric["valid_completion"])

    def test_timeout_preserves_partial_output(self):
        record, metric = adapt_browser_output(self.item, self.output(finish_reason="timeout", raw_output="partial"), **self.kwargs)
        self.assertEqual(record["status"], "timeout")
        self.assertEqual(record["raw_output"], "partial")
        self.assertEqual(metric["execution_status"], "timeout")

    def test_javascript_session_interruption_keeps_full_schedule_and_partial(self):
        items = [copy.deepcopy(self.item) for _ in range(14)]
        for n, item in enumerate(items):
            item["item_id"] = f"fixture-{n}"
            item["source"]["row_id"] = item["item_id"]
        # Exact fields emitted by taskSample when the worker's session expires.
        output = {"item_id": items[0]["item_id"], "raw_output": r"partial \boxed{1/2}",
                  "finish_reason": "interrupted", "generated_tokens": 12, "input_tokens": 5,
                  "response_latency_ms": 900, "loading_time_ms": 10,
                  "execution_error": "Session deadline exceeded", "cancelled": False,
                  "truncated": True, "valid_completion": False,
                  "webgpu_submitted_batches": 3, "local_webgpu_inference_observed": True}
        result = grade_browser_arm(items, [output], **self.kwargs)
        self.assertEqual(result["summary"]["scheduled"], 14)
        self.assertEqual(result["summary"]["interrupted"], 1)
        self.assertEqual(result["summary"]["unrun"], 13)
        self.assertEqual(result["summary"]["truncated"], 1)
        self.assertEqual(result["summary"]["all_scheduled_accuracy_lower_bound"], 0)
        self.assertEqual(result["records"][0]["status"], "error")
        self.assertEqual(result["item_metrics"][0]["raw_output"], output["raw_output"])
        self.assertEqual(result["item_metrics"][0]["grading_status"], "not_applicable")
        self.assertFalse(result["summary"]["complete_for_selection"])

    def test_duplicate_unscheduled_and_ambiguous_observations_rejected(self):
        for outputs in [[self.output(), self.output()], [self.output(item_id="not-scheduled")],
                        [self.output(finish_reason=None)], [self.output(generated_tokens=True)],
                        [self.output(response_latency_ms=float("nan"))]]:
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                grade_browser_arm([self.item], outputs, **self.kwargs)

    def test_grader_errors_keep_null_correctness_and_lower_bound(self):
        with patch("weight_space_lab.grading._worker", return_value={"status": "grader_error", "correct": None,
                                                                    "metrics": {}, "code": "evaluator_process_failed"}):
            result = grade_browser_arm([self.item], [self.output()], **self.kwargs)
        self.assertIsNone(result["item_metrics"][0]["correctness"])
        self.assertEqual(result["summary"]["unresolved_grades"], 1)
        self.assertFalse(result["summary"]["accuracy_is_resolved"])
        self.assertFalse(result["summary"]["complete_for_selection"])

    def test_heldout_is_not_runnable_through_development_adapter(self):
        item = dict(self.item, split="test")
        with self.assertRaises(ValueError):
            grade_browser_arm([item], [], **self.kwargs)


if __name__ == "__main__":
    unittest.main()
