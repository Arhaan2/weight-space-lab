"""Synthetic handoff binding and accounting cases; no target or editor calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weight_space_lab import ifeval_handoff as handoff
from weight_space_lab.grading import _known_answer_item, _known_answer_record, grade_record, ifeval_metrics
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash


POLICY = {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
          "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
          "grader_timeout_seconds": 10, "grader_retry": {"mode": "none", "max_attempts": 1}}


class IFEvalHandoffTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.items, self.records, self.grades = [], [], []
        for index in range(32):
            # The nonuniform count ensures official instruction aggregation is
            # distinct from the arithmetic mean of prompt-level fractions.
            count = 1 if index == 0 else 3
            target = {"instruction_id_list": ["fixture:instruction"] * count, "kwargs": [{}] * count}
            item = _known_answer_item("ifeval", target)
            item["item_id"] = f"fixture-{index:02}"
            item["task_text"] = f"Approved independent fixture {index}."
            record = _known_answer_record(item, "actual fixture output", "truncated" if index == 31 else "ok")
            record.update(evidence_kind="real_model", profile="browser_evaluation", stage="pilot", seed=42)
            metrics = ifeval_metrics([index == 0] * count, [True] * count)
            with patch("weight_space_lab.grading._worker", return_value={"status": "graded",
                       "correct": metrics["prompt_level_strict_accuracy"], "metrics": metrics, "code": "fixture_evaluator"}):
                grade = grade_record(record, item, POLICY)
            self.items.append(item)
            self.records.append(record)
            self.grades.append(grade)
        self.save("manifest.json", {"heldout": [{"prompt": "SECRET_HELDOUT"}]})
        self.save("baseline-config.json", {"tasks": [{"prompt": "DEV_CONFIG"}], "modelPath": "/PRIVATE_HOST_PATH",
             "artifactSha256": "b" * 64, "generation": {"temperature": 0, "max_tokens": 1024}})
        self.save("grading-policy.json", POLICY)
        self.save("baseline/graded.json", {"records": self.records, "grades": self.grades,
                  "summary": {"technically_valid": True, "resources": {"gpu_buffer_bytes": 100}, "host_path": "/PRIVATE_HOST_PATH"}})
        self.save("data/development-items.json", self.items)
        self.save("weight-information/policy.json", {"allowed_tensors": ["synthetic.weight"]})
        self.save("weight-information/exposures.jsonl", {"fixture": True})
        self.save("weight-information/inspections.json", {
            "base_sha256": self.records[0]["checkpoint"]["sha256"],
            "exposure_log_sha256": file_sha256(self.root / "weight-information/exposures.jsonl"),
            "scope": "sampled, not whole-tensor", "inspections": []})
        self.save("weight-information/host.json", {"model_path": "/unused-fixture-model"})

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_feedback_keeps_real_output_constraints_and_truncation(self):
        feedback = handoff.baseline_feedback(self.items, self.records, self.grades, POLICY)
        self.assertEqual(feedback[0]["raw_response"], "actual fixture output")
        self.assertTrue(feedback[0]["constraints"][0]["strict"])
        self.assertFalse(feedback[1]["constraints"][0]["strict"])
        summary = handoff.aggregate_feedback(feedback)
        self.assertEqual(summary["scheduled_instructions"], 94)
        self.assertEqual(summary["instruction_level_strict_accuracy"], 1 / 94)
        self.assertEqual(summary["prompt_level_strict_accuracy"], 1 / 32)
        self.assertEqual(summary["truncated"], 1)
        self.assertFalse(feedback[-1]["completion_within_budget"])

    def test_export_excludes_heldout_and_paths_and_cannot_renew(self):
        result = handoff.export_packet(self.root)
        packet = (self.root / "handoff/editor-packet.txt").read_text()
        for text in ("SECRET_HELDOUT", "PRIVATE_HOST_PATH", "DEV_CONFIG"):
            self.assertNotIn(text, packet)
        self.assertIn("Approved independent fixture 31.", packet)
        request = read_json(self.root / "handoff/request.json")
        self.assertEqual(request["editor"]["requested_effort"], "ultra")
        self.assertIsNone(request["editor"]["immutable_model_version"])
        self.assertEqual(result["frontier_requests"], 0)
        self.assertEqual(request["submission_expires_at_epoch_seconds"] - request["created_at_epoch_seconds"], 604800)
        with self.assertRaises(FileExistsError):
            handoff.export_packet(self.root)

    def test_changed_outputs_or_grades_rejected(self):
        changed = copy.deepcopy(self.records)
        changed[0]["raw_output"] = "tampered"
        changed[0]["attempts"][0]["raw_output"] = "tampered"
        with self.assertRaisesRegex(ValueError, "binding changed"):
            handoff.baseline_feedback(self.items, changed, self.grades, POLICY)
        with self.assertRaisesRegex(ValueError, "32 frozen"):
            handoff.baseline_feedback(self.items, self.records[:-1], self.grades, POLICY)

    def test_runtime_empty_nonthinking_prefix_is_not_a_reasoning_readiness_gate(self):
        records = copy.deepcopy(self.records)
        policy = dict(POLICY, reasoning_policy="think_tags")
        grades = []
        for item, record in zip(self.items, records):
            record["raw_output"] = "<think>\n\n</think>\n\nactual fixture output"
            record["attempts"][0]["raw_output"] = record["raw_output"]
            metrics = ifeval_metrics([False] * len(item["target"]["kwargs"]), [False] * len(item["target"]["kwargs"]))
            with patch("weight_space_lab.grading._worker", return_value={"status": "graded", "correct": False,
                       "metrics": metrics, "code": "fixture_evaluator"}):
                grades.append(grade_record(record, item, policy))
        feedback = handoff.baseline_feedback(self.items, records, grades, policy)
        self.assertEqual(feedback[0]["graded_response"], "\n\nactual fixture output")

    def test_resource_summary_keeps_scopes_and_observed_maxima(self):
        target = self.root / "execution/baseline-1"
        target.mkdir(parents=True)
        rows = [{"host": {"hostGpuAllocatedBytes": 1000}, "owned_processes": [
            {"pid": 99, "role": "gpu-process", "rssBytes": 25, "footprintBytes": 30}]},
            {"host": {"hostGpuAllocatedBytes": 1200}, "owned_processes": [
            {"pid": 99, "role": "gpu-process", "rssBytes": 28, "footprintBytes": 24}]}]
        (target / "resources.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        result = handoff.resource_summary(self.root, {"item_metrics": [{"item_id": "fixture", "resources": {
            "gpuBufferBytes": 15, "wasmAddressSpaceBytes": 20}}]})
        self.assertEqual(result["host_sampled_numeric_ranges"]["hostGpuAllocatedBytes"]["maximum"], 1200)
        self.assertEqual(result["owned_role_sampled_maxima"]["gpu-process"]["footprintBytes"], 30)
        self.assertNotIn('"pid"', json.dumps(result))

    def test_unresolved_grader_blocks_handoff_without_claiming_low_score(self):
        changed = copy.deepcopy(self.grades)
        changed[1].update(status="grader_error", correct=None, metrics={})
        with self.assertRaisesRegex(ValueError, "grading must resolve"):
            handoff.baseline_feedback(self.items, self.records, changed, POLICY)
        # Ordinary zero grades and truncation already pass the first test.

    def test_invalid_response_is_charged_before_parse_and_never_edits(self):
        handoff.export_packet(self.root)
        response = self.root / "response.json"
        response.write_text("{invalid")
        with patch.object(handoff, "run_checkpoint_action", side_effect=AssertionError("unexpected edit")):
            with self.assertRaises(ValueError):
                handoff.import_response(self.root, response)
            with self.assertRaisesRegex(ValueError, "already reserved"):
                handoff.import_response(self.root, response)
        request = read_json(self.root / "handoff/request.json")
        with handoff._ledger(self.root, request) as ledger:
            self.assertEqual(ledger.attempts()[0]["status"], "failed")
            self.assertEqual(ledger.summary()["charged"]["editor_requests"], 1)
            self.assertEqual(ledger.summary()["charged"]["generated_tokens"], 0)

    def test_envelope_unknown_keys_and_wrong_digest_rejected(self):
        handoff.export_packet(self.root)
        request = read_json(self.root / "handoff/request.json")
        response = {"schema_version": 1, "request_id": request["request_id"],
            "request_sha256": canonical_hash(request), "proposal": {"provenance": "editor_proposal"}}
        self.assertEqual(handoff.validate_response(response, request), response["proposal"])
        for change in ({"request_sha256": "0" * 64}, {"model_authored_usage": 100}):
            with self.assertRaisesRegex(ValueError, "envelope"):
                handoff.validate_response(dict(response, **change), request)

    def test_fixed_sampled_inspection_plan_respects_existing_scan_budget(self):
        policy = read_json("configs/candidate_policy.json")
        tensors = {name: {"shape": [1024, 3584]} for name in policy["allowed_tensors"]}
        plan = handoff.inspection_plan(policy, {"tensors": tensors})
        self.assertEqual(len(plan), 49)
        scanned = sum((p["slices"][0][1] - p["slices"][0][0]) *
                      (p["slices"][1][1] - p["slices"][1][0]) for p in plan[1:])
        self.assertEqual(scanned, 1015808)
        self.assertLessEqual(scanned, policy["inspection"]["max_total_scanned_coordinates"])


if __name__ == "__main__":
    unittest.main()
