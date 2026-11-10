"""Paired browser comparison preparation with synthetic immutable evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weight_space_lab import ifeval_comparison as comparison
from weight_space_lab.grading import _known_answer_item, _known_answer_record, grade_record, grader_identity, ifeval_metrics
from weight_space_lab.inventory import read_json


POLICY = {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
          "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
          "grader_timeout_seconds": 10, "grader_retry": {"mode": "none", "max_attempts": 1}}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.items = []
        for index in range(32):
            count = 1 if index == 0 else 3
            item = _known_answer_item("ifeval", {"instruction_id_list": ["fixture:constraint"] * count,
                "kwargs": [{}] * count})
            item["item_id"] = f"fixture-{index:02}"
            self.items.append(item)
        record = _known_answer_record(self.items[0], "fixture")
        self.config = {"runId": "synthetic-baseline", "configurationSha256": "a" * 64,
            "convertedArtifactSha256": "c" * 64, "settings": {"seed": 42, "thinking": False, "maxTokens": 1024},
            "items": [{"item_id": item["item_id"], "prompt": item["task_text"]} for item in self.items],
            "warmupPrompt": "Reply OK.", "totalLimitMs": 5400000}
        self.manifest = {"checkpoint": record["checkpoint"], "policy": POLICY, "grader": grader_identity("ifeval"),
            "items_path": str(self.root / "data/development-items.json")}
        self.save("manifest.json", self.manifest)
        self.save("baseline-config.json", self.config)
        self.save("data/development-items.json", self.items)
        self.save("data/manifest.json", {"fixture": "independent study source manifest"})
        self.save("handoff/import-result.json", {"accepted": True, "audit": {"result_sha256": "b" * 64}})
        self.artifact = {"source_checkpoint_sha256": "b" * 64, "artifact_sha256": "d" * 64,
            "baseline_artifact_sha256": "c" * 64,
            "source_identity": {"source_original_sha256": record["checkpoint"]["sha256"]}}
        self.save("candidate-artifact.json", self.artifact)
        self.save("baseline/graded.json", self.evidence("baseline", 1))

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def evidence(self, arm, passing):
        records, grades = [], []
        for index, item in enumerate(self.items):
            record = _known_answer_record(item, "independent synthetic output")
            record.update(evidence_kind="real_model", profile="browser_evaluation", stage="pilot", seed=42,
                run_id=f"synthetic-{arm}", configuration_sha256="a" * 64 if arm == "baseline" else "e" * 64)
            if arm == "candidate":
                record["checkpoint"]["sha256"] = "b" * 64
            flags = [index < passing] * len(item["target"]["kwargs"])
            metrics = ifeval_metrics(flags, flags)
            with patch("weight_space_lab.grading._worker", return_value={"status": "graded", "correct": all(flags),
                       "metrics": metrics, "code": "fixture_evaluator"}):
                grades.append(grade_record(record, item, POLICY))
            records.append(record)
        return {"records": records, "grades": grades, "summary": {"technically_valid": True}}

    def freeze_candidate(self, root, *, candidate_artifact, authorization):
        candidate = copy.deepcopy(self.manifest)
        candidate["checkpoint"]["sha256"] = "b" * 64
        candidate["result_dir"] = "candidate"
        config = copy.deepcopy(self.config)
        config.update(runId="synthetic-candidate", configurationSha256="e" * 64, convertedArtifactSha256="d" * 64)
        self.save("candidate-comparison/manifest.json", candidate)
        self.save("candidate-comparison/baseline-config.json", config)
        return candidate

    def test_plan_has_no_invented_candidate_identity_or_execution(self):
        result = comparison.prepare(self.root)
        plan = read_json(self.root / "candidate-comparison-plan.json")
        self.assertFalse(result["candidate_executed"])
        self.assertIsNone(plan["candidate_source_checkpoint_sha256"])
        self.assertIsNone(plan["candidate_artifact_sha256"])
        self.assertEqual(plan["candidate_generations_authorized_by_baseline_stage"], 0)
        with self.assertRaises(FileExistsError):
            comparison.prepare(self.root)

    def test_old_or_unaccepted_artifact_never_reaches_runner(self):
        comparison.prepare(self.root)
        self.artifact["source_checkpoint_sha256"] = "f" * 64
        self.save("candidate-artifact.json", self.artifact)
        with patch("weight_space_lab.ifeval_baseline.freeze_candidate", create=True,
                   side_effect=AssertionError("must reject before runner")):
            with self.assertRaisesRegex(ValueError, "accepted pristine-origin"):
                comparison.bind(self.root, self.root / "candidate-artifact.json", "unused-fixture-authorization")

    def test_comparison_reuses_official_instruction_denominator(self):
        comparison.prepare(self.root)
        with patch("weight_space_lab.ifeval_baseline.freeze_candidate", create=True, side_effect=self.freeze_candidate):
            result = comparison.bind(self.root, self.root / "candidate-artifact.json", "unused-fixture-authorization")
        self.assertFalse(result["candidate_executed"])
        self.save("candidate-comparison/candidate/graded.json", self.evidence("candidate", 2))
        result = comparison.compare(self.root)
        self.assertEqual(result["selection"]["selected_arm"], "candidate")
        official = result["candidate_projection"]["official_metrics"]["instruction_level_strict_accuracy"]
        self.assertEqual(official["denominator"], 94)
        self.assertEqual(official["numerator"], 4)
        self.assertEqual(result["selection"]["heldout_generations"], 0)

    def test_tampered_baseline_stops_future_run_before_dispatch(self):
        comparison.prepare(self.root)
        with patch("weight_space_lab.ifeval_baseline.freeze_candidate", create=True, side_effect=self.freeze_candidate):
            comparison.bind(self.root, self.root / "candidate-artifact.json", "unused-fixture-authorization")
        (self.root / "baseline/graded.json").write_text("{}")
        with patch("weight_space_lab.ifeval_baseline.execute", side_effect=AssertionError("unexpected dispatch")):
            with self.assertRaisesRegex(ValueError, "baseline evidence changed"):
                comparison.run(self.root)

    def test_failed_cleanup_blocks_selection_despite_all_outputs(self):
        comparison.prepare(self.root)
        with patch("weight_space_lab.ifeval_baseline.freeze_candidate", create=True, side_effect=self.freeze_candidate):
            comparison.bind(self.root, self.root / "candidate-artifact.json", "unused-fixture-authorization")
        evidence = self.evidence("candidate", 32)
        evidence["summary"]["technically_valid"] = False
        self.save("candidate-comparison/candidate/graded.json", evidence)
        result = comparison.compare(self.root)
        self.assertFalse(result["selection"]["technical_and_grading_complete"])
        self.assertIsNone(result["selection"]["selected_arm"])


if __name__ == "__main__":
    unittest.main()
