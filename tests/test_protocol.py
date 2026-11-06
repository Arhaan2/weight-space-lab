import copy
import json
import unittest
from pathlib import Path

from weight_space_lab.protocol import validate_protocol, validate_split_manifest


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "configs/qwen_math_design.json").read_text())

    def test_template_never_claims_run_readiness(self):
        result = validate_protocol(self.config)
        self.assertTrue(result["valid_design"])
        self.assertFalse(result["run_configuration_complete"])
        self.assertTrue(result["pending_run_requirements"])

    def test_ready_mode_exposes_unpinned_revisions(self):
        result = validate_protocol(self.config, require_ready=True)
        self.assertTrue(any("tokenizer_revision" in issue for issue in result["issues"]))

    def test_moving_revision_rejected(self):
        config = copy.deepcopy(self.config)
        config["pins"]["model_revision"] = "main"
        self.assertIn("pin is not immutable: model_revision", validate_protocol(config, require_ready=True)["issues"])

    def test_boolean_budget_rejected(self):
        self.config["probe_limits"]["max_calls"] = True
        self.assertFalse(validate_protocol(self.config)["valid_design"])

    def test_overlap_rejected_even_when_identifier_differs(self):
        rows = [
            {"item_id": "a", "split": "development", "task_text": "What is 2 + 2?"},
            {"item_id": "b", "split": "validation", "task_text": "Different task"},
            {"item_id": "c", "split": "test", "task_text": "What   is 2 + 2?"},
        ]
        with self.assertRaises(ValueError):
            validate_split_manifest(rows)

    def test_distinct_splits_have_content_hash(self):
        rows = [{"item_id": str(i), "split": s, "task_text": f"task {i}"}
                for i, s in enumerate(["development", "validation", "test"])]
        result = validate_split_manifest(rows)
        self.assertEqual(result["counts"]["test"], 1)
        self.assertEqual(len(result["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
