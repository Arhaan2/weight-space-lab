"""No inference: state, provenance, and single-pair orchestration acceptance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_checkpoint import checkpoint_fixture, fixture_policy, EDITABLE
from weight_space_lab import mechanistic_study as study
from weight_space_lab.inventory import inventory_model, file_sha256
from weight_space_lab.protocol import canonical_hash


class MechanisticStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.model = checkpoint_fixture(self.base)
        self.root = self.base / "study"
        self.root.mkdir()
        self.config = json.loads(study.DEFAULT_CONFIG.read_text())
        self.policy = fixture_policy()
        self.policy["direction_strength_range"] = [0, 6]
        self.policy["max_abs_change"] = 100
        self.inventory = inventory_model(self.model)
        self.source_sha = canonical_hash(self.inventory["files"])
        self.baseline_manifest = self.base / "baseline-manifest.json"
        study._write(self.baseline_manifest, {"fixture": True})
        template = {"fixture": True}
        self.manifest = {"schema_version": 1, "kind": "mechanistic_browser_study", "study_id": "fixture",
            "config": self.config, "source_model": str(self.model), "source_sha256": self.source_sha,
            "source_files": self.inventory["files"], "policy_sha256": canonical_hash(self.policy),
            "items_sha256": canonical_hash([]), "scheduled_prompts": 0,
            "baseline_template_sha256": canonical_hash(template),
            "baseline_manifest": str(self.baseline_manifest), "baseline_manifest_sha256": file_sha256(self.baseline_manifest),
            "baseline_artifact": {"artifact_sha256": "b" * 64},
            "source_files_sha256": {str(Path(study.__file__)): file_sha256(study.__file__)},
            "activation_scope": "CPU source diagnostics", "claim_scope": "adaptive development"}
        for name, value in {"study": self.manifest, "ready": {"study_sha256": canonical_hash(self.manifest)},
                            "config": self.config, "policy": self.policy, "items": [], "baseline-template": template}.items():
            study._write(self.root / (name + ".json"), value)
        self.state = {"study_sha256": canonical_hash(self.manifest), "active": None, "actions": [],
                      "baseline": {"status": "completed"}, "candidate": None, "proposal": None, "deployment": None}
        self.save_state()

    def save_state(self):
        study._write(self.root / "state.json", self.state, replace=True)

    def proposal(self):
        return {"schema_version": 1, "base_sha256": self.source_sha, "provenance": "editor_proposal",
                "composition": "independent", "operations": [{"op": "remove_direction_from_columns",
                  "tensor": EDITABLE[0], "columns": [0], "direction": [1, 0], "strength": 6}]}

    def test_one_benchmark_one_pair_contract(self):
        self.assertEqual(study.validate_config(self.config)["schedule"]["runs_per_arm"], 1)
        for change in ({"accuracy_benchmark": "math"}, {"scope": "heldout"}):
            with self.assertRaises(ValueError): study.validate_config({**self.config, **change})
        bad = copy.deepcopy(self.config)
        bad["schedule"]["runs_per_arm"] = 4
        with self.assertRaises(ValueError): study.validate_config(bad)
        bad = copy.deepcopy(self.config)
        bad["schedule"]["separate_resource_workload"] = True
        with self.assertRaises(ValueError): study.validate_config(bad)

    def test_audited_directional_submission_only_once_and_original_preserved(self):
        before = inventory_model(self.model)["files"]
        result = study.submit(self.root, self.proposal(), rationale="Fixture only")
        self.assertNotEqual(result["source_sha256"], self.source_sha)
        self.assertEqual(inventory_model(self.model)["files"], before)
        self.assertEqual(json.loads((self.root / "state.json").read_text())["actions"][0]["status"], "completed")
        with self.assertRaises(ValueError): study.submit(self.root, self.proposal())

    def test_invalid_submission_spends_slot_without_publishing_candidate(self):
        bad = self.proposal()
        bad["base_sha256"] = "f" * 64
        with self.assertRaises(ValueError): study.submit(self.root, bad)
        state = json.loads((self.root / "state.json").read_text())
        self.assertIsNone(state["proposal"])
        self.assertEqual(state["actions"][0]["status"], "failed")
        self.assertFalse((self.root / "candidate-source").exists())
        with self.assertRaisesRegex(ValueError, "capacity exhausted"):
            study.submit(self.root, self.proposal())

    def test_interrupted_action_cannot_dispatch_again(self):
        self.state["active"] = {"kind": "browser-baseline", "id": "interrupted"}
        self.save_state()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with study._action(self.root, "browser-baseline", 1):
                self.fail("Action must not start")

    def test_inspection_is_real_and_audited_and_closes_after_submission(self):
        request = {"study_sha256": canonical_hash(self.manifest), "tool": "inspect",
                   "arguments": {"op": "metadata", "tensors": [EDITABLE[0]]}}
        # The real checkpoint metadata path is exercised, rather than substituting values.
        result = study.tool(self.root, request)
        self.assertEqual(result["tool"], "inspect")
        self.assertTrue((self.root / "weight-exposures.jsonl").is_file())
        study.submit(self.root, self.proposal())
        with self.assertRaises(ValueError): study.tool(self.root, request)

    def test_tools_reject_foreign_study_and_incomplete_baseline(self):
        request = {"study_sha256": "f" * 64, "tool": "capture", "arguments": {}}
        with patch("weight_space_lab.mechanistic_tools.capture_activations") as capture:
            with self.assertRaises(ValueError): study.tool(self.root, request)
            self.state["baseline"] = {"status": "incomplete"}
            self.save_state()
            request["study_sha256"] = canonical_hash(self.manifest)
            with self.assertRaises(ValueError): study.tool(self.root, request)
            capture.assert_not_called()

    def test_failed_capture_is_retained_and_not_refunded(self):
        request = {"study_sha256": canonical_hash(self.manifest), "tool": "capture",
                   "arguments": {"source_sha256": self.source_sha}}
        with patch("weight_space_lab.mechanistic_tools.capture_activations", side_effect=TimeoutError("fixture timeout")):
            with self.assertRaises(TimeoutError): study.tool(self.root, request)
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(len(state["actions"]), 1)
        self.assertEqual(state["actions"][0]["kind"], "capture")
        self.assertEqual(state["actions"][0]["status"], "failed")
        self.assertIsNone(state["active"])

    def test_mutated_policy_or_template_cannot_continue(self):
        policy = copy.deepcopy(self.policy)
        policy["direction_strength_range"] = [0, 100]
        study._write(self.root / "policy.json", policy, replace=True)
        with self.assertRaisesRegex(ValueError, "Frozen"):
            study.load(self.root)
        study._write(self.root / "policy.json", self.policy, replace=True)
        study._write(self.root / "baseline-template.json", {"fixture": False}, replace=True)
        with self.assertRaisesRegex(ValueError, "Frozen"):
            study.load(self.root)

    def test_derivation_cannot_read_foreign_capture(self):
        request = {"study_sha256": canonical_hash(self.manifest), "tool": "derive",
                   "arguments": {"capture_id": "../../foreign", "top_k": 64}}
        with patch("weight_space_lab.mechanistic_tools.derive_directions") as derive:
            with self.assertRaisesRegex(ValueError, "completed capture"):
                study.tool(self.root, request)
            derive.assert_not_called()

    def test_source_drift_cannot_disclose_wrong_checkpoint_weights(self):
        # Freeze binds the original source. A fresh inspector would otherwise
        # happily fingerprint the replacement and disclose its different data.
        (self.model / "tokenizer.json").write_text(json.dumps({"fixture": "changed"}))
        request = {"study_sha256": canonical_hash(self.manifest), "tool": "inspect",
                   "arguments": {"op": "metadata", "tensors": [EDITABLE[0]]}}
        with self.assertRaisesRegex(ValueError, "frozen source"):
            study.tool(self.root, request)
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["actions"][-1]["status"], "failed")
        self.assertFalse((self.root / "diagnostics" / state["actions"][-1]["id"] / "result.json").exists())

    def test_derivation_rejects_resealed_replacement_capture(self):
        capture_id = "completed-capture"
        original = {"payload": {"source_sha256": self.source_sha, "arrays_sha256": "1" * 64},
                    "sha256": "2" * 64}
        directory = self.root / "diagnostics" / capture_id
        study._write(directory / "result.json", original)
        replacement = {"payload": {"source_sha256": self.source_sha, "arrays_sha256": "3" * 64},
                       "sha256": "4" * 64}
        study._write(directory / "capture/manifest.json", replacement)
        self.state["actions"].append({"id": capture_id, "kind": "capture", "status": "completed",
                                      "result_sha256": canonical_hash(original)})
        self.save_state()
        request = {"study_sha256": canonical_hash(self.manifest), "tool": "derive",
                   "arguments": {"capture_id": capture_id}}
        with patch("weight_space_lab.mechanistic_tools.derive_directions") as derive:
            with self.assertRaisesRegex(ValueError, "completed diagnostic receipt"):
                study.tool(self.root, request)
            derive.assert_not_called()
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["actions"][-1]["status"], "failed")

    def test_preparation_and_report_do_not_dispatch_models(self):
        self.state["baseline"] = None
        self.save_state()
        with patch("weight_space_lab.joint_browser.execute") as execute, \
             patch("weight_space_lab.mechanistic_tools.capture_activations") as capture:
            value = study.report(self.root)
            self.assertEqual(value["status"], "not_run")
            self.assertEqual(value["separate_resource_generations"], 0)
            self.assertEqual(value["deployment_outcome"], "not_converted")
            self.assertIsNone(value["conversion_erases_edit"])
            self.assertFalse(value["candidate_evaluation_skipped"])
            execute.assert_not_called()
            capture.assert_not_called()

    def conversion_fixture(self, *, erased, artifact_sha):
        proposal = study.submit(self.root, self.proposal())
        artifact_path = self.base / "candidate-artifact.json"
        study._write(artifact_path, {"source_checkpoint_sha256": proposal["source_sha256"],
                                    "artifact_sha256": artifact_sha})
        with patch("weight_space_lab.browser_candidate.convert") as convert, \
             patch("weight_space_lab.browser_candidate.package_candidate", return_value={
                 "artifact_manifest": str(artifact_path), "conversion_erases_edit": erased}):
            result = study.convert_candidate(self.root, "unused-fixture-python")
            convert.assert_called_once()
        return result

    def test_erased_edit_is_recorded_without_spending_candidate_browser_capacity(self):
        self.assertTrue(self.conversion_fixture(erased=True, artifact_sha="a" * 64)["conversion_erases_edit"])
        before = json.loads((self.root / "state.json").read_text())
        self.assertTrue(before["deployment"]["conversion_erases_edit"])
        self.assertEqual(before["actions"][-1]["kind"], "convert")
        self.assertEqual(before["actions"][-1]["status"], "completed")
        with patch("weight_space_lab.joint_browser.prepare_session") as prepare, \
             patch("weight_space_lab.joint_browser.execute") as execute:
            with self.assertRaisesRegex(ValueError, "Quantization erased"):
                study.run_arm(self.root, "candidate")
            prepare.assert_not_called()
            execute.assert_not_called()
        self.assertEqual(json.loads((self.root / "state.json").read_text()), before)
        with self.assertRaises(ValueError):
            study.submit(self.root, self.proposal())

    def test_identical_artifact_overrides_a_non_erased_conversion_flag(self):
        result = self.conversion_fixture(erased=False, artifact_sha="b" * 64)
        self.assertTrue(result["conversion_erases_edit"])
        with self.assertRaisesRegex(ValueError, "Quantization erased"):
            study.run_arm(self.root, "candidate")

    def test_report_discloses_erased_edit_without_a_candidate_result(self):
        self.conversion_fixture(erased=True, artifact_sha="a" * 64)
        state = json.loads((self.root / "state.json").read_text())
        baseline = {"status": "completed", "fixture": True}
        state["baseline"]["sha256"] = canonical_hash(baseline)
        study._write(self.root / "state.json", state, replace=True)
        with patch("weight_space_lab.joint_browser.verify_report", return_value=baseline), \
             patch("weight_space_lab.joint_browser.execute") as execute:
            result = study.report(self.root)
            execute.assert_not_called()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["deployment_outcome"], "erased_by_quantization")
        self.assertTrue(result["conversion_erases_edit"])
        self.assertTrue(result["candidate_evaluation_skipped"])
        self.assertEqual(result["arms"], {"baseline": baseline})
        self.assertIsNone(result["comparison"])


if __name__ == "__main__":
    unittest.main()
