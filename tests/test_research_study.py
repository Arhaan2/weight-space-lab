"""Research-stage isolation and a single final comparison; no target inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_benchmark_data import item
from test_checkpoint import checkpoint_fixture, EDITABLE
from weight_space_lab import research_study as study
from weight_space_lab.inventory import inventory_model, file_sha256
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.research_checkpoint import EditWorkspace, verify_materialized


class ResearchStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.model = checkpoint_fixture(self.base)
        self.root = self.base / "study"
        self.root.mkdir()
        for name in ("practice", "work", "snapshots", "diagnostics"):
            (self.root / name).mkdir()
        self.config = json.loads(study.DEFAULT_CONFIG.read_text())
        self.practice = [item("ifeval", "practice", "Write a practice sentence.")]
        self.final = [item("ifeval", "final", "This final prompt must stay private.", "test")]
        self.final[0]["target"]["instruction_id_list"] = ["fixture:heldout"]
        self.overlap = study.validate_splits(self.practice, self.final)
        self.manifest_path = self.base / "baseline.json"
        study._write(self.manifest_path, {"fixture": True})
        inventory = inventory_model(self.model)
        self.source_sha = canonical_hash(inventory["files"])
        self.template = {"fixture": True}
        self.manifest = {"schema_version": 1, "kind": "research_browser_study", "study_id": "fixture",
            "config": self.config, "source_model": str(self.model), "source_sha256": self.source_sha,
            "source_files": inventory["files"], "practice_sha256": canonical_hash(self.practice),
            "final_sha256": canonical_hash(self.final), "overlap_sha256": canonical_hash(self.overlap),
            "baseline_template_sha256": canonical_hash(self.template),
            "baseline_manifest": str(self.manifest_path), "baseline_manifest_sha256": file_sha256(self.manifest_path),
            "baseline_artifact": {"artifact_sha256": "b" * 64, "source_checkpoint_sha256": self.source_sha},
            "source_files_sha256": {}, "benchmark_files_sha256": {}, "scheduled_prompts": 1,
            "claim_scope": "Withheld from this editor session", "activation_scope": "CPU source diagnostics",
            "official_ifeval_root": str(self.base / "official"), "benchmark": str(self.base / "benchmark"),
            "environment": {"fixture": True}}
        for name, value in {"study.json": self.manifest, "ready.json": {"study_sha256": canonical_hash(self.manifest)},
                "config.json": self.config, "practice/items.json": self.practice, "final-items.json": self.final,
                "baseline-template.json": self.template, "overlap-audit.json": self.overlap}.items():
            study._write(self.root / name, value)
        self.state = {"study_sha256": canonical_hash(self.manifest), "active": None, "actions": [],
            "kernel": None, "snapshots": {}, "frozen": None, "development_closed": False,
            "deployment": None, "baseline": None, "candidate": None}
        self.save_state()
        self.workspace = EditWorkspace(self.model, self.root / "work")

    def save_state(self):
        study._write(self.root / "state.json", self.state, replace=True)

    def edit(self, amount=1):
        values = self.workspace.load_original(EDITABLE[0])
        values[0, 0] += amount
        self.workspace.save_edit(EDITABLE[0], values)
        return values

    def freeze(self):
        self.edit()
        study.snapshot(self.root, "best", evidence="Synthetic fixture only")
        with patch("weight_space_lab.research_kernel.stop_kernel"):
            return study.freeze(self.root, "best", rationale="A fixture, not a model-performance claim")

    def test_schedule_has_separate_research_and_single_final_budgets(self):
        config = study.validate_config(self.config)
        self.assertEqual(config["limits"]["python_calls"], 300)
        self.assertEqual(config["limits"]["forward_calls"], 50)
        for section, key, value in (("schedule", "runs_per_arm", 4), ("schedule", "separate_resource_workload", True),
                ("limits", "candidate_submissions", 2), ("limits", "conversions", 2), ("limits", "automatic_retries", 1)):
            invalid = copy.deepcopy(config)
            invalid[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_config(invalid)

    def test_practice_final_identity_text_family_and_source_overlap_rejected(self):
        for field in ("item_id", "task_text", "target", "source"):
            final = copy.deepcopy(self.final)
            final[0][field] = copy.deepcopy(self.practice[0][field])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlap"):
                study.validate_splits(self.practice, final)

    def test_packet_withholds_final_items_and_allows_prebaseline_research(self):
        result = study.editor_packet(self.root)
        encoded = json.dumps(result)
        self.assertIn(self.practice[0]["task_text"], encoded)
        self.assertNotIn(self.final[0]["task_text"], encoded)
        self.assertNotIn("fixture:heldout", encoded)
        self.assertNotIn("baseline_saved_outputs", result)
        self.assertIn("python", result["tools"])
        self.assertIn("forward", result["tools"])

    def test_final_prompts_blocked_in_raw_chats_custom_generation_and_ids(self):
        text = self.final[0]["task_text"]
        for request in ({"texts": ["Prefix " + text + " continuation"]}, {"prompts": [text]},
                {"chats": [[{"role": "user", "content": text}]]}, {"problems": ["final"]}):
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, "Final workload"):
                study._reject_final_context(self.root, request)

    def test_multiple_edits_restore_and_freeze_saved_choice_not_current(self):
        expected = self.edit(1)
        study.snapshot(self.root, "first")
        self.edit(2)
        study.snapshot(self.root, "second")
        study.restore(self.root, "first")
        self.assertEqual(self.workspace.load(EDITABLE[0]).tolist(), expected.tolist())
        self.edit(3)
        state = read_state(self.root)
        state["kernel"] = {"control_dir": str(self.root / "fixture-control")}
        study._write(self.root / "state.json", state, replace=True)
        with patch("weight_space_lab.research_kernel.stop_kernel") as stop:
            frozen = study.freeze(self.root, "first", rationale="Freeze a named earlier candidate")
            stop.assert_called_once()
        audit = verify_materialized(self.model, frozen["path"])
        self.assertEqual(audit["result_sha256"], frozen["source_sha256"])
        self.assertNotEqual(frozen["source_sha256"], self.source_sha)
        self.assertEqual(inventory_model(self.model)["files"], self.manifest["source_files"])

    def test_freeze_closes_all_editor_routes_before_final_results(self):
        self.freeze()
        calls = [lambda: study.execute_python(self.root, "1+1"), lambda: study.forward(self.root, {}),
            lambda: study.snapshot(self.root, "later"), lambda: study.restore(self.root, "best"),
            lambda: study.start_kernel(self.root), lambda: study.editor_packet(self.root),
            lambda: study.freeze(self.root, "best", rationale="Again")]
        for call in calls:
            with self.assertRaisesRegex(ValueError, "closed"):
                call()

    def test_failed_freeze_spends_slot_and_closes_development(self):
        with patch("weight_space_lab.research_kernel.stop_kernel"):
            with self.assertRaisesRegex(ValueError, "completed named snapshot"):
                study.freeze(self.root, "missing", rationale="Invalid selection")
        state = read_state(self.root)
        self.assertTrue(state["development_closed"])
        self.assertIsNone(state["frozen"])
        self.assertEqual(state["actions"][-1]["status"], "failed")
        with self.assertRaisesRegex(ValueError, "closed"):
            study.snapshot(self.root, "reopen")

    def test_failed_probe_is_spent_preserved_and_not_a_final_run(self):
        with patch("weight_space_lab.research_probes.run_probe", side_effect=TimeoutError("fixture")) as probe:
            with self.assertRaises(TimeoutError):
                study.forward(self.root, {"type": "teacher_forced", "model": "original", "texts": ["practice"]})
            probe.assert_called_once()
        state = read_state(self.root)
        self.assertEqual([(a["kind"], a["status"]) for a in state["actions"]], [("forward", "failed")])
        self.assertIsNone(state["baseline"])
        self.assertIsNone(state["candidate"])
        self.assertTrue((self.root / "diagnostics" / state["actions"][0]["id"] / "request.json").exists())

    def test_paired_probe_binds_a_materialized_candidate_and_practice_only(self):
        self.edit()
        result = {"payload": {"forward_sequences": 2}, "sha256": "a" * 64}
        def observed(*args, **kwargs):
            self.assertTrue(args[1].is_dir())
            return result
        with patch("weight_space_lab.research_probes.run_probe", side_effect=observed) as probe:
            study.forward(self.root, {"type": "generate", "model": "both", "problems": ["practice"]})
        args, kwargs = probe.call_args
        self.assertFalse(args[1].exists())
        self.assertTrue((args[1].parent / "audit.json").exists())
        self.assertEqual(kwargs["practice_items"], self.practice)
        self.assertEqual(kwargs["expected_original_sha256"], self.source_sha)
        self.assertNotEqual(kwargs["expected_candidate_sha256"], self.source_sha)
        self.assertEqual(len(read_state(self.root)["actions"]), 1)

    def test_forward_budget_exhaustion_and_crash_do_not_restart(self):
        self.state["actions"] = [{"kind": "forward", "status": "failed", "elapsed_seconds": 0}] * 50
        self.save_state()
        with self.assertRaisesRegex(ValueError, "capacity"):
            study.forward(self.root, {})
        self.state["active"] = {"kind": "forward", "id": "interrupted"}
        self.save_state()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            study.execute_python(self.root, "1")

    def test_source_drift_cannot_reach_kernel(self):
        (self.model / "tokenizer.json").write_text('{"changed":true}')
        with patch("weight_space_lab.research_kernel.execute_kernel") as execute:
            with self.assertRaisesRegex(ValueError, "source checkpoint changed"):
                study.execute_python(self.root, "load(NAMES[0])")
            execute.assert_not_called()

    def test_no_final_baseline_before_freeze_and_conversion(self):
        with patch("weight_space_lab.joint_browser.execute") as execute:
            with self.assertRaisesRegex(ValueError, "Freeze"):
                study.run_arm(self.root, "baseline")
            self.freeze()
            with self.assertRaisesRegex(ValueError, "audited browser candidate"):
                study.run_arm(self.root, "baseline")
            execute.assert_not_called()

    def test_exactly_one_joint_final_run_per_arm_with_identical_items(self):
        frozen = self.freeze()
        candidate = {"artifact_sha256": "c" * 64, "source_checkpoint_sha256": frozen["source_sha256"]}
        study._write(self.root / "deployment.json", candidate)
        state = read_state(self.root)
        state["deployment"] = {"manifest": str(self.root / "deployment.json"), "sha256": canonical_hash(candidate),
                               "conversion_erases_edit": False}
        study._write(self.root / "state.json", state, replace=True)
        result = {"status": "completed", "quality": {"strict": 1}, "resources": {"tokens": 7}}
        with patch.object(study, "_task_config", return_value={}), \
                patch("weight_space_lab.joint_browser.prepare_session") as prepare, \
                patch("weight_space_lab.joint_browser.execute", return_value=result) as execute:
            with self.assertRaisesRegex(ValueError, "baseline"):
                study.run_arm(self.root, "candidate")
            self.assertEqual(study.run_arm(self.root, "baseline"), result)
            self.assertEqual(study.run_arm(self.root, "candidate"), result)
            for arm in ("baseline", "candidate"):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    study.run_arm(self.root, arm)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual([c.kwargs["items"] for c in prepare.call_args_list], [self.final, self.final])
        self.assertTrue(read_state(self.root)["development_closed"])

    def test_mutated_final_prompts_rejected_before_any_action(self):
        study._write(self.root / "final-items.json", [], replace=True)
        with self.assertRaisesRegex(ValueError, "Frozen"):
            study.editor_packet(self.root)

    def test_explicit_kernel_restart_uses_new_control_path_and_requires_cleanup(self):
        with patch("weight_space_lab.research_kernel.start_kernel", return_value={"status": "ready"}) as start:
            study.start_kernel(self.root)
            first = start.call_args.args[2]
            with patch("weight_space_lab.research_kernel.kernel_status", return_value={"status": "ready"}):
                with self.assertRaisesRegex(ValueError, "Stop the existing"):
                    study.start_kernel(self.root)
            self.assertEqual(start.call_count, 1)
            with patch("weight_space_lab.research_kernel.kernel_status",
                       return_value={"status": "stopped", "process_exit_confirmed": True}):
                study.start_kernel(self.root)
            self.assertNotEqual(start.call_args.args[2], first)
            self.assertEqual(start.call_count, 2)

    def test_cleanup_without_started_kernel_and_erased_edit_reporting(self):
        self.assertEqual(study.stop_kernel(self.root)["status"], "not_started")
        self.state["deployment"] = {"conversion_erases_edit": True}
        self.save_state()
        result = study.report(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["deployment_outcome"], "erased_by_quantization")
        self.assertTrue(result["final_comparison_skipped"])

    def test_prepare_freezes_complete_limits_and_practice_final_separation_without_inference(self):
        from weight_space_lab.ifeval_baseline import SETTINGS
        template = {"settings": SETTINGS, "sourceCheckpointSha256": self.source_sha,
            "convertedArtifactSha256": "b" * 64,
            "items": [{"item_id": "final", "prompt": self.final[0]["task_text"], "rendered_input_tokens": 20}]}
        template_path = self.base / "template.json"
        study._write(template_path, template)
        new_root = self.base / "prepared"
        def bench(_source, destination):
            destination.mkdir()
            return destination
        with patch("weight_space_lab.browser_candidate.verify_baseline",
                   return_value={"arms": {"baseline": self.manifest["baseline_artifact"]}}), \
                patch("weight_space_lab.grading.verify_ifeval_assets", return_value={"fixture": True}), \
                patch("weight_space_lab.browser_resource.environment_identity", return_value={"fixture": True}), \
                patch.object(study, "prepare_benchmark", side_effect=bench), \
                patch("weight_space_lab.research_probes.run_probe") as probe, \
                patch("weight_space_lab.joint_browser.execute") as browser:
            result = study.prepare(new_root, source_model=self.model, baseline_manifest=self.manifest_path,
                baseline_config=template_path, benchmark=self.base / "input-bench",
                practice_items=self.root / "practice/items.json", final_items=self.root / "final-items.json",
                official_ifeval_root=self.base / "official", probe_python="/fixture/pinned/python")
            probe.assert_not_called()
            browser.assert_not_called()
        self.assertEqual(result["target_generations"], 0)
        _, manifest, _ = study.load(new_root)
        self.assertEqual(manifest["config"]["kernel_limits"]["wall_seconds"], 900)
        self.assertEqual(manifest["config"]["probe_limits"]["max_texts"], 1024)
        self.assertEqual(manifest["probe_python"], "/fixture/pinned/python")
        self.assertNotIn(self.final[0]["task_text"], json.dumps(study.editor_packet(new_root)))


def read_state(root):
    return json.loads((root / "state.json").read_text())


if __name__ == "__main__":
    unittest.main()
