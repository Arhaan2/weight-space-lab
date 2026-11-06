"""Offline replay orchestration checks; synthetic inputs never run a model."""
import contextlib
import copy
import importlib
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from test_benchmark_data import fixture_reference, item
from weight_space_lab import published_study as study
from weight_space_lab.protocol import canonical_hash

REPO = Path(__file__).parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReplaySelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.plan = study._plan(REPO / "configs/published_replay_plan.json")

    def test_published_counts_are_benchmark_specific_and_stage_budgets_distinct(self):
        expected = {"math": (500, [0, 1, 2, 3]), "gsm8k": (1319, [0, 1]), "ifeval": (541, [0, 1])}
        self.assertEqual({k: (v["items"], v["seeds"]) for k, v in self.plan["full_benchmarks"].items()}, expected)
        self.assertEqual([self.plan["stages"][s]["max_new_tokens"] for s in ("smoke", "pilot", "full")], [16, 128, 32768])
        self.assertEqual(sum(n * len(seeds) for n, seeds in expected.values()) * len(study.ARMS), 28600)
        for stage in ("smoke", "pilot"):
            self.assertIsNotNone(self.plan["stages"][stage]["items_per_benchmark"])

    def test_plan_rejects_schedule_metric_and_analysis_drift(self):
        changes = [
            lambda p: p["full_benchmarks"]["gsm8k"].update(seeds=[0, 1, 2, 3]),
            lambda p: p["full_benchmarks"]["ifeval"].update(items=66),
            lambda p: p["full_benchmarks"]["ifeval"].update(metric="prompt_level_loose_accuracy"),
            lambda p: p["stages"]["smoke"].update(max_new_tokens=32768),
            lambda p: p["analysis"].update(bootstrap_samples=1),
            lambda p: p.update(decoding_selection="chosen on final test"),
            lambda p: p.update(alternative_decoding="greedy decoding"),
            lambda p: p.update(unfrozen_extra_option=True),
            lambda p: p["full_benchmarks"].update(extra_benchmark={"items": 1}),
        ]
        for change in changes:
            plan = copy.deepcopy(self.plan)
            change(plan)
            path = self.base / "changed-plan.json"
            write_json(path, plan)
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                study._plan(path)

    def test_task_wrapper_keeps_source_targets_unchanged_and_ifeval_exact(self):
        original = [item(name, "source-1", "Exact  source\nwith whitespace.") for name in study.BENCHMARKS]
        preserved = copy.deepcopy(original)
        tasks = study.replay_tasks(original)
        self.assertEqual(original, preserved)
        for row in tasks:
            suffix = study.MATH_SUFFIX if row["benchmark"] in {"math", "gsm8k"} else ""
            self.assertEqual(row["messages"], [{"role": "user", "content": original[0]["task_text"] + suffix}])
            self.assertNotIn("target", row)
            self.assertNotIn("metadata", row)
        self.assertEqual(study.MATH_SUFFIX, "\nPlease reason step by step, and put your final answer within \\boxed{}.")

    def test_development_selection_is_order_independent_and_never_reads_final_tasks(self):
        data = self.base / "data"
        rows = [item(name, identity, f"Independent fixture {identity}.")
                for name in study.BENCHMARKS for identity in ("z", "b", "a")]
        write_json(data / "source-reference.json", {"sources": []})
        write_json(data / "editor-development/items.json", rows)
        with patch.object(study, "load_source", side_effect=AssertionError("development stage must not load final sources")):
            smoke = study._selected_items(data, self.base, "smoke", self.plan)
            pilot = study._selected_items(data, self.base, "pilot", self.plan)
        self.assertEqual({i["item_id"] for i in smoke}, {"a"})
        self.assertEqual({i["item_id"] for i in pilot}, {"a", "b"})
        write_json(data / "editor-development/items.json", list(reversed(rows)))
        self.assertEqual(pilot, study._selected_items(data, self.base, "pilot", self.plan))
        rows[0]["split"] = "test"
        write_json(data / "editor-development/items.json", rows)
        with self.assertRaisesRegex(ValueError, "held-out"):
            study._selected_items(data, self.base, "smoke", self.plan)

    def test_full_source_projection_restores_every_ifeval_row_without_mutating_bundle(self):
        cache, data = self.base / "cache", self.base / "data"
        cache.mkdir()
        reference = fixture_reference(cache)
        write_json(data / "source-reference.json", reference)
        # Deliberately tiny custom held-out subset: full replay must ignore it.
        heldout = [item("ifeval", "only-custom-heldout", split="test")]
        write_json(data / "evaluator-final/items.json", heldout)
        plan = copy.deepcopy(self.plan)
        plan["full_benchmarks"]["math"]["items"] = 1
        plan["full_benchmarks"]["gsm8k"]["items"] = 1
        plan["full_benchmarks"]["ifeval"]["items"] = 12
        selected = study._selected_items(data, cache, "full", plan)
        prompts = [row for row in selected if row["benchmark"] == "ifeval"]
        self.assertEqual(len(prompts), 12)
        self.assertEqual({i["item_id"] for i in prompts}, {str(n) for n in range(12)})
        self.assertTrue(all(i["split"] == "test" for i in selected))
        self.assertTrue(all(i["source"]["split"] == "partition" for i in prompts))
        self.assertEqual(json.loads((data / "evaluator-final/items.json").read_text()), heldout)
        self.assertTrue(all("source_file_sha256" in i["metadata"] for i in selected))
        plan["full_benchmarks"]["ifeval"]["items"] = 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            study._selected_items(data, cache, "full", plan)


class ReplayWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.editor = self.base / "replay", self.base / "fresh-editor"

    def test_disjoint_boundary_persists_and_rejects_changed_reservation(self):
        study.guard_workspace(self.root, self.editor, create=True)
        self.assertEqual(study.guard_workspace(self.root), self.root)
        with self.assertRaisesRegex(ValueError, "changed"):
            study.guard_workspace(self.root, self.base / "different-editor")
        self.assertFalse(self.editor.exists())

    def test_either_containment_direction_or_same_directory_is_rejected(self):
        for root, editor in ((self.root, self.root), (self.root, self.root / "editor"),
                             (self.editor / "replay", self.editor)):
            with self.subTest(root=root, editor=editor), self.assertRaisesRegex(ValueError, "disjoint"):
                study.guard_workspace(root, editor, create=True)

    def test_new_root_does_not_adopt_existing_files(self):
        self.root.mkdir()
        (self.root / "existing-evidence.json").write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "empty"):
            study.guard_workspace(self.root, self.editor, create=True)
        self.assertEqual((self.root / "existing-evidence.json").read_text(), "preserve me")
        self.assertFalse((self.root / "workspace-boundary.json").exists())

    def test_rechecks_editor_symlink_alias_on_every_use(self):
        study.guard_workspace(self.root, self.editor, create=True)
        self.editor.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            study.guard_workspace(self.root)

    def test_artifact_parent_symlink_cannot_escape_replay_root(self):
        study.guard_workspace(self.root, self.editor, create=True)
        self.editor.mkdir()
        (self.root / "full").symlink_to(self.editor, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            study._inside(self.root, self.root / "full/reports/new/report.json")
        with self.assertRaisesRegex(ValueError, "escapes"):
            study._inside(self.root, self.root / "../fresh-editor/report.json")
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            study.guard_workspace(alias)


class ReplayFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.editor = self.base / "replay", self.base / "editor"
        self.data, self.cache = self.base / "data", self.base / "cache"
        self.acquisition, self.reconstruction, self.identity = [self.base / p for p in ("acquisition", "reconstruction", "identity")]
        self.inventory = {"files": {"config.json": {"sha256": canonical_hash("synthetic model bytes"), "size_bytes": 1}},
                          "chat_templates": {"chat_template.jinja": {"sha256": canonical_hash("synthetic chat template")}}}
        self.rebuilt = copy.deepcopy(self.inventory)
        self.rebuilt["files"]["config.json"]["sha256"] = canonical_hash("synthetic changed model bytes")
        write_json(self.acquisition / "acquisition.json", {"fixture": "acquisition audit"})
        write_json(self.reconstruction / "audit.json", {"fixture": "reconstruction audit"})
        write_json(self.identity / "audit.json", {"fixture": "identity audit"})
        (self.acquisition / "patch").mkdir()
        (self.acquisition / "patch/edit.npz").write_bytes(b"synthetic patch, never loaded")
        write_json(self.data / "source-reference.json", {"sources": []})
        write_json(self.data / "editor-development/items.json",
                   [item(name, f"development-{n}") for name in study.BENCHMARKS for n in (0, 1)])
        self.options = dict(root=self.root, fresh_editor_workspace=self.editor, acquisition=self.acquisition,
                            reconstruction=self.reconstruction, identity=self.identity, data=self.data,
                            cache=self.cache, backend="transformers", stage="smoke",
                            reference=REPO / "configs/published_p50c.json",
                            plan_path=REPO / "configs/published_replay_plan.json",
                            policy_path=REPO / "configs/grading-policy.json",
                            smoke_measurement=REPO / "reports/native-smoke/report.json")

    def freeze(self, **changes):
        # Hash/file verification is tested in artifact suites. Here only small
        # independent metadata fixtures enter the real orchestration/validation.
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(study, "verify_acquisition", return_value={"model_inventory": self.inventory}))
            stack.enter_context(patch.object(study, "verify_reconstruction", return_value={"model_inventory": self.rebuilt}))
            stack.enter_context(patch.object(study, "verify_identity", return_value=self.inventory))
            stack.enter_context(patch.object(study, "check_benchmark_bundle", return_value={"overlap_audit_sha256": canonical_hash("synthetic overlap audit")}))
            forbidden = stack.enter_context(patch("weight_space_lab.native_runner.run_schedule", side_effect=AssertionError("freezing must never execute inference")))
            result = study.prepare_study(**(self.options | changes))
            forbidden.assert_not_called()
            return result

    def certificate(self, *, passed=True, binding=None, stage="smoke", freeze_hash=None):
        frozen = study.load_study(self.root, "smoke")
        value = {"stage": stage, "freeze_sha256": freeze_hash or frozen["freeze_sha256"],
                 "development_validation": {"passed": passed,
                     "binding_sha256": binding or frozen["declaration"]["binding_sha256"]}}
        path = self.root / "smoke/reports/fixture/report.json"
        write_json(path, study._sealed(value, "report_sha256"))
        return path

    def test_freeze_builds_matched_arms_and_full_denominators_without_execution(self):
        result = self.freeze()
        self.assertFalse(result["inference_executed"])
        self.assertEqual(result["scheduled_generations"], 15)
        frozen = study.load_study(self.root, "smoke", execution=True)
        for benchmark in study.BENCHMARKS:
            jobs = frozen["jobs"]
            primary = jobs[f"{benchmark}-original"]["config"]
            changed = jobs[f"{benchmark}-reconstructed"]["config"]
            identity = jobs[f"{benchmark}-identity"]["config"]
            self.assertEqual(primary["backend"], changed["backend"])
            self.assertEqual(primary["checkpoint"], identity["checkpoint"])
            self.assertNotEqual(primary["checkpoint"], changed["checkpoint"])
            alternative = jobs[f"{benchmark}-original_temperature1"]["config"]
            expected = copy.deepcopy(primary["backend"])
            expected["sampling"]["temperature"] = 1.0
            self.assertEqual(alternative["backend"], expected)
        self.assertFalse((self.root / "full").exists())
        self.assertFalse((self.root / "smoke/runs").exists())
        with self.assertRaisesRegex(ValueError, "already frozen"):
            self.freeze()

    def test_pilot_and_full_cannot_freeze_without_development_certificate(self):
        self.freeze()
        for stage in ("pilot", "full"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "development smoke"):
                self.freeze(stage=stage)
            self.assertFalse((self.root / stage / "inputs").exists())

    def test_development_gate_rejects_failed_stale_and_wrong_stage_reports(self):
        self.freeze()
        frozen = study.load_study(self.root, "smoke")
        binding = frozen["declaration"]["binding_sha256"]
        for changes in ({"passed": False}, {"binding": canonical_hash("different runtime")},
                        {"stage": "full"}, {"freeze_hash": canonical_hash("other smoke")}):
            path = self.certificate(**changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "validation"):
                study._development_gate(self.root, path, binding)
        path = self.certificate()
        self.assertEqual(study._development_gate(self.root, path, binding), canonical_hash(json.loads(path.read_text())))
        value = json.loads(path.read_text())
        value["development_validation"]["passed"] = False
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "digest"):
            study._development_gate(self.root, path, binding)

    def test_successful_development_binding_unlocks_only_explicit_pilot(self):
        self.freeze()
        certificate = self.certificate()
        pilot = self.freeze(stage="pilot", development_report=certificate)
        self.assertEqual(pilot["scheduled_generations"], 30)
        frozen = study.load_study(self.root, "pilot", execution=True)
        self.assertTrue(all(job["config"]["profile"] == "fixed_patch_evaluation" for job in frozen["jobs"].values()))
        self.assertTrue(all(job["config"]["backend"]["sampling"]["max_new_tokens"] == 128 for job in frozen["jobs"].values()))
        self.assertFalse((self.root / "full").exists())
        self.assertFalse((self.root / "pilot/runs").exists())

    def test_gate_rejects_different_backend_even_with_valid_development_report(self):
        self.freeze()
        certificate = self.certificate()
        with self.assertRaisesRegex(ValueError, "validation"):
            self.freeze(stage="full", backend="vllm", development_report=certificate)
        self.assertFalse((self.root / "full/inputs").exists())

    def test_frozen_source_or_inputs_drift_is_rejected_before_any_run(self):
        self.freeze()
        with patch.object(study, "implementation_identity", return_value=canonical_hash("changed implementation")):
            with self.assertRaisesRegex(ValueError, "source changed"):
                study.load_study(self.root, "smoke")
        path = self.root / "smoke/inputs/study.json"
        frozen = json.loads(path.read_text())
        frozen["benchmarks"]["math"]["tasks"][0]["messages"][0]["content"] = "changed prompt"
        write_json(path, frozen)
        with self.assertRaisesRegex(ValueError, "digest"):
            study.load_study(self.root, "smoke", execution=True)

    def test_run_executes_only_requested_stage_and_unknown_job_does_nothing(self):
        self.freeze()
        result = {"statuses": {"ok": 1, "truncated": 0, "error": 0, "timeout": 0, "missing": 0}}
        with patch("weight_space_lab.native_runner.run_schedule", return_value=result) as runner:
            selected = study.run_study(self.root, "smoke", acquisition=self.acquisition,
                                       reconstruction=self.reconstruction, identity=self.identity,
                                       job="gsm8k-original")
            self.assertEqual(set(selected["jobs"]), {"gsm8k-original"})
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0]["stage"], "smoke")
            with self.assertRaisesRegex(ValueError, "unknown"):
                study.run_study(self.root, "smoke", acquisition=self.acquisition,
                                reconstruction=self.reconstruction, identity=self.identity, job="full")
            self.assertEqual(runner.call_count, 1)
        self.assertFalse((self.root / "full").exists())


class ReplayNoImplicitExecutionTests(unittest.TestCase):
    def test_import_and_standard_build_metadata_have_no_execution_hook(self):
        with patch("weight_space_lab.native_runner.run_schedule", side_effect=AssertionError("import launched inference")) as runner:
            importlib.reload(study)
            importlib.import_module("weight_space_lab.__main__")
            runner.assert_not_called()
        metadata = tomllib.loads((REPO / "pyproject.toml").read_text())
        self.assertEqual(metadata["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(metadata["project"]["scripts"], {"weight-space-lab": "weight_space_lab.__main__:main"})
        self.assertFalse((REPO / "setup.py").exists())
        self.assertFalse((REPO / "setup.cfg").exists())


class ReplayCommandTests(unittest.TestCase):
    def test_stage_commands_dispatch_exactly_once_without_promotion(self):
        from weight_space_lab.__main__ import main
        common = ["--root", "fixture-replay", "--acquisition", "fixture-base",
                  "--reconstruction", "fixture-rebuilt", "--identity", "fixture-identity"]
        for stage in ("smoke", "pilot", "full"):
            with self.subTest(stage=stage), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(study, "run_study", return_value={"successful_execution": True}) as run:
                    self.assertEqual(main(["replay-" + stage, *common, "--resume"]), 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.args, (Path("fixture-replay"), stage))
                self.assertTrue(run.call_args.kwargs["resume"])
        with contextlib.redirect_stdout(io.StringIO()), patch.object(study, "run_study", return_value={"successful_execution": False}):
            self.assertEqual(main(["replay-smoke", *common]), 2)

    def test_prepare_command_never_dispatches_execution_even_for_full(self):
        from weight_space_lab.__main__ import main
        args = ["prepare-published-replay", "--stage", "full", "--backend", "vllm"]
        for name in ("root", "fresh-editor-workspace", "acquisition", "reconstruction", "identity", "data", "cache"):
            args.extend(["--" + name, "fixture-" + name])
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(study, "prepare_study", return_value={"inference_executed": False}) as prepare:
                with patch.object(study, "run_study", side_effect=AssertionError("prepare dispatched full inference")) as execute:
                    self.assertEqual(main(args), 0)
                    execute.assert_not_called()
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["stage"], "full")
            self.assertIsNone(prepare.call_args.kwargs["development_report"])

    def test_failed_development_report_surfaces_nonzero_exit_without_generation(self):
        from weight_space_lab.__main__ import main
        report = {"kind": "published_patch_replay_report", "stage": "smoke", "freeze_sha256": "a" * 64,
                  "report_sha256": "b" * 64, "measurement_status": "partial", "measured_result_pending": True,
                  "development_validation": {"passed": False}}
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(study, "report_study", return_value=report) as reporting:
                with patch.object(study, "run_study", side_effect=AssertionError("report generated model output")) as execute:
                    self.assertEqual(main(["report-published-replay", "--root", "fixture-root", "--stage", "smoke",
                                           "--report-id", "attempt-one"]), 2)
                    execute.assert_not_called()
            reporting.assert_called_once()


if __name__ == "__main__":
    unittest.main()
