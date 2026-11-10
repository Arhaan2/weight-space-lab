"""Offline continuation authorization, accounting, history and fake-driver checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from weight_space_lab import ifeval_candidate_continuation as runner
from weight_space_lab import ifeval_supervisor_gate as gate
from weight_space_lab.editor_budget import BudgetLedger, BudgetExhausted


class ContinuationTests(unittest.TestCase):
    def config(self):
        return {"arm": "candidate", "runId": runner.RUN_ID,
            "configurationSha256": "a" * 64, "convertedArtifactSha256": runner.CANDIDATE_ARTIFACT,
            "sourceCheckpointSha256": runner.CANDIDATE_SOURCE,
            "items": [{"item_id": f"item-{i}", "prompt": "fixture"} for i in range(94, 128)]}

    def intent(self, index, config=None):
        config = config or self.config()
        return {"kind": "load" if index == 0 else "generation", "arm": "candidate",
            "request_id": "load-1" if index == 0 else f"generation-{index}",
            "attempt_id": "candidate-1", "code_version_sha256": runner.canonical_hash({}),
            "configuration_sha256": config["configurationSha256"], "artifact_sha256": config["convertedArtifactSha256"],
            "run_id": config["runId"], "warmup": index == 1, "sequence_index": index - 1,
            "max_tokens": 16 if index == 1 else 1024,
            "item_id": "__warmup__" if index == 1 else f"item-{index + 92}"}

    def authorize_fixture(self, root, manifest):
        runner.save(root / "continuation-plan.json", {"fixture": True})
        source = root / "fixture-approval.txt"
        source.write_text("Fixture-only explicit approval; no model execution")
        with patch.object(runner, "load", return_value=(root, manifest)):
            return runner.authorize(root, source, runner.canonical_hash({"fixture": True}))

    def test_unauthorized_execution_and_grading_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "nonexistent"
            with patch.object(runner, "_collector") as collector, patch.object(runner, "BudgetLedger") as ledger:
                with self.assertRaises(PermissionError):
                    runner.execute(root)
                with self.assertRaises(PermissionError):
                    runner.grade(root)
                collector.assert_not_called()
                ledger.assert_not_called()
            self.assertEqual(list(parent.iterdir()), [])

    def test_only_exact_tail_intents_are_allowed(self):
        config = self.config()
        for index in range(36):
            runner.validate_intent(self.intent(index), config, runner.canonical_hash({}), index)
        for index in (-1, 36, 96):
            with self.assertRaises(ValueError):
                runner.validate_intent(self.intent(index), config, runner.canonical_hash({}), index)
        for key, value in (("item_id", "item-93"), ("arm", "baseline"), ("max_tokens", 1025)):
            intent = self.intent(2)
            intent[key] = value
            with self.assertRaises(ValueError):
                runner.validate_intent(intent, config, runner.canonical_hash({}), 2)

    def test_new_authorization_requires_exact_plan_and_rejects_historical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, root = Path(directory), Path(directory) / "new"
            root.mkdir()
            old = parent / "original"
            old.mkdir()
            (old / "authorization-source.txt").write_text("Original authorization")
            plan = {"fixture": True}
            runner.save(root / "continuation-plan.json", plan)
            approved = root / "fixture-approval.txt"
            approved.write_text("Original authorization\n")
            manifest = {"arm": "candidate", "original_root": str(old), "files": {}}
            with patch.object(runner, "load", return_value=(root, manifest)):
                with self.assertRaises(PermissionError):
                    runner.authorize(root, approved, "f" * 64)
                with self.assertRaises(PermissionError):
                    runner.authorize(root, approved, runner.canonical_hash(plan))
                self.assertFalse((root / "generation-budget.sqlite3").exists())
                self.assertFalse((root / "authorization-source.txt").exists())
                approved.write_text("New explicit fixture approval")
                result = runner.authorize(root, approved, runner.canonical_hash(plan))
                self.assertTrue(result["authorized"])
                with self.assertRaises(FileExistsError):
                    runner.authorize(root, approved, runner.canonical_hash(plan))
            self.assertEqual((old / "authorization-source.txt").read_text(), "Original authorization")

    def test_exact_new_capacity_cannot_extend_or_borrow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"arm": "candidate", "original_root": str(root / "old"), "files": {}}
            self.authorize_fixture(root, manifest)
            with runner.ledger(root, manifest) as book:
                for index in range(1, 36):
                    rid = f"generation-{index}"
                    book.reserve(rid, kind="probe", sequences=1, max_generated_tokens=16 if index == 1 else 1024,
                                 timeout_seconds=120, max_billed_units=0)
                    book.mark_dispatched(rid)
                    book.finish(rid, status="completed", completed_sequences=1, failed_sequences=0,
                                generated_tokens=0, elapsed_seconds=0, billed_units=0)
                self.assertEqual(book.summary()["reserved"]["generated_tokens"], 34832)
                with self.assertRaises(BudgetExhausted):
                    book.reserve("generation-36", kind="probe", sequences=1, max_generated_tokens=1,
                                 timeout_seconds=1, max_billed_units=0)

    def _run_fake_driver(self, fault=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            manifest = {"arm": "candidate", "original_root": str(root / "old"), "files": {}}
            runner.save(root / "baseline-config.json", config)
            self.authorize_fixture(root, manifest)
            events, generated, complete, raced = [], set(), {"value": False}, {"value": False}
            target = root / "execution/candidate-1"
            def fake_events(path):
                previous_snapshot, inject_race = copy.deepcopy(events), False
                if not (target / "load-intent.json").exists():
                    runner.save(target / "load-intent.json", self.intent(0))
                if (target / "load-ack.json").exists():
                    for index in range(1, 36):
                        predecessor = target / ("load-ack.json" if index == 1 else f"generation-{index-1}-ack.json")
                        intent = target / f"generation-{index}-intent.json"
                        if predecessor.exists() and not intent.exists():
                            runner.save(intent, self.intent(index))
                        if (target / f"generation-{index}-ack.json").exists() and index not in generated:
                            if index == 5 and not raced["value"]:
                                inject_race, raced["value"] = True, True
                            generated.add(index)
                            events.append({"type": "warmup-sample" if index == 1 else "task-sample", "data": {
                                "item_id": "__warmup__" if index == 1 else f"item-{index+92}",
                                "request_id": f"generation-{index}", "warmup": index == 1,
                                "raw_output": "fixture response", "generated_tokens": 1,
                                "response_latency_ms": 1, "finish_reason": "stop", "execution_error": None,
                                "local_webgpu_inference_observed": True, "webgpu_submitted_batches": 1}})
                if len(generated) == 35 and not complete["value"]:
                    complete["value"] = True
                    terminal = {"status": "completed",
                        "cleanup": {key: True for key in runner.CLEANUP_KEYS}, "driver": {"chrome_pid": 123456789},
                        "configuration_sha256": config["configurationSha256"],
                        "source_checkpoint_sha256": config["sourceCheckpointSha256"],
                        "converted_artifact_sha256": config["convertedArtifactSha256"]}
                    if fault == "malformed_driver":
                        terminal["driver"] = "invalid fixture"
                    if fault == "malformed_outputs":
                        terminal["outputs"] = None
                    runner.save(target / "result.json", terminal)
                return previous_snapshot if inject_race else copy.deepcopy(events)
            class Process:
                def poll(self):
                    if complete["value"] and fault == "late_runtime_failure" and not complete.get("late_fault"):
                        complete["late_fault"] = True
                        events.append({"type": "driver-error", "data": {"error": "late fixture runtime failure"}})
                    return 0 if complete["value"] else None
            with patch.object(runner, "load", return_value=(root, manifest)), \
                    patch.object(runner, "_collector", return_value=None), \
                    patch.object(runner, "verify_original_process_closure", return_value={"verified": True}), \
                    patch.object(runner, "sample_resources", return_value={"host": {"hostMemoryPressureLevel": 1, "hostThermalState": 0}}), \
                    patch.object(runner.subprocess, "Popen", return_value=Process()) as spawn, \
                    patch.object(runner, "read_events", side_effect=fake_events), \
                    patch.object(gate, "read_events", side_effect=fake_events), \
                    patch.object(runner, "assert_predecessor", wraps=gate.assert_predecessor) as predecessor, \
                    patch.object(runner, "verify_loaded"), patch.object(runner, "deadline_watchdog"), \
                    patch.object(runner, "_stop_process", return_value={"controller_exit_verified": True}), \
                    patch.object(runner, "alive", return_value=False), \
                    patch.object(runner.time, "sleep"), patch("builtins.print"):
                result = runner.execute(root)
                self.assertEqual(result["status"], "stopped" if fault else "completed", result["error"])
                if fault:
                    self.assertIsNotNone(result["error"])
                    self.assertTrue((target / "supervisor-result.json").exists())
                    self.assertEqual(runner._original_ledger(root / "generation-budget.sqlite3")[0]["cancelled"], 1)
                self.assertEqual(result["generation_dispatches"], 35)
                self.assertEqual(result["scored_dispatches"], 34)
                self.assertEqual(result["warmup_dispatches"], 1)
                self.assertEqual([o["item_id"] for o in result["outputs"]], [f"item-{i}" for i in range(94, 128)])
                self.assertEqual(result["generation_budget"]["observed"]["generated_tokens"], 35)
                self.assertEqual(spawn.call_count, 1)
                self.assertTrue(raced["value"])
                self.assertEqual(predecessor.call_count, 35)
                self.assertEqual(len(list(target.glob('*-ack.json'))), 36)
                self.assertEqual(len(runner._original_ledger(root / "generation-budget.sqlite3")[1]), 35)
                with self.assertRaises(ValueError):
                    runner.execute(root)
                self.assertEqual(spawn.call_count, 1)

    def test_complete_35_dispatch_supervised_loop_with_fake_driver(self):
        self._run_fake_driver()

    def test_late_runtime_failure_cannot_be_reported_as_success(self):
        self._run_fake_driver("late_runtime_failure")

    def test_malformed_terminal_closes_ledger_and_saves_failure(self):
        for fault in ("malformed_driver", "malformed_outputs"):
            with self.subTest(fault=fault):
                self._run_fake_driver(fault)

    def make_prepared_fixture(self, parent):
        """Synthetic old run and pending plan; no model, grading, or authorization."""
        old, root = parent / "old", parent / "new"
        old.mkdir(); root.mkdir()
        items = [{"benchmark": "ifeval", "split": "test", "item_id": f"item-{i}", "task_text": "fixture",
                  "target": {"instruction_id_list": ["fixture"] * (2 if i < 34 or 94 <= i < 105 else 1)}}
                 for i in range(128)]
        runner.save(old / "evaluation-items.json", items)
        freeze = {"evaluation_items_sha256": runner.canonical_hash(items)}
        runner.save(old / "evaluation-freeze.json", freeze)
        config = {**self.config(), "study": "ifeval-browser-v1", "evaluationStage": "heldout-continuation",
            "originalScheduleOffset": 94, "originalScheduleLength": 128,
            "settings": runner.SETTINGS, "totalLimitMs": 5400000, "warmupPrompt": "Reply with OK.",
            "model": {"fixture": True}, "assets": [],
            "items": [{"item_id": i["item_id"], "prompt": i["task_text"], "rendered_input_tokens": 40} for i in items[94:]]}
        old_config = {**config, "items": [{"item_id": i["item_id"], "prompt": i["task_text"], "rendered_input_tokens": 40} for i in items]}
        runner.save(old / "candidate/baseline-config.json", old_config)
        runner.save(old / "candidate/manifest.json", {"fixture": True})
        config["configurationSha256"] = runner.canonical_hash({k:v for k,v in config.items() if k != "configurationSha256"})
        runner.save(root / "baseline-config.json", config)
        runner.save(root / "remaining-items.json", items[94:])
        runner.save(root / "merge-plan.json", {"fixture": True})
        cleanup = {k: True for k in runner.CLEANUP_KEYS}
        cleanup.update(unload_confirmed=False, retained_owned_pids_alive=[])
        runner.save(old / "candidate/execution/candidate-1/supervisor-result.json", {"status": "stopped",
            "scored_dispatches": 94, "generation_dispatches": 95, "warmup_dispatches": 1,
            "outputs": [{"item_id": i["item_id"]} for i in items[:94]], "cleanup": cleanup})
        runner.save(old / "baseline/execution/baseline-1/supervisor-result.json", {"cleanup": cleanup})
        runner.save(old / "candidate/candidate/graded.json", {"records": [
            {"item_id": i["item_id"], "status": "ok" if n < 94 else "missing", "attempts": [{}] if n < 94 else []}
            for n,i in enumerate(items)]})
        with BudgetLedger(old / "candidate/generation-budget.sqlite3", run_id="old-fixture", limits={**runner.GEN_LIMITS,
                "model_probes": 95, "requested_sequences": 95, "generated_tokens": 100000}, binding={}, create=True) as book:
            for index in range(1,96):
                rid = f"generation-{index}"
                book.reserve(rid, kind="probe", sequences=1, max_generated_tokens=16 if index==1 else 1024,
                             timeout_seconds=120, max_billed_units=0)
                book.mark_dispatched(rid)
                book.finish(rid, status="completed", completed_sequences=1, failed_sequences=0,
                            generated_tokens=0, elapsed_seconds=0, billed_units=0)
            book.cancel(reason="fixture_stopped")
        manifest = {"arm": "candidate", "experiment_id": runner.EXPERIMENT_ID, "limits": runner.LIMITS,
            "settings": runner.SETTINGS, "policy": runner.POLICY, "grader": {"fixture": True},
            "checkpoint": {"sha256": runner.CANDIDATE_SOURCE}, "artifact_sha256": runner.CANDIDATE_ARTIFACT,
            "configuration_sha256": config["configurationSha256"], "original_root": str(old),
            "original_evaluation_freeze_sha256": runner.canonical_hash(freeze),
            "original_candidate_manifest_sha256": runner.canonical_hash({"fixture": True}),
            "original_candidate_configuration_file_sha256": runner.file_sha256(old / "candidate/baseline-config.json"),
            "items_path": str(root / "remaining-items.json"), "items_sha256": runner.canonical_hash(items[94:]),
            "official_ifeval_root": str(parent / "official-fixture"), "official_assets": {"fixture": True},
            "files": {str(p.resolve()): runner.file_sha256(p) for p in old.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))}}
        manifest["files"].update({str(p.resolve()): runner.file_sha256(p) for p in [root / "remaining-items.json",root / "baseline-config.json"]})
        runner.save(root / "manifest.json", manifest)
        plan = {"kind": "frozen_continuation_preparation", "execution_authorized": False, "authorization_status": "pending",
            "limits": runner.LIMITS, "manifest_sha256": runner.canonical_hash(manifest), "task_order": [i["item_id"] for i in items[94:]],
            "merge_plan_sha256": runner.canonical_hash({"fixture": True}), "original_schedule_offset": 94, "original_schedule_length": 128,
            "original_evaluation_freeze_sha256": runner.canonical_hash(freeze)}
        runner.save(root / "continuation-plan.json", plan)
        runner.save(root / "ready.json", {"continuation_plan_sha256": runner.canonical_hash(plan)})
        return root, old, runner.canonical_hash(freeze)

    def test_prepared_load_reads_old_ledger_without_budgetledger_and_blocks_changed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root, old, freeze = self.make_prepared_fixture(Path(directory))
            old_ledger = old / "candidate/generation-budget.sqlite3"
            before = runner.file_sha256(old_ledger)
            with patch.object(runner, "ORIGINAL_FREEZE", freeze), \
                    patch.object(runner, "grader_identity", return_value={"fixture": True}), \
                    patch.object(runner, "verify_ifeval_assets", return_value={"fixture": True}), \
                    patch.object(runner, "BudgetLedger") as budget:
                runner.load(root)
                budget.assert_not_called()
                self.assertEqual(runner.file_sha256(old_ledger), before)
                path = old / "candidate/candidate/graded.json"
                changed = json.loads(path.read_text())
                changed["records"][94]["status"] = "ok"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.load(root)
                budget.assert_not_called()
            self.assertFalse((root / "generation-budget.sqlite3").exists())
            self.assertFalse((root / "execution").exists())

    def test_corrupt_final_events_keep_validated_evidence_and_ambiguous_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"arm": "candidate", "original_root": str(root / "old"), "files": {}}
            self.authorize_fixture(root, manifest)
            warm = {"request_id": "generation-1", "item_id": "__warmup__", "warmup": True,
                    "raw_output": "OK", "generated_tokens": 1, "response_latency_ms": 1,
                    "finish_reason": "stop", "local_webgpu_inference_observed": True, "webgpu_submitted_batches": 1}
            events = [{"type": "warmup-sample", "data": warm}]
            with runner.ledger(root, manifest) as book:
                for index in (1, 2):
                    book.reserve(f"generation-{index}", kind="probe", sequences=1,
                        max_generated_tokens=16 if index == 1 else 1024, timeout_seconds=120, max_billed_units=0)
                    book.mark_dispatched(f"generation-{index}")
                runner.reconcile(book, warm, 1, self.config())
                seen = {1: runner.canonical_hash(warm)}
                with patch.object(runner, "read_events", side_effect=ValueError("malformed JSONL fixture")):
                    recovered_warm, scored, errors = runner.recover_observations(root, {}, events, self.config(), book, seen)
                self.assertEqual(recovered_warm, [warm])
                self.assertEqual(scored[0]["item_id"], "item-94")
                self.assertEqual(scored[0]["dispatch_outcome"], "acknowledged_or_ambiguous")
                self.assertIsNone(scored[0]["raw_output"])
                self.assertIn("final_event_read_failed:ValueError", errors)
                unknown = {"type": "task-sample", "data": {**warm, "item_id": "item-93", "request_id": "generation-2", "warmup": False}}
                with patch.object(runner, "read_events", return_value=events + [unknown]):
                    _, scored, errors = runner.recover_observations(root, {}, events, self.config(), book, seen)
                self.assertEqual(scored[0]["item_id"], "item-94")
                self.assertIsNone(scored[0]["raw_output"])
                self.assertIn("conflicting_or_unscheduled_final_completion", errors)

    def test_old_process_start_identity_distinguishes_recycled_pids(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory)
            for arm in ("baseline", "candidate"):
                target = old / arm / "execution" / (arm + "-1")
                target.mkdir(parents=True)
                (target / "resources.jsonl").write_text(json.dumps({"owned_processes": [{"pid": 111, "startId": "100"}]}) + "\n")
                runner.save(target / "result.json", {"driver": {"chrome_pid": 111}})
            with patch.object(runner, "alive", return_value=True):
                result = runner.verify_original_process_closure({"original_root": str(old)},
                    SimpleNamespace(usage=lambda _: {"startId": "200"}))
                self.assertTrue(result["verified"])
                with self.assertRaises(ValueError):
                    runner.verify_original_process_closure({"original_root": str(old)},
                        SimpleNamespace(usage=lambda _: {"startId": "100"}))


if __name__ == "__main__":
    unittest.main()
