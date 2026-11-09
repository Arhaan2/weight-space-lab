"""Focused held-out scheduling checks; no browser, model, or official grader calls."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from weight_space_lab import ifeval_heldout_runner as runner
from weight_space_lab.editor_budget import BudgetExhausted
from weight_space_lab.grading import _known_answer_item


class HeldoutRunnerTests(unittest.TestCase):
    def config(self, arm="baseline"):
        return {"configurationSha256": "a"*64, "convertedArtifactSha256": "b"*64,
            "arm": arm, "runId": "ifeval-browser-v1-heldout-"+arm,
            "items": [{"item_id": f"item-{i}"} for i in range(128)]}

    def intent(self, index, arm="baseline"):
        config = self.config(arm)
        return {"kind": "load" if index == 0 else "generation", "arm": arm,
            "request_id": "load-1" if index == 0 else f"generation-{index}",
            "attempt_id": arm+"-1", "code_version_sha256": "c"*64,
            "configuration_sha256": config["configurationSha256"],
            "artifact_sha256": config["convertedArtifactSha256"], "run_id": config["runId"],
            "warmup": index == 1, "sequence_index": index-1,
            "max_tokens": 16 if index == 1 else 1024,
            "item_id": "__warmup__" if index == 1 else f"item-{index-2}"}

    def test_exact_schedule_for_both_arms(self):
        for arm in ("baseline", "candidate"):
            for index in range(130):
                runner.validate_intent(self.intent(index, arm), self.config(arm), "c"*64, index)
            for index in (-1, 130):
                with self.assertRaises(ValueError):
                    runner.validate_intent(self.intent(index, arm), self.config(arm), "c"*64, index)
            for key, value in (("max_tokens", 1024), ("warmup", False), ("sequence_index", 1)):
                intent = self.intent(1, arm)
                intent[key] = value
                with self.assertRaises(ValueError):
                    runner.validate_intent(intent, self.config(arm), "c"*64, 1)
            intent = self.intent(129, arm)
            intent["item_id"] = "item-0"
            with self.assertRaises(ValueError):
                runner.validate_intent(intent, self.config(arm), "c"*64, 129)

    def test_full_two_arm_capacity_is_exact_and_nontransferable(self):
        with tempfile.TemporaryDirectory() as directory:
            for arm in ("baseline", "candidate"):
                root = Path(directory)/arm
                with runner.ledger(root, {"arm": arm}, create=True) as book:
                    for index in range(1, 130):
                        request = f"generation-{index}"
                        book.reserve(request, kind="probe", sequences=1,
                            max_generated_tokens=16 if index == 1 else 1024,
                            timeout_seconds=120, max_billed_units=0)
                        book.mark_dispatched(request)
                        book.finish(request, status="completed", completed_sequences=1, failed_sequences=0,
                            generated_tokens=0, elapsed_seconds=0.1, billed_units=0)
                    self.assertEqual(book.summary()["reserved"]["generated_tokens"], 131088)
                    self.assertEqual(book.summary()["charged"]["requested_sequences"], 129)
                    with self.assertRaises(BudgetExhausted):
                        book.reserve("generation-130", kind="probe", sequences=1,
                            max_generated_tokens=1, timeout_seconds=1, max_billed_units=0)
                with self.assertRaises(FileExistsError):
                    runner.ledger(root, {"arm": arm}, create=True)

    def test_test_identity_missing_and_attempted_failure_remain_distinct(self):
        item = _known_answer_item("ifeval", {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]})
        item["split"] = "test"
        before = copy.deepcopy(item)
        args = {"run_id": "fixture", "checkpoint": {"model_id": "fixture", "revision": "a"*40,
            "sha256": "b"*64}, "configuration_sha256": "c"*64, "evidence_kind": "fixture"}
        record, metric = runner.adapt_browser_output(item, None, **args)
        self.assertEqual((record["status"], record["attempts"], metric["execution_status"]), ("missing", [], "unrun"))
        self.assertEqual((record["split"], record["stage"]), ("test", "full"))
        record, metric = runner.adapt_browser_output(item, {"item_id": item["item_id"],
            "raw_output": None, "finish_reason": "error", "execution_error": "interrupted"}, **args)
        self.assertEqual(record["status"], "error")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(metric["execution_status"], "error")
        self.assertEqual(item, before)
        raw = '<think>\n\n</think>\n\n{"ok":true}'
        record, metric = runner.adapt_browser_output(item, {"item_id": item["item_id"],
            "raw_output": raw, "finish_reason": "length", "generated_tokens": 1020,
            "input_tokens": 28, "response_latency_ms": 20000}, **args)
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(record["status"], "truncated")
        self.assertEqual(record["token_counts"]["output"], 1020)

    def test_deadline_starts_before_first_load_and_bounds_second_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            runner.save(parent/"evaluation-freeze.json", {"fixture": True})
            base, candidate = parent/"baseline", parent/"candidate"
            base.mkdir()
            candidate.mkdir()
            grant = runner.grant_load(base, {"arm": "baseline"}, now=1000, mono=100)
            self.assertEqual(grant["deadline_monotonic"], 5500)
            self.assertTrue((parent/"aggregate-clock.json").exists())
            with self.assertRaises(ValueError):
                runner.grant_load(base, {"arm": "baseline"}, now=1001, mono=101)
            with self.assertRaises(ValueError):
                runner.grant_load(candidate, {"arm": "candidate"}, now=6000, mono=5100)
            cleanup = {key: True for key in runner.CLEANUP_KEYS}
            cleanup["retained_owned_pids_alive"] = []
            runner.save(base/"execution/baseline-1/supervisor-result.json", {"status": "completed",
                "error": None, "engine_load_attempts": 1, "generation_dispatches": 129, "cleanup": cleanup})
            runner.save(base/"execution/baseline-1/result.json", {"driver": {"chrome_pid": None}})
            grant = runner.grant_load(candidate, {"arm": "candidate"}, now=6500, mono=5600)
            self.assertEqual(grant["deadline_monotonic"], 10900)
            self.assertEqual(grant["deadline_epoch_ms"], 11800000)
            self.assertLess(grant["deadline_monotonic"]-grant["started_monotonic"], 5400)
            with self.assertRaises(FileExistsError):
                runner.grant_load(candidate, {"arm": "candidate"}, now=6501, mono=5601)

    def test_complete_supervised_loop_with_fake_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, arm = Path(directory), "baseline"
            root = parent/arm
            root.mkdir()
            config = self.config()
            config["sourceCheckpointSha256"] = "d"*64
            manifest = {"arm": arm, "files": {}, "fixture": True}
            runner.save(root/"baseline-config.json", config)
            runner.save(parent/"evaluation-freeze.json", {"fixture": True})
            with runner.ledger(root, manifest, create=True):
                pass
            events, generated = [], set()
            target = root/"execution/baseline-1"
            complete = {"value": False}

            def read_events(path):
                if not (target/"load-intent.json").exists():
                    intent = self.intent(0)
                    intent["code_version_sha256"] = runner.canonical_hash({})
                    runner.save(target/"load-intent.json", intent)
                if (target/"load-ack.json").exists():
                    for index in range(1, 130):
                        prior = target/("load-ack.json" if index == 1 else f"generation-{index-1}-ack.json")
                        intent_path = target/f"generation-{index}-intent.json"
                        if prior.exists() and not intent_path.exists():
                            intent = self.intent(index)
                            intent["code_version_sha256"] = runner.canonical_hash({})
                            runner.save(intent_path, intent)
                        if (target/f"generation-{index}-ack.json").exists() and index not in generated:
                            generated.add(index)
                            events.append({"type": "warmup-sample" if index == 1 else "task-sample", "data": {
                                "item_id": "__warmup__" if index == 1 else f"item-{index-2}",
                                "request_id": f"generation-{index}", "warmup": index == 1,
                                "generated_tokens": 1, "response_latency_ms": 1, "finish_reason": "stop",
                                "local_webgpu_inference_observed": True, "webgpu_submitted_batches": 1}})
                if len(generated) == 129 and not complete["value"]:
                    complete["value"] = True
                    cleanup = {key: True for key in runner.CLEANUP_KEYS}
                    runner.save(target/"result.json", {"status": "completed", "cleanup": cleanup,
                        "driver": {"chrome_pid": 123456789},
                        "configuration_sha256": config["configurationSha256"],
                        "source_checkpoint_sha256": config["sourceCheckpointSha256"],
                        "converted_artifact_sha256": config["convertedArtifactSha256"]})
                return events

            class Process:
                def poll(self):
                    return 0 if complete["value"] else None

            with patch.object(runner, "load", return_value=(root, manifest)), \
                    patch.object(runner, "_collector", return_value=None), \
                    patch.object(runner, "sample_resources", return_value={"host": {
                        "hostMemoryPressureLevel": 1, "hostThermalState": 0}}), \
                    patch.object(runner.subprocess, "Popen", return_value=Process()), \
                    patch.object(runner, "read_events", side_effect=read_events), \
                    patch.object(runner, "verify_loaded"), patch.object(runner, "deadline_watchdog"), \
                    patch.object(runner, "_stop_process", return_value={"controller_exit_verified": True}), \
                    patch.object(runner, "alive", return_value=False), \
                    patch.object(runner.time, "sleep"), patch("builtins.print"):
                result = runner.execute(root)
                self.assertEqual(result["status"], "completed", result["error"])
                self.assertEqual(result["generation_dispatches"], 129)
                self.assertEqual(result["scored_dispatches"], 128)
                self.assertEqual(result["warmup_dispatches"], 1)
                self.assertEqual(len(result["outputs"]), 128)
                self.assertEqual(result["generation_budget"]["charged"]["observed_generated_tokens"], 129)
                with self.assertRaises(ValueError):
                    runner.execute(root)

    def test_candidate_requires_complete_verified_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            runner.save(parent/"aggregate-clock.json", {})
            cleanup = {key: True for key in runner.CLEANUP_KEYS}
            cleanup["retained_owned_pids_alive"] = []
            cleanup["unload_confirmed"] = False
            runner.save(parent/"baseline/execution/baseline-1/supervisor-result.json", {"status": "completed",
                "error": None, "engine_load_attempts": 1, "generation_dispatches": 129, "cleanup": cleanup})
            with self.assertRaises(ValueError):
                runner.verify_execution_order(parent/"candidate", {"arm": "candidate"})


if __name__ == "__main__":
    unittest.main()
