import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_space_lab.inventory import read_json
from weight_space_lab.native_runner import (
    _Journal, _Process, _inputs, _lock, run_schedule, validate_fixture_spec,
)
from weight_space_lab.protocol import build_benchmark_schedule, canonical_hash
from weight_space_lab.records import validate_run_records


def fixture_inputs(*, behavior="success", delay=0.0, retries=1, items=2):
    """Synthetic task and checkpoint identities, never a scientific evaluation."""
    config = {
        "schema_version": 1, "kind": "native_execution", "run_id": "fixture-run",
        "evidence_kind": "fixture", "profile": "native_development_smoke", "stage": "smoke",
        "checkpoint": {"model_id": "synthetic-fixture", "revision": canonical_hash("fixture revision"),
                       "sha256": canonical_hash("fixture checkpoint")},
        "retry_policy": {"mode": "none" if retries == 1 else "retry_errors", "max_attempts": retries},
        "execution_limits": {"request_timeout_seconds": 3.0, "startup_timeout_seconds": 5.0,
                             "max_duration_seconds": 30.0, "max_sequences": items * retries},
        "backend": {"name": "fixture", "behavior": behavior, "delay_seconds": delay},
        "implementation_sha256": canonical_hash("fixture runner implementation"),
    }
    tasks = [{"benchmark": "synthetic", "item_id": f"fixture-{i}", "split": "development",
              "dataset_revision": canonical_hash("fixture tasks"),
              "messages": [{"role": "user", "content": f"Synthetic input {i}"}]}
             for i in range(items)]
    schedule = build_benchmark_schedule("synthetic", [task["item_id"] for task in tasks], [17])
    return config, tasks, schedule


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "run"

    def run_fixture(self, inputs=None, **kwargs):
        return run_schedule(*(inputs or fixture_inputs()), self.output, **kwargs)

    def records(self):
        return validate_run_records(read_json(self.output / "records.json"))

    def events(self):
        return [read_json(path)["event"] for path in sorted((self.output / "events").glob("*.json"))]

    def test_real_process_execution_materializes_phase_one_records_and_metadata(self):
        summary = self.run_fixture()
        records = self.records()
        self.assertEqual(summary["statuses"]["ok"], 2)
        self.assertEqual(summary["attempted_sequences"], 2)
        self.assertFalse(summary["scientific_scores"])
        self.assertFalse(summary["experiment_executed"])
        self.assertEqual([row["item_id"] for row in records], ["fixture-0", "fixture-1"])
        self.assertTrue(all(row["evidence_kind"] == "fixture" for row in records))
        self.assertTrue(all(row["raw_output"] == "SYNTHETIC FIXTURE OUTPUT" for row in records))
        finished = [event for event in self.events() if event["kind"] == "attempt_finished"]
        self.assertTrue(all(event["metadata"]["worker"]["synthetic"] for event in finished))
        self.assertTrue(all(event["elapsed_seconds"] >= event["outcome"]["duration_seconds"] for event in finished))
        manifest = read_json(self.output / "manifest.json")
        self.assertEqual(records[0]["configuration_sha256"], canonical_hash({key: manifest[key] for key in ("configuration", "tasks", "schedule")}))

    def test_resume_never_repeats_success_and_recreates_missing_export(self):
        self.run_fixture()
        originals = {path.name: path.read_bytes() for path in (self.output / "events").iterdir()}
        records = self.records()
        (self.output / "records.json").unlink()
        with patch("weight_space_lab.native_runner._Process", side_effect=AssertionError("unexpected new inference")):
            summary = self.run_fixture(resume=True)
        self.assertEqual(self.records(), records)
        self.assertEqual(summary["attempted_sequences"], 2)
        self.assertEqual({path.name: path.read_bytes() for path in (self.output / "events").iterdir()}, originals)

    def test_resume_rejects_changed_config_tasks_schedule_and_implementation(self):
        inputs = fixture_inputs()
        self.run_fixture(inputs)
        changes = []
        for field, value in (("run_id", "another-run"), ("checkpoint", dict(inputs[0]["checkpoint"], sha256=canonical_hash("changed weights"))),
                             ("implementation_sha256", canonical_hash("changed implementation")),
                             ("retry_policy", {"mode": "retry_errors", "max_attempts": 2})):
            changed = copy.deepcopy(inputs)
            changed[0][field] = value
            changes.append(changed)
        changed = copy.deepcopy(inputs)
        changed[0]["execution_limits"]["request_timeout_seconds"] += 1
        changes.append(changed)
        changed = copy.deepcopy(inputs)
        changed[1][0]["messages"][0]["content"] = "changed task"
        changes.append(changed)
        changed = copy.deepcopy(inputs)
        changed[2].reverse()
        changes.append(changed)
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "incompatible resume"):
                self.run_fixture(changed, resume=True)
        self.assertEqual(len(self.events()), 4)

    def test_explicit_resume_required_and_nonexistent_resume_rejected(self):
        with self.assertRaisesRegex(ValueError, "without a frozen manifest"):
            self.run_fixture(resume=True)
        self.run_fixture()
        with self.assertRaisesRegex(ValueError, "explicit resume"):
            self.run_fixture()

    def test_retry_policy_retains_error_before_success(self):
        inputs = fixture_inputs(behavior="error_once", retries=2, items=1)
        summary = self.run_fixture(inputs)
        record = self.records()[0]
        self.assertEqual([a["status"] for a in record["attempts"]], ["error", "ok"])
        self.assertEqual([a["attempt_index"] for a in record["attempts"]], [0, 1])
        self.assertEqual(record["raw_output"], record["attempts"][-1]["raw_output"])
        self.assertEqual(summary["attempted_sequences"], 2)
        self.assertEqual([e["kind"] for e in self.events()], ["attempt_started", "attempt_finished"] * 2)

    def test_exhausted_errors_do_not_retry_on_resume(self):
        inputs = fixture_inputs(behavior="error", retries=2, items=1)
        self.run_fixture(inputs)
        before = self.records()
        with patch("weight_space_lab.native_runner._Process", side_effect=AssertionError("retry budget exceeded")):
            self.run_fixture(inputs, resume=True)
        self.assertEqual(self.records(), before)
        self.assertEqual(len(before[0]["attempts"]), 2)

    def test_timeout_terminates_real_child_and_missing_rows_remain_explicit(self):
        inputs = fixture_inputs(behavior="timeout", delay=5.0)
        inputs[0]["execution_limits"].update(request_timeout_seconds=0.1, max_sequences=1)
        started = time.monotonic()
        summary = self.run_fixture(inputs)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(summary["statuses"], {"ok": 0, "truncated": 0, "error": 0, "timeout": 1, "missing": 1})
        first, second = self.records()
        self.assertEqual(first["error"]["code"], "request_timeout")
        self.assertIsNone(first["token_counts"]["output"])
        self.assertEqual(second["attempts"], [])
        self.assertIsNone(second["duration_seconds"])
        self.assertIsNone(second["raw_output"])
        self.run_fixture(inputs, resume=True)
        self.assertEqual(len(self.events()), 2)

    def test_worker_process_crash_is_a_failed_attempt(self):
        summary = self.run_fixture(fixture_inputs(behavior="crash", items=1))
        self.assertEqual(summary["statuses"]["error"], 1)
        self.assertEqual(self.records()[0]["error"]["code"], "worker_exited")

    def test_startup_timeout_is_bounded_and_preserved(self):
        inputs = fixture_inputs(items=1)
        inputs[0]["execution_limits"]["startup_timeout_seconds"] = 0.00001
        summary = self.run_fixture(inputs)
        self.assertEqual(summary["statuses"]["timeout"], 1)
        self.assertEqual(self.records()[0]["attempts"][0]["status"], "timeout")
        self.assertEqual(self.records()[0]["error"]["code"], "startup_timeout")

    def test_unfinished_attempt_is_preserved_and_only_explicit_retry_runs(self):
        inputs = fixture_inputs(retries=2, items=1)
        manifest, records, _ = _inputs(*inputs)
        with _lock(self.output):
            journal = _Journal(self.output, manifest, records, resume=False)
            journal.append("attempt_started", 0, 0)
        summary = self.run_fixture(inputs, resume=True)
        record = self.records()[0]
        self.assertEqual([attempt["status"] for attempt in record["attempts"]], ["error", "ok"])
        self.assertEqual(record["attempts"][0]["error"]["code"], "interrupted")
        self.assertIsNone(record["attempts"][0]["duration_seconds"])
        self.assertIsNone(record["attempts"][0]["token_counts"]["output"])
        self.assertGreaterEqual(summary["accounted_wall_seconds"], 8.0)

    def test_interruption_without_retry_budget_cannot_create_another_attempt(self):
        inputs = fixture_inputs(items=1)
        manifest, records, _ = _inputs(*inputs)
        with _lock(self.output):
            journal = _Journal(self.output, manifest, records, resume=False)
            journal.append("attempt_started", 0, 0)
        with patch("weight_space_lab.native_runner._Process", side_effect=AssertionError("undeclared retry")):
            self.run_fixture(inputs, resume=True)
        self.assertEqual(len(self.records()[0]["attempts"]), 1)
        self.assertEqual(self.records()[0]["status"], "error")

    def test_hard_parent_interruption_recovers_durable_start(self):
        inputs = fixture_inputs(behavior="timeout", delay=10, retries=2, items=1)
        inputs[0]["execution_limits"]["max_sequences"] = 1
        config_path = Path(self.temporary.name) / "inputs.json"
        config_path.write_text(json.dumps(inputs))
        script_path = Path(self.temporary.name) / "interrupt.py"
        worker_pid_path = Path(self.temporary.name) / "worker.pid"
        script_path.write_text(
            "import json,sys\nfrom pathlib import Path\nimport weight_space_lab.native_runner as runner\n"
            "class ObservedProcess(runner._Process):\n"
            "    def __init__(self, *args):\n"
            "        super().__init__(*args)\n"
            "        Path(sys.argv[3]).write_text(str(self.process.pid))\n"
            "if __name__ == '__main__':\n"
            "    runner._Process = ObservedProcess\n"
            "    runner.run_schedule(*json.load(open(sys.argv[1])),sys.argv[2])\n")
        process = subprocess.Popen([sys.executable, str(script_path), str(config_path), str(self.output), str(worker_pid_path)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
        try:
            deadline = time.monotonic() + 10
            while not (self.output / "events/00000000.json").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail("fixture parent exited before durable attempt start")
                time.sleep(0.02)
            self.assertTrue((self.output / "events/00000000.json").exists())
            while not worker_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(worker_pid_path.exists())
            worker_pid = int(worker_pid_path.read_text())
            # Let the worker enter generation, then simulate uncatchable parent loss.
            time.sleep(0.4)
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
            worker_deadline = time.monotonic() + 5
            while time.monotonic() < worker_deadline:
                try:
                    os.kill(worker_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail("worker survived parent loss")
            self.run_fixture(inputs, resume=True)
            record = self.records()[0]
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["error"]["code"], "interrupted")
            self.assertEqual(len(record["attempts"]), 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def test_corrupt_or_deleted_committed_events_are_rejected(self):
        self.run_fixture()
        final_path = self.output / "events/00000003.json"
        original = final_path.read_bytes()
        envelope = json.loads(original)
        envelope["event"]["outcome"]["raw_output"] = "altered output"
        final_path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_fixture(resume=True)
        final_path.write_bytes(original)
        final_path.unlink()
        with self.assertRaisesRegex(ValueError, "lost committed progress"):
            self.run_fixture(resume=True)
        final_path.write_bytes(original)
        (self.output / "events/00000001.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            self.run_fixture(resume=True)

    def test_completion_committed_before_export_crash_recovers_without_inference(self):
        original_export = _Journal.export
        def interrupted_export(journal):
            if journal.events and journal.events[-1]["event"]["kind"] == "attempt_finished":
                raise OSError("synthetic export interruption")
            original_export(journal)
        inputs = fixture_inputs(items=1)
        with patch.object(_Journal, "export", interrupted_export), self.assertRaises(OSError):
            self.run_fixture(inputs)
        self.assertTrue((self.output / "events/00000001.json").exists())
        self.assertEqual(self.records()[0]["status"], "missing")
        with patch("weight_space_lab.native_runner._Process", side_effect=AssertionError("repeated committed generation")):
            self.run_fixture(inputs, resume=True)
        self.assertEqual(self.records()[0]["status"], "ok")
        self.assertEqual(len(self.events()), 2)

    def test_unpublished_partial_event_does_not_become_an_attempt(self):
        self.run_fixture()
        (self.output / "events/.pending-interrupted-write").write_bytes(b'{"event":')
        self.run_fixture(resume=True)
        self.assertEqual(len(self.events()), 4)
        self.assertEqual(len(self.records()[0]["attempts"]), 1)

    def test_writer_lock_covers_recovery_and_execution(self):
        self.run_fixture()
        with _lock(self.output), self.assertRaisesRegex(ValueError, "already locked"):
            self.run_fixture(resume=True)

    def test_fixture_cannot_claim_real_model_evidence(self):
        inputs = fixture_inputs()
        inputs[0]["evidence_kind"] = "real_model"
        with self.assertRaisesRegex(ValueError, "evidence kind disagree"):
            self.run_fixture(inputs)
        self.assertFalse(self.output.exists())

    def test_unknown_fixture_settings_and_incomplete_task_schedule_rejected(self):
        with self.assertRaises(ValueError):
            validate_fixture_spec({"name": "fixture", "behavior": "success", "delay_seconds": 0, "score": 1})
        inputs = fixture_inputs()
        inputs[1].pop()
        with self.assertRaisesRegex(ValueError, "match the predetermined schedule"):
            self.run_fixture(inputs)
        self.assertFalse(self.output.exists())

    def test_invalid_backend_result_is_recorded_without_unsafe_error_text(self):
        class BadProcess:
            metadata = {"backend": "fixture"}
            def __init__(self, *args):
                pass
            def ready(self, timeout):
                pass
            def generate(self, *args):
                return {"outcome": {"raw_output": "unvalidated"}, "metadata": {}}
            def close(self):
                pass
        with patch("weight_space_lab.native_runner._Process", BadProcess):
            self.run_fixture(fixture_inputs(items=1))
        record = self.records()[0]
        self.assertEqual(record["error"], {"code": "invalid_backend_result", "retryable": False})
        self.assertIsNone(record["raw_output"])


if __name__ == "__main__":
    unittest.main()
