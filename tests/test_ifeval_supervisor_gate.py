"""Deterministic gate-race regression fixtures; no browser, model, or grader."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from weight_space_lab import ifeval_supervisor_gate as gate
from weight_space_lab.editor_budget import BudgetExhausted, BudgetLedger


class SupervisorGateTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"items": [{"item_id": f"tail-{i}"} for i in range(34)]}
            limits = {"editor_requests": 0, "model_probes": 35, "requested_sequences": 35,
                "generated_tokens": 34832, "elapsed_seconds": 5400, "billed_units": 0}
            with BudgetLedger(root / "budget.sqlite3", run_id="offline-gate-fixture",
                    limits=limits, binding={"fixture": True}, create=True) as book:
                yield root, book, config, {0}, {}, {
                    "deadline_epoch_ms": (time.time() + 300) * 1000,
                    "deadline_monotonic": time.monotonic() + 300}

    def dispatch(self, book, handled, index):
        book.reserve(f"generation-{index}", kind="probe", sequences=1,
            max_generated_tokens=16 if index == 1 else 1024,
            timeout_seconds=120, max_billed_units=0)
        book.mark_dispatched(f"generation-{index}")
        handled.add(index)

    def event(self, index, **updates):
        sample = {"item_id": "__warmup__" if index == 1 else f"tail-{index-2}",
            "request_id": f"generation-{index}", "warmup": index == 1,
            "raw_output": "Synthetic fixture response.", "generated_tokens": 2,
            "response_latency_ms": 10, "finish_reason": "stop",
            "execution_error": None, "cancelled": False,
            "local_webgpu_inference_observed": True, "webgpu_submitted_batches": 1}
        sample.update(updates)
        return {"type": "warmup-sample" if index == 1 else "task-sample", "data": sample}

    def append(self, root, event, *, newline=True):
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + ("\n" if newline else ""))

    def complete_warmup(self, root, book, config, handled, seen):
        self.dispatch(book, handled, 1)
        self.append(root, self.event(1))
        gate.reconcile_events(root, book, config, handled, seen)

    def test_resource_workload_uses_64_token_60_second_reservations(self):
        with self.fixture() as (root,book,config,handled,seen,grant):
            config["settings"]={"maxTokens":64,"requestLimitMs":60000}
            for index in (1,2):
                book.reserve(f"generation-{index}",kind="probe",sequences=1,
                    max_generated_tokens=16 if index==1 else 64,timeout_seconds=60,max_billed_units=0)
                book.mark_dispatched(f"generation-{index}"); handled.add(index)
                self.append(root,self.event(index,generated_tokens=2 if index==1 else 64))
            gate.reconcile_events(root,book,config,handled,seen)
            self.assertEqual(len(seen),2)
        for updates in ({"generated_tokens":65},{"response_latency_ms":60001}):
            with self.assertRaises(ValueError):
                gate._validate_sample(self.event(2,**updates),2,["__warmup__","tail-0"],
                    {"maxTokens":64,"requestLimitMs":60000})

    def test_stale_snapshot_old_guard_fails_but_gate_reread_reconciles_once(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            self.dispatch(book, handled, 2)
            # A supervisor iteration takes its event snapshot before completion.
            gate.reconcile_events(root, book, config, handled, seen)
            self.assertNotIn(2, seen)
            # The browser then durably records completion and its following gate.
            self.append(root, self.event(2))
            self.append(root, {"type": "gate-request", "data": {"request_id": "generation-3"}})
            (root / "generation-3-intent.json").write_text('{"request_id":"generation-3"}')
            with self.assertRaisesRegex(ValueError, "sequentially"):
                if not grant or 2 not in handled or 2 not in seen:
                    raise ValueError("requests must execute sequentially after warmup")
            observations = []
            with patch.object(gate, "reconcile", wraps=gate.reconcile) as reconciler:
                events = gate.assert_predecessor(3, root, book, config, handled, seen, grant,
                    on_observed=lambda index, sample: observations.append(index))
                self.assertEqual(len(events), 3)
                self.assertEqual(observations, [2])
                self.assertEqual(reconciler.call_count, 1)
                gate.assert_predecessor(3, root, book, config, handled, seen, grant)
                self.assertEqual(reconciler.call_count, 1)
            self.assertEqual(len(book.attempts()), 2)  # The helper creates no capacity.
            self.assertFalse((root / "generation-3-ack.json").exists())
            self.dispatch(book, handled, 3)
            with self.assertRaisesRegex(ValueError, "replay forbidden"):
                gate.assert_predecessor(3, root, book, config, handled, seen, grant)
            self.assertEqual(book.attempts()[1]["status"], "completed")

    def test_true_future_gate_and_missing_predecessor_are_rejected(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            self.dispatch(book, handled, 2)
            for index in (3, 4):
                with self.assertRaisesRegex(ValueError, "durable predecessor"):
                    gate.assert_predecessor(index, root, book, config, handled, seen, grant)
            self.assertEqual(book.attempts()[1]["status"], "dispatched")
            self.assertEqual(len(book.attempts()), 2)

    def test_partial_trailing_completion_is_withheld_until_complete(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            self.dispatch(book, handled, 2)
            self.append(root, self.event(2), newline=False)
            with self.assertRaisesRegex(ValueError, "durable predecessor"):
                gate.assert_predecessor(3, root, book, config, handled, seen, grant)
            self.assertNotIn(2, seen)
            with (root / "events.jsonl").open("a") as stream:
                stream.write("\n")
            gate.assert_predecessor(3, root, book, config, handled, seen, grant)
            self.assertIn(2, seen)

    def test_duplicate_conflicting_and_unexpected_completions_stop(self):
        mutations = [self.event(1), self.event(2, item_id="unscheduled"), self.event(3),
                     {"type": "task-sample", "data": None}]
        for event in mutations:
            with self.subTest(event=event), self.fixture() as (root, book, config, handled, seen, grant):
                self.complete_warmup(root, book, config, handled, seen)
                self.dispatch(book, handled, 2)
                self.append(root, event)
                with self.assertRaises(ValueError):
                    gate.assert_predecessor(3, root, book, config, handled, seen, grant)
                self.assertEqual(book.attempts()[1]["status"], "dispatched")
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            (root / "events.jsonl").write_text(json.dumps(self.event(1, raw_output="Changed fixture")) + "\n")
            with self.assertRaisesRegex(ValueError, "completion changed"):
                gate.assert_predecessor(2, root, book, config, handled, seen, grant)

    def test_malformed_json_and_removed_prefix_stop(self):
        for malformed in ('{"unfinished":\n', 'null\n', '{"data":{}}\n'):
            with self.subTest(malformed=malformed), self.fixture() as (root, book, config, handled, seen, grant):
                (root / "events.jsonl").write_text(malformed)
                with self.assertRaises(ValueError):
                    gate.assert_predecessor(1, root, book, config, handled, seen, grant)
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            (root / "events.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "removed or truncated"):
                gate.assert_predecessor(2, root, book, config, handled, seen, grant)

    def test_completed_ledger_row_is_bound_even_when_memory_state_is_empty(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            # A recovered in-memory map cannot cause another finish transition.
            gate.assert_predecessor(2, root, book, config, handled, {}, grant)
            completion_events = [row for row in book._db.execute("SELECT kind FROM events")
                                 if row[0] == "completed"]
            self.assertEqual(len(completion_events), 1)
            (root / "events.jsonl").write_text(json.dumps(self.event(1, generated_tokens=3)) + "\n")
            with self.assertRaisesRegex(ValueError, "previously reconciled ledger"):
                gate.assert_predecessor(2, root, book, config, handled, {}, grant)

    def test_device_loss_runtime_failure_and_generation_timeout_stop(self):
        failures = [
            {"type": "event", "data": {"type": "device-lost"}},
            {"type": "driver-error", "data": {"error": "fixture"}},
            {"type": "done", "data": {"status": "failed"}},
            self.event(1, finish_reason="timeout"),
            self.event(1, response_latency_ms=120001),
            self.event(1, response_latency_ms=float("nan")),
        ]
        for event in failures:
            with self.subTest(event=event), self.fixture() as (root, book, config, handled, seen, grant):
                self.dispatch(book, handled, 1)
                self.append(root, event)
                with self.assertRaises(ValueError):
                    gate.assert_predecessor(2, root, book, config, handled, seen, grant)
                self.assertEqual(book.attempts()[0]["status"], "dispatched")
                self.assertEqual(seen, {})

    def test_pending_request_and_grant_deadlines_stop_without_approval(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.dispatch(book, handled, 1)
            with patch.object(book, "_expired", return_value=True):
                with self.assertRaises(BudgetExhausted):
                    gate.assert_predecessor(2, root, book, config, handled, seen, grant)
            expired = {**grant, "deadline_monotonic": time.monotonic() - 1}
            with self.assertRaises(TimeoutError):
                gate.assert_predecessor(2, root, book, config, handled, seen, expired)
            self.assertEqual(len(book.attempts()), 1)

    def test_unacknowledged_completion_and_completed_request_replay_stop(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.append(root, self.event(1))
            with self.assertRaisesRegex(ValueError, "without an acknowledged"):
                gate.assert_predecessor(1, root, book, config, handled, seen, grant)
        with self.fixture() as (root, book, config, handled, seen, grant):
            self.complete_warmup(root, book, config, handled, seen)
            with self.assertRaisesRegex(ValueError, "replay forbidden"):
                gate.assert_predecessor(1, root, book, config, handled, seen, grant)
            self.assertEqual(len(book.attempts()), 1)

    def test_exact_34_item_suffix_capacity_and_last_gate(self):
        with self.fixture() as (root, book, config, handled, seen, grant):
            observations = []
            for index in range(1, 36):
                gate.assert_predecessor(index, root, book, config, handled, seen, grant,
                    on_observed=lambda i, sample: observations.append(i))
                self.dispatch(book, handled, index)
                self.append(root, self.event(index, finish_reason="length" if index > 1 else "stop"))
            gate.reconcile_events(root, book, config, handled, seen,
                on_observed=lambda i, sample: observations.append(i))
            self.assertEqual(observations, list(range(1, 36)))
            self.assertEqual(set(seen), set(range(1, 36)))
            self.assertEqual(book.summary()["reserved"]["generated_tokens"], 34832)
            self.assertEqual(len(book.attempts()), 35)
            with self.assertRaisesRegex(ValueError, "outside the frozen"):
                gate.assert_predecessor(36, root, book, config, handled, seen, grant)
            with self.assertRaises(BudgetExhausted):
                book.reserve("generation-36", kind="probe", sequences=1,
                    max_generated_tokens=1, timeout_seconds=120, max_billed_units=0)


if __name__ == "__main__":
    unittest.main()
