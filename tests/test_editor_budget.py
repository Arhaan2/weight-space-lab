"""Accounting fixtures only; these tests never execute an editor or model."""

import concurrent.futures
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_space_lab.editor_budget import (
    BudgetCancelled, BudgetError, BudgetExhausted, BudgetLedger,
)


def limits(**changes):
    return {"editor_requests": 3, "model_probes": 3, "requested_sequences": 6,
            "generated_tokens": 60, "elapsed_seconds": 60.0,
            "billed_units": 600, **changes}


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.sqlite3"
        self.binding = {"development_bundle_sha256": "a" * 64,
                        "billing_unit": "fixture_units_non_scientific"}

    def open(self, *, create=False, budget=None, binding=None):
        ledger = BudgetLedger(self.path, run_id="fixture", limits=budget or limits(),
                              binding=self.binding if binding is None else binding, create=create)
        self.addCleanup(ledger.close)
        return ledger

    def reserve(self, ledger, request_id="request-1", **changes):
        return ledger.reserve(request_id, **{"kind": "editor", "sequences": 1,
                                             "max_generated_tokens": 10,
                                             "timeout_seconds": 10, "max_billed_units": 100,
                                             **changes})

    def finish(self, ledger, request_id="request-1", **changes):
        return ledger.finish(request_id, **{"status": "completed", "completed_sequences": 1,
                                            "failed_sequences": 0, "generated_tokens": 3,
                                            "elapsed_seconds": 0.5, "billed_units": 25,
                                            **changes})

    def test_reserves_full_bounds_before_dispatch_and_reconciles_measured_usage(self):
        ledger = self.open(create=True)
        row = self.reserve(ledger)
        self.assertEqual(row["status"], "reserved")
        before = ledger.summary()
        self.assertEqual(before["charged"]["generated_tokens"], 10)
        self.assertEqual(before["charged"]["editor_requests"], 1)
        self.assertEqual(before["charged"]["requested_sequences"], 1)
        self.assertIsNone(before["observed"]["generated_tokens"])
        ledger.mark_dispatched("request-1")
        self.finish(ledger)
        after = ledger.summary()
        self.assertEqual(after["reserved"]["generated_tokens"], 10)
        self.assertEqual(after["reserved"]["pending_generated_tokens"], 0)
        self.assertEqual(after["charged"]["generated_tokens"], 3)
        self.assertEqual(after["observed"]["generated_tokens"], 3)
        self.assertEqual(after["charged"]["completed_sequences"], 1)
        self.assertEqual(after["charged"]["failed_sequences"], 0)
        self.assertEqual(after["remaining"]["billed_units"], 575)
        self.assertEqual([e["kind"] for e in ledger.events()], ["created", "reserved", "dispatched", "completed"])

    def test_budget_exhaustion_is_atomic_and_zero_limits_deny_kinds(self):
        for key, value in (("editor_requests", 0), ("requested_sequences", 0),
                           ("generated_tokens", 9), ("elapsed_seconds", 9), ("billed_units", 99)):
            with self.subTest(key=key):
                self.path = Path(self.temp.name) / f"{key}.sqlite3"
                ledger = self.open(create=True, budget=limits(**{key: value}))
                with self.assertRaises(BudgetExhausted):
                    self.reserve(ledger)
                self.assertEqual(ledger.attempts(), [])
        self.path = Path(self.temp.name) / "probes.sqlite3"
        ledger = self.open(create=True, budget=limits(model_probes=0))
        with self.assertRaises(BudgetExhausted):
            self.reserve(ledger, kind="probe")

    def test_missing_database_or_changed_configuration_cannot_reset_budget(self):
        with self.assertRaises(BudgetError):
            self.open()
        ledger = self.open(create=True)
        self.reserve(ledger)
        with self.assertRaises(FileExistsError):
            self.open(create=True)
        reopened = self.open()
        self.assertEqual(reopened.summary()["charged"]["editor_requests"], 1)
        with self.assertRaises(BudgetError):
            self.open(budget=limits(generated_tokens=1000))
        with self.assertRaises(BudgetError):
            self.open(binding={**self.binding, "architecture": "changed"})

    def test_interrupted_reservation_spends_unknown_maxima_and_fails_sequences(self):
        ledger = self.open(create=True)
        self.reserve(ledger, sequences=2)
        ledger.mark_dispatched("request-1")
        resumed = self.open()
        self.assertEqual(resumed.recover_interrupted(), [])
        recovered = resumed.recover_interrupted(executor_stopped=True)
        self.assertEqual(recovered[0]["status"], "interrupted")
        self.assertIsNone(recovered[0]["observed_generated_tokens"])
        summary = resumed.summary()
        self.assertEqual(summary["charged"]["generated_tokens"], 10)
        self.assertEqual(summary["charged"]["elapsed_seconds"], 10)
        self.assertEqual(summary["charged"]["billed_units"], 100)
        self.assertEqual(summary["charged"]["failed_sequences"], 2)
        self.assertEqual(summary["charged"]["pending_requests"], 0)
        self.assertEqual(resumed.recover_interrupted(executor_stopped=True), [])
        with self.assertRaises(BudgetError):
            self.reserve(resumed)

    def test_process_exit_after_commit_cannot_lose_reservation(self):
        ledger = self.open(create=True)
        code = """
import json, os, sys
from weight_space_lab.editor_budget import BudgetLedger
ledger=BudgetLedger(sys.argv[1],run_id='fixture',limits=json.loads(sys.argv[2]),binding=json.loads(sys.argv[3]))
ledger.reserve('crash-request',kind='probe',sequences=2,max_generated_tokens=20,timeout_seconds=8,max_billed_units=80)
ledger.mark_dispatched('crash-request')
os._exit(23)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.path), json.dumps(limits()),
                                 json.dumps(self.binding)], capture_output=True)
        self.assertEqual(result.returncode, 23, result.stderr.decode())
        resumed = self.open()
        resumed.recover_interrupted(executor_stopped=True)
        summary = resumed.summary()
        self.assertEqual(summary["charged"]["model_probes"], 1)
        self.assertEqual(summary["charged"]["failed_sequences"], 2)
        self.assertEqual(summary["charged"]["generated_tokens"], 20)

    def test_retry_requires_failed_parent_and_charges_another_full_attempt(self):
        ledger = self.open(create=True)
        self.reserve(ledger)
        with self.assertRaises(BudgetError):
            self.reserve(ledger, "request-2", retry_of="request-1")
        ledger.mark_dispatched("request-1")
        self.finish(ledger, status="failed", completed_sequences=0, failed_sequences=1,
                    generated_tokens=None, elapsed_seconds=None, billed_units=None)
        retry = self.reserve(ledger, "request-2", retry_of="request-1")
        self.assertEqual(retry["attempt_index"], 1)
        self.assertEqual(ledger.summary()["charged"]["requested_sequences"], 2)
        self.assertEqual(ledger.summary()["charged"]["generated_tokens"], 20)
        self.assertEqual(ledger.summary()["charged"]["retries"], 1)
        with self.assertRaises(BudgetError):
            self.reserve(ledger, "request-3", retry_of="request-1")
        with self.assertRaises(BudgetError):
            self.reserve(ledger, "probe-retry", kind="probe", retry_of="request-1")

    def test_successful_attempt_cannot_be_retried_or_rewritten(self):
        ledger = self.open(create=True)
        self.reserve(ledger)
        ledger.mark_dispatched("request-1")
        self.finish(ledger)
        with self.assertRaises(BudgetError):
            self.reserve(ledger, "request-2", retry_of="request-1")
        with self.assertRaises(BudgetError):
            self.finish(ledger, generated_tokens=0)
        with self.assertRaises(BudgetError):
            ledger.mark_dispatched("request-1")

    def test_persistent_cancellation_blocks_existing_and_new_dispatches(self):
        ledger = self.open(create=True)
        self.reserve(ledger)
        ledger.cancel()
        resumed = self.open()
        self.assertTrue(resumed.summary()["cancelled"])
        with self.assertRaises(BudgetCancelled):
            resumed.check_active("request-1")
        with self.assertRaises(BudgetCancelled):
            resumed.mark_dispatched("request-1")
        with self.assertRaises(BudgetCancelled):
            self.reserve(resumed, "request-2")
        # Cancellation still allows stopped requests to be accounted for.
        resumed.finish("request-1", status="cancelled", completed_sequences=0, failed_sequences=1)
        self.assertEqual(resumed.summary()["charged"]["failed_sequences"], 1)

    def test_deadline_is_checked_before_dispatch_and_during_execution(self):
        with patch("weight_space_lab.editor_budget.time.time", return_value=100):
            ledger = self.open(create=True)
            self.reserve(ledger)
        with patch("weight_space_lab.editor_budget.time.time", return_value=111):
            with self.assertRaises(BudgetExhausted):
                ledger.mark_dispatched("request-1")
            with self.assertRaises(BudgetExhausted):
                ledger.check_active("request-1")
            self.assertEqual(len(ledger.recover_interrupted()), 1)

    def test_clock_regression_cannot_extend_deadline_on_reopen(self):
        ledger = self.open(create=True)
        row = self.reserve(ledger)
        resumed = self.open()
        with patch("weight_space_lab.editor_budget.time.time", return_value=row["reserved_at"] - 1):
            with self.assertRaises(BudgetExhausted):
                resumed.check_active("request-1")
            with self.assertRaises(BudgetExhausted):
                self.reserve(resumed, "request-2")
        with patch("weight_space_lab.editor_budget.time.monotonic", return_value=row["reserved_monotonic"] - 1):
            with self.assertRaises(BudgetExhausted):
                resumed.mark_dispatched("request-1")
            self.assertEqual(len(resumed.recover_interrupted()), 1)
        self.assertEqual(resumed.summary()["charged"]["generated_tokens"], 10)
        self.assertEqual(resumed.summary()["charged"]["failed_sequences"], 1)

    def test_monotonic_deadline_expires_even_if_wall_time_is_frozen(self):
        ledger = self.open(create=True)
        row = self.reserve(ledger)
        with patch("weight_space_lab.editor_budget.time.time", return_value=row["reserved_at"] + 1), \
                patch("weight_space_lab.editor_budget.time.monotonic", return_value=row["deadline_monotonic"] + 1):
            with self.assertRaises(BudgetExhausted):
                ledger.check_active("request-1")
            self.assertEqual(len(ledger.recover_interrupted()), 1)

    def test_reported_overage_is_not_clipped_and_persistently_cancels(self):
        ledger = self.open(create=True)
        self.reserve(ledger)
        ledger.mark_dispatched("request-1")
        result = self.finish(ledger, generated_tokens=11)
        self.assertEqual(result["observed_generated_tokens"], 11)
        self.assertEqual(result["charged_generated_tokens"], 11)
        self.assertEqual(result["limit_violation"], 1)
        self.assertTrue(ledger.summary()["cancelled"])
        with self.assertRaises(BudgetCancelled):
            self.reserve(ledger, "request-2")

    def test_billing_unknown_is_null_and_configured_limit_requires_bound(self):
        ledger = self.open(create=True)
        with self.assertRaises(BudgetError):
            self.reserve(ledger, max_billed_units=None)
        self.path = Path(self.temp.name) / "unavailable.sqlite3"
        unbilled = self.open(create=True, budget=limits(billed_units=None))
        self.reserve(unbilled, max_billed_units=None)
        unbilled.mark_dispatched("request-1")
        self.finish(unbilled, billed_units=None)
        self.assertIsNone(unbilled.summary()["observed"]["billed_units"])
        self.assertIsNone(unbilled.summary()["charged"]["billed_units"])
        self.assertIsNone(unbilled.summary()["remaining"]["billed_units"])

    def test_concurrent_reservations_cannot_overspend(self):
        budget = limits(editor_requests=1)
        self.open(create=True, budget=budget)

        def reserve_one(index):
            with BudgetLedger(self.path, run_id="fixture", limits=budget, binding=self.binding) as ledger:
                try:
                    self.reserve(ledger, f"concurrent-{index}")
                    return True
                except BudgetExhausted:
                    return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            result = list(pool.map(reserve_one, range(4)))
        self.assertEqual(sum(result), 1)
        self.assertEqual(self.open(budget=budget).summary()["charged"]["editor_requests"], 1)

    def test_invalid_usage_and_sequence_counts_do_not_erase_reservation(self):
        ledger = self.open(create=True)
        self.reserve(ledger)
        with self.assertRaises(BudgetError):
            self.finish(ledger)
        ledger.mark_dispatched("request-1")
        for change in ({"generated_tokens": -1}, {"generated_tokens": True},
                       {"elapsed_seconds": float("nan")}, {"completed_sequences": 0},
                       {"status": "failed"}, {"billed_units": 1.5}):
            with self.subTest(change=change), self.assertRaises(BudgetError):
                self.finish(ledger, **change)
        self.assertEqual(ledger.attempts()[0]["status"], "dispatched")
        self.assertEqual(ledger.summary()["charged"]["generated_tokens"], 10)

    def test_database_is_private_and_uses_full_sync(self):
        ledger = self.open(create=True)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(ledger._db.execute("PRAGMA synchronous").fetchone()[0], 2)
        self.assertEqual(ledger._db.execute("PRAGMA journal_mode").fetchone()[0], "wal")


if __name__ == "__main__":
    unittest.main()
