"""Synthetic-only checks for the single session development comparison."""

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_checkpoint import EDITABLE, checkpoint_fixture, fixture_policy
from weight_space_lab.checkpoint import CheckpointEditor
from weight_space_lab.editor_budget import BudgetExhausted, BudgetLedger
from weight_space_lab.session_comparison import run_development_comparison


class SessionComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = checkpoint_fixture(self.root)
        self.policy = fixture_policy()
        editor = CheckpointEditor(self.model, self.policy,
                                  exposure_log=self.root / "editor-exposures.jsonl")
        self.original_sha256 = editor.original_sha256
        self.candidate = self.root / "candidate"
        editor.propose({"schema_version": 1, "base_sha256": self.original_sha256,
                        "provenance": "editor_proposal", "composition": "independent",
                        "operations": [{"op": "scale_columns", "tensor": EDITABLE[0],
                                        "columns": [0], "factor": 0.5}]}, self.candidate)
        self.development = [{"item_id": "authored-development-1", "split": "development",
                             "messages": [{"role": "user", "content": "Synthetic development input."}]}]
        self.probe = {"backend": {"name": "fixture", "behavior": "success", "delay_seconds": 0},
                      "seeds": [17], "max_new_tokens": 8, "timeout_seconds": 10}
        self.limits = {"editor_requests": 0, "model_probes": 2, "requested_sequences": 2,
                       "generated_tokens": 16, "elapsed_seconds": 20, "billed_units": None}
        self.binding = {"test": "synthetic_session_comparison"}
        self.ledger = BudgetLedger(self.root / "budget.sqlite3", run_id="comparison-test",
                                   limits=self.limits, binding=self.binding, create=True)
        self.addCleanup(lambda: self.ledger.close())
        self.output = self.root / "comparison"

    def run_comparison(self, **overrides):
        arguments = {"original_model": self.model, "candidate": self.candidate,
                     "policy": self.policy, "development": self.development,
                     "probe": self.probe, "ledger": self.ledger, "output": self.output,
                     "expected_original_sha256": self.original_sha256}
        arguments.update(overrides)
        return run_development_comparison(**arguments)

    def test_exactly_one_sequence_per_arm_and_private_immutable_evidence(self):
        result = self.run_comparison()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evidence_kind"], "fixture")
        self.assertFalse(result["scientific_result"])
        self.assertTrue(result["private_development_evidence"])
        self.assertEqual([row["arm"] for row in result["arms"]], ["baseline", "candidate"])
        for row in result["arms"]:
            self.assertEqual(row["seed"], 17)
            self.assertEqual(row["item_id"], "authored-development-1")
            self.assertEqual(row["outcome"]["raw_output"], "SYNTHETIC FIXTURE OUTPUT")
            self.assertEqual(row["generated_tokens"], 3)
            self.assertTrue(row["backend_metadata"]["startup"]["synthetic"])
        summary = self.ledger.summary()
        self.assertEqual(summary["charged"]["model_probes"], 2)
        self.assertEqual(summary["charged"]["requested_sequences"], 2)
        self.assertEqual(summary["charged"]["completed_sequences"], 2)
        self.assertEqual(summary["observed"]["generated_tokens"], 6)
        self.assertEqual(summary["reserved"]["generated_tokens"], 16)
        self.assertEqual(summary["charged"]["retries"], 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["sequence_counts"], {"baseline": 1, "candidate": 1})
        self.assertEqual(manifest["development"], self.development)
        self.assertEqual(manifest["probe"], self.probe)
        events = [event["kind"] for event in self.ledger.events()]
        self.assertLess(max(index for index, kind in enumerate(events) if kind == "reserved"),
                        min(index for index, kind in enumerate(events) if kind == "dispatched"))
        before = (self.output / "result.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "no rerun"):
            self.run_comparison()
        self.assertEqual((self.output / "result.json").read_bytes(), before)

    def test_held_out_multiple_items_seeds_and_undeclared_limits_rejected(self):
        cases = []
        held_out = copy.deepcopy(self.development)
        held_out[0]["split"] = "final"
        cases.append({"development": held_out})
        second = copy.deepcopy(self.development[0])
        second["item_id"] = "second"
        cases.append({"development": self.development + [second]})
        cases.append({"probe": self.probe | {"seeds": [17, 18]}})
        cases.append({"probe": self.probe | {"max_new_tokens": 257}})
        cases.append({"probe": self.probe | {"extra": "undeclared"}})
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.run_comparison(**arguments)
        self.assertEqual(self.ledger.attempts(), [])
        self.assertFalse(self.output.exists())

    def test_candidate_and_original_verification_precede_any_generation(self):
        config = self.candidate / "model" / "config.json"
        config.chmod(0o644)
        config.write_text('{"architecture":"undeclared change"}')
        with patch("weight_space_lab.session_comparison._Process") as process:
            result = self.run_comparison()
        process.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["arms"]), 2)
        self.assertTrue(all(not row["dispatched"] for row in result["arms"]))
        self.assertEqual(result["ledger"]["charged"]["failed_sequences"], 2)

    def test_original_digest_mismatch_stops_both_arms(self):
        with patch("weight_space_lab.session_comparison._Process") as process:
            result = self.run_comparison(expected_original_sha256="0" * 64)
        process.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["ledger"]["charged"]["completed_sequences"], 0)

    def test_second_reservation_exhaustion_cannot_dispatch_first_arm(self):
        self.ledger.close()
        self.limits["model_probes"] = 1
        self.ledger = BudgetLedger(self.root / "small-budget.sqlite3", run_id="comparison-test",
                                   limits=self.limits, binding=self.binding, create=True)
        with patch("weight_space_lab.session_comparison._Process") as process, patch(
                "weight_space_lab.session_comparison.run_checkpoint_action") as verification:
            result = self.run_comparison()
        process.assert_not_called()
        verification.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["ledger"]["charged"]["model_probes"], 1)
        self.assertEqual(result["ledger"]["charged"]["completed_sequences"], 0)

    def test_failure_has_no_retry_or_candidate_generation(self):
        self.probe["backend"]["behavior"] = "error_once"
        result = self.run_comparison()
        self.assertEqual(result["status"], "failed")
        self.assertEqual([row["dispatched"] for row in result["arms"]], [True, False])
        self.assertIsNone(result["arms"][0]["generated_tokens"])
        self.assertEqual(result["ledger"]["charged"]["retries"], 0)
        self.assertEqual(result["ledger"]["charged"]["generated_tokens"], 8)

    def test_measured_token_overage_is_recorded_and_permanently_stops_attempt(self):
        # This deliberately inconsistent synthetic backend always emits three
        # synthetic tokens. Real transformers generation receives the declared
        # max_new_tokens in its validated backend configuration before dispatch.
        self.probe["max_new_tokens"] = 2
        result = self.run_comparison()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["arms"][0]["generated_tokens"], 3)
        self.assertEqual(result["ledger"]["observed"]["generated_tokens"], 3)
        self.assertTrue(result["ledger"]["cancelled"])
        self.assertEqual(result["ledger"]["charged"]["limit_violations"], 1)
        self.assertFalse(result["arms"][1]["dispatched"])

    def test_timeout_stops_worker_and_consumes_unknown_generation_bound(self):
        self.probe["timeout_seconds"] = 1
        self.probe["backend"]["delay_seconds"] = 5
        started = time.monotonic()
        result = self.run_comparison()
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["arms"][0]["status"], "timed_out")
        self.assertEqual(result["ledger"]["charged"]["generated_tokens"], 8)
        self.assertEqual(result["ledger"]["charged"]["pending_requests"], 0)

    def test_persistent_cancellation_stops_worker_and_skips_candidate(self):
        self.probe["backend"]["delay_seconds"] = 5
        errors = []

        def cancel():
            try:
                with BudgetLedger(self.ledger.path, run_id="comparison-test",
                                  limits=self.limits, binding=self.binding) as other:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        if any(row["status"] == "dispatched" for row in other.attempts()):
                            time.sleep(0.1)
                            other.cancel()
                            return
                        time.sleep(0.01)
                    errors.append("no dispatch observed")
            except Exception as error:
                errors.append(type(error).__name__)

        thread = threading.Thread(target=cancel)
        thread.start()
        started = time.monotonic()
        result = self.run_comparison()
        thread.join(timeout=6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(result["arms"][0]["status"], "cancelled")
        self.assertFalse(result["arms"][1]["dispatched"])
        self.assertTrue(result["ledger"]["cancelled"])

    def test_whole_attempt_deadline_is_checked_during_forward_generation(self):
        self.probe["backend"]["delay_seconds"] = 5
        deadline = time.monotonic() + 0.75

        def active():
            if time.monotonic() >= deadline:
                raise BudgetExhausted("persisted whole-attempt deadline expired")

        started = time.monotonic()
        result = self.run_comparison(check_active=active)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(result["arms"][0]["status"], "timed_out")
        self.assertTrue(result["arms"][0]["dispatched"])
        self.assertFalse(result["arms"][1]["dispatched"])
        self.assertEqual(result["ledger"]["charged"]["pending_requests"], 0)

    def test_restart_preserves_full_reservations_and_refuses_automatic_replay(self):
        with patch("weight_space_lab.session_comparison._Process", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_comparison()
        self.ledger.close()
        self.ledger = BudgetLedger(self.root / "budget.sqlite3", run_id="comparison-test",
                                   limits=self.limits, binding=self.binding)
        self.assertEqual(self.ledger.summary()["charged"]["generated_tokens"], 16)
        self.assertEqual(self.ledger.summary()["charged"]["pending_requests"], 2)
        with self.assertRaisesRegex(ValueError, "no rerun"):
            self.run_comparison()
        recovered = self.ledger.recover_interrupted(executor_stopped=True)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(self.ledger.summary()["charged"]["generated_tokens"], 16)
        with patch("weight_space_lab.session_comparison._Process") as process:
            result = self.run_comparison(output=self.root / "retry")
        process.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.ledger.summary()["charged"]["model_probes"], 2)


if __name__ == "__main__":
    unittest.main()
