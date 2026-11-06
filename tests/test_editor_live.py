"""Readiness never confuses configuration or offline fixtures with live proof."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_checkpoint import checkpoint_fixture, fixture_policy
from weight_space_lab.__main__ import main
from weight_space_lab.editor_controller import EditorController
from weight_space_lab.editor_live import (LiveDispatchBlocked, _budget_snapshot, connection_test_plan,
                                         live_readiness, require_live_ready, validate_connection_response)


class LiveReadinessTests(unittest.TestCase):
    def test_readiness_keeps_provider_caps_separate_from_local_reservations(self):
        result = live_readiness(inspect_cli=False)
        self.assertFalse(result["ready"])
        self.assertFalse(result["real_editor_request_sent"])
        rows = {row["id"]: row for row in result["conditions"]}
        for identifier in ("live_adapter_missing", "actual_launch_isolation_unverified",
                           "frontier_generation_cap_not_demonstrated"):
            self.assertIn(identifier, result["dispatch_blockers"])
            self.assertTrue(rows[identifier]["expected_evidence"])
            self.assertTrue(rows[identifier]["smallest_next_step"])
        self.assertEqual(rows["frontier_generation_cap_not_demonstrated"]["classification"], "unavailable_interface_capability")
        self.assertNotIn("frontier_billed_cap_not_demonstrated", result["dispatch_blockers"])
        self.assertFalse(rows["frontier_billed_cap_not_demonstrated"]["mandatory"])

    def test_response_identity_is_post_response_not_circular_prelaunch_evidence(self):
        result = live_readiness(inspect_cli=False)
        for identifier in ("accepted_selector_not_observed", "response_model_metadata_not_observed"):
            row = next(row for row in result["conditions"] if row["id"] == identifier)
            self.assertEqual(row["stage"], "post_response")
            self.assertNotIn(identifier, result["dispatch_blockers"])
        self.assertIsNone(result["selector_accepted_by_live_interface"])
        self.assertIsNone(result["immutable_model_version"])
        self.assertFalse(result["model_self_description_used"])

    def test_search_protocol_is_reported_unchanged_and_not_used_as_connection_gate(self):
        result = live_readiness(inspect_cli=False)
        self.assertFalse(result["legacy_search_check"]["run_configuration_complete"])
        self.assertIn("missing pin: editor_model_version", result["legacy_search_check"]["pending_run_requirements"])
        self.assertNotIn("immutable_model_version_not_exposed", result["dispatch_blockers"])

    def test_controller_and_command_use_same_fail_closed_gate_without_execution(self):
        with patch("weight_space_lab.editor_live.installed_cli", return_value={"available": True, "version": "fixture"}), \
             patch("weight_space_lab.editor_sandbox.SandboxRunner.run", side_effect=AssertionError("unexpected launch")):
            with self.assertRaises(LiveDispatchBlocked) as caught:
                EditorController.dispatch_live(None, model="gpt-6-astra", verified=True, hard_cap=True)
            self.assertFalse(caught.exception.report["ready"])
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["editor-live-readiness"]), 2)
            self.assertEqual(json.loads(output.getvalue())["dispatch_blockers"], caught.exception.report["dispatch_blockers"])

    def test_prepared_connection_has_no_tools_changes_search_retry_or_identity_claim(self):
        plan = connection_test_plan()
        self.assertEqual(plan["request"]["tools"], [])
        self.assertEqual(plan["automatic_retries"], 0)
        self.assertEqual(plan["bounds"]["candidate_proposals"], 0)
        self.assertEqual(plan["bounds"]["target_model_probes"], 0)
        self.assertEqual(plan["bounds"]["editor_requests"], 1)
        self.assertFalse(plan["weight_changes_allowed"])
        self.assertEqual(validate_connection_response({"status": "acknowledged"}), {"status": "acknowledged"})
        for extra in ({"action": {"tool": "propose"}}, {"model": "gpt-6-astra"}, {"version": "0.163.0"}):
            with self.assertRaises(ValueError):
                validate_connection_response({"status": "acknowledged", **extra})

    def test_preparing_plan_creates_no_budget_and_never_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            with patch("weight_space_lab.editor_budget.BudgetLedger.reserve", side_effect=AssertionError("reservation forbidden")):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["editor-prepare-connection", "--out", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text()), connection_test_plan())
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["plan.json"])

    def test_budget_snapshot_never_resets_or_promotes_exhausted_offline_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = checkpoint_fixture(root)
            controller = EditorController.create(root / "session", model=model, policy=fixture_policy(),
                development=[{"item_id": "dev", "split": "development", "messages": [{"role": "user", "content": "ack"}]}],
                limits={"editor_requests": 1, "model_probes": 0, "requested_sequences": 1,
                        "generated_tokens": 64, "elapsed_seconds": 30, "billed_units": None},
                interface_observation={"status": "offline"})
            controller.export_request("not dispatched to provider", max_output_tokens=64, timeout_seconds=30)
            controller.recover()
            controller.ledger.close()
            before = (root / "session/budget.sqlite3").read_bytes()
            result = _budget_snapshot(root / "session")
            self.assertTrue(result["available"])
            self.assertFalse(result["can_authorize_live_execution"])
            self.assertEqual(result["remaining"]["editor_requests"], 0)
            self.assertEqual(result["remaining"]["generated_tokens"], 0)
            self.assertEqual((root / "session/budget.sqlite3").read_bytes(), before)
            self.assertFalse(_budget_snapshot(root / "missing")["available"])
            self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
