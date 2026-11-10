"""Synthetic checks for the explicitly exploratory manual handoff.

The proposal data and target outputs in these tests are fixtures. No authenticated
session or real target model is called, and no fixture is offered as an Astra
proposal. The actual manual attempt must receive its response from the operator.
"""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from test_checkpoint import EDITABLE, checkpoint_fixture, fixture_policy
from weight_space_lab.editor_contract import read_json_bounded
from weight_space_lab.editor_controller import _implementation
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.session_controller import SessionController, session_readiness
from weight_space_lab.session_manual import ManualSessionController
from weight_space_lab.session_protocol import default_session_protocol


class ManualSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = checkpoint_fixture(self.root)
        self.policy = fixture_policy()
        self.policy["inspection"]["allowed_tensors"] = EDITABLE.copy()
        self.protocol = default_session_protocol(max_new_tokens=128, target_timeout_seconds=300,
                                                 local_elapsed_seconds=900, effort="high")
        self.probe = {"backend": {"name": "fixture", "behavior": "success", "delay_seconds": 0},
                      "seeds": [17], "max_new_tokens": 128, "timeout_seconds": 300}
        self.development = [{"item_id": "synthetic-manual-development", "split": "development",
                             "messages": [{"role": "user", "content": "PRIVATE SYNTHETIC DEVELOPMENT INPUT"}]}]
        self.attempt = self.root / "attempt"

    def create(self, *, historical=False):
        arguments = dict(model=self.model, policy=self.policy, development=self.development,
                         protocol=self.protocol, probe=self.probe,
                         interface_observation={"kind": "synthetic_fixture_only"})
        if historical:
            old = copy.deepcopy(_implementation())
            old["files"]["session_controller.py"] = "0" * 64
            with patch("weight_space_lab.session_controller._implementation", return_value=old), patch(
                    "weight_space_lab.editor_controller._implementation", return_value=old):
                controller = SessionController.create(self.attempt, **arguments)
        else:
            controller = SessionController.create(self.attempt, **arguments)
        controller.prepare_request("PRIVATE SYNTHETIC MANUAL PREPARATION")
        self.addCleanup(controller.ledger.close)
        return controller

    def enable(self, *, historical=False):
        old = self.create(historical=historical)
        old.ledger.close()
        controller = ManualSessionController.enable(
            self.attempt, operator_model="Astra", operator_mode="Ultra", submission_hours=48)
        self.addCleanup(controller.ledger.close)
        return controller

    def response_file(self, controller, *, factor=0.5, mutation=None):
        request = read_json_bounded(self.attempt / "manual" / "request.json")
        response = {"schema_version": 1, "kind": "editor_response", "request_id": request["request_id"],
                    "request_sha256": canonical_hash(request), "evidence_kind": "session_handoff",
                    "model_identity": {"provider": None, "model": None, "version": None, "interface": None},
                    "action": {"tool": "propose", "arguments": {"proposal": {
                        "schema_version": 1,
                        "base_sha256": controller.manifest["model_info"]["checkpoint_sha256"],
                        "provenance": "editor_proposal", "composition": "independent",
                        "operations": [{"op": "scale_columns", "tensor": EDITABLE[0],
                                        "columns": [0], "factor": factor}],
                    }}}, "usage": {"generated_tokens": None, "billed_units": None},
                    "scientific_result": False}
        if mutation is not None:
            mutation(response)
        path = self.root / "response.json"
        path.write_text(json.dumps(response))
        return path

    def test_enable_preserves_historical_manifest_budget_and_events(self):
        old = self.create(historical=True)
        manifest = (self.attempt / "session.json").read_bytes()
        ready = (self.attempt / "ready.json").read_bytes()
        binding = old.ledger.configuration_sha256
        events = old.ledger.events()
        old.ledger.close()
        with patch("weight_space_lab.session_controller._session_transport", side_effect=AssertionError("no launcher")):
            manual = ManualSessionController.enable(
                self.attempt, operator_model="Astra", operator_mode="Ultra", submission_hours=48)
        self.addCleanup(manual.ledger.close)
        self.assertEqual((self.attempt / "session.json").read_bytes(), manifest)
        self.assertEqual((self.attempt / "ready.json").read_bytes(), ready)
        self.assertEqual(manual.ledger.configuration_sha256, binding)
        self.assertEqual(manual.ledger.events(), events)
        self.assertEqual(manual.ledger.attempts(), [])
        self.assertFalse(session_readiness()["ready"])
        self.assertTrue((self.attempt / "manual" / "editor-packet.txt").is_file())
        self.assertEqual(json.loads(manifest)["protocol"]["editor"]["requested_effort"], "high")

    def test_operator_ultra_does_not_authenticate_runtime_or_change_old_protocol(self):
        controller = self.enable()
        request = read_json_bounded(self.attempt / "manual" / "request.json")
        amendment = read_json_bounded(self.attempt / "manual" / "amendment.json")
        packet = (self.attempt / "manual" / "editor-packet.txt").read_text()
        self.assertEqual(amendment["operator_selection"], {
            "model": "Astra", "mode": "Ultra", "requested_selector": "gpt-6-astra",
            "requested_effort": "ultra", "source": "operator_instruction"})
        self.assertTrue(all(value is None for value in amendment["runtime_provenance"].values()))
        self.assertEqual(amendment["protocol"]["editor"]["requested_effort"], "ultra")
        self.assertEqual(amendment["editor_isolation"], "unverified")
        self.assertFalse(amendment["controlled_research_run"])
        self.assertIn("unverified", packet.lower())
        self.assertNotIn(str(self.root), packet)
        self.assertEqual(request["allowed_tools"], ["propose"])
        self.assertEqual(controller.manifest["protocol"]["editor"]["requested_effort"], "high")
        public = json.dumps(controller.status())
        self.assertNotIn("PRIVATE SYNTHETIC", public)
        self.assertNotIn(str(self.root), public)

    def test_import_builds_one_candidate_then_comparison_uses_existing_capacity(self):
        controller = self.enable()
        response = self.response_file(controller)
        with patch("weight_space_lab.session_controller._session_transport", side_effect=AssertionError("no launcher")):
            accepted = controller.import_response_file(response)
        self.assertEqual(accepted["status"], "accepted")
        self.assertGreater(accepted["changed_stored_entries"], 0)
        self.assertNotEqual(accepted["result_sha256"], controller.manifest["model_info"]["checkpoint_sha256"])
        charged = controller.ledger.summary()["charged"]
        self.assertEqual((charged["editor_requests"], charged["model_probes"], charged["requested_sequences"]), (1, 0, 1))
        with patch("weight_space_lab.session_controller._session_transport", side_effect=AssertionError("no launcher")):
            result = controller.compare()
        self.assertEqual(result["status"], "completed")
        self.assertIn("SYNTHETIC FIXTURE OUTPUT", json.dumps(result))
        charged = controller.ledger.summary()["charged"]
        self.assertEqual((charged["editor_requests"], charged["model_probes"], charged["requested_sequences"],
                          charged["completed_sequences"], charged["generated_tokens"], charged["retries"]),
                         (1, 2, 3, 3, 6, 0))
        self.assertEqual(controller.ledger.limits["generated_tokens"], 256)
        with self.assertRaises(ValueError):
            controller.import_response_file(response)
        with self.assertRaises(ValueError):
            controller.compare()
        self.assertEqual(controller.ledger.summary()["charged"], charged)

    def test_generated_code_and_invalid_operations_consume_only_submission_slot(self):
        controller = self.enable()
        def invalid(response):
            response["action"]["arguments"]["proposal"]["operations"][0] = {
                "op": "execute_python", "code": "raise RuntimeError('never execute generated code')"}
        response = self.response_file(controller, mutation=invalid)
        with patch("weight_space_lab.session_comparison._Process") as worker:
            result = controller.import_response_file(response)
        worker.assert_not_called()
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(controller.ledger.summary()["charged"]["failed_sequences"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)
        with self.assertRaises(ValueError):
            controller.import_response_file(response)
        with self.assertRaises(ValueError):
            controller.compare()

    def test_malformed_json_is_an_invalid_consumed_submission(self):
        controller = self.enable()
        path = self.root / "response.json"
        path.write_text('{"schema_version": 1, invalid')
        result = controller.import_response_file(path)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(controller.ledger.summary()["charged"]["editor_requests"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["failed_sequences"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)

    def test_expired_submission_cannot_build_candidate_or_generate(self):
        controller = self.enable()
        response = self.response_file(controller)
        future = time.time() + 49 * 3600
        with patch("weight_space_lab.session_manual.time.time", return_value=future), patch(
                "weight_space_lab.session_manual.run_checkpoint_action") as checkpoint, patch(
                "weight_space_lab.session_comparison._Process") as worker:
            result = controller.import_response_file(response)
        checkpoint.assert_not_called()
        worker.assert_not_called()
        self.assertEqual(result["status"], "expired")
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)

    def test_zero_effective_change_is_reported_without_invented_improvement(self):
        controller = self.enable()
        accepted = controller.import_response_file(self.response_file(controller, factor=1.0))
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["changed_stored_entries"], 0)
        self.assertEqual(accepted["result_sha256"], controller.manifest["model_info"]["checkpoint_sha256"])
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)

    def test_target_failure_is_recorded_without_retry_or_candidate_generation(self):
        self.probe["backend"]["behavior"] = "error"
        controller = self.enable()
        controller.import_response_file(self.response_file(controller))
        result = controller.compare()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 2)
        self.assertEqual(controller.ledger.summary()["charged"]["retries"], 0)
        self.assertEqual(controller.ledger.summary()["charged"]["completed_sequences"], 1)
        with self.assertRaises(ValueError):
            controller.compare()

    def test_old_spending_cannot_be_converted_into_fresh_manual_capacity(self):
        old = self.create()
        old.ledger.reserve("already-spent", kind="editor", sequences=1, max_generated_tokens=0, timeout_seconds=900)
        old.ledger.mark_dispatched("already-spent")
        old.ledger.finish("already-spent", status="failed", completed_sequences=0, failed_sequences=1)
        spending = old.ledger.summary()["charged"]
        with self.assertRaises(ValueError):
            ManualSessionController.enable(
                self.attempt, operator_model="Astra", operator_mode="Ultra", submission_hours=48)
        self.assertEqual(old.ledger.summary()["charged"], spending)

    def test_truncated_fixture_outputs_remain_explicit_and_consume_exact_token_cap(self):
        controller = self.enable()
        controller.import_response_file(self.response_file(controller))
        from weight_space_lab.session_comparison import _receive

        def truncated(worker, expected, active):
            value = _receive(worker, expected, active)
            if expected == "result":
                value["outcome"].update(status="truncated", finish_reason="length",
                                        raw_output="SYNTHETIC TRUNCATED FIXTURE OUTPUT")
                value["outcome"]["token_counts"] = {"input": 2, "output": 128, "total": 130}
            return value

        with patch("weight_space_lab.session_comparison._receive", side_effect=truncated):
            result = controller.compare()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["comparison_evidence_kind"], "fixture")
        self.assertEqual([row["outcome"]["status"] for row in result["arms"]], ["truncated", "truncated"])
        self.assertEqual([row["generated_tokens"] for row in result["arms"]], [128, 128])
        self.assertEqual(controller.ledger.summary()["charged"]["generated_tokens"], 256)
        self.assertFalse(result["benchmark_improvement_established"])
        self.assertFalse(result["controlled_research_run"])

    def test_local_deadline_prevents_late_comparison_without_remote_cancellation_claim(self):
        controller = self.enable()
        controller.import_response_file(self.response_file(controller))
        future = time.time() + 901
        with patch("weight_space_lab.session_manual.time.time", return_value=future), patch(
                "weight_space_lab.session_comparison._Process") as worker:
            with self.assertRaises(ValueError):
                controller.compare()
        worker.assert_not_called()
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)
        with patch("weight_space_lab.session_controller._session_transport", side_effect=AssertionError("no launcher")):
            cancelled = controller.cancel()
        self.assertFalse(cancelled["remote_cancellation_attempted"])

    def test_interrupted_import_keeps_budget_until_conservative_local_recovery(self):
        controller = self.enable()
        response = self.response_file(controller)
        with patch("weight_space_lab.session_manual.run_checkpoint_action", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            controller.import_response_file(response)
        old_config = controller.ledger.configuration_sha256
        controller.ledger.close()
        reopened = ManualSessionController(self.attempt)
        self.addCleanup(reopened.ledger.close)
        self.assertEqual(reopened.ledger.configuration_sha256, old_config)
        self.assertEqual(reopened.ledger.summary()["charged"]["pending_requests"], 1)
        with patch("weight_space_lab.session_controller._session_transport", side_effect=AssertionError("no launcher")):
            self.assertEqual(reopened.recover()["recovered"], [])
            with patch("weight_space_lab.session_manual.time.time", return_value=time.time() + 901):
                recovered = reopened.recover()
        self.assertEqual(len(recovered["recovered"]), 1)
        charged = reopened.ledger.summary()["charged"]
        self.assertEqual((charged["editor_requests"], charged["failed_sequences"], charged["elapsed_seconds"],
                          charged["retries"], charged["model_probes"]), (1, 1, 900, 0, 0))
        self.assertFalse(recovered["remote_cancellation_attempted"])

    def test_manual_freeze_rejects_changed_implementation_and_cannot_renew_packet(self):
        controller = self.enable()
        original_amendment = (self.attempt / "manual" / "amendment.json").read_bytes()
        changed = copy.deepcopy(_implementation())
        changed["files"]["session_manual.py"] = "0" * 64
        with patch("weight_space_lab.session_manual._implementation", return_value=changed), self.assertRaises(ValueError):
            ManualSessionController(self.attempt)
        with self.assertRaises((ValueError, OSError)):
            ManualSessionController.enable(self.attempt, operator_model="Astra", operator_mode="Ultra", submission_hours=48)
        self.assertEqual((self.attempt / "manual" / "amendment.json").read_bytes(), original_amendment)
        self.assertEqual(controller.ledger.attempts(), [])

    def test_missing_ledger_cannot_reinitialize_manual_capacity(self):
        controller = self.enable()
        controller.ledger.close()
        (self.attempt / "budget.sqlite3").unlink()
        with self.assertRaises(ValueError):
            ManualSessionController(self.attempt)
        with self.assertRaises(ValueError):
            ManualSessionController.enable(self.attempt, operator_model="Astra", operator_mode="Ultra", submission_hours=48)
        self.assertFalse((self.attempt / "budget.sqlite3").exists())

    def test_delivered_text_packet_is_bound_to_immutable_request(self):
        controller = self.enable()
        (self.attempt / "manual" / "editor-packet.txt").write_text("changed packet")
        with self.assertRaisesRegex(ValueError, "packet differs"):
            ManualSessionController(self.attempt)
        self.assertEqual(controller.ledger.attempts(), [])

    def test_missing_response_path_does_not_consume_a_submission(self):
        controller = self.enable()
        with self.assertRaisesRegex(ValueError, "missing"):
            controller.import_response_file(self.root / "not-transferred-yet.json")
        self.assertEqual(controller.ledger.attempts(), [])
        self.assertFalse((self.attempt / "manual" / "submission-intent.json").exists())

    def test_oversized_numeric_factor_is_invalid_and_consumes_submission(self):
        controller = self.enable()
        result = controller.import_response_file(self.response_file(controller, factor=10**400))
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(controller.ledger.summary()["charged"]["failed_sequences"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)

    def test_checkpoint_execution_failure_is_distinct_from_invalid_proposal(self):
        controller = self.enable()
        response = self.response_file(controller)
        with patch("weight_space_lab.session_manual.run_checkpoint_action", side_effect=RuntimeError("synthetic worker failure")):
            result = controller.import_response_file(response)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_stage"], "checkpoint_validation_and_execution")
        self.assertFalse(result["candidate_validated"])
        self.assertEqual(controller.ledger.summary()["charged"]["failed_sequences"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)

    def test_exact_cli_import_and_compare_commands_execute_fixture_and_reject_duplicate(self):
        from weight_space_lab.__main__ import main
        controller = self.enable()
        response = self.response_file(controller)
        base = ["--root", str(self.attempt), "--manual-exploratory"]
        commands = (["editor-session-import", *base, "--response", str(response)],
                    ["editor-session-compare", *base])
        for command in commands:
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(command)
            self.assertEqual(status, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["editor_isolation"], "unverified")
            self.assertFalse(result["controlled_research_run"])
        self.assertEqual(result["comparison_evidence_kind"], "fixture")
        self.assertEqual(len(result["arms"]), 2)
        with redirect_stdout(io.StringIO()):
            status = main(commands[0])
        self.assertEqual(status, 2)
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 2)


if __name__ == "__main__":
    unittest.main()
