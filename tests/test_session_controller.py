"""Synthetic session transport/checkpoint/forward-worker integration only.

No test creates an authenticated session or calls a frontier model. The mock
transport is explicitly synthetic and cannot register a production adapter.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_checkpoint import EDITABLE, checkpoint_fixture, fixture_policy
from weight_space_lab.checkpoint import verify_candidate
from weight_space_lab.editor_budget import BudgetError
from weight_space_lab.editor_contract import read_json_bounded
from weight_space_lab.editor_controller import EditorController
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.session_controller import (
    SessionBoundaryUnavailable, SessionController, session_readiness,
)
from weight_space_lab.session_protocol import default_session_protocol


class SyntheticTransport:
    """Exercise controller plumbing; this never supplies real model identity."""

    def __init__(self):
        self.boundary_calls = 0
        self.receipt_calls = 0
        self.stop_calls = 0
        self.confirm_stop = True
        self.evidence_override = None

    def verify_boundary(self, manifest):
        self.boundary_calls += 1
        return {"evidence_kind": "synthetic_fixture", "session_sha256": canonical_hash(manifest)}

    def verify_response(self, request, response, launch):
        self.receipt_calls += 1
        if launch["session_sha256"] != request["session_sha256"]:
            raise ValueError("synthetic launch binding mismatch")
        if response["request_sha256"] != canonical_hash(request):
            raise ValueError("synthetic response binding mismatch")
        result = {
            "evidence_kind": "synthetic_fixture",
            "provenance": {
                "schema_version": 1, "source": "unavailable",
                "requested_selector": "gpt-6-astra", "requested_effort": "high",
                "accepted_selector": None, "observed_model": None, "observed_effort": None,
                "immutable_model_version": None,
                "interface": {"name": "synthetic-fixture", "version": None},
                "evidence_sha256": None,
            },
            "usage": {"generated_tokens": None, "billed_units": None},
        }
        return copy.deepcopy(self.evidence_override or result)

    def stop_and_confirm(self, launch):
        self.stop_calls += 1
        if launch["evidence_kind"] != "synthetic_fixture":
            raise ValueError("only a synthetic launch is supported by this fixture")
        return self.confirm_stop


class SessionControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = checkpoint_fixture(self.root)
        self.policy = fixture_policy()
        # The one-proposal variant exposes no extra inspection-only tensors.
        self.policy["inspection"]["allowed_tensors"] = EDITABLE.copy()
        self.development = [{"item_id": "synthetic-dev-one", "split": "development",
                             "messages": [{"role": "user", "content": "PRIVATE SYNTHETIC DEVELOPMENT INPUT"}]}]
        self.protocol = default_session_protocol(max_new_tokens=8, target_timeout_seconds=10,
                                                 local_elapsed_seconds=60, effort="high")
        self.probe = {"backend": {"name": "fixture", "behavior": "success", "delay_seconds": 0},
                      "seeds": [17], "max_new_tokens": 8, "timeout_seconds": 10}
        self.transport = SyntheticTransport()

    def create(self):
        controller = SessionController.create(
            self.root / "session", model=self.model, policy=self.policy,
            development=self.development, protocol=self.protocol, probe=self.probe,
            interface_observation={"source": "synthetic_test_only"})
        self.addCleanup(controller.ledger.close)
        return controller

    def prepare(self, controller):
        return controller.prepare_request("PRIVATE SYNTHETIC EDITOR INSTRUCTION")

    def export(self, controller):
        self.prepare(controller)
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            return controller.export_request()

    def response(self, controller, request):
        return {
            "schema_version": 1, "kind": "editor_response", "request_id": request["request_id"],
            "request_sha256": canonical_hash(request), "evidence_kind": "session_handoff",
            "model_identity": {"provider": None, "model": None, "version": None, "interface": None},
            "action": {"tool": "propose", "arguments": {"proposal": {
                "schema_version": 1, "base_sha256": controller.manifest["model_info"]["checkpoint_sha256"],
                "operations": [{"op": "scale_columns", "tensor": EDITABLE[0], "columns": [0], "factor": 0.5}],
                "provenance": "editor_proposal", "composition": "independent",
            }}},
            "usage": {"generated_tokens": None, "billed_units": None}, "scientific_result": False,
        }

    def reopen(self, controller):
        controller.ledger.close()
        reopened = SessionController(controller.root)
        self.addCleanup(reopened.ledger.close)
        return reopened

    def test_default_gate_blocks_before_reservation_or_export(self):
        controller = self.create()
        prepared = self.prepare(controller)
        self.assertFalse(prepared["exported"])
        self.assertFalse(prepared["editor_dispatched"])
        self.assertFalse(session_readiness()["ready"])
        with self.assertRaises(SessionBoundaryUnavailable):
            controller.export_request()
        self.assertEqual(controller.ledger.attempts(), [])
        self.assertEqual(list((controller.root / "requests").iterdir()), [])
        self.assertFalse((controller.root / "launch.json").exists())
        public = json.dumps(controller.status())
        for private in (str(self.root), "PRIVATE SYNTHETIC", "chat_template", "model_path"):
            self.assertNotIn(private, public)
        self.assertIsNone(controller.status()["observed_frontier_usage"]["generated_tokens"])

    def test_request_projects_approved_metadata_without_inventory_paths(self):
        controller = self.create()
        self.prepare(controller)
        request = read_json_bounded(controller.root / "prepared-request.json")
        context = request["approved_context"]
        self.assertEqual(set(context), {"model_info", "policy", "development", "feedback"})
        self.assertIsNone(context["feedback"])
        self.assertEqual(context["development"], self.development)
        info = context["model_info"]
        self.assertEqual(set(info), {"checkpoint_sha256", "architectures", "stored_parameter_count", "allowed_tensors"})
        self.assertEqual(set(info["allowed_tensors"]), set(self.policy["allowed_tensors"]))
        for tensor in info["allowed_tensors"].values():
            self.assertEqual(set(tensor), {"shape", "dtype", "parameter_count"})
        encoded = json.dumps(request)
        for forbidden in (str(self.root), "model_path", "shard", "data_offsets", "data_start", "chat_template",
                          "model_config", "tokenizer_config", "final_answers", "final_results"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(request["allowed_tools"], ["propose"])

    def test_synthetic_transport_builds_one_checkpoint_and_comparison(self):
        controller = self.create()
        request = self.export(controller)
        self.assertEqual(controller.ledger.summary()["charged"]["editor_requests"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["generated_tokens"], 0)
        response = self.response(controller, request)
        # Model-authored counts cannot become trusted frontier usage or consume
        # the independently enforced target-model token budget.
        response["usage"] = {"generated_tokens": 999999, "billed_units": 123}
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            summary = controller.import_response(response)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["evidence_kind"], "synthetic_fixture")
        self.assertEqual(summary["target_evidence_kind"], "fixture")
        self.assertFalse(summary["editor_identity_authenticated"])
        self.assertTrue(summary["candidate_validated"])
        self.assertTrue(summary["development_comparison_executed"])
        self.assertFalse(summary["scientific_result"])
        self.assertFalse(summary["experiment_executed"])
        self.assertFalse(summary["final_evaluation_executed"])
        candidate = controller.root / "candidates" / request["request_id"]
        audit = verify_candidate(self.model, candidate, self.policy)
        self.assertGreater(audit["changed_stored_entries"], 0)
        self.assertEqual(len(list((controller.root / "candidates").iterdir())), 1)
        self.assertEqual(summary["generated_tokens"], {"baseline": 3, "candidate": 3})
        totals = controller.ledger.summary()["charged"]
        self.assertEqual((totals["editor_requests"], totals["model_probes"], totals["requested_sequences"],
                          totals["completed_sequences"], totals["generated_tokens"], totals["retries"]),
                         (1, 2, 3, 3, 6, 0))
        self.assertIsNone(summary["observed_frontier_usage"]["generated_tokens"])
        encoded = json.dumps(summary)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("PRIVATE SYNTHETIC", encoded)
        self.assertEqual((self.transport.boundary_calls, self.transport.receipt_calls), (1, 1))

        reopened = self.reopen(controller)
        self.assertEqual(reopened.ledger.summary()["charged"], totals)
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            with self.assertRaises(BudgetError):
                reopened.export_request()
            with self.assertRaises(BudgetError):
                reopened.import_response(self.response(reopened, request))
        self.assertEqual(reopened.ledger.summary()["charged"], totals)

    def test_invalid_proposal_cannot_create_candidate_or_probe(self):
        controller = self.create()
        request = self.export(controller)
        response = self.response(controller, request)
        response["action"]["arguments"]["proposal"]["operations"][0]["op"] = "optimizer_update"
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport), patch(
                "weight_space_lab.session_comparison.run_development_comparison") as comparison:
            with self.assertRaises(ValueError):
                controller.import_response(response)
        comparison.assert_not_called()
        self.assertEqual(list((controller.root / "candidates").iterdir()), [])
        self.assertFalse((controller.root / "comparison").exists())
        self.assertEqual(controller.ledger.summary()["charged"]["failed_sequences"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)
        self.assertEqual(controller.ledger.summary()["charged"]["retries"], 0)

    def test_invalid_response_envelope_after_verified_receipt_consumes_slot(self):
        controller = self.create()
        request = self.export(controller)
        response = self.response(controller, request)
        response["action"] = {"tool": "train", "arguments": {}}
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport), patch(
                "weight_space_lab.session_controller.run_checkpoint_action") as checkpoint:
            with self.assertRaises(ValueError):
                controller.import_response(response)
        checkpoint.assert_not_called()
        self.assertTrue((controller.root / "session-receipt.json").exists())
        self.assertEqual(controller.ledger.summary()["charged"]["failed_sequences"], 1)
        self.assertEqual(controller.ledger.summary()["charged"]["pending_requests"], 0)
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            with self.assertRaises(BudgetError):
                controller.export_request()

    def test_new_attempt_preserves_old_spent_ledger_and_mode_identity(self):
        old_limits = {"editor_requests": 1, "model_probes": 0, "requested_sequences": 1,
                      "generated_tokens": 32, "elapsed_seconds": 120, "billed_units": None}
        previous = EditorController.create(
            self.root / "old-offline", model=self.model, policy=self.policy,
            development=self.development, limits=old_limits,
            interface_observation={"source": "synthetic_test_only"})
        self.addCleanup(previous.ledger.close)
        previous.export_request("Synthetic prior budget spending", max_output_tokens=32, timeout_seconds=60)
        spent = previous.ledger.summary()["charged"]
        new = self.create()
        self.assertEqual(previous.ledger.summary()["charged"], spent)
        self.assertEqual(new.ledger.attempts(), [])
        self.assertNotEqual(previous.ledger.configuration_sha256, new.ledger.configuration_sha256)
        with self.assertRaisesRegex(ValueError, "modes cannot be mixed"):
            SessionController(previous.root)
        self.assertEqual(previous.ledger.summary()["charged"], spent)

    def test_interrupted_export_retains_spending_until_stop_confirmed(self):
        controller = self.create()
        self.export(controller)
        reopened = self.reopen(controller)
        before = reopened.ledger.summary()["charged"]
        self.assertEqual(before["pending_requests"], 1)
        self.transport.confirm_stop = False
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            with self.assertRaisesRegex(SessionBoundaryUnavailable, "ambiguous"):
                reopened.recover()
        self.assertEqual(reopened.ledger.summary()["charged"], before)
        self.transport.confirm_stop = True
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            recovery = reopened.recover()
            self.assertEqual(len(recovery["recovered"]), 1)
            with self.assertRaises(BudgetError):
                reopened.export_request()
        charged = reopened.ledger.summary()["charged"]
        self.assertEqual(charged["editor_requests"], 1)
        self.assertEqual(charged["failed_sequences"], 1)
        self.assertEqual(charged["elapsed_seconds"], self.protocol["limits"]["local_elapsed_seconds"])
        self.assertEqual(charged["retries"], 0)

    def test_cancellation_is_persistent_even_when_remote_stop_is_unknown(self):
        controller = self.create()
        self.export(controller)
        self.transport.confirm_stop = False
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            with self.assertRaisesRegex(SessionBoundaryUnavailable, "unconfirmed"):
                controller.cancel()
        reopened = self.reopen(controller)
        self.assertTrue(reopened.ledger.summary()["cancelled"])
        self.assertEqual(reopened.ledger.summary()["charged"]["pending_requests"], 1)
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            with self.assertRaises(BudgetError):
                reopened.export_request()
        self.assertEqual(reopened.ledger.summary()["charged"]["editor_requests"], 1)

    def test_remote_stop_cannot_reconcile_pending_local_probe_after_restart(self):
        controller = self.create()
        request = self.export(controller)
        # Simulate a controller crash after local probe dispatch. No worker or
        # model is actually launched; the durable ledger state is the fixture.
        controller.ledger.finish(request["request_id"], status="completed",
                                 completed_sequences=1, failed_sequences=0,
                                 generated_tokens=0, billed_units=None)
        probe_id = "f" * 32
        controller.ledger.reserve(probe_id, kind="probe", sequences=1,
                                  max_generated_tokens=8, timeout_seconds=10,
                                  max_billed_units=0)
        controller.ledger.mark_dispatched(probe_id)
        reopened = self.reopen(controller)
        charged = reopened.ledger.summary()["charged"]
        self.assertEqual(charged["pending_requests"], 1)
        self.assertEqual(charged["generated_tokens"], 8)
        self.transport.confirm_stop = True
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport):
            with self.assertRaisesRegex(SessionBoundaryUnavailable, "local probe termination unconfirmed"):
                reopened.recover()
        self.assertEqual(reopened.ledger.summary()["charged"], charged)
        row = next(row for row in reopened.ledger.attempts() if row["request_id"] == probe_id)
        self.assertEqual(row["status"], "dispatched")
        self.assertEqual(self.transport.stop_calls, 0)

    def test_missing_ledger_cannot_reset_and_modes_cannot_be_mixed(self):
        controller = self.create()
        with self.assertRaisesRegex(ValueError, "modes cannot be mixed"):
            EditorController(controller.root)
        with self.assertRaisesRegex(ValueError, "no simulated editor"):
            controller.simulate()
        with self.assertRaises(ValueError):
            self.create()
        controller.ledger.close()
        (controller.root / "budget.sqlite3").unlink()
        with self.assertRaisesRegex(BudgetError, "ledger missing"):
            SessionController(controller.root)

    def test_substitution_in_negative_fixture_receipt_cannot_authenticate(self):
        controller = self.create()
        request = self.export(controller)
        response = self.response(controller, request)
        # Fabricated hostile receipt for a rejection test, never live evidence.
        evidence = self.transport.verify_response(request, response, {
            "session_sha256": request["session_sha256"]})
        evidence["evidence_kind"] = "authenticated_session"
        evidence["provenance"].update(source="transport_metadata", accepted_selector="gpt-6-astra",
                                      observed_model="other-model", observed_effort="high", evidence_sha256="d" * 64)
        self.transport.evidence_override = evidence
        with patch("weight_space_lab.session_controller._session_transport", return_value=self.transport), patch(
                "weight_space_lab.session_controller.run_checkpoint_action") as checkpoint:
            with self.assertRaises(ValueError):
                controller.import_response(response)
        checkpoint.assert_not_called()
        self.assertEqual(list((controller.root / "candidates").iterdir()), [])
        self.assertEqual(controller.ledger.summary()["charged"]["model_probes"], 0)


if __name__ == "__main__":
    unittest.main()
