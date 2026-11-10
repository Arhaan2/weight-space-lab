"""Phase 7 integration checks use only synthetic checkpoints and development input."""

import copy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import threading
import time
import unittest

from test_checkpoint import checkpoint_fixture, fixture_policy, EDITABLE
from weight_space_lab.editor_budget import BudgetExhausted
from weight_space_lab.editor_controller import EditorController
from weight_space_lab.protocol import canonical_hash


def response_for(request, tool="model_info", arguments=None):
    return {"schema_version": 1, "kind": "editor_response", "request_id": request["request_id"],
            "request_sha256": canonical_hash(request), "evidence_kind": "unverified_file_handoff",
            "model_identity": {"provider": None, "model": None, "version": None, "interface": "manual_file"},
            "action": {"tool": tool, "arguments": arguments or {}},
            "usage": {"generated_tokens": None, "billed_units": None}, "scientific_result": False}


class EditorControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = checkpoint_fixture(self.root)
        self.limits = {"editor_requests": 8, "model_probes": 4, "requested_sequences": 12,
                       "generated_tokens": 4096, "elapsed_seconds": 600, "billed_units": None}
        self.development = [{"item_id": "authored-dev", "split": "development",
                             "messages": [{"role": "user", "content": "Compute 7 + 5."}]}]

    def create(self, *, probe=None):
        controller = EditorController.create(self.root / "session", model=self.model, policy=fixture_policy(),
            development=self.development, limits=self.limits, probe=probe,
            interface_observation={"status": "pending", "actual_returned_model_identity": None})
        self.addCleanup(controller.ledger.close)
        return controller

    def exchange(self, controller, tool="model_info", arguments=None):
        request = controller.export_request("Offline integration check", max_output_tokens=64, timeout_seconds=10)
        return request, controller.import_response(response_for(request, tool, arguments))

    def proposal(self, controller):
        return {"schema_version": 1, "base_sha256": controller.editor.original_sha256,
                "provenance": "editor_proposal", "composition": "independent",
                "operations": [{"op": "scale_columns", "tensor": EDITABLE[0], "columns": [0], "factor": 0.5}]}

    def probe_spec(self, behavior="success", delay=0, timeout=3):
        return {"backend": {"name": "fixture", "behavior": behavior, "delay_seconds": delay},
                "seeds": [17], "max_new_tokens": 4, "timeout_seconds": timeout}

    def test_file_round_trip_preserves_unverified_identity_and_charges_unknowns(self):
        controller = self.create()
        request = controller.export_request("Offline only", max_output_tokens=12, timeout_seconds=10)
        self.assertNotIn("model_path", json.dumps(request))
        self.assertNotIn(str(self.root), json.dumps(request))
        response = response_for(request)
        response["model_identity"]["model"] = "claimed-name-is-not-attestation"
        response["usage"]["generated_tokens"] = 0
        feedback = controller.import_response(response)
        self.assertFalse(feedback["editor_identity_authenticated"])
        self.assertFalse(feedback["scientific_result"])
        snapshot = controller.status()["budget"]
        self.assertEqual(snapshot["charged"]["generated_tokens"], 12)
        self.assertIsNone(snapshot["observed"]["generated_tokens"])
        self.assertEqual(snapshot["charged"]["completed_sequences"], 1)
        with self.assertRaises(ValueError):
            controller.import_response(response)

    def test_no_live_dispatch_or_silent_substitution(self):
        controller = self.create()
        with self.assertRaisesRegex(ValueError, "live Astra access pending"):
            controller.dispatch_live(model="anything")
        request = controller.export_request("Offline only", max_output_tokens=10, timeout_seconds=10)
        response = response_for(request)
        response["evidence_kind"] = "live_model"
        with self.assertRaises(ValueError):
            controller.import_response(response)
        self.assertEqual(controller.status()["budget"]["charged"]["failed_sequences"], 1)

    def test_test_data_and_tools_never_enter_the_editor_contract(self):
        self.development[0]["split"] = "final-test"
        with self.assertRaises(ValueError):
            self.create()
        self.development[0]["split"] = "development"
        controller = self.create()
        for tool in ("read_file", "network", "train", "gradient", "optimizer", "change_architecture", "evaluate_final"):
            request = controller.export_request("Invalid tool check", max_output_tokens=1, timeout_seconds=1)
            with self.assertRaises(ValueError):
                controller.import_response(response_for(request, tool, {"path": "/any/test/results"}))
        self.assertEqual(controller.status()["budget"]["charged"]["failed_sequences"], 7)

    def test_metadata_is_projected_and_inspection_budget_persists(self):
        controller = self.create()
        _, feedback = self.exchange(controller, "inspect", {"op": "metadata", "tensors": [EDITABLE[0]]})
        info = feedback["result"]["tensors"][EDITABLE[0]]
        self.assertEqual(set(info), {"shape", "dtype", "parameter_count"})
        reopened = EditorController(controller.root)
        self.addCleanup(reopened.ledger.close)
        self.assertEqual(len(reopened.editor.exposure_snapshot()), 2)
        _, feedback = self.exchange(reopened, "inspect", {"op": "slice", "tensor": "../../final", "slices": [[0, 1]]})
        self.assertEqual(feedback["status"], "rejected")

    def test_valid_and_invalid_proposals_do_not_change_architecture_or_evaluation(self):
        controller = self.create()
        proposal = self.proposal(controller)
        request, feedback = self.exchange(controller, "propose", {"proposal": proposal})
        self.assertEqual(feedback["status"], "ok")
        saved = controller.root / "candidates" / request["request_id"] / "model"
        self.assertEqual((saved / "config.json").read_bytes(), (self.model / "config.json").read_bytes())
        for mutation in ("training", "architecture", "evaluation"):
            invalid = copy.deepcopy(proposal)
            invalid[mutation] = {"enabled": True}
            req, rejected = self.exchange(controller, "propose", {"proposal": invalid})
            self.assertEqual(rejected["status"], "rejected")
            self.assertFalse((controller.root / "candidates" / req["request_id"]).exists())

    def test_restart_recovery_consumes_reservation_and_records_retry(self):
        self.limits["editor_requests"] = 2
        controller = self.create()
        request = controller.export_request("Interrupted", max_output_tokens=20, timeout_seconds=10)
        reopened = EditorController(controller.root)
        self.addCleanup(reopened.ledger.close)
        recovered = reopened.recover()
        self.assertEqual(recovered["budget"]["charged"]["generated_tokens"], 20)
        self.assertEqual(recovered["budget"]["charged"]["failed_sequences"], 1)
        retry = reopened.export_request("Retry", max_output_tokens=20, timeout_seconds=10, retry_of=request["request_id"])
        reopened.import_response(response_for(retry))
        with self.assertRaises(BudgetExhausted):
            reopened.export_request("No budget remains", max_output_tokens=1, timeout_seconds=1)
        self.assertEqual(reopened.status()["budget"]["charged"]["retries"], 1)
        with self.assertRaises(ValueError):
            reopened.import_response(response_for(request))

    def test_missing_ledger_and_changed_session_cannot_reset(self):
        controller = self.create()
        controller.ledger.close()
        (controller.root / "budget.sqlite3").unlink()
        with self.assertRaises(ValueError):
            EditorController(controller.root)
        with self.assertRaises(ValueError):
            self.create()

    def test_probe_uses_only_frozen_development_schedule_and_counts_both_kinds(self):
        controller = self.create(probe=self.probe_spec())
        _, feedback = self.exchange(controller, "probe", {"task_ids": ["authored-dev"], "candidate_id": "base"})
        result = feedback["result"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evidence_kind"], "fixture")
        self.assertEqual(result["outputs"][0]["seed"], 17)
        totals = controller.status()["budget"]["charged"]
        self.assertEqual((totals["editor_requests"], totals["model_probes"], totals["requested_sequences"]), (1, 1, 2))
        with self.assertRaises(ValueError):
            self.exchange(controller, "probe", {"task_ids": ["final-test-id"], "candidate_id": "base"})
        self.assertEqual(controller.status()["budget"]["charged"]["model_probes"], 1)

    def test_probe_exhaustion_denies_dispatch_and_timeout_retains_unknown_usage(self):
        self.limits["model_probes"] = 1
        controller = self.create(probe=self.probe_spec(behavior="timeout", delay=5, timeout=1))
        started = time.monotonic()
        _, feedback = self.exchange(controller, "probe", {"task_ids": ["authored-dev"], "candidate_id": "base"})
        self.assertLess(time.monotonic() - started, 4)
        self.assertNotEqual(feedback["result"]["status"], "completed")
        self.assertEqual(controller.status()["budget"]["charged"]["failed_sequences"], 2)
        self.assertTrue(controller.status()["budget"]["cancelled"])
        # Measured teardown overrun is retained and cancels future execution.
        with self.assertRaises(ValueError):
            self.exchange(controller, "probe", {"task_ids": ["authored-dev"], "candidate_id": "base"})
        self.assertEqual(controller.status()["budget"]["charged"]["model_probes"], 1)

    def test_zero_probe_budget_rejects_before_worker_dispatch(self):
        self.limits["model_probes"] = 0
        controller = self.create(probe=self.probe_spec())
        _, rejected = self.exchange(controller, "probe", {"task_ids": ["authored-dev"], "candidate_id": "base"})
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(controller.status()["budget"]["charged"]["model_probes"], 0)

    def test_active_probe_cancellation_stops_worker_and_persists(self):
        controller = self.create(probe=self.probe_spec(behavior="timeout", delay=10, timeout=10))
        errors = []
        def cancel():
            try:
                time.sleep(0.25)
                other = EditorController(controller.root)
                other.cancel()
                other.ledger.close()
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=cancel)
        thread.start()
        started = time.monotonic()
        _, feedback = self.exchange(controller, "probe", {"task_ids": ["authored-dev"], "candidate_id": "base"})
        thread.join()
        self.assertEqual(errors, [])
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(feedback["result"]["status"], "cancelled")
        reopened = EditorController(controller.root)
        self.addCleanup(reopened.ledger.close)
        with self.assertRaises(ValueError):
            reopened.export_request("Cancelled", max_output_tokens=1, timeout_seconds=1)

    def test_candidate_path_traversal_and_extra_probe_parameters_rejected(self):
        controller = self.create(probe=self.probe_spec())
        for args in ({"task_ids": ["authored-dev"], "candidate_id": "../final"},
                     {"task_ids": ["authored-dev"], "candidate_id": "base", "seed": 100},
                     {"task_ids": ["authored-dev"], "candidate_id": "base", "max_new_tokens": 100000}):
            request = controller.export_request("Invalid probe", max_output_tokens=10, timeout_seconds=10)
            with self.assertRaises(ValueError):
                controller.import_response(response_for(request, "probe", args))

    def test_isolated_simulator_round_trip_is_non_scientific(self):
        from weight_space_lab.editor_sandbox import SandboxUnavailable
        controller = self.create()
        try:
            result = controller.simulate(tool="inspect", arguments={"op": "metadata", "tensors": [EDITABLE[0]]})
        except SandboxUnavailable as error:
            # The same test runs without a skip in the separately recorded host suite.
            self.assertEqual(controller.status()["budget"]["charged"]["failed_sequences"], 1)
            self.skipTest(str(error))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evidence_kind"], "simulated")
        self.assertFalse(result["scientific_result"])
        self.assertFalse(result["editor_identity_authenticated"])
        self.assertEqual(controller.status()["budget"]["observed"]["generated_tokens"], 0)
        self.assertGreater(controller.status()["budget"]["observed"]["elapsed_seconds"], 0)

    def test_cli_uses_persisted_session_and_rejects_invalid_response(self):
        from weight_space_lab.__main__ import main
        controller = self.create()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["editor-export", "--root", str(controller.root),
                                   "--instruction", "Offline CLI contract check"]), 0)
        exported = json.loads(output.getvalue())
        request = json.loads((controller.root / exported["handoff_file"]).read_text())
        response = response_for(request, "read_file", {"path": "/forbidden"})
        response_path = self.root / "invalid-response.json"
        response_path.write_text(json.dumps(response))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["editor-import", "--root", str(controller.root),
                                   "--response", str(response_path)]), 2)
        self.assertEqual(controller.status()["budget"]["charged"]["failed_sequences"], 1)


if __name__ == "__main__":
    unittest.main()
