import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_inventory import write_shard
from weight_space_lab.__main__ import main
from weight_space_lab.acquisition import _remove_stage
from weight_space_lab.candidate_demo import DEMO_TENSOR, DEVELOPMENT_INPUT, demo_candidate_edit
from weight_space_lab.inventory import file_sha256, inventory_model, read_json


def candidate_fixture(root):
    model = root / "model"
    model.mkdir()
    for name, value in (
        ("config.json", {"model_type": "synthetic"}),
        ("tokenizer_config.json", {"chat_template": "{{ messages }}"}),
        ("tokenizer.json", {"fixture": True}),
    ):
        (model / name).write_text(json.dumps(value))
    (model / "LICENSE").write_text("Synthetic candidate demo fixture.\n")
    write_shard(model / "model.safetensors", [
        (DEMO_TENSOR, "BF16", [4, 3], np.full((4, 3), 0x3F80, dtype="<u2").tobytes()),
        ("untouched.f32", "F32", [2], np.array([1, -2], dtype="<f4").tobytes()),
        ("untouched.buffer", "I64", [], np.array(7, dtype="<i8").tobytes()),
    ])
    return model


class CandidateDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.model = candidate_fixture(self.root)
        self.output = self.root / "demonstration"
        self.addCleanup(_remove_stage, self.output)

    def test_identity_edit_rejection_and_development_provenance(self):
        before = inventory_model(self.model)
        with patch("weight_space_lab.candidate_demo._development_generation") as inference:
            report = demo_candidate_edit(self.model, self.output)
            inference.assert_not_called()
        self.assertEqual(report["intervention_origin"], "hand_authored_engineering_edit")
        self.assertFalse(report["astra_discovered"])
        self.assertFalse(report["editor_called"])
        self.assertFalse(report["experiment_executed"])
        self.assertFalse(report["scientific_result"])
        self.assertFalse(report["inference_executed"])
        self.assertTrue(report["identity_verified"])
        self.assertTrue(report["valid_edit_verified"])
        self.assertTrue(report["rejected_edit_left_no_candidate"])
        self.assertEqual(inventory_model(self.model), before)
        self.assertEqual(inventory_model(self.output / "identity/model"), before)
        edited = inventory_model(self.output / "engineering/model")
        self.assertNotEqual(edited["tensors"][DEMO_TENSOR]["sha256"], before["tensors"][DEMO_TENSOR]["sha256"])
        self.assertEqual(edited["tensors"]["untouched.buffer"], before["tensors"]["untouched.buffer"])
        self.assertEqual(edited["tensors"]["untouched.f32"], before["tensors"]["untouched.f32"])
        self.assertEqual(read_json(self.output / "development-input.json"), DEVELOPMENT_INPUT)
        self.assertEqual(report["exposure_log_sha256"], file_sha256(self.output / "exposures.jsonl"))
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_native_smoke_is_explicit_and_never_scored(self):
        with patch("weight_space_lab.candidate_demo._development_generation", return_value={"status": "truncated"}) as generation:
            report = demo_candidate_edit(self.model, self.output, backend="transformers")
        self.assertEqual(generation.call_count, 2)
        for call in generation.call_args_list:
            self.assertEqual(call.args[2], DEVELOPMENT_INPUT)
            self.assertEqual(call.args[3], "transformers")
        self.assertEqual(report["generation_statuses"], {"identity": "truncated", "engineering": "truncated"})
        self.assertTrue(report["inference_executed"])
        self.assertFalse(report["scientific_result"])

    def test_failed_second_candidate_does_not_publish_demonstration(self):
        from weight_space_lab.checkpoint import CheckpointEditor
        original = CheckpointEditor.propose

        def fail_edit(editor, proposal, output):
            if proposal["operations"]:
                raise OSError("injected write failure")
            return original(editor, proposal, output)

        before = inventory_model(self.model)
        with patch.object(CheckpointEditor, "propose", fail_edit):
            with self.assertRaises(OSError):
                demo_candidate_edit(self.model, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".demonstration.incomplete-*")), [])
        self.assertEqual(inventory_model(self.model), before)

    def test_demo_cli_and_immutable_existing_destination(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["demo-candidate-edit", "--model", str(self.model),
                                   "--out", str(self.output)]), 0)
        self.assertFalse(json.loads(stream.getvalue())["astra_discovered"])
        digest = file_sha256(self.output / "report.json")
        with self.assertRaises(FileExistsError):
            demo_candidate_edit(self.model, self.output)
        self.assertEqual(file_sha256(self.output / "report.json"), digest)

    def test_verifier_cli_reopens_saved_candidate(self):
        demo_candidate_edit(self.model, self.output)
        arguments = ["verify-candidate", "--model", str(self.model),
                     "--candidate", str(self.output / "engineering"),
                     "--policy", str(self.output / "policy.json")]
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(arguments), 0)
        self.assertEqual(set(json.loads(stream.getvalue())), {"verified", "result_sha256", "audit_sha256"})
        changed = self.output / "engineering/model/LICENSE"
        changed.chmod(0o644)
        changed.write_text("tampered fixture asset")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 2)

    def test_edit_cli_only_exposes_receipt_and_rejects_bad_policy(self):
        demo_candidate_edit(self.model, self.output)
        candidate = self.root / "cli-candidate"
        self.addCleanup(_remove_stage, candidate)
        arguments = ["edit-checkpoint", "--model", str(self.model),
                     "--policy", str(self.output / "policy.json"),
                     "--log", str(self.output / "exposures.jsonl"),
                     "--out", str(candidate), "--patch", str(self.output / "engineering-proposal.json")]
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(arguments), 0)
        receipt = json.loads(stream.getvalue())
        self.assertEqual(set(receipt), {"accepted", "result_sha256", "audit_sha256"})
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["audit_sha256"], file_sha256(candidate / "audit.json"))
        rejected = self.root / "cli-rejected"
        arguments[arguments.index("--out") + 1] = str(rejected)
        arguments[-1] = str(self.output / "rejected-proposal.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 2)
        self.assertFalse(rejected.exists())

    def test_bad_backend_or_nested_output_does_not_mutate_model(self):
        before = inventory_model(self.model)
        with self.assertRaises(ValueError):
            demo_candidate_edit(self.model, self.output, backend="automatic")
        with self.assertRaises(ValueError):
            demo_candidate_edit(self.model, self.model / "nested")
        self.assertEqual(inventory_model(self.model), before)


if __name__ == "__main__":
    unittest.main()
