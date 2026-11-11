"""No-model tests of the new diagnostic and directional intervention boundaries."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from test_checkpoint import checkpoint_fixture, fixture_policy, EDITABLE, files, tensor_bytes
from weight_space_lab.inventory import file_sha256, inventory_model, read_json
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.mechanistic_tools import (
    _capture_with_hooks, _sealed, _tokenize, apply_proposal, build_direction_proposal,
    capture_activations, contrast_direction, derive_directions, inspect_weights,
    policy_for_inventory, validate_capture_request,
)


def request(source="0" * 64):
    return {"schema_version": 1, "source_sha256": source, "layers": [0],
            "tail_tokens": 2, "max_tokens": 32, "cpu_threads": 1,
            "texts": [{"id": f"case-{condition}", "pair_id": "case", "condition": condition,
                       "text": "rendered diagnostic context", "token_span": None}
                      for condition in ("reference", "failure")]}


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModule:
    def __init__(self):
        self.hooks, self.prehooks = [], []

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return SimpleNamespace(remove=lambda: self.hooks.remove(callback))

    def register_forward_pre_hook(self, callback):
        self.prehooks.append(callback)
        return SimpleNamespace(remove=lambda: self.prehooks.remove(callback))


class MechanisticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = checkpoint_fixture(self.root)
        self.policy = fixture_policy()
        self.policy["direction_strength_range"] = [0, 6]
        self.policy["max_abs_change"] = 30
        self.source = canonical_hash(inventory_model(self.model)["files"])

    def capture_fixture(self):
        directory = self.root / "capture"
        directory.mkdir()
        arrays = {"residual_L0_T0": np.array([[0, 0], [0, 0]], dtype=np.float32),
                  "residual_L0_T1": np.array([[2, 0], [2, 0]], dtype=np.float32),
                  "mlp_L0_T0": np.array([[0, 0], [0, 0]], dtype=np.float32),
                  "mlp_L0_T1": np.array([[0, 1], [0, 1]], dtype=np.float32)}
        np.savez(directory / "activations.npz", **arrays)
        payload = {"schema_version": 1, "kind": "mechanistic_activation_capture",
                   "source_sha256": self.source, "layers": [0],
                   "records": [{"index": i, "pair_id": "case", "condition": condition}
                               for i, condition in enumerate(("reference", "failure"))],
                   "arrays_sha256": file_sha256(directory / "activations.npz")}
        (directory / "manifest.json").write_text(json.dumps(_sealed(payload)))
        return directory

    def test_raw_weight_inspection_is_requested_and_journaled(self):
        result = inspect_weights(self.model, self.policy,
            {"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 2], [0, 1]]},
            self.root / "exposure.jsonl")
        self.assertEqual(result["payload"]["response"]["values"], [[1], [3]])
        self.assertEqual(result["payload"]["source_sha256"], self.source)
        self.assertEqual(result["sha256"], canonical_hash(result["payload"]))
        self.assertEqual(len((self.root / "exposure.jsonl").read_text().splitlines()), 2)

    def test_inventory_expands_reads_without_mutation_rights(self):
        template = copy.deepcopy(self.policy)
        template["allowed_tensors"] = [EDITABLE[0]]
        template["allowed_columns"] = {EDITABLE[0]: "all"}
        template["inspection"]["allowed_tensors"] = [EDITABLE[0]]
        resolved = policy_for_inventory(template, inventory_model(self.model))
        self.assertEqual(resolved["inspection"]["allowed_tensors"], sorted(inventory_model(self.model)["tensors"]))
        self.assertEqual(template["inspection"]["allowed_tensors"], [EDITABLE[0]])
        result = inspect_weights(self.model, resolved,
            {"op": "slice", "tensor": "vector", "slices": [[0, 2]]}, self.root / "reads.jsonl")
        self.assertEqual(result["payload"]["response"]["values"], [10, 20])
        proposal = {"schema_version": 1, "base_sha256": self.source, "provenance": "editor_proposal",
                    "composition": "independent", "operations": [
                        {"op": "scale_columns", "tensor": EDITABLE[2], "columns": [0], "factor": 0.5}]}
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            apply_proposal(self.model, resolved, proposal, self.root / "rejected", self.root / "reads.jsonl")

    def test_directions_use_activation_contrast_not_weight_norm(self):
        direction, scores, consistency = contrast_direction(
            [[0, 0], [0, 0]], [[2, 0], [4, 0]],
            [[0, 0, 0], [0, 0, 0]], [[2, 0, 1], [2, 0, 1]],
            [[1, 100, 3], [0, 500, 0]])
        np.testing.assert_allclose(direction, [1, 0])
        np.testing.assert_allclose(scores, [2, 0, 3])
        np.testing.assert_array_equal(consistency, [2, 0, 2])

    def test_zero_nonfinite_and_dimension_mismatches_rejected(self):
        for values in (
            ([[0]], [[0]], [[0]], [[1]], [[1]]),
            ([[0]], [[float("nan")]], [[0]], [[1]], [[1]]),
            ([[0, 0]], [[1, 0]], [[0]], [[1]], [[1]]),
        ):
            with self.assertRaises(ValueError):
                contrast_direction(*values)

    def test_capture_request_pair_budget_and_unknown_fields(self):
        validate_capture_request(request())
        mutations = [lambda r: r.update(extra=True),
                     lambda r: r["texts"].pop(),
                     lambda r: r.update(layers=[0, 0]),
                     lambda r: r.update(source_sha256="unbound"),
                     lambda r: r["texts"][0].update(token_span=[0, 33]),
                     lambda r: r["texts"][0].update(condition="arbitrary")]
        for mutate in mutations:
            r = request()
            mutate(r)
            with self.assertRaises(ValueError):
                validate_capture_request(r)
        with self.assertRaises(ValueError):
            validate_capture_request(request(), {"max_text_bytes": 2})

    def test_tokenization_never_truncates_and_records_exact_selected_span(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [1, 2, 3])
        rows = _tokenize(tokenizer, request())
        self.assertEqual(rows[0]["token_span"], [1, 3])
        self.assertEqual(rows[0]["token_ids"], [1, 2, 3])
        short = request()
        short["max_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "truncation"):
            _tokenize(tokenizer, short)
        invalid = request()
        invalid["texts"][0]["token_span"] = [0, 4]
        with self.assertRaisesRegex(ValueError, "actual input"):
            _tokenize(tokenizer, invalid)

    def test_hook_capture_observes_preprojection_and_residual_and_cleans_up(self):
        block = FakeModule()
        down = FakeModule()
        block.mlp = SimpleNamespace(down_proj=down)

        class Backbone:
            layers = [block]

            def __call__(self, *, input_ids, use_cache):
                self_test.assertFalse(use_cache)
                value = FakeTensor(np.ones((1, 3, 2)))
                for hook in down.prehooks:
                    hook(down, (value,))
                for hook in block.hooks:
                    hook(block, (), FakeTensor(value.value * 3))

        self_test = self
        model = SimpleNamespace(model=SimpleNamespace(language_model=Backbone()))
        torch = SimpleNamespace(long="long", tensor=lambda value, **kwargs: FakeTensor(value),
                                inference_mode=contextlib.nullcontext)
        records = [{"index": 0, "token_span": [1, 3], "token_count": 3, "token_ids": [1, 2, 3]}]
        arrays = _capture_with_hooks(model, records, [0], {0: [2, 2]}, torch)
        np.testing.assert_array_equal(arrays["residual_L0_T0"], np.full((2, 2), 3))
        np.testing.assert_array_equal(arrays["mlp_L0_T0"], np.ones((2, 2)))
        self.assertEqual(block.hooks, [])
        self.assertEqual(down.prehooks, [])
        records[0]["token_count"] = 4
        with self.assertRaisesRegex(ValueError, "shape"):
            _capture_with_hooks(model, records, [0], {0: [2, 2]}, torch)
        self.assertEqual(block.hooks + down.prehooks, [])

    def test_derive_and_strength_six_edit_preserve_orthogonal_component_and_original(self):
        original = files(self.model)
        capture = self.capture_fixture()
        result = derive_directions(self.model, capture, self.root / "directions.json", self.policy)
        layer = result["payload"]["layers"][0]
        self.assertEqual(layer["ranked_columns"][0]["column"], 1)
        np.testing.assert_allclose(layer["direction"], [1, 0])
        proposal = build_direction_proposal(self.model, self.policy, self.root / "directions.json",
            [{"layer": 0, "columns": [1], "strength": 6}], self.root / "proposal.json")
        binding = read_json(self.root / "proposal.json.evidence.json")
        self.assertEqual(binding["payload"]["proposal_sha256"], canonical_hash(proposal))
        audit = apply_proposal(self.model, self.policy, proposal, self.root / "candidate", self.root / "log.jsonl")
        self.assertTrue(audit["complete"])
        raw = np.frombuffer(tensor_bytes(self.root / "candidate" / "model", EDITABLE[0]), dtype="<u2")
        from weight_space_lab.checkpoint_numeric import decode
        np.testing.assert_array_equal(decode(raw, "BF16").reshape(2, 2), [[1, -10], [3, 4]])
        self.assertEqual(files(self.model), original)

    def test_direction_and_activation_tampering_rejected(self):
        capture = self.capture_fixture()
        with (capture / "activations.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "bytes"):
            derive_directions(self.model, capture, self.root / "bad.json", self.policy)
        self.assertFalse((self.root / "bad.json").exists())

    def test_oversized_activation_header_rejected_before_numpy_load(self):
        capture = self.capture_fixture()
        with np.load(capture / "activations.npz") as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["residual_L0_T0"] = np.zeros((33, 2), dtype=np.float32)
        np.savez(capture / "activations.npz", **arrays)
        payload = read_json(capture / "manifest.json")["payload"]
        payload["arrays_sha256"] = file_sha256(capture / "activations.npz")
        (capture / "manifest.json").write_text(json.dumps(_sealed(payload)))
        with patch("weight_space_lab.mechanistic_tools.np.load") as load:
            with self.assertRaisesRegex(ValueError, "header"):
                derive_directions(self.model, capture, self.root / "bad.json", self.policy)
            load.assert_not_called()

    def test_dose_and_model_binding_rejected_before_output(self):
        derive_directions(self.model, self.capture_fixture(), self.root / "directions.json", self.policy)
        for strength in (6.1, -1, float("nan")):
            with self.assertRaises(ValueError):
                build_direction_proposal(self.model, self.policy, self.root / "directions.json",
                    [{"layer": 0, "columns": [1], "strength": strength}], self.root / "bad-proposal.json")
        self.assertFalse((self.root / "bad-proposal.json").exists())

    def test_worker_rss_limit_kills_group_and_never_publishes(self):
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        with patch("weight_space_lab.mechanistic_tools.subprocess.Popen", return_value=process), \
             patch("weight_space_lab.mechanistic_tools._rss_bytes", return_value=101), \
             patch("weight_space_lab.mechanistic_tools.os.killpg") as kill:
            with self.assertRaises(MemoryError):
                capture_activations(self.model, request(self.source), self.root / "capture", {"max_rss_bytes": 100})
        kill.assert_called_once()
        process.wait.assert_called_once()
        self.assertFalse((self.root / "capture").exists())
        self.assertEqual(list(self.root.glob(".capture-capture-*")), [])

    def test_worker_wall_limit_kills_group_and_never_publishes(self):
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        with patch("weight_space_lab.mechanistic_tools.subprocess.Popen", return_value=process), \
             patch("weight_space_lab.mechanistic_tools.time.monotonic", side_effect=[0, 2]), \
             patch("weight_space_lab.mechanistic_tools.os.killpg") as kill:
            with self.assertRaises(TimeoutError):
                capture_activations(self.model, request(self.source), self.root / "capture", {"wall_seconds": 1})
        kill.assert_called_once()
        self.assertFalse((self.root / "capture").exists())


if __name__ == "__main__":
    unittest.main()
