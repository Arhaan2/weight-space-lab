"""Synthetic source/deployment audits for general edits; no target inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_inventory import write_shard
from weight_space_lab.inventory import inventory_model
from weight_space_lab.checkpoint_numeric import encode
from weight_space_lab.research_checkpoint import EditWorkspace, materialize, verify_materialized, verify_snapshot
from weight_space_lab.research_deployment import audit_effective_pair, mapped_sources


P = "model.language_model.layers.3."
Q = P + "self_attn.q_proj.weight"
K = P + "self_attn.k_proj.weight"
V = P + "self_attn.v_proj.weight"
N = P + "input_layernorm.weight"
G = P + "mlp.gate_proj.weight"
U = P + "mlp.up_proj.weight"
D = P + "mlp.down_proj.weight"
B = "model.language_model.layers.0.linear_attn.out_proj.weight"
C = "model.language_model.layers.0.linear_attn.conv1d.weight"
LN = "model.language_model.layers.0.linear_attn.norm.weight"


def fixture(root):
    root.mkdir()
    for name, value in [("config.json", {"model_type": "fixture"}),
                        ("tokenizer_config.json", {"chat_template": "{{ messages }}"}),
                        ("tokenizer.json", {"fixture": True})]:
        (root / name).write_text(json.dumps(value))
    tensors = []
    for name in (Q, K, V, G, U, D):
        data = (np.arange(64).reshape(2, 32) + 1).astype("<f4")
        tensors.append((name, "F32", [2, 32], data.tobytes()))
    for name in (N, "model.language_model.norm.weight", "model.language_model.embed_tokens.weight"):
        tensors.append((name, "F32", [2], np.array([1, 2], dtype="<f4").tobytes()))
    tensors.append((B, "BF16", [2, 32], encode(np.arange(64).reshape(2, 32) + 1, "BF16").tobytes()))
    tensors.append((C, "F32", [2, 1, 4], np.arange(8, dtype="<f4").reshape(2, 1, 4).tobytes()))
    tensors.append((LN, "F32", [2], np.array([1, 2], dtype="<f4").tobytes()))
    write_shard(root / "model.safetensors", tensors)
    return root


def cache(root, *, weight=None, scale=None, norm=None, gate_weight=None, convolution=None, linear_norm=None):
    root.mkdir()
    arrays = {
        "model.layers.3.self_attn.c_attn.q_weight": np.full((6, 4), 0x77777777, dtype="<u4") if weight is None else weight,
        "model.layers.3.self_attn.c_attn.q_scale": np.ones((6, 1), dtype="<f2") if scale is None else scale,
        "model.layers.3.input_layernorm.weight": np.array([2, 3], dtype="<f2") if norm is None else norm,
        "model.layers.3.mlp.gate_up_proj.q_weight": np.full((4, 4), 0x77777777, dtype="<u4") if gate_weight is None else gate_weight,
        "model.layers.3.mlp.gate_up_proj.q_scale": np.ones((4, 1), dtype="<f2"),
        "model.layers.3.mlp.down_proj.q_weight": np.full((2, 4), 0x77777777, dtype="<u4"),
        "model.layers.3.mlp.down_proj.q_scale": np.ones((2, 1), dtype="<f2"),
        "model.layers.0.linear_attn.out_proj.q_weight": np.full((2, 4), 0x77777777, dtype="<u4"),
        "model.layers.0.linear_attn.out_proj.q_scale": np.ones((2, 1), dtype="<f2"),
        "model.layers.0.linear_attn.conv1d_weight": np.arange(8, dtype="<f2").reshape(2, 1, 4) if convolution is None else convolution,
        "model.layers.0.linear_attn.norm.weight": np.array([1, 2], dtype="<f2") if linear_norm is None else linear_norm,
    }
    raw, records = bytearray(), []
    for name, value in arrays.items():
        data = value.tobytes()
        records.append({"name": name, "shape": list(value.shape), "dtype": value.dtype.name,
                        "format": "raw", "byteOffset": len(raw), "nbytes": len(data)})
        raw.extend(data)
    (root / "data.bin").write_bytes(raw)
    (root / "tensor-cache.json").write_text(json.dumps({"metadata": {"ParamSize": len(arrays)},
        "records": [{"dataPath": "data.bin", "nbytes": len(raw), "md5sum": hashlib.md5(raw).hexdigest(), "records": records}]}))
    return root


class ResearchCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = fixture(self.root / "source")
        self.original = inventory_model(self.source)
        self.work = EditWorkspace(self.source, self.root / "work")

    def changed(self, name, amount=3):
        array = self.work.load_original(name)
        array.flat[0] += amount
        return array

    def bundle(self, edits=None):
        for name in (edits or [Q]):
            self.work.save_edit(name, self.changed(name))
        self.work.snapshot(self.root / "snap")
        materialize(self.source, self.root / "snap", self.root / "bundle")
        return self.root / "bundle"

    def test_general_attention_mlp_norm_edits_snapshot_restore_and_saved_bytes(self):
        for name in (Q, G, N):
            self.work.save_edit(name, self.changed(name))
        first = self.work.snapshot(self.root / "first")
        self.work.drop_edit(Q)
        self.work.save_edit(V, self.changed(V))
        self.work.restore(self.root / "first")
        self.assertEqual(self.work.edited(), sorted([Q, G, N]))
        np.testing.assert_array_equal(self.work.load(Q), self.changed(Q))
        self.assertEqual(verify_snapshot(self.source, self.root / "first"), first)
        audit = materialize(self.source, self.root / "first", self.root / "candidate")
        self.assertEqual(verify_materialized(self.source, self.root / "candidate"), audit)
        self.assertEqual(audit["changed_stored_entries"], 3)
        self.assertEqual(set(audit["per_tensor"]), {Q, G, N})
        self.assertEqual(inventory_model(self.source), self.original)
        with self.assertRaises(FileExistsError):
            self.work.snapshot(self.root / "first")

    def test_scope_scalar_shape_and_nonfinite_rejections(self):
        for name in ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight"):
            with self.assertRaisesRegex(ValueError, "internal"):
                self.work.save_edit(name, self.changed(name))
        with self.assertRaisesRegex(ValueError, "scalar"):
            self.work.save_edit(Q, self.work.load_original(Q) * 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.work.save_edit(Q, np.ones((32, 2), np.float32))
        value = self.changed(Q)
        value.flat[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            self.work.save_edit(Q, value)
        self.assertEqual(self.work.edited(), [])

    def test_head_ablation_allowed_and_load_is_independent(self):
        value = self.work.load(Q)
        value[0] = 0
        self.work.save_edit(Q, value)
        exposed = self.work.load(Q)
        exposed[:] = 999
        np.testing.assert_array_equal(self.work.load(Q), value)
        self.assertEqual(inventory_model(self.source), self.original)

    def test_snapshot_and_materialized_tampering_rejected(self):
        bundle = self.bundle()
        tokenizer = bundle / "model/tokenizer.json"
        tokenizer.chmod(0o600)
        tokenizer.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "unrelated file"):
            verify_materialized(self.source, bundle)
        snapshot = self.root / "snap"
        array = next((snapshot / "edits").iterdir())
        array.chmod(0o600)
        array.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "identity"):
            verify_snapshot(self.source, snapshot)

    def test_source_tampering_detected_even_before_freeze(self):
        path = self.source / "model.safetensors"
        data = bytearray(path.read_bytes())
        tensor = self.original["tensors"][Q]
        data[tensor["data_start"] + tensor["data_offsets"][0]] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "original tensor"):
            self.work.load_original(Q)

    def test_fused_attention_rows_and_quantization_spillover_audited(self):
        bundle = self.bundle()
        weight = np.full((6, 4), 0x77777777, dtype="<u4")
        weight[0] = 0x88888888
        base = cache(self.root / "base", weight=weight)
        scale = np.ones((6, 1), dtype="<f2")
        scale[0, 0] = 2
        candidate = cache(self.root / "deployed", weight=weight, scale=scale)
        audit = audit_effective_pair(self.source, bundle, base, candidate)
        self.assertEqual(audit["decoded_values_changed"], 32)
        self.assertEqual(audit["decoded_changed_outside_source_coordinates"], 31)
        self.assertFalse(audit["conversion_erases_edit"])
        wrong = weight.copy()
        wrong[2, 0] = 0x77777778  # K row cannot change when only Q was edited.
        candidate2 = cache(self.root / "bad", weight=wrong)
        with self.assertRaisesRegex(ValueError, "fused rows"):
            audit_effective_pair(self.source, bundle, base, candidate2)

    def test_fused_mlp_and_offset_norm_supported(self):
        bundle = self.bundle([G, N])
        base = cache(self.root / "base")
        weight = np.full((4, 4), 0x77777777, dtype="<u4")
        weight[0, 0] = 0x77777778
        candidate = cache(self.root / "deployed", gate_weight=weight, norm=np.array([5, 3], dtype="<f2"))
        audit = audit_effective_pair(self.source, bundle, base, candidate)
        self.assertEqual(audit["decoded_values_changed"], 2)
        self.assertEqual(audit["per_deployed_tensor"]["model.layers.3.input_layernorm.weight"]["loader_operation"], "add_one_then_cast")
        self.assertEqual(mapped_sources("model.layers.0.linear_attn.conv1d_weight"),
                         (["model.language_model.layers.0.linear_attn.conv1d.weight"], "direct"))
        self.assertEqual(mapped_sources("model.layers.0.linear_attn.A_log"),
                         (["model.language_model.layers.0.linear_attn.A_log"], "direct"))

    def test_unrelated_deployed_tensor_change_rejected(self):
        bundle = self.bundle()
        base = cache(self.root / "base")
        candidate = cache(self.root / "bad", norm=np.array([3, 3], dtype="<f2"))
        with self.assertRaisesRegex(ValueError, "unrelated deployed"):
            audit_effective_pair(self.source, bundle, base, candidate)

    def test_bf16_submission_rounding_and_multidimensional_linear_attention(self):
        value = self.work.load_original(B)
        value[0, 0] = np.float32(1 + 2 ** -8)  # BF16 tie rounds to even 1.0.
        value[0, 1] = np.float32(2.02)
        self.work.save_edit(B, value)
        np.testing.assert_array_equal(self.work.load(B), value)  # Notebook keeps F32 precision.
        for name in (C, LN):
            self.work.save_edit(name, self.changed(name))
        self.work.snapshot(self.root / "snap")
        audit = materialize(self.source, self.root / "snap", self.root / "bundle")
        self.assertEqual(audit["per_tensor"][B]["changed_stored_entries"], 1)
        base = cache(self.root / "base")
        conv = np.arange(8, dtype="<f2").reshape(2, 1, 4)
        conv[0, 0, 0] = 3
        candidate = cache(self.root / "deployed", convolution=conv, linear_norm=np.array([4, 2], dtype="<f2"))
        deployed = audit_effective_pair(self.source, self.root / "bundle", base, candidate)
        self.assertEqual(deployed["decoded_values_changed"], 2)
        self.assertEqual(deployed["per_deployed_tensor"]["model.layers.0.linear_attn.norm.weight"]["loader_operation"], "direct")
        conv[1, 0, 1] += 1
        bad = cache(self.root / "bad-conv", convolution=conv)
        with self.assertRaisesRegex(ValueError, "unquantized edit escaped"):
            audit_effective_pair(self.source, self.root / "bundle", base, bad)


if __name__ == "__main__":
    unittest.main()
