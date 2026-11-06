import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from weight_space_lab.replay import apply_sparse_bf16, replay_sparse_checkpoint


# Independently specified BF16 bytes: [1, -2, +0, -0] -> [0.5, -2, -0, -0].
BASE_BYTES = bytes.fromhex("803f00c000000080")
EDITED_BYTES = bytes.fromhex("003f00c000800080")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.base = np.frombuffer(BASE_BYTES, dtype="<u2").reshape(2, 2)
        self.indices = np.array([0, 1, 2], dtype=np.int64)
        self.values = np.array([0x3F00, 0xC000, 0x8000], dtype="<u2")
        self.spec = {
            "name": "layer.weight", "shape": [2, 2], "dtype": "bfloat16",
            "form": "sparse", "changed_entries": 2,
            "base_sha256": hashlib.sha256(BASE_BYTES).hexdigest(),
            "edited_sha256": hashlib.sha256(EDITED_BYTES).hexdigest(),
        }

    def test_known_bits_signed_zero_and_unchanged_assignment(self):
        result, audit = apply_sparse_bf16(self.base, self.spec, self.indices, self.values)
        self.assertEqual(result.tobytes(), EDITED_BYTES)
        self.assertEqual(audit["changed_entries"], 2)
        self.assertEqual(self.base.tobytes(), BASE_BYTES)
        self.assertFalse(np.shares_memory(result, self.base))
        result[0, 0] = 0
        self.assertEqual(self.base.tobytes(), BASE_BYTES)

    def test_hashes_counts_and_schema_are_checked(self):
        for key, value in [("base_sha256", "0" * 64), ("edited_sha256", "0" * 64),
                           ("changed_entries", 3), ("changed_entries", True),
                           ("shape", [4]), ("shape", [True, 2]),
                           ("form", "lowrank"), ("dtype", "float16")]:
            with self.subTest(key=key, value=value):
                spec = {**self.spec, key: value}
                with self.assertRaises(ValueError):
                    apply_sparse_bf16(self.base, spec, self.indices, self.values)
                self.assertEqual(self.base.tobytes(), BASE_BYTES)

    def test_invalid_indices_and_values(self):
        invalid_indices = [np.array([0, 0, 2]), np.array([-1, 1, 2]),
                           np.array([0, 1, 4]), np.array([0., 1., 2.]),
                           np.array([False, True, True]), np.array([[0, 1, 2]])]
        for indices in invalid_indices:
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    apply_sparse_bf16(self.base, self.spec, indices, self.values)
        for values in [self.values.astype(np.float32), self.values[:2],
                       self.values.reshape(1, 3), self.values.astype(">u2")]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    apply_sparse_bf16(self.base, self.spec, self.indices, values)
        with self.assertRaises(ValueError):
            apply_sparse_bf16(self.base.astype(np.float32), self.spec, self.indices, self.values)

    def fixture(self, directory):
        root = Path(directory)
        header = {
            "__metadata__": {"format": "pt", "test": "preserve metadata"},
            "layer.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
            "untouched": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        }
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        prefix = len(encoded).to_bytes(8, "little") + encoded
        untouched = bytes.fromhex("0000803f000000c0")
        base = root / "base.safetensors"
        base.write_bytes(prefix + BASE_BYTES + untouched)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"tensors": [self.spec], "changed_tensors": 1,
                                       "changed_entries_total": 2}))
        archive = root / "edit.npz"
        np.savez(archive, t0_idx=self.indices, t0_val=self.values)
        return base, manifest, archive, root / "edited.safetensors", prefix + EDITED_BYTES + untouched

    def test_checkpoint_preserves_all_other_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base, manifest, archive, output, expected = self.fixture(directory)
            output = Path(directory) / "new-checkpoint" / "edited.safetensors"
            original = base.read_bytes()
            audit = replay_sparse_checkpoint(base, manifest, archive, output)
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(base.read_bytes(), original)
            self.assertEqual(audit["changed_entries_total"], 2)

    def test_existing_output_and_base_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base, manifest, archive, output, _ = self.fixture(directory)
            output.write_bytes(b"keep")
            for destination in (base, output):
                with self.assertRaises(FileExistsError):
                    replay_sparse_checkpoint(base, manifest, archive, destination)
            self.assertEqual(output.read_bytes(), b"keep")

    def test_failed_write_removes_incomplete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base, manifest, archive, output, _ = self.fixture(directory)
            original = base.read_bytes()
            with patch("weight_space_lab.replay.shutil.copyfileobj", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    replay_sparse_checkpoint(base, manifest, archive, output)
            self.assertFalse(output.exists())
            self.assertEqual(base.read_bytes(), original)

    def test_bad_archive_and_object_array_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base, manifest, archive, output, _ = self.fixture(directory)
            cases = [{"t0_idx": self.indices, "t0_val": self.values, "extra": np.array([1])},
                     {"t0_idx": self.indices, "t0_val": np.array([1, 2, 3], dtype=object)},
                     {"t0_idx": self.indices, "t0_val": self.values ^ 1}]
            for arrays in cases:
                with self.subTest(arrays=list(arrays)):
                    np.savez(archive, **arrays)
                    with self.assertRaises(ValueError):
                        replay_sparse_checkpoint(base, manifest, archive, output)
                    self.assertFalse(output.exists())

    def test_manifest_totals_and_duplicate_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base, manifest, archive, output, _ = self.fixture(directory)
            valid = json.loads(manifest.read_text())
            cases = [{**valid, "changed_entries_total": 3}, {**valid, "changed_tensors": True},
                     {**valid, "tensors": [self.spec, copy.deepcopy(self.spec)], "changed_tensors": 2}]
            for changed in cases:
                manifest.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    replay_sparse_checkpoint(base, manifest, archive, output)
                self.assertFalse(output.exists())

    def test_truncated_or_overlapping_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base, manifest, archive, output, _ = self.fixture(directory)
            original = base.read_bytes()
            length = int.from_bytes(original[:8], "little")
            header = json.loads(original[8:8 + length])
            header["untouched"]["data_offsets"] = [0, 8]
            encoded = json.dumps(header).encode()
            invalid = len(encoded).to_bytes(8, "little") + encoded + original[8 + length:]
            for content in (original[:-1], invalid):
                base.write_bytes(content)
                with self.assertRaises(ValueError):
                    replay_sparse_checkpoint(base, manifest, archive, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
