import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from weight_space_lab.__main__ import main
from weight_space_lab.acquisition import _freeze, _remove_stage, verify_acquisition
from weight_space_lab.inventory import inventory_model
from weight_space_lab.reconstruction import reconstruct_published, verify_reconstruction


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_shard(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def fixture(root):
    """Synthetic two-shard acquisition; remote identities are fixture-only."""
    bundle = root / "acquisition"
    model, patches = bundle / "model", bundle / "patch"
    model.mkdir(parents=True)
    patches.mkdir()
    first = np.full((4, 3), 0x3F80, dtype="<u2")
    second = np.full((2, 3), 0x4000, dtype="<u2")
    first_indices, second_indices = np.array([1, 4, 7]), np.array([0, 4])
    first_values = np.array([0x3F00, 0x4000, 0x3F80], dtype="<u2")
    second_values = np.array([0xBF80, 0x4000], dtype="<u2")
    first_edited, second_edited = first.copy(), second.copy()
    first_edited.flat[first_indices] = first_values
    second_edited.flat[second_indices] = second_values
    write_shard(model / "model-00001-of-00002.safetensors", [
        ("layer.first", "BF16", [4, 3], first.tobytes()),
        ("untouched.f32", "F32", [2], bytes.fromhex("0000803f000000c0")),
    ])
    write_shard(model / "model-00002-of-00002.safetensors", [
        ("untouched.i64", "I64", [], bytes.fromhex("0100000000000000")),
        ("layer.second", "BF16", [2, 3], second.tobytes()),
    ])
    write_json(model / "model.safetensors.index.json", {
        "metadata": {"total_size": 52}, "weight_map": {
            "layer.first": "model-00001-of-00002.safetensors",
            "untouched.f32": "model-00001-of-00002.safetensors",
            "layer.second": "model-00002-of-00002.safetensors",
            "untouched.i64": "model-00002-of-00002.safetensors"}})
    write_json(model / "config.json", {"model_type": "synthetic", "dtype": "bfloat16"})
    write_json(model / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
    write_json(model / "tokenizer.json", {"fixture": True})
    (model / "LICENSE").write_text("Synthetic fixture license.\n")
    specs = []
    for name, base, edited, changed in (("layer.first", first, first_edited, 2),
                                        ("layer.second", second, second_edited, 1)):
        specs.append({"name": name, "shape": list(base.shape), "dtype": "bfloat16",
                      "form": "sparse", "changed_entries": changed,
                      "base_sha256": digest(base.tobytes()),
                      "edited_sha256": digest(edited.tobytes())})
    # One full-column intervention has an omitted rounded-to-base coordinate.
    # Both tensors also contain an explicitly stored no-op substitution.
    specs[0].update(changed_columns=1, changed_rows=3)
    write_json(patches / "manifest.json", {"changed_tensors": 2,
                                           "changed_entries_total": 3, "tensors": specs})
    np.savez(patches / "edit.npz", t0_idx=first_indices, t0_val=first_values,
             t1_idx=second_indices, t1_val=second_values)
    reference = {
        "schema_version": 1, "claim_type": "published_patch_replay",
        "source_repository": "https://github.com/fixture/release",
        "source_revision": "1" * 40, "base_model": "fixture/synthetic-model",
        "base_revision": "2" * 40, "manifest_path": "release/manifest.json",
        "archive_path": "release/edit.npz", "tensor_names": ["layer.first", "layer.second"],
        "expected_changed_stored_entries": 3, "reported_targeted_entries": 6,
    }
    reference_path = root / "reference.json"
    write_json(reference_path, reference)
    write_json(bundle / "reference.json", reference)
    files, remote_files = {}, {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        name, data = path.relative_to(bundle).as_posix(), path.read_bytes()
        files[name] = {"sha256": digest(data), "size_bytes": len(data)}
        if name == "reference.json":
            continue
        remote_files[name] = {
            "algorithm": "git_sha1",
            "digest": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            "size_bytes": len(data),
            "source": "base_model" if name.startswith("model/") else "published_patch",
            "source_path": name.removeprefix("model/") if name.startswith("model/")
            else "release/" + path.name,
        }
    write_json(bundle / "acquisition.json", {
        "schema_version": 1, "kind": "pinned_acquisition", "complete": True,
        "resolved_source_revision": reference["source_revision"],
        "resolved_model_revision": reference["base_revision"],
        "reference_sha256": files["reference.json"]["sha256"],
        "model_inventory": inventory_model(model), "files": files, "remote_files": remote_files,
        "verification_scope": "Synthetic fixture only.",
    })
    verify_acquisition(bundle, reference_path)
    _freeze(bundle)
    return bundle, reference_path


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.bundle, self.reference = fixture(self.root)
        self.addCleanup(_remove_stage, self.bundle)
        self.output = self.root / "reconstructed"
        self.originals = {p.relative_to(self.bundle).as_posix(): p.read_bytes()
                          for p in self.bundle.rglob("*") if p.is_file()}

    def reconstruct(self):
        return reconstruct_published(self.bundle, self.output, self.reference)

    def assert_base_unchanged(self):
        self.assertEqual({p.relative_to(self.bundle).as_posix(): p.read_bytes()
                          for p in self.bundle.rglob("*") if p.is_file()}, self.originals)

    def test_two_shards_counts_saved_bytes_and_all_assets(self):
        audit = self.reconstruct()
        self.assertEqual(audit["changed_stored_entries"], 3)
        self.assertEqual(audit["sparse_substitutions"], 5)
        self.assertEqual(audit["targeted_coordinates"], 6)
        self.assertEqual(audit["targeted_but_unchanged_stored_entries"], 3)
        self.assertEqual(audit["unchanged_tensor_count"], 2)
        self.assertEqual(audit["tensors"][0]["targeted_columns"], [1])
        self.assertEqual(audit["tensors"][0]["targeted_coordinates"], 4)
        self.assertEqual(audit["tensors"][1]["targeted_coordinate_basis"], "archive_indices")
        self.assertFalse(audit["inference_executed"])
        self.assertFalse(audit["experiment_executed"])
        self.assertEqual(audit["artifact_verification"], "passed")
        self.assertEqual({p.name for p in self.output.iterdir()}, {"model", "audit.json"})
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE",
                     "model.safetensors.index.json"):
            self.assertEqual((self.output / "model" / name).read_bytes(),
                             (self.bundle / "model" / name).read_bytes())
        self.assert_base_unchanged()
        self.assertEqual(verify_reconstruction(self.bundle, self.output, self.reference), audit)

    def test_verifier_does_not_call_edit_implementation(self):
        self.reconstruct()
        with patch("weight_space_lab.reconstruction.apply_sparse_bf16",
                   side_effect=AssertionError("edit implementation used by verifier")):
            verify_reconstruction(self.bundle, self.output, self.reference)

    def test_saved_target_and_unrelated_tensors_are_independently_checked(self):
        audit = self.reconstruct()
        for name in ("layer.first", "layer.second", "untouched.f32", "untouched.i64"):
            with self.subTest(name=name):
                tensor = audit["model_inventory"]["tensors"][name]
                path = self.output / "model" / tensor["shard"]
                original = path.read_bytes()
                changed = bytearray(original)
                changed[tensor["data_start"] + tensor["data_offsets"][0]] ^= 1
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    verify_reconstruction(self.bundle, self.output, self.reference)
                path.write_bytes(original)
        self.assert_base_unchanged()

    def test_header_and_assets_are_independently_checked(self):
        self.reconstruct()
        shard = self.output / "model/model-00001-of-00002.safetensors"
        original = shard.read_bytes()
        shard.write_bytes(original.replace(b'"pt"', b'"tf"', 1))
        with self.assertRaisesRegex(ValueError, "header"):
            verify_reconstruction(self.bundle, self.output, self.reference)
        shard.write_bytes(original)
        for name in ("LICENSE", "config.json", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors.index.json"):
            with self.subTest(name=name):
                path = self.output / "model" / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    verify_reconstruction(self.bundle, self.output, self.reference)
                path.write_bytes(original)

    def test_forged_audit_counts_and_missing_audit_rejected(self):
        self.reconstruct()
        path = self.output / "audit.json"
        original = path.read_bytes()
        for field in ("changed_stored_entries", "targeted_coordinates", "sparse_substitutions"):
            audit = json.loads(original)
            audit[field] += 1
            write_json(path, audit)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "audit"):
                verify_reconstruction(self.bundle, self.output, self.reference)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_reconstruction(self.bundle, self.output, self.reference)
        path.write_bytes(original)
        (self.output / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            verify_reconstruction(self.bundle, self.output, self.reference)

    def test_copy_write_audit_and_verification_failures_leave_no_output(self):
        failures = (
            ("weight_space_lab.reconstruction.shutil.copyfile", OSError("fixture copy failure")),
            ("weight_space_lab.reconstruction.os.fsync", OSError("fixture write failure")),
            ("weight_space_lab.reconstruction.json.dumps", OSError("fixture audit failure")),
            ("weight_space_lab.reconstruction.verify_reconstruction", ValueError("fixture verification failure")),
        )
        for target, failure in failures:
            with self.subTest(target=target):
                with patch(target, side_effect=failure), self.assertRaises(type(failure)):
                    self.reconstruct()
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".reconstructed.incomplete-*")), [])
                self.assert_base_unchanged()

    def test_existing_output_and_nested_base_output_refused(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.reconstruct()
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaises(ValueError):
            reconstruct_published(self.bundle, self.bundle / "nested", self.reference)
        self.assert_base_unchanged()

    def test_base_tampering_prevents_reconstruction(self):
        asset = self.bundle / "model/LICENSE"
        asset.chmod(0o644)
        asset.write_text("modified fixture")
        with self.assertRaisesRegex(ValueError, "payload"):
            self.reconstruct()
        self.assertFalse(self.output.exists())

    def test_cli_offline_end_to_end_and_status_separation(self):
        def run(*args):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = main([str(arg) for arg in args])
            return status, json.loads(stream.getvalue())

        status, acquired = run("verify-acquisition", "--reference", self.reference, "--bundle", self.bundle)
        self.assertEqual(status, 0)
        self.assertFalse(acquired["inference_executed"])
        status, created = run("reconstruct-published", "--reference", self.reference,
                              "--acquisition", self.bundle, "--out", self.output)
        self.assertEqual(status, 0)
        self.assertEqual(created["changed_stored_entries"], 3)
        self.assertFalse(created["experiment_executed"])
        status, verified = run("verify-reconstruction", "--reference", self.reference,
                               "--acquisition", self.bundle, "--out", self.output)
        self.assertEqual(status, 0)
        self.assertEqual(verified, created)
        inventory_path = self.root / "inventory.json"
        status, count = run("inventory-model", "--model", self.output / "model", "--out", inventory_path)
        self.assertEqual(status, 0)
        self.assertEqual(count["tensor_count"], 4)
        status, error = run("reconstruct-published", "--reference", self.reference,
                            "--acquisition", self.bundle, "--out", self.output)
        self.assertEqual(status, 2)
        self.assertEqual(error["type"], "FileExistsError")
        self.assertNotIn(str(self.root), json.dumps(error))


if __name__ == "__main__":
    unittest.main()
