import copy
import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

from weight_space_lab.inventory import inventory_model, read_json, safe_relative, file_sha256


def write_shard(path, tensors):
    header = {"__metadata__": {"format": "pt", "fixture": "synthetic"}}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    Path(path).write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class InventoryTests(unittest.TestCase):
    def test_file_hash_rejects_dataless_without_opening(self):
        info = SimpleNamespace(st_mode=0o100444, st_flags=0x40000000)
        with mock.patch.object(Path, "lstat", return_value=info), mock.patch.object(Path, "open") as opened:
            with self.assertRaisesRegex(ValueError, "dataless"):
                file_sha256(Path("unavailable"))
            opened.assert_not_called()

    def test_file_hash_rejects_short_reads_with_positive_size(self):
        path = self.root / "empty.bin"
        path.write_bytes(b"")
        info = path.stat()
        fake = SimpleNamespace(st_mode=info.st_mode, st_flags=0, st_size=12,
                               st_mtime_ns=info.st_mtime_ns, st_ino=info.st_ino)
        with mock.patch.object(Path, "lstat", return_value=fake):
            with self.assertRaisesRegex(ValueError, "ended before"):
                file_sha256(path)
        self.assertEqual(file_sha256(path), hashlib.sha256(b"").hexdigest())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base_bytes = bytes.fromhex("803f00c000000080")
        self.other_bytes = bytes.fromhex("0000803f000000c0")
        self.config = {"model_type": "synthetic", "dtype": "bfloat16"}
        self.tokenizer_config = {"chat_template": "{{ messages }}", "eos_token": "</s>"}
        for name, value in (("config.json", self.config),
                            ("tokenizer_config.json", self.tokenizer_config),
                            ("tokenizer.json", {"fixture": True})):
            (self.root / name).write_text(json.dumps(value))
        (self.root / "LICENSE").write_text("Synthetic fixture license notice.\n")
        write_shard(self.root / "model.safetensors", [
            ("layer.weight", "BF16", [2, 2], self.base_bytes),
            ("other", "F32", [2], self.other_bytes),
            ("scalar", "I64", [], bytes.fromhex("0100000000000000")),
            ("empty", "U8", [0], b""),
        ])

    def sharded(self):
        (self.root / "model.safetensors").unlink()
        write_shard(self.root / "model-00001-of-00002.safetensors", [
            ("layer.weight", "BF16", [2, 2], self.base_bytes)])
        write_shard(self.root / "model-00002-of-00002.safetensors", [
            ("other", "F32", [2], self.other_bytes)])
        index = {"metadata": {"total_size": 16}, "weight_map": {
            "layer.weight": "model-00001-of-00002.safetensors",
            "other": "model-00002-of-00002.safetensors"}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        return index

    def test_complete_inventory_hashes_stored_types_counts_and_assets(self):
        originals = {file.name: file.read_bytes() for file in self.root.iterdir()}
        result = inventory_model(self.root)
        self.assertEqual(result["parameter_count"], 7)
        self.assertEqual(result["stored_tensor_bytes"], 24)
        self.assertEqual(result["model_config"], self.config)
        self.assertEqual(result["shards"], ["model.safetensors"])
        self.assertIsNone(result["index"])
        self.assertEqual(result["tensors"]["layer.weight"]["dtype"], "BF16")
        self.assertEqual(result["tensors"]["other"]["dtype"], "F32")
        self.assertEqual(result["tensors"]["scalar"]["parameter_count"], 1)
        self.assertEqual(result["tensors"]["empty"]["parameter_count"], 0)
        weight = result["tensors"]["layer.weight"]
        self.assertEqual(weight["shape"], [2, 2])
        self.assertEqual(weight["data_offsets"], [0, 8])
        self.assertEqual(weight["sha256"], hashlib.sha256(self.base_bytes).hexdigest())
        for name, data in originals.items():
            self.assertEqual(result["files"][name], {
                "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
            self.assertEqual((self.root / name).read_bytes(), data)
        template = result["chat_templates"]["tokenizer_config.json#chat_template/default"]
        self.assertEqual(template["content"], "{{ messages }}")
        self.assertEqual(template["sha256"], hashlib.sha256(b"{{ messages }}").hexdigest())
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_actual_shards_and_index_recorded(self):
        index = self.sharded()
        result = inventory_model(self.root)
        self.assertEqual(result["index"]["content"], index)
        self.assertEqual(result["parameter_count"], 6)
        for tensor, shard in index["weight_map"].items():
            self.assertEqual(result["tensors"][tensor]["shard"], shard)

    def test_index_rejects_missing_extra_wrong_shard_and_size(self):
        index = self.sharded()
        cases = []
        for key, value in (("layer.weight", "missing.safetensors"),
                           ("layer.weight", "../outside.safetensors"),
                           ("layer.weight", "model-00002-of-00002.safetensors"),
                           ("extra", "model-00001-of-00002.safetensors")):
            altered = copy.deepcopy(index)
            altered["weight_map"][key] = value
            cases.append(altered)
        altered = copy.deepcopy(index)
        del altered["weight_map"]["other"]
        cases.append(altered)
        cases.extend([{**index, "metadata": {"total_size": value}} for value in (15, True, "16")])
        for altered in cases:
            with self.subTest(altered=altered):
                (self.root / "model.safetensors.index.json").write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    inventory_model(self.root)

    def test_shards_require_index_and_duplicate_tensors_fail(self):
        self.sharded()
        (self.root / "model.safetensors.index.json").unlink()
        with self.assertRaisesRegex(ValueError, "require.*index"):
            inventory_model(self.root)
        write_shard(self.root / "model-00002-of-00002.safetensors", [
            ("layer.weight", "BF16", [2, 2], self.base_bytes)])
        with self.assertRaisesRegex(ValueError, "unique"):
            inventory_model(self.root)

    def test_unindexed_extra_empty_shard_is_rejected(self):
        self.sharded()
        write_shard(self.root / "extra.safetensors", [])
        with self.assertRaisesRegex(ValueError, "layout"):
            inventory_model(self.root)

    def test_missing_required_assets_and_template_rejected(self):
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            content = (self.root / name).read_bytes()
            (self.root / name).unlink()
            with self.subTest(name=name), self.assertRaises(ValueError):
                inventory_model(self.root)
            (self.root / name).write_bytes(content)
        (self.root / "tokenizer_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "chat template"):
            inventory_model(self.root)

    def test_external_and_named_embedded_templates(self):
        config = {"chat_template": [{"name": "plain", "template": "plain {{ messages }}"},
                                    {"name": "tools", "template": "tools {{ messages }}"}]}
        (self.root / "tokenizer_config.json").write_text(json.dumps(config))
        (self.root / "chat_templates").mkdir()
        (self.root / "chat_templates" / "default.jinja").write_text("external {{ messages }}\n")
        result = inventory_model(self.root)
        self.assertEqual(len(result["chat_templates"]), 3)
        self.assertEqual(result["chat_templates"]["chat_templates/default.jinja"]["content"],
                         "external {{ messages }}\n")
        config["chat_template"].append(config["chat_template"][0])
        (self.root / "tokenizer_config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "named chat"):
            inventory_model(self.root)

    def test_sentencepiece_and_bpe_tokenizer_assets(self):
        (self.root / "tokenizer.json").unlink()
        (self.root / "tokenizer.model").write_bytes(b"fixture")
        self.assertIn("tokenizer.model", inventory_model(self.root)["tokenizer_assets"])
        (self.root / "tokenizer.model").unlink()
        (self.root / "vocab.json").write_text("{}")
        (self.root / "merges.txt").write_text("# synthetic fixture\n")
        self.assertIn("merges.txt", inventory_model(self.root)["tokenizer_assets"])

    def test_symlinks_and_nonregular_files_rejected(self):
        link = self.root / "linked-config.json"
        link.symlink_to(self.root / "config.json")
        with self.assertRaisesRegex(ValueError, "link"):
            inventory_model(self.root)
        link.unlink()
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "link"):
            inventory_model(self.root)
        link.unlink()
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.root / "pipe")
            with self.assertRaisesRegex(ValueError, "nonregular"):
                inventory_model(self.root)

    def test_corrupt_tensor_spans_unknown_types_and_extra_bytes_rejected(self):
        path = self.root / "model.safetensors"
        original = path.read_bytes()
        header_length = int.from_bytes(original[:8], "little")
        header = json.loads(original[8:8 + header_length])
        for dtype in ("UNKNOWN", None):
            altered = copy.deepcopy(header)
            altered["layer.weight"]["dtype"] = dtype
            encoded = json.dumps(altered).encode()
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded
                             + original[8 + header_length:])
            with self.assertRaises(ValueError):
                inventory_model(self.root)
        for data in (original[:-1], original + b"\x00"):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                inventory_model(self.root)

    def test_strict_json_and_relative_paths(self):
        path = self.root / "strict.json"
        for content in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            path.write_text(content)
            with self.assertRaises(ValueError):
                read_json(path)
        for value in ("", "/absolute", "../escape", "a/../b", "a/./b", "a//b",
                      "a\\b", "C:/file", "x\x00", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative(value)
        self.assertEqual(safe_relative("subdir/model-00001.safetensors"),
                         "subdir/model-00001.safetensors")


if __name__ == "__main__":
    unittest.main()
