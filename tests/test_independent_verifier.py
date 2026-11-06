import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_reconstruction import fixture
from weight_space_lab.acquisition import _remove_stage
from weight_space_lab.reconstruction import reconstruct_published


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_p50c_bytes.py"
SPEC = importlib.util.spec_from_file_location("independent_byte_checker", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class IndependentVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle, self.reference = fixture(self.root)
        self.addCleanup(_remove_stage, self.bundle)
        self.output = self.root / "reconstruction"
        self.addCleanup(_remove_stage, self.output)
        reconstruct_published(self.bundle, self.output, self.reference)

    def verify(self):
        return CHECKER.run(self.bundle, self.output, self.reference)

    def test_independent_saved_bytes_and_target_support(self):
        result = self.verify()
        self.assertEqual(result["independent_verification"], "passed")
        self.assertEqual(result["changed_stored_entries"], 3)
        self.assertEqual(result["targeted_coordinates"], 6)
        self.assertEqual(result["stored_tensor_count"], 4)
        self.assertEqual(result["unchanged_tensor_count"], 2)
        self.assertTrue(result["all_non_target_bytes_equal"])
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_target_unrelated_tensor_header_and_asset_tampering(self):
        shard = self.output / "model/model-00001-of-00002.safetensors"
        original = shard.read_bytes()
        start, header = CHECKER.header(shard)
        positions = [start + header[name]["data_offsets"][0]
                     for name in ("layer.first", "untouched.f32")]
        for position in positions:
            modified = bytearray(original)
            modified[position] ^= 1
            shard.write_bytes(modified)
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.verify()
            shard.write_bytes(original)
        shard.write_bytes(original.replace(b'"pt"', b'"tf"', 1))
        with self.assertRaises(ValueError):
            self.verify()
        shard.write_bytes(original)
        asset = self.output / "model/LICENSE"
        asset.write_text("altered fixture license")
        with self.assertRaises(ValueError):
            self.verify()

    def test_reference_must_match_independent_configuration(self):
        reference = json.loads(self.reference.read_text())
        reference["base_revision"] = "3" * 40
        self.reference.write_text(json.dumps(reference))
        with self.assertRaisesRegex(ValueError, "reference differs"):
            self.verify()

    def test_incomplete_audit_duplicate_keys_and_extra_files_fail(self):
        audit_path = self.output / "audit.json"
        original = audit_path.read_bytes()
        audit = json.loads(original)
        audit["complete"] = False
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.verify()
        audit_path.write_text('{"complete": true, "complete": false}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify()
        audit_path.write_bytes(original)
        extra = self.output / "extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.verify()
        extra.unlink()
        audit_path.unlink()
        with self.assertRaises(OSError):
            self.verify()

    def test_truncated_invalid_layout_and_symlink_fail(self):
        shard = self.output / "model/model-00001-of-00002.safetensors"
        original = shard.read_bytes()
        shard.write_bytes(original[:-1])
        with self.assertRaises(ValueError):
            self.verify()
        shard.write_bytes(original)
        prefix = int.from_bytes(original[:8], "little")
        malformed = json.loads(original[8:8 + prefix])
        malformed["layer.first"]["data_offsets"] = [1, 25]
        encoded = json.dumps(malformed).encode()
        malformed_path = self.root / "malformed.safetensors"
        malformed_path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + original[8 + prefix:])
        with self.assertRaises(ValueError):
            CHECKER.header(malformed_path)
        audit = self.output / "audit.json"
        audit.unlink()
        audit.symlink_to(self.bundle / "acquisition.json")
        with self.assertRaises(ValueError):
            self.verify()

    def test_cli_checks_remain_enabled_under_optimized_python(self):
        reference = json.loads(self.reference.read_text())
        reference["base_revision"] = "3" * 40
        self.reference.write_text(json.dumps(reference))
        completed = subprocess.run([sys.executable, "-O", str(SCRIPT), str(self.bundle), str(self.output),
                                    "--reference", str(self.reference)], capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 2)
        result = json.loads(completed.stdout)
        self.assertEqual(result["independent_verification"], "failed")
        self.assertNotIn(str(self.root), completed.stdout + completed.stderr)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = CHECKER.main([str(self.bundle), str(self.output), "--reference", str(self.reference)])
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
