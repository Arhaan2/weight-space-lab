import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_reconstruction import fixture
from weight_space_lab.__main__ import main
from weight_space_lab.acquisition import _remove_stage
from weight_space_lab.inventory import file_sha256, inventory_model
from weight_space_lab.native_assets import (
    backend_spec, copy_identity, implementation_identity, validate_execution_config,
    verify_execution_assets, verify_identity,
)
from weight_space_lab.protocol import build_benchmark_schedule, canonical_hash


class NativeAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.acquisition, self.reference = fixture(self.root)
        self.addCleanup(_remove_stage, self.acquisition)
        self.identity = self.root / "identity"
        self.addCleanup(_remove_stage, self.identity)

    def test_identity_is_independent_reopened_and_exact(self):
        result = copy_identity(self.acquisition, self.identity, self.reference)
        observed = verify_identity(self.acquisition, self.identity, self.reference)
        self.assertEqual(result["changed_stored_entries"], 0)
        self.assertFalse(result["inference_executed"])
        self.assertEqual(observed, inventory_model(self.acquisition / "model"))
        source = self.acquisition / "model/LICENSE"
        copied = self.identity / "model/LICENSE"
        self.assertNotEqual(source.stat().st_ino, copied.stat().st_ino)
        copied.chmod(0o644)
        copied.write_text("changed synthetic asset")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_identity(self.acquisition, self.identity, self.reference)
        self.assertEqual(source.read_text(), "Synthetic fixture license.\n")

    def test_failed_copy_leaves_no_published_identity(self):
        with patch("weight_space_lab.native_assets.shutil.copyfile", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                copy_identity(self.acquisition, self.identity, self.reference)
        self.assertFalse(self.identity.exists())
        self.assertEqual(list(self.root.glob(".identity.incomplete-*")), [])

    def test_existing_or_nested_identity_refused(self):
        copy_identity(self.acquisition, self.identity, self.reference)
        original = file_sha256(self.identity / "audit.json")
        with self.assertRaises(FileExistsError):
            copy_identity(self.acquisition, self.identity, self.reference)
        self.assertEqual(file_sha256(self.identity / "audit.json"), original)
        with self.assertRaises(ValueError):
            copy_identity(self.acquisition, self.acquisition / "nested", self.reference)

    def test_identity_cli_output_never_contains_local_paths(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = main(["copy-identity", "--acquisition", str(self.acquisition), "--out", str(self.identity),
                           "--reference", str(self.reference)])
        self.assertEqual(status, 0)
        self.assertNotIn(str(self.root), stream.getvalue())
        self.assertEqual(json.loads(stream.getvalue())["kind"], "identity_byte_copy")

    def config(self):
        # Hand-authored schema fixture, never submitted for native inference.
        from weight_space_lab.native_assets import runtime_environment
        task = {"benchmark": "synthetic", "item_id": "toy", "split": "development",
                "dataset_revision": canonical_hash("synthetic task"),
                "messages": [{"role": "user", "content": "fixture only"}]}
        tasks = [task]
        schedule = build_benchmark_schedule("synthetic", ["toy"], [17])
        inventory = inventory_model(self.acquisition / "model")
        spec = backend_spec("transformers", canonical_hash("fixture template"))
        config = {
            "schema_version": 1, "kind": "native_execution", "run_id": "fixture-schema-only",
            "evidence_kind": "real_model", "profile": "native_development_smoke", "stage": "smoke",
            "checkpoint": {"model_id": "fixture/synthetic-model", "revision": "2" * 40,
                           "sha256": canonical_hash(inventory["files"])},
            "retry_policy": {"mode": "none", "max_attempts": 1},
            "execution_limits": {"max_sequences": 1, "startup_timeout_seconds": 300,
                                 "request_timeout_seconds": 180, "max_duration_seconds": 600},
            "backend": spec, "runtime_environment": runtime_environment(),
            "artifacts": {"files": inventory["files"], "inventory_sha256": canonical_hash(inventory),
                          "tokenizer_revision": "2" * 40, "audit_sha256": file_sha256(self.acquisition / "acquisition.json"),
                          "arm": "original", "source_revision": "1" * 40},
            "implementation_sha256": implementation_identity(),
            "tasks_sha256": canonical_hash(tasks), "schedule_sha256": canonical_hash(schedule),
            "methodological_variations": ["Synthetic input validation only; never executed."],
        }
        return config, tasks, schedule

    def test_frozen_inputs_validate_and_changes_are_rejected(self):
        config, tasks, schedule = self.config()
        self.assertEqual(validate_execution_config(config, tasks, schedule), config)
        cases = [
            ("task", lambda c, t, s: t[0]["messages"][0].update(content="changed")),
            ("seed", lambda c, t, s: s[0].update(seed=19)),
            ("source", lambda c, t, s: c.update(implementation_sha256=canonical_hash("different source"))),
            ("runtime", lambda c, t, s: c["runtime_environment"].update(python="different")),
            ("stop", lambda c, t, s: c["backend"]["stopping"].update(ignore_eos=True)),
            ("unknown", lambda c, t, s: c.update(ignored_option=True)),
            ("checkpoint", lambda c, t, s: c["checkpoint"].update(sha256=canonical_hash("changed"))),
            ("stage", lambda c, t, s: c.update(stage="full")),
            ("retry", lambda c, t, s: c["retry_policy"].update(max_attempts=2)),
        ]
        for name, change in cases:
            c, t, s = copy.deepcopy((config, tasks, schedule))
            change(c, t, s)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_execution_config(c, t, s)

    def test_verification_rejects_asset_change_and_wrong_architecture(self):
        config, _, _ = self.config()
        with self.assertRaisesRegex(ValueError, "architecture"):
            verify_execution_assets(config, self.acquisition / "model")
        path = self.acquisition / "model/LICENSE"
        path.chmod(0o644)
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_execution_assets(config, self.acquisition / "model")

    def test_actual_source_and_environment_drift_reject_frozen_config(self):
        config, tasks, schedule = self.config()
        with patch("weight_space_lab.native_assets.implementation_identity",
                   return_value=canonical_hash("changed executable code")):
            with self.assertRaisesRegex(ValueError, "execution source changed"):
                validate_execution_config(config, tasks, schedule)
        changed_environment = copy.deepcopy(config["runtime_environment"])
        changed_environment["packages"]["numpy"] = "0.0.0"
        with patch("weight_space_lab.native_assets.runtime_environment", return_value=changed_environment):
            with self.assertRaisesRegex(ValueError, "runtime environment changed"):
                validate_execution_config(config, tasks, schedule)

    def test_malformed_checkpoint_and_unbounded_limits_reject_before_execution(self):
        original, tasks, schedule = self.config()
        for checkpoint in (None, [], {}, {**original["checkpoint"], "revision": "main"},
                           {**original["checkpoint"], "model_id": ""},
                           {**original["checkpoint"], "extra": "ignored"}):
            config = copy.deepcopy(original)
            config["checkpoint"] = checkpoint
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                validate_execution_config(config, tasks, schedule)
        for key in ("max_sequences", "startup_timeout_seconds", "request_timeout_seconds", "max_duration_seconds"):
            for value in (0, -1, True, None, float("inf"), float("nan"), 10 ** 1000):
                if key == "max_sequences" and value == 10 ** 1000:
                    continue  # A finite integer sequence budget is permitted.
                config = copy.deepcopy(original)
                config["execution_limits"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_execution_config(config, tasks, schedule)

    def test_resealed_tasks_still_require_development_scope_and_exact_schedule(self):
        original, tasks, schedule = self.config()
        changes = [
            lambda t: t[0].update(split="test"),
            lambda t: t.append(copy.deepcopy(t[0])),
            lambda t: t[0].update(item_id="unscheduled"),
            lambda t: t[0].update(dataset_revision="unresolved"),
            lambda t: t[0].update(messages=[]),
            lambda t: t[0]["messages"][0].update(role="tool"),
            lambda t: t[0]["messages"][0].update(content=[]),
            lambda t: t[0]["messages"][0].update(content=""),
            lambda t: t[0].update(ignored_option=True),
        ]
        for change in changes:
            config, changed = copy.deepcopy((original, tasks))
            change(changed)
            config["tasks_sha256"] = canonical_hash(changed)
            with self.subTest(tasks=changed), self.assertRaises(ValueError):
                validate_execution_config(config, changed, schedule)

    def test_tokenizer_bytes_are_verified_independently_of_weight_identity(self):
        config, _, _ = self.config()
        for asset in ("tokenizer.json", "tokenizer_config.json"):
            path = self.acquisition / "model" / asset
            original = path.read_bytes()
            path.chmod(0o644)
            path.write_bytes(original + b" ")
            with self.subTest(asset=asset), self.assertRaisesRegex(ValueError, "changed"):
                verify_execution_assets(config, self.acquisition / "model")
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
