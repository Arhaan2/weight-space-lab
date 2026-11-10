"""Native replay admission checks; no fixture submits a checkpoint for inference."""

import copy
import unittest
from unittest.mock import patch

from weight_space_lab.native_assets import (
    backend_spec, implementation_identity, runtime_environment, validate_execution_config,
)
from weight_space_lab.native_runner import _inputs
from weight_space_lab.protocol import build_benchmark_schedule, canonical_hash


def replay_inputs(stage="pilot"):
    tasks = [{"benchmark": "math", "item_id": "schema-only", "dataset_revision": "a" * 40,
              "split": "test" if stage == "full" else "development",
              "messages": [{"role": "user", "content": "Synthetic validation only."}]}]
    schedule = build_benchmark_schedule("math", ["schema-only"], [17, 29])
    files = {"synthetic.model": {"sha256": "a" * 64, "size_bytes": 12}}
    source = implementation_identity()
    config = {
        "schema_version": 1, "kind": "native_execution", "run_id": "replay-schema-only",
        "evidence_kind": "real_model", "profile": "fixed_patch_evaluation", "stage": stage,
        "checkpoint": {"model_id": "fixture/never-loaded", "revision": "b" * 40,
                       "sha256": canonical_hash(files)},
        "retry_policy": {"mode": "none", "max_attempts": 1},
        "execution_limits": {"max_sequences": 2, "startup_timeout_seconds": 300,
                             "request_timeout_seconds": 180, "max_duration_seconds": 600},
        "backend": backend_spec("transformers", "c" * 64),
        "artifacts": {"files": files, "inventory_sha256": "d" * 64,
                      "tokenizer_revision": "b" * 40, "audit_sha256": "e" * 64,
                      "arm": "original", "source_revision": "f" * 40},
        "implementation_sha256": source, "runtime_environment": runtime_environment(),
        "tasks_sha256": canonical_hash(tasks), "schedule_sha256": canonical_hash(schedule),
        "methodological_variations": ["Schema-only fixture; no native execution."],
        "study": {"kind": "published_patch_replay", "freeze_id": "1" * 64,
                  "grader_revision": "2" * 64, "policy_sha256": "3" * 64,
                  "patch_sha256": "4" * 64, "reconstruction_audit_sha256": "5" * 64,
                  "evaluation_code_revision": source, "split_audit_sha256": "6" * 64,
                  "primary_comparison": "unchanged_vs_reconstructed_p50c", "truncation_policy": "grade"},
    }
    return config, tasks, schedule


class NativeReplayTests(unittest.TestCase):
    def test_complete_pilot_and_full_arms_retain_fixed_patch_profile(self):
        for stage in ("pilot", "full"):
            with self.subTest(stage=stage):
                config, tasks, schedule = replay_inputs(stage)
                checked = validate_execution_config(config, tasks, schedule)
                self.assertEqual(checked, config)
                checked["study"]["freeze_id"] = "changed-copy"
                self.assertEqual(config["study"]["freeze_id"], "1" * 64)
                manifest, records, _ = _inputs(config, tasks, schedule)
                self.assertEqual(len(records), 2)
                self.assertTrue(all(row["profile"] == "fixed_patch_evaluation" for row in records))
                self.assertEqual(manifest["configuration"]["study"], config["study"])

    def test_replay_requires_every_study_identity_and_rejects_unknown_fields(self):
        original, tasks, schedule = replay_inputs()
        for key in original["study"]:
            config = copy.deepcopy(original)
            del config["study"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_execution_config(config, tasks, schedule)
        for name, change in (
            ("missing study", lambda c: c.pop("study")),
            ("unknown study", lambda c: c["study"].update(ignored=True)),
            ("unknown top level", lambda c: c.update(ignored=True)),
            ("wrong claim", lambda c: c["study"].update(kind="independent_discovery")),
            ("wrong primary", lambda c: c["study"].update(primary_comparison="best_alternative")),
            ("moving freeze", lambda c: c["study"].update(freeze_id="latest")),
            ("missing grader", lambda c: c["study"].update(grader_revision=None)),
            ("other implementation", lambda c: c["study"].update(evaluation_code_revision="0" * 64)),
            ("bad truncation", lambda c: c["study"].update(truncation_policy="discard")),
            ("fixture promotion", lambda c: c.update(evidence_kind="fixture")),
        ):
            config = copy.deepcopy(original)
            change(config)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_execution_config(config, tasks, schedule)

    def test_replay_cannot_admit_test_data_to_pilot_or_development_to_full(self):
        for stage, wrong_split in (("pilot", "test"), ("pilot", "validation"), ("full", "development")):
            config, tasks, schedule = replay_inputs(stage)
            tasks[0]["split"] = wrong_split
            config["tasks_sha256"] = canonical_hash(tasks)
            with self.subTest(stage=stage, split=wrong_split), self.assertRaisesRegex(ValueError, "split"):
                validate_execution_config(config, tasks, schedule)
        config, tasks, schedule = replay_inputs()
        config["stage"] = "smoke"
        with self.assertRaisesRegex(ValueError, "pilot or full"):
            validate_execution_config(config, tasks, schedule)

    def test_replay_still_rejects_runtime_source_and_schedule_drift(self):
        config, tasks, schedule = replay_inputs("full")
        with patch("weight_space_lab.native_assets.runtime_environment", return_value={}):
            with self.assertRaisesRegex(ValueError, "runtime environment changed"):
                validate_execution_config(config, tasks, schedule)
        with patch("weight_space_lab.native_assets.implementation_identity", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "execution source changed"):
                validate_execution_config(config, tasks, schedule)
        changed = copy.deepcopy(schedule)
        changed[0]["seed"] = 111
        with self.assertRaisesRegex(ValueError, "schedule differ"):
            validate_execution_config(config, tasks, changed)
        config["execution_limits"]["max_sequences"] = 1
        with self.assertRaisesRegex(ValueError, "scheduled sequences exceed"):
            validate_execution_config(config, tasks, schedule)

    def test_smoke_cannot_carry_replay_settings_and_fixed_patch_cannot_impersonate_search(self):
        config, tasks, schedule = replay_inputs()
        config.update(profile="native_development_smoke", stage="smoke")
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_execution_config(config, tasks, schedule)
        config, tasks, schedule = replay_inputs()
        config["profile"] = "fresh_editor_search"
        del config["study"]
        with self.assertRaisesRegex(ValueError, "frozen fixed-patch replay"):
            validate_execution_config(config, tasks, schedule)


if __name__ == "__main__":
    unittest.main()
