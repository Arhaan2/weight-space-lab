"""CPU contract fixtures; all generated identities below are synthetic."""

import copy
import json
import unittest
from pathlib import Path

from weight_space_lab.protocol import (
    PROFILES, build_benchmark_schedule, canonical_hash, frozen_configuration_hash,
    validate_benchmark_schedule, validate_protocol,
)


ROOT = Path(__file__).parents[1]


def template(profile):
    return json.loads((ROOT / "configs/profiles" / (profile + ".json")).read_text())


def complete(profile):
    """Make a declared-complete synthetic fixture, not an executed experiment."""
    config = template(profile)
    for name, value in config["pins"].items():
        if value is None:
            config["pins"][name] = canonical_hash(name) if name.endswith(("revision", "sha256")) else "fixture-version-1"
    if "backend" in config:
        config["backend"] = "fixture-backend"
        config["precision"] = "fixture-precision"
        config["execution_limits"] = {"max_sequences": 8, "max_duration_seconds": 60}
        for plan in config["benchmarks"]:
            plan.update(dataset_revision=canonical_hash("fixture-dataset"),
                        task_manifest_sha256=canonical_hash("fixture-task-manifest"),
                        item_ids=["fixture-a", "fixture-b"], seeds=[0, 1],
                        sampling={"mode": "greedy", "temperature": 0, "top_p": 1,
                                  "top_k": 0, "presence_penalty": 0, "thinking": False,
                                  "max_output_tokens": 8})
            if "grader_revision" in plan:
                plan.update(grader_revision=canonical_hash("fixture-grader"), truncation_policy="fail")
    if "editor_provenance" in config:
        config["editor_provenance"] = {"model_id": "fixture-editor-id", "interface": "fixture-interface",
                                       "record_sha256": canonical_hash("fixture-editor-record")}
    if "browser" in config:
        config["browser"] = {"version": "fixture-browser-version", "operating_system": "fixture-system",
            "hardware": "fixture-device", "quantization": "fixture-quantization",
            "session_plan_sha256": canonical_hash("fixture-session-plan")}
    if profile == "frozen_final_evaluation":
        freeze(config)
    return config


def freeze(config):
    config["frozen"] = True
    rows = [row for plan in config["benchmarks"] for row in
            build_benchmark_schedule(plan["benchmark"], plan["item_ids"], plan["seeds"])]
    config["freeze"] = {
        "configuration_sha256": frozen_configuration_hash(config),
        "checkpoint_sha256": config["pins"]["checkpoint_sha256"],
        "schedule_sha256": validate_benchmark_schedule(rows)["schedule_sha256"],
        "evaluation_code_revision": config["pins"]["evaluation_code_revision"],
    }


class ProfileTests(unittest.TestCase):
    def test_templates_have_explicit_pending_information(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result = validate_protocol(template(profile))
                self.assertTrue(result["valid_design"], result["issues"])
                # Acquisition can name already documented immutable sources.
                self.assertEqual(result["run_configuration_complete"], profile == "artifact_acquisition")
                self.assertIn("configuration only", result["verification_scope"])

    def test_every_profile_has_a_complete_configuration_fixture(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result = validate_protocol(complete(profile), require_ready=True)
                self.assertTrue(result["run_configuration_complete"], result["issues"])
                self.assertNotIn("scientific_result", result)

    def test_replay_does_not_need_editor_runtime_grader_or_freeze(self):
        for profile in ("artifact_acquisition", "published_patch_reconstruction"):
            config = complete(profile)
            self.assertNotIn("editor_model_version", validate_protocol(config)["required_pins"])
            self.assertTrue(validate_protocol(config)["run_configuration_complete"])
            for irrelevant in ("editor_model_version", "runtime_version", "grader_revision"):
                config["pins"][irrelevant] = "unknown"
            self.assertTrue(validate_protocol(config)["run_configuration_complete"])

    def test_fixed_patch_does_not_need_editor_or_search_data(self):
        config = complete("fixed_patch_evaluation")
        result = validate_protocol(config)
        self.assertTrue(result["run_configuration_complete"])
        self.assertNotIn("editor_model_version", result["required_pins"])
        self.assertNotIn("development_dataset_revision", result["required_pins"])

    def test_smoke_needs_only_development_and_no_grader(self):
        config = complete("native_development_smoke")
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])
        config["benchmarks"][0]["split"] = "test"
        self.assertFalse(validate_protocol(config)["valid_design"])

    def test_greedy_and_stochastic_are_explicit(self):
        config = complete("native_development_smoke")
        sampling = config["benchmarks"][0]["sampling"]
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])
        for change in ({"mode": "stochastic"}, {"temperature": 0.1}, {"top_p": 0.95},
                       {"top_k": 20}, {"samples_per_item": 4}, {"mode": None},
                       {"thinking": 1}, {"max_output_tokens": True},
                       {"do_sample": True}, {"temperature": 10 ** 1000}):
            changed = copy.deepcopy(config)
            changed["benchmarks"][0]["sampling"].update(change)
            with self.subTest(change=change):
                self.assertFalse(validate_protocol(changed)["valid_design"])
        sampling.update(mode="stochastic", temperature=0.6, top_p=0.95, top_k=20)
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])

    def test_missing_sampling_and_seeds_never_get_defaults(self):
        config = complete("native_development_smoke")
        for field in ("sampling", "seeds", "item_ids", "dataset_revision", "task_manifest_sha256"):
            changed = copy.deepcopy(config)
            changed["benchmarks"][0][field] = None
            result = validate_protocol(changed)
            with self.subTest(field=field):
                self.assertTrue(result["valid_design"])
                self.assertFalse(result["run_configuration_complete"])

    def test_fresh_search_requires_both_selection_splits_and_actual_editor(self):
        config = complete("fresh_editor_search")
        for field in ("model_id", "interface"):
            changed = copy.deepcopy(config)
            changed["editor_provenance"][field] = None
            self.assertFalse(validate_protocol(changed)["run_configuration_complete"])
        config["benchmarks"][0]["split"] = "test"
        self.assertFalse(validate_protocol(config)["valid_design"])

    def test_same_benchmark_across_selection_splits_has_one_schedule(self):
        config = complete("fresh_editor_search")
        development, validation = config["benchmarks"]
        validation["benchmark"] = development["benchmark"]
        validation["item_ids"] = ["fixture-c", "fixture-d"]
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])
        validation["seeds"] = [2, 3]
        self.assertFalse(validate_protocol(config)["valid_design"])
        validation["seeds"] = development["seeds"]
        validation["item_ids"] = ["fixture-a", "fixture-d"]
        self.assertFalse(validate_protocol(config)["valid_design"])

    def test_final_claim_cannot_bypass_editor_provenance_by_changing_origin(self):
        for origin in ("published_patch", "unchanged"):
            config = complete("frozen_final_evaluation")
            config["intervention_origin"] = origin
            freeze(config)
            self.assertFalse(validate_protocol(config)["valid_design"])
        config = complete("frozen_final_evaluation")
        config["claim_type"] = "published_patch_replay"
        freeze(config)
        self.assertFalse(validate_protocol(config)["valid_design"])

    def test_fixed_full_test_requires_frozen_evidence(self):
        config = complete("fixed_patch_evaluation")
        config["stage"] = "full"
        self.assertFalse(validate_protocol(config)["run_configuration_complete"])
        freeze(config)
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])

    def test_final_requires_editor_history_but_not_a_live_connection(self):
        config = complete("frozen_final_evaluation")
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])
        del config["editor_provenance"]["record_sha256"]
        freeze(config)
        self.assertFalse(validate_protocol(config)["run_configuration_complete"])

    def test_final_published_origin_requires_patch_not_editor(self):
        config = complete("frozen_final_evaluation")
        config.update(intervention_origin="published_patch", claim_type="published_patch_replay")
        del config["editor_provenance"]
        del config["pins"]["editor_model_version"]
        for name in ("source_revision", "patch_sha256", "reconstruction_audit_sha256"):
            config["pins"][name] = canonical_hash(name)
        freeze(config)
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])

    def test_freeze_binds_configuration_checkpoint_schedule_and_evaluator(self):
        config = complete("frozen_final_evaluation")
        changes = [lambda c: c.update(frozen=False),
                   lambda c: c.update(precision="changed"),
                   lambda c: c["pins"].update(checkpoint_sha256=canonical_hash("changed")),
                   lambda c: c["pins"].update(evaluation_code_revision=canonical_hash("changed")),
                   lambda c: c["benchmarks"][0].update(seeds=[2, 3]),
                   lambda c: c["freeze"].update(schedule_sha256=None),
                   lambda c: c["editor_provenance"].update(model_id="changed")]
        for change in changes:
            changed = copy.deepcopy(config)
            change(changed)
            self.assertFalse(validate_protocol(changed)["run_configuration_complete"])

    def test_browser_test_requires_freeze_conversion_and_identity_evidence(self):
        config = complete("browser_evaluation")
        config["benchmarks"][0]["split"] = "test"
        self.assertFalse(validate_protocol(config)["run_configuration_complete"])
        freeze(config)
        self.assertTrue(validate_protocol(config)["run_configuration_complete"])
        config["pins"]["identity_validation_sha256"] = None
        freeze(config)
        self.assertFalse(validate_protocol(config)["run_configuration_complete"])

    def test_budgets_and_profile_stages_cannot_be_misrepresented(self):
        config = complete("native_development_smoke")
        config["execution_limits"]["max_sequences"] = 3
        self.assertFalse(validate_protocol(config)["valid_design"])
        config = complete("fresh_editor_search")
        config["probe_limits"]["max_calls"] = True
        self.assertFalse(validate_protocol(config)["valid_design"])
        config = complete("frozen_final_evaluation")
        config["stage"] = "smoke"
        self.assertFalse(validate_protocol(config)["valid_design"])

    def test_null_bad_and_moving_pins_are_not_success(self):
        for value in (None, "main", "latest", "todo", True, 0, [], {}, "not-a-revision"):
            config = complete("native_development_smoke")
            config["pins"]["model_revision"] = value
            with self.subTest(value=value):
                self.assertFalse(validate_protocol(config)["run_configuration_complete"])
        config["pins"]["model_revision"] = canonical_hash("fixture-model")
        for value in ("unknown", "placeholder", "pending", False):
            config["pins"]["runtime_version"] = value
            self.assertFalse(validate_protocol(config)["run_configuration_complete"])

    def test_malformed_shapes_return_issues(self):
        for field in ("profile", "stage", "pins", "benchmarks", "execution_limits", "claim_type"):
            for value in ([], {}, True, 0):
                config = complete("native_development_smoke")
                config[field] = value
                with self.subTest(field=field, value=value):
                    self.assertFalse(validate_protocol(config)["run_configuration_complete"])


class ScheduleTests(unittest.TestCase):
    def test_different_benchmarks_can_have_different_repeat_counts(self):
        first = build_benchmark_schedule("fixture-math", ["a", "b"], [3, 5, 7, 9])
        second = build_benchmark_schedule("fixture-instructions", ["a", "c"], [11])
        result = validate_benchmark_schedule(first + second)
        self.assertEqual(result["benchmarks"]["fixture-math"]["samples_per_item"], 4)
        self.assertEqual(result["benchmarks"]["fixture-instructions"]["samples_per_item"], 1)
        self.assertEqual(result, validate_benchmark_schedule(list(reversed(first + second))))

    def test_partial_uneven_duplicate_or_ambiguous_schedules_fail(self):
        rows = build_benchmark_schedule("fixture", ["a", "b"], [3, 5])
        cases = [rows[:-1], rows + rows[:1], [],
                 [dict(row, sample_index=row["sample_index"] + 1) for row in rows],
                 [dict(row, seed=True) for row in rows],
                 [dict(row, seed=0) for row in rows],
                 [dict(row, extra=True) for row in rows]]
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_benchmark_schedule(changed)
        changed = copy.deepcopy(rows)
        changed[-1]["seed"] = 6
        with self.assertRaises(ValueError):
            validate_benchmark_schedule(changed)

    def test_builder_never_invents_seeds_or_coerces_identifiers(self):
        for items, seeds in ((["a"], []), (["a", "a"], [0]), (["a"], [True]),
                             (["a"], [0, 0]), ([1], [0]), ("a", [0])):
            with self.assertRaises(ValueError):
                build_benchmark_schedule("fixture", items, seeds)


if __name__ == "__main__":
    unittest.main()
