"""Public freeze consistency; does not open private prompts or execute a model."""
import hashlib
import json
from pathlib import Path
import unittest
import subprocess
import tempfile
import sys

from weight_space_lab.protocol import canonical_hash


ROOT = Path(__file__).resolve().parents[1]


class PublicFreezeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/ifeval_browser_v2.json").read_text())

    def test_schedule_arithmetic_and_nonrenewable_window(self):
        config = self.config
        self.assertEqual(config["submission_deadline_epoch_ms"] - config["freeze_epoch_ms"], 7 * 86400 * 1000)
        self.assertEqual(sum(config["stage_seconds"].values()), 14400)
        self.assertEqual(config["stage_seconds"]["browser"], 10800)
        schedule, limits = config["schedule"], config["limits"]
        development = schedule["rounds"] * len(schedule["development_arms_per_round"])
        final = len(schedule["final_arms"])
        scored = development * schedule["development_prompts_per_arm"] + final * schedule["final_prompts_per_arm"]
        loads = development + final
        self.assertEqual((development, final, scored, loads), (8, 3, 640, 11))
        self.assertEqual(limits["imports"], development)
        self.assertEqual(limits["conversions"], development)
        self.assertEqual(limits["packages"], development)
        self.assertEqual(limits["dispatches"], scored + loads)
        self.assertEqual(limits["generated_tokens"], scored * 1024 + loads * 16)
        self.assertEqual(limits["automatic_retries"], 0)
        self.assertEqual(limits["retained_bytes"], 32 * 1024 ** 3)

    def test_manual_route_does_not_claim_unobserved_provider_guarantees(self):
        editor = self.config["editor"]
        self.assertEqual((editor["requested_display_name"], editor["requested_mode"]), ("Astra", "Ultra"))
        for key in ("observed_model", "immutable_model_version", "observed_effort", "frontier_usage",
                    "marginal_billing", "provider_token_cap", "provider_billing_cap", "provider_cancellation_limit"):
            self.assertIsNone(editor[key], key)
        self.assertFalse(editor["planning_allowance_enforced"])
        self.assertFalse(editor["strict_automated_readiness_required"])
        self.assertEqual(editor["isolation"], "unverified")
        self.assertEqual(editor["max_response_bytes"], 262144)
        self.assertEqual(editor["max_submissions"], 4)

    def test_frozen_protocol_bytes_and_public_dataset_binding(self):
        protocol = ROOT / "docs/ifeval-browser-v2-protocol.md"
        expected = (ROOT / "docs/ifeval-browser-v2-protocol.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), expected)
        summary = json.loads((ROOT / "reports/ifeval-browser-v2/final-set-summary.json").read_text())
        self.assertEqual(self.config["final_set"]["manifest_sha256"], summary["manifest_sha256"])
        counts = summary["counts"]
        self.assertEqual((counts["prompts"], counts["instructions"]), (128, 160))
        self.assertEqual((counts["single_instruction_prompts"], counts["two_instruction_prompts"]), (96, 32))
        self.assertEqual(len(counts["instruction_types"]), 18)
        self.assertEqual(len(counts["instruction_families"]), 8)
        self.assertGreaterEqual(min(counts["instruction_types"].values()), 4)
        fixtures = summary["fixture_summary"]
        self.assertTrue(fixtures["all_passed"])
        self.assertEqual((fixtures["constructive_cases"], fixtures["negative_cases"], fixtures["combined_constraint_cases"]), (128, 160, 32))
        self.assertLessEqual(fixtures["max_rendered_input_tokens"] + 1024, 4096)
        self.assertLessEqual(fixtures["max_constructive_response_tokens"], 1020)
        self.assertTrue(summary["bundled_tokenizer_parity"])
        self.assertEqual(summary["exact_or_normalized_matches"], 0)
        self.assertEqual(summary["target_forward_passes"], 0)


class CurrentOfficialFreezeTests(unittest.TestCase):
    def test_current_zero_final_manifest_has_no_authored_denominator(self):
        config = json.loads((ROOT / 'configs/ifeval_browser_quality_resource.json').read_text())
        summary = json.loads((ROOT / 'reports/ifeval-browser-v2-narrowed/final-set-summary.json').read_text())
        self.assertEqual(config['final_set']['manifest_sha256'], summary['manifest_sha256'])
        self.assertEqual((summary['counts']['prompts'], summary['counts']['instructions']), (0, 0))
        self.assertFalse(summary['final_quality_available'])
        self.assertEqual(summary['audit_counts']['source_prompts'], 541)
        self.assertEqual(summary['audit_counts']['runtime_eligible_prompts'], 344)
        self.assertEqual(summary['audit_counts']['eligible_unexposed_prompts'], 0)
        self.assertEqual(summary['audit_counts']['exposure_counts'], {'exposed': 129, 'unresolved': 412})
        self.assertEqual(summary['target_forward_passes'], 0)
        self.assertNotIn('authorship', summary)

    def test_real_script_launcher_verifies_fixture_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'final-items.json').write_text('[]')
            manifest = {'files': {'final-items.json': hashlib.sha256(b'[]').hexdigest()},
                'items_sha256': canonical_hash([]), 'counts': {'prompts': 0, 'instructions': 0}}
            manifest['sha256'] = canonical_hash(manifest)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/freeze_ifeval_v2.py'),
                '--root', str(root), '--verify-only'], cwd=directory, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['counts'], {'prompts': 0, 'instructions': 0})


if __name__ == "__main__":
    unittest.main()
