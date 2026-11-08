"""Budget and safety checks with synthetic drivers only; no model inference."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from weight_space_lab import browser_check
from weight_space_lab.inventory import file_sha256
from weight_space_lab.protocol import canonical_hash


class BrowserCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "authorization-source.txt").write_text("Synthetic authorization fixture, not permission")
        self.write("authorization.json", {"authorization_source_sha256": file_sha256(self.root / "authorization-source.txt"),
                                         "historical_capacity_transferred": False})
        self.write("historical-preservation.json", {})
        self.write("historical-unverified.json", {"files": {}})
        browser_check.prepare(self.root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        for name in ("base.json", "edit.json"):
            self.write(name, {"configurationSha256": "a" * 64, "sourceCheckpointSha256": "b" * 64,
                "convertedArtifactSha256": "c" * 64, "items": [{}], "settings": {"maxTokens": 256}})
        self.frozen = {"benchmark": str(self.root), "arm_configs": [str(self.root / n) for n in ("base.json", "edit.json")]}

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def fake_process(self, command, **kwargs):
        arm = Path(command[-1])
        self.terminal(arm, 256)
        return SimpleNamespace(returncode=0, poll=lambda: 0)

    def terminal(self, arm, tokens):
        (arm / "result.json").write_text(json.dumps({"status": "completed", "generatedTokens": tokens,
            "configuration_sha256": "a" * 64, "source_checkpoint_sha256": "b" * 64,
            "converted_artifact_sha256": "c" * 64, "cleanup": {"unload_confirmed": True}}))

    def execute(self, *, pressure=1, process=None):
        with patch.object(browser_check, "check", return_value=(self.root, self.manifest, self.frozen)), \
             patch.object(browser_check, "_collector", return_value=SimpleNamespace(
                 host_signals=lambda: ({"hostMemoryPressureLevel": pressure, "hostThermalState": 0}, []))), \
             patch.object(browser_check.subprocess, "Popen", side_effect=process or self.fake_process) as launch, \
             patch.object(browser_check, "_stop_process", return_value={"controller_exit_verified": True}):
            result = browser_check.run(self.root)
        return result, launch

    def test_prepare_empty_fresh_ledger(self):
        with browser_check.ledger(self.root, self.manifest) as book:
            self.assertEqual(book.attempts(), [])
            self.assertEqual(book.summary()["remaining"]["requested_sequences"], 2)

    def test_authorization_mutation_rejected(self):
        (self.root / "authorization-source.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            browser_check.load(self.root)

    def test_two_arms_once_and_no_retry(self):
        result, launch = self.execute()
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(result["accounting"]["observed"]["generated_tokens"], 512)
        self.assertEqual(result["accounting"]["charged"]["requested_sequences"], 2)
        with self.assertRaisesRegex(ValueError, "retries forbidden"):
            self.execute()

    def test_host_stop_prevents_browser_launch_and_candidate(self):
        result, launch = self.execute(pressure=4)
        self.assertEqual(launch.call_count, 0)
        self.assertEqual(result["arms"][0]["generated_tokens"], 0)
        self.assertEqual(result["arms"][1]["status"], "unrun")
        self.assertTrue(result["accounting"]["cancelled"])

    def test_unknown_tokens_spend_full_reservation_and_no_second_arm(self):
        def failed(command, **kwargs):
            return SimpleNamespace(returncode=1, poll=lambda: 1)
        result, launch = self.execute(process=failed)
        self.assertEqual(launch.call_count, 1)
        self.assertIsNone(result["arms"][0]["generated_tokens"])
        self.assertEqual(result["accounting"]["charged"]["generated_tokens"], 256)
        self.assertEqual(result["arms"][1]["status"], "unrun")

    def test_token_overage_is_preserved_and_cancels(self):
        def overage(command, **kwargs):
            arm = Path(command[-1])
            self.terminal(arm, 257)
            return SimpleNamespace(returncode=0, poll=lambda: 0)
        # The ledger cancels after recording actual overage; no refund or clipping.
        result, launch = self.execute(process=overage)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(result["arms"][1]["status"], "unrun")
        with browser_check.ledger(self.root, self.manifest) as book:
            self.assertEqual(book.attempts()[0]["observed_generated_tokens"], 257)
            self.assertTrue(book.summary()["cancelled"])


if __name__ == "__main__":
    unittest.main()
