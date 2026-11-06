"""Checkpoint broker operations execute in supervised, killable workers."""

from pathlib import Path
import tempfile
import time
import unittest

from test_checkpoint import EDITABLE, checkpoint_fixture, fixture_policy, files
from weight_space_lab.checkpoint import CheckpointEditor, verify_candidate
from weight_space_lab.editor_tools import run_checkpoint_action


class EditorToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.model = checkpoint_fixture(self.root)
        self.policy = fixture_policy()
        self.exposures = self.root / "exposures.jsonl"
        self.editor = CheckpointEditor(self.model, self.policy, exposure_log=self.exposures)
        self.before = files(self.model)

    def tearDown(self):
        # Published candidates are intentionally read-only.
        from weight_space_lab.acquisition import _remove_stage
        _remove_stage(self.root)
        self.temporary.cleanup()

    def run_action(self, tool, arguments, **overrides):
        options = {"model": self.model, "policy": self.policy, "exposure_log": self.exposures,
                   "output": self.root / "candidate", "tool": tool, "arguments": arguments,
                   "timeout_seconds": 5, "check_active": lambda: None,
                   "expected_original_sha256": self.editor.original_sha256}
        options.update(overrides)
        return run_checkpoint_action(**options)

    def proposal(self):
        return {"schema_version": 1, "base_sha256": self.editor.original_sha256,
                "provenance": "editor_proposal", "composition": "independent",
                "operations": [{"op": "scale_columns", "tensor": EDITABLE[0],
                                "columns": [0], "factor": 0.5}]}

    def test_metadata_projects_only_approved_fields_and_persists_exposure(self):
        result = self.run_action("inspect", {"op": "metadata", "tensors": [EDITABLE[0]]})
        self.assertEqual(set(result["tensors"][EDITABLE[0]]), {"shape", "dtype", "parameter_count"})
        self.assertEqual(len(self.editor.exposure_snapshot()), 2)
        self.assertEqual(files(self.model), self.before)

    def test_proposal_is_published_only_after_verified_worker_completion(self):
        result = self.run_action("propose", {"proposal": self.proposal()})
        self.assertEqual(result["original_sha256"], self.editor.original_sha256)
        self.assertEqual(result["changed_stored_entries"], 2)
        audit = verify_candidate(self.model, self.root / "candidate", self.policy)
        self.assertEqual(audit["changed_stored_entries"], result["changed_stored_entries"])
        self.assertEqual(files(self.model), self.before)
        self.assertFalse(list(self.root.glob(".checkpoint-action-*")))

    def test_invalid_proposal_rejected_without_candidate_or_private_details(self):
        proposal = self.proposal()
        proposal["operations"][0]["op"] = "optimizer_step"
        with self.assertRaisesRegex(ValueError, "checkpoint operation rejected") as error:
            self.run_action("propose", {"proposal": proposal})
        self.assertNotIn(str(self.model), str(error.exception))
        self.assertFalse((self.root / "candidate").exists())
        self.assertFalse(list(self.root.glob(".checkpoint-action-*")))
        self.assertEqual(files(self.model), self.before)

    def test_deadline_kills_worker_and_cleans_unaccepted_candidate(self):
        began = time.monotonic()
        with self.assertRaisesRegex(ValueError, "deadline exceeded"):
            self.run_action("propose", {"proposal": self.proposal()}, timeout_seconds=0.01)
        self.assertLess(time.monotonic() - began, 3)
        self.assertFalse((self.root / "candidate").exists())
        self.assertFalse(list(self.root.glob(".checkpoint-action-*")))
        self.assertEqual(files(self.model), self.before)

    def test_cancellation_during_supervision_kills_worker(self):
        checks = []
        def active():
            checks.append(time.monotonic())
            if len(checks) >= 3:
                raise ValueError("private controller detail")
        with self.assertRaisesRegex(ValueError, "cancelled or inactive") as error:
            self.run_action("propose", {"proposal": self.proposal()}, check_active=active)
        self.assertNotIn("private controller detail", str(error.exception))
        self.assertGreaterEqual(len(checks), 3)
        self.assertFalse((self.root / "candidate").exists())
        self.assertFalse(list(self.root.glob(".checkpoint-action-*")))

    def test_internal_verification_binds_original_and_saved_candidate(self):
        self.assertTrue(self.run_action("verify", {"candidate": False})["verified"])
        self.run_action("propose", {"proposal": self.proposal()})
        self.assertTrue(self.run_action("verify", {"candidate": True})["verified"])
        with self.assertRaisesRegex(ValueError, "operation rejected"):
            self.run_action("verify", {"candidate": False}, expected_original_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "operation rejected"):
            self.run_action("verify", {"candidate": True}, output=self.root / "missing")

    def test_no_arbitrary_worker_or_verification_arguments(self):
        for tool, arguments in (("shell", {}), ("verify", {"candidate": "path"}),
                                ("verify", {"candidate": False, "path": "/outside"}),
                                ("propose", {"proposal": {}, "python": "code"})):
            with self.subTest(tool=tool, arguments=arguments), self.assertRaises(ValueError):
                self.run_action(tool, arguments)


if __name__ == "__main__":
    unittest.main()
