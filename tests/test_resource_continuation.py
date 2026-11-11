"""A continuation must never replace a measured or ambiguously attempted session."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "resource_continuation", Path(__file__).resolve().parents[1] / "scripts/continue_resource_followup.py"
)
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


def stopped_state():
    cleanup = {key: True for key in (*continuation.base.CLEANUP, "native_capture_exit_confirmed")}
    cleanup["retained_owned_pids_alive"] = []
    sessions = [
        {"status": "completed", "observed": {"cleanup": copy.deepcopy(cleanup)}}
        for _ in range(6)
    ]
    sessions += [{"key": "07-v2-r2", "status": "stopped", "dispatch_absence_verified": True,
                  "error": "TimeoutExpired: frozen input materialization"}]
    sessions += [{"status": "pending"} for _ in range(3)]
    return {"halted": True, "active": None, "sessions": sessions}


class ContinuationAdmission(unittest.TestCase):
    def test_only_undispatched_position_is_eligible_without_mutating_parent(self):
        state = stopped_state()
        original = copy.deepcopy(state)
        with tempfile.TemporaryDirectory() as folder, patch.object(continuation, "ORIGINAL", Path(folder)):
            continuation.check_original(state)
        self.assertEqual(state, original)

    def test_any_prior_reservation_or_dispatch_evidence_prevents_replacement(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(continuation, "ORIGINAL", Path(folder)):
            for key, value in (("capacity_reserved", {}), ("load_granted", False), ("observed", {})):
                with self.subTest(key=key):
                    state = stopped_state()
                    state["sessions"][6][key] = value
                    with self.assertRaises(ValueError):
                        continuation.check_original(state)
            (Path(folder) / "sessions/07-v2-r2").mkdir(parents=True)
            with self.assertRaises(ValueError):
                continuation.check_original(stopped_state())

    def test_uncertain_cleanup_or_still_active_parent_prevents_continuation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(continuation, "ORIGINAL", Path(folder)):
            for mutate in (
                lambda state: state.update(active={"position": 7}),
                lambda state: state.update(halted=False),
                lambda state: state["sessions"][5]["observed"]["cleanup"].update(native_capture_exit_confirmed=False),
                lambda state: state["sessions"][6].update(dispatch_absence_verified=False),
                lambda state: state["sessions"][6].update(error="RuntimeError: target failed"),
            ):
                state = stopped_state()
                mutate(state)
                with self.assertRaises(ValueError):
                    continuation.check_original(state)


if __name__ == "__main__":
    unittest.main()
