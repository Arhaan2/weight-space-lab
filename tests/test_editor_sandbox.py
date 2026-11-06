import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from weight_space_lab.editor_sandbox import (
    SandboxCancelled, SandboxExecutionError, SandboxRunner, SandboxTimeout,
    SandboxUnavailable, sandbox_diagnostics, verify_denials,
)


class SandboxContractTests(unittest.TestCase):
    def test_unavailable_platform_never_runs_an_unrestricted_fallback(self):
        with patch("weight_space_lab.editor_sandbox.sys.platform", "unsupported"), \
                patch("weight_space_lab.editor_sandbox.subprocess.Popen") as launch:
            with self.assertRaises(SandboxUnavailable):
                SandboxRunner().run("print('{}')", {}, timeout_seconds=1)
            launch.assert_not_called()

    def test_inventory_is_not_denial_evidence(self):
        self.assertFalse(sandbox_diagnostics()["boundary_verified"])
        self.assertFalse(sandbox_diagnostics()["scientific_result"])

    def test_precancelled_request_never_launches(self):
        with patch("weight_space_lab.editor_sandbox.subprocess.Popen") as launch:
            with self.assertRaises(SandboxCancelled):
                SandboxRunner().run("print('{}')", {}, timeout_seconds=1, cancelled=lambda: True)
            launch.assert_not_called()

    def test_invalid_bounds_and_nonfinite_payload_rejected_before_launch(self):
        with patch("weight_space_lab.editor_sandbox.subprocess.Popen") as launch:
            for timeout in (True, 0, -1, float("nan"), float("inf"), 3601):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    SandboxRunner().run("print('{}')", {}, timeout_seconds=timeout)
            with self.assertRaises(ValueError):
                SandboxRunner().run("print('{}')", {"n": float("nan")}, timeout_seconds=1)
            launch.assert_not_called()


class ActualSandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            SandboxRunner().run("print('{}')", {}, timeout_seconds=5)
        except SandboxUnavailable as error:
            raise unittest.SkipTest(str(error))

    def test_actual_denials_on_host_canary_and_active_network_listener(self):
        with patch.dict(os.environ, {"WSL_PRIVATE_SENTINEL": "must-not-inherit"}):
            result = verify_denials()
        self.assertTrue(result["boundary_verified"])
        self.assertGreaterEqual(len(result["checks"]), 10)
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["scientific_result"])
        self.assertFalse(result["experiment_executed"])

    def test_approved_json_handoff_works_without_project_imports(self):
        source = "import json,sys; payload=json.load(sys.stdin); print(json.dumps({'example':payload['example']}))"
        self.assertEqual(SandboxRunner().run(source, {"example": "development only"},
                                           timeout_seconds=3), {"example": "development only"})
        with self.assertRaises(SandboxExecutionError):
            SandboxRunner().run("import weight_space_lab; print('{}')", {}, timeout_seconds=3)

    def test_wall_timeout_is_enforced_without_worker_cooperation(self):
        start = time.monotonic()
        with self.assertRaises(SandboxTimeout):
            SandboxRunner().run("import time; time.sleep(10); print('{}')", {}, timeout_seconds=0.3)
        self.assertLess(time.monotonic() - start, 3)

    def test_cancellation_terminates_running_worker(self):
        start = time.monotonic()
        with self.assertRaises(SandboxCancelled):
            SandboxRunner().run("import time; time.sleep(10); print('{}')", {}, timeout_seconds=5,
                                cancelled=lambda: time.monotonic() - start > 0.25)
        self.assertLess(time.monotonic() - start, 3)

    def test_output_flood_is_bounded_and_terminated(self):
        with self.assertRaisesRegex(SandboxExecutionError, "byte limit"):
            SandboxRunner().run("while True: print('x'*1000, flush=True)", {},
                                timeout_seconds=5, max_output_bytes=2000)

    def test_invalid_response_objects_are_rejected(self):
        for output in ('[]', '{"a":1,"a":2}', '{"a":NaN}', 'not json'):
            with self.subTest(output=output), self.assertRaises(SandboxExecutionError):
                SandboxRunner().run("print(" + repr(output) + ")", {}, timeout_seconds=3)

    def test_controller_sigkill_cannot_leave_worker_running(self):
        # A fresh controller's only pipe writer vanishes on SIGKILL. The
        # independent watchdog must kill its sleeping worker without relying
        # on cleanup code in the controller or voluntary worker cancellation.
        with tempfile.TemporaryDirectory(prefix="wsl-watchdog-test-") as temporary:
            worker_source = (
                "import os,pathlib,time\n"
                "pathlib.Path('worker.pid').write_text(str(os.getpid()))\n"
                "time.sleep(20)\nprint('{}')\n"
            )
            controller = (
                "from weight_space_lab.editor_sandbox import SandboxRunner\n"
                f"SandboxRunner().run({worker_source!r}, {{}}, timeout_seconds=10)\n"
            )
            environment = dict(os.environ, TMPDIR=temporary)
            process = subprocess.Popen([sys.executable, "-c", controller], env=environment,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                pid_files = []
                while time.monotonic() < deadline:
                    pid_files = list(Path(temporary).glob("wsl-editor-*/worker.pid"))
                    if pid_files:
                        break
                    if process.poll() is not None:
                        self.fail("controller failed before worker started")
                    time.sleep(0.02)
                self.assertTrue(pid_files, "worker never started")
                worker_pid = int(pid_files[0].read_text())
                process.kill()
                process.wait(timeout=2)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        os.kill(worker_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("orphan worker survived controller SIGKILL")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
