"""Persistent notebook lifecycle and OS boundary checks, without model inference."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from weight_space_lab.research_kernel import (
    KernelStopped, KernelUnavailable, _BoundedText, _profile, _validate_paths,
    execute_kernel, kernel_status, start_kernel, stop_kernel, validate_code, validate_limits,
)


class KernelContractTests(unittest.TestCase):
    def test_general_weight_algebra_is_allowed_but_known_training_and_forward_apis_are_rejected(self):
        validate_code("import numpy as np\nw=load(NAMES[0]); u=torch.ones(w.shape[0]);\nu=u/u.norm(); w=w-6*torch.outer(u,u@w)")
        for code in ("import transformers", "from torch import optim", "torch.optim.Adam([])",
                     "x.backward()", "torch.cuda.empty_cache()", "model.generate()",
                     "import subprocess", "torch.set_grad_enabled(True)"):
            with self.subTest(code=code):
                # Importing a training module from torch needs the same guard
                # as reaching it through an attribute.
                with self.assertRaises(ValueError):
                    validate_code(code)

    def test_limits_and_code_bounds(self):
        for value in ({"unknown": 1}, {"wall_seconds": True}, {"max_output_chars": 10001}):
            with self.assertRaises(ValueError):
                validate_limits(value)
        with self.assertRaises(ValueError):
            validate_code("x" * 11, max_bytes=10)
        with self.assertRaises(ValueError):
            validate_code(" ")

    def test_work_control_source_and_evidence_must_be_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for work, control, evidence in ((source / "edit", root / "ctrl", []),
                                            (root / "edit", root / "edit/ctrl", []),
                                            (root / "edit", root / "ctrl", [root])):
                with self.assertRaises(ValueError):
                    _validate_paths(source, work, control, evidence)

    def test_unsupported_platform_never_launches_unrestricted_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            with patch("weight_space_lab.research_kernel.sys.platform", "linux"), \
                    patch("weight_space_lab.research_kernel.subprocess.Popen") as launch:
                with self.assertRaises(KernelUnavailable):
                    start_kernel(source, root / "work", root / "control")
                launch.assert_not_called()

    def test_output_capture_retains_only_allowed_characters(self):
        output = _BoundedText(8)
        self.assertEqual(output.write("abcd"), 4)
        self.assertEqual(output.write("efghijkl"), 8)
        for _ in range(1000):
            output.write("more")
        self.assertEqual(output.getvalue(), "abcdefgh")
        self.assertTrue(output.truncated)
        self.assertEqual(len(output.parts), 2)


class PersistentKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "darwin":
            raise unittest.SkipTest("Seatbelt kernel is macOS only")
        candidate = os.environ.get("WSL_KERNEL_TEST_PYTHON")
        if candidate is None:
            candidate = sys.executable if importlib.util.find_spec("torch") else str(
                Path(__file__).resolve().parents[1] / ".venv/bin/python")
        if not Path(candidate).exists():
            raise unittest.SkipTest("local CPU scientific environment unavailable")
        try:
            check = subprocess.run([candidate, "-c", "import torch,tokenizers,numpy"],
                                   capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise unittest.SkipTest("local CPU scientific dependency import exceeded 30 seconds")
        if check.returncode:
            raise unittest.SkipTest("local CPU scientific dependencies unavailable")
        cls.python = candidate

    def setUp(self):
        from test_checkpoint import checkpoint_fixture
        self.temporary = tempfile.TemporaryDirectory(prefix="wsl-research-kernel-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = checkpoint_fixture(self.root)
        (self.source / "tokenizer.json").write_text(json.dumps({
            "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
            "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None, "decoder": None,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1, "b": 2}, "unk_token": "[UNK]"},
        }))
        self.control, self.work = self.root / "control", self.root / "work"
        self.started = False
        self.addCleanup(self.cleanup_kernel)

    def cleanup_kernel(self):
        if self.started:
            try:
                stop_kernel(self.control)
            except KernelStopped:
                pass

    def start(self, **limits):
        try:
            ready = start_kernel(self.source, self.work, self.control, python_executable=self.python,
                                 limits={"wall_seconds": 5, **limits})
        except KernelUnavailable as error:
            # A host policy can prohibit nested Seatbelt even on macOS.
            detail = (self.control / "daemon.stderr").read_text() if (self.control / "daemon.stderr").exists() else ""
            if "sandbox_init" in detail or "sandbox-exec:" in detail:
                raise unittest.SkipTest(str(error))
            raise
        self.started = True
        return ready

    def test_persistent_variables_tensor_edits_and_no_silent_restart(self):
        ready = self.start()
        self.assertEqual(ready["isolation"], "macos-seatbelt")
        self.assertEqual(set(ready["installed_versions"]), {
            "scipy", "scikit-learn", "pandas", "sympy", "networkx", "safetensors", "tokenizers", "torch", "numpy"})
        self.assertEqual(ready["installed_versions"]["torch"], ready["torch_version"])
        response = execute_kernel(self.control,
            "name='model.language_model.layers.0.mlp.down_proj.weight'\n"
            "w=load(name); w[0,0]+=1; save_edit(name,w)\nanswer=40")
        self.assertEqual(response["status"], "ok", response)
        second = execute_kernel(self.control, "answer += 2\n(answer, load(name)[0,0].item(), edited(), tok.encode('a b').ids)")
        self.assertEqual(second["status"], "ok", second)
        self.assertIn("(42, 2.0,", second["output"])
        self.assertIn("[1, 2]", second["output"])
        self.assertEqual(kernel_status(self.control)["worker_pid"], ready["worker_pid"])
        self.assertTrue((self.control / "code-0002.json").is_file())
        self.assertEqual(second["code_sha256"], json.loads((self.control / "code-0002.json").read_text())["code_sha256"])
        stopped = stop_kernel(self.control)
        self.assertTrue(stopped["process_exit_confirmed"])
        with self.assertRaises(KernelStopped):
            execute_kernel(self.control, "answer")
        with self.assertRaises(FileExistsError):
            start_kernel(self.source, self.work, self.control, python_executable=self.python)

    def test_actual_source_control_network_and_process_denials(self):
        self.start()
        source_before = (self.source / "config.json").read_bytes()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            probes = [
                f"from pathlib import Path\nPath({str(self.source / 'config.json')!r}).write_text('changed')",
                f"from pathlib import Path\nPath({str(self.control / 'launch.json')!r}).read_text()",
                "__import__('socket').socket().connect(('127.0.0.1', %d))" % listener.getsockname()[1],
                "__import__('subprocess').run(['/bin/echo','forbidden'])",
            ]
            for code in probes:
                with self.subTest(code=code):
                    result = execute_kernel(self.control, code)
                    self.assertEqual(result["status"], "error", result)
                    self.assertTrue("PermissionError" in result["error"] or "BlockingIOError" in result["error"], result)
        self.assertEqual((self.source / "config.json").read_bytes(), source_before)

    def test_exception_preserves_variables_and_output_is_bounded(self):
        self.start(max_output_chars=32)
        response = execute_kernel(self.control, "value=9\nprint('x'*1000)\nraise ValueError('diagnostic failure')")
        self.assertEqual(response["status"], "error", response)
        self.assertEqual(len(response["output"]), 32)
        self.assertTrue(response["truncated"])
        self.assertEqual(execute_kernel(self.control, "value")["output"].strip(), "9")

    def test_timeout_terminates_entire_kernel_without_retry(self):
        ready = self.start(wall_seconds=1)
        result = execute_kernel(self.control, "while True: pass")
        self.assertEqual(result["status"], "terminated", result)
        self.assertFalse(result["worker_state_retained"])
        state = stop_kernel(self.control)
        self.assertEqual(state["status"], "failed")
        self.assertTrue(state["process_exit_confirmed"])
        with self.assertRaises(ProcessLookupError):
            os.kill(ready["worker_pid"], 0)

    def test_daemon_death_cannot_orphan_worker(self):
        ready = self.start()
        os.kill(ready["daemon_pid"], signal.SIGKILL)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            try:
                os.kill(ready["worker_pid"], 0)
            except ProcessLookupError:
                state = stop_kernel(self.control)
                self.assertEqual(state["status"], "failed")
                self.assertTrue(state["process_exit_confirmed"])
                return
            time.sleep(0.05)
        self.fail("watchdog left worker alive after daemon SIGKILL")

    def test_explicit_stop_interrupts_a_running_block_before_its_deadline(self):
        self.start(wall_seconds=30)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(execute_kernel, self.control, "while True: pass")
            deadline = time.monotonic() + 3
            while not (self.control / "code-0001.json").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue((self.control / "code-0001.json").exists())
            began = time.monotonic()
            stopped = stop_kernel(self.control)
            self.assertLess(time.monotonic() - began, 4)
            self.assertEqual(stopped["status"], "stopped")
            self.assertTrue(stopped["process_exit_confirmed"])
            result = pending.result(timeout=2)
            self.assertEqual(result["status"], "terminated")
            self.assertFalse(result["worker_state_retained"])


if __name__ == "__main__":
    unittest.main()
