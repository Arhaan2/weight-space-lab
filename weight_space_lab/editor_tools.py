"""Supervise trusted bounded checkpoint operations without exposing host tools.

The worker is controller-owned implementation, not editor-supplied code. It
receives only an already approved checkpoint/policy and one JSON operation.
Untrusted editor execution is separately contained by editor_sandbox.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import tempfile
import time


def _ensure_active(check_active, deadline):
    try:
        check_active()
    except Exception:
        raise ValueError("checkpoint action cancelled or inactive") from None
    if time.monotonic() >= deadline:
        raise ValueError("checkpoint action deadline exceeded")


def run_checkpoint_action(*, model, policy, exposure_log, output, tool, arguments,
                          timeout_seconds, check_active, expected_original_sha256=None):
    """Run inspect/propose with cancellation, hard timeout and crash cleanup.

    Proposals stay in a private staging directory until the supervised worker
    succeeds and the parent rechecks its active reservation. A controller crash
    leaves staging unaccepted; it never causes the operation to replay.
    """
    if tool not in {"inspect", "propose", "verify"} or type(arguments) is not dict:
        raise ValueError("only bounded checkpoint inspection and proposals are supported")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 3600 or not callable(check_active)):
        raise ValueError("checkpoint action requires an active bounded deadline")
    if tool == "propose" and (set(arguments) != {"proposal"} or type(arguments["proposal"]) is not dict):
        raise ValueError("proposal arguments are invalid")
    if tool == "verify" and (set(arguments) != {"candidate"} or type(arguments["candidate"]) is not bool):
        raise ValueError("verification arguments are invalid")
    if expected_original_sha256 is not None and (
            type(expected_original_sha256) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", expected_original_sha256)):
        raise ValueError("expected original identity must be a SHA256 digest")
    from .acquisition import _remove_stage
    from .editor_sandbox import _WATCHDOG
    deadline = time.monotonic() + timeout_seconds
    _ensure_active(check_active, deadline)
    destination = Path(output).absolute() if tool == "propose" else None
    if destination is not None:
        if destination.exists() or destination.is_symlink():
            raise ValueError("candidate destination must be new")
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-action-",
                                     dir=destination.parent if destination is not None else None)).resolve()
    temporary.chmod(0o700)
    reader = writer = -1
    process = None
    try:
        payload = {"model": str(Path(model).resolve()), "policy": policy,
                   "exposure_log": str(Path(exposure_log).absolute()),
                   "output": str(temporary / "candidate") if destination is not None else (
                       str(Path(output).absolute()) if tool == "verify" and arguments["candidate"] else None),
                   "tool": tool, "arguments": arguments,
                   "expected_original_sha256": expected_original_sha256}
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        if len(encoded) > 262144:
            raise ValueError("checkpoint action payload exceeds byte limit")
        incoming = temporary / "request.json"
        incoming.write_bytes(encoded)
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--checkpoint-worker"]
        reader, writer = os.pipe()
        _ensure_active(check_active, deadline)
        with incoming.open("rb") as stream:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _WATCHDOG,
                 str(reader), str(deadline), json.dumps(command)],
                stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=temporary, env={"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": str(temporary),
                                    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                pass_fds=(reader,), close_fds=True, start_new_session=True,
            )
        os.close(reader)
        reader = -1
        result_bytes = bytearray()
        total_bytes = 0
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, True)
            selector.register(process.stderr, selectors.EVENT_READ, False)
            while selector.get_map():
                _ensure_active(check_active, deadline)
                for key, _ in selector.select(min(0.025, max(0, deadline - time.monotonic()))):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > 262144:
                        raise RuntimeError("checkpoint worker output exceeded byte limit")
                    if key.data:
                        result_bytes.extend(chunk)
        code = process.wait(timeout=1)
        _ensure_active(check_active, deadline)
        if code == 124:
            raise ValueError("checkpoint action deadline exceeded")
        if code != 0:
            raise RuntimeError("checkpoint worker failed")
        try:
            result = json.loads(result_bytes)
            if type(result) is not dict or set(result) != {"ok", "result"} or type(result["ok"]) is not bool:
                raise ValueError
        except (ValueError, UnicodeError):
            raise RuntimeError("checkpoint worker returned invalid output") from None
        if not result["ok"]:
            raise ValueError("checkpoint operation rejected")
        if type(result["result"]) is not dict:
            raise RuntimeError("checkpoint worker returned invalid output")
        _ensure_active(check_active, deadline)
        if destination is not None:
            if destination.exists() or destination.is_symlink():
                raise ValueError("candidate destination appeared during execution")
            candidate = temporary / "candidate"
            # Darwin requires write permission on a moved directory itself.
            # Only its container mode changes; verified file bytes stay frozen.
            candidate.chmod(0o755)
            candidate.rename(destination)
            destination.chmod(0o555)
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return result["result"]
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("checkpoint worker execution failed") from None
    finally:
        # EOF on this sole parent-held lifeline makes the independent guardian
        # kill the entire worker process group, including on controller SIGKILL.
        for descriptor in (reader, writer):
            if descriptor >= 0:
                os.close(descriptor)
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for stream in (process.stdout, process.stderr):
                if stream:
                    stream.close()
        _remove_stage(temporary)


def _worker_main():
    # -I ignores cwd/PYTHONPATH. Only this fixed, trusted package location is
    # inserted so the implementation also works in an uninstalled checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from weight_space_lab.checkpoint import CheckpointEditor, verify_candidate
        payload = json.load(sys.stdin)
        editor = CheckpointEditor(payload["model"], payload["policy"],
                                  exposure_log=payload["exposure_log"])
        if (payload["expected_original_sha256"] is not None
                and editor.original_sha256 != payload["expected_original_sha256"]):
            raise ValueError("immutable original identity changed")
        if payload["tool"] == "inspect":
            result = editor.inspect(payload["arguments"])
            if payload["arguments"]["op"] == "metadata":
                result = {"tensors": {name: {key: info[key] for key in ("shape", "dtype", "parameter_count")}
                                      for name, info in result["tensors"].items()}}
        elif payload["tool"] == "propose":
            audit = editor.propose(payload["arguments"]["proposal"], payload["output"])
            result = {"original_sha256": audit["base_sha256"],
                      **{key: audit[key] for key in ("changed_stored_entries", "changed_numerical_entries", "targeted_coordinates")}}
        elif payload["tool"] == "verify":
            if payload["arguments"]["candidate"]:
                verify_candidate(payload["model"], payload["output"], payload["policy"])
            result = {"verified": True, "original_sha256": editor.original_sha256}
        else:
            raise ValueError("unsupported checkpoint action")
        print(json.dumps({"ok": True, "result": result}, allow_nan=False), flush=True)
    except Exception:
        # No paths, raw exception text or tracebacks enter editor-visible output.
        print(json.dumps({"ok": False, "result": None}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--checkpoint-worker"]:
        raise SystemExit("internal checkpoint worker only")
    _worker_main()
