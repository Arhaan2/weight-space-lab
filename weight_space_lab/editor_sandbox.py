"""Fail-closed OS boundary for development-only editor integration workers.

The host controller is trusted. Its deliberately selected worker receives only
JSON on stdin, a fresh private scratch directory, the Python standard library,
and OS runtime files. It receives no repository mount, credentials, network,
package imports, subprocess permission, or caller file descriptors. A separate
trusted watchdog kills the worker on timeout *and* controller death.

This is an integration transport, not an Astra API or a scientific model.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Callable


class SandboxUnavailable(RuntimeError):
    pass


class SandboxExecutionError(RuntimeError):
    pass


class SandboxTimeout(SandboxExecutionError):
    pass


class SandboxCancelled(SandboxExecutionError):
    pass


# This process remains outside the untrusted worker's sandbox. Closing the sole
# parent-held pipe writer, including by SIGKILL, terminates the whole worker
# process group. No polling of recycled parent PIDs is involved.
_WATCHDOG = r'''
import json, os, selectors, signal, subprocess, sys, time
fd = int(sys.argv[1])
deadline = float(sys.argv[2])
command = json.loads(sys.argv[3])
os.set_blocking(fd, False)
watch = selectors.DefaultSelector()
watch.register(fd, selectors.EVENT_READ)
child = subprocess.Popen(command, stdin=sys.stdin, stdout=sys.stdout,
                         stderr=sys.stderr, close_fds=True, start_new_session=True)
try:
    while child.poll() is None:
        if time.monotonic() >= deadline:
            sys.exit(124)
        if watch.select(min(0.05, max(0, deadline - time.monotonic()))):
            if os.read(fd, 1) == b'':
                sys.exit(125)
    sys.exit(child.returncode if child.returncode >= 0 else 126)
finally:
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()
'''


def _quoted(path: Path | str) -> str:
    # JSON string escaping is also valid for the sandbox profile string atoms.
    return json.dumps(str(path), ensure_ascii=True)


def _runtime() -> tuple[Path, Path, Path]:
    executable = Path(sys.executable).resolve()
    standard_library = Path(sysconfig.get_path("stdlib")).resolve()
    prefix = Path(sys.base_prefix).resolve()
    # Homebrew's bin/python is a trampoline using posix_spawn. Start the real
    # framework binary so a zero process-creation hard limit is compatible.
    framework_binary = prefix / "Resources/Python.app/Contents/MacOS/Python"
    if sys.platform == "darwin" and framework_binary.is_file():
        executable = framework_binary
    return executable, standard_library, prefix


def sandbox_diagnostics() -> dict:
    """Inventory only; successful hostile probes are separate evidence."""
    executable, _, _ = _runtime()
    present = sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()
    return {
        "schema_version": 1,
        "implementation": "macos-seatbelt-stdlib-worker-v1",
        "platform": sys.platform,
        "launcher_present": present,
        "python_version": sys.version.split()[0],
        "python_executable_name": executable.name,
        "boundary_verified": False,
        "network": "deny",
        "subprocesses": "deny",
        "host_filesystem": "deny_except_explicit_runtime_and_private_scratch",
        "watchdog": "independent_process_with_parent_pipe_and_monotonic_deadline",
        "scientific_result": False,
    }


def _profile(scratch: Path) -> str:
    executable, standard_library, prefix = _runtime()
    # Reads of the base runtime are enough under -I -S. Do not allow all of
    # /opt/homebrew, a virtualenv, /Users, /tmp, or the caller's project.
    readonly = [
        f"(literal {_quoted(executable)})",
        f"(literal {_quoted(prefix / 'Python')})",
        f"(subpath {_quoted(standard_library)})",
        '(subpath "/System/Library")',
        '(subpath "/usr/lib")',
        '(literal "/dev/null")',
        '(literal "/dev/urandom")',
        '(literal "/dev/random")',
        '(literal "/")',  # dyld requires the root directory itself, no descendants
    ]
    ancestors = {parent for path in (executable, standard_library, prefix, scratch,
                                    Path("/System/Library"), Path("/usr/lib"))
                 for parent in path.parents}
    metadata = " ".join(f"(literal {_quoted(path)})" for path in sorted(ancestors))
    return "\n".join([
        "(version 1)", "(deny default)",
        f"(allow file-read-metadata {metadata})",
        "(allow sysctl-read)",
        f"(allow process-exec (literal {_quoted(executable)}))",
        "(allow file-read* " + " ".join(readonly) + ")",
        f"(deny file-read* (subpath {_quoted(standard_library / 'site-packages')}))",
        f"(allow file-read* file-write* (subpath {_quoted(scratch)}))",
        '(allow file-write-data (literal "/dev/null"))',
        "(deny network*)", "(deny process-fork)",
    ])


def _json_object(raw: bytes) -> dict:
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("nonfinite JSON")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("worker response must be a JSON object")
    return result


class SandboxRunner:
    """Run only a controller-selected source; never accept editor code as tools."""

    def run(self, worker_source: str, payload: dict, *, timeout_seconds: float,
            cancelled: Callable[[], bool] | None = None,
            max_output_bytes: int = 262144) -> dict:
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600):
            raise ValueError("sandbox timeout must be finite and in (0, 3600]")
        if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 4_194_304:
            raise ValueError("invalid output byte limit")
        if not isinstance(worker_source, str) or len(worker_source.encode()) > 262144:
            raise ValueError("invalid worker source")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        encoded = json.dumps(payload, allow_nan=False).encode()
        if len(encoded) > 262144:
            raise ValueError("payload exceeds byte limit")
        if cancelled and cancelled():
            raise SandboxCancelled("execution cancelled before launch")
        if not sandbox_diagnostics()["launcher_present"]:
            raise SandboxUnavailable("enforced sandbox unavailable; unsandboxed fallback prohibited")

        executable, _, _ = _runtime()
        deadline = time.monotonic() + timeout_seconds
        with tempfile.TemporaryDirectory(prefix="wsl-editor-") as temporary:
            scratch = Path(temporary).resolve()
            scratch.chmod(0o700)
            worker = scratch / "worker.py"
            # File-size, fd, and CPU ceilings complement the wall-time watchdog.
            bootstrap = (
                "import resource\n"
                f"resource.setrlimit(resource.RLIMIT_CPU, ({math.ceil(timeout_seconds) + 1}, {math.ceil(timeout_seconds) + 1}))\n"
                "resource.setrlimit(resource.RLIMIT_FSIZE, (4194304, 4194304))\n"
                "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
                "resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))\n"
            )
            worker.write_text(bootstrap + worker_source, encoding="utf-8")
            profile = scratch / "sandbox.sb"
            profile.write_text(_profile(scratch), encoding="utf-8")
            request = scratch / "request.json"
            request.write_bytes(encoded)
            command = ["/usr/bin/sandbox-exec", "-f", str(profile),
                       str(executable), "-I", "-S", "-B", str(worker)]
            read_fd, write_fd = os.pipe()
            process = None
            reason = None
            output, errors = bytearray(), bytearray()
            try:
                with request.open("rb") as incoming:
                    process = subprocess.Popen(
                        [str(executable), "-I", "-S", "-c", _WATCHDOG,
                         str(read_fd), str(deadline), json.dumps(command)],
                        stdin=incoming, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=scratch, env={"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": str(scratch)},
                        pass_fds=(read_fd,), close_fds=True, start_new_session=True,
                    )
                os.close(read_fd)
                read_fd = -1
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, output)
                    selector.register(process.stderr, selectors.EVENT_READ, errors)
                    while selector.get_map():
                        if reason is None:
                            if cancelled and cancelled():
                                reason = SandboxCancelled("execution cancelled")
                            elif time.monotonic() >= deadline:
                                reason = SandboxTimeout("sandbox execution deadline exceeded")
                            if reason is not None:
                                os.close(write_fd)
                                write_fd = -1
                        for key, _ in selector.select(0.05):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            target = key.data
                            if len(output) + len(errors) + len(chunk) > max_output_bytes:
                                if reason is None:
                                    reason = SandboxExecutionError("worker output exceeds byte limit")
                                    os.close(write_fd)
                                    write_fd = -1
                            else:
                                target.extend(chunk)
                    code = process.wait(timeout=1)
                if reason:
                    raise reason
                if code == 124:
                    raise SandboxTimeout("sandbox execution deadline exceeded")
                if code != 0:
                    # Do not turn arbitrary error output into editor feedback.
                    if b"sandbox_init" in errors or b"sandbox-exec:" in errors:
                        raise SandboxUnavailable("OS rejected sandbox launch; no fallback executed")
                    raise SandboxExecutionError(f"isolated worker exited unsuccessfully ({code})")
                try:
                    return _json_object(bytes(output))
                except (ValueError, UnicodeDecodeError) as error:
                    raise SandboxExecutionError("worker returned invalid JSON object") from error
            finally:
                for fd in (read_fd, write_fd):
                    if fd >= 0:
                        os.close(fd)
                if process is not None:
                    # Closing the lifeline lets the watchdog kill the sandbox.
                    # Do not kill the watchdog first and orphan its worker.
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    if process.stdout:
                        process.stdout.close()
                    if process.stderr:
                        process.stderr.close()


_DENIAL_PROBE = r'''
import errno, json, os, pathlib, socket, subprocess, sys
payload = json.load(sys.stdin)
result = {}
def denied(name, action, accepted=(errno.EACCES, errno.EPERM)):
    try:
        action()
    except OSError as error:
        result[name] = error.errno in accepted
    else:
        result[name] = False
secret = pathlib.Path(payload["forbidden_file"])
denied("absolute_read_denied", lambda: secret.read_bytes())
denied("metadata_denied", lambda: secret.stat())
denied("traversal_read_denied", lambda: pathlib.Path(os.path.relpath(secret)).read_bytes())
link = pathlib.Path("escape-link")
link.symlink_to(secret)
denied("symlink_read_denied", lambda: link.read_bytes())
denied("filesystem_write_denied", lambda: secret.write_text("overwritten"))
denied("subprocess_denied", lambda: subprocess.run(["/bin/cat", str(secret)], check=True, capture_output=True), (errno.EACCES, errno.EPERM, errno.EAGAIN))
denied("python_subprocess_denied", lambda: subprocess.run([sys.executable, "-I", "-S", "-c", "pass"], check=True, capture_output=True), (errno.EACCES, errno.EPERM, errno.EAGAIN))
denied("inherited_descriptor_denied", lambda: os.read(payload["forbidden_fd"], 1), (errno.EBADF, errno.EACCES, errno.EPERM))
def connect():
    with socket.socket() as client:
        client.settimeout(0.2)
        client.connect(("127.0.0.1", payload["listening_port"]))
denied("network_denied", connect)
result["environment_clean"] = "WSL_PRIVATE_SENTINEL" not in os.environ
print(json.dumps(result))
'''


def verify_denials(*, timeout_seconds: float = 5.0) -> dict:
    """Exercise an actual boundary with a host-only sentinel and listening socket.

    Failure or unavailable host support is explicit. The sentinel is never a
    benchmark asset; no final-test file is opened by this verification.
    """
    import socket

    with tempfile.TemporaryDirectory(prefix="wsl-evaluator-only-") as temporary:
        sentinel = Path(temporary) / "final-test-sentinel.json"
        sentinel.write_text("HOST-ONLY CANARY; NOT BENCHMARK DATA", encoding="utf-8")
        with sentinel.open("rb") as forbidden, socket.socket() as listener:
            os.set_inheritable(forbidden.fileno(), True)
            try:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
            except OSError as error:
                raise SandboxUnavailable("host could not prepare active loopback canary") from error
            checks = SandboxRunner().run(
                _DENIAL_PROBE,
                {"forbidden_file": str(sentinel.resolve()),
                 "forbidden_fd": forbidden.fileno(),
                 "listening_port": listener.getsockname()[1]},
                timeout_seconds=timeout_seconds,
            )
        checks["sentinel_preserved"] = sentinel.read_text() == "HOST-ONLY CANARY; NOT BENCHMARK DATA"
    if not all(value is True for value in checks.values()):
        raise SandboxExecutionError("enforced boundary denial verification failed")
    return {"schema_version": 1, "boundary_verified": True, "checks": checks,
            "implementation": sandbox_diagnostics()["implementation"],
            "scientific_result": False, "experiment_executed": False}
