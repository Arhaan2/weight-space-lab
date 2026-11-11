"""Persistent CPU notebook for weight-space research, separate from model probes.

The trusted daemon owns the socket and receipts. Editor Python runs in a
different, Seatbelt-confined process: source/dependencies/probe files are read
only, only its work directory is writable, and network/process creation/GPU
access are not granted. Unsupported platforms fail closed. A parent-pipe
watchdog kills the worker if its daemon dies, including during editor code.

The OS boundary is distinct from the scientific contract. General Python cannot
prove that arbitrary algebra is not a manually implemented forward pass or
training algorithm. Known prohibited APIs are guarded and every code block is
recorded; no-training/no-unmetered-inference also requires review of those logs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback


DEFAULT_LIMITS = {
    "wall_seconds": 900,
    "max_rss_bytes": 8_589_934_592,
    "max_output_chars": 10_000,
    "max_code_bytes": 262_144,
    "idle_seconds": 3600,
    "session_seconds": 86_400,
    "startup_seconds": 120,
    "cpu_threads": 4,
}
_MAX_WIRE = 1_048_576
_OWNED_DAEMONS = {}


def _reap_owned():
    for path, process in list(_OWNED_DAEMONS.items()):
        if process.poll() is not None:
            process.wait()
            del _OWNED_DAEMONS[path]


def _alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class KernelUnavailable(RuntimeError):
    """The enforced scientific kernel could not be started."""


class KernelStopped(RuntimeError):
    """This kernel has ended; its in-memory state must not be silently replaced."""


def validate_limits(limits=None):
    supplied = limits or {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_LIMITS):
        raise ValueError("unknown research kernel limit")
    result = {**DEFAULT_LIMITS, **supplied}
    for name, maximum in DEFAULT_LIMITS.items():
        value = result[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return result


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _write(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_code(code, max_bytes=DEFAULT_LIMITS["max_code_bytes"]):
    """Reject known prohibited APIs; this is not an arbitrary-Python proof."""
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > max_bytes:
        raise ValueError("nonempty bounded Python source required")
    tree = ast.parse(code, filename="<astra-research>", mode="exec")
    blocked_modules = {"transformers", "vllm", "mlx", "requests", "urllib", "http", "socket",
                       "subprocess", "multiprocessing", "ctypes", "pickle", "cloudpickle", "dill"}
    blocked_attributes = {"backward", "autograd", "optim", "requires_grad_", "enable_grad",
                          "set_grad_enabled", "cuda", "mps", "from_pretrained", "generate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [value.name.split(".")[0] for value in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
            if names == ["torch"] and (set((node.module or "").split(".")) & blocked_attributes
                    or any(value.name in blocked_attributes for value in node.names)):
                raise ValueError("training or accelerator module is prohibited")
        else:
            names = []
        if set(names) & blocked_modules:
            raise ValueError("module belongs to metered inference or prohibited external execution")
        if isinstance(node, ast.Attribute) and node.attr in blocked_attributes:
            raise ValueError("training, accelerator, or unmetered model API is prohibited")
    return tree


def _paths_overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def _validate_paths(source, workdir, control_dir, read_paths):
    source, workdir, control_dir = (Path(path).resolve() for path in (source, workdir, control_dir))
    if not source.is_dir():
        raise ValueError("source checkpoint directory required")
    if any(_paths_overlap(a, b) for a, b in ((source, workdir), (source, control_dir), (workdir, control_dir))):
        raise ValueError("source, writable work, and trusted control must be disjoint")
    allowed = []
    for path in read_paths:
        path = Path(path).resolve()
        if not path.exists() or _paths_overlap(path, control_dir) or _paths_overlap(path, workdir):
            raise ValueError("read-only evidence paths must exist and exclude work/control")
        allowed.append(str(path))
    return source, workdir, control_dir, allowed


def start_kernel(source, workdir, control_dir, *, python_executable=None, read_paths=(), limits=None):
    """Start one persistent kernel. Control directories are single-use.

    ``read_paths`` should be separate practice/probe-view directories; never
    supply a parent containing the final evaluation or controller state.
    """
    _reap_owned()
    limits = validate_limits(limits)
    source, workdir, control_dir, read_paths = _validate_paths(source, workdir, control_dir, read_paths)
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise KernelUnavailable("macOS Seatbelt is unavailable; no unrestricted kernel fallback")
    if control_dir.exists() or control_dir.is_symlink():
        raise FileExistsError("kernel control directory must be new; a stopped kernel cannot be restarted")
    workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    control_dir.mkdir(parents=True, mode=0o700)
    socket_dir = Path(tempfile.mkdtemp(prefix="wsl-research-", dir="/private/tmp")).resolve()
    socket_dir.chmod(0o700)
    executable = os.path.abspath(python_executable or sys.executable)  # preserve virtualenv launcher
    configuration = {
        "schema_version": 1, "source": str(source), "workdir": str(workdir),
        "control_dir": str(control_dir), "socket_dir": str(socket_dir),
        "socket": str(socket_dir / "kernel.sock"), "token": secrets.token_hex(32),
        "read_paths": read_paths, "limits": limits,
    }
    _write(control_dir / "launch.json", configuration)
    command = [executable, "-I", "-B", str(Path(__file__).resolve()), "--daemon",
               str(control_dir / "launch.json")]
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(workdir),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
    with (control_dir / "daemon.stdout").open("xb") as stdout, (control_dir / "daemon.stderr").open("xb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   cwd=workdir, env=environment, start_new_session=True, close_fds=True)
    deadline = time.monotonic() + limits["startup_seconds"] + 5
    while time.monotonic() < deadline:
        state = control_dir / "state.json"
        if state.is_file():
            status = _json(state.read_bytes())
            if status["status"] == "ready":
                _OWNED_DAEMONS[str(control_dir)] = process
                return status
            if status["status"] in {"failed", "stopped"}:
                process.wait(timeout=5)
                raise KernelUnavailable(status.get("reason", "kernel setup failed"))
        if process.poll() is not None:
            raise KernelUnavailable("kernel daemon exited before readiness; inspect private daemon log")
        time.sleep(0.05)
    # Terminating the daemon closes its lifeline. Its independent watchdog
    # terminates the worker even when Python startup cannot return normally.
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    raise KernelUnavailable("kernel startup deadline exceeded")


def _rpc(control_dir, operation, **arguments):
    control = Path(control_dir)
    configuration = _json((control / "launch.json").read_bytes())
    request = {"token": configuration["token"], "operation": operation, **arguments}
    encoded = json.dumps(request, allow_nan=False).encode() + b"\n"
    if len(encoded) > _MAX_WIRE:
        raise ValueError("kernel request exceeds transport limit")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(configuration["limits"]["wall_seconds"] + 15)
        try:
            client.connect(configuration["socket"])
            client.sendall(encoded)
            raw = bytearray()
            while not raw.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    raise KernelStopped("kernel connection ended; no automatic restart")
                raw.extend(chunk)
                if len(raw) > _MAX_WIRE:
                    raise RuntimeError("kernel response exceeds transport limit")
        except (ConnectionError, FileNotFoundError, socket.timeout) as error:
            raise KernelStopped("kernel is unavailable; its in-memory state was not restarted") from error
    return _json(raw)


def execute_kernel(control_dir, code):
    # Rejections are returned and journaled by the trusted daemon. The caller
    # must reserve a code attempt before this operation, including rejected code.
    return _rpc(control_dir, "execute", code=code)


def kernel_status(control_dir):
    return _rpc(control_dir, "status")


def stop_kernel(control_dir):
    control = Path(control_dir)
    _reap_owned()
    configuration = _json((control / "launch.json").read_bytes())
    state = _json((control / "state.json").read_bytes())
    if state["status"] not in {"stopped", "failed"}:
        if _alive(state.get("daemon_pid")):
            # A stop RPC would queue behind a long-running code block. This
            # controller-only marker is checked by the supervisor while code
            # executes; the sandbox cannot read or write the control directory.
            _write(control / "stop-request.json", {"token": configuration["token"]})
        else:
            # A killed daemon cannot update its state. Its independent
            # watchdog still closes the worker; verify that outcome rather
            # than killing a potentially recycled PID or starting a replacement.
            deadline = time.monotonic() + 5
            while _alive(state.get("worker_pid")) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _alive(state.get("worker_pid")):
                raise KernelStopped("daemon died and worker cleanup could not be confirmed")
            state.update(status="failed", reason="daemon died; watchdog worker exit verified",
                         process_exit_confirmed=True)
            _write(control / "state.json", state)
            shutil.rmtree(configuration["socket_dir"], ignore_errors=True)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = _json((control / "state.json").read_bytes())
        if state["status"] in {"stopped", "failed"} and state.get("process_exit_confirmed"):
            process = _OWNED_DAEMONS.pop(str(control.resolve()), None)
            if process is not None:
                process.wait(timeout=2)
            return state
        time.sleep(0.02)
    raise KernelStopped("kernel stop acknowledged but cleanup not confirmed")


def _profile(source, workdir, read_paths):
    # No study/repository parent is exposed. The fixed package code is readable,
    # as are runtime/package locations from this selected Python environment.
    from weight_space_lab.editor_sandbox import _quoted, _runtime
    executable, standard_library, prefix = _runtime()
    package = Path(__file__).resolve().parent
    runtime_paths = {standard_library, prefix, package}
    for value in sys.path:
        if value and Path(value).exists() and ("site-packages" in value or "lib-dynload" in value):
            runtime_paths.add(Path(value).resolve())
    readonly = [source, *[Path(value) for value in read_paths], *runtime_paths]
    reads = [f"({'subpath' if p.is_dir() else 'literal'} {_quoted(p)})" for p in readonly]
    # FileFinder lists this one directory to find the approved package. This
    # grants the directory itself, not any repository siblings or descendants.
    reads.append(f"(literal {_quoted(package.parent)})")
    reads += ['(subpath "/System/Library")', '(subpath "/usr/lib")',
              '(literal "/dev/null")', '(literal "/dev/urandom")', '(literal "/dev/random")',
              '(literal "/")']
    ancestors = {parent for p in [*readonly, workdir, executable] for parent in p.parents}
    metadata = " ".join(f"(literal {_quoted(p)})" for p in sorted(ancestors))
    profile = "\n".join([
        "(version 1)", "(deny default)",
        f"(allow file-read-metadata {metadata})", "(allow sysctl-read)",
        f"(allow process-exec (literal {_quoted(executable)}))",
        "(allow file-read* " + " ".join(reads) + ")",
        f"(allow file-read* file-write* (subpath {_quoted(workdir)}))",
        '(allow file-write-data (literal "/dev/null"))',
        "(deny network*)", "(deny process-fork)",
    ])
    return profile, executable, [str(p) for p in runtime_paths]


def _read_worker(process, seconds, max_rss, *, worker_pid=None, cancelled=None):
    started, peak, raw, errors = time.monotonic(), 0, bytearray(), bytearray()
    with selectors.DefaultSelector() as poll:
        poll.register(process.stdout, selectors.EVENT_READ, raw)
        poll.register(process.stderr, selectors.EVENT_READ, errors)
        while time.monotonic() - started < seconds:
            if cancelled is not None and cancelled():
                raise KernelStopped("explicit stop requested")
            if worker_pid:
                observed = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(worker_pid)],
                                          capture_output=True, text=True, timeout=2)
                value = observed.stdout.strip()
                if value.isdigit():
                    peak = max(peak, int(value) * 1024)
                    if peak > max_rss:
                        raise MemoryError("kernel sampled RSS allowance exceeded")
                elif process.poll() is None:
                    raise RuntimeError("kernel memory measurement unavailable")
            for key, _ in poll.select(0.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("kernel worker exited or closed its protocol stream")
                key.data.extend(chunk)
                if len(raw) + len(errors) > _MAX_WIRE:
                    raise RuntimeError("kernel emitted excessive unframed output")
                if b"\n" in raw:
                    # Worker output is one JSON line; arbitrary direct writes to
                    # stdout cause a protocol failure, never trusted metadata.
                    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
                        raise RuntimeError("kernel protocol was corrupted")
                    return _json(raw), peak, errors.decode("utf-8", errors="replace")
            if process.poll() is not None:
                raise RuntimeError("kernel worker ended")
    raise TimeoutError("kernel code wall-time allowance exhausted")


def _receive(client):
    client.settimeout(5)
    raw = bytearray()
    while not raw.endswith(b"\n"):
        value = client.recv(65536)
        if not value:
            raise ValueError("empty kernel request")
        raw.extend(value)
        if len(raw) > _MAX_WIRE:
            raise ValueError("kernel request too large")
    return _json(raw)


def _run_daemon(configuration):
    from weight_space_lab.editor_sandbox import _WATCHDOG
    source, work = Path(configuration["source"]), Path(configuration["workdir"])
    control, limits = Path(configuration["control_dir"]), configuration["limits"]
    reader = writer = -1
    process = None
    started, last_used, sequence = time.monotonic(), time.monotonic(), 0
    state = {"schema_version": 1, "status": "starting", "daemon_pid": os.getpid(),
             "worker_pid": None, "control_dir": str(control), "workdir": str(work),
             "limits": limits, "isolation": "macos-seatbelt", "automatic_restart": False,
             "scientific_contract_enforcement": "known-API guards plus recorded-code review",
             "process_exit_confirmed": False}

    def cancelled():
        path = control / "stop-request.json"
        if not path.is_file():
            return False
        value = _json(path.read_bytes())
        return secrets.compare_digest(str(value.get("token", "")), configuration["token"])

    _write(control / "state.json", state)
    try:
        profile, executable, package_paths = _profile(source, work, configuration["read_paths"])
        profile_path = control / "sandbox.sb"
        profile_path.write_text(profile)
        profile_path.chmod(0o600)
        command = ["/usr/bin/sandbox-exec", "-f", str(profile_path), str(executable), "-I", "-S", "-B",
                   str(Path(__file__).resolve()), "--worker"]
        reader, writer = os.pipe()
        process = subprocess.Popen(
            [str(executable), "-I", "-S", "-B", "-c", _WATCHDOG, str(reader),
             str(started + limits["session_seconds"]), json.dumps(command)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=work, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(work),
                          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                          "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(limits["cpu_threads"]),
                          "OPENBLAS_NUM_THREADS": str(limits["cpu_threads"])},
            pass_fds=(reader,), close_fds=True, start_new_session=True)
        os.close(reader)
        reader = -1
        payload = {"source": str(source), "workdir": str(work), "package_paths": package_paths,
                   "read_paths": configuration["read_paths"], "limits": limits}
        process.stdin.write(json.dumps(payload).encode() + b"\n")
        process.stdin.flush()
        ready, _, startup_errors = _read_worker(process, limits["startup_seconds"], limits["max_rss_bytes"],
                                                cancelled=cancelled)
        if ready.get("status") != "ready" or type(ready.get("worker_pid")) is not int:
            raise KernelUnavailable("worker dependency or checkpoint initialization failed: " + str(ready.get("error", "")))
        state.update(ready)
        state["startup_stderr"] = startup_errors[:limits["max_output_chars"]]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(configuration["socket"])
            os.chmod(configuration["socket"], 0o600)
            server.listen(1)
            server.settimeout(0.2)
            _write(control / "state.json", state)
            while True:
                if cancelled():
                    state.update(status="stopped", reason="explicit stop")
                    break
                if process.poll() is not None:
                    raise KernelStopped("persistent worker ended; no automatic restart")
                if time.monotonic() - last_used > limits["idle_seconds"]:
                    state.update(status="stopped", reason="idle allowance exhausted")
                    break
                if time.monotonic() - started > limits["session_seconds"]:
                    state.update(status="stopped", reason="session lifetime allowance exhausted")
                    break
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                with client:
                    should_stop = False
                    try:
                        request = _receive(client)
                        if not secrets.compare_digest(str(request.get("token", "")), configuration["token"]):
                            raise ValueError("kernel client authentication failed")
                        operation = request.get("operation")
                        if operation == "status" and set(request) == {"token", "operation"}:
                            result = dict(state)
                        elif operation == "stop" and set(request) == {"token", "operation"}:
                            result = {"status": "stopping"}
                            state.update(status="stopped", reason="explicit stop")
                            should_stop = True
                        elif operation == "execute" and set(request) == {"token", "operation", "code"}:
                            sequence += 1
                            code = request["code"]
                            attempt = {"sequence": sequence, "code": code,
                                       "code_sha256": _hash(code) if isinstance(code, str) else None}
                            _write(control / f"code-{sequence:04}.json", attempt)
                            try:
                                validate_code(code, limits["max_code_bytes"])
                            except (ValueError, SyntaxError) as error:
                                result = {"status": "rejected", "error": str(error), "output": "",
                                          "truncated": False, "worker_state_retained": True}
                            else:
                                began = time.monotonic()
                                process.stdin.write(json.dumps({"code": code, "sequence": sequence}).encode() + b"\n")
                                process.stdin.flush()
                                try:
                                    result, peak, stderr = _read_worker(process, limits["wall_seconds"],
                                        limits["max_rss_bytes"], worker_pid=state["worker_pid"], cancelled=cancelled)
                                    if result.get("sequence") != sequence or result.get("status") not in {"ok", "error"}:
                                        raise RuntimeError("kernel response does not match submitted code")
                                    result.update(elapsed_seconds=time.monotonic() - began,
                                                  sampled_peak_rss_bytes=peak, worker_state_retained=True)
                                    if stderr:
                                        result["native_stderr"] = stderr[:limits["max_output_chars"]]
                                except Exception as error:
                                    result = {"status": "terminated", "error": str(error),
                                              "elapsed_seconds": time.monotonic() - began,
                                              "worker_state_retained": False}
                                    state.update(status="stopped" if isinstance(error, KernelStopped) else "failed",
                                                 reason=str(error))
                                    should_stop = True
                            result.update(sequence=sequence, code_sha256=attempt["code_sha256"])
                            result["output_sha256"] = _hash(result.get("output", ""))
                            _write(control / f"result-{sequence:04}.json", result)
                            state["executed_attempts"] = sequence
                            _write(control / "state.json", state)
                            last_used = time.monotonic()
                        else:
                            raise ValueError("invalid kernel operation")
                    except Exception as error:
                        result = {"status": "rejected", "error": str(error)}
                    try:
                        client.sendall(json.dumps(result, allow_nan=False).encode() + b"\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # daemon still journals work if a client goes away
                    if should_stop:
                        break
    except Exception as error:
        state.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        for descriptor in (reader, writer):
            if descriptor >= 0:
                os.close(descriptor)
        if process is not None:
            # Close the lifeline before waiting: watchdog owns worker process
            # group termination and reaping. Do not kill watchdog first.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state["cleanup_error"] = "watchdog did not confirm worker termination"
            else:
                state["process_exit_confirmed"] = True
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    stream.close()
        if state["status"] not in {"stopped", "failed"}:
            state.update(status="failed", reason="daemon ended unexpectedly")
        _write(control / "state.json", state)
        shutil.rmtree(configuration["socket_dir"], ignore_errors=True)


class _BoundedText(io.TextIOBase):
    def __init__(self, limit):
        self.limit, self.parts, self.count, self.truncated = limit, [], 0, False

    def write(self, value):
        value = str(value)
        remainder = max(0, self.limit - self.count)
        self.parts.append(value[:remainder]) if remainder else None
        self.count += min(len(value), remainder)
        self.truncated |= len(value) > remainder
        return len(value)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)


def _worker_main():
    import contextlib
    import importlib.metadata
    import resource
    incoming, outgoing = sys.stdin, sys.stdout
    configuration = _json(incoming.buffer.readline(_MAX_WIRE))
    # The real framework binary uses -S: import only predeclared site paths.
    for path in configuration["package_paths"]:
        if path not in sys.path:
            sys.path.insert(0, path)
    package_parent = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, package_parent)
    limits = configuration["limits"]
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4_294_967_296, 4_294_967_296))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    try:
        import numpy as np
        import torch
        from tokenizers import Tokenizer
        from weight_space_lab.research_checkpoint import EditWorkspace
        torch.set_num_threads(limits["cpu_threads"])
        torch.set_grad_enabled(False)
        torch.set_default_device("cpu")
        workspace = EditWorkspace(configuration["source"], configuration["workdir"])
        tokenizer = Tokenizer.from_file(str(Path(configuration["source"]) / "tokenizer.json"))

        def load_original(name):
            return torch.from_numpy(workspace.load_original(name)).clone()

        def load(name):
            return torch.from_numpy(workspace.load(name)).clone()

        def save_edit(name, value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu" or value.requires_grad:
                    raise ValueError("edits must be CPU tensors without gradients")
                value = value.detach().float().numpy()
            return workspace.save_edit(name, value)

        namespace = {"__name__": "__astra_research__", "np": np, "torch": torch,
                     "NAMES": workspace.NAMES, "load_original": load_original, "load": load,
                     "save_edit": save_edit, "drop_edit": workspace.drop_edit,
                     "edited": workspace.edited, "tok": tokenizer,
                     "WORK": Path(configuration["workdir"]),
                     "READ_PATHS": tuple(Path(path) for path in configuration["read_paths"])}
        installed_versions = {}
        for distribution in ("scipy", "scikit-learn", "pandas", "sympy", "networkx",
                             "safetensors", "tokenizers", "torch", "numpy"):
            try:
                installed_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                installed_versions[distribution] = None
        ready = {"status": "ready", "worker_pid": os.getpid(),
                 "source_sha256": workspace.original_sha256,
                 "tensor_count": len(workspace.NAMES), "torch_version": torch.__version__,
                 "numpy_version": np.__version__, "tokenizer_interface": "tokenizers.Tokenizer",
                 "installed_versions": installed_versions}
        outgoing.write(json.dumps(ready) + "\n")
        outgoing.flush()
    except Exception as error:
        outgoing.write(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}) + "\n")
        outgoing.flush()
        return
    while True:
        line = incoming.buffer.readline(_MAX_WIRE)
        if not line:
            break
        request = _json(line)
        captured = _BoundedText(limits["max_output_chars"])
        result = {"sequence": request["sequence"], "status": "ok", "error": None}
        try:
            tree = validate_code(request["code"], limits["max_code_bytes"])
            with torch.no_grad(), contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    expression = ast.Expression(tree.body.pop().value)
                    exec(compile(tree, "<astra-research>", "exec"), namespace)
                    value = eval(compile(expression, "<astra-research>", "eval"), namespace)
                    namespace["_"] = value
                    if value is not None:
                        print(repr(value))
                else:
                    exec(compile(tree, "<astra-research>", "exec"), namespace)
        except BaseException as error:
            result.update(status="error", error=f"{type(error).__name__}: {str(error)[:2000]}")
        result.update(output=captured.getvalue(), truncated=captured.truncated)
        outgoing.write(json.dumps(result, allow_nan=False) + "\n")
        outgoing.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    # Both internal entry points execute this fixed module under -I.
    if args.daemon:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        _run_daemon(_json(Path(args.daemon).read_bytes()))
    elif args.worker:
        _worker_main()
    else:
        parser.error("internal entry point; use research_study for metered execution")


if __name__ == "__main__":
    main()
