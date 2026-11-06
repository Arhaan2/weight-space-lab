"""Path-free environment observations; never evidence of model execution.

The default path uses only the standard library and installed distribution
metadata. Optional accelerator inspection runs in a bounded child process.
No model is loaded, no tensor is allocated, and no network request is made.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


DEPENDENCIES = (
    "numpy", "torch", "transformers", "tokenizers", "safetensors",
    "accelerate", "huggingface-hub", "vllm", "mlx", "mlx-lm", "math-verify",
)
_VERSION = re.compile(
    r"\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?"
    r"(?:\+(?:cpu|cu\d+|rocm\d+(?:\.\d+)*))?", re.ASCII
)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _version(value):
    return value if isinstance(value, str) and len(value) <= 80 and _VERSION.fullmatch(value) else None


def _dependencies():
    result = {}
    for name in DEPENDENCIES:
        try:
            version = _version(importlib.metadata.version(name))
            result[name] = {"status": "installed", "version": version,
                            "version_status": "reported" if version else "unreportable"}
        except importlib.metadata.PackageNotFoundError:
            result[name] = {"status": "not_installed", "version": None,
                            "version_status": "unavailable"}
        except Exception:
            result[name] = {"status": "metadata_error", "version": None,
                            "version_status": "unavailable"}
    return result


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=2, check=False)
        return result.stdout if result.returncode == 0 and len(result.stdout) <= 65536 else ""
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""


def _memory(system):
    total = available = None
    basis = "unavailable"
    try:
        total = _count(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    if system == "Linux":
        try:
            fields = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                match = re.fullmatch(r"(MemTotal|MemAvailable):\s+(\d+) kB", line)
                if match:
                    fields[match[1]] = int(match[2]) * 1024
            total = fields.get("MemTotal", total)
            available = fields.get("MemAvailable")
            if available is not None:
                basis = "kernel_mem_available"
        except (OSError, UnicodeError, ValueError):
            pass
    elif system == "Darwin":
        raw_total = _command(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip()
        if raw_total.isascii() and raw_total.isdigit():
            total = int(raw_total)
        raw_pages = _command(["/usr/bin/vm_stat"])
        page_size = re.search(r"page size of (\d+) bytes", raw_pages)
        page_counts = [re.search(rf"^{name}:\s+(\d+)\.", raw_pages, re.MULTILINE)
                       for name in ("Pages free", "Pages inactive", "Pages speculative")]
        if page_size and all(page_counts):
            available = int(page_size[1]) * sum(int(match[1]) for match in page_counts)
            basis = "free_plus_inactive_plus_speculative_estimate"
    if total is not None and available is not None and available > total:
        available, basis = None, "unavailable"
    return {"host_total_bytes": total, "host_available_bytes": available,
            "available_basis": basis,
            "scope": "host memory snapshot; not a process limit or model-fit guarantee"}


_TORCH_PROBE = r'''
import contextlib
import io
import json
result = {"status": "probe_error"}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        import torch
        result = {"status": "complete", "cpu": {"available": True},
                  "cuda": {"available": False, "devices": []},
                  "mps": {"available": False, "recommended_max_bytes": None,
                          "allocated_bytes": None}}
        try:
            result["cuda"]["available"] = bool(torch.cuda.is_available())
            if result["cuda"]["available"]:
                for index in range(min(torch.cuda.device_count(), 64)):
                    free = total = None
                    try:
                        free, total = torch.cuda.mem_get_info(index)
                    except Exception:
                        pass
                    result["cuda"]["devices"].append({"index": index,
                                                      "free_bytes": free, "total_bytes": total})
        except Exception:
            result["cuda"] = {"available": None, "devices": []}
        try:
            available = bool(torch.backends.mps.is_available())
            result["mps"]["available"] = available
            if available:
                for key, method in (("recommended_max_bytes", "recommended_max_memory"),
                                    ("allocated_bytes", "current_allocated_memory")):
                    try:
                        result["mps"][key] = getattr(torch.mps, method)()
                    except Exception:
                        pass
        except Exception:
            result["mps"]["available"] = None
    except ImportError:
        result = {"status": "dependency_unavailable"}
    except Exception:
        pass
print(json.dumps(result, allow_nan=False))
'''


def _availability(value):
    return value if type(value) is bool else None


def _sanitize_probe(value):
    if not isinstance(value, dict):
        return {"status": "invalid_probe_output"}
    status = value.get("status")
    if status in ("dependency_unavailable", "probe_error"):
        return {"status": status}
    if status != "complete":
        return {"status": "invalid_probe_output"}
    cpu, cuda, mps = (value.get(key) for key in ("cpu", "cuda", "mps"))
    if not all(isinstance(backend, dict) for backend in (cpu, cuda, mps)):
        return {"status": "invalid_probe_output"}
    devices = cuda.get("devices")
    if not isinstance(devices, list) or len(devices) > 64:
        return {"status": "invalid_probe_output"}
    cleaned = []
    seen = set()
    for device in devices:
        if not isinstance(device, dict):
            return {"status": "invalid_probe_output"}
        index = _count(device.get("index"))
        if index is None or index in seen or index >= 64:
            return {"status": "invalid_probe_output"}
        seen.add(index)
        free, total = _count(device.get("free_bytes")), _count(device.get("total_bytes"))
        if free is not None and total is not None and free > total:
            free = None
        cleaned.append({"device": f"cuda:{index}", "free_bytes": free, "total_bytes": total})
    return {"status": "complete", "cpu": {"available": _availability(cpu.get("available"))},
            "cuda": {"available": _availability(cuda.get("available")), "devices": cleaned},
            "mps": {"available": _availability(mps.get("available")),
                    "recommended_max_bytes": _count(mps.get("recommended_max_bytes")),
                    "allocated_bytes": _count(mps.get("allocated_bytes"))}}


def _probe_torch(timeout_seconds):
    try:
        process = subprocess.run([sys.executable, "-I", "-c", _TORCH_PROBE],
                                 capture_output=True, text=True, timeout=timeout_seconds, check=False)
        if process.returncode != 0:
            return {"status": "probe_error"}
        if len(process.stdout) > 65536:
            return {"status": "invalid_probe_output"}
        return _sanitize_probe(json.loads(process.stdout))
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return {"status": "probe_error"}
    except (ValueError, TypeError):
        return {"status": "invalid_probe_output"}


def collect_diagnostics(*, probe_backends=False, timeout_seconds=10.0):
    """Return sanitized observations without importing optional model libraries.

    ``probe_backends`` explicitly enables a torch child process. Its CPU,
    CUDA and MPS availability flags describe runtime discovery only. Even a
    completed probe cannot establish architecture/dtype support or model fit.
    Unknown memory remains null. Exceptions, environment variables, package
    locations, hostname, user name, and provider device names are omitted.
    """
    if type(probe_backends) is not bool:
        raise ValueError("probe_backends must be a boolean")
    if (type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 30
            or not math.isfinite(timeout_seconds)):
        raise ValueError("timeout_seconds must be finite and in (0, 30]")
    system = platform.system()
    if system not in {"Linux", "Darwin", "Windows"}:
        system = "other"
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64", "x86_64", "amd64", "i386", "i686"}:
        machine = "other"
    return {
        "schema_version": 1,
        "kind": "execution_diagnostics",
        "python_version": _version(platform.python_version()),
        "system": system,
        "cpu": {"available": True, "architecture": machine, "logical_count": _count(os.cpu_count())},
        "memory": _memory(system),
        "dependencies": _dependencies(),
        "backends": {
            "torch": _probe_torch(timeout_seconds) if probe_backends else {"status": "not_probed"},
            "browser": {"status": "not_checked", "reason": "requires a separate browser runtime session"},
        },
        "model_execution": "not_attempted",
        "verification_scope": "environment discovery only; model architecture, precision, performance, and protocol readiness are unverified",
    }
