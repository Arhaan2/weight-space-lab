import importlib.metadata
import json
import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from weight_space_lab import diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_default_never_imports_or_probes_optional_backends(self):
        with patch.object(diagnostics, "_probe_torch") as probe, \
                patch.object(diagnostics, "_memory", return_value={}), \
                patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            result = diagnostics.collect_diagnostics()
        probe.assert_not_called()
        self.assertEqual(result["backends"]["torch"], {"status": "not_probed"})
        self.assertEqual(result["model_execution"], "not_attempted")
        self.assertEqual(result["dependencies"]["torch"]["status"], "not_installed")
        self.assertIsNone(result["dependencies"]["torch"]["version"])

    def test_versions_are_metadata_only_and_sanitized(self):
        versions = ["2.3.5", "2.6.0+cu124", "1.2.3rc1", "0.5.0.post2.dev1"]
        for version in versions:
            self.assertEqual(diagnostics._version(version), version)
        sensitive = os.sep + "private-runtime" + os.sep + "access-token"
        for version in (sensitive, "token=secret", "1.0+secret", "latest", None, True):
            self.assertIsNone(diagnostics._version(version))
        with patch.object(importlib.metadata, "version", return_value=sensitive):
            report = diagnostics._dependencies()
        self.assertNotIn(sensitive, json.dumps(report))
        self.assertEqual(report["torch"]["status"], "installed")
        self.assertEqual(report["torch"]["version_status"], "unreportable")

    def test_metadata_exception_text_is_never_reported(self):
        with patch.object(importlib.metadata, "version", side_effect=RuntimeError("secret-token")):
            report = diagnostics._dependencies()
        self.assertNotIn("secret-token", json.dumps(report))
        self.assertEqual(report["torch"]["status"], "metadata_error")

    def test_host_fields_use_allowlists(self):
        with patch.object(diagnostics.platform, "system", return_value="private-system"), \
                patch.object(diagnostics.platform, "machine", return_value="private-hostname"), \
                patch.object(diagnostics.platform, "python_version", return_value="secret-value"), \
                patch.object(diagnostics, "_memory", return_value={}), \
                patch.object(diagnostics, "_dependencies", return_value={}):
            result = diagnostics.collect_diagnostics()
        self.assertEqual(result["system"], "other")
        self.assertEqual(result["cpu"]["architecture"], "other")
        self.assertIsNone(result["python_version"])
        self.assertNotIn("secret", json.dumps(result))

    def test_linux_memory_is_bytes_and_uses_available_not_free(self):
        content = "MemTotal: 1000 kB\nMemFree: 7 kB\nMemAvailable: 300 kB\n"
        with patch.object(diagnostics.Path, "read_text", return_value=content):
            result = diagnostics._memory("Linux")
        self.assertEqual(result["host_total_bytes"], 1024000)
        self.assertEqual(result["host_available_bytes"], 307200)
        self.assertEqual(result["available_basis"], "kernel_mem_available")

    def test_mac_memory_is_explicit_estimate(self):
        pages = ("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                 "Pages free: 2.\nPages inactive: 3.\nPages speculative: 1.\n")
        with patch.object(diagnostics, "_command", side_effect=["1048576\n", pages]):
            result = diagnostics._memory("Darwin")
        self.assertEqual(result["host_total_bytes"], 1048576)
        self.assertEqual(result["host_available_bytes"], 6 * 4096)
        self.assertIn("estimate", result["available_basis"])

    def test_unavailable_and_inconsistent_memory_stays_unknown(self):
        with patch.object(diagnostics.os, "sysconf", side_effect=ValueError), \
                patch.object(diagnostics.Path, "read_text", side_effect=OSError("secret")):
            result = diagnostics._memory("Linux")
        self.assertIsNone(result["host_total_bytes"])
        self.assertIsNone(result["host_available_bytes"])
        with patch.object(diagnostics.Path, "read_text", return_value="MemTotal: 1 kB\nMemAvailable: 2 kB\n"):
            result = diagnostics._memory("Linux")
        self.assertIsNone(result["host_available_bytes"])

    def probe_output(self):
        return {"status": "complete", "cpu": {"available": True},
                "cuda": {"available": True, "devices": [{"index": 0, "free_bytes": 100, "total_bytes": 200}]},
                "mps": {"available": False, "recommended_max_bytes": None, "allocated_bytes": None}}

    def test_explicit_probe_has_timeout_and_isolated_process(self):
        output = self.probe_output()
        with patch.object(diagnostics.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(output))) as run:
            result = diagnostics._probe_torch(3)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cuda"]["devices"][0]["device"], "cuda:0")
        self.assertEqual(run.call_args.kwargs["timeout"], 3)
        self.assertEqual(run.call_args.args[0][1], "-I")

    def test_probe_discards_device_names_credentials_paths_and_extra_fields(self):
        output = self.probe_output()
        output["secret"] = "api-token"
        output["cuda"]["devices"][0]["name"] = os.sep + "private-gpu-driver"
        result = diagnostics._sanitize_probe(output)
        encoded = json.dumps(result)
        for secret in ("api-token", "private-gpu-driver", "secret", "name"):
            self.assertNotIn(secret, encoded)

    def test_invalid_numeric_observations_become_unknown(self):
        for bad in (True, -1, float("nan"), float("inf"), "12"):
            output = self.probe_output()
            output["cuda"]["devices"][0]["free_bytes"] = bad
            output["mps"]["recommended_max_bytes"] = bad
            output["mps"]["available"] = bad
            result = diagnostics._sanitize_probe(output)
            self.assertIsNone(result["cuda"]["devices"][0]["free_bytes"])
            self.assertIsNone(result["mps"]["recommended_max_bytes"])
            if type(bad) is not bool:
                self.assertIsNone(result["mps"]["available"])
            json.dumps(result, allow_nan=False)

    def test_malformed_probe_does_not_claim_success(self):
        invalid = [None, [], {"status": "complete"}, {"status": "ready"}]
        duplicated = self.probe_output()
        duplicated["cuda"]["devices"] *= 2
        invalid.append(duplicated)
        for output in invalid:
            self.assertEqual(diagnostics._sanitize_probe(output)["status"], "invalid_probe_output")

    def test_probe_errors_are_bounded_and_redacted(self):
        for exception, expected in ((subprocess.TimeoutExpired("secret-command", 1), "timeout"),
                                    (OSError("secret-path"), "probe_error")):
            with patch.object(diagnostics.subprocess, "run", side_effect=exception):
                result = diagnostics._probe_torch(1)
            self.assertEqual(result, {"status": expected})
        for text in ("secret-output", "x" * 65537):
            with patch.object(diagnostics.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=text)):
                self.assertEqual(diagnostics._probe_torch(1), {"status": "invalid_probe_output"})

    def test_timeout_and_flag_contract(self):
        for timeout in (True, 0, -1, 31, 10 ** 1000, float("nan"), float("inf"), "1"):
            with self.assertRaises(ValueError):
                diagnostics.collect_diagnostics(timeout_seconds=timeout)
        with self.assertRaises(ValueError):
            diagnostics.collect_diagnostics(probe_backends=1)


if __name__ == "__main__":
    unittest.main()
