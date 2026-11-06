"""Audit installed CLI controls against a synthetic, loopback-only provider.

No real provider or model is contacted. This diagnostic's network restriction
is not the Phase 7 editor filesystem-isolation boundary. Only structural
evidence is exported; prompts, auth headers, session IDs, and local paths are not.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time


NETWORK_PROFILE = ('(version 1)(allow default)(deny network-outbound)'
                   '(allow network-outbound (remote ip "localhost:*"))')
CASES = {
    "baseline": [],
    "max_output_tokens_config": ["max_output_tokens=16"],
    "model_max_output_tokens_config": ["model_max_output_tokens=16"],
    "max_tokens_config": ["max_tokens=16"],
    "max_cost_config": ["max_cost=1"],
    "max_billed_units_config": ["max_billed_units=1"],
    "token_budget_feature": ["features.token_budget=true"],
    "rollout_budget_feature": [
        "features.rollout_budget={limit_tokens=16,reminder_at_remaining_tokens=[8]}"],
    "tool_switches": ["features.shell_tool=false", "features.unified_exec=false",
        "features.view_image=false", "features.goals=false", "features.sleep_tool=false",
        "features.workspace_dependencies=false", "tools.update_plan.enabled=false",
        "tools.experimental_request_user_input.enabled=false"],
    "ambiguous_drop_zero_retries": ["features.unbounded_connection_retries=false"],
}
TOOL_MARKERS = ("exec_command", "apply_patch", "view_image", "write_stdin", "web__run",
                "request_user_input", "create_goal", "list_mcp_resources")
MAX_CAPTURE_BYTES = 4_194_304
MAX_PROCESS_OUTPUT_BYTES = 1_048_576


def _structural_request(body, *, authorization_present):
    if not isinstance(body, dict):
        return {"invalid_json_object": True}
    text = json.dumps(body.get("input", []), sort_keys=True)
    def identifier(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) else "unrecognized"
    def token_bound(value):
        return value if type(value) is int and 0 <= value < 2**63 else None
    return {
        "request_path": "/v1/responses", "requested_model": "gpt-6-astra" if body.get("model") == "gpt-6-astra" else None,
        "keys": sorted(identifier(key) for key in body),
        "max_output_tokens": token_bound(body.get("max_output_tokens")),
        "max_tokens": token_bound(body.get("max_tokens")),
        "tool_choice": body.get("tool_choice") if body.get("tool_choice") in ("auto", "none", "required") else "other",
        "tools": [identifier(tool.get("name", tool.get("type"))) for tool in body.get("tools", [])
                  if isinstance(tool, dict)],
        "authorization_present": authorization_present,
        "body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "input_tool_markers": {name: name in text for name in TOOL_MARKERS},
        "instruction_budget_markers": {name: name in text
            for name in ("rollout_budget", "token_budget", "remaining_tokens")},
    }


def _capture_server():
    """Bind only loopback and never echo the submitted request content."""
    state = {"case": None, "requests": [], "invalid_request": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/v1/responses" or not 0 < length <= MAX_CAPTURE_BYTES:
                    raise ValueError
                self.connection.settimeout(3)
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError
                body = json.loads(payload)
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, OSError):
                state["invalid_request"] = True
                self.send_response(400)
                self.end_headers()
                return
            state["requests"].append(_structural_request(
                body, authorization_present="Authorization" in self.headers))
            if state["case"] == "ambiguous_drop_zero_retries":
                self.connection.shutdown(2)
                self.connection.close()
                return
            response = json.dumps({"error": {"message": "Synthetic local capture stopped before model inference.",
                                             "type": "invalid_request_error"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    class Server(http.server.ThreadingHTTPServer):
        def handle_error(self, *_):
            state["invalid_request"] = True  # Suppress private traceback/peer details.

    server = Server(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server, state


def _bounded_process(command, *, environment, timeout_seconds):
    """Bound diagnostic wall time and captured bytes; never export raw streams."""
    process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               close_fds=True, start_new_session=True)
    deadline = time.monotonic() + timeout_seconds
    output = {"stdout": bytearray(), "stderr": bytearray()}
    reason = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                if time.monotonic() >= deadline:
                    reason = "local_timeout"
                    break
                for key, _ in selector.select(0.05):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if sum(map(len, output.values())) + len(chunk) > MAX_PROCESS_OUTPUT_BYTES:
                        reason = "local_output_limit"
                        break
                    output[key.data].extend(chunk)
                if reason is not None:
                    break
        if reason is None:
            code = process.wait(timeout=max(0.05, deadline - time.monotonic()))
        else:
            code = reason
    except subprocess.TimeoutExpired:
        code = "local_timeout"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        process.stdout.close()
        process.stderr.close()
    return code, bytes(output["stdout"]), bytes(output["stderr"])


def audit(selected_cases, *, timeout_seconds=12):
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise ValueError("loopback-only OS boundary unavailable; no unrestricted fallback")
    located = shutil.which("codex")
    if located is None:
        raise ValueError("installed Codex CLI not found")
    binary = Path(located).resolve()
    results = []
    with tempfile.TemporaryDirectory(prefix="phase7-cli-control-") as temporary:
        root = Path(temporary).resolve()
        config = root / "version-config"
        config.mkdir()
        # CODEX_HOME is used solely in its documented configuration-directory
        # role, with no copied auth/config. No user environment is inherited.
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C",
                       "CODEX_HOME": str(config), "TMPDIR": str(root)}
        code, stdout, _ = _bounded_process(
            ["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE, str(binary), "--version"],
            environment=environment, timeout_seconds=timeout_seconds)
        version = stdout.decode(errors="replace").strip()
        if code != 0 or not re.fullmatch(r"codex-cli [0-9A-Za-z.+-]+", version):
            raise ValueError("CLI or OS boundary version check failed; no requests sent")
        server, state = _capture_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            common = ["model_provider=\"phase7_capture\"",
                "model_providers.phase7_capture.name=\"Synthetic loopback capture; no model\"",
                f"model_providers.phase7_capture.base_url=\"http://127.0.0.1:{port}/v1\"",
                "model_providers.phase7_capture.wire_api=\"responses\"",
                "model_providers.phase7_capture.requires_openai_auth=false",
                "model_providers.phase7_capture.request_max_retries=0",
                "model_providers.phase7_capture.stream_max_retries=0",
                "model_providers.phase7_capture.stream_idle_timeout_ms=3000",
                "analytics.enabled=false", "feedback.enabled=false", "web_search=\"disabled\"",
                "cli_auth_credentials_store=\"file\"",
                "features.apps=false", "features.plugins=false", "features.hooks=false",
                "features.multi_agent=false", "features.browser_use=false", "features.computer_use=false",
                "features.image_generation=false", "features.enable_request_compression=false",
                "features.daemon_auto_start=false", "features.shell_snapshot=false"]
            for case in selected_cases:
                config, cwd = root / case / "config", root / case / "approved"
                config.mkdir(parents=True)
                cwd.mkdir()
                environment["CODEX_HOME"] = str(config)
                command = [str(binary), "--no-daemon", "exec", "--ignore-user-config", "--ignore-rules",
                    "--strict-config", "--skip-git-repo-check", "--ephemeral", "--json",
                    "--model", "gpt-6-astra", "--cd", str(cwd)]
                for setting in common + CASES[case]:
                    command.extend(["-c", setting])
                command.append("Synthetic transport serialization check. Do not call tools. Return no content.")
                state.update(case=case, requests=[], invalid_request=False)
                code, _, stderr = _bounded_process(
                    ["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE] + command,
                    environment=environment, timeout_seconds=timeout_seconds)
                # Export only an allowlisted field name from known parser errors.
                unknown = re.findall(rb"unknown configuration field `([a-zA-Z0-9_.]+)`", stderr)
                results.append({"case": case, "configuration_changes": CASES[case], "exit_code": code,
                    "requests": list(state["requests"]), "invalid_request": state["invalid_request"],
                    "unknown_configuration_fields": [field.decode("ascii") for field in unknown]})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    return {"schema_version": 1, "kind": "synthetic_cli_control_capture",
        "cli_version": version, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": results, "network_profile": NETWORK_PROFILE,
        "editor_filesystem_boundary_verified": False, "live_editor_interaction": False,
        "provider_model_called": False, "scientific_result": False, "experiment_executed": False,
        "requested_selector": "gpt-6-astra", "live_accepted_selector": None,
        "observed_live_model_metadata": None, "immutable_model_version": None,
        "interpretation": "Local parsing/serialization and synthetic responses only; not real provider acceptance or enforcement."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new sanitized evidence file")
    parser.add_argument("--case", choices=list(CASES), action="append", dest="cases",
                        help="run selected cases; defaults to all ten")
    parser.add_argument("--timeout-seconds", type=int, default=12, choices=range(1, 61))
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("output must be a new file")
    try:
        result = audit(list(dict.fromkeys(args.cases or CASES)), timeout_seconds=args.timeout_seconds)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if len(encoded.encode()) > MAX_PROCESS_OUTPUT_BYTES:
            raise ValueError("sanitized evidence exceeded output byte limit")
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"cli_version": result["cli_version"], "cases": len(result["cases"]),
                          "provider_model_called": False, "scientific_result": False}))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        # Never print raw provider data, user paths, stderr or environment.
        print(json.dumps({"error": "local CLI audit failed; no unrestricted fallback",
                          "provider_model_called": False, "scientific_result": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
