"""Fresh, reservation-first supervision of the existing BrowserLLM task adapter.

Arhaan2. Preparation and freezing never perform inference. Each arm can dispatch
once; a failed or interrupted run is never retried or moved to an old ledger.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse, unquote

from .editor_budget import BudgetLedger
from .inventory import file_sha256, read_json
from .local_pilot import check_history
from .native_runner import _lock
from .phase8 import _save
from .protocol import canonical_hash

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs/browser_integration_v1.json"
LIMITS = {"editor_requests": 0, "model_probes": 2, "requested_sequences": 2,
          "generated_tokens": 512, "elapsed_seconds": 1200, "billed_units": 0}


def ledger(root, manifest, *, create=False):
    return BudgetLedger(Path(root) / "budget.sqlite3", run_id="browser-integration-v1",
        limits=LIMITS, binding={"manifest_sha256": canonical_hash(manifest),
        "billing_unit": "none_local_execution"}, create=create)


def prepare(root):
    """Bind the actual private user authorization, without consuming capacity."""
    root = Path(root)
    auth = read_json(root / "authorization.json")
    if file_sha256(root / "authorization-source.txt") != auth["authorization_source_sha256"]:
        raise ValueError("private authorization source changed")
    if auth.get("historical_capacity_transferred") is not False:
        raise ValueError("historical capacity cannot be transferred")
    manifest = {"schema_version": 1, "kind": "bounded_browser_integration_v1",
        "author": "Arhaan2", "strict_ready": False, "editor_isolation": "unverified",
        "authorization_sha256": file_sha256(root / "authorization.json"),
        "authorization_source_sha256": auth["authorization_source_sha256"],
        "profile": read_json(PROFILE), "limits": LIMITS,
        "history_sha256": file_sha256(root / "historical-preservation.json"),
        "historical_unverified_sha256": file_sha256(root / "historical-unverified.json"),
        "artifact_freeze": "freeze.json must exist and verify before dispatch",
        "scope": "one fixed development smoke input; pristine then replayed existing smoke edit; no improvement claim"}
    _save(root / "manifest.json", manifest)
    _save(root / "ready.json", {"manifest_sha256": canonical_hash(manifest)})
    with ledger(root, manifest, create=True) as book:
        _save(root / "budget-before.json", book.summary())
    return {"prepared": True, "sequences_dispatched": 0, "manifest_sha256": canonical_hash(manifest)}


def load(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if (read_json(root / "ready.json") != {"manifest_sha256": canonical_hash(manifest)}
            or manifest["profile"] != read_json(PROFILE)
            or manifest["limits"] != LIMITS
            or manifest["authorization_sha256"] != file_sha256(root / "authorization.json")
            or manifest["authorization_source_sha256"] != file_sha256(root / "authorization-source.txt")
            or manifest["history_sha256"] != file_sha256(root / "historical-preservation.json")
            or manifest["historical_unverified_sha256"] != file_sha256(root / "historical-unverified.json")):
        raise ValueError("browser manifest, authorization or frozen profile differs")
    check_history(read_json(root / "historical-preservation.json"))
    return root, manifest


def stage(root, *, benchmark, conversion, tasks, port=4187):
    """Copy verified artifacts into the existing local harness and freeze payloads."""
    from .browser_conversion import verify
    root, manifest = load(root)
    benchmark, conversion, tasks = Path(benchmark).resolve(), Path(conversion), Path(tasks)
    verify(conversion)
    artifacts = read_json(conversion / "artifacts.json")
    task_manifest = read_json(tasks / "manifest.json")
    for name, digest in task_manifest["files"].items():
        if file_sha256(tasks / name) != digest:
            raise ValueError("frozen task data changed")
    smoke = read_json(tasks / "smoke-task.json")
    if smoke != read_json(tasks / "tasks.json")[0]:
        raise ValueError("smoke input must be the first frozen development task")
    if len(smoke["messages"]) != 1 or smoke["messages"][0]["role"] != "user":
        raise ValueError("browser task currently accepts one exact user message")
    origin = f"http://127.0.0.1:{port}"
    models = benchmark / "public/models"
    models.mkdir(exist_ok=True)
    library = models / ("library-" + artifacts["model_library_sha256"][:16])
    library.mkdir()
    shutil.copyfile(conversion / "model.wasm", library / "model.wasm")
    library_url = origin + "/models/" + library.name + "/model.wasm"
    prepared = {}
    for arm in ("baseline", "candidate"):
        artifact = artifacts["arms"][arm]
        source = Path(artifact["path"])
        destination = models / source.name
        shutil.copytree(source, destination)
        base_url = origin + "/models/" + destination.name + "/"
        cache = read_json(destination / "tensor-cache.json")
        required = ["mlc-chat-config.json", "tensor-cache.json", "tokenizer.json",
                    *[r["dataPath"] for r in cache["records"]]]
        assets = [{"url": base_url + name, "sha256": artifact["files"][name]["sha256"]}
                  for name in required]
        assets.append({"url": library_url, "sha256": artifacts["model_library_sha256"]})
        config = {"schema": "weight-space-browser-task-v1", "runId": "browser-integration-v1-" + arm,
            "configurationSha256": canonical_hash(manifest), "arm": arm,
            "sourceCheckpointSha256": artifact["source_checkpoint_sha256"],
            "convertedArtifactSha256": artifact["artifact_sha256"],
            "model": {"id": artifact["model_id"], "modelUrl": base_url, "modelLib": library_url,
                      "quantization": "q4f16_1", "requiredFeatures": ["shader-f16"], "bufferSizeRequiredBytes": 0},
            "assets": assets,
            "settings": {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 256,
                         "contextWindow": 4096, "warmups": 0, "warmupTokens": 16,
                         "requestLimitMs": 60000, "thinking": False},
            "totalLimitMs": 570000,
            "items": [{"item_id": smoke["item_id"], "prompt": smoke["messages"][0]["content"]}]}
        _save(root / (arm + "-config.json"), config)
        prepared[arm] = {"config": str(root / (arm + "-config.json")),
                         "artifact_sha256": artifact["artifact_sha256"], "local_assets": len(assets)}
    _save(root / "staging.json", {"arms": prepared, "task_manifest_sha256": canonical_hash(task_manifest),
        "conversion_manifest_sha256": canonical_hash(artifacts), "inference": 0})
    return prepared


def freeze(root, *, benchmark, baseline_config, candidate_config, evidence):
    """Freeze final adapter input, deployment provenance and all executable code."""
    root, manifest = load(root)
    benchmark = Path(benchmark).resolve()
    configs = [Path(baseline_config).resolve(), Path(candidate_config).resolve()]
    documents = [read_json(p) for p in configs]
    # The browser-side validator checks the complete adapter schema too.
    for arm, data in zip(("baseline", "candidate"), documents, strict=True):
        if data.get("arm") != arm:
            raise ValueError("arm identity does not match frozen order")
        if len(data["items"]) != 1 or data["settings"]["maxTokens"] != 256:
            raise ValueError("integration arm requires exactly one item and 256-token cap")
        key_map = {"contextWindowTokens": "contextWindow", "maxOutputTokens": "maxTokens",
            "temperature": "temperature", "topP": "topP", "seed": "seed",
            "enableThinking": "thinking", "requestTimeoutMs": "requestLimitMs", "warmupCount": "warmups"}
        for key, target in key_map.items():
            if data["settings"].get(target) != manifest["profile"]["settings"][key]:
                raise ValueError("effective browser setting differs: " + key)
        if data["totalLimitMs"] > 590000:
            raise ValueError("cleanup must fit within each 600-second arm")
    if documents[0]["items"] != documents[1]["items"]:
        raise ValueError("both arms must receive the same frozen development smoke item")
    if (documents[0]["convertedArtifactSha256"] == documents[1]["convertedArtifactSha256"]
            or documents[0]["model"]["modelUrl"] == documents[1]["model"]["modelUrl"]
            or documents[0]["model"]["id"] == documents[1]["model"]["id"]):
        raise ValueError("baseline and candidate artifact identities must differ")
    if documents[0]["settings"] != documents[1]["settings"]:
        raise ValueError("arm settings differ")
    paths = [*configs, Path(evidence).resolve(), PROFILE, Path(__file__).resolve()]
    paths += [p for group in ("src/benchmark", "harness", "scripts", "public/benchmark-worker")
              for p in (benchmark / group).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    paths += [benchmark / p for p in ("package.json", "package-lock.json",
        "public/benchmark-worker/benchmark.worker.js", "public/task-evaluation.js", "public/task-evaluation.html")]
    # The worker checks every consumed asset against this independently frozen map.
    # Hash all served files here as well so a changed file fails before model loading.
    for data in documents:
        for asset in data["assets"]:
            url = urlparse(asset["url"])
            if url.hostname != "127.0.0.1" or url.scheme != "http" or url.username or url.password:
                raise ValueError("deployment assets must be served by the local harness")
            path = (benchmark / "public" / unquote(url.path).lstrip("/")).resolve()
            if not path.is_relative_to(benchmark / "public") or file_sha256(path) != asset["sha256"]:
                raise ValueError("local deployment artifact hash mismatch")
            paths.append(path)
    frozen = {"schema_version": 1, "manifest_sha256": canonical_hash(manifest),
        "benchmark": str(benchmark), "arm_configs": [str(p) for p in configs],
        "conversion_evidence": str(Path(evidence).resolve()),
        "files": {str(p): file_sha256(p) for p in sorted(set(paths))}}
    _save(root / "freeze.json", frozen)
    _save(root / "freeze-ready.json", {"freeze_sha256": canonical_hash(frozen)})
    return {"frozen": True, "freeze_sha256": canonical_hash(frozen), "inference": 0}


def check(root):
    root, manifest = load(root)
    frozen = read_json(root / "freeze.json")
    if (frozen["manifest_sha256"] != canonical_hash(manifest)
            or read_json(root / "freeze-ready.json") != {"freeze_sha256": canonical_hash(frozen)}):
        raise ValueError("dispatch freeze changed")
    for name, digest in frozen["files"].items():
        if file_sha256(name) != digest:
            raise ValueError("frozen input changed: " + name)
    return root, manifest, frozen


def _stop_process(process, grace):
    """Stop only the newly launched process group, never the user's other Chrome."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=max(0.01, grace - 1))
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return {"process_group_stop_requested": True, "controller_exit_verified": False}
    return {"process_group_stop_requested": True, "controller_exit_verified": True,
            "controller_returncode": process.returncode}


def _collector(benchmark):
    sys.path.insert(0, str(benchmark))
    return importlib.import_module("harness.collectors.macos.native_collector")


def execute_arm(*, benchmark, config, arm_root, arm, book, remaining, deadline):
    """Shared one-arm driver for integration and separately authorized pilot.

    Caller reserves and marks dispatch before invoking; no new reservation,
    retry, proposal or second sequence is created here. The frozen browser
    payload determines the scheduled item count.
    """
    benchmark, arm_root = Path(benchmark), Path(arm_root)
    expected = read_json(config)
    collector = _collector(benchmark)
    arm_started = time.monotonic()
    process = None
    error = None
    cleanup = {}
    command = ["node", "scripts/run-task-evaluation.mjs", config, str(arm_root.resolve())]
    try:
        with (arm_root / "stdout.txt").open("x") as out, (arm_root / "stderr.txt").open("x") as err, \
                (arm_root / "host-signals.jsonl").open("x") as host:
            # Host preflight is part of the browser reservation, not MPS telemetry.
            next_sample = 0
            while True:
                if time.monotonic() >= next_sample:
                    values, unavailable = collector.host_signals()
                    host.write(json.dumps({"time": time.time(), "scope": "host-wide",
                        "values": values, "unavailable": unavailable}) + "\n")
                    host.flush()
                    os.fsync(host.fileno())
                    next_sample = time.monotonic() + 5
                    if ((values.get("hostMemoryPressureLevel") or 0) >= 4
                            or (values.get("hostThermalState") or 0) >= 2):
                        error = "host_resource_stop"
                        break
                if process is None:
                    process = subprocess.Popen(command, cwd=benchmark, stdout=out, stderr=err,
                                               start_new_session=True)
                if process.poll() is not None:
                    if process.returncode:
                        error = "browser_adapter_exit_" + str(process.returncode)
                    break
                if time.monotonic() >= min(deadline, arm_started + remaining) - 10:
                    error = "browser_execution_deadline"
                    break
                book.check_active(arm)
                time.sleep(min(1, max(0.01, arm_started + remaining - 10 - time.monotonic())))
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        if process is not None:
            cleanup = _stop_process(process, min(10, max(1, deadline - time.monotonic())))
        else:
            cleanup = {"browser_process_created": False, "controller_exit_verified": True}
    elapsed = time.monotonic() - arm_started
    terminal_path = arm_root / "result.json"
    terminal = None
    if terminal_path.exists():
        try:
            terminal = read_json(terminal_path)
        except (ValueError, OSError) as exc:
            error = error or "invalid_terminal_browser_evidence: " + str(exc)
    observed_tokens = None
    if terminal is not None:
        observed_tokens = terminal.get("generatedTokens")
        if observed_tokens is not None and (type(observed_tokens) is not int or observed_tokens < 0):
            observed_tokens = None
            error = error or "invalid_token_measurement"
        if terminal.get("status") not in {"completed", "complete"}:
            error = error or "browser_execution_not_complete"
        for actual, wanted in (("configuration_sha256", "configurationSha256"),
                               ("source_checkpoint_sha256", "sourceCheckpointSha256"),
                               ("converted_artifact_sha256", "convertedArtifactSha256")):
            if terminal.get(actual) != expected[wanted]:
                error = error or "terminal_identity_mismatch"
        if observed_tokens is not None and observed_tokens > expected["settings"]["maxTokens"] * len(expected["items"]):
            error = error or "generated_token_limit_exceeded"
        if not terminal.get("cleanup", {}).get("unload_confirmed"):
            error = error or "model_unload_not_confirmed"
        if not error and observed_tokens is None:
            error = "completion_token_measurement_unavailable"
    else:
        error = error or "missing_terminal_browser_evidence"
    if process is None:
        observed_tokens = 0
    status = "timed_out" if error == "browser_execution_deadline" else "failed" if error else "completed"
    return {"arm": arm, "status": status, "error": error, "elapsed_seconds": elapsed,
        "generated_tokens": observed_tokens, "cleanup": cleanup,
        "raw_result": str(terminal_path) if terminal else None}


def run(root):
    """Execute the two newly authorized arms once, with durable partial evidence."""
    root, manifest, frozen = check(root)
    with _lock(root), ledger(root, manifest) as book:
        if book.attempts():
            raise ValueError("this one-shot integration check has already dispatched; retries forbidden")
        execution = root / "execution"
        execution.mkdir(mode=0o700)  # A crash cannot renew the aggregate deadline.
        benchmark = Path(frozen["benchmark"])
        started = time.monotonic()
        deadline = started + 1200
        _save(execution / "activation.json", {"started_at": time.time(),
            "monotonic_started": started, "monotonic_deadline": deadline,
            "freeze_sha256": canonical_hash(frozen)})
        results = []
        for arm, config in zip(("baseline", "candidate"), frozen["arm_configs"], strict=True):
            arm_root = execution / arm
            arm_root.mkdir(mode=0o700)
            arm_started = time.monotonic()
            remaining = min(600, deadline - arm_started)
            if remaining <= 10:
                results.append({"arm": arm, "status": "unrun", "reason": "aggregate_deadline"})
                break
            row = book.reserve(arm, kind="probe", sequences=1, max_generated_tokens=256,
                               timeout_seconds=remaining, max_billed_units=0)
            book.mark_dispatched(arm)
            result = execute_arm(benchmark=benchmark, config=config, arm_root=arm_root,
                arm=arm, book=book, remaining=remaining, deadline=min(deadline, row["deadline_monotonic"]))
            error, status = result["error"], result["status"]
            observed_tokens, elapsed = result["generated_tokens"], result["elapsed_seconds"]
            book.finish(arm, status=status, completed_sequences=int(not error), failed_sequences=int(bool(error)),
                        generated_tokens=observed_tokens, elapsed_seconds=elapsed, billed_units=0)
            _save(arm_root / "supervisor-result.json", result)
            results.append(result)
            if error:
                book.cancel(reason="browser_stop_condition")
                break
        for arm in ("baseline", "candidate"):
            if not any(x["arm"] == arm for x in results):
                results.append({"arm": arm, "status": "unrun", "reason": "prior_stop_condition"})
        result = {"schema_version": 1, "author": "Arhaan2", "arms": results,
            "aggregate_browser_seconds": time.monotonic() - started,
            "accounting": book.summary(), "strict_ready": False,
            "scientific_improvement_claim": False, "retries": 0}
        _save(execution / "summary.json", result)
        _save(root / "budget-after.json", book.summary())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "stage", "freeze", "check", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--conversion", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "stage":
        if not all((args.benchmark, args.conversion, args.tasks)):
            parser.error("stage requires benchmark, conversion and tasks")
        result = stage(args.root, benchmark=args.benchmark, conversion=args.conversion, tasks=args.tasks)
    elif args.command == "freeze":
        if not all((args.benchmark, args.baseline_config, args.candidate_config, args.evidence)):
            parser.error("freeze requires benchmark, both arm configs and conversion evidence")
        result = freeze(args.root, benchmark=args.benchmark, baseline_config=args.baseline_config,
                        candidate_config=args.candidate_config, evidence=args.evidence)
    elif args.command == "check":
        check(args.root)
        result = {"verified": True, "inference": 0}
    else:
        if not args.execute:
            parser.error("run requires --execute; the private fresh authorization must already verify")
        result = run(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
