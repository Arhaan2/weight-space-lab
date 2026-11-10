"""One bounded pristine browser IFEval development run, with no automatic retries.

Arhaan2. Reuses the v2 browser watchdog, resource collector, raw-record adapter,
official grader and durable BudgetLedger. Preparation never creates an engine.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

from .browser_check import _collector, _stop_process
from .browser_tasks import adapt_browser_output
from .browser_v2_runner import alive, deadline_watchdog, durable_json, sample_resources
from .editor_budget import BudgetLedger
from .grading import grade_record, grader_identity, validate_grading_policy, verify_ifeval_assets
from .inventory import file_sha256, read_json
from .native_runner import _lock
from .protocol import canonical_hash

SETTINGS = {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 1024,
            "contextWindow": 4096, "warmups": 1, "warmupTokens": 16,
            "requestLimitMs": 120000, "thinking": False}
LIMITS = {"engine_load_attempts": 1, "warmup_dispatches": 1, "scored_dispatches": 32,
          "generation_dispatches": 33, "generated_tokens": 32784,
          "warmup_tokens": 16, "tokens_per_scored_request": 1024,
          "request_seconds": 120, "wall_seconds": 5400, "automatic_retries": 0,
          "candidate_generations": 0, "heldout_generations": 0, "frontier_requests": 0}
GEN_LIMITS = {"editor_requests": 0, "model_probes": 33, "requested_sequences": 33,
              "generated_tokens": 32784, "elapsed_seconds": 5400, "billed_units": 0}
CANDIDATE_LIMITS = {**LIMITS, "candidate_generations": 33}
POLICY = {"schema_version": 1, "reasoning_policy": "think_tags", "output_eos_tokens": [],
          "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
          "grader_timeout_seconds": 20, "grader_retry": {"mode": "none", "max_attempts": 1}}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    durable_json(path, value)
    path.chmod(0o600)


def ledger(root, manifest, *, create=False):
    return BudgetLedger(Path(root)/"generation-budget.sqlite3", run_id="ifeval-browser-v1-"+manifest.get("arm", "baseline"),
        limits=GEN_LIMITS, binding={"manifest_sha256": canonical_hash(manifest),
        "billing_unit": "none_local_execution", "scope": "warmup and scored requests; load/global clock separate"}, create=create)


def freeze(root, *, authorization, items, official, checkpoint, extra_files):
    """Freeze prepared artifacts, task order, exact scoring, and new capacity."""
    root = Path(root).resolve()
    config = read_json(root/"baseline-config.json")
    tasks = read_json(items)
    if (config["arm"] != "baseline" or config["settings"] != SETTINGS or len(tasks) != 32
            or [{"item_id": i["item_id"], "prompt": i["prompt"]} for i in config["items"]]
                != [{"item_id": i["item_id"], "prompt": i["task_text"]} for i in tasks]
            or any(i["benchmark"] != "ifeval" or i["split"] != "development" for i in tasks)
            or len({i["item_id"] for i in tasks}) != 32
            or checkpoint["sha256"] != config["sourceCheckpointSha256"]):
        raise ValueError("baseline must match the frozen 32 development tasks and pristine runtime")
    if not config.get("warmupPrompt") or config["totalLimitMs"] > 5400000:
        raise ValueError("explicit frozen warmup and bounded wall duration required")
    validate_grading_policy(POLICY)
    assets = verify_ifeval_assets(official)
    if not (root/"authorization-source.txt").exists():
        with Path(authorization).open("rb") as source, (root/"authorization-source.txt").open("xb") as out:
            shutil.copyfileobj(source, out)
            out.flush()
            os.fsync(out.fileno())
        (root/"authorization-source.txt").chmod(0o600)
    save(root/"grading-policy.json", POLICY)
    files = [root/"baseline-config.json", root/"grading-policy.json", Path(items), *map(Path, extra_files)]
    files += list(Path(__file__).parent.glob("*.py"))
    manifest = {"schema_version": 1, "kind": "ifeval_browser_development_baseline", "author": "Arhaan2",
        "experiment_id": "ifeval-browser-v1", "arm": "baseline", "created_epoch_ms": time.time()*1000,
        "scope": "exploratory on-device experiment; editor isolation unverified",
        "strict_ready": False, "editor_isolation": "unverified", "limits": LIMITS,
        "settings": SETTINGS, "checkpoint": checkpoint,
        "configuration_sha256": config["configurationSha256"],
        "artifact_sha256": config["convertedArtifactSha256"],
        "authorization_source_sha256": file_sha256(root/"authorization-source.txt"),
        "files": {str(p.resolve()): file_sha256(p) for p in files},
        "items_path": str(Path(items).resolve()), "items_sha256": canonical_hash(tasks),
        "official_ifeval_root": str(Path(official).resolve()), "official_assets": assets,
        "grader": grader_identity("ifeval"), "policy": POLICY,
        "primary_metric": "prompt_level_strict_accuracy",
        "secondary_metrics": ["instruction_level_strict_accuracy", "prompt_level_loose_accuracy", "instruction_level_loose_accuracy"],
        "denominator": "all 32 scheduled prompts and all scheduled instructions, including missing/failed results",
        "extraction": "WebLLM message.content; existing think_tags removes at most one complete leading think block (including runtime's empty nonthinking prefix), permits plain content, requires no closing boundary for plain content, and removes no terminal control tokens; no rewriting or repair",
        "truncation": "grade the returned text; separately report completion and length stop",
        "grader_errors": "null correctness; any zero-assigned aggregate is a bound until separately versioned repair",
        "historical_capacity_transferred": False, "candidate_execution_authorized": False,
        "heldout_execution_authorized": False, "frontier_execution_authorized": False}
    save(root/"manifest.json", manifest)
    save(root/"ready.json", {"manifest_sha256": canonical_hash(manifest)})
    with ledger(root, manifest, create=True) as book:
        save(root/"accounting-before.json", book.summary())
    return manifest


def load(root):
    root = Path(root).resolve()
    manifest = read_json(root/"manifest.json")
    expected_limits = CANDIDATE_LIMITS if manifest.get("arm") == "candidate" else LIMITS
    if (manifest["limits"] != expected_limits or manifest["settings"] != SETTINGS
            or manifest["policy"] != POLICY or manifest["grader"] != grader_identity("ifeval")
            or read_json(root/"ready.json") != {"manifest_sha256": canonical_hash(manifest)}
            or file_sha256(root/"authorization-source.txt") != manifest["authorization_source_sha256"]):
        raise ValueError("frozen baseline identity changed")
    for name, digest in manifest["files"].items():
        if file_sha256(name) != digest:
            raise ValueError("frozen file changed: "+name)
    if verify_ifeval_assets(manifest["official_ifeval_root"]) != manifest["official_assets"]:
        raise ValueError("official evaluator assets changed")
    return root, manifest


def freeze_candidate(root, *, candidate_artifact, authorization):
    """Later-stage preparation only; requires an imported original proposal.

    This function is not called by baseline preparation or handoff export. It
    creates independent candidate capacity and never reopens baseline capacity.
    The caller supplies the subsequent stage's authorization, not the baseline
    authorization which explicitly excludes candidate execution.
    """
    from .browser_candidate import _files, source_identity, verify_baseline
    root, baseline = load(root)
    baseline_result = read_json(root/"baseline/graded.json")
    if not baseline_result["summary"]["technically_valid"]:
        raise ValueError("complete, resolved pristine baseline required")
    if Path(authorization).read_bytes() == (root/"authorization-source.txt").read_bytes():
        raise ValueError("baseline authorization excludes candidate execution; supply subsequent authorization")
    accepted = read_json(root/"handoff/import-result.json")
    if accepted.get("accepted") is not True:
        raise ValueError("one original proposal must be imported first")
    artifact_path = Path(candidate_artifact).resolve()
    artifact = read_json(artifact_path)
    conversion_manifest = Path("artifacts/browser-integration-v1/conversion/artifacts.json")
    deployed = verify_baseline(conversion_manifest)
    host = read_json(root/"weight-information/host.json")
    identity, audit, _ = source_identity(host["model_path"], root/"candidate/source", deployed,
                                       policy_path=root/"weight-information/policy.json")
    receipt = accepted["audit"]
    if (receipt.get("original_sha256") != audit["base_sha256"]
            or any(receipt.get(key) != audit.get(key) for key in
                   ("result_sha256", "changed_stored_entries", "changed_numerical_entries", "targeted_coordinates"))
            or artifact.get("source_identity") != identity
            or artifact.get("source_checkpoint_sha256") != audit["result_sha256"]
            or artifact.get("baseline_artifact_sha256") != baseline["artifact_sha256"]
            or artifact["files"] != _files(artifact["path"])
            or artifact["artifact_sha256"] != canonical_hash(artifact["files"])
            or read_json(Path(artifact["path"])/"artifact-manifest.json") != artifact):
        raise ValueError("candidate deployment differs from imported pristine-derived proposal")
    binding_path = root/"candidate/converted-source-binding.json"
    binding = read_json(binding_path)
    if (canonical_hash(binding) != artifact["conversion_binding_sha256"]
            or binding.get("status") != "converted_source_bound"
            or binding.get("baseline_manifest_sha256") != file_sha256(conversion_manifest)
            or any(binding.get(k) != v for k, v in identity.items())
            or any(artifact["files"].get(k) != v for k, v in binding["converted_files"].items())):
        raise ValueError("candidate conversion source binding differs")
    if (canonical_hash(read_json(artifact_path.parent/"deployment-difference.json"))
            != artifact["effective_deployment_audit_sha256"]):
        raise ValueError("effective deployed edit audit differs")
    old = read_json(root/"baseline-config.json")
    base_files = deployed["arms"]["baseline"]["files"]
    for name in ["mlc-chat-config.json", "model.wasm", *read_json(Path(artifact["path"])/"mlc-chat-config.json")["tokenizer_files"]]:
        if artifact["files"][name] != base_files[name]:
            raise ValueError("candidate changes tokenizer, model library, or runtime configuration")
    target = root/"candidate-comparison"
    target.mkdir(mode=0o700)
    (target/"benchmark").symlink_to(root/"benchmark", target_is_directory=True)
    origin = urlparse(old["model"]["modelUrl"])
    relative = "models/candidate-"+artifact["artifact_sha256"][:16]+"/resolve/"+artifact["artifact_sha256"]+"/"
    destination = root/"benchmark/public"/relative
    shutil.copytree(artifact["path"], destination)
    base_url = f"{origin.scheme}://{origin.netloc}/"+relative
    config = json.loads(json.dumps(old))
    config.update(arm="candidate", runId="ifeval-browser-v1-candidate",
        sourceCheckpointSha256=artifact["source_checkpoint_sha256"], convertedArtifactSha256=artifact["artifact_sha256"])
    config["model"].update(id=artifact["model_id"], modelUrl=base_url)
    names = ["mlc-chat-config.json", "tensor-cache.json", "tokenizer.json",
             *[r["dataPath"] for r in read_json(destination/"tensor-cache.json")["records"]]]
    config["assets"] = [{"url": base_url+name, "sha256": artifact["files"][name]["sha256"]} for name in names]
    config["assets"].append({"url": old["model"]["modelLib"], "sha256": artifact["files"]["model.wasm"]["sha256"]})
    save(target/"baseline-config.json", config)
    save(target/"grading-policy.json", baseline["policy"])
    with Path(authorization).open("rb") as source, (target/"authorization-source.txt").open("xb") as out:
        shutil.copyfileobj(source, out)
        out.flush()
        os.fsync(out.fileno())
    manifest = json.loads(json.dumps(baseline))
    manifest.update(arm="candidate", kind="ifeval_browser_matched_candidate", limits=CANDIDATE_LIMITS,
        created_epoch_ms=time.time()*1000, artifact_sha256=artifact["artifact_sha256"],
        checkpoint={**baseline["checkpoint"], "sha256": artifact["source_checkpoint_sha256"]},
        authorization_source_sha256=file_sha256(target/"authorization-source.txt"), candidate_execution_authorized=True,
        baseline_manifest_sha256=canonical_hash(baseline), baseline_result_sha256=file_sha256(root/"baseline/graded.json"))
    manifest["files"].update({str(p.resolve()): file_sha256(p) for p in
        [target/"baseline-config.json", target/"grading-policy.json", root/"baseline/graded.json",
         root/"handoff/import-result.json", artifact_path, binding_path]})
    save(target/"manifest.json", manifest)
    save(target/"ready.json", {"manifest_sha256": canonical_hash(manifest)})
    with ledger(target, manifest, create=True) as book:
        save(target/"accounting-before.json", book.summary())
    return manifest


def read_events(target):
    path = target/"events.jsonl"
    if not path.exists():
        return []
    data = path.read_bytes()
    # A live trailing partial line is not a completed event.
    return [json.loads(line) for line in data[:data.rfind(b"\n")+1].splitlines()]


def verify_loaded(target, config, code_sha, *, expected_generation=None):
    proof = read_json(target/"fetch-preflight.json")
    if (proof.get("status") != "passed" or proof.get("engine_loads") != 0
            or proof.get("generation_dispatches") != 0 or proof.get("code_version_sha256") != code_sha
            or proof.get("requested_artifact_sha256") != config["convertedArtifactSha256"]):
        raise ValueError("actual browser fetch proof differs")
    events = read_events(target)
    loaded = [e["data"] for e in events if e["type"] == "task-artifact-loaded"]
    effective = [e["data"] for e in events if e["type"] == "task-effective-settings"]
    if len(loaded) != 1 or len(effective) != 1:
        raise ValueError("exactly one loaded artifact and effective configuration required")
    wanted = {"modelId": config["model"]["id"], "modelUrl": config["model"]["modelUrl"],
        "modelLib": config["model"]["modelLib"], "sourceCheckpointSha256": config["sourceCheckpointSha256"],
        "convertedArtifactSha256": config["convertedArtifactSha256"], "allAssetsVerified": True,
        "runtime": "WebLLM 0.2.86", "cacheBackend": "indexeddb", "executionBackend": "webgpu",
        "serverInferenceFallback": False}
    if any(loaded[0].get(k) != v for k, v in wanted.items()) or loaded[0].get("deviceCount", 0) < 1:
        raise ValueError("loaded artifact/runtime/WebGPU identity differs")
    actual_assets = {e["data"]["url"]:e["data"]["sha256"] for e in events if e["type"] == "asset-verified"}
    if actual_assets != {a["url"]:a["sha256"] for a in config["assets"]}:
        raise ValueError("actually loaded asset hashes differ")
    settings = effective[0]
    expected = {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 1024,
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
        "n": 1, "ignore_eos": False, "extra_body": {"enable_thinking": False}}
    if expected_generation is not None: expected = expected_generation
    if settings.get("contextWindow") != 4096 or settings.get("maxHistorySize") != 1 or settings.get("generation") != expected:
        raise ValueError("effective generation configuration differs")


def validate_intent(intent, config, code_sha, index):
    kind = "load" if index == 0 else "generation"
    arm = config.get("arm", "baseline")
    expected = {"kind": kind, "arm": arm, "request_id": "load-1" if index == 0 else f"generation-{index}",
        "attempt_id": arm+"-1", "code_version_sha256": code_sha,
        "configuration_sha256": config["configurationSha256"], "artifact_sha256": config["convertedArtifactSha256"],
        "run_id": config["runId"]}
    if any(intent.get(k) != v for k, v in expected.items()):
        raise ValueError("dispatch identity differs from frozen baseline")
    if index:
        if not 1 <= index <= 33:
            raise ValueError("generation capacity exceeded")
        expected.update(warmup=index == 1, max_tokens=16 if index == 1 else 1024,
                        sequence_index=index-1)
        if index > 1:
            expected["item_id"] = config["items"][index-2]["item_id"]
        if any(intent.get(k) != v for k, v in expected.items()):
            raise ValueError("generation ordering, warmup, or token cap differs")
    return expected


def reconcile(book, output, index, config):
    request_id = f"generation-{index}"
    row = next((a for a in book.attempts() if a["request_id"] == request_id), None)
    if row is None:
        raise ValueError("output without a durable dispatch")
    if row["status"] not in {"reserved", "dispatched"}:
        return
    tokens, ms = output.get("generated_tokens"), output.get("response_latency_ms")
    cap = 16 if index == 1 else 1024
    if (type(tokens) is not int or not 0 <= tokens <= cap or type(ms) not in (int, float)
            or not 0 <= ms <= 120000 or output.get("finish_reason") not in {"stop", "eos", "length"}
            or output.get("request_id") != request_id or output.get("warmup") is not (index == 1)
            or output.get("local_webgpu_inference_observed") is not True
            or output.get("webgpu_submitted_batches", 0) < 1
            or (index > 1 and output.get("item_id") != config["items"][index-2]["item_id"])):
        raise ValueError("generation integrity/resource/timeout failure")
    book.finish(request_id, status="completed", completed_sequences=1, failed_sequences=0,
                generated_tokens=tokens, elapsed_seconds=ms/1000, billed_units=0)


def execute(root):
    root, manifest = load(root)
    config = read_json(root/"baseline-config.json")
    arm = manifest.get("arm", "baseline")
    attempt_id = arm+"-1"
    code_sha = canonical_hash(manifest["files"])
    target, benchmark = root/"execution"/attempt_id, root/"benchmark"
    target.mkdir(parents=True, mode=0o700)  # Refuses repeat execution, even before a load.
    collector = _collector(benchmark)
    process, terminal, error, cleanup = None, None, None, {}
    retained, handled, seen_samples = set(), set(), set()
    watchdog, watchdog_stopped, watchdog_triggered = None, threading.Event(), threading.Event()
    began, grant = time.monotonic(), None
    with _lock(root), ledger(root, manifest) as book:
        if book.attempts() or (root/"load-grant.json").exists():
            raise ValueError("baseline capacity already consumed; retries forbidden")
        try:
            with (target/"stdout.txt").open("x") as out, (target/"stderr.txt").open("x") as err, (target/"resources.jsonl").open("x") as resources:
                next_sample = 0
                while True:
                    if time.monotonic() >= next_sample:
                        record = sample_resources(collector, process, retained)
                        resources.write(json.dumps(record)+"\n")
                        resources.flush()
                        os.fsync(resources.fileno())
                        next_sample = time.monotonic()+5
                        values = record["host"]
                        if values.get("hostMemoryPressureLevel") is None or values.get("hostThermalState") is None:
                            raise RuntimeError("resource_state_unknown")
                        if values["hostMemoryPressureLevel"] >= 4 or values["hostThermalState"] >= 2:
                            raise RuntimeError("host_resource_stop")
                    if process is None:
                        process = subprocess.Popen(["node", "scripts/run-task-evaluation.mjs", str(root/"baseline-config.json"), str(target),
                            "--attempt-id", attempt_id, "--code-version-sha", code_sha], cwd=benchmark,
                            stdout=out, stderr=err, start_new_session=True)
                    if watchdog_triggered.is_set() or (grant and time.monotonic() > grant["deadline_monotonic"]-12):
                        raise TimeoutError("global_deadline_cleanup_reserve")
                    if not grant and time.monotonic()-began > 300:
                        raise TimeoutError("pre_engine_preparation_timeout")
                    events = read_events(target)
                    warm = [e["data"] for e in events if e["type"] == "warmup-sample"]
                    scored = [e["data"] for e in events if e["type"] == "task-sample"]
                    if len(warm) > 1 or len(scored) > 32:
                        raise ValueError("unscheduled outputs")
                    for index, sample in [(1, x) for x in warm] + [(i+2, x) for i, x in enumerate(scored)]:
                        if index not in seen_samples:
                            reconcile(book, sample, index, config)
                            seen_samples.add(index)
                            print(json.dumps({"event": "generation_observed", "sequence": index, "warmup": index == 1,
                                "tokens": sample["generated_tokens"], "finish": sample["finish_reason"]}), flush=True)
                    for pending in book.attempts():
                        if pending["status"] == "dispatched":
                            book.check_active(pending["request_id"])
                    if any(e.get("type") == "event" and e.get("data", {}).get("type") == "device-lost" for e in events):
                        raise RuntimeError("gpu_device_lost")
                    for index in range(34):
                        prefix = "load" if index == 0 else f"generation-{index}"
                        path = target/(prefix+"-intent.json")
                        if index in handled or not path.exists():
                            continue
                        intent = read_json(path)
                        validate_intent(intent, config, code_sha, index)
                        if index == 0:
                            now, mono = time.time(), time.monotonic()
                            grant = {"started_epoch_ms": now*1000, "deadline_epoch_ms": (now+5400)*1000,
                                     "started_monotonic": mono, "deadline_monotonic": mono+5400,
                                     "manifest_sha256": canonical_hash(manifest), "load_attempts_consumed": 1}
                            save(root/"load-grant.json", grant)
                            watchdog = threading.Thread(target=deadline_watchdog,
                                args=(process, grant["deadline_monotonic"], watchdog_stopped, watchdog_triggered, target), daemon=True)
                            watchdog.start()
                        else:
                            if not grant or index-1 not in handled or (index > 1 and index-1 not in seen_samples):
                                raise ValueError("requests must execute sequentially after warmup")
                            if grant["deadline_monotonic"]-time.monotonic() < 140:
                                raise TimeoutError("insufficient_global_time_for_request_and_cleanup")
                            verify_loaded(target, config, code_sha)
                            book.reserve(f"generation-{index}", kind="probe", sequences=1,
                                max_generated_tokens=16 if index == 1 else 1024, timeout_seconds=120, max_billed_units=0)
                            book.mark_dispatched(f"generation-{index}")
                        ack = {"kind": intent["kind"], "arm": arm, "request_id": intent["request_id"],
                               "attempt_id": attempt_id, "code_version_sha256": code_sha, "approved": True,
                               "deadline_epoch_ms": grant["deadline_epoch_ms"]}
                        save(target/(prefix+"-ack.json"), ack)
                        handled.add(index)
                        print(json.dumps({"event": "durably_granted", "request_id": intent["request_id"]}), flush=True)
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
        except BaseException as exc:
            error = type(exc).__name__+": "+str(exc)
        finally:
            if process:
                cleanup.update(_stop_process(process, 10))
            else:
                cleanup["controller_exit_verified"] = True
            watchdog_stopped.set()
            if watchdog:
                watchdog.join(timeout=4)
        if (target/"result.json").exists():
            terminal = read_json(target/"result.json")
            cleanup.update(terminal.get("cleanup", {}))
        pid = (terminal or {}).get("driver", {}).get("chrome_pid")
        cleanup["chrome_os_exit_verified"] = not alive(pid) if pid else None
        cleanup["retained_owned_pids_alive"] = [p for p in sorted(retained) if alive(p)]
        if cleanup.get("chrome_exit_confirmed") and cleanup.get("chrome_os_exit_verified"):
            cleanup["worker_terminated"] = True
        events = read_events(target)
        warm = [e["data"] for e in events if e["type"] == "warmup-sample"]
        scored = [e["data"] for e in events if e["type"] == "task-sample"]
        # A killed worker may leave only the driver's recovered partial response.
        # Keep that attempt distinct from unexecuted slots; never fabricate usage.
        for sample in (terminal or {}).get("outputs", []):
            if not any(s["item_id"] == sample["item_id"] for s in scored):
                scored.append(sample)
        for attempt in book.attempts():
            index = int(attempt["request_id"].split("-")[-1])
            if index > 1:
                item_id = config["items"][index-2]["item_id"]
                if not any(s["item_id"] == item_id for s in scored):
                    scored.append({"item_id": item_id, "raw_output": None,
                        "finish_reason": "error", "generated_tokens": None, "input_tokens": None,
                        "response_latency_ms": None, "execution_error": "Acknowledged or ambiguous dispatch without a recovered response",
                        "dispatch_outcome": "acknowledged_or_ambiguous", "request_id": attempt["request_id"]})
        order = {item["item_id"]: i for i, item in enumerate(config["items"])}
        scored.sort(key=lambda sample: order[sample["item_id"]])
        if not warm:
            warm = (terminal or {}).get("warmupOutputs", [])
        for index, sample in [(1, x) for x in warm] + [(i+2, x) for i, x in enumerate(scored)]:
            try:
                reconcile(book, sample, index, config)
            except ValueError as exc:
                error = error or str(exc)
        clean = all(cleanup.get(k) is True for k in ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed",
            "local_harness_exit_confirmed", "controller_exit_verified", "chrome_os_exit_verified")) and not cleanup["retained_owned_pids_alive"]
        if not clean:
            error = error or "cleanup_unconfirmed"
        if terminal:
            for field, key in (("configuration_sha256", "configurationSha256"), ("source_checkpoint_sha256", "sourceCheckpointSha256"),
                               ("converted_artifact_sha256", "convertedArtifactSha256")):
                if terminal.get(field) != config[key]:
                    error = error or "terminal_identity_mismatch"
        if grant and time.monotonic() > grant["deadline_monotonic"]:
            error = error or "global_deadline_exhausted"
        if not terminal or terminal.get("status") != "completed":
            error = error or "driver_did_not_complete"
        success = bool(not error and terminal and terminal.get("status") == "completed" and len(warm) == 1 and len(scored) == 32)
        if any(a["status"] in {"reserved", "dispatched"} for a in book.attempts()):
            book.recover_interrupted(executor_stopped=not cleanup["retained_owned_pids_alive"] and process is not None and process.poll() is not None)
        book.cancel(reason="baseline_completed" if success else "baseline_stopped")
        result = {"status": "completed" if success else "stopped", "error": error,
            "engine_load_attempts": int(grant is not None), "warmup_dispatches": int(1 in handled),
            "scored_dispatches": sum(i >= 2 for i in handled), "generation_dispatches": len(book.attempts()),
            "warmup_outputs": warm, "outputs": scored, "cleanup": cleanup,
            "first_load_through_cleanup_seconds": None if not grant else time.monotonic()-grant["started_monotonic"],
            "supervisor_seconds": time.monotonic()-began, "generation_budget": book.summary(),
            "candidate_generations": len(book.attempts()) if arm == "candidate" else 0,
            "heldout_generations": 0, "frontier_requests": 0}
        save(target/"supervisor-result.json", result)
        save(root/"accounting-after.json", result)
        return result


def grade(root):
    root, manifest = load(root)
    items = read_json(manifest["items_path"])
    arm = manifest.get("arm", "baseline")
    execution = read_json(root/"execution"/(arm+"-1")/"supervisor-result.json")
    config = read_json(root/"baseline-config.json")
    by_id = {o["item_id"]:o for o in execution["outputs"]}
    if len(by_id) != len(execution["outputs"]) or set(by_id)-{i["item_id"] for i in items}:
        raise ValueError("duplicate or unscheduled observations")
    records, grades, metrics, failures = [], [], [], Counter()
    numerators = Counter()
    instruction_count = sum(len(i["target"]["instruction_id_list"]) for i in items)
    for item in items:
        record, metric = adapt_browser_output(item, by_id.get(item["item_id"]), run_id=config["runId"],
            checkpoint=manifest["checkpoint"], configuration_sha256=config["configurationSha256"])
        checked = grade_record(record, item, manifest["policy"], ifeval_root=manifest["official_ifeval_root"])
        records.append(record)
        grades.append(checked)
        metric.update(grading_status=checked["status"], correctness=checked["correct"],
                      grading_code=checked["diagnostics"]["code"], constraint_results=checked["metrics"])
        metrics.append(metric)
        if checked["status"] == "graded":
            m = checked["metrics"]
            for mode in ("strict", "loose"):
                numerators["prompt_"+mode] += int(m[f"prompt_level_{mode}_accuracy"])
                numerators["instruction_"+mode] += sum(m["instruction_"+mode])
            for name, passed in zip(item["target"]["instruction_id_list"], m["instruction_strict"], strict=True):
                if not passed:
                    failures[name] += 1
    unresolved = sum(g["status"] in {"grader_error", "grader_timeout"} for g in grades)
    summary = {"scheduled_prompts": len(items), "scheduled_instructions": instruction_count,
        "attempted": len(by_id), "execution_status_counts": dict(Counter(r["status"] for r in records)),
        "unresolved_grader_errors": unresolved, "semantic_grading_complete": unresolved == 0,
        "score_interpretation": "measured compliance" if unresolved == 0 else "lower bounds only; unresolved evaluator failure",
        "metrics": {f"{kind}_level_{mode}_accuracy": {"passed": numerators[kind+"_"+mode],
            "scheduled": len(items) if kind == "prompt" else instruction_count,
            "value": numerators[kind+"_"+mode]/(len(items) if kind == "prompt" else instruction_count)}
            for kind in ("prompt", "instruction") for mode in ("strict", "loose")},
        "strict_failures_by_instruction": dict(failures.most_common()),
        "known_generated_tokens": sum(r["token_counts"]["output"] or 0 for r in records),
        "known_input_tokens": sum(r["token_counts"]["input"] or 0 for r in records),
        "generation_seconds": sum(r["duration_seconds"] or 0 for r in records),
        "technically_valid": execution["status"] == "completed" and not unresolved}
    result = {"schema_version": 1, "kind": "browser_ifeval_development_results", "author": "Arhaan2",
        "manifest_sha256": canonical_hash(manifest), "records": records, "grades": grades,
        "item_metrics": metrics, "metrics": metrics, "summary": summary}
    save(root/arm/"graded.json", result)
    save(root/arm/"records.json", records)
    save(root/arm/"grades.json", grades)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["execute", "grade", "check"])
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "execute":
        result = execute(args.root)
        print(json.dumps({k: v for k, v in result.items() if k not in {"outputs", "warmup_outputs"}}, indent=2))
    elif args.command == "grade":
        print(json.dumps(grade(args.root), indent=2))
    else:
        _, manifest = load(args.root)
        print(json.dumps({"verified": True, "manifest_sha256": canonical_hash(manifest)}))


if __name__ == "__main__":
    main()
