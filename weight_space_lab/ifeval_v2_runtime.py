"""V2 sessions using the preserved browser driver, reconciliation gate and grader.

Arhaan2. Dynamic 32/128 schedules; no V1 ledger, manifest, or module is changed.
"""
from __future__ import annotations
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from .browser_check import _collector, _stop_process
from .browser_v2_runner import alive, deadline_watchdog, sample_resources
from .editor_budget import BudgetLedger
from .grading import grade_record, grader_identity, verify_ifeval_assets
from .ifeval_baseline import SETTINGS, POLICY, save, read_events, verify_loaded as verify_task_loaded, reconcile as reconcile_task
from .ifeval_heldout_runner import adapt_browser_output
from .browser_tasks import adapt_browser_output as development_adapter
from .ifeval_candidate_continuation import recover_observations
from .ifeval_supervisor_gate import reconcile_events, assert_predecessor
from .inventory import file_sha256, read_json
from .native_runner import _lock
from .protocol import canonical_hash


RESOURCE_IDENTITY = "browser-llm-bench-standard"


def expected_settings(manifest):
    if manifest.get("workload_identity")==RESOURCE_IDENTITY:
        from .browser_resource import RESOURCE_SETTINGS
        return RESOURCE_SETTINGS
    return SETTINGS


def load(root):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    config = read_json(root / "baseline-config.json")
    resource=manifest.get("workload_identity")==RESOURCE_IDENTITY
    count=len(config["items"])
    if (read_json(root / "ready.json") != {"manifest_sha256": canonical_hash(manifest)}
            or config["settings"] != expected_settings(manifest)
            or manifest["settings"] != config["settings"]
            or config["configurationSha256"] != canonical_hash({k:v for k,v in config.items() if k != "configurationSha256"})
            or manifest["scheduled_prompts"] != count
            or (resource and count != 60) or (not resource and not 1 <= count <= 128)):
        raise ValueError("V2 session freeze changed")
    if not resource and (manifest["policy"] != POLICY or manifest["grader"] != grader_identity("ifeval")
            or canonical_hash(read_json(manifest["items_path"])) != manifest["items_sha256"]
            or verify_ifeval_assets(manifest["official_ifeval_root"]) != manifest["official_assets"]):
        raise ValueError("V2 task data or grader changed")
    if resource:
        from .browser_resource import validate_session
        validate_session(root,manifest,config)
    for name, digest in manifest["files"].items():
        if file_sha256(name) != digest: raise ValueError("V2 session file changed")
    return root, manifest


def verify_loaded(target,config,code_sha):
    expected=None
    if config["settings"]["maxTokens"]==64:
        expected={"temperature":0,"top_p":1,"seed":42,"max_tokens":64,"extra_body":{"enable_thinking":False}}
    return verify_task_loaded(target,config,code_sha,expected_generation=expected)


def reconcile(book,output,index,config):
    # Additional workload caps precede the preserved baseline reconciliation.
    cap=16 if index==1 else config.get("settings",SETTINGS)["maxTokens"]
    ms=output.get("response_latency_ms"); tokens=output.get("generated_tokens")
    if (type(tokens) is not int or not 0<=tokens<=cap or type(ms) not in (int,float)
            or not 0<=ms<=config.get("settings",SETTINGS)["requestLimitMs"]):
        raise ValueError("generation exceeds its frozen workload token/time cap")
    return reconcile_task(book,output,index,config)


def ledger(root, manifest, *, create=False):
    n = manifest["scheduled_prompts"]
    return BudgetLedger(Path(root) / "generation-budget.sqlite3", run_id=manifest["run_id"],
        limits={"editor_requests":0,"model_probes":n+1,"requested_sequences":n+1,
            "generated_tokens":n*expected_settings(manifest)["maxTokens"]+16,"elapsed_seconds":manifest.get("browser_seconds",5400),"billed_units":0},
        binding={"manifest_sha256": canonical_hash(manifest),"billing_unit":"none_local_execution"}, create=create)


def grant_load(root, manifest, seconds):
    if seconds < expected_settings(manifest)["requestLimitMs"]/1000+20:
        raise TimeoutError("Insufficient browser allowance for request and cleanup")
    now, mono = time.time(), time.monotonic()
    grant = {"started_epoch_ms":now*1000,"deadline_epoch_ms":(now+seconds)*1000,
        "started_monotonic":mono,"deadline_monotonic":mono+seconds,
        "manifest_sha256":canonical_hash(manifest),"load_attempts_consumed":1}
    save(Path(root)/"load-grant.json",grant)
    return grant


def validate_intent(intent, config, code_sha, index):
    if type(index) is not int or not 0 <= index <= len(config["items"])+1:
        raise ValueError("V2 dispatch capacity exceeded")
    arm = config["arm"]
    expected = {"kind":"load" if index == 0 else "generation","arm":arm,
        "request_id":"load-1" if index == 0 else f"generation-{index}",
        "attempt_id":arm+"-1","code_version_sha256":code_sha,
        "configuration_sha256":config["configurationSha256"],
        "artifact_sha256":config["convertedArtifactSha256"],"run_id":config["runId"]}
    if index:
        expected.update(warmup=index == 1,max_tokens=16 if index == 1 else config.get("settings",SETTINGS)["maxTokens"],
            sequence_index=index-1,item_id="__warmup__" if index == 1 else config["items"][index-2]["item_id"])
    if any(intent.get(k) != v for k,v in expected.items()):
        raise ValueError("V2 dispatch identity or ordering changed")
    return expected


def execute(root, *, on_load, preparation_seconds):
    root, manifest = load(root)
    config = read_json(root/"baseline-config.json")
    request_seconds=config["settings"]["requestLimitMs"]/1000
    order = {item["item_id"]: i for i, item in enumerate(config["items"])}
    arm = manifest.get("arm", "baseline")
    attempt_id = arm+"-1"
    code_sha = canonical_hash(manifest["files"])
    target, benchmark = root/"execution"/attempt_id, root/"benchmark"
    if target.exists() or (root/"load-grant.json").exists():
        raise ValueError("V2 session execution already started; retries forbidden")
    # Generated interpreter caches are not runtime source; the copied harness
    # contains no historical bytecode and collectors must not create it in place.
    sys.dont_write_bytecode=True
    collector = _collector(benchmark)
    target.mkdir(parents=True, mode=0o700)  # Refuses repeat execution, even before a load.
    process, terminal, error, cleanup = None, None, None, {}
    retained, handled, seen_samples = set(), set(), {}
    validated_events = []
    def observed(index, sample):
        print(json.dumps({"event": "generation_observed", "sequence": index, "warmup": index == 1,
            "tokens": sample["generated_tokens"], "finish": sample["finish_reason"]}), flush=True)
    watchdog, watchdog_stopped, watchdog_triggered = None, threading.Event(), threading.Event()
    began, grant = time.monotonic(), None
    with _lock(root), ledger(root, manifest) as book:
        if book.attempts() or (root/"load-grant.json").exists():
            raise ValueError("V2 session capacity already consumed; retries forbidden")
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
                            stdout=out, stderr=err, start_new_session=True,
                            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
                    if watchdog_triggered.is_set() or (grant and time.monotonic() > grant["deadline_monotonic"]-12):
                        raise TimeoutError("global_deadline_cleanup_reserve")
                    if not grant and time.monotonic()-began > preparation_seconds:
                        raise TimeoutError("pre_engine_preparation_timeout")
                    events = reconcile_events(target, book, config, handled, seen_samples, on_observed=observed)
                    validated_events = events
                    allowed_intents = {"load-intent.json", *{f"generation-{i}-intent.json" for i in range(1, len(config["items"])+2)}}
                    if {p.name for p in target.glob("*-intent.json")} - allowed_intents:
                        raise ValueError("unscheduled dispatch intent")
                    for index in range(len(config["items"])+2):
                        prefix = "load" if index == 0 else f"generation-{index}"
                        path = target/(prefix+"-intent.json")
                        if index in handled or not path.exists():
                            continue
                        intent = read_json(path)
                        validate_intent(intent, config, code_sha, index)
                        if index == 0:
                            grant = grant_load(root, manifest, on_load())
                            watchdog = threading.Thread(target=deadline_watchdog,
                                args=(process, grant["deadline_monotonic"], watchdog_stopped, watchdog_triggered, target), daemon=True)
                            watchdog.start()
                        else:
                            validated_events = assert_predecessor(index, target, book, config, handled, seen_samples, grant,
                                                                  on_observed=observed)
                            if grant["deadline_monotonic"]-time.monotonic() < request_seconds+20:
                                raise TimeoutError("insufficient_global_time_for_request_and_cleanup")
                            verify_loaded(target, config, code_sha)
                            book.reserve(f"generation-{index}", kind="probe", sequences=1,
                                max_generated_tokens=16 if index == 1 else config["settings"]["maxTokens"], timeout_seconds=request_seconds, max_billed_units=0)
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
            try:
                terminal = read_json(target/"result.json")
                if (not isinstance(terminal, dict) or not isinstance(terminal.get("cleanup", {}), dict)
                        or not isinstance(terminal.get("driver", {}), dict)
                        or any(not isinstance(terminal.get(key, []), list)
                               or any(not isinstance(row, dict) for row in terminal.get(key, []))
                               for key in ("outputs", "warmupOutputs"))):
                    raise ValueError("malformed terminal record")
                terminal_pid = terminal.get("driver", {}).get("chrome_pid")
                if terminal_pid is not None and (type(terminal_pid) is not int or terminal_pid <= 0):
                    raise ValueError("malformed terminal browser PID")
                cleanup.update(terminal.get("cleanup", {}))
            except Exception as exc:
                terminal = None
                error = error or "final_terminal_read_failed:" + type(exc).__name__
        pid = (terminal or {}).get("driver", {}).get("chrome_pid")
        cleanup["chrome_os_exit_verified"] = not alive(pid) if pid else None
        cleanup["retained_owned_pids_alive"] = [p for p in sorted(retained) if alive(p)]
        if cleanup.get("chrome_exit_confirmed") and cleanup.get("chrome_os_exit_verified"):
            cleanup["worker_terminated"] = True
        # A final completion or runtime failure may have arrived after the last
        # loop snapshot. Apply the same full helper checks before success can be
        # declared. Recovery below preserves evidence but cannot clear an error.
        try:
            validated_events = reconcile_events(target, book, config, handled, seen_samples, on_observed=observed)
        except Exception as exc:
            error = error or "final_reconciliation_failed:" + type(exc).__name__ + ": " + str(exc)
        warm, scored, recovery_errors = recover_observations(target, terminal or {}, validated_events,
                                                              config, book, seen_samples)
        if recovery_errors:
            error = error or recovery_errors[0]
        for index, sample in [(1, x) for x in warm] + [(order[x["item_id"]]+2, x) for x in scored]:
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
        success = bool(not error and terminal and terminal.get("status") == "completed" and len(warm) == 1 and len(scored) == len(config["items"]))
        if any(a["status"] in {"reserved", "dispatched"} for a in book.attempts()):
            book.recover_interrupted(executor_stopped=not cleanup["retained_owned_pids_alive"] and process is not None and process.poll() is not None)
        book.cancel(reason="v2_session_completed" if success else "v2_session_stopped")
        result = {"status": "completed" if success else "stopped", "error": error,
            "engine_load_attempts": int(grant is not None), "warmup_dispatches": int(1 in handled),
            "scored_dispatches": sum(i >= 2 for i in handled), "generation_dispatches": len(book.attempts()),
            "warmup_outputs": warm, "outputs": scored, "cleanup": cleanup,
            "recovery_integrity_errors": recovery_errors,
            "first_load_through_cleanup_seconds": None if not grant else time.monotonic()-grant["started_monotonic"],
            "supervisor_seconds": time.monotonic()-began, "generation_budget": book.summary(),
            "candidate_generations": len(book.attempts()) if arm == "candidate" else 0,
            "heldout_generations": sum(i >= 2 for i in handled), "frontier_requests": 0,
            "baseline_reruns": 0, "study_manifest_sha256": manifest["study_manifest_sha256"],
            "workload_identity":manifest.get("workload_identity","ifeval-task"),
            "deployment_limits_satisfied":success,"memory_limits_satisfied":success,
            "native_power":"unavailable unless explicitly captured; host-wide power is not model-only"}
        save(target/"supervisor-result.json", result)
        save(root/"accounting-after.json", result)
        return result


def grade(root, *, before_grade=lambda: None):
    root, manifest = load(root)
    if manifest.get("workload_identity")==RESOURCE_IDENTITY: raise ValueError("Resource workload is not IFEval; no task grading")
    items = read_json(manifest["items_path"])
    arm = manifest.get("arm", "baseline")
    execution_path = root/"execution"/(arm+"-1")/"supervisor-result.json"
    if not execution_path.exists():
        raise ValueError("Continuation grading requires explicit execution terminal evidence; pending plans are not graded")
    execution = read_json(execution_path)
    save(root/arm/"grading-intent.json", {"manifest_sha256": canonical_hash(manifest),
        "execution_file_sha256": file_sha256(execution_path) if execution_path.exists() else None,
        "grader_retries": 0, "created_epoch_ms": time.time()*1000})
    config = read_json(root/"baseline-config.json")
    by_id = {o["item_id"]:o for o in execution["outputs"]}
    if len(by_id) != len(execution["outputs"]) or set(by_id)-{i["item_id"] for i in items}:
        raise ValueError("duplicate or unscheduled observations")
    records, grades, metrics, failures = [], [], [], Counter()
    numerators = Counter()
    instruction_count = sum(len(i["target"]["instruction_id_list"]) for i in items)
    for item in items:
        before_grade()
        record, metric = (adapt_browser_output if item["split"] == "test" else development_adapter)(item, by_id.get(item["item_id"]), run_id=config["runId"],
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
        "attempted": len(by_id), "unexecuted": len(items)-len(by_id),
        "attempted_missing_outputs": sum(o.get("raw_output") is None for o in by_id.values()),
        "execution_complete": execution["status"] == "completed",
        "execution_status_counts": dict(Counter(r["status"] for r in records)),
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
        "deployment_limits_satisfied":execution.get("deployment_limits_satisfied",False),
        "memory_limits_satisfied":execution.get("memory_limits_satisfied",False),
        "natural_completions":sum(r["status"]=="ok" for r in records),
        "truncations":sum(r["status"]=="truncated" for r in records),
        "technically_valid": execution["status"] == "completed" and not unresolved}
    result = {"schema_version": 1, "kind": "browser_ifeval_v2_results", "author": "Arhaan2",
        "manifest_sha256": canonical_hash(manifest), "records": records, "grades": grades,
        "item_metrics": metrics, "metrics": metrics, "summary": summary,
        "study_manifest_sha256": manifest["study_manifest_sha256"]}
    save(root/arm/"graded.json", result)
    save(root/arm/"records.json", records)
    save(root/arm/"grades.json", grades)
    return summary
