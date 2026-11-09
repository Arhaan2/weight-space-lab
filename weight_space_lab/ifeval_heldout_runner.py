"""One frozen paired held-out schedule using the preserved IFEval browser pipeline.

Arhaan2. The original development runner and its evidence remain unchanged.
Only schedule capacity, test-split routing, and the shared two-arm deadline are
added here. Generation, extraction, reconciliation, and official grading use
the existing implementation; this module never creates or modifies a candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from .browser_check import _collector, _stop_process
from .browser_tasks import adapt_browser_output as _development_adapter
from .browser_v2_runner import alive, deadline_watchdog, sample_resources
from .editor_budget import BudgetLedger
from .grading import grade_record, grader_identity, verify_ifeval_assets
from .inventory import file_sha256, read_json
from .native_runner import _lock
from .protocol import canonical_hash
from .records import validate_run_record
from .ifeval_baseline import SETTINGS, POLICY, save, read_events, verify_loaded, reconcile

PER_ARM_LIMITS = {"engine_load_attempts": 1, "warmup_dispatches": 1,
    "scored_dispatches": 128, "generation_dispatches": 129, "generated_tokens": 131088,
    "warmup_tokens": 16, "tokens_per_scored_request": 1024, "request_seconds": 120,
    "wall_seconds": 5400, "automatic_retries": 0, "heldout_generations": 128,
    "frontier_requests": 0}
GEN_LIMITS = {"editor_requests": 0, "model_probes": 129, "requested_sequences": 129,
    "generated_tokens": 131088, "elapsed_seconds": 5400, "billed_units": 0}
AGGREGATE_LIMITS = {"engine_load_attempts": 2, "warmup_dispatches": 2,
    "scored_dispatches": 256, "generation_dispatches": 258, "generated_tokens": 262176,
    "warmup_tokens": 16, "tokens_per_scored_request": 1024, "request_seconds": 120,
    "wall_seconds_per_model": 5400, "wall_seconds": 10800, "automatic_retries": 0,
    "new_weight_proposals": 0, "development_reruns": 0, "frontier_requests": 0}
CLEANUP_KEYS = ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed",
    "local_harness_exit_confirmed", "controller_exit_verified", "chrome_os_exit_verified")


def ledger(root, manifest, *, create=False):
    return BudgetLedger(Path(root)/"generation-budget.sqlite3",
        run_id="ifeval-browser-v1-heldout-"+manifest["arm"], limits=GEN_LIMITS,
        binding={"manifest_sha256": canonical_hash(manifest),
            "billing_unit": "none_local_execution",
            "scope": "heldout warmup and scored requests; immutable per-arm half of aggregate authorization"},
        create=create)


def load(root):
    root = Path(root).resolve()
    manifest = read_json(root/"manifest.json")
    freeze = read_json(root.parent/"evaluation-freeze.json")
    from .ifeval_heldout_analysis import ANALYSIS_SPEC
    from .analysis_pipeline import validate_analysis_manifest
    if freeze["analysis_specification"] != ANALYSIS_SPEC:
        raise ValueError("frozen analysis specification changed")
    for mode in ("strict", "loose"):
        paired = read_json(root.parent/("paired-analysis-"+mode+".json"))
        validate_analysis_manifest(paired)
        if paired["manifest_sha256"] != freeze["paired_analysis_manifests"][mode]:
            raise ValueError("frozen paired analysis manifest changed")
    if (read_json(root.parent/"evaluation-ready.json") != {"evaluation_freeze_sha256": canonical_hash(freeze)}
            or freeze["execution_order"] != ["baseline", "candidate"]
            or freeze["limits"] != AGGREGATE_LIMITS
            or freeze["arms"][manifest["arm"]]["manifest_sha256"] != canonical_hash(manifest)):
        raise ValueError("aggregate heldout freeze changed")
    if (manifest["arm"] not in {"baseline", "candidate"} or root.name != manifest["arm"]
            or manifest["experiment_id"] != "ifeval-browser-v1-heldout"
            or manifest["limits"] != PER_ARM_LIMITS or manifest["settings"] != SETTINGS
            or manifest["policy"] != POLICY or manifest["grader"] != grader_identity("ifeval")
            or manifest.get("heldout_execution_authorized") is not True
            or read_json(root/"ready.json") != {"manifest_sha256": canonical_hash(manifest)}
            or file_sha256(root/"authorization-source.txt") != manifest["authorization_source_sha256"]):
        raise ValueError("frozen heldout arm identity changed")
    for name, digest in manifest["files"].items():
        if file_sha256(name) != digest:
            raise ValueError("frozen file changed: "+name)
    if verify_ifeval_assets(manifest["official_ifeval_root"]) != manifest["official_assets"]:
        raise ValueError("official evaluator assets changed")
    items, config = read_json(manifest["items_path"]), read_json(root/"baseline-config.json")
    if (len(items) != 128 or len({i["item_id"] for i in items}) != 128
            or sum(len(i["target"]["instruction_id_list"]) for i in items) != 173
            or any(i["benchmark"] != "ifeval" or i["split"] != "test" for i in items)
            or canonical_hash(items) != manifest["items_sha256"]
            or canonical_hash(items) != freeze["evaluation_items_sha256"]
            or [i["item_id"] for i in items] != freeze["task_order"]
            or freeze["scheduled_prompts"] != 128 or freeze["scheduled_instructions"] != 173
            or config["warmupPrompt"] != freeze["warmup_prompt"]
            or canonical_hash({k: v for k, v in config.items() if k != "configurationSha256"})
                != config["configurationSha256"]
            or [{"item_id": i["item_id"], "prompt": i["task_text"]} for i in items]
                != [{"item_id": i["item_id"], "prompt": i["prompt"]} for i in config["items"]]
            or config["settings"] != SETTINGS or config["totalLimitMs"] != 5400000
            or config["arm"] != manifest["arm"]
            or config["runId"] != "ifeval-browser-v1-heldout-"+manifest["arm"]
            or config["configurationSha256"] != manifest["configuration_sha256"]
            or config["sourceCheckpointSha256"] != manifest["checkpoint"]["sha256"]
            or config["convertedArtifactSha256"] != manifest["artifact_sha256"]
            or config.get("evaluationStage") != "heldout"):
        raise ValueError("heldout schedule, counts, or runtime configuration changed")
    return root, manifest


def verify_execution_order(root, manifest):
    root = Path(root)
    arm = manifest["arm"]
    if arm == "baseline":
        if (root.parent/"aggregate-clock.json").exists() or (root.parent/"candidate/execution").exists():
            raise ValueError("baseline load capacity already used or execution order changed")
        return
    base = root.parent/"baseline"
    result_path = base/"execution/baseline-1/supervisor-result.json"
    if not result_path.exists() or not (root.parent/"aggregate-clock.json").exists():
        raise ValueError("candidate requires the completed baseline and first-load clock")
    result = read_json(result_path)
    cleanup = result.get("cleanup", {})
    if (result["status"] != "completed" or result.get("error") is not None
            or result.get("engine_load_attempts") != 1 or result.get("generation_dispatches") != 129
            or any(cleanup.get(k) is not True for k in CLEANUP_KEYS)
            or cleanup.get("retained_owned_pids_alive") != []):
        raise ValueError("baseline incomplete or cleanup unconfirmed; stop paired evaluation")
    terminal = read_json(base/"execution/baseline-1/result.json")
    if alive(terminal.get("driver", {}).get("chrome_pid")):
        raise ValueError("baseline browser remains alive")


def grant_load(root, manifest, *, now=None, mono=None):
    """Consume one load slot and start deadlines before acknowledging the engine."""
    root = Path(root)
    verify_execution_order(root, manifest)
    now = time.time() if now is None else now
    mono = time.monotonic() if mono is None else mono
    freeze_sha = canonical_hash(read_json(root.parent/"evaluation-freeze.json"))
    clock_path = root.parent/"aggregate-clock.json"
    if manifest["arm"] == "baseline":
        clock = {"schema_version": 1, "kind": "heldout_first_load_clock",
            "evaluation_freeze_sha256": freeze_sha, "first_started_epoch_ms": now*1000,
            "first_started_monotonic": mono, "deadline_epoch_ms": now*1000+10800000,
            "deadline_monotonic": mono+10800}
        save(clock_path, clock)
    else:
        clock = read_json(clock_path)
    if (clock["evaluation_freeze_sha256"] != freeze_sha
            or now*1000 < clock["first_started_epoch_ms"] or mono < clock["first_started_monotonic"]
            or clock["deadline_epoch_ms"] != clock["first_started_epoch_ms"]+10800000
            or clock["deadline_monotonic"] != clock["first_started_monotonic"]+10800):
        raise ValueError("aggregate deadline binding or clock changed")
    remaining = min(5400, clock["deadline_monotonic"]-mono, clock["deadline_epoch_ms"]/1000-now)
    if remaining < 140:
        raise TimeoutError("insufficient_overall_time_for_load_and_cleanup")
    grant = {"started_epoch_ms": now*1000, "deadline_epoch_ms": (now+remaining)*1000,
        "started_monotonic": mono, "deadline_monotonic": mono+remaining,
        "manifest_sha256": canonical_hash(manifest), "evaluation_freeze_sha256": freeze_sha,
        "aggregate_clock_sha256": canonical_hash(clock), "load_attempts_consumed": 1}
    save(root/"load-grant.json", grant)
    return grant


def validate_intent(intent, config, code_sha, index):
    if type(index) is not int or not 0 <= index <= 129 or len(config["items"]) != 128:
        raise ValueError("heldout generation capacity exceeded")
    arm = config["arm"]
    expected = {"kind": "load" if index == 0 else "generation", "arm": arm,
        "request_id": "load-1" if index == 0 else f"generation-{index}",
        "attempt_id": arm+"-1", "code_version_sha256": code_sha,
        "configuration_sha256": config["configurationSha256"],
        "artifact_sha256": config["convertedArtifactSha256"], "run_id": config["runId"]}
    if index:
        expected.update(warmup=index == 1, max_tokens=16 if index == 1 else 1024, sequence_index=index-1)
        if index > 1:
            expected["item_id"] = config["items"][index-2]["item_id"]
    if any(intent.get(k) != v for k, v in expected.items()):
        raise ValueError("dispatch identity, order, warmup, or token cap differs from heldout freeze")
    return expected


def adapt_browser_output(item, output, **kwargs):
    """Route test identity through the unchanged raw adapter, then validate it.

    The adapter's development-only guard is its sole split-specific behavior.
    This local view never changes a saved item, prompt, observation, or target.
    The original test identity is restored before official grading and hashing.
    """
    if item["benchmark"] != "ifeval" or item["split"] != "test":
        raise ValueError("heldout adapter requires the original IFEval test identity")
    record, metric = _development_adapter({**item, "split": "development"}, output, **kwargs)
    record.update(split=item["split"], stage="full")
    return validate_run_record(record), metric


def execute(root):
    root, manifest = load(root)
    config = read_json(root/"baseline-config.json")
    order = {item["item_id"]: i for i, item in enumerate(config["items"])}
    arm = manifest.get("arm", "baseline")
    attempt_id = arm+"-1"
    code_sha = canonical_hash(manifest["files"])
    target, benchmark = root/"execution"/attempt_id, root/"benchmark"
    verify_execution_order(root, manifest)
    target.mkdir(parents=True, mode=0o700)  # Refuses repeat execution, even before a load.
    collector = _collector(benchmark)
    process, terminal, error, cleanup = None, None, None, {}
    retained, handled, seen_samples = set(), set(), set()
    watchdog, watchdog_stopped, watchdog_triggered = None, threading.Event(), threading.Event()
    began, grant = time.monotonic(), None
    aggregate_clock = read_json(root.parent/"aggregate-clock.json") if arm == "candidate" else None
    with _lock(root.parent), _lock(root), ledger(root, manifest) as book:
        if book.attempts() or (root/"load-grant.json").exists():
            raise ValueError("heldout capacity already consumed; retries forbidden")
        try:
            with (target/"stdout.txt").open("x") as out, (target/"stderr.txt").open("x") as err, (target/"resources.jsonl").open("x") as resources:
                next_sample = 0
                while True:
                    if aggregate_clock and (time.monotonic() > aggregate_clock["deadline_monotonic"]-12
                            or time.time()*1000 > aggregate_clock["deadline_epoch_ms"]-12000):
                        raise TimeoutError("aggregate_deadline_cleanup_reserve")
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
                    if len(warm) > 1 or len(scored) > 128:
                        raise ValueError("unscheduled outputs")
                    for index, sample in [(1, x) for x in warm] + [(order[x["item_id"]]+2, x) for x in scored]:
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
                    allowed_intents = {"load-intent.json", *{f"generation-{i}-intent.json" for i in range(1, 130)}}
                    if {p.name for p in target.glob("*-intent.json")} - allowed_intents:
                        raise ValueError("unscheduled dispatch intent")
                    for index in range(130):
                        prefix = "load" if index == 0 else f"generation-{index}"
                        path = target/(prefix+"-intent.json")
                        if index in handled or not path.exists():
                            continue
                        intent = read_json(path)
                        validate_intent(intent, config, code_sha, index)
                        if index == 0:
                            grant = grant_load(root, manifest)
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
        success = bool(not error and terminal and terminal.get("status") == "completed" and len(warm) == 1 and len(scored) == 128)
        if any(a["status"] in {"reserved", "dispatched"} for a in book.attempts()):
            book.recover_interrupted(executor_stopped=not cleanup["retained_owned_pids_alive"] and process is not None and process.poll() is not None)
        book.cancel(reason="heldout_completed" if success else "heldout_stopped")
        result = {"status": "completed" if success else "stopped", "error": error,
            "engine_load_attempts": int(grant is not None), "warmup_dispatches": int(1 in handled),
            "scored_dispatches": sum(i >= 2 for i in handled), "generation_dispatches": len(book.attempts()),
            "warmup_outputs": warm, "outputs": scored, "cleanup": cleanup,
            "first_load_through_cleanup_seconds": None if not grant else time.monotonic()-grant["started_monotonic"],
            "supervisor_seconds": time.monotonic()-began, "generation_budget": book.summary(),
            "candidate_generations": len(book.attempts()) if arm == "candidate" else 0,
            "heldout_generations": sum(i >= 2 for i in handled), "frontier_requests": 0}
        save(target/"supervisor-result.json", result)
        save(root/"accounting-after.json", result)
        return result


def grade(root):
    root, manifest = load(root)
    items = read_json(manifest["items_path"])
    arm = manifest.get("arm", "baseline")
    execution_path = root/"execution"/(arm+"-1")/"supervisor-result.json"
    if not execution_path.exists() and (root/"execution").exists():
        raise ValueError("execution started but terminal evidence is unavailable; reconcile before grading")
    execution = read_json(execution_path) if execution_path.exists() else {"status": "unexecuted", "outputs": []}
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
        "technically_valid": execution["status"] == "completed" and not unresolved}
    result = {"schema_version": 1, "kind": "browser_ifeval_heldout_results", "author": "Arhaan2",
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
