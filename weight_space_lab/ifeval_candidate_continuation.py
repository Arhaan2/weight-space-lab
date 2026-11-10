"""Separately authorized execution of only candidate held-out slots 95–128.

Arhaan2. Preparation is not authorization. This module keeps old ledgers and
evidence read-only, uses a new clock and budget only after explicit approval,
and retains the original generation, extraction, grading and stop policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from urllib.parse import quote

from .browser_check import _collector, _stop_process
from .browser_v2_runner import alive, deadline_watchdog, sample_resources
from .editor_budget import BudgetLedger
from .grading import grade_record, grader_identity, verify_ifeval_assets
from .ifeval_baseline import SETTINGS, POLICY, save, read_events, verify_loaded, reconcile
from .ifeval_heldout_runner import adapt_browser_output
from .ifeval_supervisor_gate import reconcile_events, assert_predecessor
from .inventory import file_sha256, read_json
from .native_runner import _lock
from .protocol import canonical_hash

LIMITS = {"engine_load_attempts": 1, "warmup_dispatches": 1, "scored_dispatches": 34,
    "generation_dispatches": 35, "generated_tokens": 34832, "warmup_tokens": 16,
    "tokens_per_scored_request": 1024, "request_seconds": 120, "wall_seconds": 5400,
    "automatic_retries": 0, "new_weight_proposals": 0, "development_reruns": 0,
    "baseline_reruns": 0, "completed_candidate_reruns": 0, "frontier_requests": 0}
GEN_LIMITS = {"editor_requests": 0, "model_probes": 35, "requested_sequences": 35,
    "generated_tokens": 34832, "elapsed_seconds": 5400, "billed_units": 0}
EXPERIMENT_ID = "ifeval-browser-v1-heldout-continuation"
RUN_ID = EXPERIMENT_ID + "-candidate"
CANDIDATE_SOURCE = "f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf"
CANDIDATE_ARTIFACT = "6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b"
ORIGINAL_FREEZE = "89ed9703869ab7c8b1032b3d8c5fa59fefed0215d9d0341016684d60b39b8a27"
CLEANUP_KEYS = ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed",
    "local_harness_exit_confirmed", "controller_exit_verified", "chrome_os_exit_verified")
PROCESS_CLEANUP_KEYS = tuple(k for k in CLEANUP_KEYS if k != "unload_confirmed")


def _authorization(root, plan):
    """Read an independently created authorization; never infer it from a plan."""
    path, source = root / "execution-authorization.json", root / "authorization-source.txt"
    if not path.is_file() or not source.is_file():
        raise PermissionError("Explicit later authorization for this exact continuation plan is required")
    auth = read_json(path)
    if (auth.get("execution_authorized") is not True
            or auth.get("continuation_plan_sha256") != canonical_hash(plan)
            or auth.get("authorization_source_sha256") != file_sha256(source)
            or auth.get("limits") != LIMITS or not source.read_bytes().strip()):
        raise PermissionError("Continuation authorization does not bind the frozen plan and limits")
    return auth


def ledger(root, manifest, *, create=False):
    root = Path(root).resolve()
    plan = read_json(root / "continuation-plan.json")
    auth = _authorization(root, plan)
    return BudgetLedger(root / "generation-budget.sqlite3", run_id=RUN_ID,
        limits=GEN_LIMITS, binding={"manifest_sha256": canonical_hash(manifest),
            "continuation_plan_sha256": canonical_hash(plan),
            "execution_authorization_sha256": canonical_hash(auth),
            "billing_unit": "none_local_execution",
            "scope": "separately authorized candidate slots 95 through 128 plus one warmup; no historical capacity transfer"},
        create=create)


def _original_ledger(path):
    """The old ledger is never opened through mutable BudgetLedger."""
    path = Path(path).resolve()
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Original ledger has an unresolved WAL")
    db = sqlite3.connect("file:" + quote(str(path)) + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Original ledger integrity failed")
        meta = dict(db.execute("SELECT * FROM metadata WHERE singleton=1").fetchone())
        rows = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY reserved_at,request_id")]
    finally:
        db.close()
    return meta, rows


def load(root, require_authorization=False):
    root = Path(root).resolve()
    # This first check occurs before reading the plan, importing the collector,
    # making a directory, taking a lock, or opening any generation ledger.
    if require_authorization and not (root / "execution-authorization.json").is_file():
        raise PermissionError("Explicit later continuation authorization is required; preparation does not authorize execution")
    plan = read_json(root / "continuation-plan.json")
    manifest = read_json(root / "manifest.json")
    if require_authorization:
        _authorization(root, plan)
        if not (root / "generation-budget.sqlite3").is_file():
            raise PermissionError("Authorized continuation ledger is absent; execution cannot create capacity")
    if (plan.get("kind") != "frozen_continuation_preparation"
            or plan.get("execution_authorized") is not False or plan.get("authorization_status") != "pending"
            or plan.get("limits") != LIMITS or plan.get("manifest_sha256") != canonical_hash(manifest)
            or read_json(root / "ready.json") != {"continuation_plan_sha256": canonical_hash(plan)}
            or plan.get("merge_plan_sha256") != canonical_hash(read_json(root / "merge-plan.json"))
            or plan.get("original_schedule_offset") != 94 or plan.get("original_schedule_length") != 128):
        raise ValueError("Frozen continuation preparation identity changed")
    if (manifest.get("arm") != "candidate" or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("limits") != LIMITS or manifest.get("settings") != SETTINGS
            or manifest.get("policy") != POLICY or manifest.get("grader") != grader_identity("ifeval")
            or manifest["checkpoint"]["sha256"] != CANDIDATE_SOURCE
            or manifest.get("artifact_sha256") != CANDIDATE_ARTIFACT):
        raise ValueError("Continuation candidate, settings, or grading identity changed")
    old = Path(manifest["original_root"]).resolve()
    if old == root or old in root.parents:
        raise ValueError("Continuation requires a separate destination outside the old evidence tree")
    old_candidate = old / "candidate"
    required = [old / "evaluation-freeze.json", old / "evaluation-items.json",
        old_candidate / "manifest.json", old_candidate / "baseline-config.json",
        old_candidate / "generation-budget.sqlite3", old_candidate / "candidate/graded.json",
        old_candidate / "execution/candidate-1/supervisor-result.json",
        old / "baseline/execution/baseline-1/supervisor-result.json",
        root / "baseline-config.json", Path(manifest["items_path"])]
    if any(str(p.resolve()) not in manifest["files"] for p in required):
        raise ValueError("Continuation is missing required historical and schedule pins")
    for path, pin in manifest["files"].items():
        if file_sha256(path) != pin:
            raise ValueError("Frozen continuation or historical file changed: " + path)
    original_freeze = read_json(old / "evaluation-freeze.json")
    if (canonical_hash(original_freeze) != ORIGINAL_FREEZE
            or plan["original_evaluation_freeze_sha256"] != ORIGINAL_FREEZE
            or manifest["original_evaluation_freeze_sha256"] != ORIGINAL_FREEZE
            or canonical_hash(read_json(old_candidate / "manifest.json")) != manifest["original_candidate_manifest_sha256"]
            or file_sha256(old_candidate / "baseline-config.json") != manifest["original_candidate_configuration_file_sha256"]):
        raise ValueError("Original experiment identity changed")
    old_items, old_config = read_json(old / "evaluation-items.json"), read_json(old_candidate / "baseline-config.json")
    items, config = read_json(manifest["items_path"]), read_json(root / "baseline-config.json")
    if (len(old_items) != 128 or canonical_hash(old_items) != original_freeze["evaluation_items_sha256"]
            or items != old_items[94:] or len(items) != 34 or len({i["item_id"] for i in items}) != 34
            or sum(len(i["target"]["instruction_id_list"]) for i in items) != 45
            or any(i["benchmark"] != "ifeval" or i["split"] != "test" for i in items)
            or canonical_hash(items) != manifest["items_sha256"]
            or plan["task_order"] != [i["item_id"] for i in items]
            or config["items"] != old_config["items"][94:]
            or config.get("study") != "ifeval-browser-v1" or config.get("evaluationStage") != "heldout-continuation"
            or config.get("originalScheduleOffset") != 94 or config.get("originalScheduleLength") != 128
            or config["settings"] != SETTINGS or config["totalLimitMs"] != 5400000
            or config["arm"] != "candidate" or config["runId"] != RUN_ID
            or config["warmupPrompt"] != old_config["warmupPrompt"]
            or config["warmupPrompt"] != "Reply with OK."
            or config["sourceCheckpointSha256"] != CANDIDATE_SOURCE
            or config["convertedArtifactSha256"] != CANDIDATE_ARTIFACT
            or config["configurationSha256"] != manifest["configuration_sha256"]
            or canonical_hash({k:v for k,v in config.items() if k != "configurationSha256"}) != config["configurationSha256"]
            or any(config[k] != old_config[k] for k in ("model", "assets"))):
        raise ValueError("Continuation must contain exactly the original 34 unexecuted candidate slots")
    old_execution = read_json(old_candidate / "execution/candidate-1/supervisor-result.json")
    old_grades = read_json(old_candidate / "candidate/graded.json")
    meta, attempts = _original_ledger(old_candidate / "generation-budget.sqlite3")
    if (old_execution["status"] != "stopped" or old_execution["scored_dispatches"] != 94
            or old_execution["generation_dispatches"] != 95 or old_execution["warmup_dispatches"] != 1
            or [o["item_id"] for o in old_execution["outputs"]] != [i["item_id"] for i in old_items[:94]]
            or meta["cancelled"] != 1 or len(attempts) != 95
            or [a["request_id"] for a in attempts] != [f"generation-{i}" for i in range(1, 96)]
            or any(a["status"] != "completed" or a["retry_of"] is not None for a in attempts)
            or (old_candidate / "execution/candidate-1/generation-96-ack.json").exists()
            or len(old_grades["records"]) != 128
            or [r["item_id"] for r in old_grades["records"]] != [i["item_id"] for i in old_items]
            or any(r["status"] not in {"ok", "truncated"} or len(r["attempts"]) != 1 for r in old_grades["records"][:94])
            or any(r["status"] != "missing" or r["attempts"] for r in old_grades["records"][94:])):
        raise ValueError("Original completed prefix or unexecuted continuation slots changed")
    for arm in ("baseline", "candidate"):
        prior = read_json(old / arm / "execution" / (arm + "-1") / "supervisor-result.json")
        if (any(prior["cleanup"].get(k) is not True for k in PROCESS_CLEANUP_KEYS)
                or prior["cleanup"].get("retained_owned_pids_alive") != []):
            raise ValueError("Original owned process closure is not verified")
    if verify_ifeval_assets(manifest["official_ifeval_root"]) != manifest["official_assets"]:
        raise ValueError("Official evaluator assets changed")
    return root, manifest


def authorize(root, authorization, approved_plan_sha256):
    """Bind a later explicit approval to the reviewed plan and create new capacity."""
    root, manifest = load(root)
    plan = read_json(root / "continuation-plan.json")
    if approved_plan_sha256 != canonical_hash(plan):
        raise PermissionError("Approval must name the exact frozen continuation plan hash")
    content = Path(authorization).read_bytes()
    if not content.strip():
        raise PermissionError("A nonempty explicit later authorization is required")
    old = Path(manifest["original_root"])
    old_authorizations = [old / "authorization-source.txt", old / "candidate/authorization-source.txt",
        old.parent / "ifeval-browser-v1/authorization-source.txt",
        old.parent / "ifeval-browser-v1/candidate-comparison/authorization-source.txt"]
    if any(p.exists() and p.read_bytes().strip() == content.strip() for p in old_authorizations):
        raise PermissionError("Historical authorization cannot authorize this new continuation")
    with _lock(root):
        if any((root / p).exists() for p in ("authorization-source.txt", "execution-authorization.json", "generation-budget.sqlite3", "execution", "load-grant.json")):
            raise FileExistsError("Continuation authorization or capacity already exists; do not recreate it")
        with (root / "authorization-source.txt").open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        (root / "authorization-source.txt").chmod(0o600)
        auth = {"schema_version": 1, "kind": "explicit_candidate_continuation_authorization",
            "execution_authorized": True, "continuation_plan_sha256": approved_plan_sha256,
            "authorization_source_sha256": file_sha256(root / "authorization-source.txt"),
            "limits": LIMITS, "created_epoch_ms": time.time()*1000,
            "historical_capacity_transferred": False, "automatic_retries": 0}
        save(root / "execution-authorization.json", auth)
        with ledger(root, manifest, create=True) as book:
            save(root / "accounting-before.json", book.summary())
    return {"authorized": True, "continuation_plan_sha256": approved_plan_sha256,
            "limits": LIMITS, "engine_loads": 0, "generation_dispatches": 0}


def verify_original_process_closure(manifest, collector):
    """Compare PID plus saved process start identity, allowing recycled PIDs."""
    old = Path(manifest["original_root"])
    checked = 0
    for arm in ("baseline", "candidate"):
        target = old / arm / "execution" / (arm + "-1")
        identities = {}
        for line in (target / "resources.jsonl").read_text().splitlines():
            for process in json.loads(line).get("owned_processes", []):
                if process.get("startId") is not None:
                    identities.setdefault(process["pid"], set()).add(str(process["startId"]))
        terminal = read_json(target / "result.json")
        chrome = terminal.get("driver", {}).get("chrome_pid")
        if chrome and alive(chrome) and chrome not in identities:
            raise ValueError("Original live browser PID lacks a recorded process start identity")
        for pid, starts in identities.items():
            checked += 1
            if not alive(pid):
                continue
            try:
                current = str(collector.usage(pid)["startId"])
            except ProcessLookupError:
                continue
            if current in starts:
                raise ValueError("An original owned process still exists; continuation blocked")
    return {"verified": True, "checked_saved_process_identities": checked,
            "comparison": "PID and native process startId; recycled PIDs are distinct processes",
            "original_graceful_candidate_unload_required": False}


def grant_load(root, manifest):
    plan = read_json(root / "continuation-plan.json")
    auth = _authorization(root, plan)
    now, mono = time.time(), time.monotonic()
    grant = {"started_epoch_ms": now*1000, "deadline_epoch_ms": now*1000+5400000,
        "started_monotonic": mono, "deadline_monotonic": mono+5400,
        "manifest_sha256": canonical_hash(manifest), "continuation_plan_sha256": canonical_hash(plan),
        "execution_authorization_sha256": canonical_hash(auth), "load_attempts_consumed": 1,
        "clock_scope": "separately authorized continuation load through cleanup; original clock untouched"}
    save(root / "load-grant.json", grant)
    return grant


def validate_intent(intent, config, code_sha, index):
    if (type(index) is not int or not 0 <= index <= 35 or len(config["items"]) != 34
            or config["arm"] != "candidate"):
        raise ValueError("Continuation request exceeds its candidate-only 34-item schedule")
    expected = {"kind": "load" if index == 0 else "generation", "arm": "candidate",
        "request_id": "load-1" if index == 0 else f"generation-{index}",
        "attempt_id": "candidate-1", "code_version_sha256": code_sha,
        "configuration_sha256": config["configurationSha256"],
        "artifact_sha256": config["convertedArtifactSha256"], "run_id": config["runId"]}
    if index:
        expected.update(warmup=index == 1, max_tokens=16 if index == 1 else 1024,
                        sequence_index=index-1, item_id="__warmup__" if index == 1 else config["items"][index-2]["item_id"])
    if any(intent.get(k) != v for k,v in expected.items()):
        raise ValueError("Continuation dispatch identity, order, warmup or token cap changed")
    return expected


def recover_observations(target, terminal, validated_events, config, book, seen_samples):
    """Keep validated completions if the final event file is corrupt or changes.

    Raw files remain untouched. Unknown or conflicting observations are never
    assigned to another slot. A consumed slot without a trustworthy response
    becomes the same explicit ambiguous-dispatch placeholder used originally.
    """
    schedule = ["__warmup__", *[i["item_id"] for i in config["items"]]]
    rows = {a["request_id"]: a for a in book.attempts()}
    kept = {e["data"]["request_id"]: e["data"] for e in validated_events
            if e.get("type") in {"warmup-sample", "task-sample"}}
    errors = []
    def usable(sample, index):
        if not isinstance(sample, dict):
            return False
        tokens, duration = sample.get("generated_tokens"), sample.get("response_latency_ms")
        return (sample.get("raw_output") is None or isinstance(sample.get("raw_output"), str)) and (
            tokens is None or type(tokens) is int and 0 <= tokens <= (16 if index == 1 else 1024)) and (
            duration is None or type(duration) in (int, float) and math.isfinite(duration) and 0 <= duration <= 120000) and (
            sample.get("finish_reason") in {"stop", "eos", "length", "error", "timeout", "cancelled", "interrupted"})
    try:
        current = read_events(target)
    except Exception as exc:
        current = []
        errors.append("final_event_read_failed:" + type(exc).__name__)
    count = 0
    for event in current:
        if not isinstance(event, dict):
            errors.append("malformed_final_event")
            break
        if event.get("type") not in {"warmup-sample", "task-sample"}:
            continue
        count += 1
        sample = event.get("data")
        rid = f"generation-{count}"
        if (count > len(schedule) or not isinstance(sample, dict)
                or rid not in rows or sample.get("request_id") != rid
                or sample.get("item_id") != schedule[count-1]
                or sample.get("warmup") is not (count == 1)
                or not usable(sample, count)
                or (count in seen_samples and canonical_hash(sample) != seen_samples[count])):
            errors.append("conflicting_or_unscheduled_final_completion")
            break
        kept[rid] = sample
    if count < len(seen_samples):
        errors.append("final_completion_prefix_removed")
    recovered = [(True, s) for s in terminal.get("warmupOutputs", [])] + [(False, s) for s in terminal.get("outputs", [])]
    by_item = {item_id: index for index,item_id in enumerate(schedule, 1)}
    for warmup, sample in recovered:
        index = by_item.get(sample.get("item_id")) if isinstance(sample, dict) else None
        rid = f"generation-{index}"
        if index is None or rid not in rows or warmup is not (index == 1) or not usable(sample, index):
            errors.append("unscheduled_terminal_recovery")
            continue
        if rid not in kept:
            kept[rid] = sample
    warm, scored = [], []
    for index, item_id in enumerate(schedule, 1):
        rid = f"generation-{index}"
        if rid not in rows:
            continue
        sample = kept.get(rid)
        if sample is None:
            sample = {"item_id": item_id, "raw_output": None, "finish_reason": "error",
                "generated_tokens": None, "input_tokens": None, "response_latency_ms": None,
                "execution_error": "Acknowledged or ambiguous dispatch without a recovered response",
                "dispatch_outcome": "acknowledged_or_ambiguous", "request_id": rid, "warmup": index == 1}
        (warm if index == 1 else scored).append(sample)
    return warm, scored, errors


def execute(root):
    root, manifest = load(root, require_authorization=True)
    config = read_json(root/"baseline-config.json")
    order = {item["item_id"]: i for i, item in enumerate(config["items"])}
    arm = manifest.get("arm", "baseline")
    attempt_id = arm+"-1"
    code_sha = canonical_hash(manifest["files"])
    target, benchmark = root/"execution"/attempt_id, root/"benchmark"
    if target.exists() or (root/"load-grant.json").exists():
        raise ValueError("Continuation execution already started; retries forbidden")
    collector = _collector(benchmark)
    previous_cleanup = verify_original_process_closure(manifest, collector)
    target.mkdir(parents=True, mode=0o700)  # Refuses repeat execution, even before a load.
    save(target/"original-process-closure.json", previous_cleanup)
    process, terminal, error, cleanup = None, None, None, {}
    retained, handled, seen_samples = set(), set(), {}
    validated_events = []
    def observed(index, sample):
        print(json.dumps({"event": "generation_observed", "sequence": index, "warmup": index == 1,
            "original_schedule_position": None if index == 1 else 94 + index - 1,
            "tokens": sample["generated_tokens"], "finish": sample["finish_reason"]}), flush=True)
    watchdog, watchdog_stopped, watchdog_triggered = None, threading.Event(), threading.Event()
    began, grant = time.monotonic(), None
    with _lock(root), ledger(root, manifest) as book:
        if book.attempts() or (root/"load-grant.json").exists():
            raise ValueError("continuation capacity already consumed; retries forbidden")
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
                    events = reconcile_events(target, book, config, handled, seen_samples, on_observed=observed)
                    validated_events = events
                    allowed_intents = {"load-intent.json", *{f"generation-{i}-intent.json" for i in range(1, 36)}}
                    if {p.name for p in target.glob("*-intent.json")} - allowed_intents:
                        raise ValueError("unscheduled dispatch intent")
                    for index in range(36):
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
                            validated_events = assert_predecessor(index, target, book, config, handled, seen_samples, grant,
                                                                  on_observed=observed)
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
        success = bool(not error and terminal and terminal.get("status") == "completed" and len(warm) == 1 and len(scored) == 34)
        if any(a["status"] in {"reserved", "dispatched"} for a in book.attempts()):
            book.recover_interrupted(executor_stopped=not cleanup["retained_owned_pids_alive"] and process is not None and process.poll() is not None)
        book.cancel(reason="continuation_completed" if success else "continuation_stopped")
        result = {"status": "completed" if success else "stopped", "error": error,
            "engine_load_attempts": int(grant is not None), "warmup_dispatches": int(1 in handled),
            "scored_dispatches": sum(i >= 2 for i in handled), "generation_dispatches": len(book.attempts()),
            "warmup_outputs": warm, "outputs": scored, "cleanup": cleanup,
            "recovery_integrity_errors": recovery_errors,
            "first_load_through_cleanup_seconds": None if not grant else time.monotonic()-grant["started_monotonic"],
            "supervisor_seconds": time.monotonic()-began, "generation_budget": book.summary(),
            "candidate_generations": len(book.attempts()) if arm == "candidate" else 0,
            "heldout_generations": sum(i >= 2 for i in handled), "frontier_requests": 0,
            "original_schedule_offset": 94, "original_schedule_length": 128,
            "baseline_reruns": 0, "completed_candidate_reruns": 0,
            "continuation_plan_sha256": canonical_hash(read_json(root/"continuation-plan.json"))}
        save(target/"supervisor-result.json", result)
        save(root/"accounting-after.json", result)
        return result



def grade(root):
    root, manifest = load(root, require_authorization=True)
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
    result = {"schema_version": 1, "kind": "browser_ifeval_candidate_continuation_results", "author": "Arhaan2",
        "manifest_sha256": canonical_hash(manifest), "records": records, "grades": grades,
        "item_metrics": metrics, "metrics": metrics, "summary": summary,
        "original_schedule_offset": 94, "original_schedule_length": 128,
        "continuation_plan_sha256": canonical_hash(read_json(root/"continuation-plan.json"))}
    save(root/arm/"graded.json", result)
    save(root/arm/"records.json", records)
    save(root/arm/"grades.json", grades)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "authorize", "execute", "grade"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    if args.command == "authorize":
        if not args.authorization or not args.approved_plan_sha256:
            parser.error("authorize requires a later explicit --authorization and --approved-plan-sha256")
        result = authorize(args.root, args.authorization, args.approved_plan_sha256)
    elif args.command == "execute":
        result = execute(args.root)
        result = {k:v for k,v in result.items() if k not in {"outputs", "warmup_outputs"}}
    elif args.command == "grade":
        result = grade(args.root)
    else:
        root, manifest = load(args.root)
        result = {"prepared_plan_verified": True, "continuation_plan_sha256": canonical_hash(read_json(root / "continuation-plan.json")),
                  "execution_authorized": (root / "execution-authorization.json").exists(),
                  "manifest_sha256": canonical_hash(manifest), "engine_loads": 0, "generation_dispatches": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

