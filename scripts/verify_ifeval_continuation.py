#!/usr/bin/env python3
"""Independent terminal continuation audit. Arhaan Aggarwal (Arhaan2).

Reads evidence and immutable SQLite only; never loads a model or invokes a
grader. Public output contains aggregate checks and identities. Detailed file,
request and process evidence remains in the private continuation destination.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

from verify_ifeval_heldout import (read, digest, file_digest, file_map, ledger,
                                 SETTINGS, GENERATION, EXPECTED_FREEZE,
                                 EXPECTED_ARTIFACTS, CLEANUP)

PLAN = "f366b08d4d9a25d54307af6638b1c5fa66f8ee9b87174e087de18d1ad8f9dad8"
MANIFEST = "0faa9ec2b548c5070106701fbe965640b44a678c1ee6b185c3d25aa7fcd4ab83"
RUNNERS = {
    "ifeval_heldout_runner.py": "cfbd51cbb79349442770022bb0dfcbb3a7b089debc3437bab9baeab69cca8bcf",
    "ifeval_candidate_continuation.py": "bda28582f9efbc04b2fd1cb624ce33baa5edbbc5db85097cf965caf79fb54e65",
    "ifeval_supervisor_gate.py": "5609efba2f7bf9ef334c75b8abdb2345e315dc32de8d3e5066d6cb25529a6c66",
}
LIMITS = {"engine_load_attempts": 1, "warmup_dispatches": 1, "scored_dispatches": 34,
          "generation_dispatches": 35, "generated_tokens": 34832, "warmup_tokens": 16,
          "tokens_per_scored_request": 1024, "request_seconds": 120, "wall_seconds": 5400,
          "automatic_retries": 0, "new_weight_proposals": 0, "development_reruns": 0,
          "baseline_reruns": 0, "completed_candidate_reruns": 0, "frontier_requests": 0}
GEN_LIMITS = {"editor_requests": 0, "model_probes": 35, "requested_sequences": 35,
              "generated_tokens": 34832, "elapsed_seconds": 5400, "billed_units": 0}
CORE = ("item_id", "raw_output", "finish_reason", "generated_tokens", "input_tokens",
        "response_latency_ms", "loading_time_ms", "execution_error", "cancelled", "truncated",
        "valid_completion", "webgpu_submitted_batches", "local_webgpu_inference_observed",
        "request_id", "warmup", "assistant_prefix_tokens", "request_max_tokens")


def event_rows(target):
    return [json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()]


def select(events, kind, rid=None):
    return [e for e in events if e["type"] == kind and
            (rid is None or isinstance(e.get("data"), dict) and e["data"].get("request_id") == rid)]


def core(rows):
    return [{key: row.get(key) for key in CORE} for row in rows]


def original_dispatch_audit(old, items, check):
    details, sessions = {}, {}
    for arm, count in (("baseline", 128), ("candidate", 94)):
        target = old / arm / "execution" / (arm + "-1")
        events = event_rows(target)
        good, meta, attempts, _ = ledger(old / arm / "generation-budget.sqlite3")
        calls = select(events, "generation-call-intent")
        samples = select(events, "task-sample")
        ids = [f"generation-{i}" for i in range(1, count + 2)]
        grades = read(old / arm / arm / "graded.json")
        check("original_" + arm + "_exact_dispatch_prefix", good and meta["cancelled"] == 1
              and [r["request_id"] for r in attempts] == ids
              and all(r["status"] == "completed" and r["retry_of"] is None for r in attempts)
              and [e["data"]["request_id"] for e in calls] == ids
              and [e["data"]["item_id"] for e in samples] == [i["item_id"] for i in items[:count]]
              and {p.name for p in target.glob("generation-*-ack.json")} == {i + "-ack.json" for i in ids}
              and len(select(events, "warmup-sample")) == 1
              and len(grades["records"]) == 128
              and all(r["status"] in {"ok", "truncated"} and len(r["attempts"]) == 1
                      for r in grades["records"][:count]))
        result = read(target / "supervisor-result.json")
        sessions[arm] = {"loads": result["engine_load_attempts"],
                         "warmups": result["warmup_dispatches"],
                         "dispatches": len(attempts), "scored_dispatches": count,
                         "generated_tokens": sum(r["observed_generated_tokens"] or 0 for r in attempts),
                         "elapsed_seconds": result["first_load_through_cleanup_seconds"]}
        details[arm] = {"events_file_sha256": file_digest(target / "events.jsonl"),
                        "ledger_file_sha256": file_digest(old / arm / "generation-budget.sqlite3"),
                        "request_ids": ids, "scored_item_ids": [i["item_id"] for i in items[:count]]}
        if arm == "candidate":
            pending = "generation-96"
            before = select(events, "task-sample", "generation-95")
            after = select(events, "gate-request", pending)
            denial = select(events, "driver-gate-error")
            check("original_suffix_never_dispatched_and_not_ambiguous",
                  not set(i["item_id"] for i in items[94:]) & {e["data"]["item_id"] for e in calls}
                  and not (target / (pending + "-ack.json")).exists()
                  and (target / (pending + "-intent.json")).is_file()
                  and all(r["request_id"] != pending for r in attempts)
                  and not select(events, "generation-call-intent", pending)
                  and not select(events, "gate-ack-received", pending)
                  and not select(events, "driver-gate-ack-forwarded", pending)
                  and len(before) == len(after) == 1
                  and before[0]["receivedEpochMs"] < after[0]["receivedEpochMs"]
                  and denial[-1]["data"]["acknowledged"] is False
                  and denial[-1]["data"]["execution_state"] == "no approval forwarded"
                  and all(r["status"] == "missing" and not r["attempts"] for r in grades["records"][94:]))
            details[arm]["last_completion_epoch_ms"] = before[0]["receivedEpochMs"]
            details[arm]["unacknowledged_gate_epoch_ms"] = after[0]["receivedEpochMs"]
    return sessions, details


def process_recheck(targets, benchmark):
    sys.path.insert(0, str(benchmark))
    path = benchmark / "harness/collectors/macos/native_collector.py"
    spec = importlib.util.spec_from_file_location("continuation_audit_native_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = {}
    for name, target in targets.items():
        identities = {}
        for line in (target / "resources.jsonl").read_text().splitlines():
            for process in json.loads(line).get("owned_processes", []):
                if process.get("startId") is not None:
                    identities.setdefault(process["pid"], set()).add(str(process["startId"]))
        rows = []
        for pid, starts in sorted(identities.items()):
            current, state = None, "absent"
            try:
                os.kill(pid, 0)
                current = str(module.usage(pid)["startId"])
                state = "same_process_alive" if current in starts else "pid_reused"
            except ProcessLookupError:
                pass
            rows.append({"pid": pid, "saved_start_ids": sorted(starts), "current_start_id": current, "state": state})
        terminal = read(target / "result.json") if (target / "result.json").exists() else {}
        chrome = terminal.get("driver", {}).get("chrome_pid")
        if chrome and chrome not in identities:
            try:
                os.kill(chrome, 0)
                rows.append({"pid": chrome, "state": "live_browser_identity_unavailable"})
            except ProcessLookupError:
                rows.append({"pid": chrome, "state": "absent"})
        results[name] = rows
    return results


def verify(root):
    root = Path(root).resolve()
    target = root / "execution/candidate-1"
    # These reads refuse an audit before execution and grading have terminated.
    execution = read(target / "supervisor-result.json")
    evidence = read(root / "candidate/graded.json")
    checks, details = {}, {}
    def check(name, value):
        checks[name] = bool(value)
    manifest, plan, auth = (read(root / name) for name in
                            ("manifest.json", "continuation-plan.json", "execution-authorization.json"))
    old = Path(manifest["original_root"])
    config, items = read(root / "baseline-config.json"), read(root / "remaining-items.json")
    old_items, freeze = read(old / "evaluation-items.json"), read(old / "evaluation-freeze.json")
    old_config = read(old / "candidate/baseline-config.json")
    check("exact_prepared_plan_manifest_and_freeze", digest(plan) == PLAN and digest(manifest) == MANIFEST
          and digest(freeze) == EXPECTED_FREEZE and plan["manifest_sha256"] == MANIFEST
          and read(root / "ready.json") == {"continuation_plan_sha256": PLAN}
          and plan["limits"] == manifest["limits"] == LIMITS
          and digest(read(root / "merge-plan.json")) == plan["merge_plan_sha256"])
    check("new_explicit_authorization_bound", auth["execution_authorized"] is True
          and auth["continuation_plan_sha256"] == PLAN and auth["limits"] == LIMITS
          and auth["authorization_source_sha256"] == file_digest(root / "authorization-source.txt")
          and bool((root / "authorization-source.txt").read_bytes().strip())
          and auth["historical_capacity_transferred"] is False and auth["automatic_retries"] == 0
          and auth["created_epoch_ms"] > plan["created_epoch_ms"]
          and all((root / "authorization-source.txt").read_bytes().strip() != p.read_bytes().strip()
                  for p in (old / "authorization-source.txt", old / "candidate/authorization-source.txt")))
    historical = read(root / "preservation-before.json")["files"]
    historical_bad = [p for p, pin in historical.items() if file_digest(p) != pin]
    pins_bad = [p for p, pin in manifest["files"].items() if file_digest(p) != pin]
    check("858_historical_files_preserved_byte_for_byte", len(historical) == 858 and not historical_bad)
    check("1117_frozen_source_build_dependency_and_evidence_pins", len(manifest["files"]) == 1117 and not pins_bad)
    details["changed_historical_files"], details["changed_manifest_files"] = historical_bad, pins_bad
    repo = Path(__file__).resolve().parents[1]
    check("original_and_continuation_runner_identities_separate", all(
          file_digest(repo / "weight_space_lab" / name) == pin for name, pin in RUNNERS.items()))
    check("exact_disjoint_remaining_schedule_and_settings", items == old_items[94:] and len(items) == 34
          and sum(len(i["target"]["instruction_id_list"]) for i in items) == 45
          and plan["task_order"] == [i["item_id"] for i in items]
          and config["items"] == old_config["items"][94:]
          and config["settings"] == manifest["settings"] == old_config["settings"] == SETTINGS
          and manifest["checkpoint"]["sha256"] == config["sourceCheckpointSha256"] == EXPECTED_ARTIFACTS["candidate"][0]
          and manifest["artifact_sha256"] == config["convertedArtifactSha256"] == EXPECTED_ARTIFACTS["candidate"][1]
          and config["assets"] == old_config["assets"] and config["model"] == old_config["model"]
          and config["warmupPrompt"] == old_config["warmupPrompt"] == "Reply with OK."
          and config["originalScheduleOffset"] == 94 and config["originalScheduleLength"] == 128
          and config["totalLimitMs"] == 5400000
          and config["configurationSha256"] == manifest["configuration_sha256"]
          == digest({k: v for k, v in config.items() if k != "configurationSha256"}))
    for arm, (source, deployed) in EXPECTED_ARTIFACTS.items():
        observed = freeze["artifact_verification"][arm]
        check(arm + "_actual_source_and_deployed_model_bytes", file_map(observed["source_path"]) == observed["source_files"]
              and digest(observed["source_files"]) == source
              and file_map(observed["deployed_path"]) == observed["deployed_files"]
              and digest(observed["deployed_files"]) == deployed)
    check("continuation_served_artifact_and_build_bytes", all(file_digest(root / "benchmark/public" /
          urlparse(a["url"]).path.lstrip("/")) == a["sha256"] for a in config["assets"] + config["servedBuild"]))
    historical_sessions, details["original_dispatch_audit"] = original_dispatch_audit(old, old_items, check)
    merge_plan = read(root / "merge-plan.json")
    for arm, count in (("baseline", 128), ("candidate", 94)):
        original = read(old / arm / arm / "graded.json")
        refs = [{"item_id": r["item_id"], "run_id": r["run_id"],
                 "configuration_sha256": r["configuration_sha256"],
                 "run_record_sha256": digest(r), "grade_record_sha256": digest(g)}
                for r, g in zip(original["records"][:count], original["grades"][:count])]
        check("original_" + arm + "_record_grade_references_preserved", refs ==
              merge_plan["original_" + arm + "_record_grade_references"])
    good, meta, attempts, budget_events = ledger(root / "generation-budget.sqlite3")
    frozen = json.loads(meta["frozen_json"])
    grant = read(root / "load-grant.json") if (root / "load-grant.json").exists() else None
    events = event_rows(target) if (target / "events.jsonl").exists() else []
    terminal = read(target / "result.json") if (target / "result.json").exists() else {}
    complete = execution["status"] == "completed"
    code_sha = digest(manifest["files"])
    samples = [e["data"] for e in events if e["type"] in {"warmup-sample", "task-sample"}]
    outputs = execution["warmup_outputs"] + execution["outputs"]
    calls = select(events, "generation-call-intent")
    expected_ids = [f"generation-{i}" for i in range(1, len(attempts) + 1)]
    check("dedicated_ledger_authorization_and_plan_binding", good and frozen["limits"] == GEN_LIMITS
          and frozen["run_id"] == config["runId"]
          and frozen["binding"]["continuation_plan_sha256"] == PLAN
          and frozen["binding"]["manifest_sha256"] == MANIFEST
          and frozen["binding"]["execution_authorization_sha256"] == digest(auth)
          and meta["created_at"] * 1000 >= auth["created_epoch_ms"])
    before = read(root / "accounting-before.json")
    check("new_ledger_started_empty_and_closed_without_retry", before["cancelled"] is False
          and before["charged"]["model_probes"] == before["charged"]["generated_tokens"] == 0
          and meta["cancelled"] == 1
          and meta["cancellation_reason"] == ("continuation_completed" if complete else "continuation_stopped")
          and [a["request_id"] for a in attempts] == expected_ids and len(attempts) <= 35
          and all(a["retry_of"] is None and a["attempt_index"] == 0 and a["kind"] == "probe"
                  and a["status"] not in {"reserved", "dispatched"} for a in attempts)
          and read(root / "accounting-after.json") == execution)
    check("exact_dispatch_capacity_and_consumption", execution["engine_load_attempts"] == int(grant is not None)
          and execution["warmup_dispatches"] <= 1 and execution["scored_dispatches"] <= 34
          and execution["generation_dispatches"] == len(attempts)
          and all(a["requested_sequences"] == 1 and a["timeout_seconds"] == 120
                  and a["max_generated_tokens"] == (16 if i == 0 else 1024) for i, a in enumerate(attempts))
          and sum(a["max_generated_tokens"] for a in attempts) <= 34832
          and sum(a["charged_generated_tokens"] for a in attempts) <= 34832)
    check("all_observed_consumption_preserved", sum(a["observed_generated_tokens"] or 0 for a in attempts)
          == sum(o["generated_tokens"] or 0 for o in outputs)
          and execution["generation_budget"]["charged"]["generated_tokens"]
          == sum(a["charged_generated_tokens"] for a in attempts)
          and execution["generation_budget"]["charged"]["pending_requests"] == 0
          and all(o.get("generated_tokens") is None or 0 <= o["generated_tokens"] <=
                  (16 if o.get("warmup") else 1024) for o in outputs))
    by_id = {a["request_id"]: a for a in attempts}
    gates = True
    expected_ack_stems = set()
    for index in range(0 if grant else 1, len(attempts) + 1):
        rid = "load-1" if index == 0 else f"generation-{index}"
        stem = "load" if index == 0 else rid
        expected_ack_stems.add(stem + "-ack.json")
        intent, ack = (read(target / (stem + suffix + ".json")) for suffix in ("-intent", "-ack"))
        identity = {"kind": "load" if index == 0 else "generation", "request_id": rid,
                    "arm": "candidate", "attempt_id": "candidate-1", "code_version_sha256": code_sha}
        gates &= all(intent.get(k) == ack.get(k) == value for k, value in identity.items())
        gates &= (ack["approved"] is True and ack["deadline_epoch_ms"] == grant["deadline_epoch_ms"]
                  and intent["configuration_sha256"] == config["configurationSha256"]
                  and intent["artifact_sha256"] == config["convertedArtifactSha256"]
                  and intent["run_id"] == config["runId"])
        stages = [select(events, kind, rid) for kind in
                  ("gate-request", "driver-gate-ack-forwarded", "gate-ack-received")]
        if complete:
            gates &= all(len(s) == 1 for s in stages)
        if all(len(s) == 1 for s in stages):
            gates &= events.index(stages[0][0]) < events.index(stages[1][0]) < events.index(stages[2][0])
        call = select(events, "engine-call-intent") if index == 0 else select(events, "generation-call-intent", rid)
        if complete:
            gates &= len(call) == 1
        if call:
            gates &= len(call) == 1 and len(stages[2]) == 1 and events.index(stages[2][0]) < events.index(call[0])
        if index:
            row = by_id[rid]
            gates &= (intent["warmup"] is (index == 1) and intent["sequence_index"] == index - 1
                      and intent["max_tokens"] == (16 if index == 1 else 1024)
                      and intent["item_id"] == ("__warmup__" if index == 1 else items[index - 2]["item_id"])
                      and row["reserved_at"] <= row["dispatched_at"] <= row["finished_at"])
            if row["status"] == "completed":
                gates &= [e["kind"] for e in budget_events if e["request_id"] == rid] == ["reserved", "dispatched", "completed"]
            if index > 1:
                previous = select(events, "warmup-sample" if index == 2 else "task-sample", f"generation-{index - 1}")
                gates &= len(previous) == 1 and len(stages[0]) == 1 and events.index(previous[0]) < events.index(stages[0][0])
    check("durable_gates_exact_order_and_no_extra_dispatch", gates
          and {p.name for p in target.glob("*-ack.json")} == expected_ack_stems
          and len(select(events, "engine-call-intent")) <= int(grant is not None)
          and len(calls) <= len(attempts) and len({e["data"]["request_id"] for e in calls}) == len(calls)
          and all(e["data"]["request_id"] in by_id for e in calls))
    known = [o for o in outputs if o.get("raw_output") is not None]
    check("durable_samples_and_records_preserve_observed_text", core(samples) == core(known)
          and all(a["observed_generated_tokens"] == o["generated_tokens"]
                  and math.isclose(a["observed_elapsed_seconds"], o["response_latency_ms"] / 1000, abs_tol=1e-8)
                  for o in known for a in [by_id[o["request_id"]]]))
    if grant:
        check("fresh_load_through_cleanup_clock_bound", grant["continuation_plan_sha256"] == PLAN
              and grant["manifest_sha256"] == MANIFEST and grant["execution_authorization_sha256"] == digest(auth)
              and grant["load_attempts_consumed"] == 1
              and math.isclose(grant["deadline_monotonic"] - grant["started_monotonic"], 5400, abs_tol=1e-6)
              and math.isclose(grant["deadline_epoch_ms"] - grant["started_epoch_ms"], 5400000, abs_tol=0.01)
              and grant["started_epoch_ms"] >= auth["created_epoch_ms"]
              and 0 <= execution["first_load_through_cleanup_seconds"] <= 5400
              and (not terminal or terminal["driver"]["deadline_epoch_ms"] == grant["deadline_epoch_ms"]))
    effective = select(events, "task-effective-settings")
    loaded = select(events, "task-artifact-loaded")
    if attempts:
        audits = select(events, "task-input-token-audit")
        assets = {e["data"]["url"]: e["data"]["sha256"] for e in select(events, "asset-verified")}
        check("frozen_runtime_generation_and_tokenizer_observed", len(effective) == len(loaded) == 1
              and effective[0]["data"]["generation"] == GENERATION
              and effective[0]["data"]["contextWindow"] == 4096 and effective[0]["data"]["maxHistorySize"] == 1
              and loaded[0]["data"]["sourceCheckpointSha256"] == EXPECTED_ARTIFACTS["candidate"][0]
              and loaded[0]["data"]["convertedArtifactSha256"] == EXPECTED_ARTIFACTS["candidate"][1]
              and loaded[0]["data"]["runtime"] == "WebLLM 0.2.86"
              and loaded[0]["data"]["executionBackend"] == "webgpu"
              and loaded[0]["data"]["serverInferenceFallback"] is False
              and loaded[0]["data"]["allAssetsVerified"] is True
              and assets == {a["url"]: a["sha256"] for a in config["assets"]}
              and all(a["data"]["effective_context_window"] == 4096 for a in audits)
              and {a["data"]["item_id"]: a["data"]["rendered_input_tokens"] for a in audits if not a["data"]["warmup"]}
              == {i["item_id"]: i["rendered_input_tokens"] for i in config["items"][:len(audits) - 1]})
        fetched = read(target / "fetch-preflight.json")
        check("fetch_preflight_bound_without_model_activity", fetched["status"] == "passed"
              and fetched["code_version_sha256"] == code_sha
              and fetched["engine_loads"] == fetched["generation_dispatches"] == 0)
    records, grades = evidence["records"], evidence["grades"]
    check("only_suffix_records_and_grade_references", len(records) == len(grades) == 34
          and records == read(root / "candidate/records.json") and grades == read(root / "candidate/grades.json")
          and evidence["manifest_sha256"] == MANIFEST and evidence["continuation_plan_sha256"] == PLAN
          and [r["item_id"] for r in records] == [g["item_id"] for g in grades] == [i["item_id"] for i in items]
          and all(g["run_record_sha256"] == digest(r) and g["item_sha256"] == digest(i)
                  and g["policy_sha256"] == digest(manifest["policy"]) and g["grader"] == manifest["grader"]
                  and r["configuration_sha256"] == manifest["configuration_sha256"]
                  and r["checkpoint"] == manifest["checkpoint"] and r["seed"] == g["seed"] == 42
                  and r["run_id"] == config["runId"] and r["stage"] == "full" and r["split"] == "test"
                  and r["benchmark"] == g["benchmark"] == "ifeval"
                  and r["sample_index"] == g["sample_index"] == 0
                  and r["retry_policy"] == {"mode": "none", "max_attempts": 1}
                  for i, r, g in zip(items, records, grades)))
    observed = {o["item_id"]: o for o in execution["outputs"]}
    check("suffix_records_match_actual_outputs_and_attempts", all(
          (r["status"] == "missing" and not r["attempts"] and r["raw_output"] is None) if r["item_id"] not in observed else
          (r["raw_output"] == observed[r["item_id"]]["raw_output"] and len(r["attempts"]) == 1
           and r["token_counts"]["output"] == observed[r["item_id"]]["generated_tokens"])
          for r in records))
    intent = read(root / "candidate/grading-intent.json")
    check("new_grading_bound_to_terminal_execution_without_retry", intent["manifest_sha256"] == MANIFEST
          and intent["execution_file_sha256"] == file_digest(target / "supervisor-result.json")
          and intent["grader_retries"] == 0
          and (not grant or intent["created_epoch_ms"] >= grant["started_epoch_ms"] +
               execution["first_load_through_cleanup_seconds"] * 1000 - 10))
    counts, valid = Counter(), True
    for i, g in zip(items, grades):
        if g["status"] != "graded":
            continue
        n, metrics = len(i["target"]["instruction_id_list"]), g["metrics"]
        for mode in ("strict", "loose"):
            flags = metrics["instruction_" + mode]
            valid &= len(flags) == n and all(type(f) is bool for f in flags)
            valid &= metrics[f"prompt_level_{mode}_accuracy"] == all(flags)
            valid &= metrics[f"instruction_{mode}_correct"] == sum(flags)
            counts["prompt_" + mode] += int(all(flags))
            counts["instruction_" + mode] += sum(flags)
    expected_metrics = {f"{kind}_level_{mode}_accuracy": {"passed": counts[kind + "_" + mode],
                        "scheduled": 34 if kind == "prompt" else 45,
                        "value": counts[kind + "_" + mode] / (34 if kind == "prompt" else 45)}
                        for kind in ("prompt", "instruction") for mode in ("strict", "loose")}
    summary = evidence["summary"]
    unresolved = sum(g["status"] in {"grader_error", "grader_timeout"} for g in grades)
    check("suffix_grades_independently_aggregated", valid and summary["metrics"] == expected_metrics
          and summary["scheduled_prompts"] == 34 and summary["scheduled_instructions"] == 45
          and summary["unresolved_grader_errors"] == unresolved
          and summary["execution_status_counts"] == dict(Counter(r["status"] for r in records))
          and summary["known_generated_tokens"] == sum(r["token_counts"]["output"] or 0 for r in records)
          and math.isclose(summary["generation_seconds"], sum(r["duration_seconds"] or 0 for r in records), abs_tol=1e-6))
    if complete:
        check("complete_suffix_exactly_34_with_no_execution_failure", len(attempts) == len(calls) == len(samples) == 35
              and len(execution["warmup_outputs"]) == 1 and len(execution["outputs"]) == 34
              and [o["item_id"] for o in execution["outputs"]] == [i["item_id"] for i in items]
              and execution["error"] is None and execution["recovery_integrity_errors"] == []
              and terminal["status"] == "completed" and core(outputs) == core(terminal["warmupOutputs"] + terminal["outputs"])
              and all(a["status"] == "completed" and a["conservative"] == a["limit_violation"] == 0 for a in attempts)
              and all(0 <= o["response_latency_ms"] <= 120000 and o["local_webgpu_inference_observed"] is True
                      and o["webgpu_submitted_batches"] > 0 for o in outputs)
              and not any(e["type"] in {"driver-error", "driver-gate-error", "failure-classification"}
                          or e["type"] == "event" and isinstance(e.get("data"), dict)
                          and e["data"].get("type") == "device-lost" for e in events))
    cleanup = execution["cleanup"]
    check("terminal_process_cleanup_recorded", all(cleanup.get(k) is True for k in CLEANUP if k != "unload_confirmed")
          and cleanup.get("retained_owned_pids_alive") == []
          and (not complete or cleanup.get("unload_confirmed") is True))
    targets = {"baseline": old / "baseline/execution/baseline-1",
               "candidate_interrupted": old / "candidate/execution/candidate-1", "candidate_continuation": target}
    details["process_identity_rechecks"] = process_recheck(targets, root / "benchmark")
    check("all_three_sessions_owned_process_identities_absent", all(
          r["state"] in {"absent", "pid_reused"} for rows in details["process_identity_rechecks"].values() for r in rows))
    closure = read(target / "original-process-closure.json")
    check("prior_processes_checked_before_continuation", closure["verified"] is True
          and closure["checked_saved_process_identities"] > 0)
    sessions = {"baseline": historical_sessions["baseline"],
                "candidate_interrupted": historical_sessions["candidate"],
                "candidate_continuation": {"loads": execution["engine_load_attempts"],
                    "warmups": execution["warmup_dispatches"], "dispatches": len(attempts),
                    "scored_dispatches": execution["scored_dispatches"],
                    "generated_tokens": sum(a["observed_generated_tokens"] or 0 for a in attempts),
                    "elapsed_seconds": execution["first_load_through_cleanup_seconds"] or 0}}
    cumulative = {key: sum(v[key] for v in sessions.values()) for key in sessions["baseline"]}
    check("cumulative_count_reconciliation", cumulative["loads"] <= 3 and cumulative["warmups"] <= 3
          and cumulative["dispatches"] <= 259 and cumulative["scored_dispatches"] <= 256
          and (not complete or (cumulative["loads"], cumulative["warmups"], cumulative["dispatches"], cumulative["scored_dispatches"])
               == (3, 3, 259, 256)))
    details["remaining_incomplete_item_ids"] = [r["item_id"] for r in records if r["status"] not in {"ok", "truncated"}]
    details["ledger_attempts"] = attempts
    details["verification_epoch_ms"] = time.time() * 1000
    result = {"schema_version": 1, "kind": "independent_ifeval_continuation_verification", "author": "Arhaan2",
              "passed": all(checks.values()), "checks": checks,
              "complete_suffix_execution_and_grading": complete and unresolved == 0,
              "identities": {"continuation_plan_sha256": PLAN, "continuation_manifest_sha256": MANIFEST,
                  "original_evaluation_freeze_sha256": EXPECTED_FREEZE, "verifier_sha256": file_digest(__file__),
                  "runner_source_sha256": RUNNERS, "candidate_source_sha256": EXPECTED_ARTIFACTS["candidate"][0],
                  "candidate_artifact_sha256": EXPECTED_ARTIFACTS["candidate"][1]},
              "measurements": {"historical_files_preserved": len(historical), "prepared_file_pins_verified": len(manifest["files"]),
                  "session_usage": sessions, "cumulative_usage": cumulative, "suffix_metrics": expected_metrics,
                  "suffix_completion_counts": dict(Counter(r["status"] for r in records)), "unresolved_grader_errors": unresolved,
                  "remaining_incomplete_prompts": len(details["remaining_incomplete_item_ids"]),
                  "process_identities_rechecked": {k: len(v) for k, v in details["process_identity_rechecks"].items()},
                  "continuation_graceful_unload_confirmed": cleanup.get("unload_confirmed") is True,
                  "original_candidate_graceful_unload_confirmed": False},
              "scope": "Read-only independent bytes, dispatch, authorization, ledger, clock, process identity and grade-reference audit. Grades aggregated without regrading; no claim of editor isolation or answer correctness.",
              "engine_loads_by_verifier": 0, "grader_calls_by_verifier": 0, "input_mutations": 0}
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/ifeval-browser-v1-heldout-continuation")
    parser.add_argument("--output", default="reports/ifeval-browser-v1-heldout-continuation-combined/verification.json")
    parser.add_argument("--private-output", default="artifacts/ifeval-browser-v1-heldout-continuation/independent-verification-private.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and any(Path(p).exists() for p in (args.output, args.private_output)):
        parser.error("Audit output exists; preserve it and use a new destination or --check-only")
    result, details = verify(args.root)
    if not args.check_only:
        for filename, value in ((args.private_output, details), (args.output, result)):
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
