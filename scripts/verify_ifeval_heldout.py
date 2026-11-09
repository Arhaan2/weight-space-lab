#!/usr/bin/env python3
"""Independent read-only final-subset evidence verification. Arhaan2.

Run only after held-out execution and scoring have finished. This stdlib-only
verifier never invokes inference, a converter, or a grader. Its optional public
report contains hashes and aggregate checks, never prompts or machine paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote, urlparse


EXPECTED_FREEZE = "89ed9703869ab7c8b1032b3d8c5fa59fefed0215d9d0341016684d60b39b8a27"
EXPECTED_ARTIFACTS = {
    "baseline": ("f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d",
                 "5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0"),
    "candidate": ("f9c767c06ec3c17dbdfedc43b695d64d025ae7393d3f425cee71ef45c358d6bf",
                  "6e18cc4906ce18786b9197bcc7fb5f93ffc706d808b0809150349546eef3627b"),
}
CLEANUP = ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed",
           "local_harness_exit_confirmed", "controller_exit_verified", "chrome_os_exit_verified")
SETTINGS = {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 1024,
            "contextWindow": 4096, "warmups": 1, "warmupTokens": 16,
            "requestLimitMs": 120000, "thinking": False}
GENERATION = {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 1024,
              "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
              "n": 1, "ignore_eos": False, "extra_body": {"enable_thinking": False}}
LEDGER_LIMITS = {"editor_requests": 0, "model_probes": 129, "requested_sequences": 129,
                 "generated_tokens": 131088, "elapsed_seconds": 5400, "billed_units": 0}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_map(folder):
    paths = sorted(Path(folder).iterdir())
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("artifact contains a nonregular file")
    return {p.name: {"sha256": file_digest(p), "size_bytes": p.stat().st_size}
            for p in paths if p.name != "artifact-manifest.json"}


def ledger(path):
    # Terminal ledgers must have no unapplied WAL; immutable mode guarantees
    # that even SQLite sidecar files are not mutated by this verifier.
    path = Path(path).resolve()
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("terminal ledger still has unapplied WAL")
    db = sqlite3.connect("file:" + quote(str(path)) + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    try:
        good = db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        meta = dict(db.execute("SELECT * FROM metadata WHERE singleton=1").fetchone())
        attempts = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY reserved_at,request_id")]
        events = [dict(r) for r in db.execute("SELECT * FROM events ORDER BY event_index")]
        return good, meta, attempts, events
    finally:
        db.close()


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def verify_grades(arm_root, arm, manifest, items, check):
    evidence = read(arm_root / arm / "graded.json")
    records, grades = evidence["records"], evidence["grades"]
    check(arm + "_grade_schedule", len(records) == len(grades) == len(items) == 128
          and records == read(arm_root / arm / "records.json")
          and grades == read(arm_root / arm / "grades.json")
          and [r["item_id"] for r in records] == [g["item_id"] for g in grades] == [i["item_id"] for i in items])
    check(arm + "_all_record_and_grade_hashes", evidence["manifest_sha256"] == digest(manifest)
          and all(g["run_record_sha256"] == digest(r) and g["item_sha256"] == digest(item)
              and g["policy_sha256"] == digest(manifest["policy"]) and g["grader"] == manifest["grader"]
              and r["configuration_sha256"] == manifest["configuration_sha256"]
              and r["checkpoint"] == manifest["checkpoint"] and r["seed"] == g["seed"] == 42
              and r["sample_index"] == g["sample_index"] == 0 and r["split"] == "test"
              and r["stage"] == "full" and r["benchmark"] == g["benchmark"] == "ifeval"
              and r["retry_policy"] == {"mode": "none", "max_attempts": 1}
              for item, r, g in zip(items, records, grades)))
    counts, failures = Counter(), Counter()
    valid = True
    for item, grade in zip(items, grades):
        if grade["status"] != "graded":
            continue
        n, m = len(item["target"]["instruction_id_list"]), grade["metrics"]
        for mode in ("strict", "loose"):
            flags = m["instruction_" + mode]
            valid &= (len(flags) == n and all(type(f) is bool for f in flags)
                      and m[f"prompt_level_{mode}_accuracy"] == all(flags)
                      and m[f"instruction_{mode}_correct"] == sum(flags)
                      and m[f"instruction_level_{mode}_accuracy"] == sum(flags) / n)
            counts["prompt_" + mode] += int(all(flags))
            counts["instruction_" + mode] += sum(flags)
        valid &= grade["correct"] == all(m["instruction_strict"]) and m["instruction_count"] == n
        failures.update(name for name, flag in zip(item["target"]["instruction_id_list"], m["instruction_strict"]) if not flag)
    expected = {f"{kind}_level_{mode}_accuracy": {
        "passed": counts[kind + "_" + mode], "scheduled": 128 if kind == "prompt" else 173,
        "value": counts[kind + "_" + mode] / (128 if kind == "prompt" else 173)}
        for kind in ("prompt", "instruction") for mode in ("strict", "loose")}
    summary = evidence["summary"]
    unresolved = sum(g["status"] in {"grader_error", "grader_timeout"} for g in grades)
    check(arm + "_independent_official_aggregation", valid and summary["metrics"] == expected
          and summary["scheduled_prompts"] == 128 and summary["scheduled_instructions"] == 173
          and summary["strict_failures_by_instruction"] == dict(failures)
          and summary["execution_status_counts"] == dict(Counter(r["status"] for r in records))
          and summary["unresolved_grader_errors"] == unresolved
          and summary["semantic_grading_complete"] is (unresolved == 0))
    check(arm + "_scored_resource_totals", summary["known_generated_tokens"] == sum(r["token_counts"]["output"] or 0 for r in records)
          and summary["known_input_tokens"] == sum(r["token_counts"]["input"] or 0 for r in records)
          and math.isclose(summary["generation_seconds"], sum(r["duration_seconds"] or 0 for r in records), abs_tol=1e-6))
    return evidence, expected


def verify_arm(root, arm, manifest, config, items, freeze, check):
    target = root / arm
    evidence, metrics = verify_grades(target, arm, manifest, items, check)
    good, meta, attempts, ledger_events = ledger(target / "generation-budget.sqlite3")
    frozen = json.loads(meta["frozen_json"])
    check(arm + "_fresh_ledger_binding", good and frozen["run_id"] == config["runId"]
          and frozen["binding"]["manifest_sha256"] == digest(manifest)
          and frozen["limits"] == LEDGER_LIMITS
          and all(a["retry_of"] is None and a["kind"] == "probe" and a["attempt_index"] == 0 for a in attempts))
    execution_dir = target / "execution" / (arm + "-1")
    if not execution_dir.exists():
        check(arm + "_unexecuted_slots_retained", not attempts and meta["cancelled"] == 0
              and all(r["status"] == "missing" and r["attempts"] == [] for r in evidence["records"]))
        return {"complete": False, "engine_loads": 0, "warmups": 0, "scored_dispatches": 0,
                "dispatches": 0, "generated_tokens": 0, "metrics": metrics,
                "cleanup_verified": None, "records": evidence["records"], "grades": evidence["grades"]}
    executed = read(execution_dir / "supervisor-result.json")
    terminal = read(execution_dir / "result.json") if (execution_dir / "result.json").exists() else {}
    events = [json.loads(line) for line in (execution_dir / "events.jsonl").read_text().splitlines()]
    grant = read(target / "load-grant.json") if (target / "load-grant.json").exists() else None
    code_sha = digest(manifest["files"])
    complete = executed["status"] == "completed"
    outputs = executed["warmup_outputs"] + executed["outputs"]
    observed_tokens = sum(o["generated_tokens"] for o in outputs if type(o.get("generated_tokens")) is int)
    check(arm + "_terminal_ledger_and_no_retries", meta["cancelled"] == 1 and len(attempts) <= 129
          and all(a["status"] not in {"reserved", "dispatched"} and a["limit_violation"] == 0 for a in attempts)
          and executed["generation_dispatches"] == len(attempts)
          and executed["engine_load_attempts"] == int(grant is not None)
          and read(target / "accounting-after.json") == executed)
    check(arm + "_observed_request_bounds", all(o.get("generated_tokens") is None or
          0 <= o["generated_tokens"] <= (16 if o.get("warmup") else 1024) for o in outputs)
          and all(o.get("response_latency_ms") is None or 0 <= o["response_latency_ms"] <= 120000 for o in outputs)
          and observed_tokens <= 131088)
    check(arm + "_ledger_observation_parity", sum(a["observed_generated_tokens"] or 0 for a in attempts) == observed_tokens
          and sum(a["max_generated_tokens"] for a in attempts) <= 131088
          and executed["generation_budget"]["observed"]["generated_tokens"] == observed_tokens
          and all(a["requested_sequences"] == 1 and a["timeout_seconds"] == 120 for a in attempts))
    cleanup = executed["cleanup"]
    clean = all(cleanup.get(key) is True for key in CLEANUP) and cleanup.get("retained_owned_pids_alive") == []
    process_clean = all(cleanup.get(key) is True for key in CLEANUP if key != "unload_confirmed") and cleanup.get("retained_owned_pids_alive") == []
    check(arm + "_cleanup_record_matches_os", not alive(terminal.get("driver", {}).get("chrome_pid"))
          and not any(alive(pid) for pid in cleanup.get("retained_owned_pids_alive", []))
          and process_clean and (clean or not complete))
    if grant:
        check(arm + "_bounded_load_through_cleanup_clock", grant["manifest_sha256"] == digest(manifest)
              and grant["evaluation_freeze_sha256"] == digest(freeze)
              and 0 < grant["deadline_monotonic"] - grant["started_monotonic"] <= 5400.000001
              and 0 < grant["deadline_epoch_ms"] - grant["started_epoch_ms"] <= 5400000.01
              and 0 <= executed["first_load_through_cleanup_seconds"] <= 5400
              and (not terminal or terminal.get("driver", {}).get("deadline_epoch_ms") == grant["deadline_epoch_ms"]))
    if complete:
        check(arm + "_complete_schedule", len(outputs) == 129 and len(executed["warmup_outputs"]) == 1
              and [o["item_id"] for o in executed["outputs"]] == [i["item_id"] for i in items]
              and executed["engine_load_attempts"] == executed["warmup_dispatches"] == 1
              and executed["scored_dispatches"] == 128 and executed["generation_dispatches"] == 129
              and terminal["status"] == "completed" and executed["error"] is None
              and all(a["status"] == "completed" and a["conservative"] == 0 for a in attempts))
    pending_intents, stop_diagnosis = set(), None
    if attempts and all(a["status"] == "completed" for a in attempts):
        scored_count = len(attempts) - 1
        check(arm + "_completed_dispatch_prefix_and_unexecuted_slots", len(outputs) == len(attempts)
              and len(executed["warmup_outputs"]) == 1 and executed["warmup_dispatches"] == 1
              and executed["scored_dispatches"] == len(executed["outputs"]) == scored_count
              and [o["item_id"] for o in executed["outputs"]] == [i["item_id"] for i in items[:scored_count]]
              and all(r["status"] == "missing" and r["attempts"] == [] and r["raw_output"] is None
                      for r in evidence["records"][scored_count:]))
        core_fields = ("item_id", "raw_output", "finish_reason", "generated_tokens", "input_tokens",
                       "response_latency_ms", "loading_time_ms", "execution_error", "cancelled", "truncated",
                       "valid_completion", "webgpu_submitted_batches", "local_webgpu_inference_observed",
                       "request_id", "warmup", "assistant_prefix_tokens", "request_max_tokens")
        core = lambda rows: [{key: row.get(key) for key in core_fields} for row in rows]
        check(arm + "_event_terminal_supervisor_parity", core(outputs) == core(terminal["warmupOutputs"] + terminal["outputs"])
              == core([e["data"] for e in events if e["type"] in ("warmup-sample", "task-sample")]))
        check(arm + "_records_match_raw_observations", all(r["raw_output"] == o["raw_output"]
              and r["token_counts"]["output"] == o["generated_tokens"] and r["token_counts"]["input"] == o["input_tokens"]
              and r["duration_seconds"] == o["response_latency_ms"] / 1000 and len(r["attempts"]) == 1
              for r, o in zip(evidence["records"], executed["outputs"])))
        check(arm + "_local_gpu_execution", all(o["local_webgpu_inference_observed"] is True
              and o["webgpu_submitted_batches"] > 0 for o in outputs)
              and not any(e["type"] == "event" and e["data"].get("type") == "device-lost" for e in events))
        effective = [e["data"] for e in events if e["type"] == "task-effective-settings"]
        audits = [e["data"] for e in events if e["type"] == "task-input-token-audit"]
        check(arm + "_effective_runtime_and_tokenizer", len(effective) == 1 and effective[0]["generation"] == GENERATION
              and effective[0]["contextWindow"] == 4096 and effective[0]["maxHistorySize"] == 1
              and len(attempts) <= len(audits) <= min(129, len(attempts) + 1)
              and all(a["effective_context_window"] == 4096 for a in audits)
              and {a["item_id"]: a["rendered_input_tokens"] for a in audits if not a["warmup"]}
                  == {i["item_id"]: i["rendered_input_tokens"] for i in config["items"][:len(audits) - 1]})
        loaded = [e["data"] for e in events if e["type"] == "task-artifact-loaded"]
        assets = {e["data"]["url"]: e["data"]["sha256"] for e in events if e["type"] == "asset-verified"}
        fetched = read(execution_dir / "fetch-preflight.json")
        check(arm + "_fetched_and_loaded_artifact", len(loaded) == 1
              and loaded[0]["sourceCheckpointSha256"] == config["sourceCheckpointSha256"]
              and loaded[0]["convertedArtifactSha256"] == config["convertedArtifactSha256"]
              and loaded[0]["modelLib"] == config["model"]["modelLib"]
              and loaded[0]["modelId"] == config["model"]["id"]
              and loaded[0]["allAssetsVerified"] is True and loaded[0]["executionBackend"] == "webgpu"
              and loaded[0]["serverInferenceFallback"] is False and loaded[0]["runtime"] == "WebLLM 0.2.86"
              and assets == {a["url"]: a["sha256"] for a in config["assets"]}
              and fetched["status"] == "passed" and fetched["engine_loads"] == fetched["generation_dispatches"] == 0
              and fetched["code_version_sha256"] == code_sha)
        def indices(kind, rid=None):
            return [i for i, e in enumerate(events) if e["type"] == kind
                    and (rid is None or e["data"].get("request_id") == rid)]
        gates, stems = True, set()
        attempts_by_id = {a["request_id"]: a for a in attempts}
        for index in range(len(attempts) + 1):
            stem, rid = ("load", "load-1") if index == 0 else (f"generation-{index}", f"generation-{index}")
            stems.add(stem)
            intent, ack = (read(execution_dir / (stem + suffix + ".json")) for suffix in ("-intent", "-ack"))
            expected = {"kind": "load" if index == 0 else "generation", "request_id": rid,
                        "arm": arm, "attempt_id": arm + "-1", "code_version_sha256": code_sha}
            gates &= all(intent.get(k) == v and ack.get(k) == v for k, v in expected.items())
            gates &= (intent["configuration_sha256"] == config["configurationSha256"]
                      and intent["artifact_sha256"] == config["convertedArtifactSha256"]
                      and intent["run_id"] == config["runId"] and ack["approved"] is True
                      and ack["deadline_epoch_ms"] == grant["deadline_epoch_ms"])
            stages = [indices("gate-request", rid), indices("driver-gate-ack-forwarded", rid),
                      indices("gate-ack-received", rid), indices("engine-call-intent") if index == 0 else indices("generation-call-intent", rid)]
            gates &= all(len(stage) == 1 for stage in stages)
            if all(len(stage) == 1 for stage in stages):
                gates &= stages[0][0] < stages[1][0] < stages[2][0] < stages[3][0]
            if index:
                row = attempts_by_id[rid]
                cap = 16 if index == 1 else 1024
                gates &= (intent["warmup"] is (index == 1) and intent["max_tokens"] == cap
                          and intent["sequence_index"] == index - 1 and row["max_generated_tokens"] == cap
                          and row["reserved_at"] <= row["dispatched_at"] <= row["finished_at"]
                          and intent["created_at_epoch_ms"] <= row["reserved_at"] * 1000 + 1
                          and row["dispatched_at"] * 1000 <= (execution_dir / (stem + "-ack.json")).stat().st_mtime * 1000 + 1
                          and [e["kind"] for e in ledger_events if e["request_id"] == rid] == ["reserved", "dispatched", "completed"])
                if index > 1:
                    gates &= intent["item_id"] == items[index - 2]["item_id"]
                    previous = indices("warmup-sample") if index == 2 else [i for i, e in enumerate(events)
                        if e["type"] == "task-sample" and e["data"]["item_id"] == items[index - 3]["item_id"]]
                    gates &= len(previous) == 1 and len(stages[0]) == 1 and previous[0] < stages[0][0]
        pending_intents = {p.name for p in execution_dir.glob("*-intent.json")} - {s + "-intent.json" for s in stems}
        pending_valid = not pending_intents
        if not complete and pending_intents == {f"generation-{len(attempts) + 1}-intent.json"}:
            rid = f"generation-{len(attempts) + 1}"
            intent = read(execution_dir / (rid + "-intent.json"))
            previous = indices("task-sample", f"generation-{len(attempts)}")
            following = indices("gate-request", rid)
            pending_valid = (intent["item_id"] == items[len(attempts) - 1]["item_id"]
                and intent["request_id"] == rid and intent["sequence_index"] == len(attempts)
                and intent["kind"] == "generation" and intent["warmup"] is False and intent["max_tokens"] == 1024
                and intent["configuration_sha256"] == config["configurationSha256"]
                and intent["artifact_sha256"] == config["convertedArtifactSha256"]
                and intent["code_version_sha256"] == code_sha
                and rid not in attempts_by_id and not indices("generation-call-intent", rid)
                and not indices("driver-gate-ack-forwarded", rid) and not indices("gate-ack-received", rid)
                and len(previous) == len(following) == 1 and previous[0] < following[0])
            if pending_valid:
                prior_sample_time = events[previous[0]]["receivedEpochMs"]
                next_gate_time = events[following[0]]["receivedEpochMs"]
                prior_finished_time = attempts_by_id[f"generation-{len(attempts)}"]["finished_at"] * 1000
                stop_diagnosis = {
                    "last_completed_scored_schedule_position": len(attempts) - 1,
                    "next_unexecuted_scored_schedule_position": len(attempts),
                    "next_generation_intent_without_reservation_ack_or_model_call": True,
                    "next_gate_after_prior_sample_ms": next_gate_time - prior_sample_time,
                    "next_durable_intent_after_prior_sample_ms": intent["created_at_epoch_ms"] - prior_sample_time,
                    "prior_ledger_completion_after_next_durable_intent_ms": prior_finished_time - intent["created_at_epoch_ms"],
                    "observed_worker_request_order_valid": True,
                    "inferred_mechanism": "The supervisor reads an event snapshot before scanning newly written intent files. The preserved order is consistent with a newer next-request intent being seen before the preceding completion is added to its in-memory seen_samples set. The exact supervisor snapshot was not recorded, so this mechanism is an inference.",
                }
        check(arm + "_exact_durable_dispatch_gates", gates and pending_valid
              and {p.name for p in execution_dir.glob("*-ack.json")} == {s + "-ack.json" for s in stems}
              and len(indices("engine-call-intent")) == 1 and len(indices("generation-call-intent")) == len(attempts))
    return {"complete": complete, "engine_loads": executed["engine_load_attempts"],
            "warmups": executed["warmup_dispatches"], "scored_dispatches": executed["scored_dispatches"],
            "dispatches": len(attempts), "generated_tokens": observed_tokens,
            "metrics": metrics, "cleanup_verified": clean,
            "process_cleanup_verified": process_clean, "graceful_unload_confirmed": cleanup.get("unload_confirmed") is True,
            "forced_termination": cleanup.get("forced_termination") is True,
            "denied_terminal_generation_intents": len(pending_intents),
            "supervisor_error": executed["error"],
            "stop_diagnosis": stop_diagnosis,
            "unexecuted_prompts": 128 - executed["scored_dispatches"],
            "unexecuted_instructions": sum(len(i["target"]["instruction_id_list"]) for i in items[executed["scored_dispatches"]:]),
            "grant": grant, "load_through_cleanup_seconds": executed["first_load_through_cleanup_seconds"],
            "records": evidence["records"], "grades": evidence["grades"],
            "completion_counts": evidence["summary"]["execution_status_counts"],
            "grader_errors": evidence["summary"]["unresolved_grader_errors"]}


def verify(root):
    root = Path(root).resolve()
    old = root.parent / "ifeval-browser-v1"
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
    freeze, ready = read(root / "evaluation-freeze.json"), read(root / "evaluation-ready.json")
    run = read(root / "run-result.json")  # Refuse verification while execution is live.
    run_intent = read(root / "single-run-intent.json")
    check("exact_predeclared_freeze", digest(freeze) == EXPECTED_FREEZE
          and ready == {"evaluation_freeze_sha256": EXPECTED_FREEZE}
          and freeze["execution_order"] == ["baseline", "candidate"]
          and freeze["label"] == "128-example IFEval on-device subset"
          and freeze["scheduled_prompts"] == 128 and freeze["scheduled_instructions"] == 173)
    check("private_authorization_bound", file_digest(root / "authorization-source.txt") == freeze["authorization_source_sha256"]
          and run["evaluation_freeze_sha256"] == run_intent["evaluation_freeze_sha256"] == EXPECTED_FREEZE
          and run_intent["execution_order"] == ["baseline", "candidate"] and run_intent["automatic_retries"] == 0)
    before = read(root / "prior-execution-audit.json")
    check("no_prior_heldout_dispatches", before["heldout_dispatches_found"] == 0
          and all(file_digest(row["path"]) == row["sha256"] for row in before["dispatch_metadata"]))
    historical = read(root / "preservation-before.json")["files"]
    check("historical_evidence_unchanged", all(file_digest(p) == pin for p, pin in historical.items()))
    check("historical_ledgers_unchanged", all(file_digest(row["path"]) == row["sha256"]
          and len(ledger(row["path"])[2]) == row["attempts"] for row in before["historical_ledgers"]))
    data, items = read(old / "data/manifest.json"), read(root / "evaluation-items.json")
    check("exact_original_source_and_subset", digest(data) == freeze["original_data_manifest_sha256"]
          and file_digest(old / "data/manifest.json") == freeze["original_data_manifest_file_sha256"]
          and items == read(old / "data/evaluation-items.json")
          and digest(items) == freeze["evaluation_items_sha256"] == data["evaluation"]["items_sha256"]
          and [i["item_id"] for i in items] == freeze["task_order"] == data["evaluation"]["item_ids"]
          and len(items) == len({i["item_id"] for i in items}) == 128
          and sum(len(i["target"]["instruction_id_list"]) for i in items) == 173
          and all(i["split"] == "test" and i["benchmark"] == "ifeval" for i in items)
          and freeze["original_source"] == data["sources"]["evaluation"])
    heldout_ids = {i["item_id"] for i in items}
    historical_event_files, historical_event_matches = 0, 0
    for execution_root in (old / "execution", old / "candidate-comparison/execution"):
        for path in execution_root.rglob("events.jsonl"):
            historical_event_files += 1
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event["type"] in {"generation-call-intent", "task-sample", "gate-request"}:
                    metadata = event.get("data", {})
                    historical_event_matches += metadata.get("item_id") in heldout_ids
    check("historical_event_metadata_has_no_heldout_execution", historical_event_files == 2
          and historical_event_matches == 0)
    check("proposal_provenance_preserved", freeze["proposal_session_provenance"] == read(old / "proposal-provenance.json")
          and freeze["editor_isolation"] == "unverified" and freeze["selection_changed"] is False)
    split = read(root / "paired-split-manifest.json")
    paired = {mode: read(root / ("paired-analysis-" + mode + ".json")) for mode in ("strict", "loose")}
    refs = sorted([{"benchmark": i["benchmark"], "item_id": i["item_id"], "dataset_revision": i["dataset_revision"],
                    "item_sha256": digest(i), "instruction_count": len(i["target"]["instruction_id_list"])}
                   for i in items], key=lambda r: r["item_id"])
    check("paired_analysis_frozen_denominators", all(p["manifest_sha256"] == digest({k: v for k, v in p.items() if k != "manifest_sha256"})
          == freeze["paired_analysis_manifests"][mode] and p["split_manifest_sha256"] == digest(split)
          and p["source_manifest_sha256"] == digest(data) and p["items"] == refs
          and p["policy"] == freeze["policy"] and p["grader"] == freeze["grader"]
          and p["split"] == "test" and p["stage"] == "full" and p["metric"] == "prompt_level_" + mode + "_accuracy"
          for mode, p in paired.items()))
    arms, pin_count = {}, 0
    for arm, (source_sha, artifact_sha) in EXPECTED_ARTIFACTS.items():
        target = root / arm
        manifest, config = read(target / "manifest.json"), read(target / "baseline-config.json")
        observed = freeze["artifact_verification"][arm]
        check(arm + "_actual_source_and_deployed_hashes", file_map(observed["source_path"]) == observed["source_files"]
              and digest(observed["source_files"]) == observed["source_checkpoint_sha256"] == source_sha
              and file_map(observed["deployed_path"]) == observed["deployed_files"]
              and digest(observed["deployed_files"]) == observed["artifact_sha256"] == artifact_sha
              and manifest["checkpoint"]["sha256"] == config["sourceCheckpointSha256"] == source_sha
              and manifest["artifact_sha256"] == config["convertedArtifactSha256"] == artifact_sha)
        check(arm + "_all_frozen_source_pins", all(file_digest(p) == value for p, value in manifest["files"].items()))
        pin_count += len(manifest["files"])
        check(arm + "_manifest_and_configuration", digest(manifest) == freeze["arms"][arm]["manifest_sha256"]
              and read(target / "ready.json") == {"manifest_sha256": digest(manifest)}
              and digest({k: v for k, v in config.items() if k != "configurationSha256"}) == config["configurationSha256"]
              and config["configurationSha256"] == manifest["configuration_sha256"]
              and config["settings"] == manifest["settings"] == SETTINGS and config["evaluationStage"] == "heldout"
              and config["study"] == "ifeval-browser-v1" and config["totalLimitMs"] == 5400000
              and config["warmupPrompt"] == freeze["warmup_prompt"] == "Reply with OK."
              and config["model"]["quantization"] == "q4f16_1"
              and [{"item_id": i["item_id"], "prompt": i["task_text"]} for i in items]
                  == [{"item_id": i["item_id"], "prompt": i["prompt"]} for i in config["items"]])
        check(arm + "_served_asset_and_build_pins", all(file_digest(root / "benchmark/public" / urlparse(e["url"]).path.lstrip("/")) == e["sha256"]
              for e in config["assets"] + config["servedBuild"]))
        arms[arm] = verify_arm(root, arm, manifest, config, items, freeze, check)
    totals = {key: sum(a[key] for a in arms.values()) for key in
              ("engine_loads", "warmups", "scored_dispatches", "dispatches", "generated_tokens")}
    check("aggregate_authorized_capacity", totals["engine_loads"] <= 2 and totals["warmups"] <= 2
          and totals["scored_dispatches"] <= 256 and totals["dispatches"] <= 258 and totals["generated_tokens"] <= 262176
          and run["automatic_retries"] == 0 and freeze["limits"]["automatic_retries"] == 0
          and all(freeze["limits"][k] == 0 for k in ("new_weight_proposals", "development_reruns", "frontier_requests")))
    overall_seconds = None
    if totals["engine_loads"]:
        clock = read(root / "aggregate-clock.json")
        check("freeze_precedes_first_engine_load", before["checked_epoch_seconds"] * 1000 <= freeze["created_epoch_ms"]
              <= (root / "evaluation-freeze.json").stat().st_mtime * 1000
              <= run_intent["started_epoch_seconds"] * 1000 <= clock["first_started_epoch_ms"])
        last = arms["candidate"] if arms["candidate"].get("grant") else arms["baseline"]
        overall_seconds = last["grant"]["started_monotonic"] + last["load_through_cleanup_seconds"] - clock["first_started_monotonic"]
        check("overall_first_load_through_final_cleanup_limit", clock["evaluation_freeze_sha256"] == EXPECTED_FREEZE
              and clock["deadline_monotonic"] == clock["first_started_monotonic"] + 10800
              and 0 <= overall_seconds <= 10800
              and all(a.get("grant") is None or a["grant"]["aggregate_clock_sha256"] == digest(clock) for a in arms.values()))
        if arms["candidate"].get("grant"):
            check("sequential_engines_with_verified_unload", arms["baseline"]["complete"]
                  and arms["baseline"]["cleanup_verified"] is True
                  and arms["baseline"]["grant"]["started_monotonic"] + arms["baseline"]["load_through_cleanup_seconds"]
                      <= arms["candidate"]["grant"]["started_monotonic"])
    flips = {}
    for mode in ("strict", "loose"):
        changes = Counter()
        for b, c in zip(arms["baseline"]["grades"], arms["candidate"]["grades"]):
            if b["status"] != "graded" or c["status"] != "graded":
                changes["unresolved"] += 1
            else:
                d = int(c["metrics"][f"prompt_level_{mode}_accuracy"]) - int(b["metrics"][f"prompt_level_{mode}_accuracy"])
                changes["improved" if d > 0 else "regressed" if d < 0 else "tied"] += 1
        flips[mode] = {key: changes[key] for key in ("improved", "regressed", "tied", "unresolved")}
    complete = all(a["complete"] and a.get("grader_errors", 0) == 0 for a in arms.values())
    check("reported_completion_matches_evidence", (run["status"] == "completed") is complete)
    return {"schema_version": 1, "kind": "independent_ifeval_heldout_verification", "author": "Arhaan2",
        "label": "128-example IFEval on-device subset", "passed": all(checks.values()), "checks": checks,
        "complete_paired_evaluation": complete,
        "identities": {"evaluation_freeze_sha256": EXPECTED_FREEZE, "verifier_sha256": file_digest(__file__),
                       "artifacts": {a: {"source_checkpoint_sha256": v[0], "deployed_artifact_sha256": v[1]}
                                     for a, v in EXPECTED_ARTIFACTS.items()}},
        "measurements": {"aggregate_usage": totals, "overall_load_through_cleanup_seconds": overall_seconds,
            "metrics": {a: v["metrics"] for a, v in arms.items()}, "paired_prompt_outcomes": flips,
            "cleanup_verified": {a: v["cleanup_verified"] for a, v in arms.items()},
            "process_cleanup_verified": {a: v.get("process_cleanup_verified") for a, v in arms.items()},
            "graceful_unload_confirmed": {a: v.get("graceful_unload_confirmed") for a, v in arms.items()},
            "forced_termination": {a: v.get("forced_termination") for a, v in arms.items()},
            "supervisor_error": {a: v.get("supervisor_error") for a, v in arms.items()},
            "stop_diagnosis": {a: v.get("stop_diagnosis") for a, v in arms.items()},
            "denied_terminal_generation_intents": {a: v.get("denied_terminal_generation_intents", 0) for a, v in arms.items()},
            "unexecuted_prompts": {a: v.get("unexecuted_prompts", 128) for a, v in arms.items()},
            "unexecuted_instructions": {a: v.get("unexecuted_instructions", 173) for a, v in arms.items()},
            "historical_files_preserved": len(historical), "frozen_source_pins_verified": pin_count,
            "historical_ledgers_preserved": len(before["historical_ledgers"]),
            "historical_execution_event_files_checked": historical_event_files},
        "scope": "A passing audit validates preserved evidence and accounting; it does not imply a complete evaluation or graceful unload. Read-only byte, schedule, ledger, timing, process-cleanup and grade-record verification; official grades independently aggregated, not rerun. No claim of editor isolation or substantive answer correctness.",
        "engine_loads_by_verifier": 0, "grader_calls_by_verifier": 0, "input_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/ifeval-browser-v1-heldout")
    parser.add_argument("--output", default="reports/ifeval-browser-v1-heldout/verification.json")
    parser.add_argument("--check-only", action="store_true", help="Print sanitized verification without writing a report")
    args = parser.parse_args()
    output = Path(args.output)
    if not args.check_only and output.exists():
        raise SystemExit("Verification report already exists; preserve it")
    try:
        result = verify(args.root)
    except Exception as error:
        result = {"schema_version": 1, "kind": "independent_ifeval_heldout_verification",
                  "passed": False, "complete_verification": False, "error_category": type(error).__name__,
                  "engine_loads_by_verifier": 0, "grader_calls_by_verifier": 0, "input_mutations": 0}
    if not args.check_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
