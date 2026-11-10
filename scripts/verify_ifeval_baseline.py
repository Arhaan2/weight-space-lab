#!/usr/bin/env python3
"""Read-only independent evidence verification; never dispatches model work.

Only the explicitly requested new verification output is written. All assertions
use independent stdlib joins, hashes, SQLite reads, and event-order checks.
The emitted JSON contains aggregate counts and hashes, never prompts, item IDs,
machine paths, or private diagnostic contents. Run after execution and grading.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    checks, measurements = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)

    manifest = read(root / "manifest.json")
    config = read(root / "baseline-config.json")
    data = read(root / "data/manifest.json")
    source = read(root / "data/source-manifest.json")
    items = read(root / "data/development-items.json")
    evaluation = read(root / "data/evaluation-items.json")
    graded = read(root / "baseline/graded.json")
    records = read(root / "baseline/records.json")
    grades = read(root / "baseline/grades.json")
    execution_dir = root / "execution/baseline-1"
    executed = read(execution_dir / "supervisor-result.json")
    terminal = read(execution_dir / "result.json")
    events = [json.loads(line) for line in (execution_dir / "events.jsonl").read_text().splitlines()]
    grant = read(root / "load-grant.json")
    manifest_sha, code_sha = digest(manifest), digest(manifest["files"])
    check("manifest_binding", read(root / "ready.json") == {"manifest_sha256": manifest_sha}
          and graded["manifest_sha256"] == manifest_sha and grant["manifest_sha256"] == manifest_sha)
    check("frozen_file_hashes", all(file_digest(path) == pin for path, pin in manifest["files"].items()))
    check("authorization_hash", file_digest(root / "authorization-source.txt") == manifest["authorization_source_sha256"])
    check("data_manifest_self_hash", digest({k: v for k, v in data.items() if k != "sha256"}) == data["sha256"])
    check("source_manifest_references", source["sources"] == data["sources"])
    practice_path = root / "data/source" / Path(source["sources"]["practice"]["path"]).name
    google_path = repo / "artifacts/phase04-benchmark-cache/ifeval"
    check("source_file_pins", file_digest(practice_path) == source["sources"]["practice"]["sha256"]
          and file_digest(google_path) == source["sources"]["evaluation"]["sha256"])
    check("source_license_hashes", all(file_digest(root / "data/source" / name) == pin
          for name, pin in source["cached_ai2_licenses"].items()))
    google = {str(row["key"]): row for row in [json.loads(line) for line in google_path.read_text().splitlines()]}
    check("original_evaluation_prompts_and_arguments_preserved", all(
        item["task_text"] == google[item["item_id"]]["prompt"]
        and item["target"] == {k: google[item["item_id"]][k] for k in ("instruction_id_list", "kwargs")}
        and item["source"]["row_id"] == item["item_id"] for item in evaluation))
    check("selected_data_and_task_order", len(items) == 32 and len(evaluation) == 128
          and digest(items) == data["development"]["items_sha256"] == manifest["items_sha256"]
          and digest(evaluation) == data["evaluation"]["items_sha256"]
          and [i["item_id"] for i in items] == data["development"]["item_ids"]
          and [i["item_id"] for i in evaluation] == data["evaluation"]["item_ids"]
          and [{"item_id": i["item_id"], "prompt": i["task_text"]} for i in items]
              == [{"item_id": i["item_id"], "prompt": i["prompt"]} for i in config["items"]])
    token_audit = read(root / "data/tokenizer-count-audit.json")
    check("tokenizer_context_feasibility", file_digest(root / "data/tokenizer-count-audit.json") == data["bundled_tokenizer_audit"]["sha256"]
          and len(token_audit["counts"]) == 160
          and {i["item_id"]: i["metadata"]["rendered_input_tokens"] for i in items + evaluation}
              == {i["item_id"]: i["rendered_input_tokens"] for i in token_audit["counts"]}
          and all(i["rendered_input_tokens"] + 1024 <= 4096 for i in token_audit["counts"])
          and token_audit["engine_loads"] == token_audit["target_forward_passes"] == 0)
    check("official_grader_assets", all(file_digest(Path(manifest["official_ifeval_root"]) / name) == pin
          for name, pin in manifest["official_assets"]["files"].items())
          and file_digest(Path(manifest["official_ifeval_root"]) / "punkt_tab.zip") == manifest["official_assets"]["nltk_archive_sha256"])
    known = read(root / "data/mapping-known-answers.json")
    check("adapter_known_answers", known["passed"] is True and len(known["cases"]) == 42
          and len(known["covered_functions"]) == 18 and all(c["passed"] for c in known["cases"])
          and known["grader"] == manifest["grader"])
    expected_limits = {"engine_load_attempts": 1, "warmup_dispatches": 1, "scored_dispatches": 32,
                       "generation_dispatches": 33, "generated_tokens": 32784, "warmup_tokens": 16,
                       "tokens_per_scored_request": 1024, "request_seconds": 120, "wall_seconds": 5400,
                       "automatic_retries": 0, "candidate_generations": 0, "heldout_generations": 0, "frontier_requests": 0}
    check("authorized_bounds_and_mode", manifest["limits"] == expected_limits and manifest["arm"] == "baseline"
          and manifest["strict_ready"] is False and manifest["editor_isolation"] == "unverified"
          and not any(manifest[k] for k in ("candidate_execution_authorized", "heldout_execution_authorized", "frontier_execution_authorized")))
    expected_settings = {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 1024, "contextWindow": 4096,
                         "warmups": 1, "warmupTokens": 16, "requestLimitMs": 120000, "thinking": False}
    effective = [e["data"] for e in events if e["type"] == "task-effective-settings"]
    expected_generation = {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 1024,
                           "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
                           "n": 1, "ignore_eos": False, "extra_body": {"enable_thinking": False}}
    check("frozen_and_actual_generation_settings", manifest["settings"] == config["settings"] == expected_settings
          and len(effective) == 1 and effective[0]["generation"] == expected_generation
          and effective[0]["contextWindow"] == 4096 and effective[0]["maxHistorySize"] == 1
          and config["warmupPrompt"] == "Reply with OK.")
    observed_input = [e["data"] for e in events if e["type"] == "task-input-token-audit"]
    check("actual_loaded_tokenizer_and_context", len(observed_input) == 33
          and all(a["effective_context_window"] == 4096 and a["compiled_prefill_chunk_size"] == 1024 for a in observed_input)
          and {a["item_id"]: a["rendered_input_tokens"] for a in observed_input if not a["warmup"]}
              == {i["item_id"]: i["metadata"]["rendered_input_tokens"] for i in items})
    check("runtime_identity", config["configurationSha256"] == manifest["configuration_sha256"] == terminal["configuration_sha256"]
          and config["sourceCheckpointSha256"] == manifest["checkpoint"]["sha256"] == terminal["source_checkpoint_sha256"]
          and config["convertedArtifactSha256"] == manifest["artifact_sha256"] == terminal["converted_artifact_sha256"])
    check("raw_and_grade_schedule", len(records) == len(grades) == 32 and records == graded["records"] and grades == graded["grades"]
          and [r["item_id"] for r in records] == [i["item_id"] for i in items] == [g["item_id"] for g in grades]
          and all(r["seed"] == g["seed"] == 42 and r["sample_index"] == g["sample_index"] == 0
                  and r["split"] == "development" and r["benchmark"] == g["benchmark"] == "ifeval"
                  for r, g in zip(records, grades)))
    check("grade_content_bindings", all(g["run_record_sha256"] == digest(r) and g["item_sha256"] == digest(i)
          and g["policy_sha256"] == digest(manifest["policy"]) and g["grader"] == manifest["grader"]
          and r["configuration_sha256"] == manifest["configuration_sha256"] and r["checkpoint"] == manifest["checkpoint"]
          for i, r, g in zip(items, records, grades)))
    instruction_total = sum(len(i["target"]["instruction_id_list"]) for i in items)
    numerators, failures = Counter(), Counter()
    metric_vectors_valid = True
    for item, grade in zip(items, grades):
        if grade["status"] != "graded":
            continue
        m = grade["metrics"]
        n = len(item["target"]["instruction_id_list"])
        for mode in ("strict", "loose"):
            vector = m["instruction_" + mode]
            metric_vectors_valid &= len(vector) == n and all(type(v) is bool for v in vector)
            metric_vectors_valid &= m[f"prompt_level_{mode}_accuracy"] == all(vector)
            metric_vectors_valid &= m[f"instruction_{mode}_correct"] == sum(vector)
            metric_vectors_valid &= m[f"instruction_level_{mode}_accuracy"] == sum(vector) / n
            numerators["prompt_" + mode] += int(all(vector))
            numerators["instruction_" + mode] += sum(vector)
        metric_vectors_valid &= grade["correct"] == all(m["instruction_strict"]) and m["instruction_count"] == n
        failures.update(name for name, passed in zip(item["target"]["instruction_id_list"], m["instruction_strict"]) if not passed)
    check("official_metric_vectors", metric_vectors_valid)
    expected_metrics = {f"{kind}_level_{mode}_accuracy": {"passed": numerators[kind + "_" + mode],
                        "scheduled": 32 if kind == "prompt" else instruction_total,
                        "value": numerators[kind + "_" + mode] / (32 if kind == "prompt" else instruction_total)}
                        for kind in ("prompt", "instruction") for mode in ("strict", "loose")}
    summary = graded["summary"]
    unresolved = sum(g["status"] in ("grader_error", "grader_timeout") for g in grades)
    check("official_aggregation_and_failure_accounting", summary["metrics"] == expected_metrics
          and summary["scheduled_prompts"] == 32 and summary["scheduled_instructions"] == instruction_total
          and summary["unresolved_grader_errors"] == unresolved
          and summary["semantic_grading_complete"] == (unresolved == 0)
          and summary["strict_failures_by_instruction"] == dict(failures)
          and summary["execution_status_counts"] == dict(Counter(r["status"] for r in records)))
    outputs = executed["warmup_outputs"] + executed["outputs"]
    event_outputs = [e["data"] for e in events if e["type"] in ("warmup-sample", "task-sample")]
    terminal_outputs = terminal["warmupOutputs"] + terminal["outputs"]
    core_fields = ("item_id", "raw_output", "finish_reason", "generated_tokens", "input_tokens", "response_latency_ms",
                   "loading_time_ms", "execution_error", "cancelled", "truncated", "valid_completion",
                   "webgpu_submitted_batches", "local_webgpu_inference_observed", "request_id", "warmup",
                   "assistant_prefix_tokens", "request_max_tokens")
    core = lambda observations: [{key: observation.get(key) for key in core_fields} for observation in observations]
    check("event_supervisor_terminal_output_core_parity", core(event_outputs) == core(outputs) == core(terminal_outputs))
    check("complete_generation_schedule", len(outputs) == 33 and len(executed["warmup_outputs"]) == 1
          and [o["item_id"] for o in executed["outputs"]] == [i["item_id"] for i in items]
          and executed["engine_load_attempts"] == executed["warmup_dispatches"] == 1
          and executed["generation_dispatches"] == 33 and executed["scored_dispatches"] == 32)
    check("raw_outputs_match_browser_observations", all(r["raw_output"] == o["raw_output"]
          and r["token_counts"]["output"] == o["generated_tokens"]
          and r["token_counts"]["input"] == o["input_tokens"]
          and r["duration_seconds"] == o["response_latency_ms"] / 1000
          for r, o in zip(records, executed["outputs"])))
    check("no_retries_and_request_caps", all(len(r["attempts"]) == 1 and r["retry_policy"] == {"mode": "none", "max_attempts": 1} for r in records)
          and all(0 <= o["generated_tokens"] <= (16 if index == 0 else 1024)
                  and 0 <= o["response_latency_ms"] <= 120000 for index, o in enumerate(outputs)))
    check("gpu_dispatch_observations", all(o["local_webgpu_inference_observed"] is True
          and o["webgpu_submitted_batches"] > 0 for o in outputs)
          and not any(e["type"] == "event" and e["data"].get("type") == "device-lost" for e in events))
    db = sqlite3.connect("file:" + quote(str(root / "generation-budget.sqlite3")) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        check("ledger_integrity", db.execute("PRAGMA quick_check").fetchone()[0] == "ok")
        meta = dict(db.execute("SELECT * FROM metadata WHERE singleton=1").fetchone())
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY reserved_at,request_id")]
        ledger_events = [dict(row) for row in db.execute("SELECT * FROM events ORDER BY event_index")]
    finally:
        db.close()
    frozen = json.loads(meta["frozen_json"])
    check("ledger_manifest_binding", frozen["run_id"] == "ifeval-browser-v1-baseline"
          and frozen["binding"]["manifest_sha256"] == manifest_sha
          and frozen["limits"]["model_probes"] == 33 and frozen["limits"]["generated_tokens"] == 32784)
    check("ledger_terminal", meta["cancelled"] == 1 and meta["cancellation_reason"] == "baseline_completed"
          and len(attempts) == 33 and all(a["status"] == "completed" and a["retry_of"] is None
          and a["conservative"] == a["limit_violation"] == 0 for a in attempts))
    generation_tokens = sum(o["generated_tokens"] for o in outputs)
    generation_seconds = sum(o["response_latency_ms"] / 1000 for o in outputs)
    check("ledger_observed_tokens_and_time", sum(a["observed_generated_tokens"] for a in attempts) == generation_tokens
          and math.isclose(sum(a["observed_elapsed_seconds"] for a in attempts), generation_seconds, abs_tol=1e-6)
          and executed["generation_budget"]["observed"]["generated_tokens"] == generation_tokens
          and terminal["generatedTokens"] == generation_tokens
          and generation_tokens <= 32784
          and summary["known_generated_tokens"] == sum(o["generated_tokens"] for o in executed["outputs"]))
    check("global_deadline_and_time", math.isclose(grant["deadline_epoch_ms"] - grant["started_epoch_ms"], 5400000, abs_tol=0.01)
          and 0 <= executed["first_load_through_cleanup_seconds"] <= 5400
          and terminal["driver"]["deadline_epoch_ms"] == grant["deadline_epoch_ms"])

    def indices(kind, request_id=None):
        return [i for i, e in enumerate(events) if e["type"] == kind
                and (request_id is None or e["data"].get("request_id") == request_id)]

    gate_valid = True
    expected_intents, expected_acks = {"load-intent.json"}, {"load-ack.json"}
    for index in range(34):
        stem, rid = ("load", "load-1") if index == 0 else (f"generation-{index}", f"generation-{index}")
        expected_intents.add(stem + "-intent.json")
        expected_acks.add(stem + "-ack.json")
        intent, ack = read(execution_dir / (stem + "-intent.json")), read(execution_dir / (stem + "-ack.json"))
        expected = {"kind": "load" if index == 0 else "generation", "request_id": rid, "arm": "baseline",
                    "attempt_id": "baseline-1", "code_version_sha256": code_sha}
        gate_valid &= all(intent.get(k) == v and ack.get(k) == v for k, v in expected.items())
        gate_valid &= intent["configuration_sha256"] == config["configurationSha256"] and intent["artifact_sha256"] == config["convertedArtifactSha256"]
        gate_valid &= ack["approved"] is True and ack["deadline_epoch_ms"] == grant["deadline_epoch_ms"]
        stages = [indices("gate-request", rid), indices("driver-gate-ack-forwarded", rid), indices("gate-ack-received", rid),
                  indices("engine-call-intent") if index == 0 else indices("generation-call-intent", rid)]
        gate_valid &= all(len(stage) == 1 for stage in stages)
        if all(len(stage) == 1 for stage in stages):
            gate_valid &= stages[0][0] < stages[1][0] < stages[2][0] < stages[3][0]
        if index:
            row = next(a for a in attempts if a["request_id"] == rid)
            expected_cap = 16 if index == 1 else 1024
            gate_valid &= intent["warmup"] is (index == 1) and intent["max_tokens"] == expected_cap and intent["sequence_index"] == index - 1
            gate_valid &= row["max_generated_tokens"] == expected_cap and row["requested_sequences"] == 1
            gate_valid &= row["timeout_seconds"] == 120 and row["reserved_at"] <= row["dispatched_at"] <= row["finished_at"]
            gate_valid &= intent["created_at_epoch_ms"] <= row["reserved_at"] * 1000 + 1
            gate_valid &= row["dispatched_at"] * 1000 <= (execution_dir / (stem + "-ack.json")).stat().st_mtime * 1000 + 1
            lifecycle = [e["kind"] for e in ledger_events if e["request_id"] == rid]
            gate_valid &= lifecycle == ["reserved", "dispatched", "completed"]
            if index > 1:
                gate_valid &= intent["item_id"] == items[index - 2]["item_id"]
                previous = indices("warmup-sample") if index == 2 else [i for i, e in enumerate(events)
                    if e["type"] == "task-sample" and e["data"]["item_id"] == items[index - 3]["item_id"]]
                gate_valid &= len(previous) == 1 and len(stages[0]) == 1 and previous[0] < stages[0][0]
    check("durable_dispatch_gate_order", gate_valid
          and {p.name for p in execution_dir.glob("*-intent.json")} == expected_intents
          and {p.name for p in execution_dir.glob("*-ack.json")} == expected_acks)
    check("exact_engine_and_generation_call_count", len(indices("engine-call-intent")) == 1
          and len(indices("generation-call-intent")) == 33 and len(indices("warmup-sample")) == 1 and len(indices("task-sample")) == 32)
    loaded = [e["data"] for e in events if e["type"] == "task-artifact-loaded"]
    assets = {e["data"]["url"]: e["data"]["sha256"] for e in events if e["type"] == "asset-verified"}
    check("loaded_artifact_and_asset_proof", len(loaded) == 1
          and loaded[0]["sourceCheckpointSha256"] == config["sourceCheckpointSha256"]
          and loaded[0]["convertedArtifactSha256"] == config["convertedArtifactSha256"]
          and loaded[0]["modelLib"] == config["model"]["modelLib"] and loaded[0]["modelId"] == config["model"]["id"]
          and loaded[0]["allAssetsVerified"] is True and loaded[0]["executionBackend"] == "webgpu"
          and loaded[0]["serverInferenceFallback"] is False
          and assets == {a["url"]: a["sha256"] for a in config["assets"]})
    preflight = read(execution_dir / "fetch-preflight.json")
    check("actual_origin_fetch_preflight", preflight["status"] == "passed" and preflight["engine_loads"] == 0
          and preflight["generation_dispatches"] == 0 and preflight["code_version_sha256"] == code_sha)
    cleanup_keys = ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed", "local_harness_exit_confirmed",
                    "controller_exit_verified", "chrome_os_exit_verified")
    check("terminal_cleanup_evidence", all(executed["cleanup"].get(k) is True for k in cleanup_keys)
          and executed["cleanup"]["retained_owned_pids_alive"] == []
          and terminal["cleanup"]["unload_confirmed"] is True and executed["status"] == terminal["status"] == "completed")
    execution_children = sorted(p.name for p in (root / "execution").iterdir() if p.is_dir())
    evaluation_ids = {i["item_id"] for i in evaluation}
    dispatched_ids = {e["data"].get("item_id") for e in events if e["type"] == "generation-call-intent"}
    check("excluded_execution_absence_in_experiment_scope", execution_children == ["baseline-1"]
          and not evaluation_ids & dispatched_ids
          and not (root / "candidate/execution").exists()
          and executed["candidate_generations"] == executed["heldout_generations"] == executed["frontier_requests"] == 0
          and not any(a["kind"] == "editor" for a in attempts))
    preservation = read(root / "preservation-before.json")
    matched, changed, unavailable = 0, 0, 0
    for name, pin in preservation["files"].items():
        path = repo / name
        if not path.is_file() or (getattr(path.stat(), "st_flags", 0) & 0x40000000):
            unavailable += 1
        elif path.stat().st_size == pin["bytes"] and file_digest(path) == pin["sha256"]:
            matched += 1
        else:
            changed += 1
    check("prior_snapshot_preserved", len(preservation["files"]) == matched == 1387 and changed == unavailable == 0
          and preservation["historical_capacity_transferred"] is False)
    bundle_manifest = root / "handoff/bundle-manifest.json"
    bundle_present = bundle_manifest.is_file()
    if bundle_present:
        binding = read(bundle_manifest)
        request = read(root / "handoff/request.json")
        handoff_ready = read(root / "handoff/ready.json")
        appendix = read(root / "handoff/resource-appendix.json")
        core_packet = (root / "handoff/editor-packet.txt").read_text()
        bundle_text = (root / "handoff/editor-packet-bundle.txt").read_text()
        check("handoff_bundle_file_bindings", binding["kind"] == "ifeval_private_handoff_bundle"
              and binding["original_request_binding_unchanged"] is True and binding["score_changes"] == 0
              and all(file_digest(root / name) == pin for name, pin in binding["files"].items())
              and binding["original_request_sha256"] == digest(request) == handoff_ready["request_sha256"]
              and request["frozen_study_sha256"] == manifest_sha
              and handoff_ready["packet_sha256"] == file_digest(root / "handoff/editor-packet.txt")
              and bundle_text.startswith(core_packet))
        check("handoff_development_evidence", len(request["development"]) == 32
              and all(f["item_id"] == i["item_id"] and f["prompt"] == i["task_text"]
                      and f["raw_response"] == r["raw_output"] and f["token_counts"] == r["token_counts"]
                      and f["strict_prompt_compliant"] == g["correct"]
                      for f, i, r, g in zip(request["development"], items, records, grades)))
        points = [e["data"] for e in events if e["type"] == "point"]
        peaks = {}
        for key in ("gpuBufferBytes", "wasmBytes", "heapBytes", "eventLoopDelayMs"):
            values = [point[key] for point in points if type(point.get(key)) in (int, float)]
            peaks[key] = max(values) if values else None
        check("resource_appendix_matches_immutable_evidence", appendix["joined_observations"] == 32
              and appendix["terminal_result_sha256"] == file_digest(execution_dir / "result.json")
              and appendix["events_sha256"] == file_digest(execution_dir / "events.jsonl")
              and appendix["resources"]["worker_sampled_maxima"] == peaks
              and appendix["resources"]["worker_sample_count"] == len(points)
              and appendix["cleanup"] == executed["cleanup"])
        check("handoff_excludes_evaluation_prompts", not any(i["task_text"] in bundle_text for i in evaluation)
              and not evaluation_ids & {f["item_id"] for f in request["development"]})
        check("editor_request_and_observation_separated", request["editor"]["requested_display_name"] == "Astra"
              and request["editor"]["requested_mode"] == "Ultra"
              and all(request["editor"][k] is None for k in ("observed_model", "observed_effort", "immutable_model_version", "frontier_usage", "billing")))
    measurements.update(scheduled_development_prompts=32, scheduled_development_instructions=instruction_total,
        heldout_subset_prompts=len(evaluation), heldout_generations=0,
        engine_loads=executed["engine_load_attempts"], generation_dispatches=len(attempts),
        observed_generated_tokens=generation_tokens, observed_generation_seconds=generation_seconds,
        first_load_through_cleanup_seconds=executed["first_load_through_cleanup_seconds"], metrics=expected_metrics,
        unresolved_grader_errors=unresolved, prior_preservation_matched=matched,
        prior_preservation_unavailable=unavailable, prior_preservation_changed=changed,
        historical_unavailable_files=len(preservation["unavailable_cloud_placeholders"]), handoff_bundle_present=bundle_present)
    return {"schema_version": 1, "kind": "independent_ifeval_baseline_verification", "author": "Arhaan2",
            "passed": all(checks.values()), "checks": checks, "measurements": measurements,
            "identities": {"manifest_sha256": manifest_sha, "data_manifest_sha256": data["sha256"],
                           "verifier_sha256": file_digest(__file__), "raw_records_sha256": file_digest(root / "baseline/records.json"),
                           "grades_sha256": file_digest(root / "baseline/grades.json")},
            "scope": "Recorded evidence and explicit experiment execution directories; no claim about unrelated external sessions or historical unavailable files",
            "target_forward_passes": 0, "engine_loads_by_verifier": 0, "input_mutations": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out")
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(args.root) / "verification.json"
    if out.exists():
        raise SystemExit("Verification output exists; preserve it and use a new output.")
    try:
        result = verify(args.root, args.repo)
    except Exception as error:
        result = {"schema_version": 1, "kind": "independent_ifeval_baseline_verification", "passed": False,
                  "error_category": type(error).__name__, "complete_verification": False,
                  "target_forward_passes": 0, "engine_loads_by_verifier": 0, "input_mutations": 0}
    with out.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
