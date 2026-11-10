#!/usr/bin/env python3
"""Read-only verification of a provenance-bound IFEval candidate continuation.

Uses only stdlib evidence joins, hashes, byte checks, and read-only SQLite.
Never executes a model, converter, or grader. Writes one new sanitized report.
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_map(folder):
    return {p.name: {"sha256": file_digest(p), "size_bytes": p.stat().st_size}
            for p in sorted(Path(folder).iterdir()) if p.name != "artifact-manifest.json"}


def read_ledger(path):
    connection = sqlite3.connect("file:" + quote(str(Path(path).resolve())) + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        meta = dict(connection.execute("SELECT * FROM metadata WHERE singleton=1").fetchone())
        attempts = [dict(row) for row in connection.execute("SELECT * FROM attempts ORDER BY reserved_at,request_id")]
        events = [dict(row) for row in connection.execute("SELECT * FROM events ORDER BY event_index")]
        return integrity, meta, attempts, events
    finally:
        connection.close()


def preservation(snapshot, repo, exceptions=()):
    counts = Counter(matched=0, changed=0, unavailable=0, authorized_source_changes=0)
    for name, pin in snapshot["files"].items():
        path = repo / name
        if not path.is_file() or getattr(path.stat(), "st_flags", 0) & 0x40000000:
            counts["unavailable"] += 1
            continue
        expected = pin["sha256"] if isinstance(pin, dict) else pin
        same_size = not isinstance(pin, dict) or path.stat().st_size == pin["bytes"]
        if same_size and file_digest(path) == expected:
            counts["matched"] += 1
        else:
            counts["authorized_source_changes" if name in exceptions else "changed"] += 1
    return dict(counts)


def grade_evidence(folder, arm, manifest, items, check):
    evidence = read(folder / arm / "graded.json")
    records, grades = evidence["records"], evidence["grades"]
    check(arm + "_raw_grade_schedule", len(records) == len(grades) == len(items) == 32
          and records == read(folder / arm / "records.json") and grades == read(folder / arm / "grades.json")
          and [r["item_id"] for r in records] == [g["item_id"] for g in grades] == [i["item_id"] for i in items])
    check(arm + "_grade_bindings", evidence["manifest_sha256"] == digest(manifest)
          and all(g["run_record_sha256"] == digest(r) and g["item_sha256"] == digest(item)
                  and g["policy_sha256"] == digest(manifest["policy"]) and g["grader"] == manifest["grader"]
                  and r["configuration_sha256"] == manifest["configuration_sha256"]
                  and r["checkpoint"] == manifest["checkpoint"]
                  and r["seed"] == g["seed"] == 42 and r["sample_index"] == g["sample_index"] == 0
                  and r["split"] == "development" and r["benchmark"] == g["benchmark"] == "ifeval"
                  for item, r, g in zip(items, records, grades)))
    counts, failures = Counter(), Counter()
    valid = True
    instructions = sum(len(i["target"]["instruction_id_list"]) for i in items)
    for item, grade in zip(items, grades):
        if grade["status"] != "graded":
            valid = False
            continue
        m = grade["metrics"]
        n = len(item["target"]["instruction_id_list"])
        for mode in ("strict", "loose"):
            flags = m["instruction_" + mode]
            valid &= (len(flags) == n and all(type(flag) is bool for flag in flags)
                      and m[f"prompt_level_{mode}_accuracy"] == all(flags)
                      and m[f"instruction_{mode}_correct"] == sum(flags)
                      and m[f"instruction_level_{mode}_accuracy"] == sum(flags) / n)
            counts["prompt_" + mode] += int(all(flags))
            counts["instruction_" + mode] += sum(flags)
        valid &= grade["correct"] == all(m["instruction_strict"]) and m["instruction_count"] == n
        failures.update(name for name, passed in zip(item["target"]["instruction_id_list"], m["instruction_strict"]) if not passed)
    expected = {f"{kind}_level_{mode}_accuracy": {
        "passed": counts[kind + "_" + mode], "scheduled": len(items) if kind == "prompt" else instructions,
        "value": counts[kind + "_" + mode] / (len(items) if kind == "prompt" else instructions)}
        for kind in ("prompt", "instruction") for mode in ("strict", "loose")}
    summary = evidence["summary"]
    check(arm + "_official_aggregation", valid and summary["metrics"] == expected
          and summary["scheduled_prompts"] == len(items) and summary["scheduled_instructions"] == instructions
          and summary["strict_failures_by_instruction"] == dict(failures)
          and summary["execution_status_counts"] == dict(Counter(r["status"] for r in records))
          and summary["unresolved_grader_errors"] == 0 and summary["semantic_grading_complete"] is True
          and summary["technically_valid"] is True)
    check(arm + "_scored_token_time_totals", summary["known_generated_tokens"] == sum(r["token_counts"]["output"] for r in records)
          and summary["known_input_tokens"] == sum(r["token_counts"]["input"] for r in records)
          and math.isclose(summary["generation_seconds"], sum(r["duration_seconds"] for r in records), abs_tol=1e-6))
    return evidence, expected


def verify(root, repo, user_request):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    recovery, target = root / "recovery", root / "candidate-comparison"
    checks, measurements = {}, {}

    def check(name, value):
        checks[name] = bool(value)

    base, candidate = read(root / "manifest.json"), read(target / "manifest.json")
    base_config, config = read(root / "baseline-config.json"), read(target / "baseline-config.json")
    plan = read(root / "candidate-comparison-plan.json")
    paired, bound = read(target / "paired-analysis-manifest.json"), read(target / "comparison-ready.json")
    items, heldout = read(root / "data/development-items.json"), read(root / "data/evaluation-items.json")
    data = read(root / "data/manifest.json")
    check("original_and_candidate_manifest_bindings", read(root / "ready.json") == {"manifest_sha256": digest(base)}
          and read(target / "ready.json") == {"manifest_sha256": digest(candidate)}
          and candidate["baseline_manifest_sha256"] == digest(base)
          and candidate["baseline_result_sha256"] == file_digest(root / "baseline/graded.json")
          and all(file_digest(name) == pin for name, pin in base["files"].items())
          and all(file_digest(name) == pin for name, pin in candidate["files"].items()))
    check("original_authorization_preserved_and_recovery_separate",
          file_digest(root / "authorization-source.txt") == base["authorization_source_sha256"]
          and file_digest(target / "authorization-source.txt") == candidate["authorization_source_sha256"]
          and (recovery / "authorization.txt").read_bytes() == (recovery / "request.txt").read_bytes()
          == Path(user_request).read_bytes() == (target / "authorization-source.txt").read_bytes()
          and (recovery / "authorization.txt").read_bytes() != (root / "candidate-authorization.txt").read_bytes())
    check("frozen_pair_plan_and_binding", read(root / "candidate-comparison-plan-ready.json") == {"plan_sha256": digest(plan)}
          and plan["baseline_manifest_sha256"] == digest(base)
          and plan["baseline_config_file_sha256"] == file_digest(root / "baseline-config.json")
          and bound["plan_sha256"] == digest(plan) and bound["paired_manifest_sha256"] == digest(paired)
          and bound["candidate_manifest_sha256"] == digest(candidate)
          and bound["baseline_graded_file_sha256"] == file_digest(root / "baseline/graded.json")
          and paired["manifest_sha256"] == digest({k: v for k, v in paired.items() if k != "manifest_sha256"})
          and paired["split"] == "development" and paired["metric"] == "prompt_level_strict_accuracy"
          and paired["policy"] == base["policy"] == candidate["policy"]
          and paired["grader"] == base["grader"] == candidate["grader"])
    split_manifest = read(target / "paired-split-manifest.json")
    item_refs = sorted([{"benchmark": item["benchmark"], "item_id": item["item_id"],
        "dataset_revision": item["dataset_revision"], "item_sha256": digest(item),
        "instruction_count": len(item["target"]["instruction_id_list"])} for item in items], key=lambda row: row["item_id"])
    check("paired_denominator_contains_exact_frozen_items", paired["items"] == item_refs
          and paired["schedule"] == [{"benchmark": "ifeval", "item_id": row["item_id"], "seed": 42, "sample_index": 0}
                                     for row in item_refs]
          and paired["split_manifest_sha256"] == digest(split_manifest)
          and paired["source_manifest_sha256"] == digest(data)
          and read(target / "paired-source-binding.json")["source_manifest_sha256"] == digest(data)
          and all(paired["arms"][arm] == {"run_id": cfg["runId"], "profile": "browser_evaluation",
              "configuration_sha256": cfg["configurationSha256"], "checkpoint": manifest["checkpoint"]}
                  for arm, cfg, manifest in (("baseline", base_config, base), ("candidate", config, candidate))))
    expected_settings = {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 1024, "contextWindow": 4096,
                         "warmups": 1, "warmupTokens": 16, "requestLimitMs": 120000, "thinking": False}
    expected_limits = {"engine_load_attempts": 1, "warmup_dispatches": 1, "scored_dispatches": 32,
                       "generation_dispatches": 33, "generated_tokens": 32784, "warmup_tokens": 16,
                       "tokens_per_scored_request": 1024, "request_seconds": 120, "wall_seconds": 5400,
                       "automatic_retries": 0, "candidate_generations": 33, "heldout_generations": 0, "frontier_requests": 0}
    check("candidate_only_authorized_bounds", candidate["arm"] == "candidate" and candidate["limits"] == expected_limits
          and candidate["candidate_execution_authorized"] is True
          and candidate["heldout_execution_authorized"] is False and candidate["frontier_execution_authorized"] is False
          and candidate["strict_ready"] is False and candidate["editor_isolation"] == "unverified")
    check("identical_frozen_profile_and_schedule", all(config[k] == base_config[k] for k in ("settings", "items", "warmupPrompt", "totalLimitMs"))
          and config["settings"] == base["settings"] == candidate["settings"] == expected_settings
          and config["totalLimitMs"] == 5400000 and config["warmupPrompt"] == plan["warmup_prompt"]
          and config["runId"] != base_config["runId"] and config["arm"] == "candidate"
          and plan["task_items_sha256"] == digest(config["items"])
          and plan["task_order"] == [i["item_id"] for i in config["items"]] == [i["item_id"] for i in items]
          and all(a["prompt"] == b["task_text"] for a, b in zip(config["items"], items))
          and len(items) == 32 and len(heldout) == 128
          and digest(items) == data["development"]["items_sha256"]
          and digest(heldout) == data["evaluation"]["items_sha256"])
    check("official_grader_assets_unchanged", all(file_digest(Path(base["official_ifeval_root"]) / name) == pin
          for name, pin in base["official_assets"]["files"].items())
          and file_digest(Path(base["official_ifeval_root"]) / "punkt_tab.zip") == base["official_assets"]["nltk_archive_sha256"])
    request, imported = read(root / "handoff/request.json"), read(root / "handoff/import-result.json")
    good, import_meta, import_attempts, import_events = read_ledger(root / "handoff/import-ledger.sqlite3")
    frozen_import = json.loads(import_meta["frozen_json"])
    check("import_remains_single_consumed_submission", good and len(import_attempts) == 1 and imported["accepted"] is True
          and frozen_import["binding"]["request_sha256"] == digest(request)
          and frozen_import["limits"]["editor_requests"] == frozen_import["limits"]["requested_sequences"] == 1
          and frozen_import["limits"]["elapsed_seconds"] == 900
          and all(a["request_id"] == request["request_id"] and a["status"] == "completed" and a["retry_of"] is None
                  and a["observed_generated_tokens"] == 0 for a in import_attempts)
          and [e["kind"] for e in import_events if e["request_id"] == request["request_id"]]
              == ["reserved", "dispatched", "completed"])
    source = root / "candidate/source"
    source_audit, source_proposal = read(source / "audit.json"), read(source / "proposal.json")
    artifact = read(root / "candidate/deployed/candidate-artifact.json")
    binding = read(root / "candidate/converted-source-binding.json")
    deployment_audit = read(root / "candidate/deployed/deployment-difference.json")
    conversion = read(repo / "artifacts/browser-integration-v1/conversion/artifacts.json")
    check("saved_source_and_original_proposal_binding", read(root / "response.json") == read(root / "handoff/accepted-response.json")
          and source_proposal == read(root / "response.json")["proposal"]
          and source_audit["proposal_sha256"] == digest(source_proposal)
          and source_audit["result_sha256"] == imported["audit"]["result_sha256"] == candidate["checkpoint"]["sha256"]
          == artifact["source_checkpoint_sha256"] == binding["source_checkpoint_sha256"]
          and source_audit["base_sha256"] == base["checkpoint"]["sha256"] == source_proposal["base_sha256"]
          and all(source_audit[k] == imported["audit"][k] for k in ("targeted_coordinates", "changed_stored_entries", "changed_numerical_entries"))
          and file_map(source / "model") == source_audit["model_inventory"]["files"]
          and digest(source_audit["model_inventory"]["files"]) == source_audit["result_sha256"])
    check("converted_and_deployed_file_hashes", binding["status"] == "converted_source_bound"
          and binding["converted_files"] == file_map(root / "candidate/converted")
          and binding["converted_files_sha256"] == digest(binding["converted_files"])
          and binding["baseline_manifest_sha256"] == file_digest(repo / "artifacts/browser-integration-v1/conversion/artifacts.json")
          and artifact["files"] == file_map(artifact["path"]) and artifact["artifact_sha256"] == digest(artifact["files"])
          and read(Path(artifact["path"]) / "artifact-manifest.json") == artifact
          and artifact["conversion_binding_sha256"] == digest(binding)
          and all(artifact["files"][name] == pin for name, pin in binding["converted_files"].items())
          and bound["candidate_artifact_file_sha256"] == file_digest(root / "candidate/deployed/candidate-artifact.json"))
    check("source_and_deployment_audit_binding", artifact["source_identity"]["source_audit_sha256"] == digest(source_audit)
          and read(root / "candidate/deployed/source-audit.json") == source_audit
          and artifact["effective_deployment_audit_sha256"] == digest(deployment_audit)
          and artifact["baseline_artifact_sha256"] == base["artifact_sha256"]
          and deployment_audit["parameter_contract_verified"] is True
          and deployment_audit["quantization"] == "q4f16_1" and deployment_audit["group_size"] == 32
          and set(deployment_audit["per_source_tensor"]) == {op["tensor"] for op in source_proposal["operations"]}
          and all(row["decoded_changed_outside_source_target_groups"] == 0 for row in deployment_audit["per_source_tensor"].values()))
    tensor_ids = deployment_audit["tensor_storage_identities"]
    changed_tensor_names = sorted(name for name, pin in tensor_ids["baseline"].items() if pin != tensor_ids["candidate"][name])
    allowed_names = {op["tensor"].replace("model.language_model.", "model.", 1).removesuffix(".weight") + suffix
                     for op in source_proposal["operations"] for suffix in (".q_weight", ".q_scale")}
    check("quantization_audit_internal_consistency", set(tensor_ids["baseline"]) == set(tensor_ids["candidate"])
          and deployment_audit["parameter_count"] == len(tensor_ids["baseline"])
          and sorted(deployment_audit["changed_tensor_names"]) == changed_tensor_names
          and set(changed_tensor_names) <= allowed_names
          and deployment_audit["unchanged_tensor_count"] == len(tensor_ids["baseline"]) - len(changed_tensor_names)
          and all(deployment_audit["per_source_tensor"][op["tensor"]]["target_columns"] == sorted(op["columns"])
                  and deployment_audit["per_source_tensor"][op["tensor"]]["source_targeted_quantization_groups"]
                      == sorted({column // 32 for column in op["columns"]}) for op in source_proposal["operations"])
          and deployment_audit["conversion_erases_edit"] is
              (not any(row["decoded_values_changed"] for row in deployment_audit["per_source_tensor"].values())))
    base_files = conversion["arms"]["baseline"]["files"]
    check("shared_runtime_assets_identical", all(artifact["files"][name]["sha256"] == pin == base_files[name]["sha256"]
          for name, pin in artifact["shared_runtime_assets"].items())
          and conversion["webllm_version"] == "0.2.86" and conversion["quantization"] == "q4f16_1")
    # Recovery provenance is validated in its separate, immutable source record.
    repair = read(recovery / "repair-record.json")
    recovery_binding = binding["launcher_recovery"]
    check("repaired_launcher_and_authorization_bound", recovery_binding["record_sha256"] == file_digest(recovery / "repair-record.json")
          and all(recovery_binding[key] == repair[key] for key in ("original_source_sha256", "repaired_launcher_sha256",
                  "authorization_file_sha256", "failure_file_sha256"))
          and repair["original_source_sha256"] == file_digest(repo / "weight_space_lab/browser_candidate.py")
          and repair["repaired_launcher_sha256"] == file_digest(repo / "weight_space_lab/browser_candidate_recovery.py")
          and repair["authorization_file_sha256"] == file_digest(recovery / "authorization.txt")
          and repair["failure_file_sha256"] == file_digest(root / "candidate-stage-failure.json"))
    environment = read(recovery / "environment-verification.json")
    preflight = read(recovery / "preflight.json")
    requirements = repo / "artifacts/browser-integration-v1/conversion/converter-requirements.txt"
    dependency_pins = dict(line.split("==", 1) for line in requirements.read_text().splitlines() if line and not line.startswith("#"))
    observed = environment["observed"]
    check("production_environment_and_all_dependency_pins", environment["passed"] is True
          and environment["through_production_helper"] == "weight_space_lab.browser_candidate_recovery.run_converter_python"
          and environment["helper_source_sha256"] == repair["repaired_launcher_sha256"]
          and environment["invoked_interpreter"] == observed["sys_executable"] == binding["command"][0]
          and observed["sys_prefix"] != observed["sys_base_prefix"]
          and Path(observed["sys_prefix"]) == Path(binding["command"][0]).parent.parent
          and environment["expected_environment_active"] is True and environment["changed_subprocess_cwd_verified"] is True
          and observed["packages"] == dependency_pins
          and environment["dependency_pin_count"] == len(dependency_pins)
          and environment["lockfile_sha256"] == file_digest(requirements)
          and environment["expected_converter_packages"] == conversion["converter_packages"]
          and all(Path(location).is_relative_to(observed["sys_prefix"]) for location in observed["module_locations"].values())
          and environment["target_generations"] == environment["engine_loads"] == environment["installations"] == environment["dependency_upgrades"] == 0)
    check("repair_revision_and_focused_checks_bound", repair["source_revision"]["historical_source_unchanged"] is True
          and repair["source_revision"]["module_sha256"] == repair["repaired_launcher_sha256"]
          and repair["source_revision"]["regression_test_source_sha256"] == file_digest(repo / "tests/test_browser_candidate.py")
          and repair["source_revision"]["repair_diff_sha256"] == file_digest(recovery / "launcher-repair.diff")
          and repair["verification"]["focused_tests_log_sha256"] == file_digest(recovery / "launcher-focused-tests-final.txt")
          and repair["verification"]["environment_verification_sha256"] == file_digest(recovery / "environment-verification.json")
          and repair["verification"]["preflight_sha256"] == file_digest(recovery / "preflight.json")
          and repair["verification"]["focused_tests_passed"] > 0
          and repair["conversion_timeout_seconds"] == 1800 and repair["package_check_timeout_seconds"] == 30)
    check("recovery_started_from_verified_unused_candidate_capacity", preflight["source_reverified"] is True
          and preflight["source_checkpoint_sha256"] == source_audit["result_sha256"]
          and preflight["source_audit_sha256"] == digest(source_audit) and preflight["recorded_audit_identical"] is True
          and preflight["baseline_manifest_sha256"] == digest(base) and preflight["baseline_frozen_files_verified"] is True
          and preflight["prior_imports"] == 1 and preflight["prior_import_completed"] is True
          and preflight["conversion_subprocesses_before_recovery"] == preflight["candidate_loads_before_recovery"]
              == preflight["candidate_dispatches_before_recovery"] == preflight["target_generations"] == 0
          and preflight["baseline_dispatches"] == 33 and preflight["baseline_ledger_terminal"] is True
          and preflight["frozen_handoff_command_file_sha256"] == file_digest(root / "handoff/next-commands.sh"))
    intent, outcome = read(recovery / "intent.json"), read(recovery / "result.json")
    check("single_manual_continuation_accounting", intent["manual_conversion_continuations"] == outcome["manual_conversion_continuations"] == 1
          and intent["automatic_retries"] == outcome["automatic_retries"] == intent["imports"] == outcome["imports"] == 0
          and outcome["status"] == "completed" and outcome["returncode"] == 0
          and outcome["intent_sha256"] == file_digest(recovery / "intent.json")
          and outcome["stdout_sha256"] == file_digest(recovery / "continuation.stdout.txt")
          and outcome["stderr_sha256"] == file_digest(recovery / "continuation.stderr.txt")
          and intent["command_file_sha256"] == file_digest(recovery / "continue.sh")
          and intent["runner_source_sha256"] == file_digest(recovery / "run-continuation.py")
          and intent["authorization_file_sha256"] == repair["authorization_file_sha256"]
          and intent["repair_record_sha256"] == file_digest(recovery / "repair-record.json")
          and intent["prior_failure_sha256"] == repair["failure_file_sha256"]
          and intent["import_ledger_before_sha256"] == outcome["import_ledger_after_sha256"] == file_digest(root / "handoff/import-ledger.sqlite3")
          and intent["baseline_ledger_before_sha256"] == outcome["baseline_ledger_after_sha256"] == file_digest(root / "generation-budget.sqlite3")
          and all(outcome[key] is True for key in ("converted", "packaged", "bound", "browser_result_present", "comparison_present")))
    check("same_configured_converter_for_probe_and_conversion", binding["converter_packages"] == conversion["converter_packages"]
          and Path(binding["command"][0]).is_absolute()
          and Path(binding["command"][0]) == (repo / "artifacts/browser-integration-v1/conversion/venv/bin/python")
          and Path(binding["command"][0]).resolve() != Path(binding["command"][0])
          and file_digest(binding["command"][1]) == conversion["converter_adapter_sha256"])
    base_evidence, base_metrics = grade_evidence(root, "baseline", base, items, check)
    evidence, metrics = grade_evidence(target, "candidate", candidate, items, check)
    execution_dir = target / "execution/candidate-1"
    executed, terminal = read(execution_dir / "supervisor-result.json"), read(execution_dir / "result.json")
    events = [json.loads(line) for line in (execution_dir / "events.jsonl").read_text().splitlines()]
    grant = read(target / "load-grant.json")
    code_sha = digest(candidate["files"])
    outputs = executed["warmup_outputs"] + executed["outputs"]
    core_fields = ("item_id", "raw_output", "finish_reason", "generated_tokens", "input_tokens", "response_latency_ms",
                   "loading_time_ms", "execution_error", "cancelled", "truncated", "valid_completion",
                   "webgpu_submitted_batches", "local_webgpu_inference_observed", "request_id", "warmup",
                   "assistant_prefix_tokens", "request_max_tokens")
    core = lambda rows: [{key: row.get(key) for key in core_fields} for row in rows]
    check("candidate_event_supervisor_terminal_parity", core(outputs) == core(terminal["warmupOutputs"] + terminal["outputs"])
          == core([e["data"] for e in events if e["type"] in ("warmup-sample", "task-sample")]))
    check("runtime_artifact_identity", config["configurationSha256"] == candidate["configuration_sha256"] == terminal["configuration_sha256"]
          and config["sourceCheckpointSha256"] == candidate["checkpoint"]["sha256"] == terminal["source_checkpoint_sha256"]
          and config["convertedArtifactSha256"] == candidate["artifact_sha256"] == terminal["converted_artifact_sha256"]
          == artifact["artifact_sha256"])
    effective = [e["data"] for e in events if e["type"] == "task-effective-settings"]
    expected_generation = {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 1024,
                           "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
                           "n": 1, "ignore_eos": False, "extra_body": {"enable_thinking": False}}
    token_audits = [e["data"] for e in events if e["type"] == "task-input-token-audit"]
    check("actual_effective_settings_and_input_tokens", len(effective) == 1 and effective[0]["generation"] == expected_generation
          and effective[0]["contextWindow"] == 4096 and effective[0]["maxHistorySize"] == 1
          and len(token_audits) == 33
          and all(a["effective_context_window"] == 4096 for a in token_audits)
          and {a["item_id"]: a["rendered_input_tokens"] for a in token_audits if not a["warmup"]}
              == {i["item_id"]: i["metadata"]["rendered_input_tokens"] for i in items})
    check("complete_generation_schedule_and_caps", len(outputs) == 33 and len(executed["warmup_outputs"]) == 1
          and [o["item_id"] for o in executed["outputs"]] == [i["item_id"] for i in items]
          and executed["engine_load_attempts"] == executed["warmup_dispatches"] == 1
          and executed["scored_dispatches"] == 32 and executed["generation_dispatches"] == 33
          and all(0 <= o["generated_tokens"] <= (16 if index == 0 else 1024)
                  and 0 <= o["response_latency_ms"] <= 120000 for index, o in enumerate(outputs)))
    check("raw_records_match_observed_generation", all(r["raw_output"] == o["raw_output"]
          and r["token_counts"]["output"] == o["generated_tokens"] and r["token_counts"]["input"] == o["input_tokens"]
          and r["duration_seconds"] == o["response_latency_ms"] / 1000
          and len(r["attempts"]) == 1 and r["retry_policy"] == {"mode": "none", "max_attempts": 1}
          for r, o in zip(evidence["records"], executed["outputs"])))
    check("local_gpu_execution_observed", all(o["local_webgpu_inference_observed"] is True
          and o["webgpu_submitted_batches"] > 0 for o in outputs)
          and not any(e["type"] == "event" and e["data"].get("type") == "device-lost" for e in events))
    good, meta, attempts, ledger_events = read_ledger(target / "generation-budget.sqlite3")
    frozen = json.loads(meta["frozen_json"])
    check("fresh_candidate_ledger_identity_and_terminal_state", good
          and frozen["run_id"] == config["runId"] and frozen["binding"]["manifest_sha256"] == digest(candidate)
          and frozen["limits"] == {"editor_requests": 0, "model_probes": 33, "requested_sequences": 33,
              "generated_tokens": 32784, "elapsed_seconds": 5400, "billed_units": 0}
          and meta["cancelled"] == 1 and len(attempts) == 33
          and all(a["status"] == "completed" and a["retry_of"] is None
                  and a["conservative"] == a["limit_violation"] == 0 for a in attempts))
    all_tokens = sum(o["generated_tokens"] for o in outputs)
    all_seconds = sum(o["response_latency_ms"] / 1000 for o in outputs)
    check("ledger_observed_usage_matches_outputs", sum(a["observed_generated_tokens"] for a in attempts) == all_tokens
          and math.isclose(sum(a["observed_elapsed_seconds"] for a in attempts), all_seconds, abs_tol=1e-6)
          and executed["generation_budget"]["observed"]["generated_tokens"] == terminal["generatedTokens"] == all_tokens
          and all_tokens <= 32784 and executed["candidate_generations"] == 33
          and read(target / "accounting-after.json") == executed)
    check("browser_deadline_and_cleanup_duration", grant["manifest_sha256"] == digest(candidate)
          and math.isclose(grant["deadline_epoch_ms"] - grant["started_epoch_ms"], 5400000, abs_tol=0.01)
          and 0 <= executed["first_load_through_cleanup_seconds"] <= 5400
          and terminal["driver"]["deadline_epoch_ms"] == grant["deadline_epoch_ms"])

    def indices(kind, rid=None):
        return [index for index, event in enumerate(events) if event["type"] == kind
                and (rid is None or event["data"].get("request_id") == rid)]

    gates = True
    expected_intents, expected_acks = set(), set()
    for index in range(34):
        stem, rid = ("load", "load-1") if index == 0 else (f"generation-{index}", f"generation-{index}")
        expected_intents.add(stem + "-intent.json")
        expected_acks.add(stem + "-ack.json")
        intent, ack = read(execution_dir / (stem + "-intent.json")), read(execution_dir / (stem + "-ack.json"))
        expected = {"kind": "load" if index == 0 else "generation", "request_id": rid, "arm": "candidate",
                    "attempt_id": "candidate-1", "code_version_sha256": code_sha}
        gates &= all(intent.get(key) == value and ack.get(key) == value for key, value in expected.items())
        gates &= intent["configuration_sha256"] == config["configurationSha256"] and intent["artifact_sha256"] == config["convertedArtifactSha256"]
        gates &= ack["approved"] is True and ack["deadline_epoch_ms"] == grant["deadline_epoch_ms"]
        stages = [indices("gate-request", rid), indices("driver-gate-ack-forwarded", rid), indices("gate-ack-received", rid),
                  indices("engine-call-intent") if index == 0 else indices("generation-call-intent", rid)]
        gates &= all(len(stage) == 1 for stage in stages)
        if all(len(stage) == 1 for stage in stages):
            gates &= stages[0][0] < stages[1][0] < stages[2][0] < stages[3][0]
        if index:
            row = next(a for a in attempts if a["request_id"] == rid)
            cap = 16 if index == 1 else 1024
            gates &= intent["warmup"] is (index == 1) and intent["max_tokens"] == cap and intent["sequence_index"] == index - 1
            gates &= row["max_generated_tokens"] == cap and row["requested_sequences"] == 1 and row["timeout_seconds"] == 120
            gates &= row["reserved_at"] <= row["dispatched_at"] <= row["finished_at"]
            gates &= intent["created_at_epoch_ms"] <= row["reserved_at"] * 1000 + 1
            gates &= row["dispatched_at"] * 1000 <= (execution_dir / (stem + "-ack.json")).stat().st_mtime * 1000 + 1
            gates &= [e["kind"] for e in ledger_events if e["request_id"] == rid] == ["reserved", "dispatched", "completed"]
            if index > 1:
                gates &= intent["item_id"] == items[index - 2]["item_id"]
                previous = indices("warmup-sample") if index == 2 else [i for i, e in enumerate(events)
                    if e["type"] == "task-sample" and e["data"]["item_id"] == items[index - 3]["item_id"]]
                gates &= len(previous) == len(stages[0]) == 1 and previous[0] < stages[0][0]
    check("exact_durable_gate_schedule_and_order", gates
          and {p.name for p in execution_dir.glob("*-intent.json")} == expected_intents
          and {p.name for p in execution_dir.glob("*-ack.json")} == expected_acks
          and len(indices("engine-call-intent")) == 1 and len(indices("generation-call-intent")) == 33
          and len(indices("warmup-sample")) == 1 and len(indices("task-sample")) == 32)
    loaded = [e["data"] for e in events if e["type"] == "task-artifact-loaded"]
    assets = {e["data"]["url"]: e["data"]["sha256"] for e in events if e["type"] == "asset-verified"}
    fetch = read(execution_dir / "fetch-preflight.json")
    check("loaded_and_fetched_artifact_proof", len(loaded) == 1
          and loaded[0]["sourceCheckpointSha256"] == config["sourceCheckpointSha256"]
          and loaded[0]["convertedArtifactSha256"] == config["convertedArtifactSha256"]
          and loaded[0]["modelLib"] == config["model"]["modelLib"] and loaded[0]["modelId"] == config["model"]["id"]
          and loaded[0]["allAssetsVerified"] is True and loaded[0]["executionBackend"] == "webgpu"
          and loaded[0]["serverInferenceFallback"] is False
          and assets == {a["url"]: a["sha256"] for a in config["assets"]}
          and fetch["status"] == "passed" and fetch["engine_loads"] == fetch["generation_dispatches"] == 0
          and fetch["code_version_sha256"] == code_sha)
    cleanup_keys = ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed", "local_harness_exit_confirmed",
                    "controller_exit_verified", "chrome_os_exit_verified")
    check("complete_execution_and_cleanup", executed["status"] == terminal["status"] == "completed"
          and executed["error"] is None and all(executed["cleanup"].get(key) is True for key in cleanup_keys)
          and executed["cleanup"]["retained_owned_pids_alive"] == []
          and terminal["cleanup"]["unload_confirmed"] is True)
    comparison = read(target / "comparison.json")
    flips = {}
    for mode in ("strict", "loose"):
        changes = [int(c["metrics"][f"prompt_level_{mode}_accuracy"]) - int(b["metrics"][f"prompt_level_{mode}_accuracy"])
                   for b, c in zip(base_evidence["grades"], evidence["grades"])]
        flips[mode] = {"improved": sum(d > 0 for d in changes), "regressed": sum(d < 0 for d in changes), "tied": sum(d == 0 for d in changes)}
    expected_per_item = sorted([{"item_id": item["item_id"], "scheduled_repeats": 1,
        "baseline_mean": float(b["correct"]), "candidate_mean": float(c["correct"]), "delta": float(c["correct"]) - float(b["correct"])}
        for item, b, c in zip(items, base_evidence["grades"], evidence["grades"])], key=lambda row: row["item_id"])
    projection_valid = True
    for arm, expected in (("baseline", base_metrics), ("candidate", metrics)):
        observed = comparison[arm + "_projection"]["official_metrics"]
        projection_valid &= observed == {name: {"numerator": v["passed"], "denominator": v["scheduled"], "value": v["value"]}
                                         for name, v in expected.items()}
    primary = "prompt_level_strict_accuracy"
    selected = "candidate" if metrics[primary]["value"] > base_metrics[primary]["value"] else "baseline"
    check("paired_results_and_frozen_selection", projection_valid and comparison["manifest_sha256"] == paired["manifest_sha256"]
          and comparison["semantic_grading_complete"] is True and comparison["comparison"]["per_item"] == expected_per_item
          and all(comparison["comparison"]["paired_flips"][key] == value for key, value in flips["strict"].items())
          and math.isclose(comparison["comparison"]["delta"], metrics[primary]["value"] - base_metrics[primary]["value"], abs_tol=1e-12)
          and comparison["selection"]["technical_and_grading_complete"] is True
          and comparison["selection"]["selected_arm"] == selected
          and comparison["selection"]["rule"] == plan["selection"] and comparison["selection"]["heldout_generations"] == 0
          and comparison["comparison"]["confidence_interval"]["resamples"] == plan["bootstrap"]["samples"]
          and comparison["comparison"]["confidence_interval"]["seed"] == plan["bootstrap"]["seed"])
    dispatched = {e["data"].get("item_id") for e in events if e["type"] == "generation-call-intent"}
    check("no_unscheduled_or_heldout_experiment_execution", sorted(p.name for p in (target / "execution").iterdir()) == ["candidate-1"]
          and sorted(p.name for p in (root / "execution").iterdir()) == ["baseline-1"]
          and not {i["item_id"] for i in heldout} & dispatched
          and executed["heldout_generations"] == executed["frontier_requests"] == 0
          and all(a["kind"] == "probe" for a in attempts))
    good, base_meta, base_attempts, _ = read_ledger(root / "generation-budget.sqlite3")
    check("baseline_ledger_terminal_without_rerun", good and base_meta["cancelled"] == 1 and len(base_attempts) == 33
          and all(a["status"] == "completed" and a["retry_of"] is None for a in base_attempts))
    snapshot = read(recovery / "preservation-before.json")
    historical = read(root / "preservation-before.json")
    current_counts = preservation(snapshot, repo, exceptions={"tests/test_browser_candidate.py"})
    historical_counts = preservation(historical, repo)
    check("recovery_preservation_except_authorized_regression_test", current_counts["changed"] == current_counts["unavailable"] == 0
          and current_counts["matched"] + current_counts["authorized_source_changes"] == len(snapshot["files"])
          and file_digest(recovery / "original-source/tests/test_browser_candidate.py") == snapshot["files"]["tests/test_browser_candidate.py"]
          and file_digest(recovery / "original-source/weight_space_lab/browser_candidate.py")
              == file_digest(repo / "weight_space_lab/browser_candidate.py"))
    check("historical_available_evidence_preserved", historical_counts["changed"] == historical_counts["unavailable"] == 0
          and historical_counts["matched"] == len(historical["files"]))
    points = [e["data"] for e in events if e["type"] == "point"]
    peaks = {key: max([p[key] for p in points if type(p.get(key)) in (int, float)], default=None)
             for key in ("gpuBufferBytes", "wasmBytes", "heapBytes", "eventLoopDelayMs")}
    measurements.update(baseline_metrics=base_metrics, candidate_metrics=metrics, paired_prompt_changes=flips,
        selected_arm=selected, candidate_engine_loads=executed["engine_load_attempts"], generation_dispatches=len(attempts),
        observed_generated_tokens=all_tokens, scored_generated_tokens=evidence["summary"]["known_generated_tokens"],
        scored_generation_seconds=evidence["summary"]["generation_seconds"],
        first_load_through_cleanup_seconds=executed["first_load_through_cleanup_seconds"],
        candidate_completion_counts=evidence["summary"]["execution_status_counts"], worker_sampled_maxima=peaks,
        worker_resource_scope="GPUBuffer descriptor accounting and WASM address space, not physical VRAM",
        recovery_preservation=current_counts, historical_preservation=historical_counts,
        historical_unavailable_files_not_verified=len(historical["unavailable_cloud_placeholders"]),
        manual_conversion_continuations=outcome["manual_conversion_continuations"], import_attempts=len(import_attempts),
        frozen_dependency_pins_verified=len(dependency_pins), decoded_quantization_recomputed_by_verifier=False,
        heldout_generations=0, experimental_frontier_requests=0)
    return {"schema_version": 1, "kind": "independent_ifeval_candidate_recovery_verification", "author": "Arhaan2",
        "passed": all(checks.values()), "checks": checks, "measurements": measurements,
        "identities": {"verifier_sha256": file_digest(__file__), "baseline_manifest_sha256": digest(base),
            "candidate_manifest_sha256": digest(candidate), "candidate_checkpoint_sha256": source_audit["result_sha256"],
            "candidate_artifact_sha256": artifact["artifact_sha256"], "repair_record_sha256": file_digest(recovery / "repair-record.json"),
            "comparison_file_sha256": file_digest(target / "comparison.json")},
        "scope": "Recorded recovery and matched development evidence; no external-session or unavailable historical-file claims. Official grades are hash-bound and independently aggregated, not rerun.",
        "target_forward_passes": 0, "engine_loads_by_verifier": 0, "grader_calls_by_verifier": 0, "input_mutations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/ifeval-browser-v1")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--user-request", required=True, help="Original recovery request file; location is never published")
    args = parser.parse_args()
    out = Path(args.repo) / "reports/ifeval-browser-v1-candidate-recovery/verification.json"
    if out.exists():
        raise SystemExit("Verification report exists; preserve it.")
    try:
        result = verify(args.root, args.repo, args.user_request)
    except Exception as error:
        result = {"schema_version": 1, "kind": "independent_ifeval_candidate_recovery_verification",
            "passed": False, "complete_verification": False, "error_category": type(error).__name__,
            "target_forward_passes": 0, "engine_loads_by_verifier": 0, "grader_calls_by_verifier": 0, "input_mutations": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
