#!/usr/bin/env python3
"""Export only sanitized aggregate evidence from the explicit V2 continuation.

Reads saved files; never loads the target model or dispatches an editor/browser.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def environment(execution: Path):
    result = execution / "result.json"
    if not result.exists():
        return None
    value = read(result)
    version = execution / "chrome-profile/Last Version"
    return {"chrome_version": version.read_text().strip() if version.exists() else None,
            "user_agent": value.get("user_agent"), "platform": value.get("platform"),
            "hardware_concurrency": value.get("hardware_concurrency")}


def report(root: Path):
    state = read(root / "state.json")
    study = read(root / "study.json")
    continuation = read(root / "continuation.json")
    original_root = Path(study["benchmark_source"]).parent
    original_matches = sum((original_root / name).is_file() and digest(original_root / name) == expected
                           for name, expected in continuation["original_inventory"].items())
    original_unchanged = original_matches == len(continuation["original_inventory"])
    verification_time = datetime.now(timezone.utc).isoformat()
    baseline_path = Path(study["baseline"]["baseline_root"]) / "baseline/graded.json"
    baseline = read(baseline_path)["summary"]
    baseline_environment = environment(baseline_path.parents[1] / "execution/baseline-1")
    def metrics(summary):
        return {
            "strict_passes": summary["metrics"]["prompt_level_strict_accuracy"]["passed"],
            "loose_passes": summary["metrics"]["prompt_level_loose_accuracy"]["passed"],
            "denominator": summary["scheduled_prompts"],
            "scored_output_tokens": summary["known_generated_tokens"],
            "natural_completions": summary["execution_status_counts"].get("ok", 0),
            "truncations": summary["execution_status_counts"].get("truncated", 0),
            "generation_seconds": summary.get("generation_seconds"),
            "technically_valid": summary["technically_valid"],
            "unresolved_grader_errors": summary["unresolved_grader_errors"],
        }
    rounds = []
    for number in range(1, 5):
        key = f"r{number}-astra"
        slot = state["slots"][key]
        dispatch = root / "editor" / f"round-{number}" / "dispatch-provenance.json"
        row = {"round": number, "status": slot["status"], "editor_dispatched": dispatch.exists(),
               "metrics": None, "qualifies": None, "artifact_sha256": slot.get("artifact_sha256")}
        row["editor_delivery_turns"] = read(dispatch).get("editor_delivery_turns", 1) if dispatch.exists() else 0
        if slot.get("session"):
            session = Path(slot["session"])
            arm = read(session / "manifest.json")["arm"]
            graded = session / arm / "graded.json"
            if graded.exists():
                summary = read(graded)["summary"]
                row["metrics"] = metrics(summary)
                row["graded_evidence_sha256"] = digest(graded)
                row["qualifies"] = bool(slot.get("completed") and slot.get("technically_valid")
                    and slot.get("independently_graded") and slot.get("deployment_limits_satisfied")
                    and slot.get("memory_limits_satisfied") and slot.get("strict_passes", 0) > 16
                    and slot.get("scored_output_tokens", float("inf")) <= 13941)
                row["deployment_limits_satisfied"] = slot.get("deployment_limits_satisfied")
                row["memory_limits_satisfied"] = slot.get("memory_limits_satisfied")
                feedback = root / "slots" / key / "feedback.json"
                if feedback.exists():
                    row["paired_outcomes"] = read(feedback)["paired_outcomes"]
        audit = root / "slots" / key / "source-audit.json"
        if audit.exists():
            row["source_edit"] = {k: v for k, v in read(audit).items() if k in (
                "changed_stored_entries", "changed_numerical_entries", "targeted_coordinates", "result_sha256")}
        proposal = root / "slots" / key / "proposal.json"
        if proposal.exists():
            value = read(proposal)
            row["weight_operations"] = value["operations"]
            row["base_checkpoint_sha256"] = value["base_sha256"]
            row["composition"] = value["composition"]
        supervisor = root / "sessions" / key / "execution/candidate-1/supervisor-result.json"
        if supervisor.exists():
            execution = read(supervisor)
            row["execution"] = {k: execution.get(k) for k in (
                "status", "engine_load_attempts", "warmup_dispatches", "scored_dispatches", "generation_dispatches")}
            row["execution"]["cleanup"] = execution.get("cleanup")
            row["execution"]["warmup_sampled_tokens"] = sum(r["generated_tokens"] for r in execution.get("warmup_outputs", []))
            error = execution.get("error")
            row["execution"]["error"] = error if error in (None, "RuntimeError: resource_state_unknown", "RuntimeError: host_resource_stop") else error.split(":", 1)[0] + ": see retained execution evidence"
            row["environment"] = environment(supervisor.parent)
            row["same_browser_as_saved_baseline"] = bool(row["environment"] and baseline_environment and
                row["environment"]["chrome_version"] == baseline_environment["chrome_version"])
        if number == 4 and not dispatch.exists() and slot["status"] == "invalid":
            row["reason"] = "Closed without an Astra call: the preserved failed pre-dispatch stage consumed one of four browser-development attempts."
        rounds.append(row)
    selection = state.get("selection")
    selected = None if selection is None else {
        "named_arms": selection["named_arms"],
        "selection": selection["selected"]["astra"]["selection"],
        "resource_order": selection["resource_order"],
        "frozen_before_final_results": selection["frozen_before_final_results"],
    }
    resource = {"status": "pending", "sessions": [], "reason": "Candidate selection has not resolved."}
    resource_path = root / "resource-analysis.json"
    if resource_path.exists():
        raw = read(resource_path)
        keys = ("status", "scheduled_requests", "measured_requests", "scored_output_tokens", "warmup_output_tokens",
                "response_ms_sum", "load_ms", "session_ms", "natural_completions", "truncations",
                "gpu_buffer_descriptor_bytes_sampled_max", "wasm_address_space_bytes_sampled_max",
                "native_peak_footprint_gb", "native_idle_footprint_gb", "native_recovery_footprint_gb",
                "cpu_capacity_percent", "gpu_time_ms_per_second", "host_power_watts", "cpu_coverage", "gpu_coverage", "power_coverage")
        resource = {"status": raw["status"], "scheduled_sessions": raw["scheduled_sessions"],
                    "complete_sessions": raw["complete_sessions"], "matched_environment": raw["matched_environment"],
                    "sessions": [{"position": r["position"], "arm": r["scheduled_arm"],
                                  "summary": None if not r.get("summary") else {k: r["summary"].get(k) for k in keys}}
                                 for r in raw["sessions"]],
                    "reason": "Pristine retained; no redundant comparison was run." if raw["status"] == "pristine_retained_no_comparison" else raw["interpretation"]}
    outcome = read(root / "study-result.json") if (root / "study-result.json").exists() else {}
    recovery_result = root / "grading-recovery/result.json"
    recovery = None
    if recovery_result.exists():
        saved = read(recovery_result)
        harness = read(root / "grading-recovery/harness-restoration.json")
        restored_environment = read(root / "grading-recovery/environment-restoration.json")
        preflight = read(root / "grading-recovery/frozen-input-preflight-verified.json")
        recovery = {"status": saved["status"], "charged_seconds": saved["charged_seconds"],
            "pre_recovery_state_sha256": saved["pre_recovery_state_sha256"],
            "original_failed_stage_preserved": saved["original_failed_stage_preserved"],
            "target_loads": saved["target_loads"], "target_generations": saved["target_generations"],
            "editor_calls": saved["editor_calls"], "calendar_and_global_limits_unchanged": saved["calendar_and_global_limits_unchanged"],
            "harness_files_restored_exactly": len(harness["restored_files"]),
            "evaluator_payloads_restored": restored_environment["restored_payload_count"],
            "unavailable_generated_bytecode_quarantined": restored_environment["quarantined_bytecode_count"],
            "evaluator_versions_changed": restored_environment["versions_changed"],
            "frozen_inputs_verified_after_restoration": preflight["matched_files"],
            "interpretation": "Round 3 browser execution and cleanup completed. Offline grading initially stopped before any checker call when files became dataless. Exact frozen bytes/pinned package payloads were restored, existing project bytecode was bypassed, and one separately charged offline pass graded the preserved outputs. The failed stage and halt snapshot remain retained."}
    observations = [r["observed"] for r in state.get("generation_sessions", {}).values() if r.get("observed")]
    def observed_sum(key):
        return sum(r.get(key, 0) for r in observations)
    token_observations = [r.get("generation_budget", {}).get("observed", {}).get("generated_tokens") for r in observations]
    actual_usage = {"engine_load_attempts": observed_sum("engine_load_attempts"),
        "warmup_dispatches": observed_sum("warmup_dispatches"), "scored_dispatches": observed_sum("scored_dispatches"),
        "generation_dispatches": observed_sum("generation_dispatches"),
        "sampled_tokens_including_warmups": sum(token_observations) if all(v is not None for v in token_observations) else None,
        "unresolved_generation_sessions": sum(not r.get("observed") for r in state.get("generation_sessions", {}).values()),
        "failed_target_sequences": sum(r.get("generation_budget", {}).get("charged", {}).get("failed_sequences", 0) for r in observations),
        "generation_retries": sum(r.get("generation_budget", {}).get("charged", {}).get("retries", 0) for r in observations),
        "preserved_pre_dispatch_failures": 1,
        "charged_local_seconds": sum(r["charged_seconds"] for r in state["stages"]),
        "charged_browser_seconds": sum(r["charged_seconds"] for r in state["stages"] if r["stage"] in (
            "browser_development", "browser_final", "browser_resource")),
        "scope": "All newly observed development and resource sessions, including warmups; historical saved baseline excluded. Original zero-dispatch failure remains separately counted."}
    return {"schema_version": 1, "kind": "sanitized_v2_collaboration_result", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_sha256": state["study_sha256"], "continuation_sha256": state["continuation_sha256"],
        "status": "complete" if outcome else "halted" if state["halted"] else "in_progress",
        "joint_goal_status": outcome.get("joint_goal_status", "not_established"),
        "model": "Qwen3.5-0.8B", "runtime": "WebLLM 0.2.86 / MLC", "quantization": "q4f16_1",
        "benchmark": "32-prompt Ai2 RLVR-IFeval development set", "baseline": metrics(baseline),
        "baseline_environment": baseline_environment,
        "rounds": rounds, "selection": selected, "resource": resource, "untouched_final_prompts": 0,
        "offline_grading_recovery": recovery,
        "actual_browser_usage": actual_usage,
        "accounting": {"model_loads": sum(r.get("execution", {}).get("engine_load_attempts", 0) for r in rounds),
            "warmups": sum(r.get("execution", {}).get("warmup_dispatches", 0) for r in rounds),
            "scored_generations": sum(r.get("execution", {}).get("scored_dispatches", 0) for r in rounds),
            "scored_output_tokens": sum(r["metrics"]["scored_output_tokens"] for r in rounds if r["metrics"]),
            "warmup_sampled_tokens": sum(r.get("execution", {}).get("warmup_sampled_tokens", 0) for r in rounds),
            "charged_local_seconds": sum(r["charged_seconds"] for r in state["stages"]),
            "charged_browser_seconds": sum(r["charged_seconds"] for r in state["stages"] if r["stage"] in (
                "browser_development", "browser_final", "browser_resource")),
            "development_browser_attempts": sum(r["stage"] == "browser_development" for r in state["stages"]),
            "count_scope": "Development candidate sessions only for loads/generations/tokens; stage times include inherited costs and any resource stages."},
        "editor": {"actual_route": "Codex collaboration subagent", "requested_model": "gpt-6-astra",
            "fresh_context_each_round": True, "effort_override": None, "ultra_verified": False,
            "actual_agent_dispatches": sum(r["editor_dispatched"] for r in rounds), "immutable_provider_version": None,
            "editor_delivery_turns": sum(r["editor_delivery_turns"] for r in rounds),
            "submission_slots_consumed": len(state["submissions"]),
            "provider_tokens": None, "provider_billing": None, "technical_isolation": "unverified"},
        "preserved_pre_dispatch_failure": {"cause": "Sandbox denied native memory-pressure query; resource_state_unknown",
            "model_loads": 0, "warmups": 0, "generations": 0, "original_state_sha256": continuation["original_state_sha256"],
            "original_files_verified": original_matches, "original_files_expected": len(continuation["original_inventory"]),
            "original_verification_at_utc": verification_time, "original_state_and_ledgers_unchanged": original_unchanged,
            "carried_local_seconds": continuation["inherited_local_seconds"], "carried_browser_seconds": continuation["inherited_browser_seconds"],
            "same_calendar_and_limits": True, "maximum_real_candidate_evaluations": 3,
            "correction": "Unchanged controller and native resource guards run outside sandbox; original failed attempt remains counted."},
        "limitations": ["No untouched final prompts are available; development results do not establish fresh held-out quality.",
            "The reused development baseline ran Chrome 156.0.8078.80; these candidates run Chrome 157.0.8103.39. The browser difference prevents an isolated weight-only causal claim from the development comparison. The resource A-B-B-A comparison, if selected, uses the current matched environment.",
            "A collaboration editor was used instead of the frozen manual-interface route; no Ultra or immutable provider identity is claimed.",
            "Two resource sessions per artifact would form a small pilot, not proof of precise resource equality.",
            "GPU allocation descriptors and WASM address space are not physical VRAM; missing power is unavailable, not zero."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/ifeval-browser-v2-collaboration-continuation"))
    parser.add_argument("--out", type=Path, default=Path("reports/ifeval-browser-v2-collaboration-run/result.json"))
    args = parser.parse_args()
    result = report(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rounds_with_metrics": sum(r["metrics"] is not None for r in result["rounds"])}))


if __name__ == "__main__":
    main()
