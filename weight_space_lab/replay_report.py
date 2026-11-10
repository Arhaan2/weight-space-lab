"""Reports for a fixed published-patch replay, with complete denominators.

This module performs no inference and selects no settings. Its inputs must be
the previously frozen analyses and independently graded execution evidence.
Unattempted schedules remain pending; an operational zero for missing output
is never presented as a measured model accuracy.
"""
from __future__ import annotations

import copy
import math

from .analysis_pipeline import compare_graded_runs, validate_analysis_manifest
from .protocol import canonical_hash
from .records import _choice, _digest, _integer, _text, validate_run_records


_ROLES = {"primary", "identity_control", "decoding_control", "alternative_decoding"}


def _finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _resources(records, scheduled, execution):
    """Count each attempt once; final outcomes duplicate the last attempt."""
    attempts = [attempt for row in records for attempt in row["attempts"]]
    tokens = {}
    for key in ("input", "output", "total"):
        known = [row["token_counts"][key] for row in attempts if row["token_counts"][key] is not None]
        tokens[key] = {"known_total": sum(known), "unknown_attempts": len(attempts) - len(known)}
    durations = [row["duration_seconds"] for row in attempts if row["duration_seconds"] is not None]
    accounted = None
    if execution is not None:
        if not isinstance(execution, dict) or "accounted_wall_seconds" not in execution:
            raise ValueError("execution summary requires accounted_wall_seconds")
        accounted = _finite(execution["accounted_wall_seconds"], "accounted wall time")
        if "attempted_sequences" in execution and execution["attempted_sequences"] != len(attempts):
            raise ValueError("execution attempt count differs from raw evidence")
        if "scheduled_sequences" in execution and execution["scheduled_sequences"] != scheduled:
            raise ValueError("execution schedule count differs from frozen denominator")
    return {
        "observed_records": len(records), "attempts": len(attempts),
        "retried_generations": sum(len(row["attempts"]) > 1 for row in records),
        "retry_attempts": sum(max(0, len(row["attempts"]) - 1) for row in records),
        "unattempted_scheduled_generations": scheduled - sum(bool(row["attempts"]) for row in records),
        "token_use_all_attempts": tokens,
        "attempt_duration_seconds": {"known_total": sum(durations),
                                     "unknown_attempts": len(attempts) - len(durations)},
        "accounted_wall_seconds": accounted,
        "duration_scope": "successful attempt durations cover generation; failure durations may cover startup; no final-outcome double counting",
        "wall_time_scope": "journal accounting includes startup and may include reserved time for interrupted attempts; excludes offline grading",
        "unknown_measurements_are_zero": False,
    }


def summarize_comparison(manifest, baseline_records, candidate_records, baseline_grades, candidate_grades,
                         *, comparison_id, comparison_role, baseline_label="original",
                         candidate_label="reconstructed", bootstrap_samples=10000, seed=0,
                         baseline_execution=None, candidate_execution=None):
    """Retain item-clustered paired analysis, failure counts and resource use.

    Primary and control contrasts use the same routine. Alternative decoding
    results stay separately labeled and never replace the primary result.
    ``baseline_execution`` / ``candidate_execution`` optionally supply native
    execution summaries with accounted wall time (including interruption
    reservations), distinct from measured generation durations.
    """
    _text(comparison_id, "comparison_id")
    _choice(comparison_role, _ROLES, "comparison role")
    for label in (baseline_label, candidate_label):
        _text(label, "arm label")
    manifest = validate_analysis_manifest(manifest)
    baseline_records, candidate_records = map(validate_run_records, (baseline_records, candidate_records))
    analysis = compare_graded_runs(baseline_records, candidate_records, baseline_grades, candidate_grades,
                                   manifest, bootstrap_samples=bootstrap_samples, seed=seed)
    comparison = analysis["comparison"]
    arms = {}
    for name, label, records, execution in (
        ("baseline", baseline_label, baseline_records, baseline_execution),
        ("candidate", candidate_label, candidate_records, candidate_execution),
    ):
        projection = analysis[name + "_projection"]
        resource = _resources(records, comparison["scheduled_cases"], execution)
        attempted = resource["attempts"] > 0
        completed = sum(projection["generation_status_counts"][status] for status in ("ok", "truncated"))
        compact = projection["records"]
        instruction_metric = manifest["metric"].startswith("instruction_")
        numerator = sum(row.get("score", 0.0) if instruction_metric else int(row["correct"]) for row in compact)
        arms[name] = {
            "label": label, "scheduled_generations": comparison["scheduled_cases"],
            "scheduled_items": comparison["items"],
            "completed_generations": completed,
            "metric": manifest["metric"],
            "estimand": "average per-generation prompt instruction fraction" if instruction_metric else "average per-generation correctness",
            "best_of_n_success": None, "best_of_n_computed": False,
            "measured_average_per_generation": comparison[name]["case_mean"] if completed else None,
            "measured_equal_item_mean": comparison[name]["mean"] if completed else None,
            "operational_zero_credit_numerator": numerator,
            "operational_zero_credit_denominator": comparison["scheduled_cases"],
            "operational_zero_credit_rate": comparison[name]["case_mean"],
            "score_interpretation": projection["score_interpretation"] if completed else (
                "pending: execution attempts produced no completed generation; operational failures are not measured model correctness"
                if attempted else "pending: no scheduled generation has been attempted"),
            "generation_status_counts": projection["generation_status_counts"],
            "grading_accounting": projection["accounting"], "resources": resource,
        }
        if "official_metrics" in projection:
            arms[name]["official_metrics"] = {
                metric: {**values, "value": values["value"] if completed else None,
                         "operational_zero_credit_value": values["value"],
                         "measurement_pending": not bool(completed)}
                for metric, values in projection["official_metrics"].items()
            }
            arms[name]["official_metrics_interpretation"] = (
                projection["score_interpretation"] if completed else
                "operational zero-credit counts only; measured official metrics remain pending without completed generations")
    any_attempt = any(arm["resources"]["attempts"] for arm in arms.values())
    both_generated = all(arm["completed_generations"] for arm in arms.values())
    failed_without_output = any(arm["resources"]["attempts"] and not arm["completed_generations"] for arm in arms.values())
    unattempted = sum(arm["resources"]["unattempted_scheduled_generations"] for arm in arms.values())
    status = "pending" if not any_attempt else "partial" if unattempted else "complete"
    if failed_without_output:
        status = "execution_failed"
    if any_attempt and not analysis["semantic_grading_complete"]:
        status = "grading_unresolved"
    # Lists remain auditable even in incomplete runs, but are explicitly
    # operational changes until both execution and semantic grading complete.
    changes = {name: [copy.deepcopy(row) for row in comparison["per_item"] if predicate(row["delta"])]
               for name, predicate in (("improved", lambda x: x > 0), ("regressed", lambda x: x < 0),
                                       ("tied", lambda x: x == 0))}
    return {
        "schema_version": 1, "kind": "published_patch_replay_comparison",
        "study_label": "published-patch replay", "comparison_id": comparison_id,
        "comparison_role": comparison_role, "benchmark": manifest["benchmark"],
        "stage": manifest["stage"], "evidence_kind": manifest["evidence_kind"],
        "manifest_sha256": manifest["manifest_sha256"], "measurement_status": status,
        "scientific_result": False,
        "measured_delta": comparison["delta"] if both_generated else None,
        "paired_item_confidence_interval": comparison["confidence_interval"] if both_generated else None,
        "uncertainty_interpretation": "paired item bootstrap over operational zero-credit scores; unresolved grader failures prevent an identified semantic effect"
            if not analysis["semantic_grading_complete"] else (
                "paired item bootstrap; all repeated generations of an item remain in the same resampling cluster"
                if both_generated else "pending paired correctness: at least one arm has no completed generation; operational zero-credit failure scores do not measure the model effect"),
        "item_changes_basis": "complete paired item means" if status == "complete" else "operational zero-credit item means; incomplete results are not semantic improvements",
        "item_changes": changes, "arms": arms, "paired_analysis": analysis,
    }


def build_replay_report(comparisons, *, stage, freeze_sha256, pending_reason=None,
                        workload_estimate=None, discrepancies=None, required_comparison_ids=None):
    """Bundle comparisons without pooling distinct benchmarks or controls.

    Caller-supplied ``required_comparison_ids`` preserves the frozen contrast
    denominator as well as each comparison's item denominator. A pending
    report can still contain complete unattempted schedule summaries.
    """
    _choice(stage, {"smoke", "pilot", "full"}, "report stage")
    _digest(freeze_sha256, "freeze_sha256")
    if not isinstance(comparisons, list):
        raise ValueError("comparisons must be a list")
    identifiers = set()
    primary_benchmarks = set()
    evidence = set()
    for result in comparisons:
        if not isinstance(result, dict) or result.get("kind") != "published_patch_replay_comparison":
            raise ValueError("report requires replay comparison summaries")
        if result["stage"] != stage or result["comparison_id"] in identifiers:
            raise ValueError("report has a mixed stage or duplicate comparison")
        if result["study_label"] != "published-patch replay":
            raise ValueError("report study label differs from published-patch replay")
        identifiers.add(result["comparison_id"])
        evidence.add(result["evidence_kind"])
        if result["comparison_role"] == "primary":
            if result["benchmark"] in primary_benchmarks:
                raise ValueError("only one primary comparison per benchmark is permitted")
            primary_benchmarks.add(result["benchmark"])
    if len(evidence) > 1:
        raise ValueError("fixture and real-model evidence must not be mixed")
    if required_comparison_ids is not None:
        if not isinstance(required_comparison_ids, list) or len(set(required_comparison_ids)) != len(required_comparison_ids):
            raise ValueError("required comparison identities must be a unique list")
        if identifiers != set(required_comparison_ids):
            raise ValueError("report does not include every frozen comparison")
    if pending_reason is not None:
        _text(pending_reason, "pending reason")
    discrepancies = [] if discrepancies is None else discrepancies
    if not isinstance(discrepancies, list) or any(not isinstance(row, str) or not row.strip() for row in discrepancies):
        raise ValueError("discrepancies must be a list of nonempty strings")
    attempted = any(arm["resources"]["attempts"] for result in comparisons for arm in result["arms"].values())
    complete = bool(comparisons) and all(row["measurement_status"] == "complete" for row in comparisons)
    status = "pending" if not attempted else "complete" if complete and pending_reason is None else "partial"
    if pending_reason is None:
        if not attempted:
            pending_reason = "No scheduled model generation has been attempted."
        elif any(not all(arm["completed_generations"] for arm in row["arms"].values()) for row in comparisons):
            pending_reason = "At least one frozen comparison lacks completed generations in one or both arms; measured paired correctness remains pending. Operational missing and failure counts are retained separately."
    return {
        "schema_version": 1, "kind": "published_patch_replay_report",
        "study_label": "published-patch replay", "stage": stage, "freeze_sha256": freeze_sha256,
        "measurement_status": status, "measured_result_pending": status != "complete",
        "pending_reason": pending_reason,
        "experiment_executed": stage == "full" and attempted and evidence == {"real_model"},
        "real_model_generation_attempted": attempted and evidence == {"real_model"},
        "evidence_kind": next(iter(evidence)) if evidence else None,
        "scientific_result": False,
        "primary_comparison": "unchanged versus reconstructed p50c weights under identical settings",
        "metric_interpretation": "average correctness over scheduled generations; never best-of-N success",
        "aggregation": "benchmarks, identity controls and alternative decoding comparisons are reported separately",
        "comparisons": copy.deepcopy(comparisons), "workload_estimate": copy.deepcopy(workload_estimate),
        "discrepancies": copy.deepcopy(discrepancies),
        "classification_rule": "published-patch replay regardless of the direction or size of the measured effect",
        "fresh_editor_access": "final replay tasks, outputs, grades and detailed item reports are evaluator-only artifacts; directory separation alone does not enforce access control",
    }


def estimate_workload(smoke_report, workload, *, target_backend, measured_backend="transformers",
                      hardware_runtime_matched=False):
    """Project an explicit generation schedule using measured smoke rates.

    ``workload`` rows contain benchmark, arm, scheduled_generations and
    max_new_tokens. The cap scenario is a planning extrapolation, not a time
    upper bound: long-context throughput and memory fit are unmeasured. A
    different target backend gets no ETA from the CPU measurement. Backend
    names alone do not identify hardware or runtime: the caller must explicitly
    verify the complete execution environment before enabling a target estimate.
    """
    for backend in (target_backend, measured_backend):
        _text(backend, "backend")
    if type(hardware_runtime_matched) is not bool:
        raise ValueError("hardware_runtime_matched must be an explicit boolean")
    if not isinstance(smoke_report, dict) or smoke_report.get("kind") != "real_model_report":
        raise ValueError("workload estimation requires measured real-model smoke evidence")
    arms = smoke_report.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("smoke report has no measured arms")
    rates, loads, measured_output, measured_seconds = [], [], 0, 0.0
    for row in arms:
        output = row.get("output_tokens")
        _integer(output, "measured output tokens", positive=True)
        duration = _finite(row.get("generation_seconds"), "measured generation seconds", positive=True)
        rates.append(output / duration)
        measured_output += output
        measured_seconds += duration
        if row.get("load_seconds") is not None:
            loads.append(_finite(row["load_seconds"], "measured load seconds"))
    if not isinstance(workload, list) or not workload:
        raise ValueError("workload must have explicit benchmark/arm rows")
    rows, seen = [], set()
    for source in workload:
        if not isinstance(source, dict) or set(source) != {"benchmark", "arm", "scheduled_generations", "max_new_tokens"}:
            raise ValueError("workload row requires benchmark, arm, scheduled_generations and max_new_tokens")
        row = copy.deepcopy(source)
        for key in ("benchmark", "arm"):
            _text(row[key], key)
        identity = row["benchmark"], row["arm"]
        if identity in seen:
            raise ValueError("duplicate benchmark/arm workload row")
        seen.add(identity)
        for key in ("scheduled_generations", "max_new_tokens"):
            _integer(row[key], key, positive=True)
        row["maximum_scheduled_output_tokens"] = row["scheduled_generations"] * row["max_new_tokens"]
        rows.append(row)
    tokens = sum(row["maximum_scheduled_output_tokens"] for row in rows)
    low_seconds, high_seconds = tokens / max(rates), tokens / min(rates)
    applicable = target_backend == measured_backend and hardware_runtime_matched
    reference = {"generation_seconds_range": [low_seconds, high_seconds],
                 "generation_days_range": [low_seconds / 86400, high_seconds / 86400]}
    return {
        "schema_version": 1, "kind": "measured_smoke_workload_extrapolation",
        "smoke_report_sha256": canonical_hash(smoke_report), "measured_backend": measured_backend,
        "target_backend": target_backend, "target_estimate_available": applicable,
        "hardware_runtime_match_verified": hardware_runtime_matched,
        "target_estimate_pending_reason": None if applicable else "Target hardware and complete runtime have not been verified against a matching measured smoke; measured-backend reference is not its ETA.",
        "measured_generations": len(arms), "measured_output_tokens": measured_output,
        "measured_generation_seconds": measured_seconds,
        "measured_rate_range_tokens_per_second": [min(rates), max(rates)],
        "measured_load_seconds_range": [min(loads), max(loads)] if loads else None,
        "workload": rows, "scheduled_generations": sum(row["scheduled_generations"] for row in rows),
        "maximum_scheduled_output_tokens": tokens,
        "measured_backend_token_cap_reference": reference,
        "target_token_cap_projection": copy.deepcopy(reference) if applicable else None,
        "long_context_memory_fit_verified": False,
        "assumptions": [
            "Serial requests on the same hardware, precision, kernels and runtime as the measured smoke.",
            "Every scheduled generation reaches its declared output cap; actual stopping may use fewer tokens.",
            "Constant short-context measured rate; this is not a validated long-context runtime or time upper bound.",
            "Generation component only; model loading, grading, queueing, retries and interruption reservations excluded.",
            "Each benchmark/arm execution requires model startup; measured load observations are reported separately.",
        ],
    }


def render_replay_markdown(report):
    """Render a compact, portable report; full item evidence stays in JSON."""
    if report.get("kind") != "published_patch_replay_report":
        raise ValueError("expected a published-patch replay report")
    lines = ["# Published-patch replay", "", f"Stage: **{report['stage']}**. Measured result: **{report['measurement_status']}**.", "",
             report["metric_interpretation"] + ".", ""]
    if report["pending_reason"]:
        lines += [report["pending_reason"], ""]
    lines += ["| Comparison | Benchmark | Role | Original/control mean | Candidate mean | Paired delta | 95% item CI | Generations per arm |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"]
    def number(value):
        return "pending" if value is None else f"{value:.6f}"
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for row in report["comparisons"]:
        ci = row["paired_item_confidence_interval"]
        interval = "pending" if ci is None else f"[{ci['low']:.6f}, {ci['high']:.6f}]"
        baseline, candidate = row["arms"]["baseline"], row["arms"]["candidate"]
        lines.append(f"| {safe(row['comparison_id'])} | {row['benchmark']} | {row['comparison_role']} | {number(baseline['measured_average_per_generation'])} | {number(candidate['measured_average_per_generation'])} | {number(row['measured_delta'])} | {interval} | {baseline['scheduled_generations']} |")
    lines += ["", "Intervals resample complete items with all repeated generations. Missing generations remain in the scheduled denominator; unresolved grader failures prevent an identified semantic effect.", "",
              "| Comparison / arm | Items | Missing | Error | Timeout | Truncated | Grader error / timeout | Known input / output tokens, all attempts | Known attempt seconds | Accounted wall seconds |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in report["comparisons"]:
        for arm in row["arms"].values():
            counts, grades, resources = arm["generation_status_counts"], arm["grading_accounting"], arm["resources"]
            token = resources["token_use_all_attempts"]
            lines.append(f"| {safe(row['comparison_id'])} / {safe(arm['label'])} | {arm['scheduled_items']} | {counts['missing']} | {counts['error']} | {counts['timeout']} | {counts['truncated']} | {grades['grader_error']} / {grades['grader_timeout']} | {token['input']['known_total']} / {token['output']['known_total']} | {resources['attempt_duration_seconds']['known_total']:.3f} | {number(resources['accounted_wall_seconds'])} |")
    lines += ["", "Token and duration sums are known measurements only. Unknown attempt measurements, retries, individual improved/regressed item IDs, and all grading diagnostics are retained in the JSON report. Accounted wall time includes startup and can include reservations for interrupted work; grading time is excluded.", ""]
    for row in report["comparisons"]:
        changes = row["item_changes"]
        lines.append(f"- {safe(row['comparison_id'])}: {len(changes['improved'])} improved, {len(changes['regressed'])} regressed, {len(changes['tied'])} tied item means ({row['item_changes_basis']}).")
    estimate = report["workload_estimate"]
    if estimate is not None:
        lines += ["", f"Full workload: {estimate['scheduled_generations']:,} generations, at most {estimate['maximum_scheduled_output_tokens']:,} scheduled output tokens. "]
        days = estimate["measured_backend_token_cap_reference"]["generation_days_range"]
        scope = "Target planning estimate" if estimate["target_estimate_available"] else "Historical measured-backend reference only"
        lines += [f"{scope}: the measured {estimate['measured_backend']} smoke extrapolates to {days[0]:,.1f}–{days[1]:,.1f} serial generation days at the token caps. This is a planning scenario, not a validated runtime or upper bound."]
        if estimate["target_estimate_pending_reason"]:
            lines += [estimate["target_estimate_pending_reason"]]
    if report["discrepancies"]:
        lines += ["", "Preserved discrepancies:", ""] + ["- " + value for value in report["discrepancies"]]
    lines += ["", report["classification_rule"] + ".", "", report["fresh_editor_access"] + ".", ""]
    return "\n".join(lines)
