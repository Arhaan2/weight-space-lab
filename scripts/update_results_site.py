#!/usr/bin/env python3
"""Synchronize public evidence, embedded data and no-JavaScript research tables. No inference."""
from __future__ import annotations
import argparse
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLIC_BASE = "https://github.com/Arhaan2/weight-space-lab/blob/codex/results-site/"
ARTIFACTS = {"pristine": "Pristine", "v1": "V1 edit", "v2-r1": "Astra round 1", "v2-r2": "Astra round 2", "v2-r3": "Astra round 3"}


def evidence_url(url):
    """Keep explanatory report prose on the site; link external raw evidence."""
    return url.removesuffix("README.md") + "result.json" if url.endswith("README.md") else url


def replace_region(document: str, name: str, markup: str) -> str:
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError(f"Expected one pair of {name} markers")
    before, rest = document.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + "\n" + markup + "\n" + end + after


def numeric(value, decimals=0, divisor=1):
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Invalid public numeric observation")
    return f"{value / divisor:,.{decimals}f}"


def table(headers, rows, caption):
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = []
    for label, values in rows:
        cells = "".join(f'<td data-label="{escape(h, quote=True)}">{escape(str(v))}</td>' for h, v in zip(headers[1:], values))
        body.append(f'<tr><th scope="row">{escape(label)}</th>{cells}</tr>')
    return f'<table class="data-table"><caption class="visually-hidden">{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def static_evidence(data):
    v1, report = data["v1"], data["v2"]["execution"]
    rows = []
    for label, record in (("Pristine", v1["development"]["baseline"]), ("V1 edit", v1["development"]["candidate"])):
        rows.append((label, [f'{record["strict"]}/{record["total"]}', f'{record["loose"]}/{record["total"]}', numeric(record["output_tokens"])]))
    for row in report["rounds"]:
        if row.get("metrics"):
            record = row["metrics"]
            rows.append((f'Astra round {row["round"]}', [f'{record["strict_passes"]}/{record["denominator"]}', f'{record["loose_passes"]}/{record["denominator"]}', numeric(record["scored_output_tokens"])]))
    content = '<h2>Development quality · 32 prompts</h2>' + table(["Artifact", "Strict passes", "Loose passes", "Output tokens"], rows, "Development results")
    content += '<p>Development-only evidence. The saved baseline and V1 used Chrome 156; fresh Astra candidates used Chrome 157. No V2 edit qualified. No verified untouched final prompts remained; V2’s conditional resource comparison was skipped.</p>'
    heldout = v1["heldout"]["metrics"]
    rows = []
    for arm, label in (("baseline", "Pristine"), ("candidate", "V1 edit")):
        rows.append((label, [f'{heldout["prompt_strict"][arm]["numerator"]}/128', f'{heldout["prompt_loose"][arm]["numerator"]}/128', numeric(v1["heldout"]["arms"][arm]["accounting"]["scored_generated_tokens"])]))
    low, high = heldout["prompt_strict"]["ci95_percentage_points"]
    content += '<h2>Historical V1 held-out quality · 128 prompts</h2>' + table(["Artifact", "Strict passes", "Loose passes", "Output tokens"], rows, "Historical V1 held-out results")
    content += f'<p>Strict prompt difference: +{numeric(heldout["prompt_strict"]["difference_percentage_points"], 2)} percentage points; paired 95% interval {numeric(low, 2)} to +{numeric(high, 2)} points. The interval includes zero. The edited evaluation was interrupted after 94 outputs and continued for 34; historical timing does not establish a controlled resource gain.</p>'
    resource = data.get("resource_study")
    content += '<h2>Independent all-artifact resource study</h2>'
    if resource and resource.get('continuation'):
        content += '<p>' + escape(continuation_summary(resource)) + '</p>'
    sessions = (resource or {}).get("sessions", [])
    if sessions:
        rows = []
        for session in sessions:
            s = session.get("summary") or {}
            rows.append((f'{session["position"]} · {ARTIFACTS.get(session["artifact_key"], session["artifact_key"])}', [session["status"].replace("_", " "), f'{numeric(s.get("measured_requests"))}/{numeric(s.get("scheduled_requests"))}', numeric(s.get("scored_output_tokens")), numeric(s.get("response_ms_sum"), 2, 1000), numeric(s.get("native_peak_footprint_gb"), 3)]))
        content += table(["Session / artifact", "Status", "Requests", "Output tokens", "Response sum (s)", "Sampled native peak (GB)"], rows, "Individual standard-workload resource sessions")
        content += '<p>Two sessions per artifact are planned in mirrored order. Native footprint is sampled owned-Chrome-process-tree footprint in decimal GB, not physical VRAM. CPU, GPU and power coverage are retained in the complete JSON. No equivalence claim is established.</p>'
    else:
        content += '<p>No measured resource session summary is published in this snapshot. No placeholders are treated as observations.</p>'
    content += f'<p><a href="results.json" download>Download complete results JSON</a> · <a href="{escape(evidence_url(data["sources"]["v1_report"]), quote=True)}">V1 evidence JSON</a> · <a href="{escape(evidence_url(data["sources"]["v2_execution"]), quote=True)}">V2 evidence JSON</a></p>'
    return content


def continuation_summary(resource):
    continuation = (resource or {}).get("continuation")
    if not continuation:
        return ""
    inherited = continuation.get("inherited_completed_positions", [])
    remaining = continuation.get("remaining_positions", [])
    gap = continuation.get("inter_session_gap_after_position6_seconds")
    gap_text = f"The gap after position {inherited[-1]} was {numeric(gap, 2, 60)} minutes." if gap is not None and inherited else "The inter-session gap is not yet recorded."
    return f"Documented continuation. Integrity verification stopped before position {remaining[0] if remaining else 'unavailable'}; {len(inherited)} completed sessions were retained. {gap_text} Both R3 observations remain included. Matching hardware/runtime does not establish uninterrupted or constant host conditions."


def fresh_evidence(fresh):
    if not fresh:
        return '<p>The new comparison has no published measurements yet. Historical observations remain available separately.</p>'
    quality = {row['artifact_key']: row for row in fresh.get('quality', {}).get('artifacts', [])}
    resources = fresh.get('resource', {})
    aggregates = resources.get('aggregates', {})
    rows, resource_rows = [], []
    for artifact in fresh.get('artifacts', []):
        key = artifact['key']
        label = artifact.get('label', ARTIFACTS.get(key, key))
        q = quality.get(key, {})
        metrics = q.get('metrics') or {}
        denominator = metrics.get('denominator')
        score = lambda name: f'{numeric(metrics.get(name))}/{numeric(denominator)}' if denominator else '—'
        rows.append((label, [score('strict_passes'), score('loose_passes'), numeric(metrics.get('scored_output_tokens')), q.get('status', 'not_run').replace('_', ' ')]))
        aggregate = aggregates.get(key, {})
        measurements = aggregate.get('metrics', {})
        mean = lambda name, decimals: numeric((measurements.get(name) or {}).get('mean'), decimals)
        resource_rows.append((label, [f"{numeric(aggregate.get('n_completed', 0))}/2", mean('mean_response_ms', 2), mean('native_peak_footprint_gb', 3), mean('cpu_capacity_percent', 2), mean('gpu_time_ms_per_second', 2), mean('host_power_watts', 2)]))
    content = '<h3>Fresh development scores</h3>'
    content += table(['Artifact', 'Strict passes', 'Loose passes', 'Output tokens', 'Status'], rows, 'Fresh IFEval development measurements for every version')
    content += '<p>All versions use the same 32 previously exposed development prompts. These are fresh executions, not untouched test-set confirmation. Warm-ups are excluded from scored token counts.</p>'
    content += '<h3>Fresh browser resource measurements</h3>'
    content += table(['Artifact', 'Sessions', 'Response (ms/request)', 'Native peak (GB)', 'CPU capacity (%)', 'GPU time (ms/s)', 'Host power (W)'], resource_rows, 'Means of two standard-workload browser sessions per artifact')
    content += '<p>Each available mean weights two complete sessions equally. Native footprint is sampled process-tree accounting, not physical VRAM. CPU and GPU values are activity rates, not total computation; unavailable power stays unavailable. Two sessions do not establish resource equivalence.</p>'
    content += '<p><a href="fresh-comparison.json" download>Download the complete fresh comparison</a></p>'
    return content


def fresh_methods(fresh, maintenance=None):
    if not fresh:
        return '<p>A fresh six-version comparison is being prepared. No new measurement is implied by this description.</p>'
    quality = fresh.get('quality', {})
    resource = fresh.get('resource', {})
    n_quality = sum(row.get('status') == 'completed' for row in quality.get('artifacts', []))
    n_resource = sum(row.get('status') == 'completed' and (row.get('summary') or {}).get('status') == 'complete' for row in resource.get('sessions', []))
    content = f'<h2>Fresh six-version study</h2><p><strong>Recorded progress.</strong> {n_quality} of 6 quality evaluations and {n_resource} of 12 resource sessions are complete. The report status is {escape(fresh.get("status", "unavailable").replace("_", " "))}.</p>'
    content += '<p>Pristine, V1 and the three prior Astra edits are evaluated first on the same 32 development prompts. A fresh-context Astra agent receives the approved weight information and all five fresh development results before proposing one fourth edit. The new edit starts from pristine and scales one contiguous 60-column block in each of layers 15 and 19, with an independent multiplier chosen from 0.5, 0.75, 1.25 and 1.5. Tensor shapes, tokenizer, grading and decoding rules stay fixed.</p>'
    content += '<p>The primary target is higher strict compliance than both fresh pristine and the best fresh prior edit. Equal or fewer output tokens is a secondary target, not evidence of equal compute. Every technically valid version proceeds to resource measurement regardless of its quality score; quality gains and cost changes are reported separately.</p>'
    editor = fresh.get('editor_provenance') or {}
    if editor.get('pre_evaluation_rationale'):
        content += '<details><summary>Astra round 4 · pre-evaluation hypothesis</summary><div class="method-body"><p>The following decision summary was saved and hashed with the proposal before the new weights were built or evaluated. It records a prediction, not an explanation established by the eventual score.</p><blockquote><p>' + escape(editor['pre_evaluation_rationale']) + '</p></blockquote>'
        content += '<p>One fresh-context collaboration agent was requested with the <code>gpt-6-astra</code> selector. It received the approved weight packet and five fresh development evaluations. Provider-side immutable model identity, accepted effort, token usage and billing are unavailable; browser inference measurements below do not include the editor’s compute.</p></div></details>'
    content += '<p>The resource schedule runs pristine, V1, Astra rounds 1–4, then reverses that order. Each of the twelve sessions includes one warm-up and 60 measured fixed-prompt requests. The browser session is the replication unit. The mirrored schedule is deterministic, not randomized, and cannot eliminate nonlinear drift or background activity.</p>'
    quality_match = 'verified' if quality.get('matched_environment') is True else 'not yet established'
    resource_match = 'verified' if resource.get('matched_environment') is True else 'not yet established'
    content += f'<p><strong>Recorded environment matching.</strong> Quality: {quality_match}. Resources: {resource_match}. Matching runtime identities does not prove constant host conditions. The actual observations, failures, completion status and accounting are preserved in the downloadable report.</p>'
    content += '<p>This adaptive development comparison supplies no untouched final-quality evaluation. Higher development scores are not a generalization claim. CPU/GPU rates and footprint are measured within their stated scopes; no FLOP count, causal energy saving or resource equivalence is inferred.</p>'
    limits = fresh.get('limits', {})
    if limits:
        content += f'<details><summary>Frozen limits and accounting</summary><div class="method-body"><p>The schedule permits {numeric(limits.get("engine_loads"))} model loads, {numeric(limits.get("warmups"))} warm-ups, {numeric(limits.get("quality_requests"))} quality requests and {numeric(limits.get("resource_requests"))} resource requests. The {numeric(limits.get("dispatches"))} total dispatches reserve at most {numeric(limits.get("reserved_target_tokens"))} output positions, including control positions; this is a capacity ceiling, not observed model output.</p>'
        content += f'<p>One editor proposal, response import, source build, conversion and package are allowed. Automatic retries are zero. Browser time is capped at {numeric(limits.get("browser_seconds"), divisor=60)} minutes, charged local time at {numeric(limits.get("local_seconds"), divisor=60)} minutes and calendar time at {numeric(limits.get("calendar_seconds"), divisor=3600)} hours. Failed preparation, integrity checks, grading and reporting consume the local allowance. Reservations, actual usage and cleanup status are retained separately in the report.</p></div></details>'
    preparation = fresh.get('preparation', {})
    if preparation.get('notes'):
        content += '<details><summary>Preparation and provenance</summary><div class="method-body">'
        content += ''.join('<p>' + escape(note) + '</p>' for note in preparation['notes'])
        failures = preparation.get('failed_preparation_stages', [])
        if failures:
            content += table(['Preparation stage', 'Target-model calls', 'Charged seconds'], [(row['stage'], [numeric(row.get('target_calls')), numeric(row.get('charged_seconds'), 2)]) for row in failures], 'Preserved preparation failures before the study freeze')
        content += '<p>These checks establish the identities and transformations actually recorded. They do not turn unavailable historical dependency metadata into verified evidence.</p></div></details>'
    if maintenance:
        operations = maintenance.get('operations', [])
        content += '<details><summary>Browser cache housekeeping</summary><div class="method-body"><p>' + escape(maintenance.get('scope', '')) + '</p>'
        notes = maintenance.get('notes', [])
        if isinstance(notes, str):
            notes = [notes]
        content += ''.join('<p>' + escape(note) + '</p>' for note in notes)
        content += f'<p>{numeric(len(operations))} completed-profile maintenance records are retained. The documented addition was introduced after {numeric(maintenance.get("introduced_after_quality_sessions"))} quality sessions; it was not part of the initial freeze. It changes neither the scheduled model calls nor the study’s limits.</p>'
        content += '<p><a href="fresh-maintenance.json" download>Download cache-maintenance receipts and source bindings</a></p></div></details>'
    content += '<p><a href="fresh-comparison.json" download>Download fresh comparison methods and evidence</a></p>'
    return content


def static_methods(methods, resource=None):
    def fields(values):
        return '<dl>' + ''.join(f'<div><dt>{escape(key.replace("_", " ")[:1].upper() + key.replace("_", " ")[1:])}</dt><dd>{escape(json.dumps(value) if isinstance(value, (dict, list)) else "Unavailable" if value is None else str(value))}</dd></div>' for key, value in values.items()) + '</dl>'
    blocks = []
    for study in methods.get("studies", []):
        sources = ' · '.join(f'<a href="{escape(evidence_url(source["url"]), quote=True)}">{escape(source["label"])}</a>' for source in study.get("sources", []))
        aggregation = f'<p><strong>Aggregation.</strong> {escape(study["aggregation"])}</p>' if study.get("aggregation") else ""
        public_prompt = f'<p><strong>Public resource workload prompt.</strong> “{escape(study["public_workload_prompt"])}”</p>' if study.get("public_workload_prompt") else ""
        limitations = '<ul>' + ''.join(f'<li>{escape(item)}</li>' for item in study.get("limitations", [])) + '</ul>'
        blocks.append(f'<details><summary>{escape(study["title"])}</summary><div class="method-body"><p>{escape(study["question"])}</p><p><strong>Scope.</strong> {escape(study["scope"])}</p><p><strong>Selection.</strong> {escape(study["selection_rule"])}</p>{aggregation}{public_prompt}{fields(study.get("generation_settings", {}))}{limitations}<p>{sources}</p></div></details>')
    if resource and resource.get("continuation"):
        continuation = resource["continuation"]
        recovery = continuation.get("recovery", {})
        gap = continuation.get("inter_session_gap_after_position6_seconds")
        facts = {"Preserved completed positions": ", ".join(map(str, continuation.get("inherited_completed_positions", []))), "Continuation positions": ", ".join(map(str, continuation.get("remaining_positions", []))), "Failure stage": continuation.get("failure_stage"), "Failure type": continuation.get("failure_type"), "Recovered file role": recovery.get("file_role"), "Recovered bytes": numeric(recovery.get("size_bytes")), "Recovered SHA-256": recovery.get("sha256"), "Recovery method": recovery.get("method"), "Gap after position 6": numeric(gap, 2) + " seconds" if gap is not None else "Not yet recorded", "Original calendar and total limits": "Unchanged" if continuation.get("original_calendar_and_total_limits_unchanged") is True else "Not established", "Replaced completed sessions": continuation.get("replaced_completed_sessions"), "Model dispatch retries": continuation.get("model_dispatch_retries")}
        blocks.append('<details><summary>Documented resource continuation</summary><div class="method-body"><p>' + escape(continuation.get("interruption_note", "")) + '</p>' + fields(facts) + '<p>Hardware and runtime matching does not establish constant background host activity or uninterrupted conditions. Both R3 observations remain in the descriptive comparison.</p></div></details>')
    if resource and resource.get("environment"):
        environment = resource["environment"]
        hardware = environment.get("hardware", {})
        try:
            memory = int(hardware.get("hw.memsize", ""))
        except (ValueError, TypeError):
            memory = None
        values = {"Processor": hardware.get("machdep.cpu.brand_string", "Unavailable"), "Logical CPUs": hardware.get("hw.logicalcpu", "Unavailable"), "Physical / unified memory": (f"{numeric(memory / 2**30, 2 if memory % 2**30 else 0)} GiB ({numeric(memory)} bytes)" if memory and memory > 0 else "Unavailable"), "Hardware model": hardware.get("hw.model", "Unavailable"), "macOS": environment.get("macos", "Unavailable"), "Chrome": environment.get("browser_version", "Unavailable"), "Runtime": environment.get("runtime", "Unavailable"), "Quantization": environment.get("quantization", "Unavailable")}
        verification = "Matched hardware and runtime verified across the completed schedule." if resource.get("matched_environment") is True else ("Hardware and runtime matching remains pending until the full schedule completes." if resource.get("status") == "in_progress" else "Hardware and runtime matching was not established for the full schedule.")
        blocks.append('<details><summary>Resource environment</summary><div class="method-body"><p>' + verification + '</p>' + fields(values) + '<p>Memory is the operating system’s reported physical / unified capacity, converted from bytes to GiB. It is not a measurement of model memory or physical GPU residency.</p></div></details>')
    blocks.append('<details><summary>Scoring and uncertainty</summary><div class="method-body">' + fields(methods.get('scoring', {})) + '</div></details>')
    telemetry = []
    for definition in methods.get("telemetry_definitions", {}).values():
        prose = ' '.join(str(definition[key]) for key in ('definition', 'scope', 'coverage', 'limitations', 'missing') if definition.get(key) is not None)
        telemetry.append(f'<div><dt>{escape(definition["label"])} · {escape(definition.get("unit", ""))}</dt><dd>{escape(prose)}</dd></div>')
    blocks.append('<details><summary>Telemetry definitions and coverage</summary><div class="method-body"><dl>' + ''.join(telemetry) + '</dl></div></details>')
    notes = methods.get("reproduction", []) + methods.get("preserved_failures", [])
    blocks.append('<details><summary>Reproduction and preserved failures</summary><div class="method-body"><ul>' + ''.join(f'<li>{escape(note)}</li>' for note in notes) + '</ul></div></details>')
    return '\n'.join(blocks)


def script_json(identifier, data):
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return f'<script id="{identifier}" type="application/json">{encoded}</script>'


JOINT_NUMERIC_FIELDS = {
    "quality": ("strict_passed", "scheduled", "strict_accuracy", "unresolved_grades"),
    "totals": ("output_tokens", "response_ms", "known_output_tokens", "known_response_ms", "measured_requests", "mean_response_ms", "full_response_tokens_per_second"),
    "resources": ("native_peak_footprint_gb", "gpu_buffer_descriptor_bytes_sampled_max", "wasm_address_space_bytes_sampled_max", "cpu_capacity_percent", "cpu_coverage", "gpu_time_ms_per_second", "gpu_coverage", "host_power_watts", "power_coverage", "observed_host_energy_joules", "host_energy_joules", "native_samples"),
    "cost_per_compliant_answer": ("tokens_per_strict_pass", "response_ms_per_strict_pass", "host_joules_per_strict_pass"),
}


def public_joint_report(report):
    """Whitelist canonical joint measurements; never publish arbitrary execution data."""
    if (report.get("schema_version") != 1 or report.get("kind") != "mechanistic_browser_comparison"
            or report.get("accuracy_benchmark") != "ifeval" or report.get("runs_per_arm") != 1
            or report.get("separate_resource_generations") != 0
            or report.get("status") not in ("not_run", "baseline_only", "complete", "incomplete")):
        raise ValueError("Expected the canonical one-pair IFEval mechanistic study report")

    def numbers(source, names):
        out = {}
        for name in names:
            value = source.get(name)
            numeric(value)  # Reject NaN, Infinity, booleans and strings; retain missing as null.
            out[name] = value
        return out

    def identity(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            raise ValueError("Invalid public request identity")
        return value

    public = {key: report[key] for key in ("schema_version", "kind", "status", "accuracy_benchmark", "runs_per_arm", "separate_resource_generations")}
    public["scheduled_prompts"] = numbers(report, ("scheduled_prompts",))["scheduled_prompts"]
    public["arms"] = {}
    for key, arm in report.get("arms", {}).items():
        if key not in ("baseline", "candidate") or arm.get("kind") != "joint_browser_accuracy_resource_result" or arm.get("status") not in ("completed", "incomplete"):
            raise ValueError("Unexpected joint evaluation arm")
        clean = {"kind": arm["kind"], "status": arm["status"]}
        for field in ("study_sha256", "manifest_sha256", "artifact_sha256", "source_checkpoint_sha256", "items_sha256", "model_lib_sha256", "harness_sha256", "evidence_sha256"):
            value = arm.get(field)
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError("Invalid joint report digest")
            clean[field] = value
        for section, fields in JOINT_NUMERIC_FIELDS.items():
            clean[section] = numbers(arm.get(section) or {}, fields)
        quality = clean["quality"]
        if (quality["scheduled"] != public["scheduled_prompts"] or not isinstance(quality["scheduled"], int)
                or not 1 <= quality["scheduled"] <= 128 or not isinstance(quality["strict_passed"], int)
                or not 0 <= quality["strict_passed"] <= quality["scheduled"]
                or quality["strict_accuracy"] != quality["strict_passed"] / quality["scheduled"]):
            raise ValueError("Inconsistent joint IFEval denominator")
        clean["items"] = []
        for item in arm.get("items", []):
            row = {"item_id": identity(str(item["item_id"])), "request_id": identity(item["request_id"])}
            # Execution/grader statuses are constrained tokens, never free-form exception text.
            row.update({name: identity(item[name]) for name in ("status", "grading_status")})
            for field in ("strict_pass", "loose_pass"):
                if item.get(field) is not None and (not isinstance(item[field], (bool, int, float)) or item[field] not in (0, 1)):
                    raise ValueError("Invalid joint grade")
                row[field] = None if item.get(field) is None else bool(item[field])
            row.update(numbers(item, ("output_tokens", "response_ms")))
            row["resources"] = numbers(item.get("resources") or {}, JOINT_NUMERIC_FIELDS["resources"])
            clean["items"].append(row)
        if len(clean["items"]) != quality["scheduled"] or len({row["item_id"] for row in clean["items"]}) != quality["scheduled"]:
            raise ValueError("Joint report must retain every scheduled request exactly once")
        if sum(row["strict_pass"] is True for row in clean["items"]) != quality["strict_passed"]:
            raise ValueError("Joint pass count differs from request grades")
        public["arms"][key] = clean
    keys = set(public["arms"])
    expected = {"not_run": set(), "baseline_only": {"baseline"}, "complete": {"baseline", "candidate"}, "incomplete": {"baseline", "candidate"}}
    if keys != expected[public["status"]]:
        raise ValueError("Joint report state disagrees with its arms")
    comparison = report.get("comparison")
    public["comparison"] = None
    if comparison:
        if comparison.get("kind") != "joint_browser_accuracy_resource_comparison" or comparison.get("status") not in ("completed", "unmatched_or_incomplete"):
            raise ValueError("Unexpected joint comparison")
        matched = comparison.get("matched_environment_and_workload") is True
        complete = comparison["status"] == "completed"
        if complete != (public["status"] == "complete") or (complete and not matched):
            raise ValueError("Inconsistent paired comparison status")
        public["comparison"] = {"kind": comparison["kind"], "status": comparison["status"], "matched_environment_and_workload": matched}
        public["comparison"].update(numbers(comparison, ("strict_accuracy_delta", "output_tokens_delta", "response_ms_delta", "native_peak_footprint_gb_delta", "host_energy_joules_delta")))
        public["comparison"]["paired_outcomes"] = numbers(comparison["paired_outcomes"], ("improved", "regressed", "tied")) if comparison.get("paired_outcomes") else None
        if complete:
            arms = public["arms"]
            if any(arms[a]["status"] != "completed" for a in arms) or [r["item_id"] for r in arms["baseline"]["items"]] != [r["item_id"] for r in arms["candidate"]["items"]]:
                raise ValueError("Completed pair requires complete arms and matching request order")
    elif len(keys) == 2:
        raise ValueError("Two-arm report requires a comparison record")
    return public


def joint_markup(report):
    """Render current observations without blending them with past protocols."""
    rows, costs, resource_rows, request_rows = [], [], [], []
    present = bool(report and report["arms"])
    missing = lambda value, digits=0, divisor=1: "Unavailable" if value is None else numeric(value, digits, divisor)
    for key, label in (("baseline", "Pristine"), ("candidate", "Frozen candidate")):
        arm = (report or {}).get("arms", {}).get(key)
        if not arm:
            rows.append((label, ["Not run", "Unavailable", "Unavailable", "Unavailable", "Unavailable"]))
            continue
        q, t, r, c = (arm[name] for name in ("quality", "totals", "resources", "cost_per_compliant_answer"))
        grade = f'{numeric(q["strict_passed"])}/{numeric(q["scheduled"])}'
        if q["unresolved_grades"]:
            grade += " · lower bound"
        rows.append((label, [arm["status"].capitalize(), grade, missing(t["output_tokens"]), missing(t["response_ms"], 2, 1000), missing(r["native_peak_footprint_gb"], 3)]))
        costs.append((label, [missing(c["tokens_per_strict_pass"], 1), missing(c["response_ms_per_strict_pass"], 2, 1000), missing(c["host_joules_per_strict_pass"], 2)]))
        coverage = lambda metric, cov: f'{missing(r[metric], 2)} · {numeric(r[cov], 1, .01)}% coverage' if r[cov] is not None else 'Unavailable'
        resource_rows.append((label, [coverage("cpu_capacity_percent", "cpu_coverage"), coverage("gpu_time_ms_per_second", "gpu_coverage"), coverage("host_power_watts", "power_coverage"), missing(r["host_energy_joules"], 2)]))
        for index, item in enumerate(arm["items"], 1):
            result = "Pass" if item["strict_pass"] is True else "Fail" if item["strict_pass"] is False else "Unresolved"
            request_rows.append((f'{label} · {index}', [item["item_id"], item["status"].replace("_", " "), result, missing(item["output_tokens"]), missing(item["response_ms"], 2, 1000), missing(item["resources"]["native_peak_footprint_gb"], 3)]))
    content = table(["Artifact", "Status", "Strict passes", "Output tokens", "Response sum (s)", "Native peak (GB)"], rows, "Same-request IFEval accuracy and browser resource measurements")
    if not present:
        return content + '<p class="small-note">No historical value is used to fill these fields. A completed record must bind grades and resource observations to the same prompts, requests, runtime and deployed artifacts.</p>'
    comparison = report.get("comparison") or {}
    if report["status"] == "complete" and comparison.get("matched_environment_and_workload"):
        content += f'<p class="harness-joint-reading">Candidate minus pristine: <strong>{numeric(comparison["strict_accuracy_delta"], 2, .01)} percentage points</strong> strict compliance; <strong>{missing(comparison["output_tokens_delta"])}</strong> output tokens; <strong>{missing(comparison["response_ms_delta"], 2, 1000)} seconds</strong> summed response time.</p>'
    else:
        content += '<p class="small-note">Paired change is unavailable until both arms complete with matched workload and environment identities.</p>'
    content += '<p class="small-note">One browser session per arm. These are descriptive observations on the same requests, including unsuccessful answers. Missing totals remain unavailable; individual recorded requests remain inspectable. Native footprint is sampled process-tree memory, not physical VRAM.</p>'
    content += '<h3>Cost per strictly compliant answer</h3>' + table(["Artifact", "Tokens / pass", "Response seconds / pass", "Host joules / pass"], costs, "Total scored workload cost divided by strict passes")
    content += '<p class="small-note">All scored outputs contribute to cost, including failed and truncated answers. A zero-pass or unresolved-grade denominator has no defined cost per pass. Warm-up and model load are outside these scored totals.</p>'
    content += '<details class="harness-request-records"><summary>Resource coverage and request records</summary>'
    content += table(["Artifact", "CPU capacity (%)", "GPU time (ms/s)", "Host power (W)", "Host energy (J)"], resource_rows, "Observed resource rates and interval coverage")
    content += '<p class="small-note">CPU and GPU are activity rates. Host energy is reported only for complete measured intervals; it is not model-only energy. Coverage below 100% is retained explicitly.</p>'
    content += table(["Artifact / request", "IFEval item", "Status", "Strict grade", "Output tokens", "Response (s)", "Native peak (GB)"], request_rows, "Each graded request and its own browser resource observations")
    return content + '</details><p><a href="joint-comparison.json" download>Download public paired observations</a></p>'


def harness_markup(contract, joint=None):
    """Publish the implemented workflow separately from measured historical results."""
    if (contract.get("kind") != "mechanistic_browser_harness_contract"
            or contract.get("status") not in ("implemented_not_run", "implemented")
            or contract.get("accuracy_benchmark") != "IFEval"
            or contract.get("evaluation", {}).get("results") not in (None, "joint-comparison.json")):
        raise ValueError("Expected the versioned IFEval harness contract; import measurements with --joint-report")
    evaluation = contract["evaluation"]
    if evaluation.get("arms") != ["pristine", "candidate"] or evaluation.get("runs_per_arm") != 1:
        raise ValueError("Harness presentation expects one baseline and one candidate evaluation")
    stages = contract["stages"]
    if [stage["key"] for stage in stages] != ["weights", "activations", "edit", "evaluate"]:
        raise ValueError("Unexpected diagnostic workflow")
    controls = ''.join(
        f'<button type="button" data-harness-stage="{escape(stage["key"])}" aria-pressed="{str(index == 0).lower()}" aria-controls="harness-stage-{escape(stage["key"])}"><span>{escape(stage["label"])}</span><small>{escape(stage["summary"])}</small></button>'
        for index, stage in enumerate(stages))
    panels = ''.join(
        f'<article id="harness-stage-{escape(stage["key"])}" class="harness-stage-detail" data-stage-detail="{escape(stage["key"])}"><h3>{escape(stage["label"])}</h3><p>{escape(stage["detail"])}</p><dl><div><dt>Recorded evidence</dt><dd>{escape(stage["evidence"])}</dd></div></dl></article>'
        for stage in stages)
    metrics = ''.join(f'<div><dt>{escape(row["label"])}</dt><dd>{escape(row["definition"])}</dd></div>' for row in contract["measurements"])
    limits = ''.join(f'<li>{escape(value)}</li>' for value in contract["limitations"] if not joint or not joint["arms"] or not value.startswith("No new paired evaluation"))
    base = contract["source_base"]
    if not base.startswith("https://github.com/Arhaan2/weight-space-lab/blob/"):
        raise ValueError("Harness source links must identify the public repository")
    sources = ''.join(f'<a href="{escape(base + row["path"], quote=True)}"><span>{escape(row["label"])}</span><span aria-hidden="true">↗</span></a>' for row in contract["sources"])
    record_status = {"not_run": "Awaiting first execution", "baseline_only": "Pristine arm recorded", "complete": "Both arms complete", "incomplete": "Incomplete or unmatched pair"}[(joint or {}).get("status", "not_run")]
    implementation_status = "Updated harness. First paired browser evaluation has not run." if not joint or joint["status"] == "not_run" else f"{record_status}. One-session observations; no repeated-run uncertainty estimate."
    return f'''<div class="page-heading"><div><p class="eyebrow">IFEval + Browser LLM Bench</p><h1 id="harness-heading">{escape(contract["title"])}</h1><p class="page-intro">{escape(contract["question"])}</p></div><a class="outline-button" href="harness.json" download>Download harness contract <span aria-hidden="true">↓</span></a></div>
    <div class="finding-strip harness-status"><span class="finding-label">Harness status</span><p>{implementation_status}</p><a href="#results">Earlier results <span aria-hidden="true">→</span></a></div>
    <div class="harness-scope"><div><span>Accuracy</span><strong>IFEval instruction following</strong></div><div><span>Resource measurement</span><strong>Browser LLM Bench / WebLLM</strong></div><div><span>Main comparison</span><strong>One pristine + one candidate run</strong></div></div>
    <section class="harness-workflow" aria-labelledby="harness-workflow-heading"><div class="section-heading"><h2 id="harness-workflow-heading">From diagnosis to one paired comparison</h2><span>Inspect each stage</span></div><div class="harness-flow"><div class="harness-stage-controls" role="group" aria-label="Inspection and evaluation stages">{controls}</div><div class="harness-stage-panels">{panels}</div></div><p class="small-note">{escape(evaluation["diagnostics"])}</p></section>
    <div class="harness-measurement-grid"><section aria-labelledby="harness-request-heading"><div class="section-heading"><h2 id="harness-request-heading">One answer, two measurements</h2><span>Joined by request identity</span></div><p class="harness-lead">{escape(contract["workload"])}</p><dl class="harness-measurements">{metrics}</dl></section><aside class="harness-comparison-note"><h2>What changes from the earlier studies</h2><p>The editor can inspect weights and activation evidence, then change a selected direction instead of being limited to scaling contiguous columns.</p><p>The accuracy workload now also supplies the browser resource workload. The standard repeated-prompt pilot remains in the historical record.</p><p>Additional accuracy benchmarks are a later extension. This version uses IFEval only.</p></aside></div>
    <section class="harness-paired-results" aria-labelledby="harness-paired-heading"><div class="section-heading"><h2 id="harness-paired-heading">Paired evaluation record</h2><span>{record_status}</span></div>{joint_markup(joint)}</section>
    <div class="harness-bottom"><details class="harness-boundaries"><summary>How to interpret the next result</summary><ul>{limits}</ul></details><section class="harness-sources" aria-labelledby="harness-sources-heading"><h2 id="harness-sources-heading">Implementation and protocol</h2>{sources}</section></div>'''


def update(args):
    report = json.loads(args.report.read_text())
    if report.get("schema_version") != 1 or report.get("kind") != "sanitized_v2_collaboration_result":
        raise ValueError("Expected the sanitized public V2 report")
    if report["untouched_final_prompts"] != 0:
        raise ValueError("Review study copy before adding new final-quality data")
    data_path, html_path, methods_path = args.site / "results.json", args.site / "index.html", args.site / "research-methodology.json"
    data = json.loads(data_path.read_text())
    evaluated = [r for r in report["rounds"] if r.get("metrics")]
    qualified = sum(r.get("qualifies") is True for r in evaluated)
    status = report["resource"]["status"]
    label = {"pristine_retained_no_comparison": "Skipped: pristine retained", "pending": "Not run"}.get(status, status.replace("_", " ").capitalize())
    description = f'{len(evaluated)} candidate development evaluations are complete. No edit qualified; the original model was retained. The combined quality and resource goal was not demonstrated.' if status == "pristine_retained_no_comparison" else f'{len(evaluated)} candidate development evaluations are recorded; {qualified} qualify.'
    data["v2"].update(status=report["status"], description=description, resource_status=status, resource_status_label=label, limitation="No verified untouched final prompts remain. V2 quality results are development evidence, without fresh held-out confirmation.", execution=report)
    if args.source_url:
        data["sources"]["v2_execution"] = args.source_url
    if args.repository_base:
        if not args.repository_base.startswith("https://github.com/Arhaan2/weight-space-lab/blob/"):
            raise ValueError("Repository base must identify the public source repository")
        data["sources"]["repository_base"] = args.repository_base.rstrip("/") + "/"
    base = data["sources"].get("repository_base", PUBLIC_BASE)
    if args.resource_report.exists():
        resource = json.loads(args.resource_report.read_text())
        if resource.get("schema_version") != 1 or not isinstance(resource.get("sessions"), list):
            raise ValueError("Expected versioned public resource report with session records")
        data["resource_study"] = resource
        data["sources"]["resource_study"] = args.resource_source_url or data["sources"].get("resource_study") or base + "reports/resource-all-edits-v1/result.json"
    if args.fresh_report.exists():
        fresh = json.loads(args.fresh_report.read_text())
        if fresh.get("schema_version") != 1 or not isinstance(fresh.get("quality"), dict) or not isinstance(fresh.get("resource"), dict) or not isinstance(fresh.get("artifacts"), list):
            raise ValueError("Expected a versioned fresh quality/resource comparison")
        data["fresh_comparison"] = fresh
        data["sources"]["fresh_comparison"] = args.fresh_source_url or data["sources"].get("fresh_comparison") or PUBLIC_BASE + "reports/fresh-baseline-r4-v1/result.json"
    fresh = data.get("fresh_comparison")
    if args.fresh_maintenance.exists():
        maintenance = json.loads(args.fresh_maintenance.read_text())
        if maintenance.get('kind') != 'fresh_profile_cache_maintenance' or not fresh or maintenance.get('study_sha256') != fresh['study_sha256']:
            raise ValueError('Fresh cache-maintenance record must bind to the fresh study')
        data['fresh_maintenance'] = maintenance
        data['sources']['fresh_maintenance'] = args.fresh_maintenance_source_url or data['sources'].get('fresh_maintenance') or PUBLIC_BASE + 'reports/fresh-baseline-r4-v1/maintenance.json'
    data["updated"] = datetime.fromisoformat(report["updated_at_utc"]).astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat()
    resource_updated = data.get("resource_study", {}).get("updated_at_utc")
    if resource_updated:
        data["updated"] = max(data["updated"], datetime.fromisoformat(resource_updated).astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat())
    if fresh and fresh.get("updated_at_utc"):
        data["updated"] = max(data["updated"], datetime.fromisoformat(fresh["updated_at_utc"]).astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat())
    methods = json.loads(methods_path.read_text())
    v1_base = data["sources"]["v1_report"].split("reports/", 1)[0]
    source_map = {"reports/ifeval-browser-v2-collaboration-run/result.json": data["sources"]["v2_execution"].replace("README.md", "result.json"), "reports/ifeval-browser-v2-collaboration-run/README.md": data["sources"]["v2_execution"], "reports/resource-all-edits-v1/result.json": data["sources"].get("resource_study", base + "reports/resource-all-edits-v1/result.json")}
    def normalize(value):
        if isinstance(value, dict):
            out = {k: normalize(v) for k, v in value.items()}
            url = out.get("url")
            if isinstance(url, str) and (url.startswith("../") or (args.repository_base and url.startswith(PUBLIC_BASE))):
                path = url[3:] if url.startswith("../") else url[len(PUBLIC_BASE):]
                historical = path.startswith("reports/ifeval-browser-v1-") or path in ("docs/ifeval-v1-reproduction.md", "docs/ifeval-v2-final-data.md")
                out["url"] = source_map.get(path, (v1_base if historical else base) + path)
            return out
        return [normalize(v) for v in value] if isinstance(value, list) else value
    methods = normalize(methods)
    if fresh:
        methods["fresh_comparison"] = {"report": "fresh-comparison.json", "source": data["sources"]["fresh_comparison"], "scope": "Planned schedule: six fresh executions on the same 32 exposed development prompts and two separate resource sessions per artifact. Actual completion, observations and failures are reported in the linked comparison.", "status": fresh['status']}
        sources = [item for item in methods.get("sources", []) if item.get("label") not in ("Fresh six-version comparison", "Fresh comparison cache-maintenance evidence")]
        sources.append({"label": "Fresh six-version comparison", "url": data["sources"]["fresh_comparison"]})
        if data.get('fresh_maintenance'):
            methods['fresh_comparison']['maintenance'] = {'report': 'fresh-maintenance.json', 'source': data['sources']['fresh_maintenance']}
            sources.append({'label': 'Fresh comparison cache-maintenance evidence', 'url': data['sources']['fresh_maintenance']})
        methods["sources"] = sources
    resource = data.get("resource_study")
    if resource:
        for study in methods.get("studies", []):
            if study["id"] in ("resource", "resource-all-edits", "resource_all_edits", "resource-all-edits-v1"):
                study["status"] = resource["status"]
    html = replace_region(html_path.read_text(), "STATIC_EVIDENCE", static_evidence(data))
    html = replace_region(html, "RESEARCH_DATA", script_json("research-data", data))
    html = replace_region(html, "METHODOLOGY_DATA", script_json("methodology-data", methods))
    html = replace_region(html, "METHODS_DETAIL", static_methods(methods, resource))
    html = replace_region(html, "FRESH_EVIDENCE", fresh_evidence(fresh))
    html = replace_region(html, "FRESH_METHODS", fresh_methods(fresh, data.get('fresh_maintenance')))
    if resource:
        count = sum(row.get("status") == "completed" and (row.get("summary") or {}).get("status") == "complete" and (row.get("summary") or {}).get("measured_requests") == 60 and (row.get("summary") or {}).get("scheduled_requests") == 60 for row in resource["sessions"])
        method_status = f"{count} of 10 planned sessions are complete in the public report. Quality eligibility was not required. Two sessions per artifact form a descriptive pilot, not an equivalence test."
        method_state = resource["status"].replace("_", " ").capitalize()
        download = '<a href="resource-study.json" download><span>Resource sessions</span><small>JSON · sessions and telemetry</small><span aria-hidden="true">↓</span></a>'
    else:
        method_status = "Measured sessions appear in Resources only when recorded in the public report. A small pilot does not establish resource equivalence."
        method_state = "Report pending"
        download = '<a href="https://github.com/Arhaan2/weight-space-lab"><span>Source repository</span><small>Protocols, code and reports</small><span aria-hidden="true">↗</span></a>'
    html = replace_region(html, "RESOURCE_METHOD_STATUS", escape(method_status))
    html = replace_region(html, "RESOURCE_METHOD_STATE", escape(method_state))
    html = replace_region(html, "RESOURCE_DOWNLOAD", download)
    finding = (f"Quality improvement remains unestablished. {count} of {len(resource['sessions'])} resource sessions are complete across {len(resource['artifacts'])} artifacts." if resource else "V1’s held-out change is inconclusive. None of the three new Astra edits improved the strict development score.")
    if fresh:
        quality_rows = {row['artifact_key']: row for row in fresh.get('quality', {}).get('artifacts', [])}
        pristine = (quality_rows.get('pristine') or {}).get('metrics') or {}
        r4 = (quality_rows.get('astra-r4') or {}).get('metrics') or {}
        completed_resources = sum(row.get('status') == 'completed' and (row.get('summary') or {}).get('status') == 'complete' for row in fresh.get('resource', {}).get('sessions', []))
        if pristine and r4:
            finding = f"Astra round 4: {numeric(r4.get('strict_passes'))}/{numeric(r4.get('denominator'))} strict passes; fresh pristine: {numeric(pristine.get('strict_passes'))}/{numeric(pristine.get('denominator'))}. {completed_resources}/12 resource sessions complete. Development evidence; no untouched confirmation."
            observed = [(key, row['metrics']) for key, row in quality_rows.items() if row.get('status') == 'completed' and row.get('metrics')]
            if observed:
                best = max(metrics['strict_passes'] for _, metrics in observed)
                leaders = [key for key, metrics in observed if metrics['strict_passes'] == best]
                names = {artifact['key']: artifact['label'] for artifact in fresh.get('artifacts', [])}
                if len(leaders) == 1:
                    finding = f"Astra round 4: {numeric(r4['strict_passes'])}/32 strict passes; fresh pristine: {numeric(pristine['strict_passes'])}/32. {names.get(leaders[0], leaders[0])} leads at {numeric(best)}/32. {completed_resources}/12 resource sessions complete. Exposed development data; no untouched confirmation."
        else:
            finding = f"The fresh six-version comparison is {fresh.get('status', 'pending').replace('_', ' ')}. Historical results remain separate."
    html = replace_region(html, "CURRENT_FINDING", escape(finding))
    harness_path = args.site / "harness.json"
    joint_path = args.site / "joint-comparison.json"
    joint_source = args.joint_report or (joint_path if joint_path.exists() else None)
    joint = public_joint_report(json.loads(joint_source.read_text())) if joint_source else None
    if harness_path.exists():
        harness = json.loads(harness_path.read_text())
        harness["status"] = "implemented" if joint and joint["arms"] else "implemented_not_run"
        harness["evaluation"]["status"] = joint["status"] if joint else "not_run"
        harness["evaluation"]["results"] = "joint-comparison.json" if joint else None
        html = replace_region(html, "HARNESS_CONTENT", harness_markup(harness, joint))
        html = replace_region(html, "HARNESS_DATA", script_json("harness-data", harness))
    outputs = {data_path: json.dumps(data, indent=2) + "\n", methods_path: json.dumps(methods, indent=2) + "\n", html_path: html}
    if harness_path.exists():
        outputs[harness_path] = json.dumps(harness, indent=2) + "\n"
    if joint:
        outputs[joint_path] = json.dumps(joint, indent=2, allow_nan=False) + "\n"
    if resource:
        outputs[args.site / "resource-study.json"] = json.dumps(resource, indent=2) + "\n"
    if fresh:
        outputs[args.site / "fresh-comparison.json"] = json.dumps(fresh, indent=2) + "\n"
    if data.get('fresh_maintenance'):
        outputs[args.site / 'fresh-maintenance.json'] = json.dumps(data['fresh_maintenance'], indent=2) + '\n'
    snapshot = f'historical: {len(evaluated)} Astra candidates, {len((resource or {}).get("sessions", []))} resource sessions'
    if fresh:
        quality_done = sum(row.get('status') == 'completed' for row in fresh.get('quality', {}).get('artifacts', []))
        resource_done = sum(row.get('status') == 'completed' and (row.get('summary') or {}).get('status') == 'complete' for row in fresh.get('resource', {}).get('sessions', []))
        snapshot += f'; fresh: {quality_done}/6 quality runs, {resource_done}/12 resource sessions'
    if args.check:
        stale = [str(path.relative_to(REPOSITORY)) for path, text in outputs.items() if not path.exists() or path.read_text() != text]
        if stale:
            raise SystemExit("Stale site evidence: " + ", ".join(stale))
        print(f'PASS: embedded/static evidence synchronized ({snapshot})')
    else:
        for path, text in outputs.items():
            path.write_text(text)
        print(f'Updated research snapshot ({snapshot})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPOSITORY / "reports/ifeval-browser-v2-collaboration-run/result.json")
    parser.add_argument("--resource-report", type=Path, default=REPOSITORY / "reports/resource-all-edits-v1/result.json")
    parser.add_argument("--fresh-report", type=Path, default=REPOSITORY / "reports/fresh-baseline-r4-v1/result.json")
    parser.add_argument("--fresh-maintenance", type=Path, default=REPOSITORY / "reports/fresh-baseline-r4-v1/maintenance.json")
    parser.add_argument("--joint-report", type=Path, help="Canonical mechanistic_study report; publishes whitelisted same-request observations separately")
    parser.add_argument("--site", type=Path, default=REPOSITORY / "site")
    parser.add_argument("--source-url", help="Public V2 report link; existing pin is retained by default")
    parser.add_argument("--resource-source-url", help="Public resource JSON link")
    parser.add_argument("--fresh-source-url", help="Immutable public fresh-comparison JSON link")
    parser.add_argument("--fresh-maintenance-source-url", help="Immutable public cache-maintenance evidence link")
    parser.add_argument("--repository-base", help="Public blob/COMMIT/ prefix for newly published methodology and sources")
    parser.add_argument("--check", action="store_true")
    update(parser.parse_args())


if __name__ == "__main__":
    main()
