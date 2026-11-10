#!/usr/bin/env python3
"""Copy the public V2 report into the static results site; performs no inference."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "https://github.com/Arhaan2/weight-space-lab/blob/codex/results-site/reports/ifeval-browser-v2-collaboration-run/README.md"


def replace_region(document: str, name: str, markup: str) -> str:
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError(f"Expected one pair of {name} markers")
    before, remainder = document.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + start + "\n" + markup + "\n        " + end + after


def measurements_row(label: str, metrics: dict, qualification: str) -> str:
    denominator = metrics["denominator"]
    if not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("Invalid development denominator")
    for field in ("strict_passes", "loose_passes"):
        if not isinstance(metrics[field], int) or not 0 <= metrics[field] <= denominator:
            raise ValueError(f"Invalid {field}")
    tokens = metrics["scored_output_tokens"]
    if not isinstance(tokens, int) or tokens < 0:
        raise ValueError("Invalid output-token count")
    return (
        f'              <tr><th scope="row">{escape(label)}</th>'
        f'<td data-label="Strict">{metrics["strict_passes"]}/{denominator}</td>'
        f'<td data-label="Loose">{metrics["loose_passes"]}/{denominator}</td>'
        f'<td data-label="Output tokens">{tokens:,}</td>'
        f'<td data-label="Qualifies">{escape(qualification)}</td></tr>'
    )


def numeric(value, decimals: int = 0, divisor: float = 1) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid numeric resource observation")
    return f"{value / divisor:,.{decimals}f}"


def resource_markup(resource: dict) -> str:
    sessions = [row for row in resource.get("sessions", []) if row.get("summary") is not None]
    if not sessions:
        return ""
    rows, telemetry = [], []
    for session in sessions:
        summary = session["summary"]
        arm = {"pristine_baseline": "Original (A)", "selected_astra": "Edited (B)"}.get(session["arm"], session["arm"])
        name = f'Session {session["position"]}: {arm}'
        requests = f'{numeric(summary.get("measured_requests"))} / {numeric(summary.get("scheduled_requests"))}'
        rows.append(
            f'<tr><th scope="row">{escape(name)}<span class="session-outcome">{escape(summary["status"].replace("_", " ").capitalize())}</span></th>'
            f'<td data-label="Requests">{requests}</td>'
            f'<td data-label="Output tokens">{numeric(summary.get("scored_output_tokens"))}</td>'
            f'<td data-label="Response sum (s)">{numeric(summary.get("response_ms_sum"), 2, 1000)}</td>'
            f'<td data-label="Sampled footprint (GB)">{numeric(summary.get("native_peak_footprint_gb"), 3)}</td></tr>'
        )
        cells = []
        for label, field, coverage in (
            ("CPU capacity (%)", "cpu_capacity_percent", "cpu_coverage"),
            ("GPU time (ms/s)", "gpu_time_ms_per_second", "gpu_coverage"),
            ("Host power (W)", "host_power_watts", "power_coverage"),
        ):
            observed = numeric(summary.get(field), 2)
            fraction = summary.get(coverage)
            coverage_label = "Coverage unavailable" if fraction is None else f'{numeric(fraction, 1, .01)}% coverage'
            cells.append(f'<td data-label="{label}">{observed}<span class="coverage-note">{coverage_label}</span></td>')
        telemetry.append(f'<tr><th scope="row">{escape(name)}</th>{"".join(cells)}</tr>')
    matching = "Matched environment verified." if resource.get("matched_environment") is True else "Matched environment is not established."
    return f'''          <div class="measured-resources">
            <h3>Measured browser usage</h3>
            <p>{len(sessions)} recorded sessions of {numeric(resource.get("scheduled_sessions"))} scheduled. {numeric(resource.get("complete_sessions"))} complete. {matching}</p>
            <table class="development-table">
              <caption>Original Browser LLM Bench standard workload. Response time is summed over measured requests, excluding warm-ups. Native process footprint is the session’s sampled peak in decimal GB.</caption>
              <thead><tr><th scope="col">Session</th><th scope="col">Requests</th><th scope="col">Output tokens</th><th scope="col">Response sum (s)</th><th scope="col">Sampled footprint (GB)</th></tr></thead>
              <tbody>{"".join(rows)}</tbody>
            </table>
            <details class="native-telemetry"><summary>CPU, GPU and power coverage</summary>
              <table class="development-table">
                <caption>Native collectors’ CPU capacity and GPU time, plus host-wide power. Coverage is the share of active request time covered by eligible sampling intervals. Missing observations remain unavailable.</caption>
                <thead><tr><th scope="col">Session</th><th scope="col">CPU capacity (%)</th><th scope="col">GPU time (ms/s)</th><th scope="col">Host power (W)</th></tr></thead>
                <tbody>{"".join(telemetry)}</tbody>
              </table>
            </details>
            <p>Native process footprint, GPUBuffer allocation descriptors and WASM address space are overlapping or different counters and are not added. Host power includes unrelated activity. Two sessions per artifact are a small pilot, not proof of resource equality.</p>
          </div>'''


def update(report_path: Path, site: Path, source_url: str | None, check: bool) -> None:
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 1 or report.get("kind") != "sanitized_v2_collaboration_result":
        raise ValueError("Expected the sanitized public V2 report")
    if report["untouched_final_prompts"] != 0:
        raise ValueError("Review the site copy before adding new final data")
    data_path, html_path = site / "results.json", site / "index.html"
    data = json.loads(data_path.read_text())
    evaluated = [row for row in report["rounds"] if row.get("metrics") is not None]
    count = len(evaluated)
    qualifying = sum(row.get("qualifies") is True for row in evaluated)
    resource_status = report["resource"]["status"]
    resource_label = {
        "pending": "Not run",
        "pristine_retained_no_comparison": "Skipped: pristine retained",
        "complete": "Complete",
        "completed": "Complete",
        "pilot_complete": "Pilot complete",
        "incomplete": "Incomplete",
        "failed": "Stopped",
    }.get(resource_status, resource_status.replace("_", " ").capitalize())
    if resource_status == "pristine_retained_no_comparison":
        description = f"{count} candidate development evaluations are complete. No edit qualified; the original model was retained. The combined quality and resource goal was not demonstrated."
    elif count:
        noun, verb = ("candidate", "has") if count == 1 else ("candidates", "have")
        qualification = ("1 qualifies." if qualifying == 1 else f"{qualifying} qualify.") if qualifying else "No candidate qualifies."
        description = f"{count} {noun} {verb} measured development results. {qualification}"
        if report["status"] == "in_progress":
            description += " The bounded study is still in progress."
        elif report["status"] == "halted":
            description += " Execution has stopped; partial evidence is retained."
    else:
        description = "No candidate development evaluation has completed. No new result is claimed."
    limitation = "No verified untouched final prompts remain. V2 quality results are development evidence, without fresh held-out confirmation."
    url = source_url or data["sources"].get("v2_execution") or DEFAULT_SOURCE
    if not url.startswith("https://github.com/Arhaan2/weight-space-lab/"):
        raise ValueError("The evidence link must identify this public repository")
    data["sources"]["v2_execution"] = url
    data["v2"].update(
        status=report["status"], description=description, resource_status=resource_status,
        resource_status_label=resource_label, limitation=limitation, execution=report,
    )
    data["updated"] = datetime.fromisoformat(report["updated_at_utc"]).astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat()
    status_markup = f'''          <div class="next-status reveal" id="v2-status" data-ui>
            <h3>V2 evidence</h3>
            <p id="v2-description">{escape(description)}</p>
            <dl class="status-list"><div><dt>Development tasks</dt><dd>{report["baseline"]["denominator"]}</dd></div><div><dt>Untouched final prompts</dt><dd>0</dd></div><div><dt>Standard resource comparison</dt><dd id="v2-resource-status">{escape(resource_label)}</dd></div></dl>
            <p id="v2-limitation">{escape(limitation)}</p>
          </div>'''
    rows = [measurements_row("Original baseline", report["baseline"], "Reference")]
    rows.extend(measurements_row(f'Astra round {row["round"]}', row["metrics"], "Yes" if row.get("qualifies") is True else "No") for row in evaluated)
    failure = report.get("preserved_pre_dispatch_failure", {})
    failure_note = ""
    if failure.get("maximum_real_candidate_evaluations") is not None:
        failure_note = (f'<p class="reveal">The initial browser attempt stopped before any model load and remains counted. '
                        f'The preserved stop limited the continuation to {int(failure["maximum_real_candidate_evaluations"])} real candidate evaluations.</p>')
    baseline_browser = report.get("baseline_environment", {}).get("chrome_version")
    candidate_browsers = sorted({row.get("environment", {}).get("chrome_version") for row in evaluated} - {None})
    if baseline_browser and candidate_browsers:
        browser_note = f'The saved baseline used Chrome {baseline_browser}; evaluated candidates used Chrome {", ".join(candidate_browsers)}.'
        if any(browser != baseline_browser for browser in candidate_browsers):
            browser_note += " Browser changes may also affect the development comparison."
        browser_note += " The separate resource protocol matches the browser for both artifacts."
    else:
        browser_note = "Browser-version provenance is incomplete; a matched-browser development comparison is not established."
    next_study = (
        '<p class="reveal"><strong>For a subsequent quality claim,</strong> a new study would need a baseline on the same current browser and a verified untouched final set.</p>'
        if report["status"] == "complete" else ""
    )
    results_markup = f'''        <div class="v2-measurements">
          <h3 class="reveal">Measured development</h3>
          <p class="reveal">V2 reuses the saved original baseline and the same 32 prompts for every evaluated candidate. These are development-only measurements.</p>
          <table class="development-table reveal" data-ui>
            <caption>Strict and loose prompt passes, sampled output tokens, and eligibility under the frozen selection rule.</caption>
            <thead><tr><th scope="col">Artifact</th><th scope="col">Strict</th><th scope="col">Loose</th><th scope="col">Output tokens</th><th scope="col">Qualifies</th></tr></thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
          </table>
          <p class="reveal">{escape(browser_note)}</p>
          {failure_note}
          <p class="v2-provenance reveal">The editor was a fresh-context gpt-6-astra collaboration agent. Ultra effort, immutable provider identity and technical isolation were not verified. <a href="{escape(url, quote=True)}">Read the execution report</a>.</p>
          {next_study}
{resource_markup(report["resource"])}
        </div>'''
    html = replace_region(html_path.read_text(), "V2_STATUS", status_markup)
    html = replace_region(html, "V2_RESULTS", results_markup)
    protocol_url = escape(data["sources"]["v2_protocol"], quote=True)
    if resource_status == "pristine_retained_no_comparison":
        resource_plan = f'''        <div class="resource-plan reveal">
          <div><h3>Resource comparison skipped</h3><p>No edit passed the frozen development gate. The original model was retained.</p></div>
          <p>{escape(report["resource"]["reason"])}</p>
        </div>
        <p class="source-note reveal">The A–B–B–A resource comparison was conditional on an eligible edit. No browser resource improvement or equality is established. <a href="{protocol_url}">Read the protocol</a>.</p>'''
    elif report["status"] == "halted" and not report["resource"].get("sessions"):
        resource_plan = f'''        <div class="resource-plan reveal">
          <div><h3>Resource comparison not reached</h3><p>Execution stopped. No standard-workload resource measurements are available.</p></div>
          <p>The comparison requires a selected eligible edit and valid execution capacity. <a href="{protocol_url}">Read the protocol</a>.</p>
        </div>'''
    else:
        resource_copy = ("A qualifying edit is required. The plan calls for four sessions on the original Browser LLM Bench standard workload, with 60 measured requests per session."
                         if resource_status == "pending" else
                         "The prespecified order compares the original and selected edit over four sessions. The measured table above records the actual completed work.")
        resource_plan = f'''        <div class="resource-plan reveal">
          <div><h3>The resource plan</h3><p>{resource_copy}</p></div>
          <div class="abba" aria-label="Prespecified conditional session order: original A, edited B, edited B, original A"><span>A<small>Original</small></span><span>B<small>Edited</small></span><span>B<small>Edited</small></span><span>A<small>Original</small></span></div>
        </div>
        <p class="source-note reveal">The protocol matches browser, device, quantization, prompts and settings. Native CPU, process-memory and GPU signals retain their own scope. Two sessions per artifact form a small pilot. <a href="{protocol_url}">Read the protocol</a>.</p>'''
    html = replace_region(html, "V2_RESOURCE_PLAN", resource_plan)
    serialized = json.dumps(data, indent=2) + "\n"
    if check:
        if data_path.read_text() != serialized or html_path.read_text() != html:
            raise SystemExit("Site V2 evidence is stale; run scripts/update_results_site.py")
        print(f"PASS: site matches public V2 report ({count} evaluated candidates)")
    else:
        data_path.write_text(serialized)
        html_path.write_text(html)
        print(f"Updated site with {count} measured candidate result(s); resource status: {resource_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPOSITORY / "reports/ifeval-browser-v2-collaboration-run/result.json")
    parser.add_argument("--site", type=Path, default=REPOSITORY / "site")
    parser.add_argument("--source-url", help="Public execution-report link; preserves an existing pinned link when omitted")
    parser.add_argument("--check", action="store_true", help="Read-only verification that static HTML and JSON match the saved report")
    args = parser.parse_args()
    update(args.report, args.site, args.source_url, args.check)


if __name__ == "__main__":
    main()
