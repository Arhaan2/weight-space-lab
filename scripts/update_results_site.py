#!/usr/bin/env python3
"""Synchronize public evidence, embedded data and no-JavaScript research tables. No inference."""
from __future__ import annotations
import argparse
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLIC_BASE = "https://github.com/Arhaan2/weight-space-lab/blob/codex/results-site/"
ARTIFACTS = {"pristine": "Pristine", "v1": "V1 edit", "v2-r1": "Astra round 1", "v2-r2": "Astra round 2", "v2-r3": "Astra round 3"}


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
    content += f'<p><a href="results.json" download>Download complete results JSON</a> · <a href="{escape(data["sources"]["v1_report"], quote=True)}">V1 report</a> · <a href="{escape(data["sources"]["v2_execution"], quote=True)}">V2 report</a></p>'
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


def static_methods(methods, resource=None):
    def fields(values):
        return '<dl>' + ''.join(f'<div><dt>{escape(key.replace("_", " ")[:1].upper() + key.replace("_", " ")[1:])}</dt><dd>{escape(json.dumps(value) if isinstance(value, (dict, list)) else str(value))}</dd></div>' for key, value in values.items()) + '</dl>'
    blocks = []
    for study in methods.get("studies", []):
        sources = ' · '.join(f'<a href="{escape(source["url"], quote=True)}">{escape(source["label"])}</a>' for source in study.get("sources", []))
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
        prose = ' '.join(str(definition.get(key, '')) for key in ('definition', 'scope', 'coverage', 'limitations', 'missing'))
        telemetry.append(f'<div><dt>{escape(definition["label"])} · {escape(definition.get("unit", ""))}</dt><dd>{escape(prose)}</dd></div>')
    blocks.append('<details><summary>Telemetry definitions and coverage</summary><div class="method-body"><dl>' + ''.join(telemetry) + '</dl></div></details>')
    notes = methods.get("reproduction", []) + methods.get("preserved_failures", [])
    blocks.append('<details><summary>Reproduction and preserved failures</summary><div class="method-body"><ul>' + ''.join(f'<li>{escape(note)}</li>' for note in notes) + '</ul></div></details>')
    return '\n'.join(blocks)


def script_json(identifier, data):
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return f'<script id="{identifier}" type="application/json">{encoded}</script>'


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
    data["updated"] = datetime.fromisoformat(report["updated_at_utc"]).astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat()
    resource_updated = data.get("resource_study", {}).get("updated_at_utc")
    if resource_updated:
        data["updated"] = max(data["updated"], datetime.fromisoformat(resource_updated).astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat())
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
    resource = data.get("resource_study")
    if resource:
        for study in methods.get("studies", []):
            if study["id"] in ("resource", "resource-all-edits", "resource_all_edits", "resource-all-edits-v1"):
                study["status"] = resource["status"]
    html = replace_region(html_path.read_text(), "STATIC_EVIDENCE", static_evidence(data))
    html = replace_region(html, "RESEARCH_DATA", script_json("research-data", data))
    html = replace_region(html, "METHODOLOGY_DATA", script_json("methodology-data", methods))
    html = replace_region(html, "METHODS_DETAIL", static_methods(methods, resource))
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
    html = replace_region(html, "CURRENT_FINDING", escape(finding))
    outputs = {data_path: json.dumps(data, indent=2) + "\n", methods_path: json.dumps(methods, indent=2) + "\n", html_path: html}
    if resource:
        outputs[args.site / "resource-study.json"] = json.dumps(resource, indent=2) + "\n"
    if args.check:
        stale = [str(path.relative_to(REPOSITORY)) for path, text in outputs.items() if not path.exists() or path.read_text() != text]
        if stale:
            raise SystemExit("Stale site evidence: " + ", ".join(stale))
        print(f'PASS: embedded/static evidence synchronized ({len(evaluated)} candidates; {len((resource or {}).get("sessions", []))} resource sessions)')
    else:
        for path, text in outputs.items():
            path.write_text(text)
        print(f'Updated research snapshot ({len(evaluated)} candidates; {len((resource or {}).get("sessions", []))} resource sessions)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPOSITORY / "reports/ifeval-browser-v2-collaboration-run/result.json")
    parser.add_argument("--resource-report", type=Path, default=REPOSITORY / "reports/resource-all-edits-v1/result.json")
    parser.add_argument("--site", type=Path, default=REPOSITORY / "site")
    parser.add_argument("--source-url", help="Public V2 report link; existing pin is retained by default")
    parser.add_argument("--resource-source-url", help="Public resource JSON link")
    parser.add_argument("--repository-base", help="Public blob/COMMIT/ prefix for newly published methodology and sources")
    parser.add_argument("--check", action="store_true")
    update(parser.parse_args())


if __name__ == "__main__":
    main()
