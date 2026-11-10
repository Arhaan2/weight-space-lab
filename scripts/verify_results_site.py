#!/usr/bin/env python3
"""Check the public site's numerical evidence against the saved public reports.

This is read-only: it performs no grading, inference, or network requests.
"""

from __future__ import annotations

import json
import hashlib
import base64
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def verify_methodology(data: dict, equal) -> None:
    """Keep the interactive edit map and embedded offline datasets evidence-bound."""
    methods = json.loads((ROOT / "site/research-methodology.json").read_text())
    html = (ROOT / "site/index.html").read_text()
    for identity, expected in (("research-data", data), ("methodology-data", methods)):
        match = re.search(r'<script id="' + identity + r'" type="application/json">(.*?)</script>',
                          html, re.DOTALL)
        if match is None:
            raise ValueError(f"Missing embedded {identity}")
        equal(json.loads(match.group(1)), expected, f"offline {identity}")
    edits = methods["edit_operations"]
    equal(edits["pristine"]["operations"], [], "pristine edit map")
    equal(edits["pristine"]["changed_source_entries"], 0, "pristine changed coordinates")
    for key, arm in (("pristine", "baseline"), ("v1", "candidate")):
        equal(edits[key]["artifact_sha256"], data["v1"]["heldout"]["arms"][arm]["deployed_artifact_sha256"],
              f"{key} methods artifact identity")
    for row in data["v2"]["execution"]["rounds"]:
        if not row.get("metrics"):
            continue
        key = f"v2-r{row['round']}"
        equal(edits[key]["artifact_sha256"], row["artifact_sha256"], f"{key} methods identity")
        compact = []
        for operation in row["weight_operations"]:
            columns = operation["columns"]
            equal(columns, list(range(min(columns), max(columns) + 1)), f"{key} contiguous edit")
            compact.append({"op": operation["op"], "tensor": operation["tensor"],
                            "columns_inclusive": [min(columns), max(columns)], "factor": operation["factor"]})
        equal(edits[key]["operations"], compact, f"{key} edit map operations")
        equal(edits[key]["changed_source_entries"], 122880, f"{key} changed coordinates")
    for item in methods["sources"]:
        if not item["url"].startswith("https://github.com/Arhaan2/weight-space-lab/blob/"):
            raise ValueError("Methodology sources must resolve from the public Pages site")
        if not re.search(r"/blob/[0-9a-f]{40}/", item["url"]):
            raise ValueError("Methodology sources must be pinned to immutable commits")
        checks_label = item["label"]
        equal(bool(checks_label), True, "named methodology source")


def verify_resource_study(data: dict, v1: dict, equal) -> None:
    """Recompute public resource aggregates independently of the exporter/UI."""
    path = ROOT / "reports/resource-all-edits-v1/result.json"
    if not path.exists():
        if data.get("resource_study") is not None:
            raise ValueError("Resource study lacks its canonical public report")
        return
    report = json.loads(path.read_text())
    equal(data.get("resource_study"), report, "resource public report")
    equal(json.loads((ROOT / "site/resource-study.json").read_text()), report,
          "standalone resource evidence download")
    manifest_path = path.with_name("source-manifest.json")
    if report["status"] == "complete" and not manifest_path.exists():
        raise ValueError("Completed study lacks the audited public source manifest")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        equal(manifest["study_sha256"], report["study_sha256"], "public source study binding")
        equal(len(manifest["files"]), 42 if report.get("continuation") else 41,
              "public frozen source-file denominator")
        for name, digest in manifest["files"].items():
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT) or Path(name).is_absolute():
                raise ValueError("Source manifest contains a path outside the repository")
            equal(hashlib.sha256(source.read_bytes()).hexdigest(), digest, f"frozen source {name}")
        if report.get("continuation"):
            equal(manifest["files"]["scripts/continue_resource_followup.py"],
                  report["continuation"]["driver_sha256"], "frozen continuation driver")
            equal(manifest["continuation_sha256"], report["continuation"]["identity"],
                  "frozen continuation source binding")
    if report.get("continuation"):
        continuation = report["continuation"]
        partial_path = path.with_name("pre-continuation-result.json")
        equal(hashlib.sha256(partial_path.read_bytes()).hexdigest(),
              continuation["preserved_partial_result_sha256"], "preserved pre-continuation report bytes")
        partial = json.loads(partial_path.read_text())
        equal(partial["status"], "stopped", "original resource stop")
        equal(partial["actual_usage"]["completed_sessions"], 6, "adopted resource session denominator")
        equal(report["sessions"][:6], partial["sessions"][:6], "all six original observations preserved")
        equal(continuation["inherited_completed_positions"], list(range(1, 7)), "adopted resource positions")
        equal(continuation["remaining_positions"], [7, 8, 9, 10], "unused continuation positions")
        equal(continuation["replaced_completed_sessions"], 0, "no replaced observations")
        equal(continuation["model_dispatch_retries"], 0, "no continued model retries")
        equal(continuation["original_calendar_and_total_limits_unchanged"], True, "unchanged continuation limits")
        equal(continuation["scientific_study_sha256"], report["study_sha256"], "unchanged scientific study")
        equal(report["uninterrupted"], False, "documented interrupted schedule")
        equal(report["matched_current_environment"], report["matched_environment"], "matched runtime scope")
        equal(report["actual_usage"]["preserved_failed_pre_dispatch_verifications"], 1,
              "preserved integrity-preflight failure")
        dependency = json.loads(path.with_name("dependency-manifest.json").read_text())
        equal(dependency["study_sha256"], report["study_sha256"], "dependency study binding")
        for key in ("distribution", "version", "archive_sha256", "installed_record_sha256",
                    "payload_files", "overlay_payload_bytes", "installed_record_matches",
                    "versions_changed", "installed_environment_modified"):
            equal(dependency[key], continuation["dependency_recovery"][key], f"dependency {key}")
        equal(len(dependency["files"]), dependency["payload_files"], "dependency payload denominator")
        equal(sum(row["size_bytes"] for row in dependency["files"].values()),
              dependency["overlay_payload_bytes"], "dependency payload byte accounting")
        matches = 0
        for name, row in dependency["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Dependency manifest contains a non-relative member")
            equal(bool(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])), True, "dependency payload SHA-256")
            for key in ("installed_record_hash", "wheel_record_hash"):
                if row[key]:
                    algorithm, encoded = row[key].split("=", 1)
                    equal(algorithm, "sha256", "dependency RECORD algorithm")
                    equal(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex(),
                          row["sha256"], "dependency RECORD proof")
            matches += bool(row["installed_record_hash"])
        equal(matches, dependency["installed_record_matches"], "dependency installed RECORD matches")
    keys = ["pristine", "v1", "v2-r1", "v2-r2", "v2-r3"]
    order = keys + list(reversed(keys))
    equal(report["protocol"]["session_order"], order, "resource mirrored order")
    standard = json.loads((ROOT / "configs/browser_resource_comparison.json").read_text())
    equal(report["protocol"]["settings"], standard["resolved_standard"]["preset"]["settings"],
          "original standard workload settings")
    artifacts = {row["key"]: row for row in report["artifacts"]}
    equal(sorted(artifacts), sorted(keys), "resource artifact set")
    expected = {
        "pristine": v1["heldout_arms"]["baseline"]["artifact_sha256"],
        "v1": v1["heldout_arms"]["candidate"]["artifact_sha256"],
        **{f"v2-r{row['round']}": row["artifact_sha256"]
           for row in data["v2"]["execution"]["rounds"] if row.get("metrics")},
    }
    for key in keys:
        equal(artifacts[key]["artifact_sha256"], expected[key], f"{key} resource identity")
    equal(artifacts["pristine"]["weight_operations"], [], "pristine resource intervention")
    for row in data["v2"]["execution"]["rounds"]:
        if row.get("metrics"):
            equal(artifacts[f"v2-r{row['round']}"]["weight_operations"], row["weight_operations"],
                  f"round {row['round']} resource intervention")
    equal(artifacts["v1"]["weight_operations"], [
        {"op": "scale_columns", "tensor": f"model.language_model.layers.{layer}.mlp.down_proj.weight",
         "columns": list(range(60)), "factor": 0.75} for layer in (15, 19)
    ], "canonical V1 resource intervention")
    libraries = {row["model_library_sha256"] for row in artifacts.values()}
    equal(len(libraries), 1, "shared resource model library")
    equal(all(re.fullmatch(r"[0-9a-f]{64}", value or "") is not None for value in libraries),
          True, "resource model library hash")
    sessions = report["sessions"]
    equal([r["position"] for r in sessions], list(range(1, 11)), "resource positions")
    equal([r["artifact_key"] for r in sessions], order, "resource session artifacts")

    def number(actual, expected_value, label):
        if expected_value is None:
            equal(actual, None, label)
        elif isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{label}: missing numeric observation")
        else:
            equal(math.isclose(actual, expected_value, rel_tol=1e-9, abs_tol=1e-9), True, label)

    for row in sessions:
        summary = row.get("summary")
        if not summary:
            continue
        label = f"resource session {row['position']}"
        measured = [r for r in row["requests"] if not r["warmup"]]
        warmups = [r for r in row["requests"] if r["warmup"]]
        equal(len(measured), summary["measured_requests"], f"{label} measured requests")
        equal(len(warmups), summary["warmups"], f"{label} warmups")
        if summary["status"] != "complete":
            continue
        equal(len(measured), 60, f"{label} complete denominator")
        equal(len(warmups), 1, f"{label} complete warmup count")
        equal(len({r["request_index"] for r in measured}), 60, f"{label} unique requests")
        equal(all(r["valid"] for r in measured + warmups), True, f"{label} valid requests")
        scored_tokens = sum(r["output_tokens"] for r in measured)
        warmup_tokens = sum(r["output_tokens"] for r in warmups)
        response_ms = sum(r["response_ms"] for r in measured)
        equal(summary["scored_output_tokens"], scored_tokens, f"{label} scored tokens")
        equal(summary["warmup_output_tokens"], warmup_tokens, f"{label} warmup tokens")
        number(summary["response_ms_sum"], response_ms, f"{label} response sum")
        number(summary["mean_response_ms"], response_ms / 60, f"{label} mean response")
        number(summary["full_response_tokens_per_second"], scored_tokens / (response_ms / 1000)
               if response_ms else None, f"{label} full-response throughput")
        number(summary["response_ms_per_output_token"], response_ms / scored_tokens
               if scored_tokens else None, f"{label} response per output token")
        equal(summary["natural_completions"], sum(r["finish_reason"] == "stop" for r in measured),
              f"{label} natural completions")
        equal(summary["truncations"], sum(r["finish_reason"] == "length" for r in measured),
              f"{label} truncations")

    usage = report["actual_usage"]
    equal(usage["completed_sessions"], sum(r["status"] == "completed" for r in sessions),
          "resource completed session accounting")
    equal(usage["attempted_sessions"], sum(bool(r.get("observed")) for r in sessions)
          + usage["unresolved_attempted_sessions"],
          "resource attempted session accounting")
    equal(usage["frontier_calls"], 0, "resource editor calls")
    equal(usage["quality_generations"], 0, "resource quality reruns")
    if report["status"] == "complete":
        equal(report["matched_environment"], True, "completed matched resource environment")
        equal(usage["completed_sessions"], 10, "full resource study denominator")
        for key, expected_total in (("engine_load_attempts", 10), ("warmup_dispatches", 10),
                                    ("scored_dispatches", 600), ("generation_dispatches", 610),
                                    ("generation_retries", 0), ("failed_target_sequences", 0),
                                    ("reserved_target_tokens", 38560)):
            equal(usage[key], expected_total, f"resource {key}")
        equal(usage["sampled_tokens_including_warmups"],
              sum(r["summary"]["scored_output_tokens"] + r["summary"]["warmup_output_tokens"]
                  for r in sessions), "resource actual sampled tokens")
    equal(sorted(report["comparisons"]), sorted(keys[1:]), "resource comparisons set")

    for key in keys:
        rows = [r for r in sessions if r["artifact_key"] == key]
        aggregate = report["aggregates"][key]
        complete = [r for r in rows if r["status"] == "completed" and r.get("summary", {})
                    and r["summary"]["status"] == "complete"]
        equal(aggregate["n_scheduled"], 2, f"{key} scheduled replications")
        equal(aggregate["n_completed"], len(complete), f"{key} complete replications")
        for metric, entry in aggregate["metrics"].items():
            values = [r["summary"].get(metric) if r in complete else None for r in rows]
            equal(entry["values"], values, f"{key} {metric} session values")
            available = len(complete) == 2 and all(v is not None for v in values)
            number(entry["mean"], sum(values) / 2 if available else None, f"{key} {metric} mean")
            number(entry["min"], min(values) if available else None, f"{key} {metric} minimum")
            number(entry["max"], max(values) if available else None, f"{key} {metric} maximum")
    for key, comparison in report["comparisons"].items():
        if key not in keys[1:]:
            raise ValueError("Unexpected resource comparison")
        for metric, entry in comparison["metrics"].items():
            base = report["aggregates"]["pristine"]["metrics"][metric]["mean"]
            candidate = report["aggregates"][key]["metrics"][metric]["mean"]
            number(entry["baseline_mean"], base, f"{key} {metric} baseline mean")
            number(entry["candidate_mean"], candidate, f"{key} {metric} candidate mean")
            delta = candidate - base if report["matched_environment"] and candidate is not None and base is not None else None
            number(entry["absolute_delta"], delta, f"{key} {metric} delta")
            number(entry["relative_percent"], delta / base * 100 if delta is not None and base else None,
                   f"{key} {metric} relative change")


def main() -> None:
    site = ROOT / "site"
    data = json.loads((site / "results.json").read_text())
    source = json.loads(
        (ROOT / "reports/ifeval-browser-v1-final/result.json").read_text()
    )
    checks = 0

    def equal(actual: object, expected: object, label: str) -> None:
        nonlocal checks
        if actual != expected:
            raise ValueError(f"{label}: site data differs from saved public evidence")
        checks += 1

    equal(data["v1"]["status"], source["status"], "V1 status")
    equal(data["v1"]["heldout"]["metrics"], source["metrics"], "held-out metrics")
    equal(
        data["v1"]["heldout"]["paired_prompt_outcomes"],
        source["paired_prompt_outcomes"],
        "paired outcomes",
    )
    for arm in ("baseline", "candidate"):
        development = source["development"][arm]["summary"]
        expected_development = {
            "strict": development["metrics"]["prompt_level_strict_accuracy"]["passed"],
            "loose": development["metrics"]["prompt_level_loose_accuracy"]["passed"],
            "total": development["scheduled_prompts"],
            "output_tokens": development["known_generated_tokens"],
            "generation_seconds": development["generation_seconds"],
        }
        equal(data["v1"]["development"][arm], expected_development, f"{arm} development")
        heldout = source["heldout_arms"][arm]
        published = data["v1"]["heldout"]["arms"][arm]
        equal(published["accounting"], heldout["accounting"], f"{arm} accounting")
        equal(
            published["allocation_sampled_maxima"],
            heldout["resources"]["worker_sampled_maxima_across_sessions"],
            f"{arm} allocation scope",
        )
        for identity, source_key in (
            ("source_checkpoint_sha256", "source_checkpoint_sha256"),
            ("deployed_artifact_sha256", "artifact_sha256"),
        ):
            equal(published[identity], heldout[source_key], f"{arm} {identity}")

    equal(data["v2"]["untouched_final_prompts"], 0, "V2 final-data availability")
    equal(
        data["v2"]["qualification"],
        {"strict_passes_greater_than": 16, "output_tokens_at_most": 13941},
        "V2 frozen qualification",
    )
    execution = data["v2"].get("execution")
    if execution is not None:
        public_v2 = json.loads(
            (ROOT / "reports/ifeval-browser-v2-collaboration-run/result.json").read_text()
        )
        equal(execution, public_v2, "V2 public execution report")
        equal(data["v2"]["status"], execution["status"], "V2 status")
        equal(data["v2"]["resource_status"], execution["resource"]["status"], "V2 resource status")
        for row in execution["rounds"]:
            metrics = row.get("metrics")
            if metrics is None:
                continue
            qualifies = bool(
                row["status"] == "completed"
                and metrics["technically_valid"]
                and metrics["unresolved_grader_errors"] == 0
                and row["deployment_limits_satisfied"]
                and row["memory_limits_satisfied"]
                and metrics["strict_passes"] > execution["baseline"]["strict_passes"]
                and metrics["scored_output_tokens"] <= execution["baseline"]["scored_output_tokens"]
            )
            equal(row["qualifies"], qualifies, f"round {row['round']} qualification")

    verify_resource_study(data, source, equal)
    verify_methodology(data, equal)
    if (ROOT / "CITATION.cff").exists():
        equal((site / "citation.cff").read_text(), (ROOT / "CITATION.cff").read_text(), "citation metadata")

    # Site source links must resolve to the evidence revision, not stale main.
    for label, url in data["sources"].items():
        if "/blob/main/" in url or "/tree/main/" in url:
            raise ValueError(f"{label}: pin the public evidence link to a commit")
        checks += 1

    allowed = {".html", ".css", ".js", ".json", ".md", ".txt", ".woff2", ".ttf", ".svg", ".csv", ".cff"}
    forbidden = re.compile(
        r"/Users/|/private/(?:tmp|var)/|(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]+"
        r"|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"
    )
    for path in site.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Site cannot contain a symlink: {path.relative_to(site)}")
        if not path.is_file():
            continue
        if path.name != ".nojekyll" and path.suffix not in allowed:
            raise ValueError(f"Unexpected publication file: {path.relative_to(site)}")
        if path.suffix not in {".woff2", ".ttf"} and forbidden.search(path.read_text()):
            raise ValueError(f"Private material in {path.relative_to(site)}")
        checks += 1

    print(f"PASS: {checks} public evidence and publication checks; no model execution")


if __name__ == "__main__":
    main()
