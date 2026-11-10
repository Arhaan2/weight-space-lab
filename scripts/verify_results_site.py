#!/usr/bin/env python3
"""Check the public site's numerical evidence against the saved public reports.

This is read-only: it performs no grading, inference, or network requests.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


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

    # Site source links must resolve to the evidence revision, not stale main.
    for label, url in data["sources"].items():
        if "/blob/main/" in url or "/tree/main/" in url:
            raise ValueError(f"{label}: pin the public evidence link to a commit")
        checks += 1

    allowed = {".html", ".css", ".js", ".json", ".md", ".txt", ".woff2", ".ttf", ".svg"}
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
