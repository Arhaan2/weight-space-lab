#!/usr/bin/env python3
"""Check public V1 closeout arithmetic; no model, grader, or ledger imports.

Arhaan Aggarwal (Arhaan2). --write creates only a new derived result.json;
default operation is read-only. Frozen confidence intervals are copied, never
recomputed. This verifies public consistency, not unreleased raw evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FINAL = REPORTS / "ifeval-browser-v1-final"
COMBINED = "ifeval-browser-v1-heldout-continuation-combined"
CONCLUSION = ("Completed experiment; inconclusive held-out compliance result. "
              "Strict prompt: 70/128 candidate versus 69/128 baseline; "
              "+0.78125 percentage points; frozen 95% interval [-5.47, +7.03] "
              "percentage points. Improvement and equivalence are unestablished.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_completed(source):
    require(source["status"] == "complete_manually_continued"
            and source["complete_paired_evaluation"], "comparison is not complete")
    expected = {"prompt_strict": (69, 70, 128), "prompt_loose": (74, 77, 128),
                "instruction_strict": (106, 108, 173),
                "instruction_loose": (112, 116, 173)}
    for key, (baseline, candidate, denominator) in expected.items():
        metric = source["metrics"][key]
        for arm, count in (("baseline", baseline), ("candidate", candidate)):
            require(metric[arm]["numerator"] == count
                    and metric[arm]["denominator"] == denominator,
                    "frozen numerator/denominator changed: " + key)
            require(isclose(metric[arm]["value"], count / denominator),
                    "reported ratio mismatch: " + key)
        require(isclose(metric["difference_percentage_points"],
                        100 * (candidate - baseline) / denominator),
                "reported effect mismatch: " + key)
        unit, mode = key.split("_")
        pairs = source["paired_" + unit + "_outcomes"][mode]
        require(sum(pairs[k] for k in ("improved", "regressed", "tied", "unresolved"))
                == denominator and pairs["unresolved"] == 0, "paired coverage mismatch")
        # Check saved paired counts; never manufacture wins/losses from net change.
        require(pairs["improved"] - pairs["regressed"] == candidate - baseline,
                "paired outcomes disagree with counts")
    require(source["metrics"]["prompt_strict"]["ci95_percentage_points"]
            == [-5.46875, 7.03125], "frozen primary interval changed")
    require(source["original_outputs_regraded"] == 0
            and source["original_records_rewritten"] == 0,
            "original evidence was reprocessed")
    sessions = source["sessions"]
    for arm, names in (("baseline", ("baseline",)),
                       ("candidate", ("candidate_interrupted", "candidate_continuation"))):
        accounting = source["arms"][arm]["accounting"]
        for key in ("natural_completions", "truncations", "scored_generated_tokens",
                    "scored_input_tokens", "scored_generation_seconds"):
            require(isclose(accounting[key], sum(sessions[n]["accounting"][key] for n in names)),
                    "session-to-arm mismatch: " + key)
        require(accounting["natural_completions"] + accounting["truncations"] == 128,
                "incomplete scored outputs")
    for key in ("engine_load_attempts", "warmup_dispatches", "scored_dispatches",
                "generation_dispatches", "known_generated_tokens_including_warmup",
                "reserved_generated_tokens", "charged_generated_tokens"):
        require(source["cumulative_usage"][key]
                == sum(s["usage"][key] for s in sessions.values()),
                "cumulative session mismatch: " + key)
    require(source["cumulative_usage"]["generation_requests_with_unknown_tokens"] == 0,
            "known full-study usage requires complete token measurements")


def build_result():
    read = lambda name: json.loads((REPORTS / name / "result.json").read_text())
    heldout = read(COMBINED)
    validate_completed(heldout)
    development = read("ifeval-browser-v1-candidate-recovery")
    baseline = read("ifeval-browser-v1")
    require(baseline["summary"] == development["baseline"]["summary"],
            "reused development baseline changed")
    budgets = [baseline["budget"], development["candidate"]["budget"]]
    usage = deepcopy(heldout["cumulative_usage"])
    for key, charged in (("automatic_retries", "retries"),
                         ("pending_requests", "pending_requests"),
                         ("limit_violations", "limit_violations")):
        usage[key] += sum(b["charged"][charged] for b in budgets)
    for budget in budgets:
        require(budget["observed"]["token_measurements"] == budget["generation_dispatches"],
                "development token usage is not fully observed")
    for key in ("engine_load_attempts", "warmup_dispatches", "scored_dispatches",
                "generation_dispatches"):
        usage[key] += sum(b[key] for b in budgets)
    for key, subkey in (("generated_tokens_including_warmup", "observed"),
                        ("known_generated_tokens_including_warmup", "observed"),
                        ("charged_generated_tokens", "charged"),
                        ("reserved_generated_tokens", "reserved")):
        usage[key] += sum(b[subkey]["generated_tokens"] for b in budgets)
    summaries = [development[a]["summary"] for a in ("baseline", "candidate")]
    usage["scored_generated_tokens"] += sum(s["known_generated_tokens"] for s in summaries)
    usage["scored_generation_seconds"] += sum(s["generation_seconds"] for s in summaries)
    usage["generation_seconds_including_warmup"] += sum(b["observed"]["elapsed_seconds"] for b in budgets)
    usage["active_load_through_cleanup_seconds"] += sum(b["first_load_through_cleanup_seconds"] for b in budgets)
    usage["warmup_generated_tokens"] = (usage["generated_tokens_including_warmup"]
                                          - usage["scored_generated_tokens"])
    # This field is unavailable in the baseline development public report.
    usage.pop("supervisor_seconds")
    usage.pop("new_weight_proposals")
    usage.pop("experimental_frontier_requests")
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
               for directory in sorted(REPORTS.glob("ifeval-browser-v1*"))
               if directory != FINAL
               for p in sorted(directory.rglob("*")) if p.is_file()}
    return {
        "schema_version": 1, "author": "Arhaan Aggarwal (Arhaan2)",
        "status": "completed_inconclusive", "conclusion": CONCLUSION,
        "source_report_files_sha256": sources,
        "metrics": heldout["metrics"],
        "paired_prompt_outcomes": heldout["paired_prompt_outcomes"],
        "paired_instruction_outcomes": heldout["paired_instruction_outcomes"],
        "analysis_specification": heldout["analysis_specification"],
        "heldout_arms": heldout["arms"],
        "heldout_sessions": heldout["sessions"],
        "heldout_cumulative_usage": heldout["cumulative_usage"],
        "heldout_timing": heldout["timing"],
        "development": {a: {k: development[a][k] for k in
                              ("summary", "budget", "load_ms", "resources")}
                        for a in ("baseline", "candidate")},
        "full_v1_browser_usage": usage,
        "full_v1_browser_scope": "Two development plus three held-out sessions; excludes editor, import, conversion, grading and earlier unrelated experiments.",
        "proposal_count": 1, "proposal_identity_and_usage_observed": False,
        "successful_import_count": 1, "conversion_stage_attempts": 2,
        "conversion_precheck_failures": 1, "conversion_subprocesses": 1,
        "browser_session_execution_failures": 1,
        "import_seconds": development["import"]["observed"]["elapsed_seconds"],
        "conversion": development["conversion"],
        "verification_scope": "Public report identity and additive consistency only; private raw-output/ledger audits remain separate. No grading, model calls or bootstrap reruns.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create a new closeout JSON; never overwrite")
    args = parser.parse_args()
    result = build_result()
    destination = FINAL / "result.json"
    if args.write:
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    require(json.loads(destination.read_text()) == result,
            "closeout differs from preserved reports")
    print("PASS: frozen metrics, paired coverage, session accounting, development reuse, and public source hashes")


if __name__ == "__main__":
    main()
