#!/usr/bin/env python3
"""One explicit continuation after a proved zero-inference candidate launch failure.

This sidecar never edits the frozen study, package, original ledgers, or baseline.
Preparation and reporting are read-only with respect to the original study.
Only an explicit `run` consumes the single continuation launch allowance.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from weight_space_lab import joint_browser, mechanistic_study
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.native_runner import _lock
from weight_space_lab.protocol import canonical_hash

write = mechanistic_study._write
ZERO_COUNTS = ("engine_load_attempts", "warmup_dispatches", "scored_dispatches", "generation_dispatches")
FORBIDDEN_EVENTS = {"gate-request", "driver-gate-ack-forwarded", "gate-ack-received", "engine-call-intent",
                    "loaded", "task-artifact-loaded", "generation-call-intent", "sample", "warmup-sample", "task-sample"}
CLEANUP = ("controller_exit_verified", "native_capture_exit_confirmed", "chrome_exit_confirmed",
           "local_harness_exit_confirmed", "chrome_os_exit_verified", "owned_browser_and_worker_resources_released")


def verify_zero_call_failure(session):
    """Inspect durable evidence only. Missing or ambiguous evidence refuses continuation."""
    session = Path(session).resolve()
    target = session / "execution/candidate-1"
    supervisor, terminal = read_json(target / "supervisor-result.json"), read_json(target / "result.json")
    if supervisor.get("status") != "stopped" or terminal.get("status") != "failed":
        raise ValueError("Only a terminal pre-load candidate failure is eligible")
    if any(type(supervisor.get(k)) is not int or supervisor[k] != 0 for k in ZERO_COUNTS):
        raise ValueError("Prior target dispatch capacity was consumed or is unknown")
    if supervisor.get("outputs") != [] or supervisor.get("warmup_outputs") != []:
        raise ValueError("Prior target observations exist or are unknown")
    driver = terminal.get("driver", {})
    if any(driver.get(k) is not False for k in ("load_intent_observed", "load_acknowledged", "generation_intent_observed",
            "generation_acknowledged", "generation_call_intent_observed")):
        raise ValueError("Prior browser dispatch evidence is not explicitly zero")
    if any(driver.get(k) != 0 for k in ("generation_acknowledgements", "generation_intents")):
        raise ValueError("Prior browser dispatch count is not zero")
    if (session / "load-grant.json").exists() or list(target.glob("*intent.json")) or list(target.glob("*ack.json")):
        raise ValueError("Prior load/generation permission exists")
    events = [json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()]
    if not events or any(e.get("type") in FORBIDDEN_EVENTS for e in events):
        raise ValueError("Prior event stream contains target dispatch or is missing")
    cleanup = supervisor.get("cleanup", {})
    if any(cleanup.get(k) is not True for k in CLEANUP) or cleanup.get("retained_owned_pids_alive") != []:
        raise ValueError("Prior owned-process cleanup is unconfirmed")
    budget = supervisor.get("generation_budget", {})
    charged = budget.get("charged", {})
    if any(charged.get(k) != 0 for k in ("model_probes", "requested_sequences", "generated_tokens", "pending_requests", "retries")):
        raise ValueError("Prior generation accounting is not empty")
    db = session / "generation-budget.sqlite3"
    wal = db.with_name(db.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Prior SQLite journal is not checkpointed; immutable audit cannot prove zero attempts")
    with sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        if connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] != 0:
            raise ValueError("Prior durable ledger contains target attempts")
    return {"prior_engine_loads": 0, "prior_target_generations": 0,
        "failure_reason": terminal.get("execution_error"),
        "preload_elapsed_seconds": supervisor["supervisor_seconds"],
        "original_manifest_sha256": canonical_hash(read_json(session / "manifest.json")),
        "original_report_sha256": canonical_hash(read_json(session / "joint-report.json")),
        "original_events_sha256": file_sha256(target / "events.jsonl"), "original_ledger_sha256": file_sha256(db)}


def original(study_root):
    root, study, state = mechanistic_study.load(study_root)
    if (state.get("active") is not None or state.get("baseline", {}).get("status") != "completed"
            or state.get("candidate", {}).get("status") != "incomplete" or not state.get("proposal")
            or not state.get("deployment") or state["deployment"].get("conversion_erases_edit") is not False):
        raise ValueError("Requires completed baseline, accepted effective candidate, and one incomplete candidate launch")
    if sum(a["kind"] == "browser-candidate" for a in state["actions"]) != 1:
        raise ValueError("Exactly one original candidate launch is required")
    baseline = joint_browser.verify_report(root / "sessions/baseline")
    failed = joint_browser.verify_report(root / "sessions/candidate")
    if any(canonical_hash(r) != state[a]["sha256"] for a, r in (("baseline", baseline), ("candidate", failed))):
        raise ValueError("Original reports no longer bind the completed study actions")
    proof = verify_zero_call_failure(root / "sessions/candidate")
    artifact = read_json(state["deployment"]["manifest"])
    if canonical_hash(artifact) != state["deployment"]["sha256"] or artifact["artifact_sha256"] != failed["artifact_sha256"]:
        raise ValueError("Accepted candidate deployment changed")
    return root, study, state, baseline, proof


def prepare(root, study_root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Continuation root must be new; no repeated preparation")
    parent, study, state, baseline, proof = original(study_root)
    if root != parent.parent / "ifeval-mechanistic-browser-v1-preload-continuation":
        raise ValueError("Use the single reserved sibling continuation directory; alternate roots are forbidden")
    prior = parent / "sessions/candidate"
    previous = read_json(prior / "manifest.json")
    config = copy.deepcopy(read_json(prior / "baseline-config.json"))
    config["runId"] += "-preload-completion"
    root.mkdir(mode=0o700)
    session = joint_browser.prepare_session(root / "session", benchmark=study["benchmark"], config=config,
        items=read_json(parent / "items.json"), checkpoint=previous["checkpoint"],
        official_ifeval_root=study["official_ifeval_root"], study_sha256=canonical_hash(study),
        environment=study["environment"], browser_seconds=previous["browser_seconds"])
    if any(session[k] != previous[k] for k in ("study_manifest_sha256", "artifact_sha256", "checkpoint", "items_sha256",
        "settings", "environment", "harness_sha256", "model_lib_sha256", "browser_seconds", "grader", "official_assets")):
        raise ValueError("Continuation changed a frozen measurement property")
    pins = {str(parent / name): file_sha256(parent / name) for name in ("study.json", "ready.json", "state.json", "submission.json")}
    for arm in ("baseline", "candidate"):
        location = parent / "sessions" / arm
        for name in ("manifest.json", "ready.json", "joint-report.json", "joint-report-binding.json", "generation-budget.sqlite3"):
            pins[str(location / name)] = file_sha256(location / name)
    pins[state["deployment"]["manifest"]] = file_sha256(state["deployment"]["manifest"])
    plan = {"schema_version": 1, "kind": "explicit_zero_inference_preload_continuation", "parent": str(parent),
        "study_sha256": canonical_hash(study), "baseline_report_sha256": canonical_hash(baseline),
        "session_manifest_sha256": canonical_hash(session), "source_script_sha256": file_sha256(__file__),
        "original_pins": pins, "prelaunch_interruption": proof,
        "amendment": "The planned no-retry policy is amended for exactly one explicit pre-load continuation after a proved zero-target-call infrastructure failure. The original failed launch, report and ledger remain preserved. No baseline, proposal, conversion, or target generation is repeated.",
        "limits": {"additional_launches": 1, "engine_loads": 1, "warmups": 1,
                   "scored_requests": previous["scheduled_prompts"], "browser_seconds": previous["browser_seconds"], "automatic_retries": 0},
        "created_epoch_seconds": time.time()}
    write(root / "plan.json", plan)
    write(root / "ready.json", {"plan_sha256": canonical_hash(plan)})
    write(root / "state.json", {"plan_sha256": canonical_hash(plan), "started": False, "status": "prepared", "report_sha256": None})
    return {"status": "prepared", "plan_sha256": canonical_hash(plan), "target_calls": 0, "limits": plan["limits"]}


def load(root):
    root = Path(root).resolve()
    plan, state = read_json(root / "plan.json"), read_json(root / "state.json")
    if (read_json(root / "ready.json") != {"plan_sha256": canonical_hash(plan)} or state["plan_sha256"] != canonical_hash(plan)
            or plan["source_script_sha256"] != file_sha256(__file__)):
        raise ValueError("Continuation preparation/source identity changed")
    for name, digest in plan["original_pins"].items():
        if file_sha256(name) != digest:
            raise ValueError("Preserved original evidence changed")
    parent, study, prior_state, baseline, proof = original(plan["parent"])
    if proof != plan["prelaunch_interruption"] or canonical_hash(baseline) != plan["baseline_report_sha256"]:
        raise ValueError("Original zero-call proof changed")
    if canonical_hash(read_json(root / "session/manifest.json")) != plan["session_manifest_sha256"]:
        raise ValueError("Continuation session identity changed")
    return root, plan, state, study, prior_state, baseline


def run(root):
    root = Path(root).resolve()
    with _lock(root):
        root, plan, state, *_ = load(root)
        if state["started"]:
            raise ValueError("The one continuation launch allowance is already spent; no replay")
        state.update(started=True, status="running", started_epoch_seconds=time.time())
        write(root / "state.json", state, replace=True)
        try:
            value = joint_browser.execute(root / "session", on_load=lambda: plan["limits"]["browser_seconds"], preparation_seconds=300)
            checked = joint_browser.verify_report(root / "session")
            if checked != value:
                raise ValueError("Continuation result verification differs")
            state.update(status=checked["status"], report_sha256=canonical_hash(checked))
            return checked
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            state["ended_epoch_seconds"] = time.time()
            write(root / "state.json", state, replace=True)


def report(root):
    root, plan, state, study, prior_state, baseline = load(root)
    arms = {"baseline": baseline}
    if state["report_sha256"] is not None:
        candidate = joint_browser.verify_report(root / "session")
        if canonical_hash(candidate) != state["report_sha256"]:
            raise ValueError("Continuation report changed")
        arms["candidate"] = candidate
    comparison = joint_browser.compare_reports(baseline, arms["candidate"]) if "candidate" in arms else None
    target = root / "session/execution/candidate-1"
    execution = read_json(target / "supervisor-result.json") if (target / "supervisor-result.json").exists() else {}
    return {"schema_version": 1, "kind": "mechanistic_browser_comparison", "study_sha256": plan["study_sha256"],
        "status": "complete" if comparison and comparison["status"] == "completed" else "incomplete",
        "accuracy_benchmark": "ifeval", "resource_benchmark": "Browser LLM Bench on the same IFEval requests",
        "scope": study["claim_scope"], "scheduled_prompts": study["scheduled_prompts"], "runs_per_arm": 1,
        "separate_resource_generations": 0, "arms": arms, "comparison": comparison,
        "deployment_outcome": "effective_edit", "conversion_erases_edit": False, "candidate_evaluation_skipped": False,
        "diagnostic_usage": {name: sum(a["kind"] == name for a in prior_state["actions"]) for name in mechanistic_study.TOOLS},
        "activation_scope": study["activation_scope"], "editor_isolation": "unverified",
        "prelaunch_interruption": {**plan["prelaunch_interruption"], "amendment": plan["amendment"],
            "continuation_plan_sha256": canonical_hash(plan), "continuation_status": state["status"],
            "original_evidence_preserved": True, "repeated_target_sequences": 0},
        "browser_launches": {"baseline": 1, "candidate": 1 + int((target / "browser-session.json").exists())},
        "model_loads": {"baseline": 1, "candidate": execution.get("engine_load_attempts", 0)},
        "limitations": ["One measured model session per arm; a failed candidate browser launch before model loading is preserved separately.",
            "This explicit pre-load continuation amends the planned no-retry protocol; it does not repeat a target generation.",
            "Editor-visible development workload; no untouched accuracy, significance, or equivalence claim.",
            "Missing resource measurements remain null; partial CPU/GPU intervals are not whole-workload totals."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "report"):
        command = sub.add_parser(name); command.add_argument("--root", required=True)
        if name == "prepare": command.add_argument("--study", required=True)
        if name == "report": command.add_argument("--output")
    args = parser.parse_args()
    value = prepare(args.root, args.study) if args.command == "prepare" else run(args.root) if args.command == "run" else report(args.root)
    if getattr(args, "output", None): write(args.output, value)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
