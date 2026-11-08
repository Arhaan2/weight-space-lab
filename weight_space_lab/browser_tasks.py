"""Frozen MATH tasks and independent grading for the existing WebLLM worker.

This module performs no model inference. It adapts browser observations to the
existing versioned execution records and Math-Verify grader. Private task and
output files must stay under ignored artifacts; the browser receives no targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from .benchmark_data import audit_overlap, validate_item
from .grading import (MATH_PACKAGES, _known_answer_item, grade_record,
                      grader_identity, validate_grading_policy)
from .native_runner import _missing, _write_json
from .protocol import canonical_hash
from .published_study import replay_tasks
from .records import validate_run_record


def browser_grading_policy(*, thinking=False):
    """Explicit browser policy; native pre-opened reasoning is never assumed."""
    if type(thinking) is not bool:
        raise ValueError("thinking must be boolean")
    policy = json.loads(Path("configs/grading-policy.json").read_text())
    policy["reasoning_policy"] = "qwen_thinking" if thinking else "think_tags"
    return validate_grading_policy(policy)


def _read(path):
    path = Path(path)
    # macOS cloud placeholders are not evidence; do not trigger a hanging read.
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise ValueError(f"required input is an unavailable cloud placeholder: {path}")
    return json.loads(path.read_text())


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value, immutable=True)


def _coverage(items):
    return {"count": len(items),
            "subjects": dict(sorted(Counter(i["metadata"].get("subject", "unspecified") for i in items).items())),
            "levels": dict(sorted(Counter(str(i["metadata"].get("level", "unspecified")) for i in items).items()))}


def prepare_manifests(root, *, development="artifacts/on-device-native-pilot-v2",
                      data="artifacts/phase04-benchmark-data-verified",
                      design="configs/phase8_development.json"):
    """Freeze existing development tasks and a reference-only MATH-500 holdout.

    Held-out answers are hashed by code to verify the existing manifest but
    never emitted or copied into browser/editor payloads. No evaluation runs.
    """
    root, development, data = Path(root), Path(development), Path(data)
    if "artifacts" not in root.parts or root.exists() or root.is_symlink():
        raise ValueError("use a new private artifacts directory")
    config = _read(design)
    items = _read(development / "items.json")
    tasks = replay_tasks(items)
    if (len(items) != 14 or canonical_hash(items) != config["selected_items_sha256"]
            or canonical_hash(tasks) != config["tasks_sha256"]
            or tasks != _read(development / "tasks.json")):
        raise ValueError("development items or task formatting differ from frozen Phase 8")
    refs = [{key: i[key] for key in ("benchmark", "item_id", "dataset_revision")}
            | {"item_sha256": canonical_hash(i)} for i in items]
    if refs != [{key: r[key] for key in ("benchmark", "item_id", "dataset_revision", "item_sha256")}
                for r in config["item_refs"]]:
        raise ValueError("development item identities differ from frozen selection")
    source = _read("configs/benchmark_sources.json")
    if canonical_hash(source) != config["source"]["source_manifest_sha256"]:
        raise ValueError("source manifest differs from frozen study")
    original_audit = _read(data / "overlap-audit.json")
    if canonical_hash(original_audit) != config["source"]["overlap_audit_sha256"] or not original_audit["passed"]:
        raise ValueError("historical split audit differs or failed")
    final_manifest = _read(data / "evaluator-final/manifest.json")
    if final_manifest["source_manifest_sha256"] != canonical_hash(source) or final_manifest["split"] != "test":
        raise ValueError("held-out source identity differs")
    heldout = [i for i in _read(data / "evaluator-final/items.json") if i["benchmark"] == "math"]
    held_refs = [r for r in final_manifest["items"] if r["benchmark"] == "math"]
    if len(heldout) != 500 or len(held_refs) != 500:
        raise ValueError("existing MATH-500 final split must contain exactly 500 items")
    checked_refs = [{key: i[key] for key in ("benchmark", "item_id", "dataset_revision")}
                    | {"item_sha256": canonical_hash(validate_item(i))} for i in heldout]
    if checked_refs != held_refs or any(i["split"] != "test" for i in heldout):
        raise ValueError("held-out items differ from their saved manifest")
    overlap = audit_overlap(items + heldout)
    if not overlap["passed"]:
        raise ValueError("development and held-out identities/text/template families overlap")
    policy = browser_grading_policy(thinking=False)
    root.mkdir(parents=True, mode=0o700)
    payloads = {
        "items-evaluator-only.json": items,
        "tasks.json": tasks,
        "smoke-task.json": tasks[0],
        "grading-policy.json": policy,
        "development-manifest.json": {
            "schema_version": 1, "kind": "browser_math_development", "author": "Arhaan2",
            "benchmark": "math", "split": "development", "items": refs,
            "selected_items_sha256": canonical_hash(items), "tasks_sha256": canonical_hash(tasks),
            "source_design_sha256": canonical_hash(config), "source_manifest_sha256": canonical_hash(source),
            "source": config["source"], "coverage": _coverage(items),
            "representativeness": config["selection"]["representativeness"],
            "prompt_adapter": "weight_space_lab.published_study.replay_tasks",
            "browser_thinking": False, "browser_grading_policy_sha256": canonical_hash(policy),
            "native_prompt_token_ids_reused": False, "seeds": [42],
            "scientific_scope": "MATH development component only; no general on-device quality claim",
        },
        "heldout-manifest.json": {
            "schema_version": 1, "kind": "browser_math_heldout", "author": "Arhaan2",
            "benchmark": "math", "split": "test", "items": held_refs,
            "source": [s for s in source["sources"] if s["key"] == "math500"],
            "source_manifest_sha256": canonical_hash(source),
            "existing_final_manifest_sha256": canonical_hash(final_manifest),
            "existing_evaluator_items": str(data / "evaluator-final/items.json"),
            "coverage": _coverage(heldout), "overlap_audit_sha256": canonical_hash(overlap),
            "execution_authorized": False, "executed_items": 0,
            "answers_included": False, "editor_exposure_permitted": False,
            "scope": "All 500 existing MATH-500 items; GSM8K and IFEval excluded from this math component",
            "limitations": overlap["limitations"],
        },
        "overlap-audit.json": overlap,
    }
    for name, value in payloads.items():
        _save(root / name, value)
    summary = {"schema_version": 1, "kind": "browser_task_preparation", "author": "Arhaan2",
               "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in payloads},
               "development_items": 14, "heldout_items": 500, "overlap_passed": True,
               "inference_sequences": 0, "heldout_evaluations": 0, "grader": grader_identity("math"),
               "private_prompts_excluded_from_public_artifacts": True}
    _save(root / "manifest.json", summary)
    return summary


def _measurement(value, name, *, integer=False):
    if value is None:
        return None
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be nonnegative {'integer' if integer else 'finite number'} or null")
    return value


def adapt_browser_output(item, output, *, run_id, checkpoint, configuration_sha256,
                         seed=42, evidence_kind="real_model"):
    """Preserve one observation, including partial/failed output, without retries."""
    validate_item(item)
    if item["benchmark"] not in {"math", "ifeval"} or item["split"] != "development":
        raise ValueError("this adapter accepts frozen MATH or IFEval development items")
    observation = copy.deepcopy(output)
    if observation is None:
        outcome, state = _missing(), "unrun"
    else:
        if observation.get("item_id") != item["item_id"]:
            raise ValueError("browser observation item identity differs")
        finish = observation.get("finish_reason")
        cancelled = observation.get("cancelled", False)
        if type(cancelled) is not bool:
            raise ValueError("cancelled must be boolean")
        if finish not in {"stop", "eos", "length", "error", "timeout", "cancelled", "interrupted"}:
            raise ValueError("browser finish reason must be explicit")
        state = ("timeout" if finish == "timeout" else "interrupted" if finish == "interrupted"
                 else "cancelled" if cancelled or finish == "cancelled"
                 else "error" if finish == "error" or observation.get("execution_error") is not None
                 else "truncated" if finish == "length" else "completed")
        status = {"completed": "ok", "truncated": "truncated", "cancelled": "error",
                  "error": "error", "timeout": "timeout", "interrupted": "error"}[state]
        raw = observation.get("raw_output")
        if raw is not None and not isinstance(raw, str):
            raise ValueError("raw_output must be text or null")
        latency = _measurement(observation.get("response_latency_ms"), "response_latency_ms")
        input_tokens = _measurement(observation.get("input_tokens"), "input_tokens", integer=True)
        output_tokens = _measurement(observation.get("generated_tokens"), "generated_tokens", integer=True)
        _measurement(observation.get("loading_time_ms"), "loading_time_ms")
        outcome = {"status": status, "raw_output": raw,
                   "finish_reason": finish if status in {"ok", "truncated"} else "timeout" if status == "timeout" else "error",
                   "token_counts": {"input": input_tokens, "output": output_tokens,
                                    "total": None if input_tokens is None or output_tokens is None else input_tokens + output_tokens},
                   "duration_seconds": None if latency is None else latency / 1000,
                   "error": None if status in {"ok", "truncated"} else {"code": f"browser_{state}", "retryable": False}}
    record = validate_run_record({"schema_version": 1, "record_type": "run", "run_id": run_id,
        "evidence_kind": evidence_kind, "profile": "browser_evaluation", "stage": "pilot",
        **{k: item[k] for k in ("benchmark", "dataset_revision", "item_id", "split")},
        "sample_index": 0, "seed": seed, "checkpoint": checkpoint,
        "configuration_sha256": configuration_sha256,
        "retry_policy": {"mode": "none", "max_attempts": 1},
        "attempts": [] if state == "unrun" else [dict(outcome, attempt_index=0)], **outcome})
    observed = observation or {}
    raw = outcome["raw_output"]
    valid = (state == "completed" and isinstance(raw, str) and bool(raw.strip()) and "\ufffd" not in raw
             and (outcome["token_counts"]["output"] or 0) > 0)
    metric = {"item_id": item["item_id"], "execution_status": state,
              "raw_output": raw, "valid_completion": valid,
              "truncated": state == "truncated" or observed.get("truncated") is True,
              "generated_tokens": outcome["token_counts"]["output"],
              "response_latency_ms": observed.get("response_latency_ms"),
              "loading_time_ms": observed.get("loading_time_ms"),
              "loading_time_scope": "arm model loading; separate from per-response latency; do not sum per-item repeats",
              "execution_error": observed.get("execution_error"),
              "cancelled": state == "cancelled" or observed.get("cancelled", False),
              "cleanup": observed.get("cleanup", {"status": "unavailable", "reason": "no cleanup observation supplied"}),
              "resources": observed.get("resources", {"status": "unavailable", "reason": "no browser resource observations supplied"}),
              "resource_scope_policy": "retain browser/worker/host scopes separately; no PyTorch MPS counters inferred",
              "browser_observation": observation}
    return record, metric


def grade_browser_arm(items, outputs, *, run_id, checkpoint, configuration_sha256,
                      policy=None, seed=42, evidence_kind="real_model"):
    """Grade all scheduled items, retaining interrupted/error/unrun denominators."""
    policy = validate_grading_policy(policy or browser_grading_policy(thinking=False))
    if not items or len({i["item_id"] for i in items}) != len(items):
        raise ValueError("a nonempty unique scheduled item list is required")
    if not isinstance(outputs, list):
        raise ValueError("browser outputs must be an array")
    indexed = {}
    allowed = {i["item_id"] for i in items}
    for output in outputs:
        if not isinstance(output, dict) or output.get("item_id") not in allowed or output["item_id"] in indexed:
            raise ValueError("duplicate or unscheduled browser observation")
        indexed[output["item_id"]] = output
    records, grades, metrics = [], [], []
    for item in items:
        record, metric = adapt_browser_output(item, indexed.get(item["item_id"]), run_id=run_id,
            checkpoint=checkpoint, configuration_sha256=configuration_sha256, seed=seed, evidence_kind=evidence_kind)
        grade = grade_record(record, item, policy)
        metric.update(extracted_answer=grade["diagnostics"]["extracted_answer"], correctness=grade["correct"],
                      grading_status=grade["status"], grading_code=grade["diagnostics"]["code"],
                      denominator_score=int(grade["correct"] is True))
        records.append(record)
        grades.append(grade)
        metrics.append(metric)
    states = Counter(m["execution_status"] for m in metrics)
    correct = sum(g["correct"] is True for g in grades)
    unresolved = sum(g["status"] in {"grader_error", "grader_timeout"} for g in grades)
    summary = {"scheduled": len(items), "attempted": len(outputs),
        "evaluated": states["completed"] + states["truncated"],
        "interrupted": states["timeout"] + states["cancelled"] + states["interrupted"], "execution_errors": states["error"],
        "unrun": states["unrun"], "execution_status_counts": dict(states),
        "independently_graded": sum(g["status"] == "graded" for g in grades),
        "unresolved_grades": unresolved, "correct": correct,
        "all_scheduled_accuracy_lower_bound": correct / len(items),
        "accuracy_is_resolved": unresolved == 0,
        "valid_completions": sum(m["valid_completion"] for m in metrics),
        "truncated": sum(m["truncated"] for m in metrics),
        "known_generated_tokens": sum(m["generated_tokens"] or 0 for m in metrics),
        "attempted_items_with_unknown_token_count": sum(m["generated_tokens"] is None and m["execution_status"] != "unrun" for m in metrics),
        "complete_for_selection": len(outputs) == len(items) and not unresolved
            and not any(states[s] for s in ("timeout", "cancelled", "interrupted", "error", "unrun")),
        "denominator_policy": "all scheduled items; failed/unrun outcomes score zero, unresolved grader outcomes remain null with lower bound only",
        "selection_rule": "candidate strictly exceeds baseline correct count after complete paired execution and resolved grades; baseline wins ties",
    }
    return {"schema_version": 1, "kind": "browser_math_task_results", "author": "Arhaan2",
            "evidence_kind": evidence_kind, "policy": policy, "policy_sha256": canonical_hash(policy),
            "records": records, "grades": grades, "item_metrics": metrics, "summary": summary}


def run_browser_grader_fixtures():
    """Known answers through the actual existing grader; zero target inference."""
    versions = {name: importlib.metadata.version(name) for name in MATH_PACKAGES}
    if versions != MATH_PACKAGES:
        raise ValueError("pinned Math-Verify dependencies differ")
    item = _known_answer_item()
    checkpoint = {"model_id": "hand-authored-browser-grading-fixture", "revision": item["dataset_revision"],
                  "sha256": canonical_hash({"checkpoint": "none; hand-authored fixture"})}
    cases = [("correct", r"\boxed{2/4}", "stop", True),
             ("incorrect", r"\boxed{3/4}", "stop", False),
             ("malformed", r"\boxed{", "stop", False),
             ("truncated_answer", r"\boxed{1/2}", "length", True),
             ("truncated_no_answer", "Unfinished calculation", "length", False),
             ("unclosed_reasoning", r"<think>\boxed{1/2}", "length", False)]
    checks = []
    for name, output, finish, expected in cases:
        result = grade_browser_arm([item], [{"item_id": item["item_id"], "raw_output": output,
            "finish_reason": finish, "generated_tokens": 10, "input_tokens": 5,
            "response_latency_ms": 12, "loading_time_ms": 100}], run_id="browser-grading-fixtures",
            checkpoint=checkpoint, configuration_sha256=canonical_hash({"fixture": name}), evidence_kind="fixture")
        metric = result["item_metrics"][0]
        checks.append({"name": name, "expected_correct": expected,
                       "actual_correct": metric["correctness"], "grading_status": metric["grading_status"],
                       "extracted_answer": metric["extracted_answer"], "truncated": metric["truncated"],
                       "valid_completion": metric["valid_completion"],
                       "passed": metric["correctness"] is expected and metric["grading_status"] == "graded"})
    return {"schema_version": 1, "kind": "browser_grader_known_answers", "author": "Arhaan2",
            "evidence_kind": "fixture", "target_inference_sequences": 0, "scientific_result": False,
            "packages": versions, "grader": grader_identity("math"), "checks": checks,
            "all_passed": all(c["passed"] for c in checks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", required=True)
    fixtures = sub.add_parser("fixtures")
    fixtures.add_argument("--output", required=True)
    grade = sub.add_parser("grade")
    grade.add_argument("--items", required=True)
    grade.add_argument("--outputs", required=True)
    grade.add_argument("--config", required=True, help="JSON containing run_id, checkpoint, configuration_sha256, policy, seed")
    grade.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_manifests(args.root)
    elif args.command == "fixtures":
        result = run_browser_grader_fixtures()
        _save(args.output, result)
    else:
        result = grade_browser_arm(_read(args.items), _read(args.outputs), **_read(args.config))
        _save(args.output, result)
        result = result["summary"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
