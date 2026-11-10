"""Prepare a matched future browser candidate and compare independent grades.

No candidate is invented. ``prepare`` only saves a plan with unavailable candidate
identities. ``bind`` requires the accepted manual proposal and an independently
audited deployed artifact, and freezes the paired denominator before ``run``.
These future candidate actions are separate from the pristine baseline budget.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .analysis_pipeline import build_analysis_manifest, compare_graded_runs
from .ifeval_handoff import _save
from .inventory import file_sha256, read_json
from .protocol import canonical_hash


def prepare(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    config = read_json(root / "baseline-config.json")
    plan = {"schema_version": 1, "kind": "ifeval_matched_candidate_comparison_plan",
        "status": "awaiting_original_manual_proposal", "study_id": "ifeval-browser-v1",
        "baseline_manifest_sha256": canonical_hash(manifest),
        "baseline_config_file_sha256": file_sha256(root / "baseline-config.json"),
        "source_checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "baseline_artifact_sha256": config["convertedArtifactSha256"],
        "candidate_source_checkpoint_sha256": None, "candidate_artifact_sha256": None,
        "candidate_proposal_sha256": None,
        "task_order": [item["item_id"] for item in config["items"]],
        "task_items_sha256": canonical_hash(config["items"]),
        "settings": copy.deepcopy(config["settings"]),
        "warmup_prompt": config["warmupPrompt"], "total_limit_ms": config["totalLimitMs"],
        "grading_policy": copy.deepcopy(manifest["policy"]), "grader": copy.deepcopy(manifest["grader"]),
        "primary_metric": "prompt_level_strict_accuracy",
        "secondary_metrics": ["instruction_level_strict_accuracy", "prompt_level_loose_accuracy", "instruction_level_loose_accuracy"],
        "bootstrap": {"samples": 10000, "seed": 0, "unit": "complete item"},
        "candidate_budget": {"engine_load_attempts": 1, "warmup_dispatches": 1,
            "scored_dispatches": 32, "generation_dispatches": 33, "generated_tokens": 32784,
            "request_seconds": 120, "wall_seconds": 5400, "automatic_retries": 0},
        "heldout_generations": 0, "candidate_generations_authorized_by_baseline_stage": 0,
        "run_requirement": "A subsequent explicit candidate command is required after the original proposal import, saved-checkpoint validation, same-toolchain conversion, deployment audit, and fresh candidate ledger freeze.",
        "selection": "Compare all 32 fixed development items; prefer candidate only for strictly higher strict prompt compliance with complete technical execution and grading; ties retain pristine.",
        "heldout_stage": "Separate later frozen-candidate evaluation; not executed by these development commands."}
    _save(root / "candidate-comparison-plan.json", plan)
    _save(root / "candidate-comparison-plan-ready.json", {"plan_sha256": canonical_hash(plan)})
    return {"prepared": True, "candidate_executed": False, "plan_sha256": canonical_hash(plan)}


def _plan(root):
    root = Path(root)
    plan = read_json(root / "candidate-comparison-plan.json")
    if (read_json(root / "candidate-comparison-plan-ready.json") != {"plan_sha256": canonical_hash(plan)} or
            plan["baseline_manifest_sha256"] != canonical_hash(read_json(root / "manifest.json")) or
            plan["baseline_config_file_sha256"] != file_sha256(root / "baseline-config.json")):
        raise ValueError("frozen baseline or candidate comparison plan changed")
    return plan


def _arm_identity(manifest, config):
    return {"run_id": config["runId"], "profile": "browser_evaluation",
        "configuration_sha256": config["configurationSha256"],
        "checkpoint": copy.deepcopy(manifest["checkpoint"])}


def bind(root, candidate_artifact, authorization):
    """Future preparation after real manual import/conversion; no inference."""
    from . import ifeval_baseline
    root = Path(root)
    plan = _plan(root)
    imported = read_json(root / "handoff/import-result.json")
    artifact = read_json(candidate_artifact)
    if imported.get("accepted") is not True:
        raise ValueError("candidate binding requires accepted original manual proposal")
    if (artifact.get("source_checkpoint_sha256") != imported["audit"]["result_sha256"] or
            artifact.get("source_identity", {}).get("source_original_sha256") != plan["source_checkpoint_sha256"] or
            artifact.get("baseline_artifact_sha256") != plan["baseline_artifact_sha256"]):
        raise ValueError("deployed candidate does not bind the accepted pristine-origin proposal")
    # The trusted runner independently verifies the deployed package and creates
    # a fresh candidate-only ledger; it never reopens or reassigns baseline slots.
    candidate = ifeval_baseline.freeze_candidate(root, candidate_artifact=candidate_artifact,
                                               authorization=authorization)
    destination = root / "candidate-comparison"
    base_manifest = read_json(root / "manifest.json")
    base_config = read_json(root / "baseline-config.json")
    candidate_config = read_json(destination / "baseline-config.json")
    for key in ("settings", "items", "warmupPrompt", "totalLimitMs"):
        if candidate_config[key] != base_config[key]:
            raise ValueError("candidate task order, runtime budget, or warmup differs from baseline")
    if (candidate["policy"] != base_manifest["policy"] or
            candidate["grader"] != base_manifest["grader"] or
            candidate["checkpoint"]["sha256"] != imported["audit"]["result_sha256"]):
        raise ValueError("candidate scorer or source identity differs")
    items = read_json(base_manifest["items_path"])
    source_sha = canonical_hash(read_json(root / "data/manifest.json"))
    split_manifest = {"schema_version": 1, "kind": "benchmark_split", "split": "development",
        "source_manifest_sha256": source_sha,
        "items": [{"benchmark": item["benchmark"], "item_id": item["item_id"],
                   "dataset_revision": item["dataset_revision"], "item_sha256": canonical_hash(item)} for item in items]}
    paired = build_analysis_manifest(split_manifest, items, benchmark="ifeval", seeds=[42],
        policy=base_manifest["policy"], arms={"baseline": _arm_identity(base_manifest, base_config),
            "candidate": _arm_identity(candidate, candidate_config)}, stage="pilot",
        metric="prompt_level_strict_accuracy", evidence_kind="real_model")
    _save(destination / "paired-source-binding.json", {"source_manifest_sha256": source_sha,
        "source": "new study canonical item identities and pre-baseline manifest; no reused historical split roles"})
    _save(destination / "paired-split-manifest.json", split_manifest)
    _save(destination / "paired-analysis-manifest.json", paired)
    _save(destination / "comparison-ready.json", {"plan_sha256": canonical_hash(plan),
        "paired_manifest_sha256": canonical_hash(paired), "candidate_manifest_sha256": canonical_hash(candidate),
        "candidate_artifact_file_sha256": file_sha256(candidate_artifact),
        "baseline_graded_file_sha256": file_sha256(root / "baseline/graded.json")})
    return {"bound": True, "candidate_executed": False,
            "candidate_artifact_sha256": artifact["artifact_sha256"], "paired_manifest_sha256": canonical_hash(paired)}


def _bound(root):
    root = Path(root)
    plan = _plan(root)
    target = root / "candidate-comparison"
    ready = read_json(target / "comparison-ready.json")
    paired = read_json(target / "paired-analysis-manifest.json")
    candidate = read_json(target / "manifest.json")
    if (ready["plan_sha256"] != canonical_hash(plan) or
            ready["paired_manifest_sha256"] != canonical_hash(paired) or
            ready["candidate_manifest_sha256"] != canonical_hash(candidate) or
            ready["baseline_graded_file_sha256"] != file_sha256(root / "baseline/graded.json")):
        raise ValueError("candidate comparison or baseline evidence changed")
    return target, plan, paired, candidate


def run(root):
    """Future explicit candidate execution; never called by baseline export."""
    from . import ifeval_baseline
    target, _, _, _ = _bound(root)
    result = ifeval_baseline.execute(target)
    # Independent grading uses preserved raw responses, never regenerates them.
    grades = ifeval_baseline.grade(target)
    return {"execution": result, "grading": grades}


def compare(root):
    root = Path(root)
    target, plan, paired, candidate_manifest = _bound(root)
    baseline = read_json(root / "baseline/graded.json")
    candidate = read_json(target / candidate_manifest.get("result_dir", "candidate") / "graded.json")
    result = compare_graded_runs(baseline["records"], candidate["records"],
        baseline["grades"], candidate["grades"], paired,
        bootstrap_samples=plan["bootstrap"]["samples"], seed=plan["bootstrap"]["seed"])
    complete = (baseline.get("summary", {}).get("technically_valid") is True and
        candidate.get("summary", {}).get("technically_valid") is True and result["semantic_grading_complete"] and
        all(r["status"] in ("ok", "truncated") for r in baseline["records"] + candidate["records"]))
    metric = "prompt_level_strict_accuracy"
    base_score = result["baseline_projection"]["official_metrics"][metric]["value"]
    candidate_score = result["candidate_projection"]["official_metrics"][metric]["value"]
    result["selection"] = {"technical_and_grading_complete": complete,
        "selected_arm": "candidate" if complete and candidate_score > base_score else "baseline" if complete else None,
        "rule": plan["selection"], "heldout_generations": 0}
    _save(target / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "bind", "run", "compare"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--candidate-artifact")
    parser.add_argument("--authorization", help="Subsequent candidate-stage authorization, distinct from baseline-only authorization")
    args = parser.parse_args()
    if args.command == "bind":
        if not args.candidate_artifact or not args.authorization:
            parser.error("bind requires --candidate-artifact and --authorization")
        result = bind(args.root, args.candidate_artifact, args.authorization)
    else:
        result = {"prepare": prepare, "run": run, "compare": compare}[args.command](args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
