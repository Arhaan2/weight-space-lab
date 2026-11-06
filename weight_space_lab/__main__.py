"""Command-line entry points for validated study mechanics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import compare_paired
from .patches import apply_patch, state_sha256
from .protocol import validate_protocol, validate_split_manifest


def read_json(path: Path):
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reproduction artifacts are append-only by default.
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def demo(out: Path) -> dict:
    """Exercise mechanics using synthetic arrays and synthetic grades only."""
    base = {
        "example.mlp.weight": np.array([[1, 2, 3, 4], [2, 3, 4, 5],
                                         [3, 4, 5, 6], [4, 5, 6, 7]], dtype=np.float32),
        "example.unchanged.weight": np.eye(4, dtype=np.float32),
    }
    proposal = {
        "schema_version": 1,
        "base_sha256": state_sha256(base),
        "operations": [{
            "op": "remove_direction_from_columns", "tensor": "example.mlp.weight",
            "columns": [0, 2], "direction": [1.0, 0.0, 0.0, 0.0], "strength": 1.0,
        }],
    }
    edited, audit = apply_patch(base, proposal, allowed_tensors=["example.mlp.weight"],
                                max_changed_fraction=0.1)
    if not np.array_equal(edited["example.unchanged.weight"], base["example.unchanged.weight"]):
        raise RuntimeError("unselected tensor changed")
    schedule = [{"item_id": item, "seed": seed} for item in ("toy-a", "toy-b", "toy-c")
                for seed in (0, 1)]
    baseline_grades = [True, False, False, True, True, False]
    candidate_grades = [False, True, True, False, True, False]
    baseline = [dict(row, status="ok", correct=correct)
                for row, correct in zip(schedule, baseline_grades)]
    candidate = [dict(row, status="ok", correct=correct)
                 for row, correct in zip(schedule, candidate_grades)]
    comparison = compare_paired(baseline, candidate, schedule=schedule,
                                bootstrap_samples=1000, seed=7)
    report = {
        "status": "synthetic_mechanics_only",
        "language_model_evaluated": False,
        "scientific_result": False,
        "explanation": "These arrays and grades are hand-authored fixtures. They measure no model capability.",
        "patch_audit": audit,
        "synthetic_comparison": comparison,
    }
    if out.exists():
        raise ValueError("choose a new output directory; existing evidence is not overwritten")
    out.mkdir(parents=True)
    write_json(out / "proposal.json", proposal)
    write_json(out / "schedule.json", schedule)
    write_json(out / "baseline.json", baseline)
    write_json(out / "candidate.json", candidate)
    write_json(out / "mechanics_report.json", report)
    return {"status": report["status"], "language_model_evaluated": False,
            "changed_synthetic_scalars": audit["changed_scalars"],
            "synthetic_comparison_delta": comparison["delta"]}


def build_parser():
    parser = argparse.ArgumentParser(description="Controlled weight-editing study utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate a proposed study configuration")
    check.add_argument("config", type=Path)
    check.add_argument("--require-ready", action="store_true")
    splits = commands.add_parser("check-splits", help="check exact split overlap")
    splits.add_argument("manifest", type=Path)
    example = commands.add_parser("demo", help="run synthetic mechanics fixtures")
    example.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser("compare", help="compare externally graded JSON record arrays")
    compare.add_argument("--schedule", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--metric", choices=["accuracy", "score", "completion_within_budget"], default="accuracy")
    compare.add_argument("--truncation-policy", choices=["grade", "fail"], default="grade")
    compare.add_argument("--bootstrap-samples", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser("replay", help="replay a published sparse BF16 patch on a local checkpoint")
    replay.add_argument("--base", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--archive", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            result = validate_protocol(read_json(args.config), require_ready=args.require_ready)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if not result["issues"] else 2
        if args.command == "check-splits":
            result = validate_split_manifest(read_json(args.manifest))
        elif args.command == "demo":
            result = demo(args.out)
        elif args.command == "compare":
            result = compare_paired(read_json(args.baseline), read_json(args.candidate),
                                    schedule=read_json(args.schedule), metric=args.metric,
                                    truncation_policy=args.truncation_policy,
                                    bootstrap_samples=args.bootstrap_samples, seed=args.seed)
            write_json(args.out, result)
        elif args.command == "replay":
            from .replay import replay_sparse_checkpoint
            if args.audit.resolve() == args.out.resolve():
                raise ValueError("audit and checkpoint output must be different paths")
            if args.audit.exists():
                raise ValueError("audit output exists; choose a new output")
            result = replay_sparse_checkpoint(args.base, args.manifest, args.archive, args.out)
            write_json(args.audit, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError) as error:
        # Avoid dumping local machine paths or raw provider errors into reports.
        if isinstance(error, OSError):
            print(json.dumps({"error": "file operation failed", "type": type(error).__name__}))
        else:
            print(json.dumps({"error": str(error), "type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
