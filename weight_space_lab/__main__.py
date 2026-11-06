"""Command-line entry points for validated study mechanics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import validate_protocol, validate_split_manifest


def read_json(path: Path):
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    def reject_constant(_):
        raise ValueError("JSON must not contain nonfinite constants")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys,
                      parse_constant=reject_constant)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reproduction artifacts are append-only by default.
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def demo(out: Path) -> dict:
    """Exercise mechanics using synthetic arrays and synthetic grades only."""
    import numpy as np

    from .evaluation import compare_paired
    from .patches import apply_patch, state_sha256

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
    diagnose = commands.add_parser("diagnose", help="report dependencies, devices and memory without loading a model")
    diagnose.add_argument("--probe-backends", action="store_true",
                          help="run an isolated, time-bounded optional backend probe")
    diagnose.add_argument("--out", type=Path)
    records = commands.add_parser("check-records", help="validate a versioned raw run-record array")
    records.add_argument("records", type=Path)
    status = commands.add_parser("check-status", help="validate separate implementation and evidence statuses")
    status.add_argument("record", type=Path)
    schedule = commands.add_parser("check-schedule", help="validate a benchmark-specific item/sample/seed schedule")
    schedule.add_argument("schedule", type=Path)
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
    acquire = commands.add_parser("acquire-published", help="acquire the pinned model snapshot and selected patch artifacts")
    acquire.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
    acquire.add_argument("--out", type=Path, required=True)
    acquire.add_argument("--max-download-bytes", type=int, default=3_000_000_000)
    acquire.add_argument("--timeout", type=float, default=900.0,
                         help="per-request transfer budget and socket timeout in seconds")
    acquire.add_argument("--max-elapsed-seconds", type=float, default=1800.0,
                         help="HTTP elapsed-time budget, checked between socket operations")
    acquisition = commands.add_parser("verify-acquisition", help="recheck all pinned acquisition files offline")
    acquisition.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
    acquisition.add_argument("--bundle", type=Path, required=True)
    inventory = commands.add_parser("inventory-model", help="inventory every local model file and stored tensor")
    inventory.add_argument("--model", type=Path, required=True)
    inventory.add_argument("--out", type=Path, required=True)
    for command, help_text in (("reconstruct-published", "reconstruct and verify a complete model directory atomically"),
                               ("verify-reconstruction", "independently verify saved reconstruction bytes and audit")):
        reconstruction = commands.add_parser(command, help=help_text)
        reconstruction.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
        reconstruction.add_argument("--acquisition", type=Path, required=True)
        reconstruction.add_argument("--out", type=Path, required=True,
                                    help="reconstruction bundle (new output for reconstruct, existing for verify)")
    identity = commands.add_parser("copy-identity", help="create an independently verified byte-copy control")
    identity.add_argument("--acquisition", type=Path, required=True)
    identity.add_argument("--out", type=Path, required=True)
    identity.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
    prepare = commands.add_parser("prepare-native-smoke", help="freeze matched tiny native smoke configurations")
    prepare.add_argument("--acquisition", type=Path, required=True)
    prepare.add_argument("--reconstruction", type=Path, required=True)
    prepare.add_argument("--identity", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
    prepare.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    native = commands.add_parser("run-native", help="execute or resume a frozen development smoke schedule")
    native.add_argument("config", type=Path)
    native.add_argument("--model", type=Path, required=True)
    native.add_argument("--tasks", type=Path, required=True)
    native.add_argument("--schedule", type=Path, required=True)
    native.add_argument("--out", type=Path, required=True)
    native.add_argument("--resume", action="store_true")
    acquire_data = commands.add_parser("acquire-benchmarks", help="download and verify pinned benchmark source files")
    acquire_data.add_argument("--reference", type=Path, default=Path("configs/benchmark_sources.json"))
    acquire_data.add_argument("--out", type=Path, required=True)
    acquire_data.add_argument("--max-download-bytes", type=int, default=12_000_000)
    acquire_data.add_argument("--timeout", type=float, default=30.0)
    prepare_data = commands.add_parser("prepare-benchmark-data", help="prepare explicit audited benchmark splits and role bundles")
    prepare_data.add_argument("--reference", type=Path, default=Path("configs/benchmark_sources.json"))
    prepare_data.add_argument("--cache", type=Path, required=True)
    prepare_data.add_argument("--out", type=Path, required=True)
    bundle = commands.add_parser("check-benchmark-bundle", help="verify prepared source identities and split artifacts")
    bundle.add_argument("bundle", type=Path)
    bundle.add_argument("--cache", type=Path,
                        help="independently reconstruct expected items from verified source bytes")
    ifeval = commands.add_parser("acquire-ifeval", help="acquire pinned official evaluator code, license, and resources")
    ifeval.add_argument("--out", type=Path, required=True)
    grader_check = commands.add_parser("check-graders", help="execute independent known-answer fixtures with real evaluators")
    grader_check.add_argument("--ifeval-root", type=Path)
    grader_check.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser("freeze-analysis", help="freeze a complete split, sample schedule, and analysis policy")
    for name in ("split-manifest", "items", "seeds", "policy", "arms", "out"):
        freeze.add_argument("--" + name, type=Path, required=True)
    freeze.add_argument("--benchmark", required=True)
    freeze.add_argument("--stage", choices=["smoke", "pilot", "full"], required=True)
    freeze.add_argument("--metric", default="accuracy")
    freeze.add_argument("--evidence-kind", choices=["fixture", "real_model"], required=True)
    grade = commands.add_parser("grade-records", help="grade raw records independently and retain diagnostics")
    for name in ("records", "items", "policy", "out"):
        grade.add_argument("--" + name, type=Path, required=True)
    grade.add_argument("--ifeval-root", type=Path)
    project = commands.add_parser("project-analysis", help="project raw evidence and diagnostic grades under a frozen manifest")
    for name in ("manifest", "records", "grades", "out"):
        project.add_argument("--" + name, type=Path, required=True)
    project.add_argument("--arm", choices=["baseline", "candidate"], required=True)
    production = commands.add_parser("compare-graded", help="compare complete scheduled arms with separate model/grader failure accounting")
    for name in ("manifest", "baseline-records", "candidate-records", "baseline-grades", "candidate-grades", "out"):
        production.add_argument("--" + name, type=Path, required=True)
    production.add_argument("--bootstrap-samples", type=int, default=10000)
    production.add_argument("--seed", type=int, default=0)
    replay_prepare = commands.add_parser("prepare-published-replay", help="verify and freeze one published-patch replay stage without inference")
    replay_prepare.add_argument("--stage", choices=["smoke", "pilot", "full"], required=True)
    for name in ("root", "fresh-editor-workspace", "acquisition", "reconstruction", "identity", "data", "cache"):
        replay_prepare.add_argument("--" + name, type=Path, required=True)
    replay_prepare.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    replay_prepare.add_argument("--development-report", type=Path)
    replay_prepare.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
    replay_prepare.add_argument("--plan", type=Path, default=Path("configs/published_replay_plan.json"))
    replay_prepare.add_argument("--policy", type=Path, default=Path("configs/grading-policy.json"))
    replay_prepare.add_argument("--smoke-measurement", type=Path, default=Path("reports/native-smoke/report.json"))
    for name in ("smoke", "pilot", "full"):
        command = commands.add_parser("replay-" + name, help=f"explicitly execute/resume only the frozen {name} replay stage")
        for argument in ("root", "acquisition", "reconstruction", "identity"):
            command.add_argument("--" + argument, type=Path, required=True)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--job", help="execute only this frozen benchmark/arm job")
    replay_report = commands.add_parser("report-published-replay", help="independently grade and report a frozen replay, including missing results")
    replay_report.add_argument("--root", type=Path, required=True)
    replay_report.add_argument("--stage", choices=["smoke", "pilot", "full"], required=True)
    replay_report.add_argument("--report-id", required=True)
    replay_report.add_argument("--ifeval-root", type=Path)
    inspect_candidate = commands.add_parser("inspect-checkpoint", help="expose a declared bounded checkpoint inspection and record it")
    for argument in ("model", "policy", "request", "log"):
        inspect_candidate.add_argument("--" + argument, type=Path, required=True)
    edit_candidate = commands.add_parser("edit-checkpoint", help="apply a bounded proposal from the immutable original and verify saved bytes")
    for argument in ("model", "policy", "patch", "log", "out"):
        edit_candidate.add_argument("--" + argument, type=Path, required=True)
    verify_candidate = commands.add_parser("verify-candidate", help="reopen a candidate and verify stored changes, policy, and unrelated assets")
    for argument in ("model", "candidate", "policy"):
        verify_candidate.add_argument("--" + argument, type=Path, required=True)
    candidate_demo = commands.add_parser("demo-candidate-edit", help="demonstrate a labeled hand-authored engineering edit on development input")
    candidate_demo.add_argument("--model", type=Path, required=True)
    candidate_demo.add_argument("--out", type=Path, required=True)
    candidate_demo.add_argument("--backend", choices=["transformers"], help="explicitly enable tiny native development generations")
    candidate_demo.add_argument("--tensor", default="model.language_model.layers.15.mlp.down_proj.weight")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            result = validate_protocol(read_json(args.config), require_ready=args.require_ready)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if not result["issues"] else 2
        if args.command == "diagnose":
            from .diagnostics import collect_diagnostics
            result = collect_diagnostics(probe_backends=args.probe_backends)
            if args.out is not None:
                write_json(args.out, result)
        elif args.command == "check-records":
            from .protocol import canonical_hash
            from .records import validate_run_records
            records = validate_run_records(read_json(args.records))
            result = {"valid": True, "record_count": len(records),
                      "records_sha256": canonical_hash(records),
                      "verification_scope": "schema only; outputs and provenance require independent verification"}
        elif args.command == "check-status":
            from .records import validate_status_record
            result = validate_status_record(read_json(args.record))
        elif args.command == "check-schedule":
            from .protocol import validate_benchmark_schedule
            result = validate_benchmark_schedule(read_json(args.schedule))
        elif args.command == "check-splits":
            result = validate_split_manifest(read_json(args.manifest))
        elif args.command == "demo":
            result = demo(args.out)
        elif args.command == "compare":
            from .evaluation import compare_paired
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
        elif args.command in ("acquire-published", "verify-acquisition"):
            from .acquisition import acquire_published, verify_acquisition
            if args.command == "acquire-published":
                audit = acquire_published(args.reference, args.out,
                                          max_download_bytes=args.max_download_bytes,
                                          timeout=args.timeout, max_elapsed_seconds=args.max_elapsed_seconds)
            else:
                audit = verify_acquisition(args.bundle, args.reference)
            result = {key: audit[key] for key in ("kind", "complete", "resolved_source_revision", "resolved_model_revision")}
            result.update(stored_tensor_count=len(audit["model_inventory"]["tensors"]),
                          stored_parameter_count=audit["model_inventory"]["parameter_count"],
                          inference_executed=False, experiment_executed=False)
        elif args.command == "inventory-model":
            from .inventory import inventory_model
            inventory = inventory_model(args.model)
            write_json(args.out, inventory)
            result = {"file_count": len(inventory["files"]), "tensor_count": len(inventory["tensors"]),
                      "stored_parameter_count": inventory["parameter_count"]}
        elif args.command in ("reconstruct-published", "verify-reconstruction"):
            from .reconstruction import reconstruct_published, verify_reconstruction
            operation = reconstruct_published if args.command == "reconstruct-published" else verify_reconstruction
            audit = operation(args.acquisition, args.out, args.reference)
            result = {key: value for key, value in audit.items() if key != "model_inventory"}
        elif args.command == "copy-identity":
            from .native_assets import copy_identity
            result = copy_identity(args.acquisition, args.out, args.reference)
        elif args.command == "prepare-native-smoke":
            from .native_assets import prepare_smoke
            result = prepare_smoke(args.acquisition, args.reconstruction, args.identity,
                                   args.out, args.reference, backend=args.backend)
        elif args.command == "run-native":
            from .native_runner import run_schedule
            config, tasks, schedule = read_json(args.config), read_json(args.tasks), read_json(args.schedule)
            if config.get("profile") != "native_development_smoke" or config.get("stage") != "smoke":
                raise ValueError("run-native is smoke-only; use the explicit frozen replay stage command")
            result = run_schedule(config, tasks, schedule, args.out, model_path=args.model, resume=args.resume)
        elif args.command == "prepare-published-replay":
            from .published_study import prepare_study
            result = prepare_study(root=args.root, fresh_editor_workspace=args.fresh_editor_workspace,
                acquisition=args.acquisition, reconstruction=args.reconstruction, identity=args.identity,
                data=args.data, cache=args.cache, backend=args.backend, stage=args.stage,
                reference=args.reference, plan_path=args.plan, policy_path=args.policy,
                development_report=args.development_report, smoke_measurement=args.smoke_measurement)
        elif args.command in {"replay-smoke", "replay-pilot", "replay-full"}:
            from .published_study import run_study
            result = run_study(args.root, args.command.removeprefix("replay-"), acquisition=args.acquisition,
                reconstruction=args.reconstruction, identity=args.identity, resume=args.resume, job=args.job)
        elif args.command == "report-published-replay":
            from .published_study import report_study
            report = report_study(args.root, args.stage, report_id=args.report_id, ifeval_root=args.ifeval_root)
            result = {key: report[key] for key in ("kind", "stage", "freeze_sha256", "report_sha256", "measurement_status", "measured_result_pending")}
            if "development_validation" in report:
                result["development_validation"] = report["development_validation"]
        elif args.command in {"inspect-checkpoint", "edit-checkpoint"}:
            from .checkpoint import CheckpointEditor
            editor = CheckpointEditor(args.model, read_json(args.policy), exposure_log=args.log)
            if args.command == "inspect-checkpoint":
                result = editor.inspect(read_json(args.request))
            else:
                from .inventory import file_sha256
                audit = editor.propose(read_json(args.patch), args.out)
                result = {"accepted": True, "result_sha256": audit["result_sha256"],
                          "audit_sha256": file_sha256(args.out / "audit.json")}
        elif args.command == "verify-candidate":
            from .checkpoint import verify_candidate
            from .inventory import file_sha256
            audit = verify_candidate(args.model, args.candidate, read_json(args.policy))
            result = {"verified": True, "result_sha256": audit["result_sha256"],
                      "audit_sha256": file_sha256(args.candidate / "audit.json")}
        elif args.command == "demo-candidate-edit":
            from .candidate_demo import demo_candidate_edit
            result = demo_candidate_edit(args.model, args.out, backend=args.backend, tensor=args.tensor)
        elif args.command == "acquire-benchmarks":
            from .benchmark_data import acquire_benchmarks
            result = acquire_benchmarks(args.reference, args.out,
                                       max_download_bytes=args.max_download_bytes, timeout=args.timeout)
        elif args.command == "prepare-benchmark-data":
            from .benchmark_data import prepare_benchmark_data
            result = prepare_benchmark_data(args.reference, args.cache, args.out)
        elif args.command == "check-benchmark-bundle":
            from .benchmark_data import check_benchmark_bundle
            result = check_benchmark_bundle(args.bundle, cache=args.cache)
        elif args.command == "acquire-ifeval":
            from .grading import acquire_ifeval
            result = acquire_ifeval(args.out)
        elif args.command in {"check-graders", "freeze-analysis", "grade-records", "project-analysis", "compare-graded"}:
            if args.out.exists():
                raise ValueError("output exists; choose a new artifact destination")
            if args.command == "check-graders":
                from .grading import run_known_answer_checks
                result = run_known_answer_checks(ifeval_root=args.ifeval_root)
            elif args.command == "freeze-analysis":
                from .analysis_pipeline import build_analysis_manifest
                result = build_analysis_manifest(
                    read_json(args.split_manifest), read_json(args.items),
                    benchmark=args.benchmark, seeds=read_json(args.seeds), policy=read_json(args.policy),
                    arms=read_json(args.arms), stage=args.stage, metric=args.metric,
                    evidence_kind=args.evidence_kind)
            elif args.command == "grade-records":
                from .benchmark_data import validate_item
                from .grading import grade_record, validate_grading_policy
                from .records import validate_run_records
                records = validate_run_records(read_json(args.records))
                policy = validate_grading_policy(read_json(args.policy))
                items = read_json(args.items)
                if not isinstance(items, list):
                    raise ValueError("items must be an array")
                indexed = {}
                for item in items:
                    item = validate_item(item)
                    key = item["benchmark"], item["item_id"]
                    if key in indexed:
                        raise ValueError("duplicate benchmark item")
                    indexed[key] = item
                if any((row["benchmark"], row["item_id"]) not in indexed for row in records):
                    raise ValueError("raw record references an unknown benchmark item")
                result = [grade_record(row, indexed[row["benchmark"], row["item_id"]], policy,
                                       ifeval_root=args.ifeval_root) for row in records]
            elif args.command == "project-analysis":
                from .analysis_pipeline import project_arm
                result = project_arm(read_json(args.records), read_json(args.grades),
                                     read_json(args.manifest), arm=args.arm)
            else:
                from .analysis_pipeline import compare_graded_runs
                result = compare_graded_runs(
                    read_json(args.baseline_records), read_json(args.candidate_records),
                    read_json(args.baseline_grades), read_json(args.candidate_grades),
                    read_json(args.manifest), bootstrap_samples=args.bootstrap_samples, seed=args.seed)
            write_json(args.out, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "check-graders" and not result["all_passed"]:
            return 2
        if args.command == "grade-records" and any(row["status"] in {"grader_error", "grader_timeout"} for row in result):
            return 2
        if args.command == "run-native" and any(result["statuses"][key] for key in ("error", "timeout", "missing")):
            return 2
        if args.command in {"replay-smoke", "replay-pilot", "replay-full"} and not result["successful_execution"]:
            return 2
        if args.command == "report-published-replay" and "development_validation" in result and not result["development_validation"]["passed"]:
            return 2
        if args.command == "demo-candidate-edit" and any(status not in {"ok", "truncated"} for status in result["generation_statuses"].values()):
            return 2
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        # Avoid dumping local machine paths or raw provider errors into reports.
        if isinstance(error, OSError):
            print(json.dumps({"error": "file operation failed", "type": type(error).__name__}))
        elif isinstance(error, KeyError):
            print(json.dumps({"error": "artifact has missing required fields", "type": "KeyError"}))
        else:
            print(json.dumps({"error": str(error), "type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
