"""Explicit, offline-frozen published-patch replay orchestration.

Preparation never generates. Execution never selects settings. Reports retain
all scheduled cases. Final replay artifacts live in a reserved evaluator root;
this path guard is not the later editor's OS/tool access boundary.
"""
from __future__ import annotations

import copy
import re
import shutil
import tempfile
import time
from pathlib import Path

from .acquisition import verify_acquisition
from .analysis_pipeline import build_analysis_manifest, validate_analysis_manifest
from .benchmark_data import check_benchmark_bundle, load_source, to_generation_tasks
from .grading import grader_identity, validate_grading_policy
from .inventory import file_sha256, read_json
from .native_assets import (backend_spec, implementation_identity, runtime_environment,
                            validate_execution_config, verify_identity, _write)
from .protocol import build_benchmark_schedule, canonical_hash
from .reconstruction import verify_reconstruction

BENCHMARKS = ("math", "gsm8k", "ifeval")
ARMS = ("original", "identity", "reconstructed", "original_temperature1", "reconstructed_temperature1")
COMPARISONS = (("primary", "primary", "original", "reconstructed"),
               ("identity", "identity_control", "original", "identity"),
               ("temperature1", "alternative_decoding", "original_temperature1", "reconstructed_temperature1"),
               ("decoding_original", "alternative_decoding", "original", "original_temperature1"))
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DISCREPANCIES = [
    "Published-patch replay, regardless of agreement with the published improvement; no fresh editor discovery claim.",
    "The full conditional-generation checkpoint is retained; upstream used a text-only export.",
    "Our native runtime versions are pinned now; the upstream installed vLLM version was not pinned.",
    "Each sample is a separate request with its own explicit seed; upstream used n samples with shared seed 0. Random streams differ.",
    "Explicit EOS union [248044,248046]; upstream exported implicit generation metadata was not fully pinned.",
    "Independent pinned graders differ from upstream: Qwen output extraction requires a closed thinking block; upstream math grading could inspect unclosed reasoning.",
    "GSM8K uses the declared Decimal numeric final-answer policy, while upstream used Math-Verify for its last boxed answer.",
    "IFEval uses all 541 official source prompts for replay, including development examples; this is not a fresh held-out IFEval estimate.",
    "Math-Verify, official IFEval code/resources, extraction, retries and truncation rules are independently pinned; no exact upstream evaluator-environment reproduction is claimed.",
    "Identity control is an independently reopened byte copy, not a framework reserialization.",
    "Smoke and pilot use short output caps and development subsets; their scores do not estimate full benchmark correctness.",
    "Temperature 1.0 is a predeclared alternative, not a development-selected winner or the primary comparison.",
]


def _sealed(value, field):
    return {**value, field: canonical_hash(value)}


def _check_seal(value, field):
    if value.get(field) != canonical_hash({k: v for k, v in value.items() if k != field}):
        raise ValueError("frozen artifact digest mismatch")


def guard_workspace(root, fresh_editor_workspace=None, *, create=False):
    """Reject containment and symlink aliases, on preparation AND every use."""
    root = Path(root)
    marker = root / "workspace-boundary.json"
    if root.is_symlink() or marker.is_symlink():
        raise ValueError("replay root and boundary must not be symlinks")
    if marker.exists():
        boundary = read_json(marker)
        if set(boundary) != {"kind", "fresh_editor_workspace"} or boundary["kind"] != "replay_only_workspace":
            raise ValueError("invalid replay workspace boundary")
        reserved = Path(boundary["fresh_editor_workspace"])
        if fresh_editor_workspace is not None and Path(fresh_editor_workspace).absolute() != reserved:
            raise ValueError("fresh-editor workspace reservation changed")
    elif create and fresh_editor_workspace is not None:
        reserved = Path(fresh_editor_workspace).absolute()
    else:
        raise ValueError("a separate fresh-editor workspace must be reserved explicitly")
    actual, editor = root.resolve(), reserved.resolve()
    if actual == editor or actual in editor.parents or editor in actual.parents:
        raise ValueError("replay and fresh-editor workspaces must be disjoint")
    if not marker.exists():
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(root.iterdir()):
            raise ValueError("new replay workspace must be empty")
        _write(marker, {"kind": "replay_only_workspace", "fresh_editor_workspace": str(reserved)})
    return root


def _inside(root, path):
    path = Path(path)
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError("replay artifact escapes its dedicated workspace")
    return path


def _stage(stage):
    if stage not in {"smoke", "pilot", "full"}:
        raise ValueError("replay stage must be smoke, pilot or full")
    return stage


def _plan(path):
    plan = read_json(path)
    # This implementation supports one predeclared study, not a post-test optimizer.
    expected = {"math": (500, [0, 1, 2, 3], "accuracy"), "gsm8k": (1319, [0, 1], "accuracy"),
                "ifeval": (541, [0, 1], "prompt_level_strict_accuracy")}
    fields = {"schema_version", "kind", "primary_comparison", "primary_benchmark", "decoding_selection",
              "alternative_decoding", "stages", "full_benchmarks", "analysis", "published_reference_scores", "reference_revision"}
    if (set(plan) != fields or plan["schema_version"] != 1 or plan["kind"] != "published_patch_replay_plan"
            or plan["primary_comparison"] != "unchanged_vs_reconstructed_p50c"
            or plan["primary_benchmark"] != "math"
            or plan["decoding_selection"] != "none; all settings fixed a priori, no development or test score selection"
            or plan["alternative_decoding"] != "temperature_1_thinking; only temperature changes from 0.6 to 1.0"
            or set(plan["full_benchmarks"]) != set(BENCHMARKS)
            or set(plan["stages"]) != {"smoke", "pilot", "full"}
            or plan["analysis"] != {"bootstrap_samples": 10000, "seed": 0, "confidence_level": 0.95,
                "unit": "paired_source_item", "multiplicity": "math primary; GSM8K, IFEval and controls descriptive; no pooled benchmark score"}):
        raise ValueError("unsupported replay plan or analysis selection")
    for name, (count, seeds, metric) in expected.items():
        if plan["full_benchmarks"][name] != {"items": count, "seeds": seeds, "metric": metric}:
            raise ValueError("full published benchmark schedule changed")
    stages = {"smoke": (1, [0], 16, 180), "pilot": (2, [0], 128, 600), "full": (None, None, 32768, 7200)}
    for name, (count, seeds, cap, timeout) in stages.items():
        if plan["stages"][name] != dict(items_per_benchmark=count, seeds=seeds, max_new_tokens=cap, request_timeout_seconds=timeout):
            raise ValueError("stage limits differ from the predeclared replay plan")
    return plan


def replay_tasks(items):
    tasks = to_generation_tasks(items)
    for task in tasks:
        if task["benchmark"] in {"math", "gsm8k"}:
            task["messages"][0]["content"] += MATH_SUFFIX
    return tasks


def _selected_items(data, cache, stage, plan):
    reference = read_json(Path(data) / "source-reference.json")
    if stage == "full":
        # Do not reuse the 66-prompt held-out Phase 4 subset as official IFEval.
        items = []
        for spec in reference["sources"]:
            if spec["role"] == "final" or spec["benchmark"] == "ifeval":
                for item in load_source(spec, cache):
                    item["split"] = "test"
                    items.append(item)
        for name in BENCHMARKS:
            if sum(i["benchmark"] == name for i in items) != plan["full_benchmarks"][name]["items"]:
                raise ValueError("full source coverage differs from the published replay schedule")
    else:
        development = read_json(Path(data) / "editor-development" / "items.json")
        if any(i["split"] != "development" for i in development):
            raise ValueError("development selection contains held-out data")
        count = plan["stages"][stage]["items_per_benchmark"]
        items = []
        for name in BENCHMARKS:
            selected = sorted((i for i in development if i["benchmark"] == name), key=lambda i: i["item_id"])[:count]
            if len(selected) != count:
                raise ValueError("insufficient development items")
            items.extend(selected)
    return sorted(items, key=lambda i: (i["benchmark"], i["item_id"]))


def _split(items, source_hash):
    return {"schema_version": 1, "kind": "benchmark_split", "split": items[0]["split"],
            "source_manifest_sha256": source_hash,
            "items": [{"benchmark": i["benchmark"], "item_id": i["item_id"],
                       "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in items]}


def _arm_identity(config, tasks, schedule):
    return {"run_id": config["run_id"], "profile": config["profile"], "checkpoint": config["checkpoint"],
            "configuration_sha256": canonical_hash({"configuration": config, "tasks": tasks, "schedule": schedule})}


def load_study(root, stage, *, execution=False):
    root, stage = guard_workspace(root), _stage(stage)
    study = read_json(_inside(root, root / stage / "inputs" / "study.json"))
    _check_seal(study, "freeze_sha256")
    if study["kind"] != "published_patch_replay" or study["stage"] != stage:
        raise ValueError("study stage or claim mismatch")
    if study["declaration"]["implementation_sha256"] != implementation_identity():
        raise ValueError("study source changed; prepare and validate a new study")
    for name in BENCHMARKS:
        if study["declaration"]["graders"][name] != grader_identity(name):
            raise ValueError("grader identity changed after freeze")
    expected_jobs = {f"{name}-{arm}" for name in BENCHMARKS for arm in ARMS}
    if (set(study["jobs"]) != expected_jobs or set(study["benchmarks"]) != set(BENCHMARKS)
            or len(study["execution_order"]) != len(expected_jobs) or set(study["execution_order"]) != expected_jobs):
        raise ValueError("frozen job coverage or execution order mismatch")
    for name, job in study["jobs"].items():
        if name != f"{job['benchmark']}-{job['label']}" or job["arm"] != job["label"].removesuffix("_temperature1"):
            raise ValueError("job weight identity mismatch")
        inputs = study["benchmarks"][job["benchmark"]]
        if inputs["tasks"] != replay_tasks(inputs["items"]):
            raise ValueError("frozen prompt wrapping disagrees with source tasks")
        if job["config"]["tasks_sha256"] != canonical_hash(inputs["tasks"]) or job["config"]["schedule_sha256"] != canonical_hash(inputs["schedule"]):
            raise ValueError("job task or schedule identity mismatch")
        if execution:
            validate_execution_config(job["config"], inputs["tasks"], inputs["schedule"])
    expected_comparisons = {f"{name}-{suffix}" for name in BENCHMARKS for suffix, _, _, _ in COMPARISONS}
    if len(study["comparisons"]) != len(expected_comparisons) or {c["id"] for c in study["comparisons"]} != expected_comparisons:
        raise ValueError("frozen comparison coverage mismatch")
    for comparison in study["comparisons"]:
        manifest = validate_analysis_manifest(comparison["manifest"])
        name = manifest["benchmark"]
        inputs = study["benchmarks"][name]
        suffix, role, left, right = next(c for c in COMPARISONS if f"{name}-{c[0]}" == comparison["id"])
        if (comparison["role"], comparison["baseline"], comparison["candidate"]) != (role, f"{name}-{left}", f"{name}-{right}"):
            raise ValueError("comparison labels or arms differ from predeclaration")
        arms = {key: _arm_identity(study["jobs"][comparison[value]]["config"], inputs["tasks"], inputs["schedule"])
                for key, value in (("baseline", "baseline"), ("candidate", "candidate"))}
        seeds = [r["seed"] for r in inputs["schedule"] if r["item_id"] == inputs["schedule"][0]["item_id"]]
        expected = build_analysis_manifest(inputs["split"], inputs["items"], benchmark=name, seeds=seeds,
            policy=study["declaration"]["policy"], arms=arms, stage=stage,
            metric=study["declaration"]["plan"]["full_benchmarks"][name]["metric"])
        if manifest != expected or manifest["schedule"] != inputs["schedule"]:
            raise ValueError("analysis differs from the frozen execution inputs")
    return study


def _development_gate(root, path, binding):
    if path is None:
        raise ValueError("pilot/full freeze requires a completed development smoke report")
    path = _inside(root, path)
    smoke = load_study(root, "smoke")
    report = read_json(path)
    _check_seal(report, "report_sha256")
    certificate = report.get("development_validation", {})
    if (report.get("stage") != "smoke" or report.get("freeze_sha256") != smoke["freeze_sha256"]
            or certificate.get("passed") is not True or certificate.get("binding_sha256") != binding
            or smoke["declaration"]["binding_sha256"] != binding):
        raise ValueError("development validation is absent, failed or bound to different study settings")
    return canonical_hash(report)


def prepare_study(*, root, fresh_editor_workspace, acquisition, reconstruction, identity, data, cache,
                  backend, stage, reference="configs/published_p50c.json",
                  plan_path="configs/published_replay_plan.json", policy_path="configs/grading-policy.json",
                  development_report=None, smoke_measurement="reports/native-smoke/report.json"):
    """Verify local inputs and atomically freeze a stage; never launch inference."""
    stage, plan = _stage(stage), _plan(plan_path)
    root = guard_workspace(root, fresh_editor_workspace, create=True)
    target = _inside(root, root / stage / "inputs")
    if target.exists():
        raise ValueError("stage already frozen; use a new replay workspace")
    acquisition, reconstruction, identity, data = map(Path, (acquisition, reconstruction, identity, data))
    for path in (acquisition, reconstruction, identity, data, Path(cache)):
        if path.resolve() == root.resolve() or root.resolve() in path.resolve().parents or path.resolve() in root.resolve().parents:
            raise ValueError("replay output must not overlap its source artifact bundles")
    selected = read_json(reference)
    if plan["reference_revision"] != selected["source_revision"]:
        raise ValueError("plan and published patch source revisions disagree")
    acquired = verify_acquisition(acquisition, reference)
    rebuilt = verify_reconstruction(acquisition, reconstruction, reference)
    copied = verify_identity(acquisition, identity, reference)
    data_summary = check_benchmark_bundle(data, cache=cache)
    source_reference = read_json(data / "source-reference.json")
    inventories = {"original": acquired["model_inventory"], "identity": copied,
                   "reconstructed": rebuilt["model_inventory"]}
    audit_paths = {"original": acquisition / "acquisition.json", "identity": identity / "audit.json",
                   "reconstructed": reconstruction / "audit.json"}
    policy = validate_grading_policy(read_json(policy_path))
    # This replay fixes all output-processing decisions; changes require a new implementation/study.
    if policy != {"schema_version": 1, "reasoning_policy": "qwen_thinking",
        "output_eos_tokens": ["<|im_end|>", "<|endoftext|>"], "truncation_policy": "grade",
        "generation_retry": {"mode": "none", "max_attempts": 1}, "grader_timeout_seconds": 20,
        "grader_retry": {"mode": "none", "max_attempts": 1}}:
        raise ValueError("grading policy differs from the predeclared replay policy")
    template = inventories["original"]["chat_templates"]["chat_template.jinja"]["sha256"]
    spec = backend_spec(backend, template)
    from .grading import MATH_PACKAGES, IFEVAL_PACKAGES, IFEVAL_FILES, IFEVAL_REVISION, NLTK_DATA_REVISION, NLTK_ARCHIVE_SHA256
    grader_contract = {"math_packages": MATH_PACKAGES, "ifeval_packages": IFEVAL_PACKAGES,
        "ifeval_files": IFEVAL_FILES, "ifeval_revision": IFEVAL_REVISION,
        "nltk_data_revision": NLTK_DATA_REVISION, "nltk_archive_sha256": NLTK_ARCHIVE_SHA256,
        "math_extraction": "last balanced boxed expression after closed thinking block; pinned Math-Verify",
        "gsm8k_extraction": "last explicit #### or boxed numeric answer; Decimal equivalence; no last-number fallback",
        "ifeval_extraction": "final text after closed thinking block; official strict/loose prompt/instruction checks",
        "prompt_truncation": "reject over-context requests; never silently truncate input",
        "output_truncation": "grade final extracted answer if available; never expose unclosed reasoning",
        "missing_or_model_failure": "retain every scheduled denominator with zero credit",
        "grader_failure": "retain denominator and report unresolved zero-credit bound, not identified correctness"}
    binding = {"plan": plan, "runtime_environment": runtime_environment(), "base_backend": spec,
               "implementation_sha256": implementation_identity(), "policy": policy,
               "graders": {name: grader_identity(name) for name in BENCHMARKS},
               "grader_contract": grader_contract,
               "checkpoint_inventories": {arm: canonical_hash(inv) for arm, inv in inventories.items()},
               "source_reference": source_reference, "data_summary": data_summary,
               "model_reference": selected}
    binding_hash = canonical_hash(binding)
    development_hash = None if stage == "smoke" else _development_gate(root, development_report, binding_hash)
    items = _selected_items(data, cache, stage, plan)
    declaration = {**binding, "binding_sha256": binding_hash, "development_report_sha256": development_hash,
                   "items_sha256": canonical_hash(items), "stage": stage,
                   "discrepancies": DISCREPANCIES + (["Transformers CPU eager BF16 with preserved F32 tensors is a methodological variation from the released vLLM runtime."] if backend == "transformers" else []),
                   "output_audience": "replay evaluator only; exclude from every later fresh-editor workspace",
                   "access_boundary_enforced": False}
    declaration_id = canonical_hash(declaration)
    benchmarks, jobs, comparisons = {}, {}, []
    settings = plan["stages"][stage]
    for name in BENCHMARKS:
        subset = [i for i in items if i["benchmark"] == name]
        seeds = plan["full_benchmarks"][name]["seeds"] if stage == "full" else settings["seeds"]
        schedule = build_benchmark_schedule(name, [i["item_id"] for i in subset], seeds)
        tasks = replay_tasks(subset)
        split = _split(subset, canonical_hash(source_reference))
        benchmarks[name] = {"items": subset, "tasks": tasks, "schedule": schedule, "split": split}
        for label in ARMS:
            arm = label.removesuffix("_temperature1")
            inventory = inventories[arm]
            backend_config = copy.deepcopy(spec)
            backend_config["max_model_len"] = settings["max_new_tokens"] + 2048
            backend_config["sampling"]["max_new_tokens"] = settings["max_new_tokens"]
            if label.endswith("_temperature1"):
                backend_config["sampling"]["temperature"] = 1.0
            job_id = f"{name}-{label}"
            config = {"schema_version": 1, "kind": "native_execution", "run_id": f"replay-{stage}-{job_id}",
                      "evidence_kind": "real_model", "profile": "native_development_smoke" if stage == "smoke" else "fixed_patch_evaluation",
                      "stage": stage, "checkpoint": {"model_id": selected["base_model"], "revision": selected["base_revision"],
                                                       "sha256": canonical_hash(inventory["files"])},
                      "retry_policy": policy["generation_retry"],
                      "execution_limits": {"max_sequences": len(schedule), "startup_timeout_seconds": 300,
                          "request_timeout_seconds": settings["request_timeout_seconds"],
                          "max_duration_seconds": 300 + len(schedule) * settings["request_timeout_seconds"]},
                      "backend": backend_config, "runtime_environment": binding["runtime_environment"],
                      "artifacts": {"files": inventory["files"], "inventory_sha256": canonical_hash(inventory),
                          "tokenizer_revision": selected["base_revision"], "audit_sha256": file_sha256(audit_paths[arm]),
                          "arm": arm, "source_revision": selected["source_revision"]},
                      "implementation_sha256": binding["implementation_sha256"],
                      "tasks_sha256": canonical_hash(tasks), "schedule_sha256": canonical_hash(schedule),
                      "methodological_variations": declaration["discrepancies"]}
            if stage != "smoke":
                config["study"] = {"kind": "published_patch_replay", "freeze_id": declaration_id,
                    "grader_revision": binding["graders"][name]["revision"], "policy_sha256": canonical_hash(policy),
                    "patch_sha256": file_sha256(acquisition / "patch" / "edit.npz"),
                    "reconstruction_audit_sha256": file_sha256(reconstruction / "audit.json"),
                    "evaluation_code_revision": binding["implementation_sha256"],
                    "split_audit_sha256": data_summary["overlap_audit_sha256"],
                    "primary_comparison": plan["primary_comparison"], "truncation_policy": policy["truncation_policy"]}
            validate_execution_config(config, tasks, schedule)
            jobs[job_id] = {"benchmark": name, "arm": arm, "label": label, "config": config}
        for suffix, role, left, right in COMPARISONS:
            pair = {key: _arm_identity(jobs[f"{name}-{label}"]["config"], tasks, schedule)
                    for key, label in (("baseline", left), ("candidate", right))}
            manifest = build_analysis_manifest(split, subset, benchmark=name, seeds=seeds, policy=policy,
                arms=pair, stage=stage, metric=plan["full_benchmarks"][name]["metric"])
            comparisons.append({"id": f"{name}-{suffix}", "role": role, "baseline": f"{name}-{left}",
                                "candidate": f"{name}-{right}", "manifest": manifest})
    from .replay_report import estimate_workload
    workload = [{"benchmark": name, "arm": arm, "scheduled_generations": values["items"] * len(values["seeds"]),
                 "max_new_tokens": plan["stages"]["full"]["max_new_tokens"]}
                for name, values in plan["full_benchmarks"].items() for arm in ARMS]
    measured = read_json(smoke_measurement)
    estimate = estimate_workload(measured, workload, target_backend=backend)
    execution_order = [f"{name}-{arm}" for index, name in enumerate(BENCHMARKS)
                       for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))]
    study = _sealed({"schema_version": 1, "kind": "published_patch_replay", "stage": stage,
                     "declaration": declaration, "benchmarks": benchmarks, "jobs": jobs,
                     "execution_order": execution_order,
                     "comparisons": comparisons, "workload_estimate": estimate,
                     "smoke_measurement_sha256": canonical_hash(measured)}, "freeze_sha256")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=target.parent))
    try:
        _write(temporary / "study.json", study)
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"kind": "published_patch_replay", "stage": stage, "freeze_sha256": study["freeze_sha256"],
            "jobs": len(jobs), "scheduled_generations": sum(len(benchmarks[j["benchmark"]]["schedule"]) for j in jobs.values()),
            "inference_executed": False, "measured_result": "pending"}


def run_study(root, stage, *, acquisition, reconstruction, identity, resume=False, job=None):
    """Execute only the explicitly named stage, with validated resume per job."""
    from .native_runner import run_schedule
    root = guard_workspace(root)
    study = load_study(root, stage, execution=True)
    if job is not None and job not in study["jobs"]:
        raise ValueError("unknown frozen replay job")
    models = {"original": Path(acquisition) / "model", "identity": Path(identity) / "model",
              "reconstructed": Path(reconstruction) / "model"}
    results = {}
    # Alternate weight order across benchmarks; this ordering is fixed before execution.
    names = study["execution_order"]
    for name in names:
        if job is not None and name != job:
            continue
        entry = study["jobs"][name]
        inputs = study["benchmarks"][entry["benchmark"]]
        output = _inside(root, root / stage / "runs" / name)
        results[name] = run_schedule(entry["config"], inputs["tasks"], inputs["schedule"], output,
            model_path=models[entry["arm"]], resume=resume and output.exists())
    return {"kind": "published_patch_replay_execution", "stage": stage, "freeze_sha256": study["freeze_sha256"],
            "jobs": results, "successful_execution": all(not any(r["statuses"][s] for s in ("error", "timeout", "missing")) for r in results.values()),
            "measured_result": "awaiting_independent_grading"}


def _read_run(root, stage, name, entry, inputs):
    from .native_runner import _inputs, _Journal, _lock
    output = _inside(root, root / stage / "runs" / name)
    if not output.exists():
        return [], None, []
    manifest, empty, _ = _inputs(entry["config"], inputs["tasks"], inputs["schedule"])
    with _lock(output):
        journal = _Journal(output, manifest, empty, resume=True)
        if journal.pending is not None:
            raise ValueError("unfinished native attempt: resume its stage/job to reconcile the journal before reporting")
        records = copy.deepcopy(journal.records)
        execution = {"accounted_wall_seconds": journal.consumed_seconds,
                     "pending_attempts": int(journal.pending is not None)}
        events = copy.deepcopy(journal.events)
    return records, execution, events


def report_study(root, stage, *, report_id, ifeval_root=None):
    """Grade independently and save a new snapshot, including unattempted cases.

    Immutable grade cache files bind record, grader and policy hashes. Restarting
    reporting reuses completed grading, including recorded evaluator failures.
    It never repeats inference or selects a better score.
    """
    from .grading import grade_record, run_known_answer_checks, validate_grade_record
    from .records import run_record_sha256
    from .native_runner import _write_json
    from .replay_report import summarize_comparison, build_replay_report, render_replay_markdown
    root = guard_workspace(root)
    report_started = time.monotonic()
    new_grades, cached_grades, grading_seconds = 0, 0, 0.0
    if not isinstance(report_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", report_id) is None:
        raise ValueError("report id must be a portable identifier")
    study = load_study(root, stage)
    output = _inside(root, root / stage / "reports" / report_id)
    if output.exists():
        raise ValueError("report already exists; use a new report id")
    policy = study["declaration"]["policy"]
    raw, grades, executions, events = {}, {}, {}, {}
    for name, entry in study["jobs"].items():
        inputs = study["benchmarks"][entry["benchmark"]]
        raw[name], executions[name], events[name] = _read_run(root, stage, name, entry, inputs)
        indexed = {i["item_id"]: i for i in inputs["items"]}
        grades[name] = []
        for record in raw[name]:
            key = canonical_hash({"record": run_record_sha256(record), "policy": policy,
                                  "grader": grader_identity(entry["benchmark"])})
            path = _inside(root, root / stage / "grade-cache" / name / f"{key}.json")
            if path.exists():
                grade = read_json(path)
                validate_grade_record(grade)
                cached_grades += 1
            else:
                grade_started = time.monotonic()
                grade = grade_record(record, indexed[record["item_id"]], policy, ifeval_root=ifeval_root)
                grading_seconds += time.monotonic() - grade_started
                new_grades += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(path, grade, immutable=True)
            grades[name].append(grade)
    analysis = study["declaration"]["plan"]["analysis"]
    comparisons = [summarize_comparison(c["manifest"], raw[c["baseline"]], raw[c["candidate"]],
        grades[c["baseline"]], grades[c["candidate"]], comparison_id=c["id"], comparison_role=c["role"],
        baseline_label=c["baseline"], candidate_label=c["candidate"],
        bootstrap_samples=analysis["bootstrap_samples"], seed=analysis["seed"],
        baseline_execution=executions[c["baseline"]], candidate_execution=executions[c["candidate"]]) for c in study["comparisons"]]
    result = build_replay_report(comparisons, stage=stage, freeze_sha256=study["freeze_sha256"],
        workload_estimate=study["workload_estimate"], discrepancies=study["declaration"]["discrepancies"],
        required_comparison_ids=[c["id"] for c in study["comparisons"]])
    identity_checks = {}
    for name in BENCHMARKS:
        def token_outputs(label):
            return [e["event"].get("metadata", {}).get("generation", {}).get("output_token_ids")
                    for e in events[f"{name}-{label}"] if e["event"]["kind"] == "attempt_finished"]
        left, right = token_outputs("original"), token_outputs("identity")
        count = len(study["benchmarks"][name]["schedule"])
        checked = len(left) == len(right) == count and all(v is not None for v in left + right)
        identity_checks[name] = {"checked": checked, "token_ids_equal": checked and left == right}
    result["identity_token_checks"] = identity_checks
    result["published_reference_scores"] = study["declaration"]["plan"]["published_reference_scores"]
    result["analysis_configuration"] = analysis
    result["distinct_job_execution_accounting"] = executions
    result["evidence_hashes"] = {name: {"records": canonical_hash(raw[name]), "grades": canonical_hash(grades[name]),
                                      "events": canonical_hash(events[name])} for name in raw}
    if stage == "smoke":
        known = run_known_answer_checks(ifeval_root=ifeval_root)
        complete = all(len(raw[name]) == len(study["benchmarks"][j["benchmark"]]["schedule"])
                       and all(r["status"] in {"ok", "truncated"} for r in raw[name]) for name, j in study["jobs"].items())
        resolved = all(g["status"] == "graded" for rows in grades.values() for g in rows)
        result["development_validation"] = {"passed": complete and resolved and known["all_passed"]
            and all(v["token_ids_equal"] for v in identity_checks.values()),
            "binding_sha256": study["declaration"]["binding_sha256"], "generation_paths_complete": complete,
            "grading_resolved": resolved, "known_answer_checks_passed": known["all_passed"],
            "meaning": "development mechanics validated, including short truncation; not long-context fit or accuracy validation"}
    result["reporting_runtime"] = {"new_grades": new_grades, "cached_grades": cached_grades,
        "new_grade_wall_seconds": grading_seconds, "report_wall_seconds_before_serialization": time.monotonic() - report_started,
        "cache_note": "prior grading runtime is not remeasured; each cached grade is reused without retry"}
    result = _sealed(result, "report_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".report-", dir=output.parent))
    try:
        _write(temporary / "report.json", result)
        if stage == "smoke":
            _write(temporary / "known-answer-checks.json", known)
        for name in raw:
            (temporary / name).mkdir()
            _write(temporary / name / "records.json", raw[name])
            _write(temporary / name / "grades.json", grades[name])
        (temporary / "README.md").write_text(render_replay_markdown(result), encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result
