"""Original-style weight research followed by one joint IFEval/WebLLM pair.

Python and source-model diagnostic probes have separate development budgets.
Final prompts are withheld from the editor interface. Freeze closes development
before either final browser arm runs. Historical study controllers are unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from pathlib import Path
import re
import os
import sys
import time
import uuid

from .inventory import file_sha256, inventory_model, read_json
from .mechanistic_study import _write, _task_config, prepare_benchmark
from .native_runner import _lock
from .protocol import canonical_hash

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/ifeval_research_browser.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_BUDGETS = {"python_calls": 300, "forward_calls": 50, "snapshots": 300,
            "restores": 300, "kernel_starts": 8, "candidate_submissions": 1,
            "conversions": 1, "browser_runs": 2, "browser_seconds_per_arm": 5400,
            "automatic_retries": 0}


def validate_config(value):
    config = copy.deepcopy(value)
    if (config.get("schema_version") != 1 or config.get("accuracy_benchmark") != "ifeval"
            or config.get("scope") != "practice_then_frozen_comparison"):
        raise ValueError("Research workflow requires IFEval practice and one frozen comparison")
    schedule = config.get("schedule", {})
    if (schedule.get("arms") != ["baseline", "candidate"] or schedule.get("runs_per_arm") != 1
            or schedule.get("warmups_per_arm") != 1 or schedule.get("separate_resource_workload") is not False
            or type(schedule.get("maximum_prompts")) is not int or not 1 <= schedule["maximum_prompts"] <= 128):
        raise ValueError("Exactly one final baseline/candidate pair on the same workload is required")
    limits = config.get("limits", {})
    if set(limits) != set(_BUDGETS):
        raise ValueError("Explicit development and final budgets are required")
    for key, maximum in _BUDGETS.items():
        if type(limits[key]) is not int or not 0 <= limits[key] <= maximum:
            raise ValueError("Invalid research budget: " + key)
    for key in ("candidate_submissions", "conversions", "browser_runs", "automatic_retries"):
        if limits[key] != _BUDGETS[key]:
            raise ValueError("One freeze, one conversion, two final browser arms, no automatic retries")
    if limits["browser_seconds_per_arm"] < 140 or config.get("editor", {}).get("max_submissions") != 1:
        raise ValueError("Final comparison allowance/submission count differs")
    if any(limits[key] < 1 for key in ("python_calls", "forward_calls", "snapshots", "restores", "kernel_starts")):
        raise ValueError("Development must have positive bounded allowances")
    if not isinstance(config.get("kernel_limits"), dict) or not isinstance(config.get("probe_limits"), dict):
        raise ValueError("Kernel and forward limits must be explicit objects")
    return config


def validate_splits(practice, final, maximum=128):
    from .benchmark_data import audit_overlap, validate_item
    if (not isinstance(practice, list) or not practice or len(practice) > 16384
            or not isinstance(final, list) or not 1 <= len(final) <= maximum):
        raise ValueError("Provide nonempty bounded practice and final IFEval sets")
    for item in practice + final:
        validate_item(item)
        if item["benchmark"] != "ifeval":
            raise ValueError("IFEval is the only accuracy benchmark")
    if any(i["split"] not in ("development", "validation") for i in practice) or any(i["split"] != "test" for i in final):
        raise ValueError("Practice uses development/validation; the withheld final workload uses test")
    # Practice development and validation are both editor-visible here. Preserve
    # caller records, but audit their union against the withheld final set.
    audited = [{**i, "split": "development"} for i in practice] + copy.deepcopy(final)
    overlap = audit_overlap(audited)
    if not overlap["passed"]:
        raise ValueError("Practice/final overlap in IDs, normalized prompts or instruction families")
    source_ids = lambda rows: {(i["source"]["dataset_id"], i["source"]["split"], i["source"]["row_id"]) for i in rows}
    if source_ids(practice) & source_ids(final):
        raise ValueError("Practice/final source rows overlap")
    return overlap


def prepare(root, *, source_model, baseline_manifest, baseline_config, benchmark,
            practice_items, final_items, official_ifeval_root, config_path=DEFAULT_CONFIG, probe_python=None):
    from .browser_candidate import verify_baseline
    from .browser_resource import environment_identity
    from .grading import verify_ifeval_assets
    from .ifeval_baseline import SETTINGS
    from .research_checkpoint import EditWorkspace
    from .research_kernel import validate_limits as validate_kernel_limits
    from .research_probes import DEFAULT_LIMITS as PROBE_LIMITS, validate_limits
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Prepare a new research study; never reopen a historical ledger")
    config = validate_config(read_json(config_path))
    config["kernel_limits"] = validate_kernel_limits(config["kernel_limits"])
    config["probe_limits"] = validate_limits({**PROBE_LIMITS, **config["probe_limits"]})
    practice, final = read_json(practice_items), read_json(final_items)
    overlap = validate_splits(practice, final, config["schedule"]["maximum_prompts"])
    source = Path(source_model).resolve()
    inventory = inventory_model(source)
    source_sha = canonical_hash(inventory["files"])
    deployed = verify_baseline(baseline_manifest)
    base = deployed["arms"]["baseline"]
    if base["source_checkpoint_sha256"] != source_sha:
        raise ValueError("Browser baseline must derive from the exact inspected source checkpoint")
    template = read_json(baseline_config)
    template_items = {item["item_id"]: item for item in template["items"]}
    if (template["settings"] != SETTINGS or template["sourceCheckpointSha256"] != source_sha
            or template["convertedArtifactSha256"] != base["artifact_sha256"]):
        raise ValueError("Baseline template must bind the frozen deployment and browser settings")
    for item in final:
        row = template_items.get(item["item_id"])
        if (row is None or row["prompt"] != item["task_text"]
                or type(row.get("rendered_input_tokens")) is not int or not 1 <= row["rendered_input_tokens"] <= 3072):
            raise ValueError("Final prompts need matching frozen rendered token counts")
    official = Path(official_ifeval_root).resolve()
    official_assets = verify_ifeval_assets(official)
    root.mkdir(parents=True, mode=0o700)
    for directory in ("practice", "diagnostics", "snapshots", "work"):
        (root / directory).mkdir(mode=0o700)
    EditWorkspace(source, root / "work")
    _write(root / "practice/items.json", practice)
    _write(root / "final-items.json", final)
    _write(root / "overlap-audit.json", overlap)
    _write(root / "config.json", config)
    _write(root / "baseline-template.json", template)
    bench = prepare_benchmark(benchmark, root / "benchmark")
    study = {"schema_version": 1, "kind": "research_browser_study", "study_id": config["study_id"],
        "created_epoch_seconds": time.time(), "config": config, "source_model": str(source),
        "source_sha256": source_sha, "source_files": inventory["files"],
        "practice_sha256": canonical_hash(practice), "final_sha256": canonical_hash(final),
        "overlap_sha256": canonical_hash(overlap), "scheduled_prompts": len(final),
        "baseline_template_sha256": canonical_hash(template), "baseline_artifact": base,
        "baseline_manifest": str(Path(baseline_manifest).resolve()),
        "baseline_manifest_sha256": file_sha256(baseline_manifest),
        "official_ifeval_root": str(official), "official_assets": official_assets,
        "probe_python": os.path.abspath(probe_python or sys.executable),
        "benchmark": str(bench), "environment": environment_identity(),
        "source_files_sha256": {str(p): file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "benchmark_files_sha256": {str(p): file_sha256(p) for p in sorted(bench.rglob("*"))
            if p.is_file() and not p.is_symlink() and not any(part in p.relative_to(bench).parts
                for part in ("node_modules", "models", "__pycache__", ".git"))},
        "claim_scope": "Final workload withheld from this editor session; prior external exposure is not established",
        "activation_scope": "BF16 CPU source diagnostics, distinct from final quantized WebGPU execution"}
    _write(root / "study.json", study)
    _write(root / "ready.json", {"study_sha256": canonical_hash(study)})
    _write(root / "state.json", {"study_sha256": canonical_hash(study), "active": None, "actions": [],
        "kernel": None, "snapshots": {}, "frozen": None, "development_closed": False,
        "deployment": None, "baseline": None, "candidate": None})
    return {"status": "prepared", "study_sha256": canonical_hash(study),
            "practice_items": len(practice), "final_items": len(final), "target_generations": 0,
            "development_limits": config["limits"], "final_browser_runs": 2}


def load(root):
    root = Path(root).resolve()
    study, state = read_json(root / "study.json"), read_json(root / "state.json")
    identity = canonical_hash(study)
    if (study.get("kind") != "research_browser_study" or state["study_sha256"] != identity
            or read_json(root / "ready.json") != {"study_sha256": identity}):
        raise ValueError("Frozen research study identity changed")
    bindings = {"config.json": canonical_hash(study["config"]), "practice/items.json": study["practice_sha256"],
        "final-items.json": study["final_sha256"], "overlap-audit.json": study["overlap_sha256"],
        "baseline-template.json": study["baseline_template_sha256"]}
    if any(canonical_hash(read_json(root / path)) != digest for path, digest in bindings.items()):
        raise ValueError("Frozen configuration or workload changed")
    if file_sha256(study["baseline_manifest"]) != study["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline deployment manifest changed")
    for field in ("source_files_sha256", "benchmark_files_sha256"):
        if any(file_sha256(path) != digest for path, digest in study.get(field, {}).items()):
            raise ValueError("Implementation changed after preparation; prepare a new study")
    return root, study, state


def _source(study):
    if inventory_model(study["source_model"])["files"] != study["source_files"]:
        raise ValueError("Immutable source checkpoint changed")


def _open_development(state):
    if state["development_closed"] or state["frozen"] is not None:
        raise ValueError("Final freeze has closed all development tools")


@contextmanager
def _action(root, kind, budget, arguments=None, *, development=False):
    root = Path(root).resolve()
    with _lock(root):
        root, study, state = load(root)
        if state["active"] is not None:
            raise RuntimeError("An interrupted action is unresolved; automatic replay is forbidden")
        if development:
            _open_development(state)
        maximum = study["config"]["limits"][budget] if isinstance(budget, str) else budget
        if sum(a["kind"] == kind for a in state["actions"]) >= maximum:
            raise ValueError("Action capacity exhausted")
        action = {"id": uuid.uuid4().hex, "kind": kind, "started_epoch_seconds": time.time(),
                  "arguments_sha256": canonical_hash(arguments) if arguments is not None else None}
        state["active"] = action
        _write(root / "state.json", state, replace=True)
        started = time.monotonic()
        try:
            yield root, study, state, action
        except BaseException as error:
            action.update(status="failed", error_type=type(error).__name__)
            raise
        else:
            action.setdefault("status", "completed")
        finally:
            action["elapsed_seconds"] = time.monotonic() - started
            state["actions"].append(action)
            state["active"] = None
            _write(root / "state.json", state, replace=True)


def editor_packet(root):
    root, study, state = load(root)
    _open_development(state)
    return {"schema_version": 1, "kind": "research_editor_packet", "study_sha256": canonical_hash(study),
        "requested_editor": study["config"]["editor"], "objective": study["config"]["objective"],
        "source_sha256": study["source_sha256"], "practice_items": read_json(root / "practice/items.json"),
        "limits": study["config"]["limits"], "kernel_limits": study["config"]["kernel_limits"],
        "probe_limits": study["config"]["probe_limits"], "final_settings": __import__(
            "weight_space_lab.ifeval_baseline", fromlist=["SETTINGS"]).SETTINGS,
        "spent": {name: sum(a["kind"] == name for a in state["actions"])
                  for name in ("python", "forward", "snapshot", "restore", "kernel-start")},
        "tools": {"python": "Persistent CPU Python: NAMES, load, load_original, save_edit, drop_edit, edited, tok",
                  "forward": "Original, current edited, or both: generation and teacher-forced diagnostic probes",
                  "snapshot": "Keep a named immutable candidate with evidence", "restore": "Return to a saved candidate",
                  "freeze": "Choose exactly one snapshot and close development before the final browser pair"},
        "editing": "Any direct, shape-preserving internal text-block weight edit; no fixed layer, column, rank, direction, strength or support budget",
        "restrictions": ["No training, gradients, optimizer fitting, network or GPU in the Python kernel",
            "Run the target only through the metered forward tool; never simulate its forward computation in Python",
            "Do not edit embeddings, output head, final norm, model architecture, tokenizer or decoding settings",
            "Whole-tensor scalar-only changes are rejected; justify targeted internal edits with evidence",
            "Use disjoint practice problems; final prompts and results are unavailable until editing closes",
            "Keep a hypothesis and try relevant dose, layer, random-direction and transfer controls; no prescribed repetition method",
            "CPU diagnostic resource costs are development costs, not WebLLM performance measurements"],
        "kernel_isolation": state.get("kernel"), "snapshot_names": sorted(state["snapshots"])}


def start_kernel(root, python_executable=None):
    from .research_kernel import start_kernel as start, kernel_status, KernelStopped
    with _action(root, "kernel-start", "kernel_starts", development=True) as (root, study, state, action):
        _source(study)
        if state["kernel"] is not None:
            try:
                status = kernel_status(state["kernel"]["control_dir"])
            except KernelStopped:
                status = read_json(Path(state["kernel"]["control_dir"]) / "state.json")
            if status.get("status") not in ("stopped", "failed") or not status.get("process_exit_confirmed"):
                raise ValueError("Stop the existing kernel and verify cleanup before an explicit restart")
        control = root / "kernels" / action["id"]
        control.parent.mkdir(exist_ok=True, mode=0o700)
        result = start(study["source_model"], root / "work", control,
            python_executable=python_executable,
            read_paths=(root / "practice", root / "diagnostics", root / "snapshots"),
            limits=study["config"]["kernel_limits"])
        state["kernel"] = {**result, "control_dir": str(control)}
        return result


def stop_kernel(root):
    from .research_kernel import stop_kernel as stop
    # Cleanup is allowed after freeze, failure, or source drift and never spends
    # an inference slot. It cannot reopen development or change a frozen model.
    root = Path(root).resolve()
    state = read_json(root / "state.json")
    if state["kernel"] is None:
        return {"status": "not_started", "process_exit_confirmed": True}
    return stop(state["kernel"]["control_dir"])


def execute_python(root, code):
    from .research_kernel import execute_kernel
    with _action(root, "python", "python_calls", code, development=True) as (root, study, state, action):
        _source(study)
        directory = root / "diagnostics" / action["id"]
        directory.mkdir(mode=0o700)
        _write(directory / "request.json", {"code": code})
        if state["kernel"] is None:
            raise ValueError("Start the persistent kernel explicitly before executing Python")
        result = execute_kernel(state["kernel"]["control_dir"], code)
        _write(directory / "result.json", result)
        action["result_sha256"] = canonical_hash(result)
        if result.get("ok") is False or result.get("error"):
            action["status"] = "failed"
        return {"action_id": action["id"], "result": result}


def snapshot(root, name, *, evidence=""):
    from .research_checkpoint import EditWorkspace
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("Use a short portable snapshot name")
    with _action(root, "snapshot", "snapshots", {"name": name, "evidence": evidence}, development=True) as (root, study, state, action):
        _source(study)
        if name in state["snapshots"]:
            raise ValueError("A named snapshot cannot be overwritten")
        destination = root / "snapshots" / name
        receipt = EditWorkspace(study["source_model"], root / "work").snapshot(destination)
        state["snapshots"][name] = {"path": str(destination), "sha256": receipt["sha256"],
                                    "evidence": evidence, "action_id": action["id"]}
        return {"name": name, **state["snapshots"][name]}


def _snapshot(study, state, name):
    from .research_checkpoint import verify_snapshot
    if name not in state["snapshots"]:
        raise ValueError("Select a completed named snapshot from this study")
    row = state["snapshots"][name]
    receipt = verify_snapshot(study["source_model"], row["path"])
    if receipt["sha256"] != row["sha256"]:
        raise ValueError("Named snapshot changed")
    return row


def restore(root, name):
    from .research_checkpoint import EditWorkspace
    with _action(root, "restore", "restores", {"name": name}, development=True) as (root, study, state, action):
        _source(study)
        row = _snapshot(study, state, name)
        result = EditWorkspace(study["source_model"], root / "work").restore(row["path"])
        return {"restored": name, "result": result}


def _reject_final_context(root, request):
    from .benchmark_data import normalized_text
    texts = list(request.get("texts", [])) + list(request.get("prompts", []))
    for chat in request.get("chats", []):
        texts.extend(m.get("content", "") for m in chat if isinstance(m, dict))
    normalized = [normalized_text(text) for text in texts if isinstance(text, str)]
    final = read_json(root / "final-items.json")
    final_ids = {item["item_id"] for item in final}
    if final_ids.intersection(request.get("problems", [])) or any(
            normalized_text(item["task_text"]) in text for item in final for text in normalized):
        raise ValueError("Final workload cannot be used as a diagnostic probe")


def forward(root, request):
    from .acquisition import _remove_stage
    from .research_checkpoint import EditWorkspace, materialize
    from .research_probes import run_probe
    with _action(root, "forward", "forward_calls", request, development=True) as (root, study, state, action):
        _source(study)
        _reject_final_context(root, request)
        directory = root / "diagnostics" / action["id"]
        directory.mkdir(mode=0o700)
        _write(directory / "request.json", request)
        candidate, candidate_sha = None, None
        if request.get("model") in ("edited", "both"):
            EditWorkspace(study["source_model"], root / "work").snapshot(directory / "snapshot")
            audit = materialize(study["source_model"], directory / "snapshot", directory / "candidate")
            candidate = directory / "candidate/model"
            candidate_sha = canonical_hash(inventory_model(candidate)["files"])
            action["candidate_source_sha256"] = candidate_sha
        try:
            result = run_probe(study["source_model"], candidate, request, directory / "probe",
                limits=study["config"]["probe_limits"], practice_items=read_json(root / "practice/items.json"),
                official_ifeval_root=study["official_ifeval_root"],
                expected_original_sha256=study["source_sha256"], expected_candidate_sha256=candidate_sha,
                python_executable=study.get("probe_python"))
            _write(directory / "result.json", result)
            action["result_sha256"] = canonical_hash(result)
            action["diagnostic_accounting"] = result.get("payload", {}).get("accounting")
            return {"action_id": action["id"], "result": result}
        finally:
            progress = directory / "probe/progress.json"
            if progress.is_file():
                action["diagnostic_accounting"] = read_json(progress)
            # Retain edited arrays, source audit, identities and probe evidence.
            # Rebuild this disposable model copy from the snapshot for replay;
            # fifty probes must not retain fifty full source checkpoints.
            if candidate is not None and candidate.exists():
                candidate.parent.chmod(0o755)
                try:
                    _remove_stage(candidate)
                finally:
                    candidate.parent.chmod(0o555)
                action["temporary_model_removed"] = True


def freeze(root, name, *, rationale, editor_provenance=None):
    from .research_checkpoint import materialize, verify_materialized
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Record the hypothesis and evidence before final evaluation")
    with _action(root, "freeze", "candidate_submissions", {"name": name, "rationale": rationale}, development=True) as (root, study, state, action):
        # Even a failed freeze closes development: no result-dependent reopening.
        state["development_closed"] = True
        _write(root / "state.json", state, replace=True)
        stop_kernel(root)
        _source(study)
        row = _snapshot(study, state, name)
        destination = root / "candidate-source"
        materialize(study["source_model"], row["path"], destination)
        audit = verify_materialized(study["source_model"], destination)
        submission = {"snapshot": name, "snapshot_sha256": row["sha256"], "rationale": rationale,
            "editor_provenance": editor_provenance or {"requested_selector": "gpt-6-astra", "verified_backend_version": None},
            "source_audit_sha256": canonical_hash(audit), "source_sha256": audit["result_sha256"],
            "path": str(destination)}
        _write(root / "submission.json", submission)
        state["frozen"] = {**submission, "submission_sha256": canonical_hash(submission)}
        return state["frozen"]


def _frozen(study, state):
    from .research_checkpoint import verify_materialized
    if state["frozen"] is None or not state["development_closed"]:
        raise ValueError("Freeze one candidate before final conversion or either browser arm")
    audit = verify_materialized(study["source_model"], state["frozen"]["path"])
    if canonical_hash(audit) != state["frozen"]["source_audit_sha256"]:
        raise ValueError("Frozen source candidate changed")
    return audit


def convert_candidate(root, converter_python):
    from .research_deployment import convert, package_candidate
    with _action(root, "convert", "conversions") as (root, study, state, action):
        _source(study)
        _frozen(study, state)
        converted = root / "conversion/weights"
        convert(study["source_model"], state["frozen"]["path"], study["baseline_manifest"], converted, converter_python)
        packaged = package_candidate(study["source_model"], state["frozen"]["path"], study["baseline_manifest"],
                                     converted, root / "deployment")
        artifact = read_json(packaged["artifact_manifest"])
        if artifact["source_checkpoint_sha256"] != state["frozen"]["source_sha256"]:
            raise ValueError("Browser artifact differs from frozen candidate")
        state["deployment"] = {"manifest": packaged["artifact_manifest"], "sha256": canonical_hash(artifact),
            "conversion_erases_edit": bool(packaged["conversion_erases_edit"] or
                artifact["artifact_sha256"] == study["baseline_artifact"]["artifact_sha256"])}
        return {**packaged, "conversion_erases_edit": state["deployment"]["conversion_erases_edit"]}


def run_arm(root, arm):
    from . import joint_browser
    if arm not in ("baseline", "candidate"):
        raise ValueError("Unknown final browser arm")
    root, study, state = load(root)
    _frozen(study, state)
    if state["deployment"] is None or state["deployment"]["conversion_erases_edit"]:
        raise ValueError("An effective audited browser candidate is required before final comparison")
    if state[arm] is not None:
        raise ValueError("Each final arm runs exactly once")
    if arm == "candidate" and (state["baseline"] is None or state["baseline"]["status"] != "completed"):
        raise ValueError("Complete the one final baseline before the candidate")
    with _action(root, "browser-" + arm, 1) as (root, study, state, action):
        _source(study)
        _frozen(study, state)
        artifact = study["baseline_artifact"] if arm == "baseline" else read_json(state["deployment"]["manifest"])
        if arm == "candidate" and canonical_hash(artifact) != state["deployment"]["sha256"]:
            raise ValueError("Frozen deployment changed")
        items = read_json(root / "final-items.json")
        config = _task_config(study["benchmark"], read_json(root / "baseline-template.json"), artifact,
                              items, arm, study["study_id"] + "-" + arm)
        session = root / "sessions" / arm
        seconds = study["config"]["limits"]["browser_seconds_per_arm"]
        joint_browser.prepare_session(session, benchmark=study["benchmark"], config=config, items=items,
            checkpoint={"model_id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                        "sha256": artifact["source_checkpoint_sha256"]},
            official_ifeval_root=study["official_ifeval_root"], study_sha256=canonical_hash(study),
            environment=study["environment"], browser_seconds=seconds)
        result = joint_browser.execute(session, on_load=lambda: seconds, preparation_seconds=300)
        state[arm] = {"report": str(session / "joint-report.json"), "sha256": canonical_hash(result),
                      "status": result.get("status", "unknown")}
        return result


def report(root):
    from .joint_browser import compare_reports, verify_report
    root, study, state = load(root)
    arms = {}
    for arm in ("baseline", "candidate"):
        if state[arm] is not None:
            value = verify_report(root / "sessions" / arm)
            if canonical_hash(value) != state[arm]["sha256"]:
                raise ValueError("Final joint report changed")
            arms[arm] = value
    comparison = compare_reports(arms["baseline"], arms["candidate"]) if len(arms) == 2 else None
    erased = bool(state["deployment"] and state["deployment"]["conversion_erases_edit"])
    return {"schema_version": 1, "kind": "research_browser_comparison", "study_sha256": canonical_hash(study),
        "status": "incomplete" if erased else "complete" if comparison and comparison["status"] == "completed" else "not_run" if not arms else "incomplete",
        "deployment_outcome": "not_converted" if state["deployment"] is None else "erased_by_quantization" if erased else "effective_edit",
        "final_comparison_skipped": erased,
        "accuracy_benchmark": "ifeval", "resource_benchmark": "WebLLM on the same final IFEval responses",
        "scope": study["claim_scope"], "scheduled_prompts": study["scheduled_prompts"],
        "runs_per_arm": 1, "separate_resource_generations": 0, "comparison": comparison, "arms": arms,
        "frozen_candidate": state["frozen"], "deployment": state["deployment"],
        "development_usage": {name: {"attempted": sum(a["kind"] == name for a in state["actions"]),
            "completed": sum(a["kind"] == name and a["status"] == "completed" for a in state["actions"]),
            "elapsed_seconds": sum(a["elapsed_seconds"] for a in state["actions"] if a["kind"] == name)}
            for name in ("python", "forward", "snapshot", "restore", "kernel-start")},
        "development_actions": [a for a in state["actions"] if a["kind"] in ("python", "forward")],
        "activation_scope": study["activation_scope"],
        "limitations": ["One final session per arm; descriptive paired result, not replicated causal resource evidence",
            "Diagnostic generation/teacher forcing is source CPU work, not WebLLM performance",
            "No claim of global non-exposure or training-data independence for the final set",
            "Missing telemetry remains null; all scored failures contribute to accuracy and cost"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--root", required=True)
    for name in ("source-model", "baseline-manifest", "baseline-config", "benchmark", "practice-items", "final-items", "official-ifeval-root"):
        setup.add_argument("--" + name, required=True)
    setup.add_argument("--config", default=str(DEFAULT_CONFIG))
    setup.add_argument("--probe-python", help="Pinned native-cpu interpreter for diagnostic workers")
    for name in ("packet", "kernel-start", "kernel-stop", "python", "forward", "snapshot", "restore", "freeze", "convert", "baseline", "candidate", "compare", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        if name in ("packet", "report"):
            command.add_argument("--output")
        if name == "kernel-start":
            command.add_argument("--python-executable")
        if name == "python":
            command.add_argument("--code-file", required=True)
        if name == "forward":
            command.add_argument("--request", required=True)
        if name in ("snapshot", "restore", "freeze"):
            command.add_argument("--name", required=True)
        if name == "snapshot":
            command.add_argument("--evidence", default="")
        if name == "freeze":
            command.add_argument("--rationale-file", required=True)
            command.add_argument("--editor-provenance")
        if name == "convert":
            command.add_argument("--converter-python", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.root, **{key: getattr(args, key) for key in
            ("source_model", "baseline_manifest", "baseline_config", "benchmark", "practice_items", "final_items", "official_ifeval_root")},
            config_path=args.config, probe_python=args.probe_python)
    elif args.command == "packet": result = editor_packet(args.root)
    elif args.command == "kernel-start": result = start_kernel(args.root, args.python_executable)
    elif args.command == "kernel-stop": result = stop_kernel(args.root)
    elif args.command == "python": result = execute_python(args.root, Path(args.code_file).read_text())
    elif args.command == "forward": result = forward(args.root, read_json(args.request))
    elif args.command == "snapshot": result = snapshot(args.root, args.name, evidence=args.evidence)
    elif args.command == "restore": result = restore(args.root, args.name)
    elif args.command == "freeze": result = freeze(args.root, args.name, rationale=Path(args.rationale_file).read_text(),
        editor_provenance=read_json(args.editor_provenance) if args.editor_provenance else None)
    elif args.command == "convert": result = convert_candidate(args.root, args.converter_python)
    elif args.command in ("baseline", "candidate"): result = run_arm(args.root, args.command)
    elif args.command == "compare":
        run_arm(args.root, "baseline")
        run_arm(args.root, "candidate")
        result = report(args.root)
    else: result = report(args.root)
    if getattr(args, "output", None):
        _write(args.output, result)
    import json
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
