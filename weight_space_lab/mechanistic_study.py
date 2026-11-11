"""Adaptive inspection, one directional candidate, and one same-workload browser pair.

Preparation and reporting never run a model. Only explicit ``baseline``, ``tool``
(activation capture), and ``candidate`` commands execute it. All observations
belong to a new study; historical ledgers and experiment policies are untouched.
The local CLI is an operation boundary, not an OS sandbox for the editor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from urllib.parse import urlparse

from .editor_contract import read_json_bounded
from .inventory import file_sha256, inventory_model, read_json
from .native_runner import _lock
from .protocol import canonical_hash

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/ifeval_mechanistic_browser.json"
DEFAULT_POLICY = REPOSITORY / "configs/mechanistic_candidate_policy.json"
TOOLS = {"inspect": "weight_inspections", "capture": "activation_requests", "derive": "direction_derivations"}


def _write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("accuracy_benchmark") != "ifeval":
        raise ValueError("This version supports exactly one accuracy benchmark: IFEval")
    if config.get("scope") != "adaptive_development":
        raise ValueError("Current editor-visible workload is adaptive development, not untouched confirmation")
    schedule = config["schedule"]
    if (schedule["arms"] != ["baseline", "candidate"] or schedule["runs_per_arm"] != 1
            or schedule["warmups_per_arm"] != 1 or schedule["separate_resource_workload"] is not False
            or type(schedule["maximum_prompts"]) is not int or not 1 <= schedule["maximum_prompts"] <= 128):
        raise ValueError("Use one baseline/candidate pair on one shared workload")
    limits = config["limits"]
    expected = {"weight_inspections", "activation_requests", "direction_derivations", "candidate_submissions",
                "conversions", "browser_runs", "browser_seconds_per_arm", "automatic_retries"}
    if set(limits) != expected or any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError("Explicit nonnegative integer limits required")
    if (limits["candidate_submissions"] != 1 or limits["conversions"] != 1 or limits["browser_runs"] != 2
            or limits["automatic_retries"] != 0 or not 140 <= limits["browser_seconds_per_arm"] <= 5400
            or config["editor"]["max_submissions"] != 1):
        raise ValueError("One candidate, one conversion, and two browser runs without automatic retries")
    for key in ("weight_inspections", "activation_requests", "direction_derivations"):
        if not 1 <= limits[key] <= 100:
            raise ValueError("Diagnostic calls must have a bounded positive allowance")
    return copy.deepcopy(config)


def validate_items(items, maximum):
    from .benchmark_data import validate_item
    if not isinstance(items, list) or not 1 <= len(items) <= maximum:
        raise ValueError("Nonempty bounded IFEval workload required")
    ids = set()
    for item in items:
        validate_item(item)
        if item["benchmark"] != "ifeval" or item["split"] != "development" or item["item_id"] in ids:
            raise ValueError("Unique development IFEval items required; no hidden test data in editor tools")
        ids.add(item["item_id"])
    return items


def prepare_benchmark(source, destination):
    """Copy code into the new study, install telemetry, build without inference."""
    from .joint_browser import install_adapter
    from .browser_resource import install_adapter as install_resource_adapter
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or not (source / "scripts/run-task-evaluation.mjs").is_file():
        raise ValueError("A new destination and existing task-mode Browser LLM Bench checkout are required")
    shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns(
        ".git", "node_modules", "models", "results", ".DS_Store", "__pycache__", "*.pyc", "*.pyo"))
    dependencies = (source / "node_modules").resolve()
    if not dependencies.is_dir():
        raise ValueError("Stage the pinned browser dependencies before preparing a study")
    (destination / "node_modules").symlink_to(dependencies, target_is_directory=True)
    (destination / "public/models").mkdir(parents=True, exist_ok=True)
    if not (destination / "src/benchmark/resource-comparison/report.ts").exists():
        install_resource_adapter(destination)
    install_adapter(destination)
    subprocess.run(["node", "scripts/build-worker.mjs"], cwd=destination,
                   check=True, capture_output=True, timeout=120)
    return destination


def _task_config(benchmark, template, artifact, items, arm, run_id):
    from .browser_candidate import _files
    from .browser_resource import served_build
    from .ifeval_baseline import SETTINGS
    if _files(artifact["path"]) != artifact["files"] or canonical_hash(artifact["files"]) != artifact["artifact_sha256"]:
        raise ValueError("Deployed artifact bytes differ from their manifest")
    config = copy.deepcopy(template)
    origin = urlparse(config["model"]["modelUrl"])
    if origin.scheme != "http" or origin.hostname != "127.0.0.1":
        raise ValueError("Evaluation must use the local browser harness origin")
    base_url = f"{origin.scheme}://{origin.netloc}"
    relative = "models/mechanistic-" + artifact["artifact_sha256"] + "/resolve/" + artifact["artifact_sha256"] + "/"
    destination = Path(benchmark) / "public" / relative
    if destination.exists():
        if _files(destination) != artifact["files"]:
            raise ValueError("Previously staged artifact changed")
    else:
        shutil.copytree(artifact["path"], destination)
    model_url = base_url + "/" + relative
    config.pop("configurationSha256", None)
    config.pop("resourceComparison", None)
    template_items = {item["item_id"]: item for item in template["items"]}
    scheduled = []
    for item in items:
        old = template_items.get(item["item_id"])
        if old is None or old["prompt"] != item["task_text"] or type(old.get("rendered_input_tokens")) is not int:
            raise ValueError("Freeze exact rendered token counts in the task template before preparing this workload")
        scheduled.append({"item_id": item["item_id"], "prompt": item["task_text"],
                          "rendered_input_tokens": old["rendered_input_tokens"]})
    config.update(arm=arm, runId=run_id, sourceCheckpointSha256=artifact["source_checkpoint_sha256"],
                  convertedArtifactSha256=artifact["artifact_sha256"], settings=copy.deepcopy(SETTINGS),
                  items=scheduled)
    config["model"].update(id=artifact.get("model_id", "wsl-mechanistic-" + artifact["artifact_sha256"]),
                           modelUrl=model_url, modelLib=model_url + "model.wasm")
    names = ["mlc-chat-config.json", "tensor-cache.json", "tokenizer.json", "model.wasm",
             *[s["dataPath"] for s in read_json(destination / "tensor-cache.json")["records"]]]
    config["assets"] = [{"url": model_url + name, "sha256": artifact["files"][name]["sha256"]}
                        for name in dict.fromkeys(names)]
    config["servedBuild"] = served_build(benchmark, base_url)
    config["configurationSha256"] = canonical_hash(config)
    return config


def prepare(root, *, source_model, baseline_manifest, baseline_config, benchmark, items,
            official_ifeval_root, config_path=DEFAULT_CONFIG, policy_path=DEFAULT_POLICY):
    from .browser_candidate import verify_baseline
    from .browser_resource import environment_identity
    from .mechanistic_tools import policy_for_inventory
    from .ifeval_baseline import SETTINGS
    from .grading import verify_ifeval_assets
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Choose a new study directory; historical studies are never reopened")
    config = validate_config(read_json(config_path))
    item_values = validate_items(read_json(items), config["schedule"]["maximum_prompts"])
    source_model = Path(source_model).resolve()
    inventory = inventory_model(source_model)
    source_sha = canonical_hash(inventory["files"])
    policy = policy_for_inventory(read_json(policy_path), inventory)
    deployed = verify_baseline(baseline_manifest)
    base = deployed["arms"]["baseline"]
    if base["source_checkpoint_sha256"] != source_sha:
        raise ValueError("Baseline deployment must derive from the exact original inspected checkpoint")
    official = Path(official_ifeval_root).resolve()
    official_assets = verify_ifeval_assets(official)
    template = read_json(baseline_config)
    template_items = {i["item_id"]: i for i in template["items"]}
    if (template["settings"] != SETTINGS or template["sourceCheckpointSha256"] != source_sha
            or template["convertedArtifactSha256"] != base["artifact_sha256"]):
        raise ValueError("Task template must use the verified baseline source, deployment and generation settings")
    for item in item_values:
        old = template_items.get(item["item_id"])
        if (old is None or old["prompt"] != item["task_text"] or type(old.get("rendered_input_tokens")) is not int
                or not 1 <= old["rendered_input_tokens"] <= 3072):
            raise ValueError("Freeze matching task texts and rendered input counts in the baseline template")
    root.mkdir(parents=True, mode=0o700)
    _write(root / "config.json", config)
    _write(root / "policy.json", policy)
    _write(root / "items.json", item_values)
    _write(root / "baseline-template.json", template)
    bench = prepare_benchmark(benchmark, root / "benchmark")
    environment = environment_identity()
    study = {"schema_version": 1, "kind": "mechanistic_browser_study", "study_id": config["study_id"],
             "created_epoch_seconds": time.time(), "config": config, "source_model": str(source_model),
             "source_sha256": source_sha, "source_files": inventory["files"], "policy_sha256": canonical_hash(policy),
             "items_sha256": canonical_hash(item_values), "scheduled_prompts": len(item_values),
             "baseline_template_sha256": canonical_hash(template),
             "baseline_manifest": str(Path(baseline_manifest).resolve()),
             "baseline_manifest_sha256": file_sha256(baseline_manifest), "baseline_artifact": base,
             "official_ifeval_root": str(official), "official_assets": official_assets,
             "benchmark": str(bench), "environment": environment,
             "benchmark_files_sha256": {str(p): file_sha256(p) for p in sorted(bench.rglob("*"))
                if p.is_file() and not p.is_symlink() and not any(part in p.relative_to(bench).parts
                    for part in ("node_modules", "models", "__pycache__", ".git"))},
             "claim_scope": "adaptive development; editor sees this workload; no untouched quality claim",
             "activation_scope": "CPU source-checkpoint diagnostics, not quantized WebGPU activations",
             "source_files_sha256": {str(p): file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    _write(root / "study.json", study)
    _write(root / "ready.json", {"study_sha256": canonical_hash(study)})
    _write(root / "state.json", {"study_sha256": canonical_hash(study), "active": None, "actions": [],
                               "baseline": None, "candidate": None, "proposal": None, "deployment": None})
    return {"status": "prepared", "study_sha256": canonical_hash(study), "accuracy_benchmark": "ifeval",
            "scheduled_browser_runs": 2, "scheduled_requests_per_arm": len(item_values), "target_generations": 0}


def load(root):
    root = Path(root).resolve()
    study, state = read_json(root / "study.json"), read_json(root / "state.json")
    identity = canonical_hash(study)
    if read_json(root / "ready.json") != {"study_sha256": identity} or state["study_sha256"] != identity:
        raise ValueError("Frozen study identity changed")
    if (canonical_hash(read_json(root / "config.json")) != canonical_hash(study["config"])
            or canonical_hash(read_json(root / "policy.json")) != study["policy_sha256"]
            or canonical_hash(read_json(root / "items.json")) != study["items_sha256"]
            or canonical_hash(read_json(root / "baseline-template.json")) != study["baseline_template_sha256"]
            or file_sha256(study["baseline_manifest"]) != study["baseline_manifest_sha256"]):
        raise ValueError("Frozen workload, policy, configuration or baseline changed")
    for name, digest in study["source_files_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError("Implementation changed after study freeze; prepare a new study")
    for name, digest in study.get("benchmark_files_sha256", {}).items():
        if file_sha256(name) != digest:
            raise ValueError("Browser harness changed after study freeze")
    return root, study, state


@contextmanager
def _action(root, kind, maximum, *, arguments=None):
    """Reserve before work; crashes remain active and failed capacity is spent."""
    root = Path(root).resolve()
    with _lock(root):
        root, study, state = load(root)
        if state["active"] is not None:
            raise RuntimeError("An interrupted action is unresolved; automatic replay is forbidden")
        if sum(a["kind"] == kind for a in state["actions"]) >= maximum:
            raise ValueError("Action capacity exhausted; repeated execution is forbidden")
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
            action["status"] = "completed"
        finally:
            action["elapsed_seconds"] = time.monotonic() - started
            state["actions"].append(action)
            state["active"] = None
            _write(root / "state.json", state, replace=True)


def editor_packet(root):
    from .joint_browser import verify_report
    root, study, state = load(root)
    if not state["baseline"] or state["baseline"]["status"] != "completed" or state["proposal"] is not None:
        raise ValueError("Complete the one baseline before diagnosis, and finish diagnosis before submission")
    baseline = verify_report(root / "sessions/baseline")
    if canonical_hash(baseline) != state["baseline"]["sha256"]:
        raise ValueError("Baseline evidence no longer matches the completed action")
    return {"schema_version": 1, "kind": "adaptive_mechanistic_editor_packet", "study_sha256": canonical_hash(study),
            "requested_editor": study["config"]["editor"], "source_sha256": study["source_sha256"],
            "objective": study["config"]["objective"], "policy": read_json(root / "policy.json"),
            "development_items": read_json(root / "items.json"),
            "baseline_joint_report": baseline,
            "baseline_saved_outputs": read_json(root / "sessions/baseline/baseline/graded.json")["records"],
            "tools": {"inspect": "Read requested weight values/metadata/statistics within the policy",
                      "capture": "Bounded teacher-forced residual and MLP activations on paired diagnostic contexts",
                      "derive": "Contrast reference/failure activations and rank directional MLP writers"},
            "tool_request_envelope": {"study_sha256": canonical_hash(study), "tool": "inspect|capture|derive", "arguments": {}},
            "tool_command": "python -m weight_space_lab.mechanistic_study tool --root STUDY --request REQUEST.json",
            "limits": study["config"]["limits"], "activation_limits": study["config"]["activation_limits"],
            "observed_actions": state["actions"], "submission": "One final editor_proposal from pristine; resource results reported even when quality worsens",
            "restrictions": ["No training, gradients, network inference, or evaluation-setting changes",
                             "Use diagnostic tool calls adaptively; keep prompts and outputs in the private study",
                             "Source CPU activations are not measurements of the quantized browser execution",
                             "The workload is exposed development evidence, not an untouched final test"]}


def tool(root, request):
    from . import mechanistic_tools as diagnostic
    root, study, state = load(root)
    if not state["baseline"] or state["baseline"]["status"] != "completed" or state["proposal"] is not None:
        raise ValueError("Diagnostic tools are available after baseline and before final submission")
    if (set(request) != {"study_sha256", "tool", "arguments"} or request["study_sha256"] != canonical_hash(study)
            or request["tool"] not in TOOLS or not isinstance(request["arguments"], dict)):
        raise ValueError("Tool request must bind this study and an allowed diagnostic action")
    name, arguments = request["tool"], request["arguments"]
    limit = study["config"]["limits"][TOOLS[name]]
    with _action(root, name, limit, arguments=request) as (root, study, state, action):
        if state["proposal"] is not None:
            raise ValueError("Final submission has closed diagnostic access")
        directory = root / "diagnostics" / action["id"]
        directory.mkdir(parents=True, mode=0o700)
        _write(directory / "request.json", request)
        policy = read_json(root / "policy.json")
        if name == "inspect":
            result = diagnostic.inspect_weights(study["source_model"], policy, arguments,
                                                exposure_log=root / "weight-exposures.jsonl")
            if result["payload"]["source_sha256"] != study["source_sha256"]:
                raise ValueError("Inspected checkpoint differs from the frozen source")
        elif name == "capture":
            if arguments.get("source_sha256") != study["source_sha256"]:
                raise ValueError("Activation request must bind the original source checkpoint")
            result = diagnostic.capture_activations(study["source_model"], arguments, directory / "capture",
                                                   limits=study["config"]["activation_limits"])
        else:
            result = _derive(diagnostic, root, study, state, arguments, directory, policy)
        _write(directory / "result.json", result)
        action["result_sha256"] = canonical_hash(result)
        return {"action_id": action["id"], "tool": name, "result": result}


def _derive(diagnostic, root, study, state, arguments, directory, policy):
    capture_id = arguments.get("capture_id")
    captures = [a for a in state["actions"] if a["id"] == capture_id and a["kind"] == "capture" and a["status"] == "completed"]
    if len(captures) != 1:
        raise ValueError("Derivation requires one completed capture from this study")
    result_path = root / "diagnostics" / capture_id / "result.json"
    receipt = read_json(result_path)
    if canonical_hash(receipt) != captures[0]["result_sha256"]:
        raise ValueError("Capture receipt changed")
    if read_json(root / "diagnostics" / capture_id / "capture/manifest.json") != receipt:
        raise ValueError("Captured arrays no longer bind the completed diagnostic receipt")
    if set(arguments) - {"capture_id", "top_k"}:
        raise ValueError("Derivation accepts only capture_id and optional top_k")
    return diagnostic.derive_directions(study["source_model"], root / "diagnostics" / capture_id / "capture",
                                       directory / "directions.json", policy, top_k=arguments.get("top_k", 128))


def submit(root, proposal, *, rationale="", editor_provenance=None):
    from .checkpoint import CheckpointEditor, verify_candidate
    root, study, state = load(root)
    if not state["baseline"] or state["baseline"]["status"] != "completed" or state["proposal"] is not None:
        raise ValueError("One final submission is available after the baseline")
    with _action(root, "submit", 1, arguments=proposal) as (root, study, state, action):
        if proposal.get("provenance") != "editor_proposal" or proposal.get("base_sha256") != study["source_sha256"]:
            raise ValueError("Submit an editor proposal bound to the pristine source")
        policy = read_json(root / "policy.json")
        _write(root / "submission.json", {"proposal": proposal, "rationale": rationale,
                                          "editor_provenance": editor_provenance or {"requested_selector": "gpt-6-astra", "verified_version": None}})
        destination = root / "candidate-source"
        editor = CheckpointEditor(study["source_model"], policy, exposure_log=root / "weight-exposures.jsonl")
        editor.propose(proposal, destination)
        audit = verify_candidate(study["source_model"], destination, policy)
        if audit["base_sha256"] != study["source_sha256"]:
            raise ValueError("Candidate did not start from frozen pristine source")
        state["proposal"] = {"proposal_sha256": canonical_hash(proposal), "audit_sha256": canonical_hash(audit),
                             "source_sha256": audit["result_sha256"], "path": str(destination)}
        return state["proposal"]


def convert_candidate(root, converter_python):
    from .browser_candidate import convert, package_candidate
    from .checkpoint import verify_candidate
    root, study, state = load(root)
    if state["proposal"] is None or state["deployment"] is not None:
        raise ValueError("Submit one audited candidate before its single conversion")
    with _action(root, "convert", 1) as (root, study, state, action):
        current_audit = verify_candidate(study["source_model"], state["proposal"]["path"], read_json(root / "policy.json"))
        current_proposal = read_json(Path(state["proposal"]["path"]) / "proposal.json")
        if (canonical_hash(current_audit) != state["proposal"]["audit_sha256"]
                or canonical_hash(current_proposal) != state["proposal"]["proposal_sha256"]):
            raise ValueError("Candidate bytes or proposal differ from the accepted submission")
        converted = root / "conversion/weights"
        convert(study["source_model"], state["proposal"]["path"], study["baseline_manifest"], converted,
                converter_python, policy_path=root / "policy.json")
        packaged = package_candidate(study["source_model"], state["proposal"]["path"], study["baseline_manifest"],
                                     converted, root / "deployment", policy_path=root / "policy.json")
        artifact = read_json(packaged["artifact_manifest"])
        if artifact["source_checkpoint_sha256"] != state["proposal"]["source_sha256"]:
            raise ValueError("Converted artifact does not match the accepted candidate")
        erased = bool(packaged["conversion_erases_edit"] or
                      artifact["artifact_sha256"] == study["baseline_artifact"]["artifact_sha256"])
        state["deployment"] = {"manifest": packaged["artifact_manifest"], "sha256": canonical_hash(artifact),
                               "conversion_erases_edit": erased}
        packaged["conversion_erases_edit"] = erased
        return packaged


def run_arm(root, arm):
    from . import joint_browser
    root, study, state = load(root)
    if arm not in ("baseline", "candidate") or state[arm] is not None:
        raise ValueError("Each arm has exactly one browser run")
    if arm == "candidate" and (not state["baseline"] or state["baseline"]["status"] != "completed" or state["deployment"] is None):
        raise ValueError("Candidate evaluation requires the baseline and an audited deployed edit")
    if arm == "candidate" and state["deployment"].get("conversion_erases_edit") is True:
        raise ValueError("Quantization erased the candidate edit; no candidate browser run will be dispatched")
    with _action(root, "browser-" + arm, 1) as (root, study, state, action):
        artifact = study["baseline_artifact"] if arm == "baseline" else read_json(state["deployment"]["manifest"])
        if arm == "candidate" and canonical_hash(artifact) != state["deployment"]["sha256"]:
            raise ValueError("Frozen candidate deployment changed")
        items = read_json(root / "items.json")
        config = _task_config(study["benchmark"], read_json(root / "baseline-template.json"), artifact,
                              items, arm, study["study_id"] + "-" + arm)
        session = root / "sessions" / arm
        seconds = study["config"]["limits"]["browser_seconds_per_arm"]
        joint_browser.prepare_session(session, benchmark=study["benchmark"], config=config, items=items,
            checkpoint={"model_id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                        "sha256": artifact["source_checkpoint_sha256"]},
            official_ifeval_root=study["official_ifeval_root"], study_sha256=canonical_hash(study),
            environment=study["environment"], browser_seconds=seconds)
        report = joint_browser.execute(session, on_load=lambda: seconds, preparation_seconds=300)
        state[arm] = {"report": str(session / "joint-report.json"), "sha256": canonical_hash(report),
                      "status": report.get("status", "unknown")}
        return report


def report(root):
    from .joint_browser import compare_reports, verify_report
    root, study, state = load(root)
    arms = {}
    for arm in ("baseline", "candidate"):
        if state[arm]:
            value = verify_report(root / "sessions" / arm)
            if canonical_hash(value) != state[arm]["sha256"]:
                raise ValueError("Saved joint report changed")
            arms[arm] = value
    comparison = compare_reports(arms["baseline"], arms["candidate"]) if len(arms) == 2 else None
    deployment = state["deployment"]
    erased = deployment.get("conversion_erases_edit") is True if deployment else None
    status = ("incomplete" if erased else "complete" if comparison and comparison["status"] == "completed" else
              "incomplete" if len(arms) == 2 else "not_run" if not arms else "baseline_only")
    return {"schema_version": 1, "kind": "mechanistic_browser_comparison", "study_sha256": canonical_hash(study),
            "status": status,
            "deployment_outcome": "not_converted" if deployment is None else "erased_by_quantization" if erased else "effective_edit",
            "conversion_erases_edit": erased, "candidate_evaluation_skipped": erased is True,
            "accuracy_benchmark": "ifeval", "resource_benchmark": "Browser LLM Bench on the same IFEval requests",
            "scope": study["claim_scope"], "scheduled_prompts": study["scheduled_prompts"],
            "runs_per_arm": 1, "separate_resource_generations": 0, "arms": arms, "comparison": comparison,
            "diagnostic_usage": {name: sum(a["kind"] == name for a in state["actions"]) for name in TOOLS},
            "activation_scope": study["activation_scope"], "editor_isolation": "unverified",
            "limitations": ["One browser session per arm; descriptive comparison, not replicated causal attribution",
                            "Editor-visible workload is development evidence; no untouched accuracy estimate",
                            "Missing telemetry remains unavailable; token counts do not establish energy savings"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="Freeze a new single-pair study without inference")
    setup.add_argument("--root", required=True)
    for name in ("source-model", "baseline-manifest", "baseline-config", "benchmark", "items", "official-ifeval-root"):
        setup.add_argument("--" + name, required=True)
    setup.add_argument("--config", default=str(DEFAULT_CONFIG))
    setup.add_argument("--policy", default=str(DEFAULT_POLICY))
    for name in ("baseline", "candidate", "packet", "tool", "submit", "convert", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        if name == "tool": command.add_argument("--request", required=True)
        if name == "submit":
            command.add_argument("--proposal", required=True)
            command.add_argument("--rationale")
            command.add_argument("--editor-provenance")
        if name == "convert": command.add_argument("--converter-python", required=True)
        if name in ("packet", "report"): command.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.root, source_model=args.source_model, baseline_manifest=args.baseline_manifest,
                baseline_config=args.baseline_config, benchmark=args.benchmark, items=args.items,
                official_ifeval_root=args.official_ifeval_root, config_path=args.config, policy_path=args.policy)
        elif args.command in ("baseline", "candidate"):
            result = run_arm(args.root, args.command)
        elif args.command == "tool": result = tool(args.root, read_json_bounded(args.request))
        elif args.command == "submit":
            result = submit(args.root, read_json_bounded(args.proposal),
                rationale=Path(args.rationale).read_text() if args.rationale else "",
                editor_provenance=read_json_bounded(args.editor_provenance) if args.editor_provenance else None)
        elif args.command == "convert": result = convert_candidate(args.root, args.converter_python)
        elif args.command == "packet": result = editor_packet(args.root)
        else: result = report(args.root)
        if getattr(args, "output", None): _write(Path(args.output), result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "error": str(error), "type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
