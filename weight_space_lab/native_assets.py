"""Offline asset preparation and strict inputs for native smoke and replay runs."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path

from .acquisition import _freeze, _remove_stage, verify_acquisition
from .inventory import file_sha256, inventory_model, read_json
from .protocol import build_benchmark_schedule, canonical_hash, validate_protocol
from .reconstruction import verify_reconstruction


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def implementation_identity():
    """Bind executable project source, without machine paths or Git dirtiness."""
    root = Path(__file__).parent
    return canonical_hash({p.name: file_sha256(p) for p in sorted(root.glob("*.py"))})


def runtime_environment():
    """Freeze installed distribution identities without paths or environment dumps."""
    packages = {}
    for distribution in importlib.metadata.distributions():
        name, version = distribution.metadata.get("Name"), distribution.version
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None
                or not isinstance(version, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version) is None):
            raise ValueError("installed distribution identity cannot be recorded safely")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if packages.setdefault(name, version) != version:
            raise ValueError("conflicting installed distribution versions")
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": dict(sorted(packages.items()))}


def copy_identity(acquisition, output, reference):
    """Independent byte copy/reopen control, not a framework reserialization."""
    acquisition, output = Path(acquisition), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("identity output must be new")
    if acquisition.resolve() in output.resolve().parents:
        raise ValueError("identity output must be outside the acquisition")
    audit = verify_acquisition(acquisition, reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        shutil.copytree(acquisition / "model", stage / "model", copy_function=shutil.copyfile)
        observed = inventory_model(stage / "model")
        if observed != audit["model_inventory"]:
            raise ValueError("identity copy differs from original")
        for path in (stage / "model").rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        result = {"schema_version": 1, "kind": "identity_byte_copy", "complete": True,
                  "acquisition_sha256": file_sha256(acquisition / "acquisition.json"),
                  "model_files_sha256": canonical_hash(observed["files"]),
                  "model_inventory_sha256": canonical_hash(observed),
                  "changed_stored_entries": 0,
                  "inference_executed": False, "experiment_executed": False}
        _write(stage / "audit.json", result)
        _sync_directory(stage / "model")
        _sync_directory(stage)
        verify_acquisition(acquisition, reference)
        _freeze(stage)
        stage.rename(output)
        _sync_directory(output.parent)
        return result
    except BaseException:
        _remove_stage(stage)
        raise


def verify_identity(acquisition, identity, reference):
    audit = verify_acquisition(acquisition, reference)
    identity = Path(identity)
    if identity.is_symlink() or {p.name for p in identity.iterdir()} != {"model", "audit.json"}:
        raise ValueError("identity bundle is incomplete or has unexpected files")
    file_sha256(identity / "audit.json")
    observed = inventory_model(identity / "model")
    expected = {"schema_version": 1, "kind": "identity_byte_copy", "complete": True,
                "acquisition_sha256": file_sha256(Path(acquisition) / "acquisition.json"),
                "model_files_sha256": canonical_hash(observed["files"]),
                "model_inventory_sha256": canonical_hash(observed),
                "changed_stored_entries": 0,
                "inference_executed": False, "experiment_executed": False}
    if observed != audit["model_inventory"] or read_json(identity / "audit.json") != expected:
        raise ValueError("identity copy or audit differs from original")
    return observed


def backend_spec(name, template_hash):
    """Explicit short development settings; never the reference token limit."""
    if name not in {"transformers", "vllm"}:
        raise ValueError("unsupported native backend")
    return {
        "name": name,
        "versions": {"transformers": "5.3.0" if name == "transformers" else "4.57.6",
                     "torch": "2.10.0", "tokenizers": "0.22.2", "safetensors": "0.7.0", "jinja2": "3.1.6",
                     **({"vllm": "0.17.0"} if name == "vllm" else {})},
        "device": "cpu" if name == "transformers" else "cuda",
        "precision": "bfloat16",
        "attention_implementation": "eager" if name == "transformers" else "auto",
        "max_model_len": 2048, "cpu_threads": 4,
        "sampling": {"mode": "stochastic", "temperature": 0.6, "top_p": 0.95,
                     "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                     "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                     "max_new_tokens": 16},
        "chat": {"enable_thinking": True, "add_generation_prompt": True,
                 "chat_template_sha256": template_hash},
        "stopping": {"eos_token_ids": [248044, 248046], "stop_strings": [],
                     "ignore_eos": False, "include_stop_str_in_output": False},
        "decoding": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
    }


def prepare_smoke(acquisition, reconstruction, identity, output, reference, *, backend):
    """Freeze three matched development arms from independently checked bundles."""
    acquisition, reconstruction, identity, output = map(Path, (acquisition, reconstruction, identity, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("prepared smoke output must be new")
    acquired = verify_acquisition(acquisition, reference)
    rebuilt = verify_reconstruction(acquisition, reconstruction, reference)
    copied = verify_identity(acquisition, identity, reference)
    selected = read_json(reference)
    task_content = {"benchmark": "development-arithmetic", "item_id": "addition-7-5",
                    "split": "development", "messages": [{"role": "user", "content":
                    "Compute 7 + 5.\nPlease reason step by step, and put your final answer within \\boxed{}."}]}
    tasks = [{**task_content, "dataset_revision": canonical_hash([task_content])}]
    schedule = build_benchmark_schedule("development-arithmetic", ["addition-7-5"], [17])
    configs = {}
    for arm, inventory, audit_path in (
        ("original", acquired["model_inventory"], acquisition / "acquisition.json"),
        ("identity", copied, identity / "audit.json"),
        ("reconstructed", rebuilt["model_inventory"], reconstruction / "audit.json"),
    ):
        template = inventory["chat_templates"]["chat_template.jinja"]["sha256"]
        spec = backend_spec(backend, template)
        config = {
            "schema_version": 1, "kind": "native_execution", "run_id": f"native-smoke-{backend}-{arm}",
            "evidence_kind": "real_model", "profile": "native_development_smoke", "stage": "smoke",
            "checkpoint": {"model_id": selected["base_model"], "revision": selected["base_revision"],
                           "sha256": canonical_hash(inventory["files"])},
            "retry_policy": {"mode": "none", "max_attempts": 1},
            "execution_limits": {"max_sequences": 1, "startup_timeout_seconds": 300,
                                 "request_timeout_seconds": 180, "max_duration_seconds": 600},
            "backend": spec,
            "artifacts": {"files": inventory["files"], "inventory_sha256": canonical_hash(inventory),
                          "tokenizer_revision": selected["base_revision"], "audit_sha256": file_sha256(audit_path),
                          "arm": arm, "source_revision": selected["source_revision"]},
            "implementation_sha256": implementation_identity(),
            "runtime_environment": runtime_environment(),
            "tasks_sha256": canonical_hash(tasks), "schedule_sha256": canonical_hash(schedule),
            "methodological_variations": ["Full released conditional-generation architecture; upstream evaluated a text-only export.",
                "Development-only authored task, one seed, 16 output tokens and 2048 context; no scientific score."]
                + (["Transformers CPU eager BF16 runtime instead of the released vLLM evaluation runtime."]
                   if backend == "transformers" else []),
        }
        validate_execution_config(config, tasks, schedule)
        configs[arm] = config
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        _write(stage / "tasks.json", tasks)
        _write(stage / "schedule.json", schedule)
        for arm, config in configs.items():
            _write(stage / f"{arm}.json", config)
        _sync_directory(stage)
        stage.rename(output)
        _sync_directory(output.parent)
    except BaseException:
        _remove_stage(stage)
        raise
    return {"prepared_arms": list(configs), "scheduled_per_arm": len(schedule),
            "stage": "smoke", "inference_executed": False, "experiment_executed": False}


def validate_execution_config(config, tasks, schedule):
    """Require complete native provenance and reject unsupported study settings.

    A fixed-patch run additionally names its enclosing replay freeze. The study
    orchestrator checks that freeze and its complete matched-arm schedule; this
    validator checks the native arm's immutable inputs before every execution
    and resume. Neither check asserts editor isolation or scientific success.
    """
    from .native_backend import validate_backend_spec
    required = {"schema_version", "kind", "run_id", "evidence_kind", "profile", "stage", "checkpoint",
                "retry_policy", "execution_limits", "backend", "artifacts", "implementation_sha256",
                "tasks_sha256", "schedule_sha256", "methodological_variations", "runtime_environment"}
    if not isinstance(config, dict):
        raise ValueError("execution configuration has missing or unsupported fields")
    replay = config.get("profile") == "fixed_patch_evaluation"
    if set(config) != required | ({"study"} if replay else set()):
        raise ValueError("execution configuration has missing or unsupported fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["kind"] != "native_execution":
        raise ValueError("unsupported native execution configuration")
    if replay:
        if config["stage"] not in ("pilot", "full"):
            raise ValueError("fixed-patch execution requires a pilot or full replay stage")
        study = config["study"]
        digest_fields = {"freeze_id", "grader_revision", "policy_sha256", "patch_sha256",
                         "reconstruction_audit_sha256", "evaluation_code_revision", "split_audit_sha256"}
        if (not isinstance(study, dict)
                or set(study) != digest_fields | {"kind", "primary_comparison", "truncation_policy"}
                or study["kind"] != "published_patch_replay"
                or study["primary_comparison"] != "unchanged_vs_reconstructed_p50c"
                or study["truncation_policy"] not in ("grade", "fail")):
            raise ValueError("fixed-patch execution requires a complete published-patch study freeze")
        if any(not isinstance(study[key], str) or re.fullmatch(r"[0-9a-f]{64}", study[key]) is None
               for key in digest_fields):
            raise ValueError("replay study identities must be immutable SHA256 values")
        if study["evaluation_code_revision"] != config["implementation_sha256"]:
            raise ValueError("replay evaluation code differs from the native implementation freeze")
    elif config["profile"] != "native_development_smoke" or config["stage"] != "smoke":
        raise ValueError("native execution supports development smoke or a frozen fixed-patch replay")
    if config["evidence_kind"] != "real_model":
        raise ValueError("native artifact execution requires real_model evidence; use the synthetic runner fixture separately")
    validate_backend_spec(config["backend"])
    if not isinstance(config["run_id"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", config["run_id"]) is None:
        raise ValueError("run_id must be a portable identifier")
    if config["implementation_sha256"] != implementation_identity():
        raise ValueError("execution source changed; prepare a new configuration and output")
    if config["runtime_environment"] != runtime_environment():
        raise ValueError("runtime environment changed; prepare a new configuration and output")
    if config["tasks_sha256"] != canonical_hash(tasks) or config["schedule_sha256"] != canonical_hash(schedule):
        raise ValueError("tasks or ordered schedule differ from the frozen configuration")
    if not isinstance(config["artifacts"], dict) or set(config["artifacts"]) != {
            "files", "inventory_sha256", "tokenizer_revision", "audit_sha256", "arm", "source_revision"}:
        raise ValueError("artifact identities have missing or unknown fields")
    assets = config["artifacts"]
    if assets["arm"] not in {"original", "identity", "reconstructed"}:
        raise ValueError("unknown checkpoint arm")
    for name in ("inventory_sha256", "audit_sha256"):
        if not isinstance(assets[name], str) or re.fullmatch(r"[0-9a-f]{64}", assets[name]) is None:
            raise ValueError("artifact digest is invalid")
    for name in ("tokenizer_revision", "source_revision"):
        if not isinstance(assets[name], str) or re.fullmatch(r"[0-9a-f]{40}", assets[name]) is None:
            raise ValueError("artifact revision is invalid")
    checkpoint = config["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model_id", "revision", "sha256"}:
        raise ValueError("checkpoint identity has missing or unknown fields")
    if not isinstance(checkpoint["model_id"], str) or not checkpoint["model_id"].strip():
        raise ValueError("checkpoint model identifier is invalid")
    if not isinstance(checkpoint["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", checkpoint["revision"]) is None:
        raise ValueError("checkpoint revision is invalid")
    if not isinstance(checkpoint["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"]) is None:
        raise ValueError("checkpoint digest is invalid")
    if not isinstance(assets["files"], dict) or not assets["files"]:
        raise ValueError("complete artifact file inventory is required")
    if config["checkpoint"]["sha256"] != canonical_hash(assets["files"]):
        raise ValueError("checkpoint identity differs from complete asset inventory")
    variations = config["methodological_variations"]
    if not isinstance(variations, list) or not variations or any(not isinstance(x, str) or not x.strip() for x in variations):
        raise ValueError("methodological variations must be explicit")
    retry = config["retry_policy"]
    if (not isinstance(retry, dict) or set(retry) != {"mode", "max_attempts"}
            or retry["mode"] not in ("none", "retry_errors")
            or type(retry["max_attempts"]) is not int or retry["max_attempts"] < 1
            or (retry["mode"] == "none" and retry["max_attempts"] != 1)):
        raise ValueError("retry policy must explicitly declare a valid attempt budget")
    limits = config["execution_limits"]
    duration_keys = {"startup_timeout_seconds", "request_timeout_seconds", "max_duration_seconds"}
    if (not isinstance(limits, dict) or set(limits) != {"max_sequences", *duration_keys}
            or type(limits["max_sequences"]) is not int or limits["max_sequences"] < 1):
        raise ValueError("execution limits have missing or invalid fields")
    for key in duration_keys:
        try:
            valid = type(limits[key]) in (int, float) and math.isfinite(limits[key]) and limits[key] > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("execution duration limits must be finite and positive")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty array")
    task_keys = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"benchmark", "item_id", "split", "dataset_revision", "messages"}:
            raise ValueError("task has missing or unknown fields")
        if any(not isinstance(task[key], str) or not task[key].strip() for key in ("benchmark", "item_id")):
            raise ValueError("task identifiers must be nonempty strings")
        expected_split = "test" if replay and config["stage"] == "full" else "development"
        if task["split"] != expected_split:
            raise ValueError(f"{config['stage']} tasks must use the {expected_split} split")
        if replay and task["benchmark"] not in {"math", "gsm8k", "ifeval"}:
            raise ValueError("fixed-patch replay requires a supported pinned benchmark")
        if (not isinstance(task["dataset_revision"], str)
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", task["dataset_revision"]) is None):
            raise ValueError("task dataset revision is invalid")
        identity = task["benchmark"], task["item_id"]
        if identity in task_keys:
            raise ValueError("duplicate task identity")
        task_keys.add(identity)
        messages = task["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("task messages must be a nonempty array")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise ValueError("task messages must contain supported text roles and content")
    # Reuse Phase 1's profile and per-benchmark schedule contract.
    sampling = config["backend"]["sampling"]
    plans = []
    from .protocol import validate_benchmark_schedule
    validate_benchmark_schedule(schedule)
    if task_keys != {(row["benchmark"], row["item_id"]) for row in schedule}:
        raise ValueError("schedule and task identities must match exactly")
    for benchmark in sorted({row["benchmark"] for row in schedule}):
        rows = [row for row in schedule if row["benchmark"] == benchmark]
        items = [task for task in tasks if task["benchmark"] == benchmark]
        if not items or len({task["dataset_revision"] for task in items}) != 1:
            raise ValueError("benchmark requires one known dataset revision")
        plans.append({"benchmark": benchmark, "split": items[0]["split"], "dataset_revision": items[0]["dataset_revision"],
                      "task_manifest_sha256": canonical_hash(items), "item_ids": sorted({row["item_id"] for row in rows}),
                      "seeds": [row["seed"] for row in sorted(rows, key=lambda r: r["sample_index"])
                                if row["item_id"] == rows[0]["item_id"]],
                      "sampling": {**{key: sampling[key] for key in ("mode", "temperature", "top_p", "top_k", "presence_penalty")},
                                   "thinking": config["backend"]["chat"]["enable_thinking"],
                                   "max_output_tokens": sampling["max_new_tokens"]},
                      **({"grader_revision": study["grader_revision"],
                          "truncation_policy": study["truncation_policy"]} if replay else {})})
    profile = {"schema_version": 2, "profile": config["profile"], "stage": config["stage"],
               "claim_type": "published_patch_replay" if replay else "development_smoke",
               "model": config["checkpoint"]["model_id"],
               "backend": config["backend"]["name"], "precision": config["backend"]["precision"],
               "pins": {"model_revision": config["checkpoint"]["revision"], "tokenizer_revision": assets["tokenizer_revision"],
                        "runtime_version": canonical_hash(config["backend"]["versions"]),
                        "chat_template_sha256": config["backend"]["chat"]["chat_template_sha256"],
                        "checkpoint_sha256": config["checkpoint"]["sha256"]},
               "benchmarks": plans, "execution_limits": config["execution_limits"]}
    if replay:
        profile["pins"].update({"source_revision": assets["source_revision"],
                                **{key: study[key] for key in ("patch_sha256", "reconstruction_audit_sha256",
                                                              "evaluation_code_revision", "split_audit_sha256")}})
        if config["stage"] == "full":
            from .protocol import frozen_configuration_hash
            profile["frozen"] = True
            profile["freeze"] = {
                "configuration_sha256": frozen_configuration_hash(profile),
                "checkpoint_sha256": config["checkpoint"]["sha256"],
                "schedule_sha256": validate_benchmark_schedule(schedule)["schedule_sha256"],
                "evaluation_code_revision": study["evaluation_code_revision"],
            }
    checked = validate_protocol(profile, require_ready=True)
    if checked["issues"]:
        raise ValueError("native execution inputs do not satisfy the Phase 1 protocol: " + "; ".join(checked["issues"]))
    return copy.deepcopy(config)


def verify_execution_assets(config, model_path):
    """Rehash actual files before every execution/resume; paths are not identities."""
    observed = inventory_model(model_path)
    if (observed["files"] != config["artifacts"]["files"]
            or canonical_hash(observed) != config["artifacts"]["inventory_sha256"]):
        raise ValueError("model, tokenizer or chat assets changed since preparation")
    if observed["model_config"].get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("native runner requires the exact released architecture")
    if observed["chat_templates"].get("chat_template.jinja", {}).get("sha256") != config["backend"]["chat"]["chat_template_sha256"]:
        raise ValueError("chat template does not match execution configuration")
    return {"verified": True, "checkpoint_sha256": canonical_hash(observed["files"])}
