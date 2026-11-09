"""V2 provenance adapter around the frozen candidate/converter primitives.

The V1 launcher and recovery record remain unchanged. Both V2 arms use the
repaired interpreter helper and the same checkpoint, q4 and package audits.
Arhaan2.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

from .browser_candidate import _files, _save, audit_effective_pair, verify_baseline
from .browser_candidate_recovery import converter_python_command, run_converter_python
from .browser_conversion import tensor_index
from .checkpoint import verify_candidate
from .ifeval_v2_controls import MAPPING, validate_v2_proposal, verify_control
from .inventory import file_sha256, read_json
from .protocol import canonical_hash


def source_identity(original, candidate, manifest, *, policy_path="configs/candidate_policy.json",
                    expected_provenance="editor_proposal", control_audit=None):
    policy = read_json(policy_path)
    audit = verify_candidate(original, candidate, policy)
    base = manifest["arms"]["baseline"]["source_checkpoint_sha256"]
    if audit["base_sha256"] != base:
        raise ValueError("Candidate must start from the frozen pristine source")
    proposal = read_json(Path(candidate) / "proposal.json")
    validate_v2_proposal(proposal, base, provenance=expected_provenance)
    if expected_provenance == "engineering_edit":
        verify_control(proposal, control_audit)
        provenance = "matched_random_control"
    elif expected_provenance == "editor_proposal" and control_audit is None:
        provenance = "manual_astra_proposal"
    else:
        raise ValueError("V2 candidate provenance is inconsistent")
    return {
        "source_checkpoint_sha256": audit["result_sha256"], "source_original_sha256": audit["base_sha256"],
        "source_audit_sha256": canonical_hash(audit), "source_proposal_sha256": canonical_hash(proposal),
        "source_policy_sha256": canonical_hash(policy), "v2_provenance": provenance,
        "control_audit_sha256": canonical_hash(control_audit) if control_audit is not None else None,
    }, audit, proposal


def convert(original, candidate, baseline_manifest, output, converter_python, *,
            policy_path="configs/candidate_policy.json", expected_provenance="editor_proposal",
            control_audit=None, precheck_timeout=30, conversion_timeout=1800, on_conversion=None):
    """Convert once, with a durable precheck intent and no V1 recovery binding.

``on_conversion`` runs after the package precheck and before the conversion
subprocess. It may return the remaining conversion allowance in seconds.
The caller uses it to close/open separately budgeted durable stage clocks.
"""
    if not 0 < precheck_timeout <= 30 or not 0 < conversion_timeout <= 1800:
        raise ValueError("conversion timeouts exceed frozen per-attempt bounds")
    started = time.monotonic()
    original, candidate, output = map(Path, (original, candidate, output))
    intent = output.with_name(output.name + "-conversion-start.json")
    binding_path = output.with_name(output.name + "-source-binding.json")
    if output.exists() or intent.exists() or binding_path.exists():
        raise FileExistsError("V2 conversion attempt already exists; zero retries")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Even failed source/package checks are an attempted preparation, not a retry.
    _save(intent, {"schema_version": 1, "kind": "ifeval_browser_v2_conversion_intent",
                   "target_generations": 0, "conversion_subprocess_started": False})
    conversion_started = False
    try:
        manifest = verify_baseline(baseline_manifest)
        kwargs = {"policy_path": policy_path, "expected_provenance": expected_provenance,
                  "control_audit": control_audit}
        identity, _, _ = source_identity(original, candidate, manifest, **kwargs)
        script = Path(__file__).absolute().parent.parent / "scripts/browser_convert.py"
        if file_sha256(script) != manifest["converter_adapter_sha256"]:
            raise ValueError("Converter differs from the frozen baseline adapter")
        root = Path(baseline_manifest).parent
        packages = manifest["converter_packages"]
        mapping_files = {"qwen35_loader.py": "model/qwen35/qwen35_loader.py",
            "quantization.py": "quantization/quantization.py",
            "group_quantization.py": "quantization/group_quantization.py"}
        installed = run_converter_python(converter_python, ["-c",
            "import importlib.metadata,importlib.util,json,hashlib,pathlib; "
            "root=pathlib.Path(importlib.util.find_spec('mlc_llm').origin).parent; "
            "print(json.dumps({'packages':{k:importlib.metadata.version(k) for k in "
            + repr(list(packages)) + "},'mapping_source_sha256':{name:hashlib.sha256((root/relative).read_bytes()).hexdigest() "
            "for name,relative in " + repr(mapping_files) + ".items()}}))"], check=True, capture_output=True, text=True,
            timeout=precheck_timeout)
        environment = json.loads(installed.stdout)
        if environment.get("packages") != packages:
            raise ValueError("Converter environment differs from frozen packages")
        if environment.get("mapping_source_sha256") != MAPPING["mapping_source_sha256"]:
            raise ValueError("Converter quantization mapping differs from frozen matched-control mapping")
        if on_conversion is not None:
            remaining = on_conversion()
            if remaining is not None:
                if type(remaining) not in (int, float) or not 0 < remaining <= conversion_timeout:
                    raise ValueError("invalid remaining conversion allowance")
                conversion_timeout = remaining
        command = converter_python_command(converter_python, [str(script), "--source",
            str((candidate / "model").resolve()), "--output", str(output.resolve()),
            "--wasm-metadata", str((root / "wasm-metadata.json").resolve())])
        env = dict(os.environ, TVM_BACKTRACE="1", MLC_LLM_HOME=str((root / "cache").resolve()),
            PYTHONPYCACHEPREFIX="/tmp/wsl-browser-convert", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", TVM_NUM_THREADS="4")
        start_record = {"schema_version": 1, "kind": "ifeval_browser_v2_conversion_preparation",
            **identity, "baseline_manifest_sha256": file_sha256(baseline_manifest),
            "command": command, "converter_packages": packages, "target_generations": 0,
            "mapping_source_sha256": environment["mapping_source_sha256"],
            "new_proposals": 0, "v2_adapter_sha256": file_sha256(__file__),
            "converter_launcher_helper_sha256": file_sha256(Path(__file__).with_name("browser_candidate_recovery.py"))}
        _save(output.with_name(output.name + "-subprocess-intent.json"), start_record)
        conversion_started = True
        with output.with_name(output.name + "-convert.stdout.txt").open("x") as stdout, \
                output.with_name(output.name + "-convert.stderr.txt").open("x") as stderr:
            result = run_converter_python(command[0], command[1:], env=env, stdout=stdout,
                                          stderr=stderr, timeout=conversion_timeout)
        if result.returncode:
            raise RuntimeError("conversion subprocess failed")
        after, _, _ = source_identity(original, candidate, manifest, **kwargs)
        if after != identity:
            raise ValueError("Source candidate changed during conversion")
        files = _files(output)
        tensor_index(output)
        receipt = {**start_record, "converted_files": files,
            "converted_files_sha256": canonical_hash(files), "preparation_seconds": time.monotonic() - started,
            "status": "converted_source_bound"}
        _save(binding_path, receipt)
        return {"converted": str(output.resolve()), "source_binding_receipt": str(binding_path.resolve()),
                "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
                "target_generations": 0, "new_proposals": 0}
    except BaseException as error:
        _save(output.with_name(output.name + "-conversion-failure.json"), {
            "schema_version": 1, "error": type(error).__name__, "preparation_seconds": time.monotonic() - started,
            "conversion_subprocess_started": conversion_started, "target_generations": 0})
        raise


def package_candidate(original, candidate, baseline_manifest, converted, output, *,
                      policy_path="configs/candidate_policy.json", expected_provenance="editor_proposal",
                      control_audit=None):
    """Reuse unchanged deployment checks, retaining truthful V2 arm provenance."""
    started = time.monotonic()
    original, candidate, converted, output = map(Path, (original, candidate, converted, output))
    if output.exists():
        raise FileExistsError("Candidate package output must be new")
    manifest = verify_baseline(baseline_manifest)
    identity, audit, proposal = source_identity(original, candidate, manifest, policy_path=policy_path,
        expected_provenance=expected_provenance, control_audit=control_audit)
    binding = read_json(converted.with_name(converted.name + "-source-binding.json"))
    if (binding.get("status") != "converted_source_bound"
            or binding["baseline_manifest_sha256"] != file_sha256(baseline_manifest)
            or any(binding.get(key) != value for key, value in identity.items())
            or binding.get("v2_adapter_sha256") != file_sha256(__file__)
            or binding["converted_files"] != _files(converted)
            or binding["converted_files_sha256"] != canonical_hash(_files(converted))):
        raise ValueError("Converted artifact is not bound to its V2 source and baseline")
    policy = read_json(policy_path)
    baseline = Path(manifest["arms"]["baseline"]["path"])
    effective = audit_effective_pair(baseline, converted, proposal, policy["allowed_tensors"])
    output.mkdir(parents=True)
    stage = output / "candidate-staging"
    shutil.copytree(converted, stage)
    config = read_json(baseline / "mlc-chat-config.json")
    metadata = ["mlc-chat-config.json", "model.wasm", "LICENSE", *config["tokenizer_files"]]
    for name in metadata:
        if name not in manifest["arms"]["baseline"]["files"] or Path(name).name != name:
            raise ValueError("Unsafe or unbound baseline metadata")
        shutil.copyfile(baseline / name, stage / name)
    files = _files(stage)
    artifact_sha = canonical_hash(files)
    destination = output / f"candidate-{artifact_sha[:16]}"
    stage.rename(destination)
    artifact = {
        "schema_version": 1, "model_id": f"wsl-qwen35-v2-candidate-{artifact_sha}",
        "artifact_sha256": artifact_sha, "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "path": str(destination.resolve()), "files": files,
        "cache_namespace": f"wsl-qwen35-v2-candidate-{artifact_sha}",
        "tensor_cache_sha256": files["tensor-cache.json"]["sha256"],
        "baseline_artifact_sha256": manifest["arms"]["baseline"]["artifact_sha256"],
        "source_identity": identity, "conversion_binding_sha256": canonical_hash(binding),
        "effective_deployment_audit_sha256": canonical_hash(effective),
        "shared_runtime_assets": {name: files[name]["sha256"] for name in metadata},
        "preparation_seconds": time.monotonic() - started, "new_proposals": 0,
        "target_generations": 0, "provenance": identity["v2_provenance"],
    }
    _save(destination / "artifact-manifest.json", artifact)
    _save(output / "candidate-artifact.json", artifact)
    _save(output / "source-audit.json", audit)
    _save(output / "deployment-difference.json", effective)
    if control_audit is not None:
        _save(output / "matched-control-audit.json", control_audit)
    return {"artifact_manifest": str((output / "candidate-artifact.json").resolve()),
            "artifact_path": artifact["path"], "artifact_sha256": artifact_sha,
            "conversion_erases_edit": effective["conversion_erases_edit"],
            "target_generations": 0, "new_proposals": 0}
