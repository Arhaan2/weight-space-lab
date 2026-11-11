"""Convert and audit a frozen general weight edit against the WebLLM baseline.

Mappings follow the pinned Qwen3.5 MLC loader, including fused Q/K/V and
gate/up tensors, convolution renaming, and RMSNorm offset conversion.  Every
deployed tensor is compared. Quantization may change only groups containing
changed source coordinates, including the correct rows of fused projections.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .browser_candidate import _files, _save, verify_baseline
from .browser_conversion import decode_q4, load_tensor, tensor_index
from .checkpoint import _mapped
from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash
from .research_checkpoint import verify_materialized


def source_identity(original, candidate, manifest):
    audit = verify_materialized(original, candidate)
    if audit["base_sha256"] != manifest["arms"]["baseline"]["source_checkpoint_sha256"]:
        raise ValueError("research candidate must start from this deployment baseline")
    return {"source_checkpoint_sha256": audit["result_sha256"],
            "source_original_sha256": audit["base_sha256"],
            "source_audit_sha256": canonical_hash(audit),
            "source_snapshot_sha256": audit["snapshot_sha256"]}, audit


def convert(original, candidate, baseline_manifest, output, converter_python):
    """One explicitly invoked local conversion; never executes a model."""
    started = time.monotonic()
    original, candidate, output = Path(original), Path(candidate), Path(output)
    manifest = verify_baseline(baseline_manifest)
    identity, _ = source_identity(original, candidate, manifest)
    script = Path(__file__).resolve().parents[1] / "scripts/browser_convert.py"
    if file_sha256(script) != manifest["converter_adapter_sha256"]:
        raise ValueError("converter adapter differs from the frozen baseline")
    receipt_path = output.with_name(output.name + "-source-binding.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("conversion destination and receipt must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(baseline_manifest).parent
    packages = manifest["converter_packages"]
    installed = subprocess.run([str(Path(converter_python).resolve()), "-c",
        "import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in " + repr(list(packages)) + "}))"],
        capture_output=True, text=True, check=True, timeout=30)
    if json.loads(installed.stdout) != packages:
        raise ValueError("converter packages differ from baseline")
    command = [str(Path(converter_python).resolve()), str(script), "--source", str((candidate / "model").resolve()),
               "--output", str(output.resolve()), "--wasm-metadata", str((root / "wasm-metadata.json").resolve())]
    record = {"schema_version": 1, "kind": "research_candidate_conversion", **identity,
              "baseline_manifest_sha256": file_sha256(baseline_manifest), "command": command,
              "converter_packages": packages, "target_generations": 0}
    _save(output.with_name(output.name + "-conversion-start.json"), record)
    env = dict(os.environ, TVM_BACKTRACE="1", MLC_LLM_HOME=str((root / "cache").resolve()),
               PYTHONPYCACHEPREFIX="/tmp/wsl-research-convert", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TVM_NUM_THREADS="4")
    with output.with_name(output.name + "-stdout.txt").open("x") as stdout, output.with_name(output.name + "-stderr.txt").open("x") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=1800)
    if result.returncode:
        _save(output.with_name(output.name + "-failure.json"), {"returncode": result.returncode,
              "preparation_seconds": time.monotonic() - started, "target_generations": 0})
        raise RuntimeError("conversion failed; diagnostic evidence preserved")
    if source_identity(original, candidate, manifest)[0] != identity:
        raise ValueError("candidate source changed during conversion")
    files = _files(output)
    tensor_index(output)
    receipt = {**record, "status": "converted_source_bound", "converted_files": files,
               "converted_files_sha256": canonical_hash(files), "preparation_seconds": time.monotonic() - started}
    _save(receipt_path, receipt)
    return {"converted": str(output.resolve()), "source_binding_receipt": str(receipt_path.resolve()),
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"], "target_generations": 0}


def mapped_sources(logical_name):
    """Source tensors and loader operation for one unquantized MLC parameter."""
    source = logical_name.replace("model.", "model.language_model.", 1)
    if source.endswith(".self_attn.c_attn.weight"):
        return [source.replace("c_attn.weight", name + ".weight") for name in ("q_proj", "k_proj", "v_proj")], "concatenate_rows"
    if source.endswith(".mlp.gate_up_proj.weight"):
        return [source.replace("gate_up_proj.weight", name + ".weight") for name in ("gate_proj", "up_proj")], "concatenate_rows"
    if source.endswith(".linear_attn.conv1d_weight"):
        return [source.removesuffix("conv1d_weight") + "conv1d.weight"], "direct"
    offset = source.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight"))
    return [source], "add_one_then_cast" if offset else "direct"


def _logical_name(name):
    if name.endswith(".q_weight"):
        return name.removesuffix(".q_weight") + ".weight"
    if name.endswith(".q_scale"):
        return name.removesuffix(".q_scale") + ".weight"
    return name


def _source_masks(original, candidate, audit):
    before, after = inventory_model(original), inventory_model(Path(candidate) / "model")
    masks = {}
    for name in audit["per_tensor"]:
        with _mapped(original, before["tensors"][name]) as a, _mapped(Path(candidate) / "model", after["tensors"][name]) as b:
            unsigned = np.dtype(f"<u{a.dtype.itemsize}")
            masks[name] = np.array(a.view(unsigned) != b.view(unsigned), copy=True)
    return before, masks


def _mapped_mask(logical, inventory, masks):
    sources, operation = mapped_sources(logical)
    if not any(name in masks for name in sources):
        return None, sources, operation
    if any(name not in inventory["tensors"] for name in sources):
        raise ValueError("MLC mapped source tensor missing")
    values = [masks.get(name, np.zeros(inventory["tensors"][name]["shape"], dtype=bool)) for name in sources]
    return (np.concatenate(values, axis=0) if operation == "concatenate_rows" else values[0]), sources, operation


def audit_effective_pair(original, candidate_bundle, baseline, candidate):
    """Independently verify source support, fused-row boundaries and q4 spillover."""
    source_audit = verify_materialized(original, candidate_bundle)
    inventory, masks = _source_masks(original, candidate_bundle, source_audit)
    base, edited = tensor_index(baseline), tensor_index(candidate)
    contract = lambda index: {name: (row["shape"], row["dtype"]) for name, row in index.items()}
    if contract(base) != contract(edited):
        raise ValueError("candidate deployed parameter contract differs")
    changed = [name for name in base if base[name]["stored_sha256"] != edited[name]["stored_sha256"]]
    allowed, covered, rows = set(), set(), {}
    for logical in sorted({_logical_name(name) for name in base}):
        support, sources, operation = _mapped_mask(logical, inventory, masks)
        if support is None:
            continue
        covered.update(set(sources) & masks.keys())
        prefix = logical.removesuffix(".weight")
        qname, sname = prefix + ".q_weight", prefix + ".q_scale"
        if qname in base or sname in base:
            if qname not in base or sname not in base or support.ndim != 2 or support.shape[1] % 32:
                raise ValueError("unsupported quantized source mapping contract")
            allowed.update((qname, sname))
            bw, cw = (load_tensor(root, index[qname]) for root, index in ((baseline, base), (candidate, edited)))
            bs, cs = (load_tensor(root, index[sname]) for root, index in ((baseline, base), (candidate, edited)))
            bd, cd = decode_q4(bw, bs), decode_q4(cw, cs)
            if support.shape != bd.shape:
                raise ValueError("quantized decoded shape differs from mapped source")
            group_support = support.reshape(support.shape[0], -1, 32).any(axis=2)
            if group_support.shape != bs.shape:
                raise ValueError("quantization scale shape differs")
            permitted = np.repeat(group_support, 32, axis=1)
            nibbles = lambda weights: ((weights[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(bd.shape)
            nibble_changes = nibbles(bw) != nibbles(cw)
            if np.any(nibble_changes & ~permitted) or np.any((bs != cs) & ~group_support):
                raise ValueError("quantized edit spills outside changed source groups or fused rows")
            numerical = bd != cd
            if np.any(numerical & ~permitted):
                raise ValueError("decoded edit escaped source-targeted groups")
            delta = cd.astype(np.float64) - bd.astype(np.float64)
            row = {"quantized": True, "group_size": 32,
                   "quantized_nibbles_changed": int(nibble_changes.sum()), "quantization_scales_changed": int(np.count_nonzero(bs != cs)),
                   "decoded_values_changed": int(numerical.sum()),
                   "decoded_changed_outside_source_coordinates": int((numerical & ~support).sum()),
                   "decoded_changed_outside_source_groups": int((numerical & ~permitted).sum()),
                   "decoded_baseline_sha256": hashlib.sha256(bd.tobytes()).hexdigest(),
                   "decoded_candidate_sha256": hashlib.sha256(cd.tobytes()).hexdigest()}
        elif logical in base:
            allowed.add(logical)
            bd, cd = (load_tensor(root, index[logical]) for root, index in ((baseline, base), (candidate, edited)))
            if bd.shape != support.shape:
                raise ValueError("unquantized shape differs from mapped source")
            raw_bits = lambda value: value.view(np.dtype(f"<u{value.dtype.itemsize}"))
            if np.any((raw_bits(bd) != raw_bits(cd)) & ~support):
                raise ValueError("unquantized edit escaped changed source coordinates")
            delta = cd.astype(np.float64) - bd.astype(np.float64)
            row = {"quantized": False, "decoded_values_changed": int(np.count_nonzero(bd != cd)),
                   "decoded_changed_outside_source_coordinates": 0, "decoded_changed_outside_source_groups": 0}
        else:
            raise ValueError("edited source has no recognized deployed parameter")
        if not np.all(np.isfinite(delta)):
            raise ValueError("nonfinite effective deployed edit")
        rows[logical] = {**row, "mapped_sources": sources, "loader_operation": operation,
                         "source_changed_coordinates": int(support.sum()),
                         "decoded_max_abs_change": float(np.max(np.abs(delta))) if delta.size else 0.0,
                         "decoded_l2_change": float(np.linalg.norm(delta))}
    if covered != set(masks):
        raise ValueError("edited source tensor is not part of the deployed text model")
    if not set(changed) <= allowed:
        raise ValueError("unrelated deployed tensor changed")
    return {"schema_version": 1, "kind": "research_effective_deployment_audit",
            "quantization": "q4f16_1", "parameter_contract_verified": True,
            "source_snapshot_sha256": source_audit["snapshot_sha256"],
            "changed_tensor_names": changed, "unchanged_tensor_count": len(base) - len(changed),
            "per_deployed_tensor": rows,
            "decoded_values_changed": sum(row["decoded_values_changed"] for row in rows.values()),
            "decoded_changed_outside_source_coordinates": sum(row["decoded_changed_outside_source_coordinates"] for row in rows.values()),
            "conversion_erases_edit": not any(row["decoded_values_changed"] for row in rows.values()),
            "tensor_storage_identities": {arm: {name: row["stored_sha256"] for name, row in index.items()}
                                         for arm, index in (("baseline", base), ("candidate", edited))}}


def package_candidate(original, candidate, baseline_manifest, converted, output):
    started = time.monotonic()
    original, candidate, converted, output = map(Path, (original, candidate, converted, output))
    if output.exists():
        raise FileExistsError("candidate package destination must be new")
    manifest = verify_baseline(baseline_manifest)
    identity, audit = source_identity(original, candidate, manifest)
    binding = read_json(converted.with_name(converted.name + "-source-binding.json"))
    if (binding.get("status") != "converted_source_bound" or binding["baseline_manifest_sha256"] != file_sha256(baseline_manifest)
            or any(binding.get(key) != value for key, value in identity.items())
            or binding["converted_files"] != _files(converted)
            or binding["converted_files_sha256"] != canonical_hash(_files(converted))):
        raise ValueError("converted bytes not bound to frozen candidate and baseline")
    baseline = Path(manifest["arms"]["baseline"]["path"])
    effective = audit_effective_pair(original, candidate, baseline, converted)
    output.mkdir(parents=True)
    stage = output / "candidate-staging"
    shutil.copytree(converted, stage)
    config = read_json(baseline / "mlc-chat-config.json")
    metadata = ["mlc-chat-config.json", "model.wasm", "LICENSE", *config["tokenizer_files"]]
    for name in metadata:
        if name not in manifest["arms"]["baseline"]["files"] or Path(name).name != name:
            raise ValueError("unsafe baseline runtime asset")
        shutil.copyfile(baseline / name, stage / name)
    files = _files(stage)
    artifact_sha = canonical_hash(files)
    destination = output / ("candidate-" + artifact_sha[:16])
    stage.rename(destination)
    artifact = {"schema_version": 1, "model_id": "wsl-qwen35-research-" + artifact_sha,
                "artifact_sha256": artifact_sha, "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
                "path": str(destination.resolve()), "files": files, "cache_namespace": "wsl-qwen35-research-" + artifact_sha,
                "tensor_cache_sha256": files["tensor-cache.json"]["sha256"],
                "baseline_artifact_sha256": manifest["arms"]["baseline"]["artifact_sha256"],
                "source_identity": identity, "conversion_binding_sha256": canonical_hash(binding),
                "effective_deployment_audit_sha256": canonical_hash(effective),
                "shared_runtime_assets": {name: files[name]["sha256"] for name in metadata},
                "preparation_seconds": time.monotonic() - started, "target_generations": 0,
                "provenance": "one frozen general research edit, independently source- and deployment-audited"}
    _save(destination / "artifact-manifest.json", artifact)
    _save(output / "candidate-artifact.json", artifact)
    _save(output / "source-audit.json", audit)
    _save(output / "deployment-difference.json", effective)
    return {"artifact_manifest": str((output / "candidate-artifact.json").resolve()),
            "artifact_path": artifact["path"], "artifact_sha256": artifact_sha,
            "conversion_erases_edit": effective["conversion_erases_edit"], "target_generations": 0}
