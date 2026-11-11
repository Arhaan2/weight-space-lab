"""Bounded, offline weight/activation tools for mechanistic editing.

CPU teacher-forced diagnostics are separate from WebGPU quality/resource runs.
The public capture entry point supervises a disposable worker: it never loads a
target model in the controller. No arbitrary editor code, training or downloads.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

from .checkpoint import CheckpointEditor, _mapped, _proposal, validate_policy, verify_candidate
from .checkpoint_numeric import decode
from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash


DEFAULT_LIMITS = {
    "max_texts": 16, "max_layers": 24, "max_selected_positions": 32,
    "max_activation_elements": 16_777_216, "max_text_bytes": 65_536,
    "max_checkpoint_bytes": 4_294_967_296, "max_rss_bytes": 8_589_934_592,
    "wall_seconds": 180,
}
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_TENSOR = "model.language_model.layers.{}.mlp.down_proj.weight"


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _limits(value=None):
    result = {**DEFAULT_LIMITS, **(value or {})}
    if set(result) != set(DEFAULT_LIMITS):
        raise ValueError("unknown diagnostic limit")
    for name, maximum in DEFAULT_LIMITS.items():
        _integer(result[name], 1, maximum, name)
    return result


def _new_path(path, original):
    path, original = Path(path), Path(original).resolve()
    resolved = path.resolve()
    if resolved == original or original in resolved.parents or resolved in original.parents:
        raise ValueError("diagnostic output must be outside the original checkpoint")
    if path.exists() or path.is_symlink():
        raise FileExistsError("diagnostic output must be new")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sealed(payload):
    return {"payload": payload, "sha256": canonical_hash(payload)}


def _read_sealed(path):
    result = read_json(path)
    if (not isinstance(result, dict) or set(result) != {"payload", "sha256"}
            or result["sha256"] != canonical_hash(result["payload"])):
        raise ValueError("diagnostic content hash differs")
    return result


def inspect_weights(model, policy, request, exposure_log):
    """Expose live bounded tensor slices/statistics with the existing journal."""
    editor = CheckpointEditor(model, policy, exposure_log=exposure_log)
    response = editor.inspect(request)
    return _sealed({"kind": "mechanistic_weight_inspection", "schema_version": 1,
                    "source_sha256": editor.original_sha256,
                    "policy_sha256": canonical_hash(policy),
                    "request": request, "response": response})


def policy_for_inventory(template, inventory):
    """Freeze read access to every inventoried tensor, preserving edit limits.

    A read-only attention/gate/norm inspection must not broaden mutation rights.
    The orchestrator persists this resolved policy before any editor request.
    """
    policy = copy.deepcopy(template)
    policy["inspection"]["allowed_tensors"] = sorted(inventory["tensors"])
    return validate_policy(policy, inventory)


def validate_capture_request(request, limits=None):
    limits = _limits(limits)
    expected = {"schema_version", "source_sha256", "layers", "texts", "tail_tokens",
                "max_tokens", "cpu_threads"}
    if (not isinstance(request, dict) or set(request) != expected
            or type(request["schema_version"]) is not int or request["schema_version"] != 1):
        raise ValueError("unsupported activation request")
    if not isinstance(request["source_sha256"], str) or not _SHA.fullmatch(request["source_sha256"]):
        raise ValueError("source checkpoint hash required")
    layers = request["layers"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= limits["max_layers"]:
        raise ValueError("activation layer budget exceeded")
    for layer in layers:
        _integer(layer, 0, 23, "layer")
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate activation layer")
    _integer(request["tail_tokens"], 1, limits["max_selected_positions"], "tail_tokens")
    _integer(request["max_tokens"], 1, 512, "max_tokens")
    _integer(request["cpu_threads"], 1, 8, "cpu_threads")
    texts = request["texts"]
    if not isinstance(texts, list) or not 2 <= len(texts) <= limits["max_texts"]:
        raise ValueError("activation text budget exceeded")
    ids, pairs, byte_count = set(), {}, 0
    for item in texts:
        if not isinstance(item, dict) or set(item) != {"id", "pair_id", "condition", "text", "token_span"}:
            raise ValueError("invalid activation text fields")
        for key in ("id", "pair_id"):
            if not isinstance(item[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", item[key]):
                raise ValueError("text and pair IDs must be bounded portable identifiers")
        if item["id"] in ids:
            raise ValueError("duplicate activation text ID")
        ids.add(item["id"])
        if item["condition"] not in ("reference", "failure"):
            raise ValueError("condition must be reference or failure")
        conditions = pairs.setdefault(item["pair_id"], set())
        if item["condition"] in conditions:
            raise ValueError("duplicate pair condition")
        conditions.add(item["condition"])
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("teacher-forced text must be nonempty")
        byte_count += len(item["text"].encode("utf-8"))
        span = item["token_span"]
        if span is not None:
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("token_span must be null or [start,end]")
            _integer(span[0], 0, request["max_tokens"] - 1, "span start")
            _integer(span[1], span[0] + 1, request["max_tokens"], "span end")
            if span[1] - span[0] > limits["max_selected_positions"]:
                raise ValueError("selected activation position budget exceeded")
    if any(value != {"reference", "failure"} for value in pairs.values()):
        raise ValueError("each contrast pair requires one reference and one failure")
    if byte_count > limits["max_text_bytes"]:
        raise ValueError("activation input byte budget exceeded")
    return copy.deepcopy(request)


def _tokenize(tokenizer, request):
    records = []
    for index, item in enumerate(request["texts"]):
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if not ids or len(ids) > request["max_tokens"]:
            raise ValueError("tokenized diagnostic text exceeds limit; truncation is forbidden")
        span = item["token_span"] or [max(0, len(ids) - request["tail_tokens"]), len(ids)]
        if span[1] > len(ids):
            raise ValueError("activation token span exceeds actual input")
        records.append({"index": index, **{k: item[k] for k in ("id", "pair_id", "condition")},
                        "text_sha256": canonical_hash(item["text"]), "token_ids": ids,
                        "token_span": span, "token_count": len(ids)})
    return records


def _capture_with_hooks(model, records, layers, tensor_shapes, torch):
    """Hook decoder residual outputs and gated MLP inputs; always detach/remove."""
    backbone = model.model.language_model
    arrays = {}
    current = {}
    handles = []

    def save(kind, layer, value):
        if isinstance(value, tuple):
            value = value[0]
        start, end = current["token_span"]
        expected_width = tensor_shapes[layer][0 if kind == "residual" else 1]
        if tuple(value.shape) != (1, current["token_count"], expected_width):
            raise ValueError("observed activation shape differs from checkpoint")
        key = f"{kind}_L{layer}_T{current['index']}"
        if key in arrays:
            raise ValueError("diagnostic layer executed more than once for one text")
        values = value[0, start:end].detach().float().cpu().numpy().copy()
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite activation")
        arrays[key] = values

    try:
        for layer in layers:
            block = backbone.layers[layer]
            handles.append(block.register_forward_hook(
                lambda _module, _args, value, layer=layer: save("residual", layer, value)))
            handles.append(block.mlp.down_proj.register_forward_pre_hook(
                lambda _module, args, layer=layer: save("mlp", layer, args[0])))
        with torch.inference_mode():
            for record in records:
                current = record
                ids = torch.tensor([record["token_ids"]], dtype=torch.long, device="cpu")
                # Call text backbone directly: no LM-head vocabulary logits, no generation/cache.
                backbone(input_ids=ids, use_cache=False)
        expected = {f"{kind}_L{layer}_T{row['index']}"
                    for kind in ("residual", "mlp") for layer in layers for row in records}
        if set(arrays) != expected:
            raise ValueError("incomplete activation hooks")
        return arrays
    finally:
        for handle in handles:
            handle.remove()


def _capture_worker(model_path, request, directory, limits):
    # Pin optional dependencies only when actual capture was explicitly invoked.
    from .native_backend import (TRANSFORMERS_VERSIONS, _check_versions,
                                 _preserve_and_verify_loaded_tensors)
    _check_versions(TRANSFORMERS_VERSIONS)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    started = time.monotonic()
    before = inventory_model(model_path)
    if canonical_hash(before["files"]) != request["source_sha256"]:
        raise ValueError("activation source identity mismatch")
    if sum(item["size_bytes"] for item in before["files"].values()) > limits["max_checkpoint_bytes"]:
        raise ValueError("diagnostic checkpoint byte budget exceeded")
    shapes = {}
    for layer in request["layers"]:
        tensor = before["tensors"].get(_TENSOR.format(layer))
        if tensor is None or len(tensor["shape"]) != 2:
            raise ValueError("requested MLP layer absent from checkpoint")
        shapes[layer] = tensor["shape"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    records = _tokenize(tokenizer, request)
    positions = sum(row["token_span"][1] - row["token_span"][0] for row in records)
    expected_elements = positions * sum(sum(shape) for shape in shapes.values())
    if expected_elements > limits["max_activation_elements"]:
        raise ValueError("activation element budget exceeded")
    torch.set_num_threads(request["cpu_threads"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    verification = _preserve_and_verify_loaded_tensors(model, Path(model_path))
    arrays = _capture_with_hooks(model, records, request["layers"], shapes, torch)
    if inventory_model(model_path) != before:
        raise ValueError("source checkpoint changed during activation capture")
    directory = Path(directory)
    np.savez(directory / "activations.npz", **arrays)
    payload = {"schema_version": 1, "kind": "mechanistic_activation_capture",
               "source_sha256": request["source_sha256"], "request_sha256": canonical_hash(request),
               "backend": "transformers_cpu_teacher_forced", "versions": TRANSFORMERS_VERSIONS,
               "evidence_kind": "source_model_diagnostic", "browser_activations": False,
               "generated_tokens": 0, "forward_sequences": len(records), "training": False,
               "scope": "decoder residual output and gated MLP down-projection input; exact supplied text",
               "layers": request["layers"], "records": records,
               "tensor_shapes": {str(k): v for k, v in shapes.items()},
               "activation_elements": sum(value.size for value in arrays.values()),
               "arrays_sha256": file_sha256(directory / "activations.npz"),
               "model_loading": verification, "elapsed_seconds": time.monotonic() - started}
    _write_json(directory / "manifest.json", _sealed(payload))


def _rss_bytes(pid):
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                            text=True, timeout=2, check=False)
    value = result.stdout.strip()
    if result.returncode == 0 and value.isdigit():
        return int(value) * 1024
    return None


def capture_activations(model_path, request, output_dir, limits=None):
    """Run a timeout/RSS-supervised worker and publish one hash-bound capture.

    RSS is sampled process residency, not a hard allocator limit or browser RAM.
    The parent terminates the worker process group on an over-budget observation.
    Root orchestration is responsible for cumulative tool-call reservations.
    """
    limits = _limits(limits)
    request = validate_capture_request(request, limits)
    output = _new_path(output_dir, model_path)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-capture-", dir=output.parent))
    process = None
    try:
        _write_json(temporary / "request.json", request)
        _write_json(temporary / "limits.json", limits)
        command = [sys.executable, "-m", "weight_space_lab.mechanistic_tools", "_capture-worker",
                   "--model", str(Path(model_path).resolve()), "--request", str(temporary / "request.json"),
                   "--limits", str(temporary / "limits.json"), "--output", str(temporary)]
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false"}
        began = time.monotonic()
        peak = 0
        with (temporary / "worker.stdout").open("xb") as stdout, (temporary / "worker.stderr").open("xb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                       cwd=Path(__file__).resolve().parents[1], start_new_session=True)
            while process.poll() is None:
                if time.monotonic() - began > limits["wall_seconds"]:
                    raise TimeoutError("activation capture wall-time budget exhausted")
                observed = _rss_bytes(process.pid)
                if observed is None and process.poll() is None:
                    raise RuntimeError("activation worker memory measurement unavailable")
                peak = max(peak, observed or 0)
                if peak > limits["max_rss_bytes"]:
                    raise MemoryError("activation capture sampled RSS budget exhausted")
                if sum((temporary / name).stat().st_size for name in ("worker.stdout", "worker.stderr")) > 1_048_576:
                    raise RuntimeError("activation worker diagnostic output budget exhausted")
                time.sleep(0.1)
        if process.returncode != 0:
            # Raw provider/runtime traces may include paths; caller sees a stable failure.
            raise RuntimeError("activation capture worker failed; check pinned local runtime and request")
        manifest = _read_sealed(temporary / "manifest.json")
        if (manifest["payload"]["source_sha256"] != request["source_sha256"]
                or manifest["payload"]["request_sha256"] != canonical_hash(request)
                or manifest["payload"]["arrays_sha256"] != file_sha256(temporary / "activations.npz")):
            raise ValueError("activation worker output identity mismatch")
        for name in ("worker.stdout", "worker.stderr"):
            (temporary / name).unlink()
        _write_json(temporary / "supervision.json", _sealed({"limits": limits,
            "wall_seconds": time.monotonic() - began, "sampled_process_peak_rss_bytes": peak,
            "memory_scope": "CPU diagnostic worker sampled RSS; not WebGPU telemetry",
            "process_exit_confirmed": True}))
        if output.exists() or output.is_symlink():
            raise FileExistsError("capture destination appeared during execution")
        temporary.rename(output)
        return manifest
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        if temporary.exists():
            shutil.rmtree(temporary)


def contrast_direction(reference_residual, failure_residual, reference_mlp, failure_mlp, weights):
    """Return a unit residual contrast and activation-weighted writer scores.

    Inputs contain one token-mean vector per matched pair. Pair means receive
    equal weights; the direction points from reference toward observed failure.
    This is an association used to propose interventions, not proof of causation.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (reference_residual, failure_residual, reference_mlp, failure_mlp, weights)]
    r, f, rm, fm, w = arrays
    if (any(x.ndim != 2 or not np.all(np.isfinite(x)) for x in arrays)
            or r.shape != f.shape or rm.shape != fm.shape or r.shape[0] != rm.shape[0]
            or r.shape[0] == 0 or w.shape != (r.shape[1], rm.shape[1])):
        raise ValueError("contrast array dimensions/finiteness differ")
    direction = (f - r).mean(axis=0)
    scale = np.max(np.abs(direction))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("zero or invalid residual contrast")
    direction = direction / scale
    direction /= np.linalg.norm(direction)
    projections = direction @ w
    pair_scores = (fm - rm) * projections[None, :]
    scores = pair_scores.mean(axis=0)
    if not np.all(np.isfinite(scores)):
        raise ValueError("writer attribution overflow")
    return direction, scores, (pair_scores > 0).sum(axis=0)


def _check_activation_archive(path, meta, inventory):
    """Check NumPy headers/size before loading any supplied array allocation."""
    layers, records = meta["layers"], meta["records"]
    if (not isinstance(layers, list) or not 1 <= len(layers) <= DEFAULT_LIMITS["max_layers"]
            or any(type(layer) is not int or not 0 <= layer <= 23 for layer in layers)
            or len(set(layers)) != len(layers)
            or not isinstance(records, list) or not 2 <= len(records) <= DEFAULT_LIMITS["max_texts"]):
        raise ValueError("activation manifest limits exceeded")
    if [row["index"] for row in records] != list(range(len(records))):
        raise ValueError("activation manifest indices are not contiguous")
    expected = {}
    for layer in layers:
        tensor = inventory["tensors"].get(_TENSOR.format(layer))
        if tensor is None:
            raise ValueError("activation layer absent from source")
        for row in records:
            for kind, width in zip(("residual", "mlp"), tensor["shape"]):
                expected[f"{kind}_L{layer}_T{row['index']}.npy"] = width
    maximum_bytes = DEFAULT_LIMITS["max_activation_elements"] * 4 + len(expected) * 65536
    if path.stat().st_size > maximum_bytes:
        raise ValueError("activation archive byte budget exceeded")
    total = 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {entry.filename for entry in entries} != set(expected):
            raise ValueError("activation archive array set differs")
        if sum(entry.file_size for entry in entries) > maximum_bytes:
            raise ValueError("activation archive expanded byte budget exceeded")
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported activation array encoding")
                if (len(shape) != 2 or not 1 <= shape[0] <= DEFAULT_LIMITS["max_selected_positions"]
                        or shape[1] != expected[entry.filename] or dtype != np.dtype("float32") or fortran):
                    raise ValueError("activation array header shape/dtype differs")
                elements = math.prod(shape)
                if entry.file_size - stream.tell() != elements * dtype.itemsize:
                    raise ValueError("activation array byte length differs")
                total += elements
    if total > DEFAULT_LIMITS["max_activation_elements"]:
        raise ValueError("activation array element budget exceeded")


def derive_directions(model_path, captures_dir, output_path, policy, *, top_k=128):
    """Derive layer directions and rank any columns from actual saved activations."""
    _integer(top_k, 1, 256, "top_k")
    output = _new_path(output_path, model_path)
    before = inventory_model(model_path)
    policy = validate_policy(policy, before)
    capture = _read_sealed(Path(captures_dir) / "manifest.json")
    meta = capture["payload"]
    if meta["kind"] != "mechanistic_activation_capture" or meta["source_sha256"] != canonical_hash(before["files"]):
        raise ValueError("activation capture belongs to another source model")
    path = Path(captures_dir) / "activations.npz"
    if file_sha256(path) != meta["arrays_sha256"]:
        raise ValueError("activation bytes differ from manifest")
    _check_activation_archive(path, meta, before)
    pairs = {}
    for row in meta["records"]:
        pair = pairs.setdefault(row["pair_id"], {})
        if row["condition"] in pair:
            raise ValueError("duplicate activation condition")
        pair[row["condition"]] = row["index"]
    if not pairs or any(set(value) != {"reference", "failure"} for value in pairs.values()):
        raise ValueError("incomplete activation contrast pairs")
    results = []
    scanned = 0
    with np.load(path, allow_pickle=False) as arrays:
        for layer in meta["layers"]:
            name = _TENSOR.format(layer)
            if name not in policy["inspection"]["allowed_tensors"]:
                raise ValueError("direction derivation tensor is not inspection-allowlisted")
            tensor = before["tensors"][name]
            scanned += tensor["parameter_count"]
            if scanned > policy["inspection"]["max_total_scanned_coordinates"]:
                raise ValueError("direction derivation weight-scan budget exceeded")
            values = []
            for kind in ("residual", "mlp"):
                for condition in ("reference", "failure"):
                    observations = []
                    for pair in pairs.values():
                        array = arrays[f"{kind}_L{layer}_T{pair[condition]}"]
                        if array.ndim != 2 or array.shape[0] < 1 or not np.all(np.isfinite(array)):
                            raise ValueError("invalid saved activation array")
                        observations.append(np.mean(array, axis=0, dtype=np.float64))
                    values.append(np.stack(observations))
            with _mapped(model_path, tensor) as stored:
                weights = decode(np.array(stored, copy=True), tensor["dtype"])
            direction, scores, consistency = contrast_direction(*values, weights)
            columns = np.argsort(-scores, kind="stable")[:min(top_k, len(scores))]
            results.append({"layer": layer, "tensor": name, "direction": direction.tolist(),
                "direction_norm": float(np.linalg.norm(direction)), "pair_count": len(pairs),
                "ranked_columns": [{"column": int(column), "score": float(scores[column]),
                                    "positive_pairs": int(consistency[column])} for column in columns],
                "positive_writer_count": int((scores > 0).sum())})
    if inventory_model(model_path) != before:
        raise ValueError("source changed during direction derivation")
    result = _sealed({"schema_version": 1, "kind": "mechanistic_directions",
        "source_sha256": meta["source_sha256"], "capture_manifest_sha256": capture["sha256"],
        "policy_sha256": canonical_hash(policy), "layers": results,
        "weight_coordinates_scanned": scanned,
        "method": "mean paired failure-minus-reference residual contrast; MLP activation contrast times output projection",
        "interpretation": "diagnostic association; causal benefit requires held-out browser evaluation"})
    _write_json(output, result)
    return result


def build_direction_proposal(model_path, policy, diagnostics_path, selections, output_path):
    """Create a normal checkpoint proposal from explicit layer/column/dose choices."""
    output = _new_path(output_path, model_path)
    inventory = inventory_model(model_path)
    policy = validate_policy(policy, inventory)
    diagnostic = _read_sealed(diagnostics_path)
    payload = diagnostic["payload"]
    source_sha = canonical_hash(inventory["files"])
    if (payload.get("kind") != "mechanistic_directions" or payload["source_sha256"] != source_sha
            or payload["policy_sha256"] != canonical_hash(policy)):
        raise ValueError("direction artifact model or policy differs")
    if not isinstance(selections, list) or not selections:
        raise ValueError("at least one directional selection required")
    available = {row["layer"]: row for row in payload["layers"]}
    operations = []
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {"layer", "columns", "strength"}:
            raise ValueError("direction selection requires layer, columns, strength")
        layer = selection["layer"]
        if type(layer) is not int or layer not in available:
            raise ValueError("selected layer has no captured direction")
        operations.append({"op": "remove_direction_from_columns", "tensor": available[layer]["tensor"],
                           "columns": selection["columns"], "direction": available[layer]["direction"],
                           "strength": selection["strength"]})
    proposal = {"schema_version": 1, "base_sha256": source_sha, "operations": operations,
                "provenance": "editor_proposal", "composition": "sequential" if len(operations) > 1 else "independent"}
    _proposal(proposal, policy, inventory, source_sha)
    # Existing checkpoint schema stays unchanged. Sidecar binds the evidence.
    binding = output.with_name(output.name + ".evidence.json")
    if binding.exists() or binding.is_symlink():
        raise FileExistsError("proposal evidence destination already exists")
    _write_json(output, proposal)
    _write_json(binding, _sealed({"proposal_sha256": canonical_hash(proposal),
        "directions_sha256": diagnostic["sha256"], "capture_manifest_sha256": payload["capture_manifest_sha256"],
        "source_sha256": source_sha, "policy_sha256": canonical_hash(policy)}))
    return proposal


def apply_proposal(model, policy, proposal, output, exposure_log):
    """Build and independently verify one candidate, without inference."""
    editor = CheckpointEditor(model, policy, exposure_log=exposure_log)
    audit = editor.propose(proposal, output)
    if verify_candidate(model, output, policy) != audit:
        raise ValueError("independent candidate verification differs")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve-policy", "inspect", "capture", "derive", "propose", "apply", "_capture-worker"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        if name in ("resolve-policy", "inspect", "derive", "propose", "apply"):
            p.add_argument("--policy", required=True)
        if name in ("inspect", "capture", "_capture-worker"):
            p.add_argument("--request", required=True)
        if name in ("capture", "_capture-worker"):
            p.add_argument("--limits")
        if name in ("inspect", "apply"):
            p.add_argument("--exposure-log", required=True)
        if name != "inspect":
            p.add_argument("--output", required=True)
        if name == "derive":
            p.add_argument("--capture", required=True)
            p.add_argument("--top-k", type=int, default=128)
        if name == "propose":
            p.add_argument("--directions", required=True)
            p.add_argument("--selections", required=True)
        if name == "apply":
            p.add_argument("--proposal", required=True)
    args = parser.parse_args(argv)
    policy = read_json(args.policy) if hasattr(args, "policy") else None
    request = read_json(args.request) if hasattr(args, "request") else None
    limits = read_json(args.limits) if getattr(args, "limits", None) else None
    if args.command == "resolve-policy":
        output = _new_path(args.output, args.model)
        result = policy_for_inventory(policy, inventory_model(args.model))
        _write_json(output, result)
    elif args.command == "inspect":
        result = inspect_weights(args.model, policy, request, args.exposure_log)
    elif args.command == "capture":
        result = capture_activations(args.model, request, args.output, limits)
    elif args.command == "derive":
        result = derive_directions(args.model, args.capture, args.output, policy, top_k=args.top_k)
    elif args.command == "propose":
        result = build_direction_proposal(args.model, policy, args.directions, read_json(args.selections), args.output)
    elif args.command == "apply":
        result = apply_proposal(args.model, policy, read_json(args.proposal), args.output, args.exposure_log)
    else:
        _capture_worker(args.model, validate_capture_request(request, limits), args.output, _limits(limits))
        return
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
