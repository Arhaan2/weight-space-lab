"""Exact sparse BF16 replay, independently implemented from the release format.

Format reference and research credit:
https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math

The p50c archive stores flat indices as ``t{i}_idx`` and replacement BF16
bits as ``t{i}_val`` for each manifest tensor. No float conversion is used.
Only sparse edits are supported. Hashes identify the edited tensors, not the
whole checkpoint or its revision. This module performs no model evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _shape(value):
    if not isinstance(value, list):
        raise ValueError("shape must be an integer list")
    return tuple(_integer(item, "dimension") for item in value)


def _spec(spec):
    required = {"name", "shape", "dtype", "form", "changed_entries",
                "base_sha256", "edited_sha256"}
    if not isinstance(spec, dict) or not required <= spec.keys():
        raise ValueError("tensor specification has missing fields")
    if not isinstance(spec["name"], str) or not spec["name"]:
        raise ValueError("tensor name must be a nonempty string")
    if spec["form"] != "sparse" or spec["dtype"] != "bfloat16":
        raise ValueError("only sparse bfloat16 edits are supported")
    shape = _shape(spec["shape"])
    if _integer(spec["changed_entries"], "changed_entries") > math.prod(shape):
        raise ValueError("changed_entries exceeds tensor size")
    for key in ("base_sha256", "edited_sha256"):
        if not isinstance(spec[key], str) or not re.fullmatch(r"[0-9a-f]{64}", spec[key]):
            raise ValueError(f"{key} must be a lowercase SHA256 digest")
    return shape


def apply_sparse_bf16(base_bits, tensor_spec, indices, values):
    """Return (independent uint16 array, audit), checking exact bit hashes.

    ``base_bits`` and ``values`` must contain little-endian uint16 BF16 bit
    patterns. Indices use C-order flattening. Every input remains unchanged,
    including when validation fails. Changed entries count differing stored
    bits, including changes between positive and negative zero.
    """
    shape = _spec(tensor_spec)
    if (not isinstance(base_bits, np.ndarray) or base_bits.dtype != np.dtype("<u2")
            or base_bits.shape != shape):
        raise ValueError("base must be a uint16 BF16 array with the declared shape")
    if (not isinstance(indices, np.ndarray) or indices.ndim != 1
            or indices.dtype.kind not in "iu"):
        raise ValueError("indices must be a one-dimensional integer array")
    if (not isinstance(values, np.ndarray) or values.ndim != 1
            or values.dtype != np.dtype("<u2") or len(values) != len(indices)):
        raise ValueError("values must be a matching one-dimensional uint16 array")
    if indices.size and (indices.min() < 0 or indices.max() >= base_bits.size):
        raise ValueError("sparse index is outside the tensor")
    if np.unique(indices).size != indices.size:
        raise ValueError("sparse indices must be unique")
    before = np.ascontiguousarray(base_bits)
    base_hash = hashlib.sha256(before.tobytes()).hexdigest()
    if base_hash != tensor_spec["base_sha256"]:
        raise ValueError("base tensor SHA256 does not match")
    after = before.copy()
    after.reshape(-1)[indices] = values
    edited_hash = hashlib.sha256(after.tobytes()).hexdigest()
    if edited_hash != tensor_spec["edited_sha256"]:
        raise ValueError("edited tensor SHA256 does not match")
    changed = int(np.count_nonzero(before != after))
    if changed != tensor_spec["changed_entries"]:
        raise ValueError("changed_entries does not match the actual bit changes")
    return after, {"name": tensor_spec["name"], "changed_entries": changed,
                   "base_sha256": base_hash, "edited_sha256": edited_hash}


def _header(stream):
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(0)
    prefix = stream.read(8)
    length = int.from_bytes(prefix, "little")
    if len(prefix) != 8 or not 2 <= length <= min(64 * 1024 * 1024, size - 8):
        raise ValueError("invalid safetensors header length")
    header = json.loads(stream.read(length), object_pairs_hook=_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    widths = {"BOOL": 1, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4,
              "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("invalid safetensors tensor schema")
        dtype = tensor["dtype"]
        if not isinstance(dtype, str) or dtype not in widths:
            raise ValueError("unsupported safetensors dtype")
        shape = _shape(tensor["shape"])
        offsets = tensor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid safetensors offsets")
        start, end = (_integer(v, "offset") for v in offsets)
        if end - start != math.prod(shape) * widths[dtype]:
            raise ValueError("safetensors byte length disagrees with shape")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("safetensors data has a gap or overlap")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("safetensors data size does not match the header")
    return header, 8 + length


def replay_sparse_checkpoint(base_path, manifest_path, archive_path, output_path):
    """Replay a release-format sparse archive into a new safetensors file.

    Existing outputs are refused. The original header, metadata, and unedited
    tensor bytes are copied unchanged. All patch validation precedes output
    creation; a write exception removes its incomplete output. The audit
    reports exact changed BF16 scalars. Inputs must remain stable during replay.
    """
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be a new path")
    manifest = json.loads(Path(manifest_path).read_text(), object_pairs_hook=_object)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tensors"), list):
        raise ValueError("manifest must contain a tensors list")
    specs = manifest["tensors"]
    if not specs:
        raise ValueError("manifest must contain at least one tensor")
    for spec in specs:
        _spec(spec)
    names = [spec["name"] for spec in specs]
    if len(set(names)) != len(names) or "__metadata__" in names:
        raise ValueError("manifest tensor names must be unique and unreserved")
    if _integer(manifest.get("changed_tensors"), "changed_tensors") != len(specs):
        raise ValueError("changed_tensors does not match the tensor count")
    total = _integer(manifest.get("changed_entries_total"), "changed_entries_total")
    expected_keys = {f"t{i}_{suffix}" for i in range(len(specs)) for suffix in ("idx", "val")}
    archive = np.load(archive_path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("archive must be an NPZ file")
    edits, audits = [], []
    with archive, open(base_path, "rb") as source:
        if set(archive.files) != expected_keys or len(archive.files) != len(expected_keys):
            raise ValueError("NPZ fields do not match the sparse manifest")
        header, data_start = _header(source)
        for i, spec in enumerate(specs):
            tensor = header.get(spec["name"])
            if not tensor or tensor["dtype"] != "BF16" or tensor["shape"] != spec["shape"]:
                raise ValueError("checkpoint tensor dtype or shape does not match")
            start, end = tensor["data_offsets"]
            source.seek(data_start + start)
            bits = np.frombuffer(source.read(end - start), dtype="<u2").reshape(spec["shape"])
            edited, audit = apply_sparse_bf16(bits, spec, archive[f"t{i}_idx"], archive[f"t{i}_val"])
            edits.append((data_start + start, edited))
            audits.append(audit)
        if sum(audit["changed_entries"] for audit in audits) != total:
            raise ValueError("changed_entries_total does not match actual changes")
        output.parent.mkdir(parents=True, exist_ok=True)
        destination = output.open("xb")
        try:
            with destination:
                source.seek(0)
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                for offset, edited in edits:
                    destination.seek(offset)
                    destination.write(edited.tobytes())
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return {"changed_tensors": len(audits), "changed_entries_total": total, "tensors": audits}
