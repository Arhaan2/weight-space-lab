"""Supplementary offline verification of acquired p50c model bytes.

This checker imports no project implementation. It independently reads model
headers and compares every byte outside the published target tensor spans.
Its reference must come from an independently maintained source. Like any
offline checker, it cannot authenticate wholly replaced provenance records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path

import numpy as np


CHUNK = 8 * 1024 * 1024
WIDTHS = {"BOOL": 1, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
          "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4,
          "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def json_bytes(data):
    def pairs(items):
        value = {}
        for key, item in items:
            check(key not in value, "duplicate JSON key")
            value[key] = item
        return value

    def constant(_):
        raise ValueError("nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def json_file(path):
    regular(path)
    return json_bytes(path.read_bytes())


def regular(path):
    check(stat.S_ISREG(path.lstat().st_mode), "artifact must be a regular file")


def file_map(root):
    check(stat.S_ISDIR(root.lstat().st_mode), "artifact root must be a directory")
    result = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        check(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "links and special files are refused")
        if stat.S_ISREG(mode):
            result[path.relative_to(root).as_posix()] = path
    return result


def sha(path, start=0, length=None):
    regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        while length is None or length:
            block = stream.read(CHUNK if length is None else min(length, CHUNK))
            if not block:
                check(length is None or length == 0, "truncated span")
                break
            digest.update(block)
            if length is not None:
                length -= len(block)
    return digest.hexdigest()


def header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        size = int.from_bytes(prefix, "little")
        check(len(prefix) == 8 and 2 <= size <= min(64 * 1024 * 1024, path.stat().st_size - 8),
              "invalid header length")
        value = json_bytes(stream.read(size))
    check(isinstance(value, dict), "header must be an object")
    spans = []
    for name, tensor in value.items():
        if name == "__metadata__":
            check(isinstance(tensor, dict) and all(isinstance(v, str) for v in tensor.values()),
                  "invalid header metadata")
            continue
        check(bool(name) and isinstance(tensor, dict)
              and set(tensor) == {"dtype", "shape", "data_offsets"}, "invalid tensor header")
        shape, dtype, offsets = tensor["shape"], tensor["dtype"], tensor["data_offsets"]
        check(isinstance(shape, list) and all(integer(v) for v in shape), "invalid tensor shape")
        check(isinstance(dtype, str) and dtype in WIDTHS, "unsupported stored dtype")
        check(isinstance(offsets, list) and len(offsets) == 2 and all(integer(v) for v in offsets),
              "invalid tensor offsets")
        start, end = offsets
        check(end - start == math.prod(shape) * WIDTHS[dtype], "tensor size differs from layout")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        check(start == cursor, "tensor spans overlap or have a gap")
        cursor = end
    check(cursor == path.stat().st_size - size - 8, "tensor payload size differs")
    return size + 8, value


def equal_span(left, right, start, length):
    with left.open("rb") as a, right.open("rb") as b:
        a.seek(start)
        b.seek(start)
        while length:
            size = min(length, CHUNK)
            a_bytes, b_bytes = a.read(size), b.read(size)
            check(len(a_bytes) == size and a_bytes == b_bytes, "unrelated bytes changed")
            length -= size


def run(acquisition, reconstruction, reference_path):
    acquisition, reconstruction = Path(acquisition), Path(reconstruction)
    reference = json_file(Path(reference_path))
    check(isinstance(reference, dict) and reference.get("schema_version") == 1
          and reference.get("claim_type") == "published_patch_replay", "invalid reference")
    for key in ("source_revision", "base_revision"):
        check(isinstance(reference.get(key), str) and re.fullmatch(r"[0-9a-f]{40}", reference[key]),
              "reference revisions must be immutable")
    for key in ("expected_changed_stored_entries", "reported_targeted_entries"):
        check(integer(reference.get(key)), "invalid reference counts")
    names = reference.get("tensor_names")
    check(isinstance(names, list) and bool(names) and all(isinstance(n, str) and n for n in names)
          and len(set(names)) == len(names), "invalid reference target names")
    source_files = file_map(acquisition)
    saved_files = file_map(reconstruction)
    check(json_file(acquisition / "reference.json") == reference, "acquired reference differs")
    manifest = json_file(acquisition / "patch/manifest.json")
    source_audit = json_file(acquisition / "acquisition.json")
    output_audit = json_file(reconstruction / "audit.json")
    check(isinstance(source_audit, dict) and source_audit.get("complete") is True
          and source_audit.get("kind") == "pinned_acquisition", "incomplete acquisition")
    check(isinstance(output_audit, dict) and output_audit.get("complete") is True
          and output_audit.get("kind") == "published_patch_reconstruction", "incomplete reconstruction")
    for audit in (source_audit, output_audit):
        check(audit.get("resolved_source_revision") == reference["source_revision"]
              and audit.get("resolved_model_revision") == reference["base_revision"], "resolved revision differs")
        check(audit.get("reference_sha256") == sha(acquisition / "reference.json"), "reference hash differs")
    check(output_audit.get("acquisition_sha256") == sha(acquisition / "acquisition.json"),
          "acquisition audit binding differs")
    for field, path in (("manifest_sha256", "patch/manifest.json"), ("archive_sha256", "patch/edit.npz")):
        check(output_audit.get(field) == sha(acquisition / path), "patch audit binding differs")
    check(isinstance(source_audit.get("files"), dict)
          and set(source_files) == set(source_audit["files"]) | {"acquisition.json"}, "acquisition file set differs")
    for name, info in source_audit["files"].items():
        check(isinstance(info, dict) and info == {"sha256": sha(source_files[name]),
                                                 "size_bytes": source_files[name].stat().st_size},
              "source audit mismatch")
    original, saved = acquisition / "model", reconstruction / "model"
    old_files, new_files = file_map(original), file_map(saved)
    check(old_files.keys() == new_files.keys(), "model file set differs")
    check(set(saved_files) == {"model/" + name for name in old_files} | {"audit.json"},
          "unexpected reconstruction files")
    check({p.name for p in reconstruction.iterdir()} == {"model", "audit.json"}, "unexpected reconstruction entries")
    tensor_map, headers, starts = {}, {}, {}
    for name, path in old_files.items():
        check(path.stat().st_size == new_files[name].stat().st_size, "saved size differs")
        if name.endswith(".safetensors"):
            start, content = header(path)
            check(header(new_files[name]) == (start, content), "header schema differs")
            equal_span(path, new_files[name], 0, start)
            headers[name], starts[name] = content, start
            for tensor, info in content.items():
                if tensor != "__metadata__":
                    check(tensor not in tensor_map, "duplicate tensor across shards")
                    tensor_map[tensor] = (name, info)
    check(bool(tensor_map), "model has no tensors")
    index_path = original / "model.safetensors.index.json"
    if index_path.exists():
        index = json_file(index_path)
        check(isinstance(index, dict) and index.get("weight_map") == {
            name: shard for name, (shard, info) in tensor_map.items()}, "index differs from actual shards")
        check(set(index["weight_map"].values()) == set(headers), "unindexed shard")
    else:
        check(list(headers) == ["model.safetensors"], "noncanonical model requires an index")
    check(isinstance(manifest, dict) and isinstance(manifest.get("tensors"), list), "invalid manifest")
    specs = manifest["tensors"]
    check(all(isinstance(spec, dict) and isinstance(spec.get("name"), str) for spec in specs), "invalid target specification")
    check(len(specs) == len(names) and set(names) == {spec["name"] for spec in specs}, "target names differ")
    check(manifest.get("changed_tensors") == len(specs), "manifest target count differs")
    target_ranges = {name: [] for name in old_files}
    target_results = []
    archive = np.load(acquisition / "patch/edit.npz", allow_pickle=False)
    check(isinstance(archive, np.lib.npyio.NpzFile), "patch archive must be NPZ")
    with archive:
        expected_keys = {f"t{i}_{part}" for i in range(len(specs)) for part in ("idx", "val")}
        check(set(archive.files) == expected_keys and len(archive.files) == len(expected_keys), "archive fields differ")
        for i, spec in enumerate(specs):
            name = spec["name"]
            check(name in tensor_map, "target missing from model")
            shard, info = tensor_map[name]
            check(info["dtype"] == "BF16" and info["shape"] == spec.get("shape")
                  and spec.get("dtype") == "bfloat16" and spec.get("form") == "sparse", "target format differs")
            check(integer(spec.get("changed_entries")), "invalid expected change count")
            start, end = info["data_offsets"]
            absolute_start, length = starts[shard] + start, end - start
            target_ranges[shard].append((absolute_start, absolute_start + length))
            base_hash, edited_hash = sha(old_files[shard], absolute_start, length), sha(new_files[shard], absolute_start, length)
            check(base_hash == spec.get("base_sha256"), "published base hash differs")
            check(edited_hash == spec.get("edited_sha256"), "published saved hash differs")
            with old_files[shard].open("rb") as stream:
                stream.seek(absolute_start)
                old = np.frombuffer(stream.read(length), dtype="<u2")
            with new_files[shard].open("rb") as stream:
                stream.seek(absolute_start)
                new = np.frombuffer(stream.read(length), dtype="<u2")
            indices, values = archive[f"t{i}_idx"], archive[f"t{i}_val"]
            check(indices.ndim == 1 and indices.dtype.kind in "iu", "invalid indices")
            check(values.ndim == 1 and values.dtype == np.dtype("<u2"), "invalid stored BF16 values")
            check(len(indices) == len(values) and len(np.unique(indices)) == len(indices), "duplicate or mismatched indices")
            check(not indices.size or (indices.min() >= 0 and indices.max() < len(old)), "index out of bounds")
            check(np.array_equal(new[indices], values), "saved substitutions differ")
            differences = np.flatnonzero(old != new)
            check(not np.setdiff1d(differences, indices).size, "extra changed coordinates")
            check(len(differences) == spec["changed_entries"], "stored change count differs")
            result = {"name": name, "shard": shard, "base_sha256": base_hash,
                      "edited_sha256": edited_hash, "sparse_substitutions": len(indices),
                      "changed_stored_entries": len(differences)}
            if "changed_columns" in spec:
                check(len(info["shape"]) == 2 and info["shape"][1] > 0, "column footprint requires a matrix")
                columns = np.unique(indices % info["shape"][1])
                rows = np.unique(indices // info["shape"][1])
                check(integer(spec["changed_columns"]) and len(columns) == spec["changed_columns"], "column support differs")
                check(integer(spec.get("changed_rows")) and len(rows) == spec["changed_rows"], "row support differs")
                result.update(targeted_coordinates=info["shape"][0] * len(columns),
                              targeted_coordinate_basis="published_full_column_footprint", targeted_columns=columns.tolist())
            else:
                # Supports synthetic fixtures that explicitly enumerate their targets.
                result.update(targeted_coordinates=len(indices), targeted_coordinate_basis="archive_indices")
            target_results.append(result)
    output_files = {}
    output_inventory = output_audit.get("model_inventory")
    check(isinstance(output_inventory, dict) and isinstance(output_inventory.get("files"), dict)
          and output_inventory["files"].keys() == new_files.keys(), "output file inventory differs")
    for name, path in old_files.items():
        cursor = 0
        for start, end in sorted(target_ranges[name]):
            check(start >= cursor, "overlapping target spans")
            equal_span(path, new_files[name], cursor, start - cursor)
            cursor = end
        equal_span(path, new_files[name], cursor, path.stat().st_size - cursor)
        output_files[name] = sha(new_files[name])
        check(output_inventory["files"][name] == {"sha256": output_files[name], "size_bytes": new_files[name].stat().st_size},
              "saved file hash differs from audit")
    changed = sum(t["changed_stored_entries"] for t in target_results)
    targeted = sum(t["targeted_coordinates"] for t in target_results)
    check(integer(manifest.get("changed_entries_total"))
          and changed == reference["expected_changed_stored_entries"] == manifest["changed_entries_total"], "total changes differ")
    check(targeted == reference["reported_targeted_entries"], "target footprint differs")
    check(changed == output_audit.get("changed_stored_entries")
          and targeted == output_audit.get("targeted_coordinates"), "audit counts differ")
    return {"independent_verification": "passed", "method": "standalone stored-byte comparison; no project implementation imported",
            "numpy_version": np.__version__, "model_file_count": len(old_files), "stored_tensor_count": len(tensor_map),
            "stored_element_count": sum(math.prod(info["shape"]) for shard, info in tensor_map.values()),
            "unchanged_tensor_count": len(tensor_map) - len(specs), "changed_stored_entries": changed,
            "targeted_coordinates": targeted, "targeted_but_unchanged_stored_entries": targeted - changed,
            "all_non_target_bytes_equal": True, "tensors": target_results,
            "acquisition_audit_sha256": sha(acquisition / "acquisition.json"),
            "reconstruction_audit_sha256": sha(reconstruction / "audit.json"), "output_file_sha256": output_files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("configs/published_p50c.json"))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.acquisition, args.reconstruction, args.reference), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"independent_verification": "failed", "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
