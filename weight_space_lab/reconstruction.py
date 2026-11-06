"""Transactional full-directory replay and independent saved-byte verification.

The original snapshot is only read. Model loading, inference, and evaluation
are deliberately outside this artifact workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .inventory import file_sha256, inventory_model, read_json
from .replay import _integer, _spec, apply_sparse_bf16


def _manifest(path, reference):
    manifest = read_json(path)
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
    if set(names) != set(reference["tensor_names"]):
        raise ValueError("manifest targets differ from the pinned reference")
    if _integer(manifest.get("changed_tensors"), "changed_tensors") != len(specs):
        raise ValueError("manifest tensor count differs")
    if (_integer(manifest.get("changed_entries_total"), "changed_entries_total")
            != sum(spec["changed_entries"] for spec in specs)):
        raise ValueError("manifest changed-entry total differs")
    return manifest


def _archive(path, specs):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("patch archive must be NPZ")
    expected = {f"t{i}_{suffix}" for i in range(len(specs)) for suffix in ("idx", "val")}
    if set(archive.files) != expected or len(archive.files) != len(expected):
        archive.close()
        raise ValueError("patch archive fields differ from manifest")
    return archive


def _tensor_bytes(model, tensor):
    start, end = tensor["data_offsets"]
    with (Path(model) / tensor["shard"]).open("rb") as stream:
        stream.seek(tensor["data_start"] + start)
        payload = stream.read(end - start)
    if len(payload) != end - start:
        raise ValueError("truncated tensor payload")
    return payload


def _target_support(indices, spec):
    """Describe observed sparse support and the published full-column scope.

    The released archive omits substitutions that round back to base BF16.
    Its indices are NOT a record of every coordinate the editor targeted.
    The column footprint is derived from actual sparse support and matrix
    shape, using the release's documented full-column operation.
    """
    count = int(indices.size)
    support = {"sparse_substitutions": count, "targeted_coordinates": count,
               "targeted_coordinate_basis": "archive_indices"}
    if "changed_columns" in spec:
        if len(spec["shape"]) != 2:
            raise ValueError("published column footprint requires a matrix")
        rows, width = spec["shape"]
        columns = np.unique(indices % width)
        observed_rows = np.unique(indices // width)
        if _integer(spec["changed_columns"], "changed_columns") != len(columns):
            raise ValueError("manifest changed_columns differs from sparse support")
        if ("changed_rows" in spec and
                _integer(spec["changed_rows"], "changed_rows") != len(observed_rows)):
            raise ValueError("manifest changed_rows differs from sparse support")
        support.update(targeted_coordinates=rows * len(columns),
                       targeted_coordinate_basis="all_rows_of_observed_columns_in_published_column_edit",
                       targeted_columns=columns.tolist(), observed_sparse_rows=len(observed_rows))
    return support


def _verify_saved_model(base, saved, manifest_path, archive_path, reference):
    """Read saved files afresh; never call the edit implementation to verify it."""
    before = inventory_model(base)
    after = inventory_model(saved)
    manifest = _manifest(manifest_path, reference)
    specs = manifest["tensors"]
    target_names = {spec["name"] for spec in specs}
    if before["files"].keys() != after["files"].keys():
        raise ValueError("saved model file set differs from base")
    if before["tensors"].keys() != after["tensors"].keys():
        raise ValueError("saved tensor set differs from base")
    if not target_names <= before["tensors"].keys():
        raise ValueError("published target is missing from the checkpoint")
    target_shards = {before["tensors"][name]["shard"] for name in target_names}
    for name, info in before["files"].items():
        if name not in target_shards and info != after["files"][name]:
            raise ValueError("an unrelated model file changed")
    for name, tensor in before["tensors"].items():
        other = after["tensors"][name]
        if {k: v for k, v in tensor.items() if k != "sha256"} != {
                k: v for k, v in other.items() if k != "sha256"}:
            raise ValueError("saved tensor layout, shape, or stored dtype changed")
        if name not in target_names and tensor != other:
            raise ValueError("an unrelated tensor changed")
    # Tensor hashes alone would miss altered metadata or whitespace in headers.
    for shard in target_shards:
        boundary = next(t["data_start"] for t in before["tensors"].values() if t["shard"] == shard)
        with (Path(base) / shard).open("rb") as left, (Path(saved) / shard).open("rb") as right:
            if left.read(boundary) != right.read(boundary):
                raise ValueError("saved shard header changed")
    audits = []
    with _archive(archive_path, specs) as archive:
        for i, spec in enumerate(specs):
            name = spec["name"]
            tensor, other = before["tensors"][name], after["tensors"][name]
            if tensor["dtype"] != "BF16" or tensor["shape"] != spec["shape"]:
                raise ValueError("target dtype or shape differs from manifest")
            if tensor["sha256"] != spec["base_sha256"] or other["sha256"] != spec["edited_sha256"]:
                raise ValueError("saved/base tensor hash differs from published hash")
            old = np.frombuffer(_tensor_bytes(base, tensor), dtype="<u2")
            new = np.frombuffer(_tensor_bytes(saved, other), dtype="<u2")
            indices, values = archive[f"t{i}_idx"], archive[f"t{i}_val"]
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise ValueError("archive indices must be a one-dimensional integer array")
            if (values.ndim != 1 or values.dtype != np.dtype("<u2") or len(values) != len(indices)
                    or (indices.size and (indices.min() < 0 or indices.max() >= old.size))
                    or np.unique(indices).size != indices.size):
                raise ValueError("invalid sparse coordinates or BF16 values")
            differences = np.flatnonzero(old != new)
            if np.setdiff1d(differences, indices).size or not np.array_equal(new[indices], values):
                raise ValueError("saved values do not match the exact sparse substitutions")
            changed = int(differences.size)
            if changed != spec["changed_entries"]:
                raise ValueError("observed saved changes differ from manifest")
            audits.append({"name": name, "shard": tensor["shard"],
                           "base_sha256": tensor["sha256"], "edited_sha256": other["sha256"],
                           "changed_stored_entries": changed, **_target_support(indices, spec)})
    changed = sum(row["changed_stored_entries"] for row in audits)
    targeted = sum(row["targeted_coordinates"] for row in audits)
    if changed != manifest["changed_entries_total"] or changed != reference["expected_changed_stored_entries"]:
        raise ValueError("observed saved change total differs from pinned expectation")
    if targeted != reference["reported_targeted_entries"]:
        raise ValueError("derived targeted-coordinate total differs from pinned expectation")
    return {"model_inventory": after, "changed_stored_entries": changed,
            "targeted_coordinates": targeted,
            "sparse_substitutions": sum(row["sparse_substitutions"] for row in audits),
            "targeted_but_unchanged_stored_entries": targeted - changed,
            "unchanged_tensor_count": len(before["tensors"]) - len(specs),
            "tensors": audits}


def _audit(acquisition, output_model, acquisition_record):
    acquisition = Path(acquisition)
    reference = read_json(acquisition / "reference.json")
    observed = _verify_saved_model(acquisition / "model", output_model,
                                  acquisition / "patch/manifest.json",
                                  acquisition / "patch/edit.npz", reference)
    return {"schema_version": 1, "kind": "published_patch_reconstruction", "complete": True,
            "artifact_verification": "passed", "inference_executed": False,
            "experiment_executed": False,
            "acquisition_sha256": file_sha256(acquisition / "acquisition.json"),
            "reference_sha256": file_sha256(acquisition / "reference.json"),
            "manifest_sha256": file_sha256(acquisition / "patch/manifest.json"),
            "archive_sha256": file_sha256(acquisition / "patch/edit.npz"),
            "resolved_source_revision": acquisition_record["resolved_source_revision"],
            "resolved_model_revision": acquisition_record["resolved_model_revision"],
            **observed}


def verify_reconstruction(acquisition, output, config_path=None):
    """Recompute evidence from base, patch and saved files, then check the audit."""
    from .acquisition import verify_acquisition
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("reconstruction must be a regular bundle directory")
    if {p.name for p in output.iterdir()} != {"model", "audit.json"}:
        raise ValueError("reconstruction is incomplete or has unexpected files")
    file_sha256(output / "audit.json")  # Reject links, devices and pipes before parsing.
    acquisition_record = verify_acquisition(acquisition, config_path)
    observed = _audit(acquisition, output / "model", acquisition_record)
    if read_json(output / "audit.json") != observed:
        raise ValueError("reconstruction audit does not match independently verified saved files")
    return observed


def reconstruct_published(acquisition, output, config_path=None):
    """Copy all assets, replay exact bits, verify, then publish model and audit together.

    A hidden sibling staging directory is never a finalized artifact. Any
    handled exception removes it; process termination may leave only that
    explicitly incomplete directory. Existing destinations are refused.
    """
    from .acquisition import _remove_stage, verify_acquisition
    acquisition, output = Path(acquisition), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be a new path")
    if acquisition.resolve() in output.resolve().parents:
        raise ValueError("reconstruction output must be outside the immutable acquisition")
    acquisition_record = verify_acquisition(acquisition, config_path)
    reference = read_json(acquisition / "reference.json")
    manifest = _manifest(acquisition / "patch/manifest.json", reference)
    inventory = acquisition_record["model_inventory"]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        # copyfile creates independent writable files, never aliases to base.
        shutil.copytree(acquisition / "model", stage / "model", copy_function=shutil.copyfile)
        with _archive(acquisition / "patch/edit.npz", manifest["tensors"]) as archive:
            for i, spec in enumerate(manifest["tensors"]):
                tensor = inventory["tensors"].get(spec["name"])
                if tensor is None or tensor["dtype"] != "BF16" or tensor["shape"] != spec["shape"]:
                    raise ValueError("target is missing or incompatible with the actual model layout")
                bits = np.frombuffer(_tensor_bytes(acquisition / "model", tensor), dtype="<u2").reshape(spec["shape"])
                edited, _ = apply_sparse_bf16(bits, spec, archive[f"t{i}_idx"], archive[f"t{i}_val"])
                with (stage / "model" / tensor["shard"]).open("r+b") as destination:
                    destination.seek(tensor["data_start"] + tensor["data_offsets"][0])
                    destination.write(edited.tobytes())
                    destination.flush()
                    os.fsync(destination.fileno())
        observed = _audit(acquisition, stage / "model", acquisition_record)
        with (stage / "audit.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(observed, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck original provenance and the serialized audit before commit.
        verify_reconstruction(acquisition, stage, config_path)
        if output.exists() or output.is_symlink():
            raise FileExistsError("output appeared during reconstruction")
        stage.rename(output)
        return observed
    except BaseException:
        _remove_stage(stage)
        raise
