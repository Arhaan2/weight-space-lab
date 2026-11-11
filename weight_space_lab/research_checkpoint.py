"""General direct weight edits for the research notebook, without model execution.

The notebook can inspect every tensor and replace any floating-point tensor in
the 24 text decoder blocks.  Final norm, embeddings, output head, architecture,
tokenizer and all other assets stay immutable.  This is deliberately separate
from the historical, column-limited checkpoint proposal implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from .acquisition import _freeze, _remove_stage
from .checkpoint import _mapped
from .checkpoint_numeric import FLOAT_DTYPES, decode, encode
from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash


POLICY = {
    "name": "research_internal_tensor_replacement_v1",
    "editable_scope": "floating-point tensors in text decoder blocks 0..23",
    "forbidden": ["embedding", "output_head", "final_norm", "whole_tensor_scalar_only_edit"],
    "fixed_tensor_names_shapes_dtypes": True,
    "submission_dtype": "float32",
    "rounding": "submitted float32 rounded once to original stored dtype at materialization",
    "support_or_direction_restriction": None,
}
_BLOCK = re.compile(r"model\.language_model\.layers\.(\d+)\..+\Z")


def _write(path, value, *, replace=False):
    import json
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _outside(path, source):
    path, source = Path(path).absolute(), Path(source).resolve()
    resolved = path.resolve()
    if resolved == source or source in resolved.parents or resolved in source.parents:
        raise ValueError("edit destination must be separate from original checkpoint")
    if path.is_symlink():
        raise ValueError("edit destination must not be a symlink")
    return path


def _editable(name, tensor):
    match = _BLOCK.fullmatch(name)
    return bool(match and 0 <= int(match[1]) < 24 and tensor["dtype"] in FLOAT_DTYPES)


def _original_array(original, inventory, name):
    if name not in inventory["tensors"]:
        raise ValueError("unknown source tensor")
    tensor = inventory["tensors"][name]
    with _mapped(original, tensor) as raw:
        copied = np.array(raw, copy=True)
    if hashlib.sha256(copied.tobytes()).hexdigest() != tensor["sha256"]:
        raise ValueError("original tensor bytes changed")
    return decode(copied, tensor["dtype"]).astype(np.float32, copy=True)


def _replacement(original, inventory, name, values):
    if name not in inventory["tensors"] or not _editable(name, inventory["tensors"][name]):
        raise ValueError("only internal text decoder block tensors may be edited")
    values = np.asarray(values)
    tensor = inventory["tensors"][name]
    if values.dtype.kind != "f" or list(values.shape) != tensor["shape"]:
        raise ValueError("replacement must be floating point with original shape")
    with np.errstate(over="raise", invalid="raise"):
        try:
            values = values.astype(np.float32, copy=True)
        except FloatingPointError as error:
            raise ValueError("replacement overflows float32") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("replacement must contain only finite values")
    base = _original_array(original, inventory, name)
    if np.array_equal(values, base):
        return values
    # This mirrors the original experiment's ban on a whole-tensor scalar knob.
    # The comparison is on the submitted F32 array before source-dtype rounding.
    # Localized head/neuron changes, arbitrary directions, and internal norm
    # changes are permitted; there is no magnitude or coordinate-count cap.
    if base.size and np.any(base):
        index = int(np.argmax(np.abs(base)))
        factor = float(values.flat[index]) / float(base.flat[index])
        scaled = base.astype(np.float64) * factor
        tolerance = np.maximum(np.abs(scaled), np.abs(values)) * (4 * np.finfo(np.float32).eps)
        if np.all(np.abs(values.astype(np.float64) - scaled) <= tolerance):
            raise ValueError("whole-tensor scalar-only edits are out of scope")
    encode(values, tensor["dtype"])  # Reject overflow in the eventual stored dtype.
    return values


def _array_file(path, entry, tensor):
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > tensor["parameter_count"] * 4 + 65536
            or file_sha256(path) != entry["sha256"]):
        raise ValueError("replacement file identity or size differs")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if value.dtype != np.dtype("<f4") or list(value.shape) != tensor["shape"]:
        raise ValueError("replacement file dtype or shape differs")
    return value


class EditWorkspace:
    """Persistent mutable edits, with read-only access to pristine source bytes.

    Callers supervise the notebook and serialize its actions. File locking also
    makes snapshot selection atomic relative to save/drop operations.
    """

    def __init__(self, original, workdir):
        self.original = Path(original).resolve()
        self.inventory = inventory_model(self.original)
        self.original_sha256 = canonical_hash(self.inventory["files"])
        self.NAMES = tuple(sorted(self.inventory["tensors"]))
        self.workdir = _outside(workdir, self.original)
        self.workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.workdir / "edits").mkdir(exist_ok=True, mode=0o700)
        with self._locked():
            if not (self.workdir / "workspace.json").exists():
                _write(self.workdir / "workspace.json", {
                    "schema_version": 1, "kind": "research_edit_workspace",
                    "source_sha256": self.original_sha256, "policy": POLICY,
                    "revision": 0, "edits": {},
                })
            self._state()

    @contextmanager
    def _locked(self):
        fd = os.open(self.workdir / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _state(self):
        state = read_json(self.workdir / "workspace.json")
        if (state.get("schema_version") != 1 or state.get("kind") != "research_edit_workspace"
                or state.get("source_sha256") != self.original_sha256 or state.get("policy") != POLICY):
            raise ValueError("workspace belongs to another source or edit contract")
        return state

    def load_original(self, name):
        return _original_array(self.original, self.inventory, name)

    def load(self, name):
        with self._locked():
            state = self._state()
            if name not in state["edits"]:
                return self.load_original(name)
            entry = state["edits"][name]
            expected = hashlib.sha256(name.encode()).hexdigest() + ".npy"
            if entry.get("file") != expected:
                raise ValueError("unsafe replacement file")
            return np.array(_array_file(self.workdir / "edits" / expected, entry,
                                       self.inventory["tensors"][name]), copy=True)

    def save_edit(self, name, values):
        values = _replacement(self.original, self.inventory, name, values)
        with self._locked():
            state = self._state()
            filename = hashlib.sha256(name.encode()).hexdigest() + ".npy"
            path = self.workdir / "edits" / filename
            temporary = path.with_name("." + filename + "." + uuid.uuid4().hex)
            with temporary.open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            entry = {"file": filename, "sha256": file_sha256(path), "shape": list(values.shape),
                     "submission_dtype": "float32", "source_dtype": self.inventory["tensors"][name]["dtype"]}
            state["edits"][name] = entry
            state["revision"] += 1
            _write(self.workdir / "workspace.json", state, replace=True)
            return copy.deepcopy(entry)

    def drop_edit(self, name):
        if name not in self.inventory["tensors"]:
            raise ValueError("unknown tensor")
        with self._locked():
            state = self._state()
            entry = state["edits"].pop(name, None)
            if entry is not None:
                state["revision"] += 1
                _write(self.workdir / "workspace.json", state, replace=True)
                (self.workdir / "edits" / entry["file"]).unlink()
            return entry is not None

    def edited(self):
        with self._locked():
            return sorted(self._state()["edits"])

    def restore(self, snapshot):
        """Restore a verified named candidate without changing its frozen files."""
        snapshot = Path(snapshot)
        manifest = verify_snapshot(self.original, snapshot)
        with self._locked():
            state = self._state()
            replacement = self.workdir / (".restore-" + uuid.uuid4().hex)
            backup = self.workdir / (".previous-" + uuid.uuid4().hex)
            shutil.copytree(snapshot / "edits", replacement)
            replacement.chmod(0o700)
            for path in replacement.iterdir():
                path.chmod(0o600)
            moved = False
            try:
                (self.workdir / "edits").rename(backup)
                replacement.rename(self.workdir / "edits")
                moved = True
                state["edits"] = copy.deepcopy(manifest["payload"]["edits"])
                state["revision"] += 1
                _write(self.workdir / "workspace.json", state, replace=True)
            except BaseException:
                if moved:
                    _remove_stage(self.workdir / "edits")
                if backup.exists():
                    backup.rename(self.workdir / "edits")
                raise
            finally:
                for path in (replacement, backup):
                    if path.exists():
                        _remove_stage(path)
        return manifest

    def snapshot(self, output):
        output = _outside(output, self.original)
        if output.exists():
            raise FileExistsError("snapshot destination must be new")
        if inventory_model(self.original) != self.inventory:
            raise ValueError("source changed before snapshot")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name("." + output.name + "." + uuid.uuid4().hex)
        try:
            with self._locked():
                state = self._state()
                (temporary / "edits").mkdir(parents=True)
                for name, entry in state["edits"].items():
                    values = self.load_original(name)  # Confirms each source tensor identity.
                    del values
                    expected = hashlib.sha256(name.encode()).hexdigest() + ".npy"
                    if entry["file"] != expected:
                        raise ValueError("unsafe replacement filename")
                    shutil.copyfile(self.workdir / "edits" / expected, temporary / "edits" / expected)
                payload = {"schema_version": 1, "kind": "research_weight_snapshot",
                           "source_sha256": self.original_sha256, "policy": POLICY,
                           "workspace_revision": state["revision"], "edits": state["edits"]}
                manifest = {"payload": payload, "sha256": canonical_hash(payload)}
                _write(temporary / "manifest.json", manifest)
                verify_snapshot(self.original, temporary)
                _freeze(temporary)
                temporary.chmod(0o755)
                temporary.rename(output)
                output.chmod(0o555)
                return manifest
        except BaseException:
            if temporary.exists():
                _remove_stage(temporary)
            raise


def verify_snapshot(original, path):
    """Independently validate snapshot files, source identity and edit scope."""
    original, path = Path(original), Path(path)
    inventory = inventory_model(original)
    if path.is_symlink() or not path.is_dir() or (path / "edits").is_symlink():
        raise ValueError("snapshot must be a regular directory")
    manifest = read_json(path / "manifest.json")
    if set(manifest) != {"payload", "sha256"} or canonical_hash(manifest["payload"]) != manifest["sha256"]:
        raise ValueError("snapshot manifest hash mismatch")
    payload = manifest["payload"]
    if (set(payload) != {"schema_version", "kind", "source_sha256", "policy", "workspace_revision", "edits"}
            or payload["schema_version"] != 1 or payload["kind"] != "research_weight_snapshot"
            or payload["source_sha256"] != canonical_hash(inventory["files"]) or payload["policy"] != POLICY):
        raise ValueError("snapshot source or policy mismatch")
    expected_files = {"manifest.json"}
    for name, entry in payload["edits"].items():
        if name not in inventory["tensors"] or not _editable(name, inventory["tensors"][name]):
            raise ValueError("snapshot edits a forbidden tensor")
        filename = hashlib.sha256(name.encode()).hexdigest() + ".npy"
        if (entry.get("file") != filename or entry.get("shape") != inventory["tensors"][name]["shape"]
                or entry.get("source_dtype") != inventory["tensors"][name]["dtype"]
                or entry.get("submission_dtype") != "float32"):
            raise ValueError("snapshot replacement metadata differs")
        values = _array_file(path / "edits" / filename, entry, inventory["tensors"][name])
        _replacement(original, inventory, name, values)
        expected_files.add("edits/" + filename)
    observed = set()
    for file in path.rglob("*"):
        if file.is_symlink() or not (file.is_dir() or file.is_file()):
            raise ValueError("snapshot contains nonregular entries")
        if file.is_file():
            observed.add(file.relative_to(path).as_posix())
        elif file.relative_to(path).as_posix() != "edits":
            raise ValueError("unexpected snapshot directory")
    if observed != expected_files:
        raise ValueError("snapshot file set differs")
    return manifest


def _audit_materialized(original, snapshot, model):
    manifest = verify_snapshot(original, snapshot)
    before, after = inventory_model(original), inventory_model(model)
    edits = manifest["payload"]["edits"]
    if before["files"].keys() != after["files"].keys() or before["tensors"].keys() != after["tensors"].keys():
        raise ValueError("materialized file or tensor contract changed")
    shards = {before["tensors"][name]["shard"] for name in edits}
    for name, entry in before["files"].items():
        if entry["size_bytes"] != after["files"][name]["size_bytes"] or (name not in shards and entry != after["files"][name]):
            raise ValueError("unrelated file changed")
    for shard in shards:
        boundary = before["shard_layout"][shard]["data_start"]
        with (Path(original) / shard).open("rb") as a, (Path(model) / shard).open("rb") as b:
            if a.read(boundary) != b.read(boundary):
                raise ValueError("stored shard header changed")
    rows = {}
    for name, tensor in before["tensors"].items():
        other = after["tensors"][name]
        if {k: v for k, v in tensor.items() if k != "sha256"} != {k: v for k, v in other.items() if k != "sha256"}:
            raise ValueError("tensor shape, dtype or layout changed")
        if name not in edits:
            if tensor["sha256"] != other["sha256"]:
                raise ValueError("unselected source tensor changed")
            continue
        value = _array_file(Path(snapshot) / "edits" / edits[name]["file"], edits[name], tensor)
        expected = encode(value, tensor["dtype"])
        with _mapped(model, other) as saved:
            if saved.tobytes() != expected.tobytes():
                raise ValueError("saved values differ from frozen replacement")
        with _mapped(original, tensor) as raw:
            old = np.array(raw, copy=True)
        unsigned = np.dtype(f"<u{old.dtype.itemsize}")
        bits = old.view(unsigned) != expected.view(unsigned)
        x, y = decode(old, tensor["dtype"]).astype(np.float64), decode(expected, tensor["dtype"]).astype(np.float64)
        delta = y - x
        rows[name] = {"source_dtype": tensor["dtype"], "shape": tensor["shape"],
                      "base_sha256": tensor["sha256"], "result_sha256": other["sha256"],
                      "changed_stored_entries": int(bits.sum()), "changed_numerical_entries": int(np.count_nonzero(delta)),
                      "l2_norm": float(np.linalg.norm(delta)), "max_abs_change": float(np.max(np.abs(delta))) if delta.size else 0.0}
    if not all(math.isfinite(row["l2_norm"]) for row in rows.values()):
        raise ValueError("nonfinite stored edit norm")
    changed = sum(row["changed_stored_entries"] for row in rows.values())
    return {"schema_version": 1, "kind": "research_checkpoint_audit",
            "base_sha256": canonical_hash(before["files"]), "result_sha256": canonical_hash(after["files"]),
            "snapshot_sha256": manifest["sha256"], "policy": POLICY, "per_tensor": rows,
            "changed_stored_entries": changed,
            "changed_numerical_entries": sum(row["changed_numerical_entries"] for row in rows.values()),
            "total_stored_coordinates": before["parameter_count"],
            "changed_fraction": changed / before["parameter_count"] if before["parameter_count"] else 0,
            "unchanged_tensor_count": sum(a["sha256"] == after["tensors"][n]["sha256"] for n, a in before["tensors"].items())}


def materialize(original, snapshot, output):
    """Build an independent complete source checkpoint from one frozen snapshot."""
    original, snapshot = Path(original), Path(snapshot)
    output = _outside(output, original)
    if output.exists():
        raise FileExistsError("materialization destination must be new")
    manifest = verify_snapshot(original, snapshot)
    inventory = inventory_model(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + "." + uuid.uuid4().hex)
    try:
        temporary.mkdir()
        shutil.copytree(snapshot, temporary / "snapshot")
        shutil.copytree(original, temporary / "model")
        for name, entry in manifest["payload"]["edits"].items():
            tensor = inventory["tensors"][name]
            path = temporary / "model" / tensor["shard"]
            path.chmod(0o600)
            value = _array_file(snapshot / "edits" / entry["file"], entry, tensor)
            with _mapped(temporary / "model", tensor, "r+") as saved:
                saved[...] = encode(value, tensor["dtype"])
        audit = _audit_materialized(original, temporary / "snapshot", temporary / "model")
        _write(temporary / "audit.json", audit)
        _freeze(temporary)
        temporary.chmod(0o755)
        temporary.rename(output)
        output.chmod(0o555)
        return audit
    except BaseException:
        if temporary.exists():
            _remove_stage(temporary)
        raise


def verify_materialized(original, bundle):
    bundle = Path(bundle)
    audit = _audit_materialized(original, bundle / "snapshot", bundle / "model")
    if audit != read_json(bundle / "audit.json"):
        raise ValueError("materialized checkpoint audit differs")
    return audit
