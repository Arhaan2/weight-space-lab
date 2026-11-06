"""Bounded, dtype-aware editing of complete Safetensors model directories.

Only selected coordinates enter arithmetic. Unrelated data is streamed as
bytes, never converted through a model-wide NumPy state or framework loader.
The caller supplies the trusted original and intervention policy; this module
is an operation boundary, not an OS sandbox for an untrusted Python process.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile

import numpy as np

from .acquisition import _freeze, _remove_stage
from .checkpoint_numeric import FLOAT_DTYPES, PRECISION, STORAGE_DTYPES, decode, finite, operation
from .inventory import inventory_model, read_json
from .patches import _columns, _finite_number, _stable_norm
from .protocol import canonical_hash


_CHUNK = 65536
_OPS = {"scale_columns", "remove_direction_from_columns"}


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _names(value, available, label):
    if (not isinstance(value, list) or any(not isinstance(x, str) or x not in available for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} must list unique existing tensor names")
    return value


def validate_policy(policy, inventory):
    keys = {"schema_version", "allowed_tensors", "allowed_operations", "allowed_columns",
            "max_changed_fraction", "max_changed_coordinates", "max_targeted_coordinates",
            "max_operations", "max_working_elements", "scale_factor_range", "direction_strength_range",
            "max_abs_change", "inspection"}
    if not isinstance(policy, dict) or set(policy) != keys or type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("unsupported or incomplete candidate intervention policy")
    tensors = inventory["tensors"]
    names = _names(policy["allowed_tensors"], tensors, "allowed_tensors")
    ops = policy["allowed_operations"]
    if not isinstance(ops, list) or any(not isinstance(x, str) or x not in _OPS for x in ops) or len(set(ops)) != len(ops):
        raise ValueError("allowed_operations must list unique supported operations")
    columns = policy["allowed_columns"]
    if not isinstance(columns, dict) or set(columns) != set(names):
        raise ValueError("allowed_columns must specify every allowlisted tensor")
    for name in names:
        tensor = tensors[name]
        if len(tensor["shape"]) != 2 or tensor["dtype"] not in FLOAT_DTYPES:
            raise ValueError("editable tensors must be supported floating-point matrices")
        if columns[name] != "all":
            _columns(columns[name], tensor["shape"][1])
    fraction = _finite_number(policy["max_changed_fraction"], "max_changed_fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("max_changed_fraction must be in [0, 1]")
    for key in ("max_changed_coordinates", "max_targeted_coordinates", "max_operations", "max_working_elements"):
        _integer(policy[key], key, int(key == "max_working_elements"))
    if _finite_number(policy["max_abs_change"], "max_abs_change") < 0:
        raise ValueError("max_abs_change must be nonnegative")
    for key in ("scale_factor_range", "direction_strength_range"):
        bounds = policy[key]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{key} must be a finite [minimum, maximum]")
        lo, hi = (_finite_number(x, key) for x in bounds)
        if lo > hi:
            raise ValueError(f"{key} is reversed")
    inspection = policy["inspection"]
    if not isinstance(inspection, dict) or set(inspection) != {
            "allowed_tensors", "max_requests", "max_values", "max_scanned_coordinates",
            "max_total_values", "max_total_scanned_coordinates"}:
        raise ValueError("inspection limits must be explicit")
    _names(inspection["allowed_tensors"], tensors, "inspection allowed_tensors")
    for key, value in inspection.items():
        if key != "allowed_tensors":
            _integer(value, key)
    return copy.deepcopy(policy)


def _proposal(patch, policy, inventory, original_sha256):
    if not isinstance(patch, dict) or set(patch) != {
            "schema_version", "base_sha256", "operations", "provenance", "composition"}:
        raise ValueError("candidate proposal has missing or unknown fields")
    if type(patch["schema_version"]) is not int or patch["schema_version"] != 1:
        raise ValueError("unsupported candidate proposal version")
    if patch["base_sha256"] != original_sha256:
        raise ValueError("proposal base_sha256 differs from immutable original")
    if patch["provenance"] not in ("engineering_edit", "editor_proposal"):
        raise ValueError("proposal must declare its provenance")
    if patch["composition"] not in ("independent", "sequential"):
        raise ValueError("unsupported composition; candidates always start from original")
    ops = patch["operations"]
    if not isinstance(ops, list) or len(ops) > policy["max_operations"]:
        raise ValueError("operation budget exceeded or operations is not a list")
    if patch["composition"] == "independent" and len(ops) > 1:
        raise ValueError("multiple operations require declared sequential composition")
    support = {}
    for op in ops:
        if not isinstance(op, dict) or not isinstance(op.get("op"), str) or op["op"] not in policy["allowed_operations"]:
            raise ValueError("operation is not permitted by intervention policy")
        key = "factor" if op["op"] == "scale_columns" else "strength"
        required = {"op", "tensor", "columns", key} | ({"direction"} if key == "strength" else set())
        if set(op) != required:
            raise ValueError("operation has missing or unknown keys")
        name = op["tensor"]
        if not isinstance(name, str) or name not in policy["allowed_tensors"]:
            raise ValueError("operation tensor is not allowlisted")
        rows, width = inventory["tensors"][name]["shape"]
        columns = _columns(op["columns"], width)
        allowed = policy["allowed_columns"][name]
        if allowed != "all" and not set(columns) <= set(allowed):
            raise ValueError("operation columns violate intervention policy")
        if rows * len(columns) > policy["max_working_elements"]:
            raise ValueError("operation exceeds bounded working element limit")
        number = _finite_number(op[key], key)
        bounds = policy["scale_factor_range" if key == "factor" else "direction_strength_range"]
        if not bounds[0] <= number <= bounds[1]:
            raise ValueError("operation parameter violates intervention policy")
        if key == "strength":
            direction = op["direction"]
            if not isinstance(direction, list) or len(direction) != rows:
                raise ValueError("direction must match the tensor row count")
            values = [_finite_number(x, "direction component") for x in direction]
            if not any(values):
                raise ValueError("direction must not be zero")
        support.setdefault(name, set()).update(columns)
    targeted = sum(inventory["tensors"][name]["shape"][0] * len(cols) for name, cols in support.items())
    if targeted > policy["max_targeted_coordinates"]:
        raise ValueError("targeted coordinate budget exceeded")
    if any(inventory["tensors"][name]["shape"][0] * len(cols) > policy["max_working_elements"]
           for name, cols in support.items()):
        raise ValueError("composed tensor selection exceeds bounded working element limit")
    return support, targeted


@contextmanager
def _mapped(model, tensor, mode="r"):
    """Map one stored tensor in its exact dtype, without copying its payload."""
    if not tensor["parameter_count"]:
        yield np.empty(tensor["shape"], dtype=STORAGE_DTYPES[tensor["dtype"]])
        return
    value = np.memmap(Path(model) / tensor["shard"], dtype=STORAGE_DTYPES[tensor["dtype"]],
                      mode=mode, offset=tensor["data_start"] + tensor["data_offsets"][0],
                      shape=tuple(tensor["shape"]))
    try:
        yield value
        if mode != "r":
            value.flush()
    finally:
        value._mmap.close()


def _check_finite(model, inventory):
    for tensor in inventory["tensors"].values():
        if tensor["dtype"] not in FLOAT_DTYPES and not tensor["dtype"].startswith("F8"):
            continue
        with _mapped(model, tensor) as mapped:
            flat = mapped.reshape(-1)
            for start in range(0, flat.size, _CHUNK):
                if not np.all(finite(flat[start:start + _CHUNK], tensor["dtype"])):
                    raise ValueError("checkpoint contains nonfinite floating-point values")


def _write_operation(model, tensor, op):
    with _mapped(model, tensor, "r+") as mapped:
        selected = np.array(mapped[:, op["columns"]], copy=True)
        result = operation(selected, tensor["dtype"], op)
        if result.shape != selected.shape or result.dtype != selected.dtype:
            raise ValueError("operation changed stored shape or dtype")
        mapped[:, op["columns"]] = result


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _outside(path, original, label):
    if path.resolve() == original.resolve() or original.resolve() in path.resolve().parents:
        raise ValueError(f"{label} must be outside the immutable original")


def _verify_operation_result(original, saved, before, after, patch, support):
    """Recompute only bounded declared selections, independently of the writer.

    Reuse the specified arithmetic, but never a writer's in-memory result or
    claimed hash. Full-tensor byte comparisons below separately prove that
    the writer did not change anything outside these selections.
    """
    for name, columns in support.items():
        columns = sorted(columns)
        positions = {col: i for i, col in enumerate(columns)}
        tensor = before["tensors"][name]
        with _mapped(original, tensor) as mapped:
            expected = np.array(mapped[:, columns], copy=True)
        for op in patch["operations"]:
            if op["tensor"] == name:
                selected = [positions[col] for col in op["columns"]]
                expected[:, selected] = operation(expected[:, selected], tensor["dtype"], op)
        with _mapped(saved, after["tensors"][name]) as mapped:
            observed = np.array(mapped[:, columns], copy=True)
        unsigned = np.dtype(f"<u{expected.dtype.itemsize}")
        if not np.array_equal(expected.view(unsigned), observed.view(unsigned)):
            raise ValueError("saved selected values differ from declared operation result")


def _verify_saved(original, saved, before, policy, patch):
    """Reopen the saved bytes; count bit and numeric changes independently."""
    support, targeted = _proposal(patch, policy, before, canonical_hash(before["files"]))
    after = inventory_model(saved)
    if before["files"].keys() != after["files"].keys() or before["tensors"].keys() != after["tensors"].keys():
        raise ValueError("saved file or tensor set changed")
    target_shards = {before["tensors"][name]["shard"] for name in support}
    for name, info in before["files"].items():
        if info["size_bytes"] != after["files"][name]["size_bytes"]:
            raise ValueError("saved file size changed")
        if name not in target_shards and info != after["files"][name]:
            raise ValueError("unrelated asset or shard changed")
    for name, tensor in before["tensors"].items():
        other = after["tensors"][name]
        if {k: v for k, v in tensor.items() if k != "sha256"} != {k: v for k, v in other.items() if k != "sha256"}:
            raise ValueError("stored tensor shape, dtype or shard layout changed")
        if name not in support and tensor["sha256"] != other["sha256"]:
            raise ValueError("unrelated tensor changed")
    for shard in target_shards:
        boundary = before["shard_layout"][shard]["data_start"]
        with (Path(original) / shard).open("rb") as left, (Path(saved) / shard).open("rb") as right:
            if left.read(boundary) != right.read(boundary):
                raise ValueError("stored shard header changed")
    _check_finite(saved, after)
    _verify_operation_result(original, saved, before, after, patch, support)
    per_tensor = {}
    bits_total = numeric_total = 0
    max_change = 0.0
    norm = 0.0
    for name in sorted(support):
        tensor, other = before["tensors"][name], after["tensors"][name]
        bits_count = numeric_count = 0
        tensor_norm = tensor_max = 0.0
        with _mapped(original, tensor) as old, _mapped(saved, other) as new:
            a, b = old.reshape(-1), new.reshape(-1)
            raw_type = np.dtype(f"<u{a.dtype.itemsize}")
            width = tensor["shape"][1]
            for start in range(0, a.size, _CHUNK):
                left, right = a[start:start + _CHUNK], b[start:start + _CHUNK]
                changed = left.view(raw_type) != right.view(raw_type)
                changed_indices = np.flatnonzero(changed) + start
                if any(int(col) not in support[name] for col in np.unique(changed_indices % width)):
                    raise ValueError("saved changes escaped declared columns")
                x = decode(left[changed], tensor["dtype"]).astype(np.float64)
                y = decode(right[changed], tensor["dtype"]).astype(np.float64)
                bits_count += int(np.count_nonzero(changed))
                numeric_count += int(np.count_nonzero(x != y))
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        delta = y - x
                        tensor_norm = math.hypot(tensor_norm, _stable_norm(delta))
                    except FloatingPointError as error:
                        raise ValueError("stored change exceeds supported numeric range") from error
                tensor_max = max(tensor_max, float(np.max(np.abs(delta))) if delta.size else 0.0)
        bits_total += bits_count
        numeric_total += numeric_count
        max_change = max(max_change, tensor_max)
        norm = math.hypot(norm, tensor_norm)
        per_tensor[name] = {"dtype": tensor["dtype"], "shape": tensor["shape"], "shard": tensor["shard"],
                            "base_sha256": tensor["sha256"], "result_sha256": other["sha256"],
                            "changed_stored_entries": bits_count, "changed_numerical_entries": numeric_count,
                            "targeted_coordinates": tensor["shape"][0] * len(support[name]),
                            "l2_norm": tensor_norm, "max_abs_change": tensor_max}
    denominator = before["parameter_count"]
    fraction = bits_total / denominator if denominator else 0.0
    if bits_total > policy["max_changed_coordinates"] or fraction > policy["max_changed_fraction"]:
        raise ValueError("actual stored changed-coordinate budget exceeded")
    if max_change > policy["max_abs_change"]:
        raise ValueError("actual stored change exceeds intervention magnitude policy")
    if not math.isfinite(norm):
        raise ValueError("stored change norm exceeds supported numeric range")
    return {"model_inventory": after, "result_sha256": canonical_hash(after["files"]),
            "total_stored_coordinates": denominator, "targeted_coordinates": targeted,
            "changed_stored_entries": bits_total, "changed_numerical_entries": numeric_total,
            "bit_only_changed_entries": bits_total - numeric_total, "changed_fraction": fraction,
            "l2_norm": norm, "max_abs_change": max_change,
            "unchanged_tensor_count": sum(t["sha256"] == after["tensors"][n]["sha256"] for n, t in before["tensors"].items()),
            "per_tensor": per_tensor,
            "budget_basis": "final stored-bit differences / all stored tensor coordinates, including buffers and aliases"}


def _inspection_plan(request, inventory, limits):
    if not isinstance(request, dict):
        raise ValueError("inspection request must be an object")
    if request.get("op") == "metadata":
        if set(request) != {"op", "tensors"}:
            raise ValueError("metadata request has missing or unknown fields")
        names = _names(request["tensors"], limits["allowed_tensors"], "inspection tensors")
        return len(names), 0, None
    if request.get("op") not in ("slice", "statistics"):
        raise ValueError("unsupported bounded inspection operation")
    if (set(request) != {"op", "tensor", "slices"} or not isinstance(request["tensor"], str)
            or request["tensor"] not in limits["allowed_tensors"]):
        raise ValueError("inspection tensor or fields violate policy")
    tensor = inventory["tensors"][request["tensor"]]
    slices = request["slices"]
    if not isinstance(slices, list) or len(slices) != len(tensor["shape"]):
        raise ValueError("declare one slice per tensor dimension")
    selection, shape = [], []
    for span, size in zip(slices, tensor["shape"]):
        if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                or not 0 <= span[0] <= span[1] <= size):
            raise ValueError("slice bounds are invalid")
        selection.append(slice(*span))
        shape.append(span[1] - span[0])
    scanned = math.prod(shape)
    return scanned if request["op"] == "slice" else 6, scanned, tuple(selection)


def _inspection_response(original, inventory, request, selection, scanned):
    if selection is None:
        return {"tensors": {n: copy.deepcopy(inventory["tensors"][n]) for n in request["tensors"]}}
    tensor = inventory["tensors"][request["tensor"]]
    with _mapped(original, tensor) as mapped:
        raw = np.array(mapped[selection], copy=True)
    values = decode(raw, tensor["dtype"])
    response = {"tensor": request["tensor"], "dtype": tensor["dtype"], "shape": list(values.shape)}
    if request["op"] == "slice":
        response.update(values=values.tolist())
    else:
        # Integer extrema remain exact; mean and norm are explicitly
        # approximate binary64 summaries, never used for editing.
        work = values.astype(np.float64)
        maximum = float(np.max(np.abs(work))) if work.size else 0.0
        mean = float(np.mean(work / maximum) * maximum) if maximum else 0.0
        response["statistics"] = {"count": scanned, "min": values.min().item() if scanned else None,
            "max": values.max().item() if scanned else None, "mean": mean if scanned else None,
            "l2_norm": _stable_norm(work), "finite": bool(np.all(finite(raw, tensor["dtype"])))}
        response["statistics_precision"] = "exact extrema; binary64 approximate mean and L2"
    return response


class CheckpointEditor:
    """One immutable original, explicit policy, and persistent exposure journal."""

    def __init__(self, original_model, policy, *, exposure_log):
        self.original = Path(original_model)
        self.inventory = inventory_model(self.original)
        self.original_sha256 = canonical_hash(self.inventory["files"])
        self.policy = validate_policy(policy, self.inventory)
        self.exposure_log = Path(exposure_log)
        _outside(self.exposure_log, self.original, "exposure log")
        _check_finite(self.original, self.inventory)
        self._session = {"schema_version": 1, "kind": "checkpoint_exposure_session",
                         "original_sha256": self.original_sha256, "policy_sha256": canonical_hash(self.policy)}
        with self._journal() as (_, records):
            self._validate_records(records)

    def _assert_original(self):
        if inventory_model(self.original) != self.inventory:
            raise ValueError("immutable original changed since editor creation")

    @contextmanager
    def _journal(self):
        self.exposure_log.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.exposure_log, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("exposure journal must be a regular file without hard links")
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            content = stream.read()
            if content and not content.endswith("\n"):
                raise ValueError("incomplete exposure journal")
            records = [json.loads(line) for line in content.splitlines()]
            if not records:
                self._append(stream, self._session)
                records = [self._session]
            yield stream, records

    @staticmethod
    def _append(stream, record):
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def _validate_records(self, records):
        if not isinstance(records, list) or not records or records[0] != self._session:
            raise ValueError("exposure journal belongs to a different original or policy")
        limits = self.policy["inspection"]
        if len(records) - 1 > limits["max_requests"]:
            raise ValueError("exposure journal exceeds request budget")
        total_values = total_scanned = 0
        for i, row in enumerate(records[1:], 1):
            if (not isinstance(row, dict) or set(row) != {"sequence", "request", "response", "response_sha256", "values_exposed", "coordinates_scanned"}
                    or type(row["sequence"]) is not int or row["sequence"] != i
                    or row["response_sha256"] != canonical_hash(row["response"])):
                raise ValueError("invalid exposure journal record")
            _integer(row["values_exposed"], "values_exposed")
            _integer(row["coordinates_scanned"], "coordinates_scanned")
            exposed, scanned, selection = _inspection_plan(row["request"], self.inventory, limits)
            total_values += exposed
            total_scanned += scanned
            if (row["values_exposed"] != exposed or row["coordinates_scanned"] != scanned
                    or exposed > limits["max_values"] or scanned > limits["max_scanned_coordinates"]
                    or total_values > limits["max_total_values"] or total_scanned > limits["max_total_scanned_coordinates"]):
                raise ValueError("exposure journal accounting or limits differ from declared requests")
            expected = _inspection_response(self.original, self.inventory, row["request"], selection, scanned)
            if canonical_hash(expected) != row["response_sha256"]:
                raise ValueError("exposure journal differs from declared original weight information")

    def exposure_snapshot(self):
        with self._journal() as (_, records):
            self._validate_records(records)
            return copy.deepcopy(records)

    def inspect(self, request):
        """Return only a declared bounded view and log the exact exposed response."""
        self._assert_original()
        request = copy.deepcopy(request)
        limits = self.policy["inspection"]
        exposed, scanned, selection = _inspection_plan(request, self.inventory, limits)
        with self._journal() as (stream, records):
            self._validate_records(records)
            if (len(records) > limits["max_requests"] or exposed > limits["max_values"]
                    or scanned > limits["max_scanned_coordinates"]
                    or exposed + sum(r["values_exposed"] for r in records[1:]) > limits["max_total_values"]
                    or scanned + sum(r["coordinates_scanned"] for r in records[1:]) > limits["max_total_scanned_coordinates"]):
                raise ValueError("inspection exposure budget exceeded")
            response = _inspection_response(self.original, self.inventory, request, selection, scanned)
            self._assert_original()
            record = {"sequence": len(records), "request": request, "response": response,
                      "response_sha256": canonical_hash(response), "values_exposed": exposed, "coordinates_scanned": scanned}
            self._append(stream, record)  # Persist before revealing the response.
            return response

    def propose(self, patch, output):
        """Stage an independent byte copy, edit, reload/verify, then publish once."""
        patch = copy.deepcopy(patch)
        _proposal(patch, self.policy, self.inventory, self.original_sha256)
        output = Path(output)
        _outside(output, self.original, "candidate output")
        if output.resolve() in self.original.resolve().parents:
            raise ValueError("candidate output cannot contain the original")
        if output.resolve() in self.exposure_log.resolve().parents:
            raise ValueError("candidate output cannot contain the exposure journal")
        if output.exists() or output.is_symlink():
            raise FileExistsError("candidate output must be new")
        self._assert_original()
        exposures = self.exposure_snapshot()
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
        try:
            shutil.copytree(self.original, stage / "model", copy_function=shutil.copyfile)
            for op in patch["operations"]:
                _write_operation(stage / "model", self.inventory["tensors"][op["tensor"]], op)
            observed = _verify_saved(self.original, stage / "model", self.inventory, self.policy, patch)
            audit = _audit(self.original_sha256, self.policy, patch, exposures, observed)
            for name, value in (("proposal", patch), ("policy", self.policy), ("exposure", exposures), ("audit", audit)):
                _write_json(stage / f"{name}.json", value)
            # Reopen serialized evidence too; any failure discards the whole bundle.
            if verify_candidate(self.original, stage, self.policy) != audit:
                raise ValueError("candidate audit did not survive reload")
            self._assert_original()
            for path in stage.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for directory in [p for p in stage.rglob("*") if p.is_dir()] + [stage]:
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _freeze(stage)
            if output.exists() or output.is_symlink():
                raise FileExistsError("candidate output appeared during proposal")
            stage.rename(output)  # Acceptance point; no fallible writes follow.
            return audit
        except BaseException:
            _remove_stage(stage)
            raise


def _audit(original_sha256, policy, patch, exposures, observed):
    return {"schema_version": 1, "kind": "candidate_checkpoint", "complete": True,
            "artifact_verification": "passed", "base_sha256": original_sha256,
            "policy_sha256": canonical_hash(policy), "proposal_sha256": canonical_hash(patch),
            "exposure_sha256": canonical_hash(exposures), "exposure_request_count": len(exposures) - 1,
            "provenance": patch["provenance"], "composition": patch["composition"],
            "precision": PRECISION, "operation_count": len(patch["operations"]),
            "numeric_runtime": {"numpy_version": np.__version__},
            "inference_executed": False, "astra_discovery_claimed": False, "experiment_executed": False,
            **observed}


def verify_candidate(original_model, candidate, policy):
    """Independently recompute saved-byte evidence against the trusted original."""
    original, candidate = Path(original_model), Path(candidate)
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ValueError("candidate bundle must be a directory, not a link")
    if {p.name for p in candidate.iterdir()} != {"model", "policy.json", "proposal.json", "exposure.json", "audit.json"}:
        raise ValueError("candidate bundle is incomplete or has unexpected evidence")
    for name in ("policy", "proposal", "exposure", "audit"):
        if not stat.S_ISREG((candidate / f"{name}.json").lstat().st_mode):
            raise ValueError("candidate evidence must be regular files")
    before = inventory_model(original)
    policy = validate_policy(policy, before)
    if read_json(candidate / "policy.json") != policy:
        raise ValueError("candidate policy differs from trusted intervention policy")
    patch = read_json(candidate / "proposal.json")
    exposures = read_json(candidate / "exposure.json")
    session = {"schema_version": 1, "kind": "checkpoint_exposure_session",
               "original_sha256": canonical_hash(before["files"]), "policy_sha256": canonical_hash(policy)}
    # Validate the saved disclosure evidence without creating an exposure log.
    checker = object.__new__(CheckpointEditor)
    checker._session = session
    checker.original, checker.inventory, checker.policy = original, before, policy
    checker._validate_records(exposures)
    _check_finite(original, before)
    observed = _verify_saved(original, candidate / "model", before, policy, patch)
    expected = _audit(session["original_sha256"], policy, patch, exposures, observed)
    if read_json(candidate / "audit.json") != expected:
        raise ValueError("saved candidate audit differs from independently verified bytes")
    if inventory_model(original) != before:
        raise ValueError("immutable original changed during verification")
    return expected
