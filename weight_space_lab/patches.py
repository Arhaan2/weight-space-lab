"""Declared, bounded edits of named floating-point tensors.

Public API::

    state_sha256(state) -> str
    apply_patch(state, patch, *, allowed_tensors, max_changed_fraction)
        -> (copied_state, audit)

A patch has exactly ``schema_version``, ``base_sha256`` and ``operations``.
Version 1 accepts these operation objects::

    {"op": "scale_columns", "tensor": "name", "columns": [0], "factor": 0.9}
    {"op": "remove_direction_from_columns", "tensor": "name", "columns": [0],
     "direction": [1.0, 0.0], "strength": 1.0}

The tensor allowlist and budget are supplied separately from the patch. The
budget counts actual scalar differences in the final state, across all tensors.
This implements transformation mechanics and does not recreate the discovery
process. A separate exact replay module handles released BF16 substitutions.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _checked_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a nonempty mapping")
    checked = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be nonempty strings")
        if not isinstance(tensor, np.ndarray) or tensor.dtype.kind != "f":
            raise ValueError(f"tensor {name!r} must be a floating-point ndarray")
        if tensor.dtype.itemsize > 8:
            raise ValueError(f"tensor {name!r} uses an unsupported floating-point dtype")
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"tensor {name!r} contains nonfinite values")
        checked[name] = tensor
    if not sum(tensor.size for tensor in checked.values()):
        raise ValueError("state must contain at least one scalar")
    return checked


def state_sha256(state: Mapping[str, np.ndarray]) -> str:
    """Hash sorted tensor names, shapes, dtype descriptors and C-order bytes.

    Array layout does not affect the hash. The exact dtype, including byte
    order, is part of the checkpoint identity. Signed zeros retain their byte
    representation in the identity. Unsupported and nonfinite states fail.
    """
    checked = _checked_state(state)
    digest = hashlib.sha256(b"weight-space-state-v1\x00")
    for name in sorted(checked):
        tensor = checked[name]
        metadata = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": tensor.dtype.str},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        raw = np.ascontiguousarray(tensor).tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _columns(value: Any, width: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("columns must be a nonempty list of integers")
    if any(isinstance(col, (bool, np.bool_)) or not isinstance(col, numbers.Integral)
           for col in value):
        raise ValueError("columns must contain integers, not booleans")
    columns = [int(col) for col in value]
    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicates")
    if any(col < 0 or col >= width for col in columns):
        raise ValueError("column index is outside the tensor")
    return columns


def _stable_norm(values: np.ndarray) -> float:
    absolute = np.abs(values)
    maximum = float(np.max(absolute)) if absolute.size else 0.0
    if maximum == 0.0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = maximum * math.sqrt(float(np.sum((absolute / maximum) ** 2)))
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("change norm exceeds the supported numeric range") from error
    if not math.isfinite(result):
        raise ValueError("change norm exceeds the supported numeric range")
    return result


def apply_patch(
    state: Mapping[str, np.ndarray],
    patch: Mapping[str, Any],
    *,
    allowed_tensors: Iterable[str],
    max_changed_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply a version-1 patch to independent copies and return an audit.

    All tensors must be finite real floating arrays of at most 64 bits.
    Operations require matrices. Calculations use float64 and are cast back
    after each operation; casting overflow is rejected. A failed call leaves
    every input array unchanged. Operations execute in the declared order.

    The final-state budget uses numerical inequality; changing only a zero's
    sign changes the hash but does not count as a changed scalar. A scalar
    edited and subsequently restored does not consume the final-state budget.
    """
    checked = _checked_state(state)
    budget = _finite_number(max_changed_fraction, "max_changed_fraction")
    if not 0.0 <= budget <= 1.0:
        raise ValueError("max_changed_fraction must be between 0 and 1")
    if isinstance(allowed_tensors, (str, bytes)):
        raise ValueError("allowed_tensors must be an iterable of tensor names")
    try:
        names = list(allowed_tensors)
    except TypeError as error:
        raise ValueError("allowed_tensors must be an iterable of tensor names") from error
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("allowed_tensors must contain nonempty strings")
    allowlist = set(names)
    if allowlist - checked.keys():
        raise ValueError("allowed_tensors contains a tensor missing from the state")
    if not isinstance(patch, Mapping) or set(patch) != {
        "schema_version", "base_sha256", "operations"
    }:
        raise ValueError("patch must have exactly schema_version, base_sha256 and operations")
    if type(patch["schema_version"]) is not int or patch["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    expected = patch["base_sha256"]
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("base_sha256 must be a lowercase SHA256 digest")
    base_digest = state_sha256(checked)
    if expected != base_digest:
        raise ValueError("patch base_sha256 does not match the starting state")
    operations = patch["operations"]
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    result = {name: tensor.copy() for name, tensor in checked.items()}
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("each operation must be an object")
        op = operation.get("op")
        if op == "scale_columns":
            required = {"op", "tensor", "columns", "factor"}
        elif op == "remove_direction_from_columns":
            required = {"op", "tensor", "columns", "direction", "strength"}
        else:
            raise ValueError("unknown patch operation")
        if set(operation) != required:
            raise ValueError(f"{op} has missing or unknown keys")
        name = operation["tensor"]
        if not isinstance(name, str) or name not in result:
            raise ValueError("operation references a missing tensor")
        if name not in allowlist:
            raise ValueError(f"tensor {name!r} is not allowlisted")
        tensor = result[name]
        if tensor.ndim != 2:
            raise ValueError("column operations require a two-dimensional tensor")
        columns = _columns(operation["columns"], tensor.shape[1])
        work = tensor[:, columns].astype(np.float64)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            try:
                if op == "scale_columns":
                    work *= _finite_number(operation["factor"], "factor")
                else:
                    direction = operation["direction"]
                    if not isinstance(direction, list) or len(direction) != tensor.shape[0]:
                        raise ValueError("direction must be a list matching the tensor's row count")
                    u = np.array([_finite_number(v, "direction component") for v in direction],
                                 dtype=np.float64)
                    strength = _finite_number(operation["strength"], "strength")
                    largest = float(np.max(np.abs(u))) if u.size else 0.0
                    if largest == 0.0:
                        raise ValueError("direction must not be the zero vector")
                    # Rescaling the direction leaves its projection unchanged
                    # and avoids overflow when forming its squared norm.
                    u = u / largest
                    u = u / math.sqrt(float(u @ u))
                    work -= strength * u[:, None] * (u @ work)[None, :]
                cast = work.astype(tensor.dtype)
            except (FloatingPointError, OverflowError) as error:
                raise ValueError("patch operation overflows the supported dtype") from error
        if not np.all(np.isfinite(cast)):
            raise ValueError("patch operation produced nonfinite values")
        tensor[:, columns] = cast
    total = sum(tensor.size for tensor in checked.values())
    changed = 0
    tensor_audits = {}
    norms = []
    for name in sorted(checked):
        before, after = checked[name], result[name]
        count = int(np.count_nonzero(before != after))
        changed += count
        with np.errstate(over="raise", invalid="raise"):
            try:
                difference = after.astype(np.float64) - before.astype(np.float64)
            except FloatingPointError as error:
                raise ValueError("change exceeds the supported numeric range") from error
        norm = _stable_norm(difference)
        norms.append(norm)
        tensor_audits[name] = {
            "shape": list(before.shape), "dtype": before.dtype.str,
            "total_scalars": int(before.size), "changed_scalars": count,
            "changed_fraction": count / before.size if before.size else 0.0,
            "l2_norm": norm,
            "max_abs_change": float(np.max(np.abs(difference))) if difference.size else 0.0,
        }
    fraction = changed / total
    if fraction > budget:
        raise ValueError(f"actual changed fraction {fraction:.12g} exceeds budget {budget:.12g}")
    audit = {
        "schema_version": 1, "base_sha256": base_digest,
        "result_sha256": state_sha256(result), "operation_count": len(operations),
        "total_scalars": int(total), "changed_scalars": changed,
        "changed_fraction": fraction, "max_changed_fraction": budget,
        "l2_norm": _stable_norm(np.array(norms)), "per_tensor": tensor_audits,
        "budget_basis": "final numerical scalar differences across the whole state",
    }
    return result, audit
