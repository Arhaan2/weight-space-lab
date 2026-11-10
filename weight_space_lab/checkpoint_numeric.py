"""Dtype codecs and arithmetic for bounded Safetensors selections only.

These functions never open or materialize a checkpoint. Their caller limits
the selected coordinates and owns all file I/O. BF16 is stored as uint16 bits;
integer and Boolean buffers retain their exact native values when decoded.
The two FP8 formats can be inspected and checked for finiteness but not edited.
"""

from __future__ import annotations

import numpy as np

from .patches import _operate_float64


STORAGE_DTYPES = {name: np.dtype(dtype) for name, dtype in {
    "BOOL": "?", "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
    "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8",
    "BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8",
    "F8_E4M3": "u1", "F8_E5M2": "u1",
}.items()}

FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})

PRECISION = {
    "arithmetic": "CPU NumPy IEEE-754 binary64 for each selected column operation",
    "projection": "direction rescaled by max abs, normalized by binary64 dot product; binary64 matrix product",
    "rounding": "round to nearest, ties to even, to original stored dtype after every operation",
    "bf16_rounding": "direct binary64 to BF16; no intermediate float32 rounding",
    "underflow": "gradual underflow; round to subnormal or signed zero; no explicit flush to zero",
    "overflow": "reject nonfinite binary64 intermediates and nonfinite stored results",
    "signed_zero": "preserve sign through storage rounding; arithmetic may change it; bit and numerical counts differ",
    "composition": "operations run in declared order and each reads the preceding rounded stored result",
    "reproducibility": "NumPy/BLAS reductions may differ across runtime builds; stored hashes identify actual results",
    "editable_dtypes": sorted(FLOAT_DTYPES),
    "nonfloating_buffers": "read-only; preserve original bytes and exact integer inspection values",
}


def _checked_raw(raw, dtype):
    if dtype not in STORAGE_DTYPES:
        raise ValueError("unsupported stored tensor dtype")
    if not isinstance(raw, np.ndarray) or raw.dtype != STORAGE_DTYPES[dtype]:
        raise ValueError("selection storage dtype differs from the tensor dtype")
    return raw


def finite(raw, dtype):
    """Return a Boolean mask without converting integer or BF16 buffers."""
    raw = _checked_raw(raw, dtype)
    if dtype == "BF16":
        return (raw & np.uint16(0x7F80)) != np.uint16(0x7F80)
    if dtype == "F8_E4M3":
        # E4M3FN has finite values at exponent 15 except the all-one mantissa.
        return (raw & np.uint8(0x7F)) != np.uint8(0x7F)
    if dtype == "F8_E5M2":
        return (raw & np.uint8(0x7C)) != np.uint8(0x7C)
    if dtype in FLOAT_DTYPES:
        return np.isfinite(raw)
    return np.ones(raw.shape, dtype=np.bool_)


def decode(raw, dtype):
    """Decode a bounded selection, preserving exact integral buffer values.

    BF16/FP8 are promoted exactly to float64. F16/F32/F64 and integral arrays
    are returned as views; callers must not mutate source-backed selections.
    Nonfinite floating encodings are represented faithfully, not silently
    replaced. Call ``finite`` before statistics or arithmetic.
    """
    raw = _checked_raw(raw, dtype)
    if dtype == "BF16":
        bits = np.asarray(raw.astype("<u4") << np.uint32(16), dtype="<u4")
        return np.asarray(bits.view("<f4"), dtype=np.float64)
    if dtype in {"F8_E4M3", "F8_E5M2"}:
        mantissa_bits, bias = (3, 7) if dtype == "F8_E4M3" else (2, 15)
        magnitude = raw & np.uint8(0x7F)
        exponent = (magnitude >> mantissa_bits).astype(np.int32)
        fraction = (magnitude & np.uint8((1 << mantissa_bits) - 1)).astype(np.float64)
        mantissa = np.where(exponent == 0, fraction, fraction + (1 << mantissa_bits))
        result = np.ldexp(mantissa, np.maximum(exponent, 1) - bias - mantissa_bits)
        result = np.copysign(result, np.where(raw & np.uint8(0x80), -1.0, 1.0))
        if dtype == "F8_E4M3":
            return np.where(magnitude == 0x7F, np.nan, result)
        result = np.where(magnitude == 0x7C, np.copysign(np.inf, result), result)
        return np.where(magnitude > 0x7C, np.nan, result)
    return raw.view()


def _encode_bf16(values):
    """Round binary64 directly to BF16 with exact power-of-two scaling.

    Normal significands need eight significant binary digits. Subnormals
    use units of 2**-133. Scaling is exact for the relevant ranges, so rint
    sees the original binary64 midpoint decision without float32 rounding.
    """
    absolute = np.abs(values)
    normal = absolute >= np.ldexp(1.0, -126)
    mantissa, exponent = np.frexp(absolute)
    rounded = np.rint(np.ldexp(mantissa, 8)).astype(np.int32)
    normal_bits = ((exponent.astype(np.int32) + 126) << 7) + rounded - 128
    if np.any(normal & (normal_bits >= 0x7F80)):
        raise ValueError("patch operation overflows the stored BF16 dtype")
    # np.where evaluates both branches: mask first to avoid scaling huge
    # normal inputs up by 2**133 merely to discard their intermediate values.
    subnormal = np.where(normal, 0.0, absolute)
    subnormal_bits = np.rint(np.ldexp(subnormal, 133)).astype(np.int32)
    magnitude = np.where(normal, normal_bits, subnormal_bits).astype("<u2")
    sign = np.where(np.signbit(values), np.uint16(0x8000), np.uint16(0))
    return np.asarray(magnitude | sign, dtype="<u2")


def encode(values, dtype):
    """Round finite binary64 values to an editable stored dtype.

    Native NumPy F16/F32 casts use nearest-even, including subnormals. F64
    retains binary64 values. BF16 rounds directly without an F32 staging cast.
    Every call returns independent storage and rejects nonfloating edits.
    """
    if dtype not in FLOAT_DTYPES:
        raise ValueError("tensor dtype is read-only for arithmetic operations")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cannot store nonfinite values")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            result = (_encode_bf16(values) if dtype == "BF16"
                      else values.astype(STORAGE_DTYPES[dtype], copy=True))
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("patch operation overflows the stored dtype") from error
    if not np.all(finite(result, dtype)):
        raise ValueError("patch operation produced nonfinite stored values")
    return result


def operation(raw, dtype, op):
    """Apply one declared operation to an already bounded column selection.

    ``raw`` is a matrix containing every row of the requested columns. Tensor
    names, operation schemas and coordinate/policy bounds belong to the
    adapter; this helper validates arithmetic and rounds once per operation.
    """
    if dtype not in FLOAT_DTYPES:
        raise ValueError("tensor dtype is read-only for arithmetic operations")
    raw = _checked_raw(raw, dtype)
    if not np.all(finite(raw, dtype)):
        raise ValueError("selection contains nonfinite values")
    return encode(_operate_float64(decode(raw, dtype), op), dtype)
