import unittest

import numpy as np

from weight_space_lab.checkpoint_numeric import (
    FLOAT_DTYPES, STORAGE_DTYPES, decode, encode, finite, operation,
)


class CheckpointNumericTests(unittest.TestCase):
    def test_all_finite_bf16_encodings_round_trip_exactly(self):
        bits = np.arange(65536, dtype="<u2")
        accepted = bits[finite(bits, "BF16")]
        np.testing.assert_array_equal(encode(decode(accepted, "BF16"), "BF16"), accepted)

    def test_bf16_direct_rounding_avoids_float32_double_rounding(self):
        midpoint = 1.0 + 2.0 ** -8
        values = np.array([midpoint, np.nextafter(midpoint, np.inf),
                           np.nextafter(midpoint, -np.inf), midpoint + 2.0 ** -7])
        np.testing.assert_array_equal(encode(values, "BF16"), [0x3F80, 0x3F81, 0x3F80, 0x3F82])
        # The value just above the midpoint loses its decisive low bits in F32.
        staged = encode(values.astype(np.float32).astype(np.float64), "BF16")
        self.assertEqual(int(staged[1]), 0x3F80)

    def test_signed_zero_subnormal_ties_and_bf16_overflow_boundary(self):
        values = np.array([0.0, -0.0, 2.0 ** -134, -2.0 ** -134,
                           np.nextafter(2.0 ** -134, np.inf), 3 * 2.0 ** -134,
                           255 * 2.0 ** 120])
        expected = [0x0000, 0x8000, 0x0000, 0x8000, 0x0001, 0x0002, 0x7F7F]
        np.testing.assert_array_equal(encode(values, "BF16"), expected)
        boundary = 255.5 * 2.0 ** 120
        self.assertEqual(int(encode(np.array([np.nextafter(boundary, 0)]), "BF16")[0]), 0x7F7F)
        for value in (boundary, -boundary, np.finfo(np.float64).max):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode(np.array([value]), "BF16")

    def test_native_rounding_and_dtype_preservation(self):
        for dtype, precision in (("F16", 11), ("F32", 24)):
            midpoint = 1.0 + 2.0 ** -precision
            value = np.array([midpoint, np.nextafter(midpoint, np.inf),
                              midpoint + 2.0 ** (1 - precision), -0.0])
            result = encode(value, dtype)
            self.assertEqual(result.dtype, STORAGE_DTYPES[dtype])
            np.testing.assert_array_equal(result.astype(np.float64),
                [1.0, 1.0 + 2.0 ** (1 - precision), 1.0 + 2.0 ** (2 - precision), -0.0])
            self.assertTrue(np.signbit(result[-1]))
        values = np.array([np.nextafter(1.0, 2.0), -0.0])
        np.testing.assert_array_equal(encode(values, "F64").view("<u8"), values.view("<u8"))
        self.assertFalse(np.shares_memory(encode(values, "F64"), values))

    def test_nonfinite_and_overflow_rejected_for_each_editable_dtype(self):
        for dtype in FLOAT_DTYPES:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(dtype=dtype, value=value), self.assertRaises(ValueError):
                    encode(np.array([value]), dtype)
            if dtype != "F64":
                with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                    encode(np.array([np.finfo(np.float64).max]), dtype)
        with self.assertRaises(ValueError):
            operation(np.array([[np.finfo(np.float64).max]]), "F64",
                      {"op": "scale_columns", "factor": 2.0})

    def test_operations_share_bounded_arithmetic_and_never_modify_input(self):
        for dtype in FLOAT_DTYPES:
            before = encode(np.array([[2.0, 4.0], [3.0, 6.0]]), dtype)
            original_bytes = before.tobytes()
            scaled = operation(before, dtype, {"op": "scale_columns", "factor": 0.5})
            np.testing.assert_array_equal(decode(scaled, dtype), [[1, 2], [1.5, 3]])
            projected = operation(before, dtype, {"op": "remove_direction_from_columns",
                "direction": [2, 0], "strength": 1})
            np.testing.assert_array_equal(decode(projected, dtype), [[0, 0], [3, 6]])
            self.assertEqual(before.tobytes(), original_bytes)
            self.assertFalse(np.shares_memory(scaled, before))
            with self.assertRaises(ValueError):
                operation(before, dtype, {"op": "remove_direction_from_columns",
                    "direction": [0, 0], "strength": 1})

    def test_integer_buffers_are_exact_and_read_only(self):
        values = np.array([2 ** 63 + 1, 2 ** 64 - 1], dtype="<u8")
        self.assertEqual(decode(values, "U64").tolist(), [2 ** 63 + 1, 2 ** 64 - 1])
        self.assertTrue(np.all(finite(values, "U64")))
        for dtype in set(STORAGE_DTYPES) - FLOAT_DTYPES:
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                encode(np.array([1.0]), dtype)
        with self.assertRaises(ValueError):
            decode(values, "BF16")

    def test_scalar_and_empty_codec_shapes_are_preserved(self):
        for dtype in FLOAT_DTYPES:
            for values in (np.array(-0.0), np.empty((0, 3), dtype=np.float64)):
                with self.subTest(dtype=dtype, shape=values.shape):
                    stored = encode(values, dtype)
                    self.assertEqual(stored.shape, values.shape)
                    self.assertEqual(decode(stored, dtype).shape, values.shape)

    def test_float8_inspection_respects_format_nonfinite_encodings(self):
        bits = np.arange(256, dtype=np.uint8)
        for dtype in ("F8_E4M3", "F8_E5M2"):
            values = decode(bits, dtype)
            np.testing.assert_array_equal(finite(bits, dtype), np.isfinite(values))
            self.assertTrue(np.signbit(values[0x80]))
        e4 = decode(bits, "F8_E4M3")
        self.assertEqual(e4[0x7E], 448.0)
        self.assertEqual(e4[1], 2 ** -9)
        e5 = decode(bits, "F8_E5M2")
        self.assertEqual(e5[0x7B], 57344.0)
        self.assertEqual(e5[1], 2 ** -16)
        self.assertEqual(e5[0x7C], np.inf)
        self.assertEqual(e5[0xFC], -np.inf)


if __name__ == "__main__":
    unittest.main()
