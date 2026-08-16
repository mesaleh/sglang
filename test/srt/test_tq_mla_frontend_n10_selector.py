import unittest

import numpy as np


LEVELS = np.asarray(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)
BOUNDARIES = ((LEVELS[:-1] + LEVELS[1:]) * np.float32(0.5)).astype(np.float32)
POSITIVE_BOUNDARIES = BOUNDARIES[7:]


def _reference_bins(values: np.ndarray) -> np.ndarray:
    return np.searchsorted(BOUNDARIES, values, side="left").astype(np.int32)


def _magnitude_bins(values: np.ndarray, *, correct_negative_ties: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    negative = values < np.float32(0.0)
    magnitude_bits = values.view(np.uint32) & np.uint32(0x7FFFFFFF)
    if correct_negative_ties:
        magnitude_bits = magnitude_bits + negative.astype(np.uint32)
    magnitude = magnitude_bits.view(np.float32)
    rank = np.count_nonzero(
        magnitude[:, None] > POSITIVE_BOUNDARIES[None, :], axis=1
    ).astype(np.int32)
    return np.where(negative, 7 - rank, 7 + rank).astype(np.int32)


class TestTQMLAFrontendN10Selector(unittest.TestCase):
    def test_frozen_native_e2m1_boundaries(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        config = NativeE2M1MLAConfig(device="cpu")
        actual = config.boundaries.numpy().view(np.uint32)
        expected = BOUNDARIES.view(np.uint32)
        np.testing.assert_array_equal(actual, expected)

    def test_negative_midpoint_correction_is_required(self):
        negative_midpoints = BOUNDARIES[:7]
        reference = _reference_bins(negative_midpoints)
        uncorrected = _magnitude_bins(
            negative_midpoints, correct_negative_ties=False
        )
        corrected = _magnitude_bins(
            negative_midpoints, correct_negative_ties=True
        )
        self.assertTrue(np.all(uncorrected == reference + 1))
        np.testing.assert_array_equal(corrected, reference)

    def test_corrected_selector_matches_strict_bucketizer(self):
        directed = [
            np.float32(0.0),
            np.float32(-0.0),
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(0.0), np.float32(-1.0)),
            np.finfo(np.float32).tiny,
            -np.finfo(np.float32).tiny,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ]
        for boundary in BOUNDARIES:
            value = np.float32(boundary)
            directed.extend(
                [
                    np.nextafter(
                        np.nextafter(value, np.float32(-np.inf)),
                        np.float32(-np.inf),
                    ),
                    np.nextafter(value, np.float32(-np.inf)),
                    value,
                    np.nextafter(value, np.float32(np.inf)),
                    np.nextafter(
                        np.nextafter(value, np.float32(np.inf)),
                        np.float32(np.inf),
                    ),
                ]
            )

        dense = []
        offsets = np.arange(-256, 257, dtype=np.int32)
        for boundary in BOUNDARIES:
            center = np.asarray([boundary], dtype=np.float32).view(np.uint32)[0]
            ordered = np.where(
                offsets < 0,
                center - (-offsets).astype(np.uint32),
                center + offsets.astype(np.uint32),
            )
            dense.append(ordered.astype(np.uint32).view(np.float32))

        rng = np.random.default_rng(20260816)
        random_values = rng.integers(
            0, np.iinfo(np.uint32).max, size=2_000_000, dtype=np.uint32
        ).view(np.float32)
        random_values = random_values[np.isfinite(random_values)]
        values = np.concatenate(
            [np.asarray(directed, dtype=np.float32), *dense, random_values]
        )

        reference = _reference_bins(values)
        corrected = _magnitude_bins(values, correct_negative_ties=True)
        np.testing.assert_array_equal(corrected, reference)


if __name__ == "__main__":
    unittest.main()
