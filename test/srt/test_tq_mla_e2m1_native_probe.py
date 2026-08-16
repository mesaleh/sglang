from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np
import torch


LEVELS = np.asarray(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)
CODES = np.asarray(
    [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7],
    dtype=np.uint8,
)
BOUNDARIES = (LEVELS[:-1] + LEVELS[1:]) / np.float32(2.0)


def _adversarial_scalars() -> np.ndarray:
    values: list[np.float32] = []
    for boundary in BOUNDARIES:
        values.extend(
            [
                np.nextafter(boundary, np.float32(-np.inf), dtype=np.float32),
                boundary,
                np.nextafter(boundary, np.float32(np.inf), dtype=np.float32),
            ]
        )
    min_subnormal = np.nextafter(
        np.float32(0.0), np.float32(1.0), dtype=np.float32
    )
    min_normal = np.float32(np.finfo(np.float32).tiny)
    max_subnormal = np.nextafter(
        min_normal, np.float32(0.0), dtype=np.float32
    )
    max_finite = np.float32(np.finfo(np.float32).max)
    values.extend([np.float32(-0.0), np.float32(0.0)])
    for magnitude in (min_subnormal, max_subnormal, min_normal, max_finite):
        values.extend([-magnitude, magnitude])
    values.extend(LEVELS.tolist())
    bitwise_unique: dict[int, np.float32] = {}
    for value in values:
        scalar = np.float32(value)
        bitwise_unique[int(scalar.view(np.uint32))] = scalar
    return np.asarray(list(bitwise_unique.values()), dtype=np.float32)


def _exponent_stratified_random(count: int = 1_048_576) -> np.ndarray:
    rng = np.random.default_rng(0xA170)
    indices = np.arange(count, dtype=np.uint32)
    exponents = indices % np.uint32(255)
    signs = rng.integers(0, 2, size=count, dtype=np.uint32)
    mantissas = rng.integers(0, 1 << 23, size=count, dtype=np.uint32)
    bits = (signs << 31) | (exponents << 23) | mantissas
    values = bits.view(np.float32)
    assert np.isfinite(values).all()
    return values


def _oracle(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.searchsorted(BOUNDARIES, pairs, side="left")
    codes = CODES[bins]
    packed = codes[:, 0] | (codes[:, 1] << np.uint8(4))
    decoded = LEVELS[bins]
    return packed, decoded


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNativeE2M1Probe(unittest.TestCase):
    def test_exact_encode_decode(self) -> None:
        if torch.cuda.get_device_capability() != (10, 0):
            self.skipTest("GB200 compute capability 10.0 required")
        from sglang.kernels.jit.tq_mla_e2m1_native_probe import (
            _get_module,
            e2m1_native_probe,
        )
        from sglang.kernels.jit.tq_mla_e2m1_native_probe_sass import (
            audit_extension,
        )

        adversarial = _adversarial_scalars()
        zeros = np.zeros_like(adversarial)
        both_nibbles = np.concatenate(
            [
                np.stack([adversarial, zeros], axis=1),
                np.stack([zeros, adversarial], axis=1),
            ],
            axis=0,
        )
        grid_first, grid_second = np.meshgrid(
            adversarial, adversarial, indexing="ij"
        )
        cartesian = np.stack(
            [grid_first.reshape(-1), grid_second.reshape(-1)], axis=1
        )
        random_values = _exponent_stratified_random()
        random_pairs = random_values.reshape(-1, 2)
        pairs = np.ascontiguousarray(
            np.concatenate([both_nibbles, cartesian, random_pairs], axis=0)
        )
        expected_packed, expected_decoded = _oracle(pairs)

        device_pairs = torch.from_numpy(pairs).cuda()
        raw, corrected, decoded = e2m1_native_probe(device_pairs)
        torch.cuda.synchronize()
        raw_cpu = raw.cpu().numpy()
        corrected_cpu = corrected.cpu().numpy()
        decoded_cpu = decoded.cpu().numpy()

        corrected_mismatch = int(np.count_nonzero(corrected_cpu != expected_packed))
        decoded_mismatch = int(
            np.count_nonzero(
                decoded_cpu.view(np.uint32) != expected_decoded.view(np.uint32)
            )
        )
        signed_zero_codes = int(
            np.count_nonzero((corrected_cpu & np.uint8(0x0F)) == 8)
            + np.count_nonzero((corrected_cpu >> np.uint8(4)) == 8)
        )
        # The raw instruction must visibly exercise cases corrected by N0.
        raw_difference = int(np.count_nonzero(raw_cpu != expected_packed))
        extension = str(_get_module().__file__)
        sass_audit = audit_extension(extension)
        metrics = {
            "adversarial_scalars": int(adversarial.size),
            "cartesian_pairs": int(cartesian.shape[0]),
            "random_scalars": int(random_values.size),
            "total_pairs": int(pairs.shape[0]),
            "raw_difference": raw_difference,
            "corrected_mismatch": corrected_mismatch,
            "decoded_mismatch": decoded_mismatch,
            "signed_zero_codes": signed_zero_codes,
            "corrected_sha256": hashlib.sha256(corrected_cpu.tobytes()).hexdigest(),
            "decoded_sha256": hashlib.sha256(decoded_cpu.tobytes()).hexdigest(),
            "extension": extension,
            "sass_audit": sass_audit,
        }
        print("N0_GATE0=" + json.dumps(metrics, sort_keys=True))
        self.assertGreater(raw_difference, 0)
        self.assertEqual(corrected_mismatch, 0)
        self.assertEqual(decoded_mismatch, 0)
        # Code 8 in either nibble is forbidden after canonicalization.  Test
        # each nibble independently rather than treating byte 0x88 specially.
        self.assertEqual(signed_zero_codes, 0)
        self.assertGreaterEqual(random_values.size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
