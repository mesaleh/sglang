"""Unit tests for pick_num_kv_splits_ceiling in triton_backend.py.

The picker decides ``max_kv_splits`` at backend init, used by both the
TritonAttnBackend (NVIDIA) and WaveAttnBackend (AMD). Previously SGLang
hardcoded this ceiling at 8 (AMD-overridden to 16), which left H100 at
~48% SM occupancy at c=1 decode for gpt-oss-shaped models.

These tests exercise the picker with synthetic device/model geometries
so they run on CPU in CI.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")

from sglang.srt.layers.attention.triton_backend import (
    _AUTO_KV_SPLITS_FALLBACK,
    _AUTO_KV_SPLITS_MAX,
    _AUTO_KV_SPLITS_MIN,
    pick_num_kv_splits_ceiling,
)


class TestPickNumKvSplitsCeiling(CustomTestCase):
    def test_h100_gpt_oss_128k_saturates_ceiling(self):
        """H100 (132 SMs), gpt-oss-120b (64 attn heads, 8 KV heads, GQA),
        128K advertised context — the production configuration.

        Expected derivation:
        - num_kv_group = 64 / 8 = 8
        - BLOCK_H = min(16, 8) = 8
        - token_grid@c=1 = ceil(64 / 8) = 8
        - ext_multiplier = log2(131072/64) = 11
        - ext_sm = 132 * 11 = 1452
        - target_blocks = 1452 * 1.5 = 2178
        - raw_splits = ceil(2178 / 8) = 273
        - bounded to [8, 128] -> 128 (hits MAX)

        At fully-advertised 128K context the picker saturates the max
        ceiling, matching the top of the useful range in the 2026-05-01
        sweep (splits=64 and splits=128 both plateau near the peak).
        """
        picked = pick_num_kv_splits_ceiling(
            device_core_count=132,
            num_head=64,
            num_kv_head=8,
            max_context_len=131072,
        )
        self.assertEqual(picked, 128)

    def test_h100_gpt_oss_64k_also_saturates_ceiling(self):
        """The shipping production config is c=12 @ 64K context. Even at
        64K the log-scaled ext_sm demand is enormous (ext_multiplier =
        log2(65536/64) = 10), so the picker still saturates to MAX.
        """
        picked = pick_num_kv_splits_ceiling(
            device_core_count=132,
            num_head=64,
            num_kv_head=8,
            max_context_len=65536,
        )
        self.assertEqual(picked, 128)

    def test_h100_mha_shape_still_picks_meaningful_value(self):
        """H100 (132 SMs), hypothetical MHA with 64 heads (no GQA), 128K.

        At MHA (num_kv_group = 1), the picker takes the direct-num_head
        branch: token_grid@c=1 = 64. With 128K context, ext_multiplier
        = 11, target_blocks = 132 * 11 * 1.5 = 2178; raw = ceil(2178/64)
        = 35. Bounded [8, 128] -> 35.

        Models with larger token_grids naturally return smaller ceilings,
        because the grid is already closer to SM-saturating on its own.
        """
        picked = pick_num_kv_splits_ceiling(
            device_core_count=132,
            num_head=64,
            num_kv_head=64,
            max_context_len=131072,
        )
        self.assertEqual(picked, 35)

    def test_short_context_picks_lower_value(self):
        """Models with short advertised contexts should not over-allocate
        split buffers. With max_context_len=2048, ext_multiplier=log2(32)=5,
        target=132*5*1.5=990; raw=ceil(990/8)=124.
        """
        picked = pick_num_kv_splits_ceiling(
            device_core_count=132,
            num_head=64,
            num_kv_head=8,
            max_context_len=2048,
        )
        self.assertEqual(picked, 124)

    def test_unknown_device_core_count_returns_fallback(self):
        """When SM count is 0 or None (old CUDA runtime, exotic device),
        return the historical fallback of 8.
        """
        for sentinel in (None, 0, -1):
            self.assertEqual(
                pick_num_kv_splits_ceiling(
                    device_core_count=sentinel,
                    num_head=64,
                    num_kv_head=8,
                    max_context_len=131072,
                ),
                _AUTO_KV_SPLITS_FALLBACK,
                msg=f"sentinel={sentinel}",
            )

    def test_ceiling_bound(self):
        """Pathological case: many SMs and tiny grid caps at MAX, preventing
        scratch-memory blowup."""
        picked = pick_num_kv_splits_ceiling(
            device_core_count=1000,
            num_head=1,
            num_kv_head=1,
            max_context_len=131072,
        )
        self.assertEqual(picked, _AUTO_KV_SPLITS_MAX)

    def test_floor_bound(self):
        """Tiny device + heavy model should return at least MIN, no matter
        what the raw math gives."""
        picked = pick_num_kv_splits_ceiling(
            device_core_count=1,
            num_head=64,
            num_kv_head=64,
            max_context_len=2048,
        )
        self.assertEqual(picked, _AUTO_KV_SPLITS_MIN)

    def test_zero_num_kv_head_returns_fallback(self):
        """Defensive: never divide by zero if an upstream caller ever
        passes num_kv_head=0."""
        self.assertEqual(
            pick_num_kv_splits_ceiling(
                device_core_count=132,
                num_head=64,
                num_kv_head=0,
                max_context_len=131072,
            ),
            _AUTO_KV_SPLITS_FALLBACK,
        )

    def test_missing_max_context_len_uses_unit_multiplier(self):
        """If caller doesn't know the model's context length (None or 0),
        ext_multiplier falls back to 1.0 — no log-scaling. For H100+gpt-oss
        this gives target=132*1.5=198; raw=ceil(198/8)=25 within [8, 128].
        """
        for ctx in (None, 0, 16):  # ctx ≤ 64 uses ext=1.0
            picked = pick_num_kv_splits_ceiling(
                device_core_count=132,
                num_head=64,
                num_kv_head=8,
                max_context_len=ctx,
            )
            self.assertEqual(picked, 25, msg=f"ctx={ctx}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
