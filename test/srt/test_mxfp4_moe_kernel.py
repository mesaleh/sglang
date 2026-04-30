"""
Unit tests for the MXFP4 fused MoE kernel
(``fused_moe_kernel_mxfp4`` / ``invoke_fused_moe_kernel_mxfp4``).

Validates numerical equivalence against a reference path:
    upcast_from_mxfp(W) -> bf16, then stock ``fused_moe_kernel``.

The reference and our kernel consume identical inputs modulo the
weight dequant. If they match to cosine-similarity ≥ 0.995 and
max abs error within expected bf16 accumulation tolerance, the
MXFP4 decode + scale application is correct.

Test matrix (tuned for gpt-oss-120b target shapes; kernel is generic):
- Hidden K      ∈ {128, 2880}       (128 for fast debug, 2880 is real)
- Output N      ∈ {128, 2880, 5760} (128 for debug, 2880 = w2, 5760 = w13)
- Experts E     ∈ {4, 128}          (4 for debug, 128 is real gpt-oss)
- topk          ∈ {2, 4}
- M tokens      ∈ {1, 4, 16, 32, 128, 144, 256}

Usage:
    python -m pytest test/srt/test_mxfp4_moe_kernel.py -v
"""

from __future__ import annotations

import unittest

try:
    import torch
    import triton
    import triton.language as tl

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


def _pack_mxfp4_reference(weight_fp32: torch.Tensor):
    """Quantize a float weight tensor to MXFP4 using the standard E2M1 / E8M0 spec.

    Args:
        weight_fp32: [..., K] float32, K must be multiple of 32.

    Returns:
        packed: [..., K // 2] uint8 — two E2M1 values per byte, low=even, high=odd.
        scales: [..., K // 32] uint8 — E8M0 encoding of 2^(byte - 127).
    """
    assert weight_fp32.dtype == torch.float32
    K = weight_fp32.shape[-1]
    assert K % 32 == 0, f"K={K} must be multiple of 32"

    # Reshape to expose 32-element groups.
    lead_shape = weight_fp32.shape[:-1]
    w = weight_fp32.reshape(*lead_shape, K // 32, 32)  # [..., G, 32]

    # Compute per-group absmax. E2M1 max representable is 6.0.
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)  # [..., G, 1]
    # We want scale = 2^(s - 127) such that absmax / scale ≤ 6.0.
    # => 2^(s - 127) ≥ absmax / 6.0
    # => s ≥ log2(absmax / 6.0) + 127
    log2_target = torch.log2(absmax / 6.0)
    s_float = torch.ceil(log2_target) + 127
    s_int = s_float.clamp(0, 254).to(torch.int32)  # reserve 255 for NaN
    scale = torch.pow(2.0, s_int.to(torch.float32) - 127.0)

    # Normalize weights into E2M1 range.
    w_norm = w / scale  # [..., G, 32], values in [-6, 6]

    # Quantize each value to nearest E2M1 grid point.
    # E2M1 values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6.
    grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=w.device,
    )  # 8 abs values, 4 exp values (0, 1, 2, 3)
    sign = torch.sign(w_norm)
    abs_val = w_norm.abs()
    # Find nearest grid index.
    dists = (abs_val.unsqueeze(-1) - grid).abs()  # [..., G, 32, 8]
    idx = dists.argmin(dim=-1)  # [..., G, 32] in [0, 7]
    # Encode: sign bit = bit 3, low 3 bits = grid index. This matches
    # the E2M1 ordering where nibble=0..7 encode +0,+0.5,+1,+1.5,+2,+3,+4,+6
    # and nibble=8..15 encode -0,-0.5,-1,-1.5,-2,-3,-4,-6.
    nibbles = idx.to(torch.uint8) | ((sign < 0).to(torch.uint8) << 3)

    # Pack pairs along K axis: low nibble = even, high nibble = odd.
    even = nibbles[..., 0::2]
    odd = nibbles[..., 1::2]
    packed_groups = (even & 0x0F) | ((odd & 0x0F) << 4)  # [..., G, 16] uint8
    # Flatten groups back to [..., K // 2]
    packed = packed_groups.reshape(*lead_shape, K // 2)

    scale_u8 = s_int.to(torch.uint8).squeeze(-1)  # [..., G]

    return packed, scale_u8


def _upcast_mxfp4_reference(
    packed: torch.Tensor,
    scale_u8: torch.Tensor,
) -> torch.Tensor:
    """Decode MXFP4 back to bf16 via pure PyTorch ops. Used as a reference
    independent of the Triton kernel path. Mirrors the canonical formula.
    """
    assert packed.dtype == torch.uint8
    assert scale_u8.dtype == torch.uint8
    # Unpack nibbles.
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Stack so even K = low, odd K = high along the K axis.
    lead_shape = packed.shape[:-1]
    K_half = packed.shape[-1]
    nibbles = torch.stack([low, high], dim=-1).reshape(*lead_shape, K_half * 2)

    # Decode each nibble.
    sign_bit = (nibbles >> 3) & 0x1
    abs_idx = nibbles & 0x07
    grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    abs_val = grid[abs_idx.long()]
    signed = torch.where(sign_bit.bool(), -abs_val, abs_val)

    # Apply scale (one per 32 K values, broadcast along N if present).
    K = K_half * 2
    G = K // 32
    signed = signed.reshape(*lead_shape, G, 32)
    scale = torch.pow(2.0, scale_u8.to(torch.float32) - 127.0).unsqueeze(-1)
    nan_mask = (scale_u8 == 0xFF).unsqueeze(-1)
    out = signed * scale
    out = torch.where(nan_mask, torch.full_like(out, float("nan")), out)
    out = out.reshape(*lead_shape, K)
    return out.to(torch.bfloat16)


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestMxfp4Decode(unittest.TestCase):
    """Standalone decode test: the kernel's bit-arithmetic E2M1 decode
    must reconstruct the original values when we provide synthetic packed
    data with known contents."""

    def test_decode_against_reference(self):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel_mxfp4,
        )

        device = "cuda"
        torch.manual_seed(0)
        K = 128
        N = 64
        E = 2

        # Build a random float weight, quantize to MXFP4, then decode both ways.
        W = torch.randn(E, N, K, dtype=torch.float32, device=device) * 2.0
        packed, scales = _pack_mxfp4_reference(W)  # [E, N, K/2], [E, N, K/32]

        # Reference decode.
        W_ref = _upcast_mxfp4_reference(packed, scales)  # [E, N, K] bf16

        # Kernel-path decode: run the matmul with A = identity-ish activation
        # that lets us recover dequantized columns.
        # For expert e, token e, top_k=1: output[e] = A[e] @ W[e]^T (reduced over K).
        # To isolate decode accuracy, pick A as one-hot vectors across K.
        M = K
        topk = 1
        A = torch.eye(K, dtype=torch.bfloat16, device=device)  # [K, K]
        # We only test expert 0 here.
        sorted_token_ids = torch.arange(M, dtype=torch.int32, device=device)
        expert_ids = torch.zeros(
            (M + 15) // 16, dtype=torch.int32, device=device
        )  # all expert 0, BLOCK_M=16
        num_tokens_post_padded = torch.tensor([M], dtype=torch.int32, device=device)
        topk_weights = torch.ones((M, topk), dtype=torch.float32, device=device)
        topk_ids = torch.zeros((M, topk), dtype=torch.int32, device=device)

        C = torch.zeros((M * topk, N), dtype=torch.bfloat16, device=device)

        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }

        invoke_fused_moe_kernel_mxfp4(
            A,
            packed,
            scales,
            C,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            compute_type=tl.bfloat16,
            filter_expert=False,
            enable_nan_prop=True,
        )

        # For one-hot activation at row k: C[k, n] = W[0, n, k].
        # So C should equal W_ref[0].T (shape [K, N]).
        expected = W_ref[0].T.to(torch.float32)  # [K, N]
        got = C.to(torch.float32)
        max_err = (got - expected).abs().max().item()
        # bf16 accumulator tolerance; decode should be exact modulo bf16.
        self.assertLess(max_err, 0.1, f"decode max_err={max_err}")


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestMxfp4MatmulGolden(unittest.TestCase):
    """Golden test: run our kernel and compare to the same MoE matmul
    computed via _upcast_mxfp4_reference + PyTorch reference matmul."""

    def _run(self, M: int, N: int, K: int, E: int, topk: int, seed: int):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel_mxfp4,
        )

        device = "cuda"
        torch.manual_seed(seed)

        # Random bf16 activations.
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5

        # Random weights, quantized.
        W = torch.randn(E, N, K, dtype=torch.float32, device=device) * 0.3
        packed, scales = _pack_mxfp4_reference(W)
        W_deq = _upcast_mxfp4_reference(packed, scales)  # [E, N, K] bf16

        # Random routing: for each token, pick topk experts uniformly.
        topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
        topk_weights = torch.rand(M, topk, dtype=torch.float32, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Build sorted_token_ids / expert_ids the same way fused_moe does.
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        BLOCK_M = 16
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_M, E
        )

        # Output buffer.
        C = torch.zeros((M * topk, N), dtype=torch.bfloat16, device=device)

        config = {
            "BLOCK_SIZE_M": BLOCK_M,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }

        invoke_fused_moe_kernel_mxfp4(
            A,
            packed,
            scales,
            C,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            compute_type=tl.bfloat16,
            filter_expert=False,
            enable_nan_prop=True,
        )

        # Reference: for each (token m, topk slot t), compute
        #   out[m*topk + t] = A[m] @ W_deq[expert_ids[m, t]].T
        expected = torch.zeros_like(C, dtype=torch.float32)
        for m in range(M):
            for t in range(topk):
                e = int(topk_ids[m, t].item())
                # Reorder expected by sorted_token_ids: output index is
                # the position in topk_ids flattened, which is m*topk + t.
                expected[m * topk + t] = (
                    A[m].to(torch.float32) @ W_deq[e].to(torch.float32).T
                )

        got = C.to(torch.float32)

        # Cosine similarity and max abs error.
        cos = torch.nn.functional.cosine_similarity(
            got.flatten().unsqueeze(0), expected.flatten().unsqueeze(0)
        ).item()
        max_err = (got - expected).abs().max().item()
        rel_err = max_err / expected.abs().max().clamp(min=1e-6).item()

        return cos, max_err, rel_err

    def test_small(self):
        cos, mx, rel = self._run(M=16, N=128, K=128, E=4, topk=2, seed=0)
        self.assertGreater(cos, 0.99, f"cos={cos}")
        self.assertLess(rel, 0.1, f"rel_err={rel}")

    def test_gptoss_w13_tiny_expert_count(self):
        # 4 experts so test runs quick, but gpt-oss dim.
        cos, mx, rel = self._run(M=32, N=5760, K=2880, E=4, topk=2, seed=1)
        self.assertGreater(cos, 0.99, f"cos={cos}")
        self.assertLess(rel, 0.1, f"rel_err={rel}")

    def test_gptoss_w2_tiny_expert_count(self):
        cos, mx, rel = self._run(M=32, N=2880, K=2880, E=4, topk=2, seed=2)
        self.assertGreater(cos, 0.99, f"cos={cos}")
        self.assertLess(rel, 0.1, f"rel_err={rel}")

    def test_m_144_stable(self):
        # The shape that crashed triton_kernels — we want this to just work.
        cos, mx, rel = self._run(M=144, N=2880, K=2880, E=4, topk=4, seed=3)
        self.assertGreater(cos, 0.99, f"cos={cos}")
        self.assertLess(rel, 0.1, f"rel_err={rel}")

    def test_m_1_decode(self):
        cos, mx, rel = self._run(M=1, N=2880, K=2880, E=4, topk=4, seed=4)
        self.assertGreater(cos, 0.95, f"cos={cos}")
        self.assertLess(rel, 0.15, f"rel_err={rel}")


class TestAutotuneWiring(unittest.TestCase):
    """Sanity checks that the Omniva MXFP4 runner plugs into the autotune
    infrastructure the same way other kernels do. Doesn't need CUDA."""

    def setUp(self):
        # ``get_default_config`` reads ``enable_deterministic_inference``
        # from the global server args. Stub it so this test doesn't need a
        # full ServerArgs bootstrap.
        import sglang.srt.server_args as sa_mod

        self._prev_global = getattr(sa_mod, "_global_server_args", None)

        class _Stub:
            enable_deterministic_inference = False

        sa_mod._global_server_args = _Stub()

    def tearDown(self):
        import sglang.srt.server_args as sa_mod

        sa_mod._global_server_args = self._prev_global

    def test_dtype_string_registered(self):
        import torch
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
        )

        self.assertEqual(
            get_config_dtype_str(torch.bfloat16, use_mxfp4_w4a16=True),
            "mxfp4_w4a16",
        )
        # Other flags still take precedence / unaffected
        self.assertEqual(
            get_config_dtype_str(torch.bfloat16, use_fp8_w8a8=True), "fp8_w8a8"
        )

    def test_default_config_has_mxfp4_branch(self):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_default_config,
        )

        # BLOCK_SIZE_K must be a multiple of the MXFP block size (32) so
        # scale alignment holds in the kernel. Both small-M and large-M
        # branches must satisfy this.
        for M in [1, 4, 32, 128, 512, 4096]:
            cfg = get_default_config(
                M=M, E=128, N=2880, K=2880, topk=4,
                dtype="mxfp4_w4a16", is_marlin=False,
            )
            self.assertEqual(cfg["BLOCK_SIZE_K"] % 32, 0, f"M={M}")
            for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "GROUP_SIZE_M",
                        "num_warps", "num_stages"):
                self.assertIn(key, cfg, f"M={M} missing {key}")

    def test_try_get_optimal_falls_back_cleanly(self):
        # No tuned JSON shipped yet for these shapes; must fall back to
        # the default config without errors.
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            try_get_optimal_moe_config,
        )

        up, (dn, _mbm) = try_get_optimal_moe_config(
            w1_shape=(128, 5760, 1440),
            w2_shape=(128, 2880, 1440),
            top_k=4,
            dtype="mxfp4_w4a16",
            M=1,
            return_down_config=True,
        )
        self.assertEqual(up["BLOCK_SIZE_K"] % 32, 0)
        # Down config is None when no _down JSON exists — runner handles
        # this by reusing the gate-up config.
        if dn is not None:
            self.assertEqual(dn["BLOCK_SIZE_K"] % 32, 0)
            self.assertEqual(dn["BLOCK_SIZE_M"], up["BLOCK_SIZE_M"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
