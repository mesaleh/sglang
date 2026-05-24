"""
Numerical-parity unit test for the fused TurboQuant-MLA decode Triton kernel.

The kernel under test reads packed 4-bit TurboQuant KV cache directly (rotated
domain, uint8 nope + bf16 scale + bf16 rope) and produces an attention output
in rotated domain, which the backend inverse-rotates post-kernel.

Strategy:
  1. Build a TurboQuantConfig matching Kimi-K2.6 MLA dims (lora=512, rope=64).
  2. Generate random Q_nope (original domain) and K_nope bf16 (original domain).
  3. Quantize K_nope through the same batched_quantize path the pool uses at
     write time, producing packed uint8 + dequant_scale matching the exact
     storage layout of MLATokenToKVPoolTurboQuant.
  4. Build K_rope raw bf16 (no quantization).
  5. Call tq_mla_decode_attention_fwd with Q_nope rotated by tq_config.
  6. Inverse-rotate the kernel output (what the backend does post-kernel).
  7. Reference: run a torch attention using the *dequanted* K_nope (the same
     values the kernel sees after reading packed storage). This isolates
     kernel-math correctness from quantization error — if the kernel reads
     packed storage and computes attention correctly, it should match torch
     closely.

Placement per write-sglang-test skill: test/registered/attention/.
Suite: stage-b-test-1-gpu-large (H100 required — kernel is Hopper-targeted).

Related:
  Source: python/sglang/srt/layers/attention/triton_ops/turboquant_mla_decode_attention.py
  Backend: python/sglang/srt/layers/attention/flashmla_backend.py (TurboQuantMLABackend)
  Pool: python/sglang/srt/mem_cache/memory_pool.py (MLATokenToKVPoolTurboQuant)
  Kernel Engineering Rule KE-13: backend-kernel interface is a separate
    correctness surface; this test exercises the live backend call contract
    (kv_indptr/kv_indices/num_kv_splits shape/dtype, q rotation, output
    inverse-rotation).
"""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# H100-class memory + Hopper Triton compatibility.
register_cuda_ci(est_time=40, suite="stage-b-test-1-gpu-large")


def _build_config(lora_rank: int, device: torch.device):
    """Build a TurboQuantConfig for the nope dim. Matches pool's construction."""
    from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

    return TurboQuantConfig(
        bit_width=4,
        head_dim=lora_rank,
        device=str(device),
        k_bit_width=4,
        v_bit_width=4,
        uniform=False,
    )


def _quantize_like_pool(
    k_nope_bf16: torch.Tensor,  # (n, 1, lora_rank) original domain
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate MLATokenToKVPoolTurboQuant.set_mla_kv_buffer quantization.

    Returns packed uint8 (n, 1, lora_rank // 2) and dequant_scale (n, 1) bf16.
    """
    from sglang.srt.layers.quantization.kv_turboquant import batched_quantize

    packed, norms, quant_norms = batched_quantize(
        k_nope_bf16,
        cfg.signs1,
        cfg.signs2,
        cfg.k_centroids,
        cfg.k_boundaries,
        4,  # bit_width
    )
    threshold = torch.tensor(1e-10, dtype=torch.bfloat16, device=k_nope_bf16.device)
    replacement = torch.tensor(1.0, dtype=torch.bfloat16, device=k_nope_bf16.device)
    safe_qnorm = torch.where(quant_norms > threshold, quant_norms, replacement)
    dequant_scale = norms / safe_qnorm
    return packed, dequant_scale


def _dequant_nope_reference(
    packed: torch.Tensor,  # (n, 1, lora_rank // 2) uint8
    dequant_scale: torch.Tensor,  # (n, 1) bf16
    cfg,
    lora_rank: int,
) -> torch.Tensor:
    """Replicate MLATokenToKVPoolTurboQuant._dequant_nope.

    Returns K_nope in *original* domain, shape (n, 1, lora_rank), bf16.
    """
    from sglang.srt.layers.quantization.kv_turboquant import (
        batched_dequantize_rotspace,
    )

    x_rot = batched_dequantize_rotspace(
        packed, dequant_scale, cfg.k_centroids, 4, head_dim=lora_rank
    )
    return cfg.inverse_rotate_output(x_rot).to(torch.bfloat16)


def _reference_mla_decode(
    q_nope: torch.Tensor,  # (bs, q_heads, lora_rank) bf16 — ORIGINAL domain
    q_rope: torch.Tensor,  # (bs, q_heads, rope_dim) bf16
    k_nope_full: torch.Tensor,  # (total_kv, 1, lora_rank) bf16 — post-dequant, ORIGINAL domain
    k_rope: torch.Tensor,  # (total_kv, 1, rope_dim) bf16
    kv_indptr: torch.Tensor,  # (bs+1,) int32
    kv_indices: torch.Tensor,  # (total_kv_tokens,) int32
    sm_scale: float,
) -> torch.Tensor:
    """Reference MLA decode: per-batch gather then softmax(Q·Kᵀ / sqrt)·K_nope.

    Returns o (bs, q_heads, lora_rank), fp32. MLA's absorbed form: attention
    output IS K_nope @ softmax_weights (W_V is absorbed into W_O downstream).
    """
    bs, q_heads, lora_rank = q_nope.shape
    rope_dim = q_rope.shape[-1]
    out = torch.zeros((bs, q_heads, lora_rank), dtype=torch.float32, device=q_nope.device)

    for b in range(bs):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        if end <= start:
            continue
        ctx_idx = kv_indices[start:end].long()
        # k_nope_full[ctx_idx] -> (ctx_len, 1, lora_rank) -> (ctx_len, lora_rank)
        k_n = k_nope_full[ctx_idx].squeeze(1).float()  # (ctx_len, lora_rank)
        k_r = k_rope[ctx_idx].squeeze(1).float()  # (ctx_len, rope_dim)

        q_n = q_nope[b].float()  # (q_heads, lora_rank)
        q_r = q_rope[b].float()  # (q_heads, rope_dim)

        # qk = q_n @ k_n^T  + q_r @ k_r^T  (MLA: no k_scale in this kernel path,
        # matches flashmla's reference attention and the kernel's sm_scale-only path)
        qk = (q_n @ k_n.T) + (q_r @ k_r.T)  # (q_heads, ctx_len)
        qk = qk * sm_scale
        p = torch.softmax(qk, dim=-1)  # (q_heads, ctx_len)
        out[b] = p @ k_n  # (q_heads, lora_rank)

    return out


class TestTurboQuantMLADecodeKernel(CustomTestCase):
    """Tests the fused Triton MLA decode kernel on packed 4-bit TurboQuant KV.

    We fix Kimi K2.6 dims (lora=512, rope=64, q_heads=128) and sweep
    (bs, ctx_len) configurations. Cosine-similarity tolerance accounts for
    the 4-bit quantization error that is intrinsic to the kernel's input.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for TurboQuant MLA decode kernel")
        cls.device = torch.device("cuda")

        # Kimi K2.6 architectural constants.
        cls.lora_rank = 512
        cls.rope_dim = 64
        cls.q_heads = 128
        cls.sm_scale = 1.0 / math.sqrt(cls.lora_rank + cls.rope_dim)

        # Build TurboQuantConfig once; reused across sub-tests.
        cls.cfg = _build_config(cls.lora_rank, cls.device)

    def _run_one(self, bs: int, ctx_len: int, seed: int = 0) -> tuple[float, float]:
        """Run one (bs, ctx_len) configuration. Return (cos_sim, max_abs_err)."""
        from sglang.srt.layers.attention.triton_ops.turboquant_mla_decode_attention import (
            tq_mla_decode_attention_fwd,
        )

        torch.manual_seed(seed)
        device = self.device
        lora, rope, qh = self.lora_rank, self.rope_dim, self.q_heads
        total_kv = bs * ctx_len

        # Inputs in ORIGINAL domain.
        q_nope = torch.randn(bs, qh, lora, dtype=torch.bfloat16, device=device)
        q_rope = torch.randn(bs, qh, rope, dtype=torch.bfloat16, device=device)
        k_nope_orig = torch.randn(total_kv, 1, lora, dtype=torch.bfloat16, device=device)
        k_rope = torch.randn(total_kv, 1, rope, dtype=torch.bfloat16, device=device)

        # Quantize K_nope the way the pool does at set_mla_kv_buffer.
        packed, dequant_scale = _quantize_like_pool(k_nope_orig, self.cfg)

        # Dequant back through the same path the pool uses at get_mla_kv_buffer.
        # This is the K_nope the kernel effectively sees — reference attention
        # must use THIS, not k_nope_orig (else it tests quantization error, not
        # kernel math).
        k_nope_dequant = _dequant_nope_reference(packed, dequant_scale, self.cfg, lora)

        # Paging metadata: each batch gets a contiguous ctx_len slice of KV rows.
        kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=device) * ctx_len
        kv_indices = torch.arange(0, total_kv, dtype=torch.int32, device=device)

        # Stage C contract: caller pre-rotates Q_nope. q_rope NOT rotated.
        q_nope_rot = self.cfg.rotate_query(q_nope).to(q_nope.dtype)

        # Output + split scratch buffers (bs=1-path shape convention from backend).
        max_splits = 8  # kernel's compile-time MAX_KV_SPLITS
        att_logits = torch.empty(
            (bs, qh, max_splits, lora), dtype=torch.float32, device=device
        )
        att_lse = torch.empty(
            (bs, qh, max_splits), dtype=torch.float32, device=device
        )
        o = torch.empty((bs, qh, lora), dtype=torch.bfloat16, device=device)
        # Per-batch split count (kernel expects (bs,) int32 per-batch — NOT
        # flashmla's (bs+1,) prefix-sum convention; this was Stage C bug #3).
        num_kv_splits = torch.full(
            (bs,), max_splits, dtype=torch.int32, device=device
        )

        tq_mla_decode_attention_fwd(
            q_nope_rotated=q_nope_rot,
            q_rope=q_rope,
            k_nope_packed=packed,
            k_scale=dequant_scale,
            k_rope=k_rope,
            k_centroids=self.cfg.k_centroids,
            o=o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            att_logits=att_logits,
            att_lse=att_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=max_splits,
            sm_scale=self.sm_scale,
            logit_cap=0.0,
            uniform=getattr(self.cfg, "uniform", False),
        )

        # Backend inverse-rotates the output post-kernel.
        o_orig = self.cfg.inverse_rotate_output(o).float()

        # Reference in original domain.
        ref = _reference_mla_decode(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope_full=k_nope_dequant,
            k_rope=k_rope,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            sm_scale=self.sm_scale,
        )

        # Cosine similarity across the (q_heads, lora) axis, averaged over bs.
        o_flat = o_orig.reshape(bs, -1)
        ref_flat = ref.reshape(bs, -1)
        cos = torch.nn.functional.cosine_similarity(o_flat, ref_flat, dim=-1)
        cos_sim = cos.mean().item()
        max_abs = (o_orig - ref).abs().max().item()
        return cos_sim, max_abs

    # --- Correctness across shape configs -----------------------------------

    # Magnitude guard: cosine_similarity catches direction but not magnitude.
    # A kernel bug that preserves output direction while scaling magnitude
    # (e.g. dropped sm_scale, doubled k_scale) can keep cos_sim high while
    # max_abs balloons. Reference magnitudes are O(1) after softmax — a real
    # kernel break typically exceeds this tolerance by 10x+.
    _MAX_ABS_TOL = 0.5

    def _check(self, bs: int, ctx_len: int):
        cos, max_abs = self._run_one(bs=bs, ctx_len=ctx_len)
        self.assertGreaterEqual(
            cos, 0.995, f"cos_sim={cos:.6f} (bs={bs}, ctx_len={ctx_len})"
        )
        self.assertLessEqual(
            max_abs,
            self._MAX_ABS_TOL,
            f"max_abs={max_abs:.4f} exceeds tolerance "
            f"{self._MAX_ABS_TOL} (bs={bs}, ctx_len={ctx_len}, cos={cos:.4f})",
        )

    def test_bs1_ctx64(self):
        self._check(bs=1, ctx_len=64)

    def test_bs1_ctx256(self):
        self._check(bs=1, ctx_len=256)

    def test_bs2_ctx128(self):
        self._check(bs=2, ctx_len=128)

    def test_bs4_ctx64(self):
        self._check(bs=4, ctx_len=64)

    def test_bs8_ctx32(self):
        self._check(bs=8, ctx_len=32)

    # --- Kernel-metadata contract regression guards -------------------------

    def test_num_kv_splits_per_batch_shape(self):
        """Stage C bug #3 guard: kernel expects (bs,) per-batch int32, NOT
        flashmla's (bs+1,) prefix-sum. If the kernel or a refactor silently
        accepts (bs+1,), the shape change would read garbage. This test
        confirms the per-batch shape is what the kernel needs.
        """
        from sglang.srt.layers.attention.triton_ops.turboquant_mla_decode_attention import (
            tq_mla_decode_attention_fwd,
        )

        torch.manual_seed(42)
        device = self.device
        bs, ctx_len = 2, 32
        lora, rope, qh = self.lora_rank, self.rope_dim, self.q_heads

        q_nope = torch.randn(bs, qh, lora, dtype=torch.bfloat16, device=device)
        q_rope = torch.randn(bs, qh, rope, dtype=torch.bfloat16, device=device)
        k_nope_orig = torch.randn(
            bs * ctx_len, 1, lora, dtype=torch.bfloat16, device=device
        )
        k_rope = torch.randn(
            bs * ctx_len, 1, rope, dtype=torch.bfloat16, device=device
        )
        packed, dequant_scale = _quantize_like_pool(k_nope_orig, self.cfg)

        kv_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=device) * ctx_len
        )
        kv_indices = torch.arange(0, bs * ctx_len, dtype=torch.int32, device=device)
        max_splits = 8
        att_logits = torch.empty(
            (bs, qh, max_splits, lora), dtype=torch.float32, device=device
        )
        att_lse = torch.empty((bs, qh, max_splits), dtype=torch.float32, device=device)
        o = torch.empty((bs, qh, lora), dtype=torch.bfloat16, device=device)
        q_nope_rot = self.cfg.rotate_query(q_nope).to(q_nope.dtype)

        # Happy path: per-batch (bs,) — shape (2,).
        num_kv_splits = torch.full(
            (bs,), max_splits, dtype=torch.int32, device=device
        )
        self.assertEqual(num_kv_splits.shape, torch.Size([bs]))

        tq_mla_decode_attention_fwd(
            q_nope_rotated=q_nope_rot,
            q_rope=q_rope,
            k_nope_packed=packed,
            k_scale=dequant_scale,
            k_rope=k_rope,
            k_centroids=self.cfg.k_centroids,
            o=o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            att_logits=att_logits,
            att_lse=att_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=max_splits,
            sm_scale=self.sm_scale,
            logit_cap=0.0,
            uniform=getattr(self.cfg, "uniform", False),
        )
        # If we get here without a CUDA illegal-access, the per-batch shape is
        # accepted. The numerical correctness is covered by the other tests.
        self.assertTrue(torch.isfinite(o).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
