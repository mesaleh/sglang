"""Fused TurboQuant-MLA decode attention kernel.

Reads packed uint8 KV cache directly during MLA (Multi-head Latent Attention)
decode, eliminating the per-call full-pool dequant that Stage A pays.

Design reference:
  OmniSec/Inference/Performance Optimization/
    Design - Stage C (fused Triton MLA decode on packed KV).md
  OmniSec/Inference/Performance Optimization/
    Design - Stage C Phase 1 findings.md
  OmniSec/Inference/Kernel Engineering Rules.md

MLA differs from MHA (KE-11):
  * Single shared latent (kv_lora_rank=512 nope), not separate K/V buffers.
  * One scale per token per layer (not per K and per V).
  * RoPE stored uncompressed in bf16, separate buffer.
  * Grouped attention: many Q heads, 1 K head per token.
  * Absorbed form: acc = p @ K_nope (W_V absorbed into W_O).

Rotation handling (KE-2):
  * K_nope stored in Walsh-Hadamard-rotated domain.
  * Orthogonality trick: (H·Q_nope) · (H·K_nope) = Q_nope · K_nope.
  * Backend must rotate Q_nope before calling this kernel (tq_config.rotate_query).
  * Kernel output is in rotated nope space. Backend applies
    tq_config.inverse_rotate_output after stage-2 reduction.
  * RoPE is NOT rotated (TurboQuant skips RoPE); Q_rope · K_rope is direct.

Kernel reference ports:
  - turboquant_decode_attention.py (MHA TurboQuant, PR #23135):
    codebook lookup, 2-way N-split dot pattern, split-accumulator output.
  - rocm_mla_decode_rope.py (MLA FP8, PR #20479):
    MLA stage-1/stage-2 split, grouped-MLA head tiling.
"""

import os

import triton
import triton.language as tl


_MIN_BLOCK_KV = 32


def _get_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_power_of_two(name: str, value: int) -> int:
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")
    return value


_TQ_MLA_DECODE_BLOCK_N = _require_power_of_two(
    "SGLANG_TQ_MLA_DECODE_BLOCK_N",
    _get_int_env("SGLANG_TQ_MLA_DECODE_BLOCK_N", 64, minimum=16),
)
_TQ_MLA_DECODE_NUM_WARPS = _get_int_env("SGLANG_TQ_MLA_DECODE_NUM_WARPS", 8)
_TQ_MLA_DECODE_NUM_STAGES = _get_int_env("SGLANG_TQ_MLA_DECODE_NUM_STAGES", 2)


@triton.jit
def _lookup_4bit_codebook(
    idx,
    c0, c1, c2, c3, c4, c5, c6, c7,
    c8, c9, c10, c11, c12, c13, c14, c15,
):
    """4-bit codebook lookup via binary select tree (4 levels, 15 tl.where).
    Matches the MHA reference kernel's _lookup_4bit_codebook."""
    lo = tl.where(
        (idx & 4) != 0,
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c7, c6),
                 tl.where((idx & 1) != 0, c5, c4)),
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c3, c2),
                 tl.where((idx & 1) != 0, c1, c0)),
    )
    hi = tl.where(
        (idx & 4) != 0,
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c15, c14),
                 tl.where((idx & 1) != 0, c13, c12)),
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c11, c10),
                 tl.where((idx & 1) != 0, c9, c8)),
    )
    return tl.where((idx & 8) != 0, hi, lo)


@triton.jit
def _lookup_4bit_uniform(idx, c0, c15):
    """Uniform-quant shortcut: 1 FMA replaces 15 tl.where selects.
    Uniform layout encodes centroids as c0 + idx * (c15 - c0) / 15."""
    step = (c15 - c0) * 0.06666666666666667  # 1/15
    return idx.to(tl.float32) * step + c0


@triton.jit
def _fwd_tq_mla_decode_stage1(
    # --- Inputs ---
    # Q tensors: backend has already applied tq_config.rotate_query to Q_nope.
    Q_Nope,           # (bs, q_heads, lora_rank) bf16 (rotated domain)
    Q_Rope,           # (bs, q_heads, rope_dim) bf16 (original domain)
    # Packed KV (per-layer; pointer passed for the current layer).
    K_Nope_Packed,    # (pool_size, 1, lora_rank//2) uint8 (rotated domain)
    K_Scale,          # (pool_size, 1) bf16 (per-token dequant scale)
    K_Rope,           # (pool_size, 1, rope_dim) bf16 (original domain)
    K_Centroids,      # (16,) fp32 — 4-bit codebook
    # Paging
    kv_indptr,        # (bs+1,) int32
    kv_indices,       # (total_kv_tokens,) int32 — logical → pool row
    num_kv_splits,    # (bs,) int32
    # --- Outputs ---
    Att_Out,          # (bs, q_heads, num_splits, lora_rank) fp32 (partial, rotated)
    Att_Lse,          # (bs, q_heads, num_splits) fp32 (log-sum-exp per split)
    # --- Strides ---
    stride_q_nope_bs, stride_q_nope_h,
    stride_q_rope_bs, stride_q_rope_h,
    stride_kp_bs, stride_kp_h,       # packed nope: last dim contiguous = bytes
    stride_ksc_bs,                    # scale: last dim (head=1) assumed contiguous
    stride_krope_bs, stride_krope_h,  # rope: last dim contiguous = rope_dim
    stride_mid_ob, stride_mid_oh, stride_mid_os,  # Att_Out strides
    # --- Scalars ---
    sm_scale,
    # --- Constexpr ---
    q_head_num: tl.constexpr,
    BLOCK_N: tl.constexpr,              # K rows per inner iteration
    BLOCK_H: tl.constexpr,              # Q heads per program
    MIN_BLOCK_KV: tl.constexpr,
    LORA_RANK: tl.constexpr,            # 512 (Kimi K2.6 nope)
    ROPE_DIM: tl.constexpr,             # 64  (Kimi K2.6 rope)
    LORA_PACKED: tl.constexpr,          # 256 = lora_rank / 2 (4-bit, 2 per byte)
    BLOCK_LORA: tl.constexpr,           # next_power_of_2(LORA_RANK) = 512
    BLOCK_LORA_PACKED: tl.constexpr,    # next_power_of_2(LORA_PACKED) = 256
    BLOCK_ROPE: tl.constexpr,           # next_power_of_2(ROPE_DIM) = 64
    logit_cap: tl.constexpr,
    UNIFORM: tl.constexpr,              # True = uniform codebook (cheap lookup)
):
    # Grid: (bs, q_head_blocks, num_splits).
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # Head tile
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < q_head_num

    # Offsets
    offs_lora = tl.arange(0, BLOCK_LORA)         # output interleave axis
    mask_lora = offs_lora < LORA_RANK
    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_rope = offs_rope < ROPE_DIM
    offs_kp = tl.arange(0, BLOCK_LORA_PACKED)    # packed-byte axis
    mask_kp = offs_kp < LORA_PACKED

    # KV slice for this batch and split
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # Running state
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # Split accumulators (Fix 5 in the Phase 1 findings doc):
    # acc_even covers even-indexed lora_rank positions: 0, 2, 4, ..., 510
    # acc_odd  covers odd-indexed  lora_rank positions: 1, 3, 5, ..., 511
    # On store we interleave these into the full lora_rank output.
    acc_even = tl.zeros([BLOCK_H, BLOCK_LORA_PACKED], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, BLOCK_LORA_PACKED], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Preload codebook centroids (16 fp32 scalars)
        c0 = tl.load(K_Centroids + 0)
        c1 = tl.load(K_Centroids + 1)
        c2 = tl.load(K_Centroids + 2)
        c3 = tl.load(K_Centroids + 3)
        c4 = tl.load(K_Centroids + 4)
        c5 = tl.load(K_Centroids + 5)
        c6 = tl.load(K_Centroids + 6)
        c7 = tl.load(K_Centroids + 7)
        c8 = tl.load(K_Centroids + 8)
        c9 = tl.load(K_Centroids + 9)
        c10 = tl.load(K_Centroids + 10)
        c11 = tl.load(K_Centroids + 11)
        c12 = tl.load(K_Centroids + 12)
        c13 = tl.load(K_Centroids + 13)
        c14 = tl.load(K_Centroids + 14)
        c15 = tl.load(K_Centroids + 15)

        # Load Q_nope split into even/odd — matches MHA reference lines 226-240.
        # Even positions: lora[0, 2, 4, ..., 510]; odd: lora[1, 3, ..., 511].
        q_even_cols = 2 * offs_kp                # (BLOCK_LORA_PACKED,)
        q_odd_cols = 2 * offs_kp + 1
        mask_q_even = mask_h[:, None] & (q_even_cols[None, :] < LORA_RANK)
        mask_q_odd = mask_h[:, None] & (q_odd_cols[None, :] < LORA_RANK)

        offs_q_even = (
            cur_batch * stride_q_nope_bs
            + cur_head[:, None] * stride_q_nope_h
            + q_even_cols[None, :]
        )
        offs_q_odd = (
            cur_batch * stride_q_nope_bs
            + cur_head[:, None] * stride_q_nope_h
            + q_odd_cols[None, :]
        )
        q_nope_even = tl.load(Q_Nope + offs_q_even, mask=mask_q_even, other=0.0)
        q_nope_odd = tl.load(Q_Nope + offs_q_odd, mask=mask_q_odd, other=0.0)

        # Load Q_rope (all dims, BLOCK_H heads)
        offs_q_rope = (
            cur_batch * stride_q_rope_bs
            + cur_head[:, None] * stride_q_rope_h
            + offs_rope[None, :]
        )
        mask_q_rope = mask_h[:, None] & mask_rope[None, :]
        q_rope = tl.load(Q_Rope + offs_q_rope, mask=mask_q_rope, other=0.0)

        # Tile over K rows
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end

            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
            )

            # --- Load packed K_nope: (BLOCK_LORA_PACKED, BLOCK_N) uint8 ---
            offs_buf_kp = (
                kv_loc[None, :] * stride_kp_bs
                + 0 * stride_kp_h
                + offs_kp[:, None]
            )
            packed_k = tl.load(
                K_Nope_Packed + offs_buf_kp,
                mask=mask_n[None, :] & mask_kp[:, None],
                other=0,
            )

            # Split each byte into two 4-bit indices:
            #   bit bit_0 (low nibble)  → lora position 2*i     → k_lo
            #   bit bit_1 (high nibble) → lora position 2*i + 1 → k_hi
            k_idx_lo = (packed_k & 0x0F).to(tl.int32)
            k_idx_hi = ((packed_k >> 4) & 0x0F).to(tl.int32)

            if UNIFORM:
                k_lo = _lookup_4bit_uniform(k_idx_lo, c0, c15)
                k_hi = _lookup_4bit_uniform(k_idx_hi, c0, c15)
            else:
                k_lo = _lookup_4bit_codebook(
                    k_idx_lo, c0, c1, c2, c3, c4, c5, c6, c7,
                    c8, c9, c10, c11, c12, c13, c14, c15,
                )
                k_hi = _lookup_4bit_codebook(
                    k_idx_hi, c0, c1, c2, c3, c4, c5, c6, c7,
                    c8, c9, c10, c11, c12, c13, c14, c15,
                )
            # k_lo, k_hi shape: (BLOCK_LORA_PACKED, BLOCK_N) fp32

            # --- Q_nope · K_nope (rotated domain, needs per-token scale) ---
            # k_lo / k_hi are codebook values in rotated space, BEFORE the
            # per-token dequant scale has been applied. We scale the nope
            # contribution here, then add the un-scaled rope contribution.
            # (Bug fix 2026-04-27: previously applied k_scale to the combined
            # nope+rope qk, which wrongly scaled the rope term.)
            qk_nope = tl.dot(q_nope_even, k_lo.to(q_nope_even.dtype))
            qk_nope += tl.dot(q_nope_odd, k_hi.to(q_nope_odd.dtype))

            k_scale = tl.load(
                K_Scale + kv_loc * stride_ksc_bs + 0,
                mask=mask_n,
                other=1.0,
            ).to(tl.float32)
            qk_nope = qk_nope * k_scale[None, :]

            # --- Q_rope · K_rope (original domain, NO scale) ---
            # RoPE is stored uncompressed in bf16; TurboQuant does not touch
            # it, so no per-token dequant scale applies here.
            offs_buf_krope = (
                kv_loc[None, :] * stride_krope_bs
                + 0 * stride_krope_h
                + offs_rope[:, None]
            )
            k_rope = tl.load(
                K_Rope + offs_buf_krope,
                mask=mask_n[None, :] & mask_rope[:, None],
                other=0.0,
            )
            qk_rope = tl.dot(q_rope, k_rope.to(q_rope.dtype))

            # Combine scaled-nope + un-scaled-rope; apply softmax scale once.
            qk = (qk_nope + qk_rope) * sm_scale

            if logit_cap > 0:
                qk = logit_cap * (2.0 * tl.sigmoid(2.0 * qk / logit_cap) - 1.0)

            qk = tl.where(
                mask_h[:, None] & mask_n[None, :],
                qk,
                float("-inf"),
            )

            # --- Online softmax update ---
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc_even *= re_scale[:, None]
            acc_odd *= re_scale[:, None]

            # --- Accumulate p @ K_nope (absorbed form: V = K_nope) ---
            # MLA uses a single per-token scale that applies to BOTH the QK
            # dot (already applied above) AND the weighted-sum accumulation
            # p @ K_nope. Without this, acc is in a pre-scale rotated domain
            # and downstream inverse_rotate_output produces wrong magnitudes.
            # Matches MHA reference's V-side flow (turboquant_decode_attention.py:320
            # where p_scaled = p * v_dscale).
            p_scaled = p * k_scale[None, :]

            # p_scaled: (BLOCK_H, BLOCK_N). k_lo/k_hi: (BLOCK_LORA_PACKED, BLOCK_N).
            # We need acc_even += p_scaled @ k_lo.T — tl.dot(p_scaled, k_lo.T).
            k_lo_T = tl.trans(k_lo).to(p.dtype)  # (BLOCK_N, BLOCK_LORA_PACKED)
            k_hi_T = tl.trans(k_hi).to(p.dtype)
            acc_even += tl.dot(p_scaled.to(k_lo_T.dtype), k_lo_T)
            acc_odd += tl.dot(p_scaled.to(k_hi_T.dtype), k_hi_T)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Normalize
        acc_even /= e_sum[:, None]
        acc_odd /= e_sum[:, None]

        # --- Interleaved store into (lora_rank,) output axis ---
        # Even accumulator → lora positions [0, 2, ..., 510]
        # Odd accumulator  → lora positions [1, 3, ..., 511]
        base_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        offs_out_even = 2 * offs_kp                  # (BLOCK_LORA_PACKED,)
        offs_out_odd = 2 * offs_kp + 1
        mask_out_even = mask_h[:, None] & (offs_out_even[None, :] < LORA_RANK)
        mask_out_odd = mask_h[:, None] & (offs_out_odd[None, :] < LORA_RANK)

        tl.store(
            Att_Out + base_mid_o + offs_out_even[None, :],
            acc_even,
            mask=mask_out_even,
        )
        tl.store(
            Att_Out + base_mid_o + offs_out_odd[None, :],
            acc_odd,
            mask=mask_out_odd,
        )

        # --- Store LSE ---
        # Layout: (bs, q_heads, num_splits), one scalar per (batch, head, split).
        # Matches MHA reference: offs_mid_lse = (cur_batch * stride_ob
        #   + cur_head * stride_oh + split_kv_id * stride_os) / LORA_RANK
        offs_mid_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // LORA_RANK

        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum), mask=mask_h)


def tq_mla_decode_attention_fwd(
    q_nope_rotated: "torch.Tensor",    # (bs, q_heads, lora_rank) bf16, rotated
    q_rope: "torch.Tensor",            # (bs, q_heads, rope_dim) bf16
    k_nope_packed: "torch.Tensor",     # (pool_size, 1, lora_rank//2) uint8
    k_scale: "torch.Tensor",           # (pool_size, 1) bf16
    k_rope: "torch.Tensor",            # (pool_size, 1, rope_dim) bf16
    k_centroids: "torch.Tensor",       # (16,) fp32
    o: "torch.Tensor",                 # (bs, q_heads, lora_rank) bf16 — output
    kv_indptr: "torch.Tensor",
    kv_indices: "torch.Tensor",
    att_logits: "torch.Tensor",        # (bs, q_heads, max_splits, lora_rank) fp32
    att_lse: "torch.Tensor",           # (bs, q_heads, max_splits) fp32
    num_kv_splits: "torch.Tensor",
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    uniform: bool = False,
):
    """Run fused TurboQuant-MLA decode attention.

    The kernel output (in att_logits) is in the Walsh-Hadamard-rotated nope
    domain. The caller is responsible for applying
    tq_config.inverse_rotate_output to the final (post-stage-2) output
    before handing back to the MLA forward path.

    q_nope_rotated MUST be pre-rotated by the caller via
    tq_config.rotate_query; this kernel assumes it's in rotated space.
    """
    import torch

    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_softmax_reducev_fwd,
    )

    bs, q_heads, lora_rank = q_nope_rotated.shape
    rope_dim = q_rope.shape[-1]
    lora_packed = k_nope_packed.shape[-1]
    assert lora_rank == 2 * lora_packed, (
        f"lora_rank={lora_rank} must equal 2 * packed_dim={lora_packed} "
        f"for 4-bit TurboQuant"
    )
    assert k_rope.shape[-1] == rope_dim
    assert att_logits.shape == (bs, q_heads, max_kv_splits, lora_rank)
    assert att_lse.shape == (bs, q_heads, max_kv_splits)
    assert o.shape == (bs, q_heads, lora_rank)

    BLOCK_LORA = triton.next_power_of_2(lora_rank)
    BLOCK_LORA_PACKED = triton.next_power_of_2(lora_packed)
    BLOCK_ROPE = triton.next_power_of_2(rope_dim)
    BLOCK_N = _TQ_MLA_DECODE_BLOCK_N
    BLOCK_H = min(16, q_heads)

    grid = (
        bs,
        triton.cdiv(q_heads, BLOCK_H),
        max_kv_splits,
    )

    _fwd_tq_mla_decode_stage1[grid](
        q_nope_rotated,
        q_rope,
        k_nope_packed,
        k_scale,
        k_rope,
        k_centroids,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        att_logits,
        att_lse,
        q_nope_rotated.stride(0), q_nope_rotated.stride(1),
        q_rope.stride(0), q_rope.stride(1),
        k_nope_packed.stride(0), k_nope_packed.stride(1),
        k_scale.stride(0),
        k_rope.stride(0), k_rope.stride(1),
        att_logits.stride(0), att_logits.stride(1), att_logits.stride(2),
        sm_scale,
        q_head_num=q_heads,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        LORA_RANK=lora_rank,
        ROPE_DIM=rope_dim,
        LORA_PACKED=lora_packed,
        BLOCK_LORA=BLOCK_LORA,
        BLOCK_LORA_PACKED=BLOCK_LORA_PACKED,
        BLOCK_ROPE=BLOCK_ROPE,
        logit_cap=logit_cap,
        UNIFORM=uniform,
        num_warps=_TQ_MLA_DECODE_NUM_WARPS,
        num_stages=_TQ_MLA_DECODE_NUM_STAGES,
    )

    # Stage 2: softmax-reduce across splits. Reuses the MHA decode_attention
    # stage-2 kernel (shape-agnostic along Lv axis; we use o itself as the
    # "v_buffer" template to pick Lv=lora_rank).
    _decode_softmax_reducev_fwd(
        att_logits,
        att_lse,
        q_nope_rotated,  # used only for shape (bs, q_heads)
        o,
        1.0,              # v_scale = 1.0 (scale already applied in stage1)
        o,                # v_buffer: only shape[-1] used for Lv
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        None,             # sinks (not used)
    )
