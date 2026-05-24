"""
Microbenchmark for the fused TurboQuant-MLA decode Triton kernel.

Measures per-step latency of tq_mla_decode_attention_fwd at Kimi K2.6 MLA
dims (lora_rank=512, rope_dim=64, q_heads=128) across a sweep of
(batch_size, context_length). Uses CUDA-graph-based timing for stable
measurement matching how the kernel is invoked in production (the
TurboQuantMLABackend captures the decode path into a CUDA graph).

Intentionally has no "baseline" comparator column — the goal is a durable
speed floor for THIS kernel across commits / Triton versions / pool layout
changes. A torch reference was considered but eliminated: a torch attention
in a per-batch loop requires `.item()` host syncs that can't run inside
CUDA graph capture, and a non-cudagraph reference measures a different
path than production serves. For end-to-end comparisons (e.g. Stage C
fused decode vs flashmla-on-bf16), use the deployment-level bench
documented in the vault integration log rather than a column here.

Placement:
  The kernel lives under python/sglang/srt/layers/attention/triton_ops/,
  not under python/sglang/jit_kernel/. Following the sibling precedent
  of benchmark/bench_attention_sink/ and benchmark/turboquant/, this
  bench lives under benchmark/bench_turboquant_mla/ rather than
  jit_kernel/benchmark/.

Usage (local):
  python benchmark/bench_turboquant_mla/bench_turboquant_mla_decode.py

Usage (inside pod):
  kubectl cp benchmark/bench_turboquant_mla/bench_turboquant_mla_decode.py \\
    <pod>:/tmp/bench_turboquant_mla_decode.py
  kubectl exec <pod> -- python3 /tmp/bench_turboquant_mla_decode.py

Related:
  Kernel: python/sglang/srt/layers/attention/triton_ops/turboquant_mla_decode_attention.py
  Test:   test/registered/attention/test_turboquant_mla_decode_kernel.py
  Design: (vault) Design - Stage C (fused Triton MLA decode on packed KV).md
"""

import itertools
import math
import os

import torch
import triton
import triton.testing


# --------------------------------------------------------------------------
# Fixed Kimi K2.6 architectural dims
# --------------------------------------------------------------------------
LORA_RANK = 512  # kv_lora_rank
ROPE_DIM = 64  # qk_rope_head_dim
Q_HEADS = 128
MAX_KV_SPLITS = 8  # kernel's compile-time constant

DTYPE = torch.bfloat16
DEVICE = "cuda"


# --------------------------------------------------------------------------
# Config sweeps
# --------------------------------------------------------------------------
try:
    from sglang.utils import is_in_ci

    IS_CI = is_in_ci()
except Exception:
    IS_CI = False

# Allow a quick smoke run via env var for local sanity checking without CI.
QUICK = os.environ.get("BENCH_TQ_MLA_QUICK", "0") == "1"

if QUICK or IS_CI:
    CONTEXT_LENS = [256, 2048]
    BATCH_SIZES = [1, 2]
else:
    CONTEXT_LENS = [256, 1024, 2048, 4096, 8192, 16384, 32768]
    BATCH_SIZES = [1, 2, 4, 8, 12]


# --------------------------------------------------------------------------
# Setup helpers (build inputs once per config)
# --------------------------------------------------------------------------
def _build_cfg():
    from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

    return TurboQuantConfig(
        bit_width=4,
        head_dim=LORA_RANK,
        device=DEVICE,
        k_bit_width=4,
        v_bit_width=4,
        uniform=False,
    )


def _prepare_inputs(cfg, bs: int, ctx_len: int):
    """Build a self-contained set of inputs for one (bs, ctx_len) config.

    Matches the live backend contract: Q_nope pre-rotated, Q_rope not rotated,
    K_nope in packed uint8 rotated-domain + bf16 per-token scale, K_rope in
    raw bf16, num_kv_splits per-batch int32.
    """
    from sglang.srt.layers.quantization.kv_turboquant import batched_quantize

    total_kv = bs * ctx_len

    q_nope = torch.randn(bs, Q_HEADS, LORA_RANK, dtype=DTYPE, device=DEVICE)
    q_rope = torch.randn(bs, Q_HEADS, ROPE_DIM, dtype=DTYPE, device=DEVICE)
    k_nope_orig = torch.randn(total_kv, 1, LORA_RANK, dtype=DTYPE, device=DEVICE)
    k_rope = torch.randn(total_kv, 1, ROPE_DIM, dtype=DTYPE, device=DEVICE)

    packed, norms, quant_norms = batched_quantize(
        k_nope_orig,
        cfg.signs1,
        cfg.signs2,
        cfg.k_centroids,
        cfg.k_boundaries,
        4,
    )
    threshold = torch.tensor(1e-10, dtype=DTYPE, device=DEVICE)
    replacement = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)
    safe_qnorm = torch.where(quant_norms > threshold, quant_norms, replacement)
    dequant_scale = norms / safe_qnorm

    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=DEVICE) * ctx_len
    kv_indices = torch.arange(0, total_kv, dtype=torch.int32, device=DEVICE)
    num_kv_splits = torch.full((bs,), MAX_KV_SPLITS, dtype=torch.int32, device=DEVICE)

    att_logits = torch.empty(
        (bs, Q_HEADS, MAX_KV_SPLITS, LORA_RANK), dtype=torch.float32, device=DEVICE
    )
    att_lse = torch.empty(
        (bs, Q_HEADS, MAX_KV_SPLITS), dtype=torch.float32, device=DEVICE
    )
    o = torch.empty((bs, Q_HEADS, LORA_RANK), dtype=DTYPE, device=DEVICE)

    q_nope_rot = cfg.rotate_query(q_nope).to(q_nope.dtype)

    return dict(
        q_nope_rot=q_nope_rot,
        q_rope=q_rope,
        packed=packed,
        dequant_scale=dequant_scale,
        k_rope=k_rope,
        k_centroids=cfg.k_centroids,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        att_logits=att_logits,
        att_lse=att_lse,
        o=o,
        num_kv_splits=num_kv_splits,
        sm_scale=1.0 / math.sqrt(LORA_RANK + ROPE_DIM),
        uniform=getattr(cfg, "uniform", False),
    )


# --------------------------------------------------------------------------
# Benchmark bodies
# --------------------------------------------------------------------------
def _run_fused(inputs):
    """The Stage C Phase 3 kernel — what production serves."""
    from sglang.srt.layers.attention.triton_ops.turboquant_mla_decode_attention import (
        tq_mla_decode_attention_fwd,
    )

    tq_mla_decode_attention_fwd(
        q_nope_rotated=inputs["q_nope_rot"],
        q_rope=inputs["q_rope"],
        k_nope_packed=inputs["packed"],
        k_scale=inputs["dequant_scale"],
        k_rope=inputs["k_rope"],
        k_centroids=inputs["k_centroids"],
        o=inputs["o"],
        kv_indptr=inputs["kv_indptr"],
        kv_indices=inputs["kv_indices"],
        att_logits=inputs["att_logits"],
        att_lse=inputs["att_lse"],
        num_kv_splits=inputs["num_kv_splits"],
        max_kv_splits=MAX_KV_SPLITS,
        sm_scale=inputs["sm_scale"],
        logit_cap=0.0,
        uniform=inputs["uniform"],
    )


# --------------------------------------------------------------------------
# Timer
# --------------------------------------------------------------------------
configs = list(itertools.product(BATCH_SIZES, CONTEXT_LENS))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "ctx_len"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["fused"],
        line_names=["Fused Triton MLA decode (Stage C Phase 3)"],
        styles=[("green", "-")],
        ylabel="µs (median)",
        plot_name="tq-mla-decode-performance",
        args={},
    )
)
def benchmark(bs: int, ctx_len: int, provider: str):
    cfg = _build_cfg()
    inputs = _prepare_inputs(cfg, bs, ctx_len)

    fn = lambda: _run_fused(inputs)

    # CUDA-graph timing — reproduces how the kernel runs in production
    # (TurboQuantMLABackend captures the decode path into a CUDA graph).
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
