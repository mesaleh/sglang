"""Benchmark the fused MLA TQ4 writer with and without FP8 codebook output."""

import argparse

import torch
import triton

from sglang.srt.layers.attention.triton_ops.turboquant_quantize import (
    fused_turboquant_quantize_and_store,
)
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig


def _capture_writer(tokens: int, store_codebook: bool):
    device = torch.device("cuda")
    latent = 512
    rope_dim = 64
    pool_size = max(32, tokens)
    config = TurboQuantConfig(4, latent, str(device))
    nope = torch.randn(tokens, 1, latent, device=device, dtype=torch.bfloat16)
    rope = torch.randn(tokens, 1, rope_dim, device=device, dtype=torch.bfloat16)
    loc = torch.arange(tokens, device=device, dtype=torch.long)
    packed = torch.empty(
        pool_size, 1, latent // 2, device=device, dtype=torch.uint8
    )
    scale = torch.empty(pool_size, 1, device=device, dtype=torch.bfloat16)
    rope_pool = torch.empty(
        pool_size, 1, rope_dim, device=device, dtype=torch.bfloat16
    )
    codebook = torch.empty(pool_size, 1, 16, device=device, dtype=torch.uint8)
    unit = torch.empty(tokens, 1, latent, device=device, dtype=torch.float32)
    norms = torch.empty(tokens, 1, device=device, dtype=torch.float32)
    rotated = torch.empty(tokens, 1, latent, device=device, dtype=torch.float32)

    def write():
        fused_turboquant_quantize_and_store(
            nope,
            config.signs1,
            config.signs2,
            config.k_centroids,
            config.k_boundaries,
            4,
            packed,
            scale,
            loc,
            pre_unit=unit,
            pre_norms=norms,
            pre_y=rotated,
            codebook_buffer=(
                codebook.view(torch.float8_e4m3fn) if store_codebook else None
            ),
            rope_src=rope,
            rope_buffer=rope_pool,
        )

    for _ in range(5):
        write()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    torch.cuda.synchronize()
    return graph


def _bench(graph: torch.cuda.CUDAGraph, warmup_ms: int, rep_ms: int):
    return triton.testing.do_bench(
        graph.replay,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 5, 128])
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--rep-ms", type=int, default=2000)
    args = parser.parse_args()

    torch.manual_seed(20260723)
    for tokens in args.tokens:
        baseline = _capture_writer(tokens, store_codebook=False)
        candidate = _capture_writer(tokens, store_codebook=True)
        order = (
            ("candidate_a1", candidate),
            ("baseline_b", baseline),
            ("candidate_a2", candidate),
        )
        for label, graph in order:
            p20, p50, p80 = _bench(graph, args.warmup_ms, args.rep_ms)
            print(
                f"tokens={tokens} variant={label} "
                f"p20_us={p20 * 1000:.4f} p50_us={p50 * 1000:.4f} "
                f"p80_us={p80 * 1000:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
