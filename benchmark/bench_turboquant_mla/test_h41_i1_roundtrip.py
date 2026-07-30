"""H43 I2 selected front-end-to-codebook-reader composition gate.

This focused test covers q1/q5 eager and graph-replay consumption of the
material 338-byte TQ row. It is correctness evidence, not a timing result.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Callable
from typing import Any

import torch

from h43_aot_loader import install_h43_aot_from_environment
from sglang.jit_kernel.tq_mla_frontend import tq_mla_frontend_out
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq4

_H43_AOT_LOADER = install_h43_aot_from_environment()

LATENT = 512
ROPE = 64
HEADS = 8
PAGE = 32
MAX_CONTEXT = 256000
CACHE_ROWS = MAX_CONTEXT
CACHE_PAGES = CACHE_ROWS // PAGE
REQUEST_PAGES = MAX_CONTEXT // PAGE
FP8 = torch.float8_e4m3fn
E2M1_CENTROIDS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def assert_idle_before_cuda() -> dict[str, Any]:
    fields = "pid,process_name,used_memory"
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=" + fields,
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = completed.stdout.strip()
    if output:
        raise RuntimeError(
            f"H43 I2 requires an idle node before CUDA init; found: {output}"
        )
    return {"query": fields, "rows": []}


def reconstruct_cache(
    dense: torch.Tensor,
    packed: torch.Tensor,
    codebook: torch.Tensor,
    rope: torch.Tensor,
    *,
    chunk_pages: int = 128,
) -> None:
    for first in range(0, CACHE_PAGES, chunk_pages):
        last = min(first + chunk_pages, CACHE_PAGES)
        packed_chunk = packed[first:last]
        indices = torch.empty(
            (*packed_chunk.shape[:-1], LATENT),
            dtype=torch.uint8,
            device=packed.device,
        )
        indices[..., 0::2] = packed_chunk & 0x0F
        indices[..., 1::2] = packed_chunk >> 4
        decoded = torch.gather(
            codebook[first:last].view(FP8).float(), -1, indices.long()
        )
        dense[first:last, ..., :LATENT].copy_(decoded)
        dense[first:last, ..., LATENT:].copy_(rope[first:last])


def reconstruct_rows(
    dense: torch.Tensor,
    packed: torch.Tensor,
    codebook: torch.Tensor,
    rope: torch.Tensor,
    locations: torch.Tensor,
) -> None:
    packed_rows = packed.view(CACHE_ROWS, LATENT // 2)[locations]
    indices = torch.empty(
        locations.numel(), LATENT, dtype=torch.uint8, device=packed.device
    )
    indices[:, 0::2] = packed_rows & 0x0F
    indices[:, 1::2] = packed_rows >> 4
    codebook_rows = codebook.view(CACHE_ROWS, 16)[locations].view(FP8).float()
    decoded = torch.gather(codebook_rows, -1, indices.long()).to(FP8)
    dense_rows = dense.view(CACHE_ROWS, LATENT + ROPE)
    dense_rows[locations, :LATENT] = decoded
    dense_rows[locations, LATENT:] = rope.view(CACHE_ROWS, ROPE)[locations]


def capture_graph(fn: Callable[[], torch.Tensor]) -> torch.cuda.CUDAGraph:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(10219, 37932), required=True)
    parser.add_argument("--q-len", type=int, choices=(1, 5), required=True)
    parser.add_argument("--split-kv", type=int, choices=(40, 64), required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--correctness-atol", type=float, default=0.002)
    args = parser.parse_args()
    expected_split = {10219: 64, 37932: 40}[args.context]
    if args.split_kv != expected_split:
        raise ValueError(
            f"context {args.context} requires frozen split-kv {expected_split}"
        )

    idle_proof = assert_idle_before_cuda()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H43 I2 requires SM100")
    if not is_arch_support_pdl():
        raise RuntimeError("H43 I2 requires the SM100 PDL path")

    import flashinfer.rope

    generator = torch.Generator(device=device).manual_seed(args.seed)
    page_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    centroids = torch.tensor(E2M1_CENTROIDS, dtype=torch.float32, device=device)
    packed = torch.empty(
        CACHE_PAGES, PAGE, LATENT // 2, dtype=torch.uint8, device=device
    ).random_(0, 256, generator=generator)
    scale = torch.empty(
        CACHE_PAGES, PAGE, dtype=torch.bfloat16, device=device
    ).uniform_(0.05, 0.20, generator=generator)
    codebook = (
        (scale.float()[..., None] * centroids).to(FP8).view(torch.uint8).contiguous()
    )
    if codebook.data_ptr() % 16:
        raise AssertionError("codebook allocation must be 16-byte aligned")
    rope = torch.empty(CACHE_PAGES, PAGE, ROPE, dtype=FP8, device=device)
    rope_scratch = torch.empty_like(rope, dtype=torch.bfloat16).normal_(
        0.0, 0.1, generator=generator
    )
    rope.copy_(rope_scratch)
    del rope_scratch
    dense = torch.empty(CACHE_PAGES, PAGE, LATENT + ROPE, dtype=FP8, device=device)
    reconstruct_cache(dense, packed, codebook, rope)

    page_table = torch.randperm(
        REQUEST_PAGES,
        dtype=torch.int32,
        device=device,
        generator=page_generator,
    ).unsqueeze(0)
    seq_lens = torch.tensor([args.context], dtype=torch.int32, device=device)
    positions = torch.arange(
        args.context - args.q_len,
        args.context,
        dtype=torch.int64,
        device=device,
    )
    logical_pages = torch.div(positions, PAGE, rounding_mode="floor")
    locations = (
        page_table[0, logical_pages].to(torch.int64) * PAGE
        + torch.remainder(positions, PAGE)
    ).contiguous()
    if torch.unique(locations).numel() != args.q_len:
        raise AssertionError("allocator contract requires unique writer locations")

    def make_bf16(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.bfloat16, device=device).uniform_(
            -0.125, 0.125, generator=generator
        )

    query_latent = make_bf16(args.q_len, HEADS, LATENT)
    query_rope_raw = make_bf16(args.q_len, HEADS, ROPE)
    cache_latent = make_bf16(args.q_len, 1, LATENT)
    cache_rope_raw = make_bf16(args.q_len, 1, ROPE)
    selected_query_rope = torch.empty_like(query_rope_raw)
    selected_cache_rope = torch.empty_like(cache_rope_raw)
    query = torch.empty(args.q_len, HEADS, LATENT + ROPE, dtype=FP8, device=device)
    cos_sin_cache = torch.empty(
        MAX_CONTEXT, ROPE, dtype=torch.float32, device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    dense_workspace = torch.empty(64 << 20, dtype=torch.int8, device=device)
    tq_workspace = torch.empty_like(dense_workspace)
    dense_out = torch.empty(
        1, args.q_len, HEADS, LATENT, dtype=torch.bfloat16, device=device
    )
    tq_out = torch.empty_like(dense_out)
    config = TurboQuantConfig(
        bit_width=4,
        head_dim=LATENT,
        device=device,
        k_bit_width=4,
        v_bit_width=4,
        uniform=False,
        e2m1=True,
    )
    assert config.k_storage_code_lut is not None
    torch.testing.assert_close(config.k_centroids, centroids, rtol=0, atol=0)
    softmax_scale = 1.0 / math.sqrt(LATENT + ROPE)

    def frontend() -> None:
        flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
            q=query_rope_raw,
            k=cache_rope_raw,
            q_rope=selected_query_rope,
            k_rope=selected_cache_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            interleave=False,
        )
        tq_mla_frontend_out(
            query_latent,
            selected_query_rope,
            cache_latent,
            selected_cache_rope,
            locations,
            config.signs1,
            config.signs2,
            config.k_boundaries,
            config.k_quant_centroids,
            config.k_storage_code_lut,
            query,
            packed.view(CACHE_ROWS, 1, LATENT // 2),
            scale.view(CACHE_ROWS, 1),
            rope.view(CACHE_ROWS, 1, ROPE),
            status,
            decode_centroids=config.k_centroids,
            codebook_cache=codebook.view(CACHE_ROWS, 1, 16),
            scale_multiplier=config.k_dequant_scale_multiplier,
            rotation_fused=True,
            num_warps=8,
            strict=False,
        )

    def tq_call() -> torch.Tensor:
        return tokenspeed_mla_decode_tq4(
            query=query.view(1, args.q_len, HEADS, LATENT + ROPE),
            kv_nope_packed=packed,
            kv_nope_scale=scale,
            kv_rope=rope,
            centroids=centroids,
            workspace_buffer=tq_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=MAX_CONTEXT,
            softmax_scale=softmax_scale,
            out=tq_out,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=args.split_kv,
            kv_nope_codebook=codebook,
            fp8_rope=True,
        )

    def dense_call() -> torch.Tensor:
        return tokenspeed_mla_decode(
            query=query.view(1, args.q_len, HEADS, LATENT + ROPE),
            kv_cache=dense,
            workspace_buffer=dense_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=MAX_CONTEXT,
            softmax_scale=softmax_scale,
            out=dense_out,
            causal_mask=True,
            enable_pdl=True,
        )

    def candidate() -> torch.Tensor:
        frontend()
        return tq_call()

    def check(label: str) -> dict[str, Any]:
        reconstruct_rows(dense, packed, codebook, rope, locations)
        dense_call()
        torch.cuda.synchronize()
        max_abs_diff = float((tq_out.float() - dense_out.float()).abs().max())
        torch.testing.assert_close(
            tq_out, dense_out, rtol=0, atol=args.correctness_atol
        )
        if int(status.item()) != 0:
            raise AssertionError(f"{label} changed sticky status")
        return {
            "label": label,
            "max_abs_diff": max_abs_diff,
            "tq_checksum": float(tq_out.float().sum()),
            "dense_checksum": float(dense_out.float().sum()),
        }

    candidate()
    eager = check("eager")
    graph = capture_graph(candidate)
    allocation_before = torch.cuda.memory_allocated(device)
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    allocation_after = torch.cuda.memory_allocated(device)
    if allocation_before != allocation_after:
        raise AssertionError(
            f"graph replay allocated: {allocation_before} -> {allocation_after}"
        )
    replay = check("graph_replay")
    result = {
        "status": "PASS",
        "experiment": "H43_I2_WRITER_CODEBOOK_READER_COMPOSITION",
        "context": args.context,
        "q_len": args.q_len,
        "split_kv": args.split_kv,
        "page_size": PAGE,
        "logical_positions": positions.tolist(),
        "physical_locations": locations.tolist(),
        "unique_locations": torch.unique(locations).numel() == args.q_len,
        "partial_final_page": args.context % PAGE != 0,
        "shuffled_page_table": not bool(
            torch.equal(
                page_table[0],
                torch.arange(REQUEST_PAGES, dtype=torch.int32, device=device),
            )
        ),
        "selected_warps": 8,
        "selected_row_bytes": 338,
        "fp8_rope": True,
        "codebook_materialized": True,
        "sticky_status": int(status.item()),
        "correctness_atol": args.correctness_atol,
        "checks": [eager, replay],
        "graph_replay_allocation_bytes": {
            "before": allocation_before,
            "after": allocation_after,
        },
        "idle_proof": idle_proof,
        "seed": args.seed,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
