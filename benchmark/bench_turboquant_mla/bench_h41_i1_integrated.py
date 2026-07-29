"""H41 I1 complete 61-layer four-family CUDA-graph falsifier.

This is isolated graph evidence. It is not an endpoint or quality result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from sglang.jit_kernel.tq_mla_frontend import tq_mla_frontend_out
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
from sglang.srt.mem_cache.triton_ops.mla_buffer import set_mla_kv_buffer_triton
from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq4

LATENT = 512
ROPE = 64
HEADS = 8
TOKENS = 5
PAGE = 32
MAX_CONTEXT = 256000
CACHE_ROWS = MAX_CONTEXT
CACHE_PAGES = CACHE_ROWS // PAGE
REQUEST_PAGES = MAX_CONTEXT // PAGE
TOTAL_LAYERS = 61
SELECTED_LAYERS = 14
DENSE_BEFORE = 24
DENSE_AFTER = 23
DENSE_LAYERS = DENSE_BEFORE + DENSE_AFTER
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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p20": percentile(values, 0.20),
        "median": percentile(values, 0.50),
        "p80": percentile(values, 0.80),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=20
    )
    return completed.stdout.strip()


def assert_idle_before_cuda() -> dict[str, Any]:
    fields = "pid,process_name,used_memory"
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=" + fields,
            "--format=csv,noheader,nounits",
        ]
    )
    if output:
        raise RuntimeError(f"H41 I1 requires an idle node before CUDA init; found: {output}")
    return {"query": fields, "rows": []}


def gpu_covariates() -> dict[str, Any]:
    fields = (
        "timestamp,index,pstate,clocks.current.graphics,clocks.max.graphics,"
        "temperature.gpu,power.draw,utilization.gpu,memory.used,"
        "ecc.errors.uncorrected.volatile.total,gpu_recovery_action"
    )
    try:
        output = command_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=" + fields,
                "--format=csv,noheader,nounits",
            ]
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"error": repr(error)}
    return dict(
        zip(
            fields.split(","),
            [part.strip() for part in output.split(",")],
            strict=False,
        )
    )


def capture_graph(
    fn: Callable[[], torch.Tensor],
    *,
    warmups: int,
    debug_path: Path | None,
) -> torch.cuda.CUDAGraph:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if debug_path is not None:
        graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        graph.debug_dump(str(debug_path))
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    samples: int,
    replays_per_sample: int,
    output: torch.Tensor,
    status: torch.Tensor | None,
) -> dict[str, Any]:
    before = gpu_covariates()
    sample_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(replays_per_sample):
            graph.replay()
        end.record()
        end.synchronize()
        sample_us.append(start.elapsed_time(end) * 1000.0 / replays_per_sample)
    after = gpu_covariates()
    status_value = int(status.item()) if status is not None else None
    if status_value not in (None, 0):
        raise AssertionError(f"candidate sticky status changed to {status_value}")
    return {
        "sample_graph_us": sample_us,
        "graph_us": summarize(sample_us),
        "per_total_layer_us": summarize(
            [value / TOTAL_LAYERS for value in sample_us]
        ),
        "replays": samples * replays_per_sample,
        "gpu_before": before,
        "gpu_after": after,
        "output_checksum": float(output.float().sum()),
        "sticky_status": status_value,
    }


def operation_traces() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    control = [
        {
            "layer": layer,
            "representation": "dense_fp8",
            "operations": [
                "flashinfer_rope_quantize_fp8",
                "dense_fp8_cache_scatter",
                "tokenspeed_dense_attention",
            ],
        }
        for layer in range(TOTAL_LAYERS)
    ]
    candidate: list[dict[str, Any]] = []
    for layer in range(TOTAL_LAYERS):
        if DENSE_BEFORE <= layer < DENSE_BEFORE + SELECTED_LAYERS:
            candidate.append(
                {
                    "layer": layer,
                    "representation": "tq4_e2m1_fp8_rope",
                    "operations": [
                        "selected_bf16_rope",
                        "h41_combined_query_material_writer",
                        "tokenspeed_tq4_attention",
                    ],
                }
            )
        else:
            candidate.append(control[layer].copy())
    assert len(control) == TOTAL_LAYERS
    assert len(candidate) == TOTAL_LAYERS
    assert sum(item["representation"] == "dense_fp8" for item in candidate) == DENSE_LAYERS
    assert (
        sum(item["representation"] == "tq4_e2m1_fp8_rope" for item in candidate)
        == SELECTED_LAYERS
    )
    assert [item["layer"] for item in candidate] == list(range(TOTAL_LAYERS))
    return control, candidate


def fill_dense_layer(
    destination: torch.Tensor,
    scratch: torch.Tensor,
    generator: torch.Generator,
) -> None:
    scratch.uniform_(-0.125, 0.125, generator=generator)
    destination.copy_(scratch)


def reconstruct_tq_layer(
    destination: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    rope: torch.Tensor,
    centroids: torch.Tensor,
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
        decoded = centroids[indices.long()] * scale[first:last, ..., None].float()
        destination[first:last, ..., :LATENT].copy_(decoded)
        destination[first:last, ..., LATENT:].copy_(rope[first:last])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(10219, 37932), required=True)
    parser.add_argument("--split-kv", type=int, choices=(40, 64), required=True)
    parser.add_argument(
        "--allocation-order",
        choices=("control-first", "candidate-first"),
        required=True,
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--correctness-atol", type=float, default=0.002)
    parser.add_argument("--graph-debug-dir", type=Path)
    args = parser.parse_args()

    expected_split = {10219: 64, 37932: 40}[args.context]
    if args.split_kv != expected_split:
        raise ValueError(
            f"context {args.context} requires frozen split-kv {expected_split}"
        )
    if args.warmups < 100:
        raise ValueError("H41 I1 requires at least 100 graph warmups per arm")
    if args.samples * args.replays_per_sample < 2000:
        raise ValueError("H41 I1 requires at least 2,000 timed replays per arm")

    idle_proof = assert_idle_before_cuda()
    import flashinfer.rope

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H41 I1 requires SM100")
    if not is_arch_support_pdl():
        raise RuntimeError("H41 I1 requires the SM100 PDL path")
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    if free_before < (40 << 30):
        raise RuntimeError(
            f"H41 I1 requires 40 GiB free; found {free_before / 2**30:.2f} GiB"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    page_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    control_trace, candidate_trace = operation_traces()

    def allocate_control() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(
                TOTAL_LAYERS,
                CACHE_PAGES,
                PAGE,
                LATENT + ROPE,
                dtype=FP8,
                device=device,
            ),
            torch.empty(64 << 20, dtype=torch.int8, device=device),
        )

    def allocate_candidate() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(
                DENSE_LAYERS,
                CACHE_PAGES,
                PAGE,
                LATENT + ROPE,
                dtype=FP8,
                device=device,
            ),
            torch.empty(
                SELECTED_LAYERS,
                CACHE_PAGES,
                PAGE,
                LATENT // 2,
                dtype=torch.uint8,
                device=device,
            ),
            torch.empty(
                SELECTED_LAYERS,
                CACHE_PAGES,
                PAGE,
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.empty(
                SELECTED_LAYERS,
                CACHE_PAGES,
                PAGE,
                ROPE,
                dtype=FP8,
                device=device,
            ),
            torch.empty(64 << 20, dtype=torch.int8, device=device),
        )

    if args.allocation_order == "control-first":
        control_dense, control_workspace = allocate_control()
        candidate_allocations = allocate_candidate()
    else:
        candidate_allocations = allocate_candidate()
        control_dense, control_workspace = allocate_control()
    (
        candidate_dense,
        candidate_packed,
        candidate_scale,
        candidate_rope,
        candidate_workspace,
    ) = candidate_allocations

    centroids = torch.tensor(E2M1_CENTROIDS, dtype=torch.float32, device=device)
    candidate_packed.random_(0, 256, generator=generator)
    candidate_scale.uniform_(0.05, 0.20, generator=generator)
    rope_scratch = torch.empty(
        CACHE_PAGES, PAGE, ROPE, dtype=torch.bfloat16, device=device
    )
    dense_scratch = torch.empty(
        CACHE_PAGES, PAGE, LATENT + ROPE, dtype=torch.bfloat16, device=device
    )
    for layer in range(SELECTED_LAYERS):
        rope_scratch.normal_(0.0, 0.1, generator=generator)
        candidate_rope[layer].copy_(rope_scratch)
        reconstruct_tq_layer(
            control_dense[DENSE_BEFORE + layer],
            candidate_packed[layer],
            candidate_scale[layer],
            candidate_rope[layer],
            centroids,
        )
    for dense_index in range(DENSE_LAYERS):
        fill_dense_layer(candidate_dense[dense_index], dense_scratch, generator)
        control_index = (
            dense_index
            if dense_index < DENSE_BEFORE
            else dense_index + SELECTED_LAYERS
        )
        control_dense[control_index].copy_(candidate_dense[dense_index])
    del dense_scratch, rope_scratch
    torch.cuda.synchronize()

    page_table = torch.randperm(
        REQUEST_PAGES,
        dtype=torch.int32,
        device=device,
        generator=page_generator,
    ).unsqueeze(0)
    seq_lens = torch.tensor([args.context], dtype=torch.int32, device=device)
    logical_positions = torch.arange(
        args.context - TOKENS,
        args.context,
        dtype=torch.int64,
        device=device,
    )
    logical_pages = torch.div(logical_positions, PAGE, rounding_mode="floor")
    locations = (
        page_table[0, logical_pages].to(torch.int64) * PAGE
        + torch.remainder(logical_positions, PAGE)
    ).contiguous()

    def make_bf16(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.bfloat16, device=device).uniform_(
            -0.125, 0.125, generator=generator
        )

    query_latent = make_bf16(TOKENS, HEADS, LATENT)
    query_rope_raw = make_bf16(TOKENS, HEADS, ROPE)
    cache_latent = make_bf16(TOKENS, 1, LATENT)
    cache_rope_raw = make_bf16(TOKENS, 1, ROPE)
    cos_sin_cache = torch.empty(
        MAX_CONTEXT, ROPE, dtype=torch.float32, device=device
    ).uniform_(-1.0, 1.0, generator=generator)

    control_query = torch.empty(
        TOKENS, HEADS, LATENT + ROPE, dtype=FP8, device=device
    )
    control_k = torch.empty(TOKENS, LATENT, dtype=FP8, device=device)
    control_rope = torch.empty(TOKENS, ROPE, dtype=FP8, device=device)
    candidate_query = torch.empty_like(control_query)
    candidate_k = torch.empty_like(control_k)
    candidate_dense_rope = torch.empty_like(control_rope)
    selected_query_rope = torch.empty_like(query_rope_raw)
    selected_cache_rope = torch.empty_like(cache_rope_raw)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    control_out = torch.empty(
        1, TOKENS, HEADS, LATENT, dtype=torch.bfloat16, device=device
    )
    candidate_out = torch.empty_like(control_out)
    check_dense_out = torch.empty_like(control_out)
    check_tq_out = torch.empty_like(control_out)
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

    def dense_attention(
        cache: torch.Tensor,
        query: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        return tokenspeed_mla_decode(
            query=query.view(1, TOKENS, HEADS, LATENT + ROPE),
            kv_cache=cache,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=MAX_CONTEXT,
            softmax_scale=softmax_scale,
            out=output,
            causal_mask=True,
            enable_pdl=True,
        )

    def tq_attention(layer: int, query: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return tokenspeed_mla_decode_tq4(
            query=query.view(1, TOKENS, HEADS, LATENT + ROPE),
            kv_nope_packed=candidate_packed[layer],
            kv_nope_scale=candidate_scale[layer],
            kv_rope=candidate_rope[layer],
            centroids=centroids,
            workspace_buffer=candidate_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=MAX_CONTEXT,
            softmax_scale=softmax_scale,
            out=output,
            causal_mask=True,
            enable_pdl=True,
            split_kv_override=args.split_kv,
            kv_nope_codebook=None,
            fp8_rope=True,
        )

    def normal_frontend(
        cache: torch.Tensor,
        query_latent_input: torch.Tensor,
        query_out: torch.Tensor,
        k_out: torch.Tensor,
        rope_out: torch.Tensor,
    ) -> None:
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=query_rope_raw,
            k_rope=cache_rope_raw[:, 0],
            q_nope=query_latent_input,
            k_nope=cache_latent[:, 0],
            cos_sin_cache=cos_sin_cache,
            pos_ids=logical_positions,
            is_neox=True,
            quantize_dtype=FP8,
            q_rope_out=query_out[..., LATENT:],
            k_rope_out=rope_out,
            q_nope_out=query_out[..., :LATENT],
            k_nope_out=k_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=True,
        )
        set_mla_kv_buffer_triton(
            cache.view(CACHE_ROWS, -1), locations, k_out, rope_out
        )

    def selected_frontend(layer: int, query_latent_input: torch.Tensor) -> None:
        flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
            q=query_rope_raw,
            k=cache_rope_raw,
            q_rope=selected_query_rope,
            k_rope=selected_cache_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=logical_positions,
            interleave=False,
        )
        tq_mla_frontend_out(
            query_latent_input,
            selected_query_rope,
            cache_latent,
            selected_cache_rope,
            locations,
            config.signs1,
            config.signs2,
            config.k_boundaries,
            config.k_quant_centroids,
            config.k_storage_code_lut,
            candidate_query,
            candidate_packed[layer].view(CACHE_ROWS, 1, LATENT // 2),
            candidate_scale[layer].view(CACHE_ROWS, 1),
            candidate_rope[layer].view(CACHE_ROWS, 1, ROPE),
            status,
            scale_multiplier=config.k_dequant_scale_multiplier,
            rotation_fused=True,
            num_warps=8,
            strict=False,
        )

    def reconstruct_current_rows(layer: int, control_layer: int) -> None:
        packed_rows = candidate_packed[layer].view(CACHE_ROWS, LATENT // 2)[
            locations
        ]
        indices = torch.empty(
            TOKENS, LATENT, dtype=torch.uint8, device=device
        )
        indices[:, 0::2] = packed_rows & 0x0F
        indices[:, 1::2] = packed_rows >> 4
        scale_rows = candidate_scale[layer].view(CACHE_ROWS)[locations]
        decoded = (centroids[indices.long()] * scale_rows[:, None].float()).to(FP8)
        dense_rows = control_dense[control_layer].view(
            CACHE_ROWS, LATENT + ROPE
        )
        dense_rows[locations, :LATENT] = decoded
        dense_rows[locations, LATENT:] = candidate_rope[layer].view(
            CACHE_ROWS, ROPE
        )[locations]

    check_query = make_bf16(1, TOKENS, HEADS, LATENT + ROPE).to(FP8)
    dense_attention(
        control_dense[DENSE_BEFORE],
        check_query.view(TOKENS, HEADS, LATENT + ROPE),
        check_dense_out,
        control_workspace,
    )
    tq_attention(
        0,
        check_query.view(TOKENS, HEADS, LATENT + ROPE),
        check_tq_out,
    )
    torch.cuda.synchronize()
    max_abs_diff = float(
        (check_dense_out.float() - check_tq_out.float()).abs().max().item()
    )
    torch.testing.assert_close(
        check_tq_out,
        check_dense_out,
        rtol=0,
        atol=args.correctness_atol,
    )

    selected_frontend(0, query_latent)
    reconstruct_current_rows(0, DENSE_BEFORE)
    dense_attention(
        control_dense[DENSE_BEFORE],
        candidate_query,
        check_dense_out,
        control_workspace,
    )
    tq_attention(0, candidate_query, check_tq_out)
    torch.cuda.synchronize()
    writer_reader_max_abs_diff = float(
        (check_dense_out.float() - check_tq_out.float()).abs().max().item()
    )
    torch.testing.assert_close(
        check_tq_out,
        check_dense_out,
        rtol=0,
        atol=args.correctness_atol,
    )
    if int(status.item()) != 0:
        raise AssertionError("writer-reader round trip changed sticky status")

    def run_control() -> torch.Tensor:
        result = control_out
        layer_query_latent = query_latent
        for layer in range(TOTAL_LAYERS):
            normal_frontend(
                control_dense[layer],
                layer_query_latent,
                control_query,
                control_k,
                control_rope,
            )
            result = dense_attention(
                control_dense[layer],
                control_query,
                control_out,
                control_workspace,
            )
            layer_query_latent = control_out.view(TOKENS, HEADS, LATENT)
        return result

    def run_candidate() -> torch.Tensor:
        result = candidate_out
        layer_query_latent = query_latent
        for dense_index in range(DENSE_BEFORE):
            normal_frontend(
                candidate_dense[dense_index],
                layer_query_latent,
                candidate_query,
                candidate_k,
                candidate_dense_rope,
            )
            result = dense_attention(
                candidate_dense[dense_index],
                candidate_query,
                candidate_out,
                candidate_workspace,
            )
            layer_query_latent = candidate_out.view(TOKENS, HEADS, LATENT)
        for selected_index in range(SELECTED_LAYERS):
            selected_frontend(selected_index, layer_query_latent)
            result = tq_attention(selected_index, candidate_query, candidate_out)
            layer_query_latent = candidate_out.view(TOKENS, HEADS, LATENT)
        for dense_index in range(DENSE_BEFORE, DENSE_LAYERS):
            normal_frontend(
                candidate_dense[dense_index],
                layer_query_latent,
                candidate_query,
                candidate_k,
                candidate_dense_rope,
            )
            result = dense_attention(
                candidate_dense[dense_index],
                candidate_query,
                candidate_out,
                candidate_workspace,
            )
            layer_query_latent = candidate_out.view(TOKENS, HEADS, LATENT)
        return result

    run_control()
    run_candidate()
    torch.cuda.synchronize()
    if int(status.item()) != 0:
        raise AssertionError("candidate sticky status changed during valid eager run")
    if not bool(torch.isfinite(control_out.float()).all()):
        raise AssertionError("control output contains a non-finite value")
    if not bool(torch.isfinite(candidate_out.float()).all()):
        raise AssertionError("candidate output contains a non-finite value")

    debug_control = (
        args.graph_debug_dir / "control.dot"
        if args.graph_debug_dir is not None
        else None
    )
    debug_candidate = (
        args.graph_debug_dir / "candidate.dot"
        if args.graph_debug_dir is not None
        else None
    )
    control_graph = capture_graph(
        run_control, warmups=args.warmups, debug_path=debug_control
    )
    candidate_graph = capture_graph(
        run_candidate, warmups=args.warmups, debug_path=debug_candidate
    )
    allocation_before_replay = torch.cuda.memory_allocated(device)
    for _ in range(100):
        control_graph.replay()
        candidate_graph.replay()
    torch.cuda.synchronize()
    allocation_after_replay = torch.cuda.memory_allocated(device)
    if allocation_after_replay != allocation_before_replay:
        raise AssertionError(
            "graph replay allocated memory: "
            f"{allocation_before_replay} -> {allocation_after_replay}"
        )

    sequence: list[dict[str, Any]] = []
    for kind, graph, output, graph_status in (
        ("control", control_graph, control_out, None),
        ("candidate", candidate_graph, candidate_out, status),
        ("control", control_graph, control_out, None),
    ):
        control_number = sum(item["kind"] == "control" for item in sequence) + 1
        arm = f"control_{control_number}" if kind == "control" else "candidate"
        sequence.append(
            {
                "arm": arm,
                "kind": kind,
                **time_graph(
                    graph,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                    output=output,
                    status=graph_status,
                ),
            }
        )

    control_1 = sequence[0]["graph_us"]["mean"]
    candidate_us = sequence[1]["graph_us"]["mean"]
    control_2 = sequence[2]["graph_us"]["mean"]
    control_mean = (control_1 + control_2) / 2.0
    graph_delta = candidate_us - control_mean
    control_persistent_cache_bytes = control_dense.nbytes
    candidate_persistent_cache_bytes = (
        candidate_dense.nbytes
        + candidate_packed.nbytes
        + candidate_scale.nbytes
        + candidate_rope.nbytes
    )
    gross_persistent_cache_savings_bytes = (
        control_persistent_cache_bytes - candidate_persistent_cache_bytes
    )

    result = {
        "status": "TIMING_ONLY",
        "experiment": "H41_I1_INTEGRATED_FOUR_FAMILY",
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "context": args.context,
        "max_context": MAX_CONTEXT,
        "cache_rows": CACHE_ROWS,
        "page_size": PAGE,
        "batch": 1,
        "q_len": TOKENS,
        "heads": HEADS,
        "total_layers": TOTAL_LAYERS,
        "selected_layers": SELECTED_LAYERS,
        "selected_layer_ids": list(
            range(DENSE_BEFORE, DENSE_BEFORE + SELECTED_LAYERS)
        ),
        "dense_before": DENSE_BEFORE,
        "dense_after": DENSE_AFTER,
        "selected_warps": 8,
        "selected_row_bytes": 322,
        "selected_fp8_rope": True,
        "selected_codebook_materialized": False,
        "split_kv": args.split_kv,
        "allocation_order": args.allocation_order,
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "control_mean_graph_us": control_mean,
        "candidate_mean_graph_us": candidate_us,
        "candidate_minus_control_graph_us": graph_delta,
        "integrated_delta_per_selected_layer_us": graph_delta / SELECTED_LAYERS,
        "control_flank_drift_fraction": (control_2 - control_1) / control_mean,
        "operation_traces": {
            "control": control_trace,
            "candidate": candidate_trace,
        },
        "logical_operation_counts": {
            "control_dense_layers": TOTAL_LAYERS,
            "candidate_dense_layers": DENSE_LAYERS,
            "candidate_selected_layers": SELECTED_LAYERS,
            "operations_per_layer": 3,
        },
        "inter_layer_dependency": "attention_output_to_next_query_latent",
        "graph_debug_dir": (
            str(args.graph_debug_dir) if args.graph_debug_dir is not None else None
        ),
        "graph_replay_allocation_bytes": {
            "before": allocation_before_replay,
            "after": allocation_after_replay,
        },
        "one_selected_layer_max_abs_diff": max_abs_diff,
        "writer_reader_max_abs_diff": writer_reader_max_abs_diff,
        "correctness_atol": args.correctness_atol,
        "current_locations": locations.tolist(),
        "idle_proof": idle_proof,
        "free_bytes_before_allocations": free_before,
        "free_bytes_after_allocations": torch.cuda.mem_get_info(device)[0],
        "total_device_bytes": total_bytes,
        "control_dense_bytes": control_dense.nbytes,
        "candidate_dense_bytes": candidate_dense.nbytes,
        "candidate_tq_packed_bytes": candidate_packed.nbytes,
        "candidate_tq_scale_bytes": candidate_scale.nbytes,
        "candidate_tq_rope_bytes": candidate_rope.nbytes,
        "candidate_tq_codebook_bytes": 0,
        "control_persistent_cache_bytes": control_persistent_cache_bytes,
        "candidate_persistent_cache_bytes": candidate_persistent_cache_bytes,
        "gross_persistent_cache_savings_bytes": gross_persistent_cache_savings_bytes,
        "gross_persistent_cache_savings_gib": (
            gross_persistent_cache_savings_bytes / 2**30
        ),
        "seed": args.seed,
        "interpretation": (
            "Complete isolated 61-layer graph timing; not endpoint performance, "
            "semantic quality, production, or promotion evidence."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
