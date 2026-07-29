"""Paired H41 W2 SM100 combined-front-end timing harness.

The candidate is the real selected-layer query+writer kernel. The control is
the same one-launch FlashInfer FP8 query plus dense FP8 scatter writer used by
H41 W1. This is an isolated front-end falsifier, not endpoint evidence.
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
from typing import Any

import torch

from sglang.jit_kernel.tq_mla_frontend import tq_mla_frontend_out
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
from sglang.srt.mem_cache.triton_ops.mla_buffer import set_mla_kv_buffer_triton

LATENT = 512
ROPE = 64
HEADS = 8
MAX_CONTEXT = 256000
POOL_SIZE = MAX_CONTEXT + 32
FP8 = torch.float8_e4m3fn


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


def gpu_covariates() -> dict[str, Any]:
    fields = (
        "timestamp,index,pstate,clocks.current.graphics,clocks.max.graphics,"
        "temperature.gpu,power.draw,utilization.gpu,memory.used,"
        "ecc.errors.uncorrected.volatile.total,gpu_recovery_action"
    )
    command = [
        "nvidia-smi",
        "--id=0",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"error": repr(error), "command": command}
    names = fields.split(",")
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    return dict(zip(names, values, strict=False))


def capture_graph(fn: Callable[[], None], warmups: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    samples: int,
    replays_per_sample: int,
    checksum_tensors: tuple[torch.Tensor, ...],
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
    checksum = sum(float(tensor.float().sum()) for tensor in checksum_tensors)
    return {
        "sample_us": sample_us,
        "timing_us": summarize(sample_us),
        "replays": samples * replays_per_sample,
        "gpu_before": before,
        "gpu_after": after,
        "output_checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(10219, 37932), required=True)
    parser.add_argument(
        "--tokens", type=int, choices=(1, 2, 5, 7, 10, 15, 20, 25, 30, 35, 40), required=True
    )
    parser.add_argument("--warps", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument(
        "--allocation-order",
        choices=("control-first", "candidate-first"),
        default="control-first",
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.warmups < 100:
        raise ValueError("H41 W2 requires at least 100 graph warmups per arm")
    if args.samples * args.replays_per_sample < 2000:
        raise ValueError("H41 W2 requires at least 2,000 timed replays per arm")

    import flashinfer.rope

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H41 W2 timing requires SM100")
    if not is_arch_support_pdl():
        raise RuntimeError("H41 W2 control requires the SM100 PDL path")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def make_input(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.bfloat16, device=device).uniform_(
            -0.125, 0.125, generator=generator
        )

    query_latent = make_input(args.tokens, HEADS, LATENT)
    query_rope = make_input(args.tokens, HEADS, ROPE)
    cache_latent_2d = make_input(args.tokens, LATENT)
    cache_rope_2d = make_input(args.tokens, ROPE)
    cache_latent = cache_latent_2d.unsqueeze(1)
    cache_rope = cache_rope_2d.unsqueeze(1)
    cos_sin_cache = torch.empty(
        MAX_CONTEXT, ROPE, dtype=torch.float32, device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    positions = torch.arange(
        args.context - args.tokens,
        args.context,
        dtype=torch.int64,
        device=device,
    )
    stride = MAX_CONTEXT // args.tokens
    locations = torch.arange(args.tokens, dtype=torch.int64, device=device) * stride + 17
    permutation = torch.randperm(args.tokens, device=device)
    locations = locations[permutation].contiguous()

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

    def allocate_control() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(args.tokens, HEADS, LATENT + ROPE, dtype=FP8, device=device),
            torch.empty_like(cache_latent_2d, dtype=FP8),
            torch.empty_like(cache_rope_2d, dtype=FP8),
            torch.empty(POOL_SIZE, LATENT + ROPE, dtype=FP8, device=device),
        )

    def allocate_candidate() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(args.tokens, HEADS, LATENT + ROPE, dtype=FP8, device=device),
            torch.empty(POOL_SIZE, 1, LATENT // 2, dtype=torch.uint8, device=device),
            torch.empty(POOL_SIZE, 1, dtype=torch.bfloat16, device=device),
            torch.empty(POOL_SIZE, 1, ROPE, dtype=FP8, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    if args.allocation_order == "control-first":
        control = allocate_control()
        candidate = allocate_candidate()
    else:
        candidate = allocate_candidate()
        control = allocate_control()
    control_query, control_k, control_rope, control_cache = control
    candidate_query, candidate_packed, candidate_scale, candidate_rope, status = candidate

    def run_control() -> None:
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=query_rope,
            k_rope=cache_rope_2d,
            q_nope=query_latent,
            k_nope=cache_latent_2d,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=True,
            quantize_dtype=FP8,
            q_rope_out=control_query[..., LATENT:],
            k_rope_out=control_rope,
            q_nope_out=control_query[..., :LATENT],
            k_nope_out=control_k,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=True,
        )
        set_mla_kv_buffer_triton(
            control_cache, locations, control_k, control_rope
        )

    def run_candidate() -> None:
        tq_mla_frontend_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            locations,
            config.signs1,
            config.signs2,
            config.k_boundaries,
            config.k_quant_centroids,
            config.k_storage_code_lut,
            candidate_query,
            candidate_packed,
            candidate_scale,
            candidate_rope,
            status,
            scale_multiplier=config.k_dequant_scale_multiplier,
            rotation_fused=True,
            num_warps=args.warps,
            strict=False,
        )

    run_control()
    run_candidate()
    torch.cuda.synchronize()
    expected_control = torch.cat((control_k, control_rope), dim=-1)
    torch.testing.assert_close(
        control_cache[locations], expected_control, rtol=0, atol=0
    )
    if int(status.item()) != 0:
        raise AssertionError("candidate set the fault word for valid locations")
    if not bool(torch.isfinite(candidate_query.float()).all()):
        raise AssertionError("candidate query contains a non-finite value")

    graphs = {
        "control": capture_graph(run_control, args.warmups),
        "candidate": capture_graph(run_candidate, args.warmups),
    }
    checksums = {
        "control": (control_query, control_cache[locations[0], :1]),
        "candidate": (
            candidate_query,
            candidate_packed[locations[0], :, :1],
            candidate_scale[locations[0]],
            candidate_rope[locations[0], :, :1],
            status,
        ),
    }
    sequence: list[dict[str, Any]] = []
    for arm in ("control", "candidate", "control"):
        control_index = sum(item["kind"] == "control" for item in sequence) + 1
        label = f"control_{control_index}" if arm == "control" else "candidate"
        sequence.append(
            {
                "arm": label,
                "kind": arm,
                **time_graph(
                    graphs[arm],
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                    checksum_tensors=checksums[arm],
                ),
            }
        )

    control_1 = sequence[0]["timing_us"]["mean"]
    candidate_us = sequence[1]["timing_us"]["mean"]
    control_2 = sequence[2]["timing_us"]["mean"]
    control_mean = (control_1 + control_2) / 2.0
    result = {
        "status": "TIMING_ONLY",
        "experiment": "H41_W2_REAL_COMBINED_FRONTEND",
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "context": args.context,
        "tokens": args.tokens,
        "heads": HEADS,
        "warps": args.warps,
        "rotation_fused": True,
        "allocation_order": args.allocation_order,
        "pool_size": POOL_SIZE,
        "candidate_row_bytes": 322,
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "control_mean_us": control_mean,
        "control_flank_drift_fraction": (control_2 - control_1) / control_mean,
        "candidate_mean_us": candidate_us,
        "candidate_minus_control_us": candidate_us - control_mean,
        "candidate_over_control": candidate_us / control_mean,
        "seed": args.seed,
        "interpretation": (
            "Real candidate versus the H41 W1 normal FP8 front-end control; "
            "isolated timing only, not endpoint performance or quality evidence."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
