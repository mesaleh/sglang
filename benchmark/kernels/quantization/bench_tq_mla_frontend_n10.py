"""Paired W5/N10 SM100 fused-RoPE front-end timing harness.

The control is the v0.5.17 FlashInfer RoPE+FP8 query launch plus generic dense
FP8 scatter. W5 is the accepted 322-byte packed writer, and N10 is the
386-byte reciprocal-RoPE writer. This is an isolated front-end falsifier, not
endpoint evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kv_tq = load(
    "sglang.srt.layers.quantization.kv_turboquant",
    "/tmp/s5w3/kv_turboquant.py",
)
w5 = load("s5w5_tq_mla_frontend", "/tmp/s5w5/tq_mla_frontend.py")
n10 = load("a17_n10_tq_mla_frontend", "/tmp/n10/tq_mla_frontend_n10.py")
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

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
        "timestamp,index,pstate,clocks.current.graphics,clocks.current.sm,"
        "clocks.current.memory,clocks.max.graphics,clocks.max.sm,"
        "clocks.max.memory,clocks.applications.graphics,"
        "clocks.applications.memory,"
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
        "--tokens",
        type=int,
        choices=(1, 2, 5, 7, 10, 15, 20, 25, 30, 35, 40, 4096, 10219),
        required=True,
    )
    parser.add_argument(
        "--allocation-order",
        choices=("control-first", "candidates-first"),
        default="control-first",
    )
    parser.add_argument(
        "--candidate-order",
        choices=("w5-first", "n10-first"),
        default="w5-first",
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.warmups < 100:
        raise ValueError("N10 requires at least 100 graph warmups per arm")
    minimum_replays = 200 if args.tokens >= 4096 else 2000
    if args.samples * args.replays_per_sample < minimum_replays:
        raise ValueError(
            f"N10 requires at least {minimum_replays:,} timed replays per arm"
        )

    import flashinfer.rope

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("N10 timing requires SM100")
    if not is_arch_support_pdl():
        raise RuntimeError("N10 control requires the SM100 PDL path")
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

    config = kv_tq.NativeE2M1MLAConfig(device=device)

    def allocate_control() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(args.tokens, HEADS, LATENT + ROPE, dtype=FP8, device=device),
            torch.empty_like(cache_latent_2d, dtype=FP8),
            torch.empty_like(cache_rope_2d, dtype=FP8),
            # MLATokenToKVPool stores FP8 as uint8 and passes prequantized
            # FP8 inputs to the generic scatter as byte views.
            torch.empty(POOL_SIZE, LATENT + ROPE, dtype=torch.uint8, device=device),
        )

    def allocate_w5() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(args.tokens, HEADS, LATENT + ROPE, dtype=FP8, device=device),
            torch.empty(POOL_SIZE, 1, LATENT // 2, dtype=torch.uint8, device=device),
            torch.empty(POOL_SIZE, 1, dtype=torch.bfloat16, device=device),
            torch.empty(POOL_SIZE, 1, ROPE, dtype=FP8, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    def allocate_n10() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty(args.tokens, HEADS, LATENT, dtype=FP8, device=device),
            torch.empty_like(query_rope),
            torch.empty(POOL_SIZE, 1, LATENT // 2, dtype=torch.uint8, device=device),
            torch.empty(POOL_SIZE, 1, dtype=torch.bfloat16, device=device),
            torch.empty(POOL_SIZE, 1, ROPE, dtype=torch.bfloat16, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )

    def allocate_candidates() -> tuple[tuple[torch.Tensor, ...], ...]:
        if args.candidate_order == "w5-first":
            return allocate_w5(), allocate_n10()
        n10_buffers = allocate_n10()
        w5_buffers = allocate_w5()
        return w5_buffers, n10_buffers

    if args.allocation_order == "control-first":
        control = allocate_control()
        w5_buffers, n10_buffers = allocate_candidates()
    else:
        w5_buffers, n10_buffers = allocate_candidates()
        control = allocate_control()
    control_query, control_k, control_rope, control_cache = control
    (
        w5_query,
        w5_packed,
        w5_scale,
        w5_rope,
        w5_status,
    ) = w5_buffers
    (
        n10_query_latent,
        n10_query_rope,
        n10_packed,
        n10_scale,
        n10_rope,
        n10_status,
        n10_zero_count,
    ) = n10_buffers
    oracle_query_rope = torch.empty_like(query_rope)
    oracle_cache_rope = torch.empty_like(cache_rope)

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
            control_cache,
            locations,
            control_k.view(torch.uint8),
            control_rope.view(torch.uint8),
        )

    def run_w5() -> None:
        w5.tq_mla_frontend_rope_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            cos_sin_cache,
            positions,
            locations,
            config.signs1,
            config.signs2,
            config.boundaries,
            config.levels,
            config.codes,
            w5_query,
            w5_packed,
            w5_scale,
            w5_rope,
            w5_status,
            grid=config.grid,
            rotation_fused=True,
            num_warps=9,
            strict=False,
        )

    def run_n10() -> None:
        n10.tq_mla_n10_frontend_rope_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            cos_sin_cache,
            positions,
            locations,
            config.signs1,
            config.signs2,
            config.boundaries,
            config.levels,
            config.codes,
            n10_query_latent,
            n10_query_rope,
            n10_packed,
            n10_scale,
            n10_rope,
            n10_status,
            n10_zero_count,
            grid=config.grid,
            rotation_fused=True,
            strict=False,
        )

    run_control()
    run_w5()
    run_n10()
    torch.cuda.synchronize()
    expected_control = torch.cat((control_k, control_rope), dim=-1).view(
        torch.uint8
    )
    torch.testing.assert_close(
        control_cache[locations], expected_control, rtol=0, atol=0
    )
    if int(w5_status.item()) != 0 or int(n10_status.item()) != 0:
        raise AssertionError("a candidate set the fault word for valid inputs")
    if int(n10_zero_count.item()) != 0:
        raise AssertionError("random nonzero rows incremented the N10 zero counter")
    flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
        q=query_rope,
        k=cache_rope,
        q_rope=oracle_query_rope,
        k_rope=oracle_cache_rope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        interleave=False,
    )
    torch.cuda.synchronize()
    if not torch.equal(
        n10_query_latent.view(torch.uint8), w5_query[..., :LATENT].view(torch.uint8)
    ):
        raise AssertionError("N10 query latent differs from W5")
    if not torch.equal(
        n10_query_rope.view(torch.int16), oracle_query_rope.view(torch.int16)
    ):
        raise AssertionError("N10 query RoPE differs from the BF16 oracle")
    if not torch.equal(n10_packed[locations], w5_packed[locations]):
        raise AssertionError("N10 packed latent differs from W5")
    if not torch.equal(
        n10_scale[locations].view(torch.int16),
        w5_scale[locations].view(torch.int16),
    ):
        raise AssertionError("N10 scale differs from W5")
    reciprocal_oracle = (
        oracle_cache_rope.float() / n10_scale[locations].float().unsqueeze(-1)
    ).to(torch.bfloat16)
    if not torch.equal(
        n10_rope[locations].view(torch.int16), reciprocal_oracle.view(torch.int16)
    ):
        raise AssertionError("N10 reciprocal RoPE differs from the FP32 oracle")

    graphs = {
        "control": capture_graph(run_control, args.warmups),
        "w5": capture_graph(run_w5, args.warmups),
        "n10": capture_graph(run_n10, args.warmups),
    }
    checksums = {
        "control": (control_query, control_cache[locations[0], :1]),
        "w5": (
            w5_query,
            w5_packed[locations[0], :, :1],
            w5_scale[locations[0]],
            w5_rope[locations[0], :, :1],
            w5_status,
        ),
        "n10": (
            n10_query_latent,
            n10_query_rope,
            n10_packed[locations[0], :, :1],
            n10_scale[locations[0]],
            n10_rope[locations[0], :, :1],
            n10_status,
            n10_zero_count,
        ),
    }
    sequence: list[dict[str, Any]] = []
    candidate_arms = (
        ("w5", "n10") if args.candidate_order == "w5-first" else ("n10", "w5")
    )
    for arm in ("control", *candidate_arms, "control"):
        control_index = sum(item["kind"] == "control" for item in sequence) + 1
        label = f"control_{control_index}" if arm == "control" else arm
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
    control_2 = sequence[-1]["timing_us"]["mean"]
    w5_us = next(item["timing_us"]["mean"] for item in sequence if item["kind"] == "w5")
    n10_us = next(
        item["timing_us"]["mean"] for item in sequence if item["kind"] == "n10"
    )
    control_mean = (control_1 + control_2) / 2.0
    active_slot = int(locations[0].item())
    address_period = 2**21
    result = {
        "status": "TIMING_ONLY",
        "experiment": "A17_N10_W0_FUSED_ROPE_FRONTEND",
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "context": args.context,
        "tokens": args.tokens,
        "heads": HEADS,
        "warps": 9,
        "rotation_fused": True,
        "w5_physical_launches": 1,
        "n10_physical_launches": 1,
        "control_physical_launches": 2,
        "allocation_order": args.allocation_order,
        "candidate_order": args.candidate_order,
        "pool_size": POOL_SIZE,
        "w5_row_bytes": 322,
        "n10_row_bytes": 386,
        "buffer_address_mod_2m": {
            "control_query": control_query.data_ptr() % address_period,
            "control_cache": control_cache.data_ptr() % address_period,
            "control_cache_active": (
                control_cache.data_ptr() + active_slot * (LATENT + ROPE)
            )
            % address_period,
            "w5_query": w5_query.data_ptr() % address_period,
            "w5_packed": w5_packed.data_ptr() % address_period,
            "w5_packed_active": (
                w5_packed.data_ptr() + active_slot * (LATENT // 2)
            )
            % address_period,
            "n10_query_latent": n10_query_latent.data_ptr() % address_period,
            "n10_query_rope": n10_query_rope.data_ptr() % address_period,
            "n10_packed": n10_packed.data_ptr() % address_period,
            "n10_packed_active": (
                n10_packed.data_ptr() + active_slot * (LATENT // 2)
            )
            % address_period,
        },
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "control_mean_us": control_mean,
        "control_flank_drift_fraction": (control_2 - control_1) / control_mean,
        "w5_mean_us": w5_us,
        "w5_minus_control_us": w5_us - control_mean,
        "w5_over_control": w5_us / control_mean,
        "n10_mean_us": n10_us,
        "n10_minus_control_us": n10_us - control_mean,
        "n10_over_control": n10_us / control_mean,
        "n10_minus_w5_us": n10_us - w5_us,
        "n10_over_w5": n10_us / w5_us,
        "seed": args.seed,
        "interpretation": (
            "W5 and N10 versus the v0.5.17 normal FP8 front-end control. Each "
            "candidate is one fused NeoX BF16-RoPE, query, and packed-cache "
            "launch; isolated timing only, not endpoint evidence."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
