"""Paired W0/R3B SM100 fused-RoPE front-end timing harness.

The control is the v0.5.17 FlashInfer RoPE+FP8 query launch plus generic dense
FP8 scatter. W0 is the accepted 57-register N10 writer, and R3B is its
split-entry 56-register residency candidate. This is an isolated front-end
falsifier, not endpoint evidence.
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
w0 = load("a17_n10_w0_tq_mla_frontend", "/tmp/n10/tq_mla_frontend_n10.py")
r3b = load(
    "a17_n10_w1_r3b_tq_mla_frontend",
    "/tmp/n10r3boverlay/sglang/kernels/jit/tq_mla_frontend_n10.py",
)
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
        choices=("w0-first", "r3b-first"),
        default="w0-first",
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

    def allocate_candidate() -> tuple[torch.Tensor, ...]:
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
        if args.candidate_order == "w0-first":
            return allocate_candidate(), allocate_candidate()
        r3b_buffers = allocate_candidate()
        w0_buffers = allocate_candidate()
        return w0_buffers, r3b_buffers

    if args.allocation_order == "control-first":
        control = allocate_control()
        w0_buffers, r3b_buffers = allocate_candidates()
    else:
        w0_buffers, r3b_buffers = allocate_candidates()
        control = allocate_control()
    control_query, control_k, control_rope, control_cache = control
    (
        w0_query_latent,
        w0_query_rope,
        w0_packed,
        w0_scale,
        w0_rope,
        w0_status,
        w0_zero_count,
    ) = w0_buffers
    (
        r3b_query_latent,
        r3b_query_rope,
        r3b_packed,
        r3b_scale,
        r3b_rope,
        r3b_status,
        r3b_zero_count,
    ) = r3b_buffers

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

    def run_w0() -> None:
        w0.tq_mla_n10_frontend_rope_out(
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
            w0_query_latent,
            w0_query_rope,
            w0_packed,
            w0_scale,
            w0_rope,
            w0_status,
            w0_zero_count,
            grid=config.grid,
            rotation_fused=True,
            strict=False,
        )

    def run_r3b() -> None:
        r3b.tq_mla_n10_frontend_rope_out(
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
            r3b_query_latent,
            r3b_query_rope,
            r3b_packed,
            r3b_scale,
            r3b_rope,
            r3b_status,
            r3b_zero_count,
            grid=config.grid,
            rotation_fused=True,
            strict=False,
        )

    run_control()
    run_w0()
    run_r3b()
    torch.cuda.synchronize()
    expected_control = torch.cat((control_k, control_rope), dim=-1).view(
        torch.uint8
    )
    torch.testing.assert_close(
        control_cache[locations], expected_control, rtol=0, atol=0
    )
    if int(w0_status.item()) != 0 or int(r3b_status.item()) != 0:
        raise AssertionError("a candidate set the fault word for valid inputs")
    if int(w0_zero_count.item()) != 0 or int(r3b_zero_count.item()) != 0:
        raise AssertionError("random nonzero rows incremented a zero counter")
    if not torch.equal(
        w0_query_latent.view(torch.uint8), r3b_query_latent.view(torch.uint8)
    ):
        raise AssertionError("R3B query latent differs from W0")
    if not torch.equal(
        w0_query_rope.view(torch.int16), r3b_query_rope.view(torch.int16)
    ):
        raise AssertionError("R3B query RoPE differs from W0")
    if not torch.equal(w0_packed[locations], r3b_packed[locations]):
        raise AssertionError("R3B packed latent differs from W0")
    if not torch.equal(
        w0_scale[locations].view(torch.int16),
        r3b_scale[locations].view(torch.int16),
    ):
        raise AssertionError("R3B scale differs from W0")
    if not torch.equal(
        w0_rope[locations].view(torch.int16),
        r3b_rope[locations].view(torch.int16),
    ):
        raise AssertionError("R3B reciprocal RoPE differs from W0")

    graphs = {
        "control": capture_graph(run_control, args.warmups),
        "w0": capture_graph(run_w0, args.warmups),
        "r3b": capture_graph(run_r3b, args.warmups),
    }
    checksum_slot = int(locations[0].item())
    checksums = {
        "control": (control_query, control_cache[checksum_slot, :1]),
        "w0": (
            w0_query_latent,
            w0_query_rope,
            w0_packed[checksum_slot, :, :1],
            w0_scale[checksum_slot],
            w0_rope[checksum_slot, :, :1],
            w0_status,
            w0_zero_count,
        ),
        "r3b": (
            r3b_query_latent,
            r3b_query_rope,
            r3b_packed[checksum_slot, :, :1],
            r3b_scale[checksum_slot],
            r3b_rope[checksum_slot, :, :1],
            r3b_status,
            r3b_zero_count,
        ),
    }
    sequence: list[dict[str, Any]] = []
    candidate_arms = (
        ("w0", "r3b")
        if args.candidate_order == "w0-first"
        else ("r3b", "w0")
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
    w0_us = next(
        item["timing_us"]["mean"] for item in sequence if item["kind"] == "w0"
    )
    r3b_us = next(
        item["timing_us"]["mean"] for item in sequence if item["kind"] == "r3b"
    )
    control_mean = (control_1 + control_2) / 2.0
    active_slot = checksum_slot
    address_period = 2**21
    result = {
        "status": "TIMING_ONLY",
        "experiment": "A17_N10_W1_R3B_SPLIT_ENTRY_RESIDENCY",
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
        "w0_physical_launches": 1,
        "r3b_physical_launches": 1,
        "control_physical_launches": 2,
        "allocation_order": args.allocation_order,
        "candidate_order": args.candidate_order,
        "pool_size": POOL_SIZE,
        "w0_row_bytes": 386,
        "r3b_row_bytes": 386,
        "buffer_address_mod_2m": {
            "control_query": control_query.data_ptr() % address_period,
            "control_cache": control_cache.data_ptr() % address_period,
            "control_cache_active": (
                control_cache.data_ptr() + active_slot * (LATENT + ROPE)
            )
            % address_period,
            "w0_query_latent": w0_query_latent.data_ptr() % address_period,
            "w0_query_rope": w0_query_rope.data_ptr() % address_period,
            "w0_packed": w0_packed.data_ptr() % address_period,
            "w0_packed_active": (
                w0_packed.data_ptr() + active_slot * (LATENT // 2)
            )
            % address_period,
            "r3b_query_latent": r3b_query_latent.data_ptr() % address_period,
            "r3b_query_rope": r3b_query_rope.data_ptr() % address_period,
            "r3b_packed": r3b_packed.data_ptr() % address_period,
            "r3b_packed_active": (
                r3b_packed.data_ptr() + active_slot * (LATENT // 2)
            )
            % address_period,
        },
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "control_mean_us": control_mean,
        "control_flank_drift_fraction": (control_2 - control_1) / control_mean,
        "w0_mean_us": w0_us,
        "w0_minus_control_us": w0_us - control_mean,
        "w0_over_control": w0_us / control_mean,
        "r3b_mean_us": r3b_us,
        "r3b_minus_control_us": r3b_us - control_mean,
        "r3b_over_control": r3b_us / control_mean,
        "r3b_minus_w0_us": r3b_us - w0_us,
        "r3b_over_w0": r3b_us / w0_us,
        "seed": args.seed,
        "interpretation": (
            "W0 and R3B versus the v0.5.17 normal FP8 front-end control. Each "
            "candidate is one fused NeoX BF16-RoPE, query, and packed-cache "
            "launch. R3B differs from W0 only by split kernel entry points and "
            "a fused-only launch bound; isolated timing only, not endpoint evidence."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
