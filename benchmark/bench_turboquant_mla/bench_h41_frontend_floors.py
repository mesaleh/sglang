"""H41 W1 timing floors for the Kimi K2.6 MLA decode front end.

Measures the production normal-cache FP8 RoPE/quantize kernel, the production
MLA KV scatter writer, their ordered two-launch combination, and non-elided
one-/two-launch physical floors. All arms are captured once in CUDA graphs and
timed through raw fixed-replay samples. Results are microbenchmark evidence,
not endpoint performance or quality evidence.
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
import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.mem_cache.triton_ops.mla_buffer import set_mla_kv_buffer_triton

LATENT = 512
ROPE = 64
TOKENS = 5
HEADS = 8
MAX_CONTEXT = 256000


@triton.jit
def _non_elided_launch_floor(src, dst):
    value = tl.load(src)
    tl.store(dst, value + 1.0)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
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
        "timestamp,index,pstate,clocks.sm,clocks.mem,temperature.gpu,"
        "power.draw,utilization.gpu,memory.used"
    )
    command = [
        "nvidia-smi",
        "--id=0",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
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
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.warmups < 100:
        raise ValueError("H41 W1 requires at least 100 graph warmups per arm")
    if args.samples * args.replays_per_sample < 2000:
        raise ValueError("H41 W1 requires at least 2000 timed replays per arm")

    import flashinfer.rope

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H41 W1 front-end floors require an SM100 GPU")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    bf16 = torch.bfloat16
    fp8 = torch.float8_e4m3fn

    def make_input(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        tensor = torch.empty(*shape, device=device, dtype=torch.bfloat16)
        tensor.uniform_(-0.125, 0.125, generator=generator)
        return tensor.to(dtype) if dtype != torch.bfloat16 else tensor

    q_nope = make_input(TOKENS, HEADS, LATENT)
    q_rope = make_input(TOKENS, HEADS, ROPE)
    k_nope = make_input(TOKENS, LATENT)
    k_rope = make_input(TOKENS, ROPE)
    q_out = torch.empty(TOKENS, HEADS, LATENT + ROPE, device=device, dtype=fp8)
    k_nope_out = torch.empty_like(k_nope, dtype=fp8)
    k_rope_out = torch.empty_like(k_rope, dtype=fp8)
    cos_sin_cache = torch.empty(
        MAX_CONTEXT, ROPE, device=device, dtype=torch.float32
    )
    cos_sin_cache.uniform_(-1.0, 1.0, generator=generator)
    first_position = args.context - TOKENS
    positions = torch.arange(
        first_position,
        first_position + TOKENS,
        device=device,
        dtype=torch.int64,
    )
    kv_buffer = torch.empty(
        MAX_CONTEXT + 32, LATENT + ROPE, device=device, dtype=fp8
    )
    write_locations = torch.tensor(
        [17, 65537, 131071, 196613, 255991], device=device, dtype=torch.int64
    )
    sentinel_a = torch.ones(1, device=device, dtype=torch.float32)
    sentinel_b = torch.zeros_like(sentinel_a)
    pdl = is_arch_support_pdl()
    if not pdl:
        raise RuntimeError("H41 W1 is pinned to the SM100 PDL-enabled production path")

    def run_query() -> None:
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=True,
            quantize_dtype=fp8,
            q_rope_out=q_out[..., LATENT:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., :LATENT],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=pdl,
        )

    def run_write() -> None:
        set_mla_kv_buffer_triton(
            kv_buffer, write_locations, k_nope_out, k_rope_out
        )

    def run_combined() -> None:
        run_query()
        run_write()

    def run_one_launch() -> None:
        _non_elided_launch_floor[(1,)](sentinel_a, sentinel_b, num_warps=1)

    def run_two_launch() -> None:
        _non_elided_launch_floor[(1,)](sentinel_a, sentinel_b, num_warps=1)
        _non_elided_launch_floor[(1,)](sentinel_b, sentinel_a, num_warps=1)

    # Compile every production and floor kernel before any capture. The write
    # inputs are initialized by the query kernel before standalone write timing.
    run_combined()
    run_one_launch()
    run_two_launch()
    torch.cuda.synchronize()
    expected_k = torch.cat((k_nope_out, k_rope_out), dim=-1)
    torch.testing.assert_close(
        kv_buffer[write_locations], expected_k, rtol=0, atol=0
    )
    if not bool(torch.isfinite(q_out.float()).all()):
        raise AssertionError("normal FP8 query output contains a non-finite value")
    del expected_k

    graphs = {
        "normal_combined": capture_graph(run_combined, args.warmups),
        "one_launch_floor": capture_graph(run_one_launch, args.warmups),
        "two_launch_floor": capture_graph(run_two_launch, args.warmups),
        "normal_query": capture_graph(run_query, args.warmups),
        "normal_write": capture_graph(run_write, args.warmups),
    }
    checksums = {
        "normal_combined": (q_out, kv_buffer[17, :1]),
        "one_launch_floor": (sentinel_b,),
        "two_launch_floor": (sentinel_a, sentinel_b),
        "normal_query": (q_out, k_nope_out, k_rope_out),
        "normal_write": (kv_buffer[17, :1],),
    }
    arm_order = (
        "normal_combined",
        "one_launch_floor",
        "two_launch_floor",
        "normal_query",
        "normal_write",
        "normal_combined",
    )
    sequence: list[dict[str, Any]] = []
    combined_flank = 0
    for name in arm_order:
        if name == "normal_combined":
            combined_flank += 1
            arm_label = f"normal_combined_{combined_flank}"
        else:
            arm_label = name
        sequence.append(
            {
                "arm": arm_label,
                "kind": name,
                **time_graph(
                    graphs[name],
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                    checksum_tensors=checksums[name],
                ),
            }
        )

    combined_1 = sequence[0]["timing_us"]["mean"]
    one_launch = sequence[1]["timing_us"]["mean"]
    two_launch = sequence[2]["timing_us"]["mean"]
    query = sequence[3]["timing_us"]["mean"]
    write = sequence[4]["timing_us"]["mean"]
    combined_2 = sequence[5]["timing_us"]["mean"]
    combined_mean = (combined_1 + combined_2) / 2.0
    result = {
        "status": "TIMING_ONLY",
        "experiment": "H41_W1_FRONTEND_FLOORS",
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "context": args.context,
        "max_context": MAX_CONTEXT,
        "tokens": TOKENS,
        "heads": HEADS,
        "is_neox": True,
        "pdl_enabled": pdl,
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "normal_combined_mean_us": combined_mean,
        "normal_combined_flank_drift_fraction": (
            (combined_2 - combined_1) / combined_mean
        ),
        "normal_query_mean_us": query,
        "normal_write_mean_us": write,
        "normal_query_plus_write_us": query + write,
        "one_launch_floor_mean_us": one_launch,
        "two_launch_floor_mean_us": two_launch,
        "second_launch_increment_floor_us": two_launch - one_launch,
        "seed": args.seed,
        "interpretation": (
            "Timing-only front-end and physical-launch floors; not endpoint "
            "performance or quality evidence."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
