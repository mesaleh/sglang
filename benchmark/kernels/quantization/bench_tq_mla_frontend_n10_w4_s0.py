"""Crossed R3B/S0 SM100 fused-RoPE timing harness.

The control is the v0.5.17 FlashInfer RoPE+FP8 query launch plus generic dense
FP8 scatter. R3B is the exact split-entry residency control and S0 changes only
its E2M1 selector. This is an isolated front-end falsifier, not endpoint
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
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
r3b = load(
    "a17_n10_w1_r3b_tq_mla_frontend",
    "/tmp/n10r3boverlay/sglang/kernels/jit/tq_mla_frontend_n10.py",
)
s0 = load(
    "a17_n10_w4_s0_tq_mla_frontend",
    "/tmp/n10w4s0full/sglang/kernels/jit/tq_mla_frontend_n10.py",
)
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

LATENT = 512
ROPE = 64
HEADS = 8
MAX_CONTEXT = 256000
POOL_SIZE = MAX_CONTEXT + 32
FP8 = torch.float8_e4m3fn
ARMS = ("r3b", "s0")
MODULES = {"r3b": r3b, "s0": s0}
KERNEL_SOURCE_COMMITS = {
    "r3b": "5bf49216d0e0c864c085ccb29e12fc390f1e9b71",
    "s0": "da744f9eec98618f05a651bfbfcd72485c5fe01d",
}
EXPECTED_CONFIG_SHA256 = (
    "dbd242a672069b510cc2a8fe542d391858b71265ffa0ac4518e952e09d4c805a"
)
EXPECTED_WRAPPER_SHA256 = {
    "r3b": "128d0c5f30738c6cfaf1dbfa65f7022e0ebcc9d10a0cfade36ba0e34f4b540d9",
    "s0": "0b9fdd42e3792f6a318cba474ed30e36d3c319935f58802fd12bccd2db2c61dd",
}
EXPECTED_SOURCE_SHA256 = {
    "r3b": "13f65d194b509cbed5b5f914f7bab0e6f3a431c44521038351a1d8fc5cf5b724",
    "s0": "fdce7462acd97516993eab2263c91af6c7f2ee1ace3d69fa7827834977f3012b",
}
WILLIAMS_ORDERS = (
    ("r3b", "s0"),
    ("s0", "r3b"),
)
EXPECTED_SHARED_OBJECT_SHA256 = {
    "r3b": "0489c2dea8e0ed110c05c682d75f51642bf554daebbf3e470c78cf4f4a9eb6c3",
    "s0": "15a179a610c93dc0ac636f0606f2e0a1ec54c475cccf26c0474b4e3ba5386ee0",
}
EXPECTED_CUBIN_SHA256 = {
    "r3b": "d02b71b7c363baffbc8cefeff86a4950b5f376a03c73a9d4a267d5ef2a3eb582",
    "s0": "e7fa937662c51c324bc59eae5afa042e09c705ecf3526a3909c0d230ce1ad84a",
}
WRITER_DEBIT_BUDGET_MS = 5.206480


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(name: str, module: Any) -> dict[str, Any]:
    extension = module._get_module()
    shared_object = Path(extension.__file__).resolve()
    with tempfile.TemporaryDirectory(prefix=f"n10w4-{name}-") as directory:
        subprocess.run(
            ["cuobjdump", "--extract-elf", "all", str(shared_object)],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        cubins = sorted(Path(directory).glob("*.cubin"))
        if len(cubins) != 1:
            raise RuntimeError(f"{name} expected one cubin, found {cubins}")
        cubin_sha256 = sha256_file(cubins[0])
    wrapper = Path(module.__file__).resolve()
    source = (
        wrapper.parent
        / "csrc"
        / "tq_mla_frontend"
        / "tq_mla_frontend_n10_sm100.cu"
    )
    wrapper_sha256 = sha256_file(wrapper)
    source_sha256 = sha256_file(source)
    if wrapper_sha256 != EXPECTED_WRAPPER_SHA256[name]:
        raise AssertionError(
            f"{name} wrapper hash {wrapper_sha256} does not match recorded head"
        )
    if source_sha256 != EXPECTED_SOURCE_SHA256[name]:
        raise AssertionError(
            f"{name} source hash {source_sha256} does not match recorded head"
        )
    identity = {
        "kernel_source_commit": KERNEL_SOURCE_COMMITS[name],
        "module_name": module._MODULE_NAME,
        "wrapper": str(wrapper),
        "wrapper_sha256": wrapper_sha256,
        "source": str(source),
        "source_sha256": source_sha256,
        "shared_object": str(shared_object),
        "shared_object_sha256": sha256_file(shared_object),
        "cubin_sha256": cubin_sha256,
    }
    if identity["shared_object_sha256"] != EXPECTED_SHARED_OBJECT_SHA256[name]:
        raise AssertionError(
            f"{name} shared-object hash {identity['shared_object_sha256']} "
            "does not match the preregistered timing binary"
        )
    if identity["cubin_sha256"] != EXPECTED_CUBIN_SHA256[name]:
        raise AssertionError(
            f"{name} cubin hash {identity['cubin_sha256']} does not match "
            "the preregistered timing binary"
        )
    return identity


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
        choices=(1, 4096, 10219),
        required=True,
    )
    parser.add_argument(
        "--allocation-order",
        choices=("control-first", "candidates-first"),
        default="control-first",
    )
    parser.add_argument(
        "--timing-order-index", type=int, choices=range(2), required=True
    )
    parser.add_argument(
        "--candidate-allocation-index", type=int, choices=range(2), required=True
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

    module_names = {name: module._MODULE_NAME for name, module in MODULES.items()}
    if len(set(module_names.values())) != len(module_names):
        raise AssertionError(f"TurboQuant module names alias: {module_names}")
    artifact_identities = {
        name: artifact_identity(name, MODULES[name]) for name in ARMS
    }
    config_path = Path(kv_tq.__file__).resolve()
    config_sha256 = sha256_file(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise AssertionError(
            f"config source hash {config_sha256} does not match the exact N10 config"
        )
    if artifact_identities["r3b"]["cubin_sha256"] == artifact_identities["s0"][
        "cubin_sha256"
    ]:
        raise AssertionError("R3B and S0 resolved to the same cubin")

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

    timing_arm_order = WILLIAMS_ORDERS[args.timing_order_index]
    allocation_arm_order = WILLIAMS_ORDERS[args.candidate_allocation_index]

    def allocate_candidates() -> dict[str, tuple[torch.Tensor, ...]]:
        return {name: allocate_candidate() for name in allocation_arm_order}

    if args.allocation_order == "control-first":
        control = allocate_control()
        candidate_buffers = allocate_candidates()
    else:
        candidate_buffers = allocate_candidates()
        control = allocate_control()
    control_query, control_k, control_rope, control_cache = control
    query_latent_out = {name: values[0] for name, values in candidate_buffers.items()}
    query_rope_out = {name: values[1] for name, values in candidate_buffers.items()}
    packed_out = {name: values[2] for name, values in candidate_buffers.items()}
    scale_out = {name: values[3] for name, values in candidate_buffers.items()}
    rope_out = {name: values[4] for name, values in candidate_buffers.items()}
    status_out = {name: values[5] for name, values in candidate_buffers.items()}
    zero_count_out = {name: values[6] for name, values in candidate_buffers.items()}

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

    def run_candidate(name: str) -> None:
        MODULES[name].tq_mla_n10_frontend_rope_out(
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
            query_latent_out[name],
            query_rope_out[name],
            packed_out[name],
            scale_out[name],
            rope_out[name],
            status_out[name],
            zero_count_out[name],
            grid=config.grid,
            rotation_fused=True,
            strict=False,
        )

    candidate_functions = {
        name: (lambda candidate=name: run_candidate(candidate)) for name in ARMS
    }

    run_control()
    for name in ARMS:
        candidate_functions[name]()
    torch.cuda.synchronize()
    expected_control = torch.cat((control_k, control_rope), dim=-1).view(
        torch.uint8
    )
    torch.testing.assert_close(
        control_cache[locations], expected_control, rtol=0, atol=0
    )
    for name in ARMS:
        if int(status_out[name].item()) != 0:
            raise AssertionError(f"{name} set the fault word for valid inputs")
        if int(zero_count_out[name].item()) != 0:
            raise AssertionError(f"{name} incremented a zero counter")
    for name in ARMS[1:]:
        if not torch.equal(
            query_latent_out["r3b"].view(torch.uint8),
            query_latent_out[name].view(torch.uint8),
        ):
            raise AssertionError(f"{name} query latent differs from R3B")
        if not torch.equal(
            query_rope_out["r3b"].view(torch.int16),
            query_rope_out[name].view(torch.int16),
        ):
            raise AssertionError(f"{name} query RoPE differs from R3B")
        if not torch.equal(packed_out["r3b"][locations], packed_out[name][locations]):
            raise AssertionError(f"{name} packed latent differs from R3B")
        if not torch.equal(
            scale_out["r3b"][locations].view(torch.int16),
            scale_out[name][locations].view(torch.int16),
        ):
            raise AssertionError(f"{name} scale differs from R3B")
        if not torch.equal(
            rope_out["r3b"][locations].view(torch.int16),
            rope_out[name][locations].view(torch.int16),
        ):
            raise AssertionError(f"{name} reciprocal RoPE differs from R3B")

    graphs = {"control": capture_graph(run_control, args.warmups)}
    graphs.update(
        {
            name: capture_graph(candidate_functions[name], args.warmups)
            for name in ARMS
        }
    )
    checksum_slot = int(locations[0].item())
    checksums = {
        "control": (control_query, control_cache[checksum_slot, :1]),
    }
    checksums.update(
        {
            name: (
                query_latent_out[name],
                query_rope_out[name],
                packed_out[name][checksum_slot, :, :1],
                scale_out[name][checksum_slot],
                rope_out[name][checksum_slot, :, :1],
                status_out[name],
                zero_count_out[name],
            )
            for name in ARMS
        }
    )
    sequence: list[dict[str, Any]] = []
    for arm in ("control", *timing_arm_order, "control"):
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
    arm_means = {
        name: next(
            item["timing_us"]["mean"]
            for item in sequence
            if item["kind"] == name
        )
        for name in ARMS
    }
    control_mean = (control_1 + control_2) / 2.0
    active_slot = checksum_slot
    address_period = 2**21
    result = {
        "status": "TIMING_ONLY",
        "experiment": "A17_N10_W4_S0_EXACT_SYMMETRIC_E2M1_SELECTOR",
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
        "candidate_physical_launches": {name: 1 for name in ARMS},
        "control_physical_launches": 2,
        "allocation_order": args.allocation_order,
        "timing_order_index": args.timing_order_index,
        "timing_candidate_order": timing_arm_order,
        "candidate_allocation_index": args.candidate_allocation_index,
        "candidate_allocation_order": allocation_arm_order,
        "artifact_identities": artifact_identities,
        "config_identity": {
            "path": str(config_path),
            "sha256": config_sha256,
        },
        "pool_size": POOL_SIZE,
        "candidate_row_bytes": {name: 386 for name in ARMS},
        "buffer_address_mod_2m": {
            "control_query": control_query.data_ptr() % address_period,
            "control_cache": control_cache.data_ptr() % address_period,
            "control_cache_active": (
                control_cache.data_ptr() + active_slot * (LATENT + ROPE)
            )
            % address_period,
            "candidates": {
                name: {
                    "query_latent": query_latent_out[name].data_ptr()
                    % address_period,
                    "query_rope": query_rope_out[name].data_ptr() % address_period,
                    "packed": packed_out[name].data_ptr() % address_period,
                    "packed_active": (
                        packed_out[name].data_ptr()
                        + active_slot * (LATENT // 2)
                    )
                    % address_period,
                }
                for name in ARMS
            },
        },
        "warmups_per_graph": args.warmups,
        "samples_per_arm": args.samples,
        "replays_per_sample": args.replays_per_sample,
        "sequence": sequence,
        "control_mean_us": control_mean,
        "control_flank_drift_fraction": (control_2 - control_1) / control_mean,
        "control_flank_abs_drift_fraction": abs(control_2 - control_1)
        / control_mean,
        "passes_control_flank_drift_gate": (
            abs(control_2 - control_1) / control_mean <= 0.01
        ),
        "arm_mean_us": arm_means,
        "arm_minus_control_us": {
            name: arm_means[name] - control_mean for name in ARMS
        },
        "arm_over_control": {
            name: arm_means[name] / control_mean for name in ARMS
        },
        "s0_minus_r3b_us": arm_means["s0"] - arm_means["r3b"],
        "s0_over_r3b": arm_means["s0"] / arm_means["r3b"],
        "projected_61_layer_debit_ms": (
            61.0 * (arm_means["s0"] - control_mean) / 1000.0
        ),
        "writer_debit_budget_ms": WRITER_DEBIT_BUDGET_MS,
        "passes_writer_debit_budget": (
            61.0 * (arm_means["s0"] - control_mean) / 1000.0
            <= WRITER_DEBIT_BUDGET_MS
        ),
        "seed": args.seed,
        "interpretation": (
            "Exact R3B and S0 versus the v0.5.17 normal FP8 front-end control. "
            "Each candidate is one fused NeoX BF16-RoPE, query, and packed-cache "
            "launch. Two-arm crossed isolated timing only, not endpoint evidence."
        ),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
