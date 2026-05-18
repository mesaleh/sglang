# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
import triton
from common_utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from ray.experimental.tqdm_ray import tqdm

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel_mxfp4,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device, is_hip, is_xpu
import triton.language as tl

_is_hip = is_hip()
_is_xpu = is_xpu()


def _benchmark_mxfp4_config(
    config: BenchmarkConfig,
    *,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    mxfp4_kernel: str,  # "gate_up" or "down"
    num_iters: int,
) -> float:
    """Time ``invoke_fused_moe_kernel_mxfp4`` at the requested shape.

    Uses the same CUDA-graph capture + L2-cache-flush pattern as the
    bf16 ``benchmark_config`` path so µs-per-call numbers are directly
    comparable across quant modes in this tuner.
    """
    assert mxfp4_kernel in ("gate_up", "down"), mxfp4_kernel

    if mxfp4_kernel == "gate_up":
        # A: [M, K_hidden] bf16; W13: [E, 2*N_int, K_hidden/2] uint8
        # top_k matches the model's routing top_k.
        kernel_N = shard_intermediate_size      # 2 * intermediate
        kernel_K = hidden_size
        kernel_topk = topk
        input_M = num_tokens
    else:
        # Down takes [M*topk, N_intermediate] input and calls the kernel
        # with top_k=1 (the MoE alignment already factored in the topk
        # expansion during the gate-up step).
        kernel_N = hidden_size
        kernel_K = shard_intermediate_size // 2  # intermediate
        kernel_topk = 1
        input_M = num_tokens * topk

    device = "cuda"
    x = torch.randn(input_M, kernel_K, dtype=torch.bfloat16, device=device) * 0.3
    # Random bytes — content doesn't affect timing.
    w = torch.randint(
        0, 256, (num_experts, kernel_N, kernel_K // 2), dtype=torch.uint8, device=device
    )
    # Scale byte 127 == 2^0 == 1.0 (E8M0 convention); keeps dequant output
    # in a tame range in case we inspect C post-run.
    w_scale = torch.full(
        (num_experts, kernel_N, kernel_K // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    topk_ids = torch.randint(
        0, num_experts, (input_M, kernel_topk), dtype=torch.int32, device=device
    )
    topk_weights = torch.rand(
        input_M, kernel_topk, dtype=torch.float32, device=device
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )
    y = torch.empty(
        input_M * kernel_topk, kernel_N, dtype=torch.bfloat16, device=device
    )

    def run() -> None:
        invoke_fused_moe_kernel_mxfp4(
            x,
            w,
            w_scale,
            y,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=kernel_topk,
            config=dict(config),
            compute_type=tl.bfloat16,
            filter_expert=False,
            enable_nan_prop=False,
        )

    # JIT compile + warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph (same as bf16 path).
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()
    latencies = [
        start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)
    ]
    avg_us = sum(latencies) / (num_iters * 10) * 1000
    graph.reset()
    return avg_us


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
    use_mxfp4_w4a16: bool = False,
    mxfp4_kernel: str = "gate_up",
) -> float:
    # MXFP4 has two separate matmul kernels (gate-up and down) that share
    # a ``sorted_token_ids`` alignment but have independent tile configs.
    # For this dtype we benchmark exactly one of the two per call — the
    # outer sweep runs twice and saves two JSONs (see main()).
    if use_mxfp4_w4a16:
        return _benchmark_mxfp4_config(
            config,
            num_tokens=num_tokens,
            num_experts=num_experts,
            shard_intermediate_size=shard_intermediate_size,
            hidden_size=hidden_size,
            topk=topk,
            mxfp4_kernel=mxfp4_kernel,
            num_iters=num_iters,
        )
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    elif use_int4_w4a16:
        w1 = torch.randint(
            0,
            255,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size // 2,
            ),
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            0,
            255,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 4,
            ),
            dtype=torch.uint8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_int4_w4a16:
        block_n = 1 if (block_shape[0] == 0) else block_shape[0]
        block_k = block_shape[1]
        n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
        n_tiles_w2 = (hidden_size + block_n - 1) // block_n
        k_tiles_w1 = (hidden_size + block_k - 1) // block_k
        k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
        w1_scale = torch.randn(
            (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.bfloat16
        )
        w2_scale = torch.randn(
            (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.bfloat16
        )
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # Flush L2 cache with 256 MB data
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        prepare(i)
        start_events[i].record()
        graph.replay()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies: List[float] = []
    for i in range(num_iters):
        latencies.append(start_events[i].elapsed_time(end_events[i]))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


@ray.remote(num_gpus=1)
class BenchmarkWorker:

    def __init__(self, seed: int, server_args: ServerArgs) -> None:
        torch.set_default_device(get_device())
        torch.get_device_module().manual_seed_all(0)
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU.
        self.device_id = int(ray.get_gpu_ids()[0])
        set_global_server_args_for_scheduler(server_args)

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        use_mxfp4_w4a16: bool = False,
        mxfp4_kernel: str = "gate_up",
    ) -> Tuple[Dict[str, int], float]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
            use_mxfp4_w4a16=use_mxfp4_w4a16,
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        N = shard_intermediate_size // 2
        if use_int4_w4a16 or use_mxfp4_w4a16:
            # int4_w4a16 and mxfp4_w4a16 both pack two values per byte on
            # the inner axis, so the filename-key N halves.
            N = N // 2
        op_config = get_moe_configs(
            num_experts,
            N,
            dtype_str,
            block_n,
            block_k,
            per_channel_quant,
            down_moe=(use_mxfp4_w4a16 and mxfp4_kernel == "down"),
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                block_shape,
                use_mxfp4_w4a16=use_mxfp4_w4a16,
                mxfp4_kernel=mxfp4_kernel,
            )
        return config, kernel_time

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        use_mxfp4_w4a16: bool = False,
        mxfp4_kernel: str = "gate_up",
        return_all_timings: bool = False,
    ) -> Any:
        # Returns either ``best_config`` (legacy) or
        # ``(best_config, all_timings)`` where ``all_timings`` maps the
        # tuple key (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages)
        # to µs. The full timings are used by the MXFP4 post-processing
        # pass (see ``_harmonize_block_m``) to enforce the gate-up/down
        # BLOCK_SIZE_M invariant that ``moe_align_block_size`` requires.
        best_config = None
        best_time = float("inf")
        all_timings: Dict[Tuple[int, int, int, int, int, int], float] = {}
        with (
            torch.get_device_module().device(self.device_id)
            if _is_xpu or _is_hip
            else nullcontext()
        ):
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        use_int4_w4a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=10,
                        use_mxfp4_w4a16=use_mxfp4_w4a16,
                        mxfp4_kernel=mxfp4_kernel,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    # Some configurations may be invalid and fail to compile.
                    continue

                if return_all_timings:
                    all_timings[
                        (
                            config["BLOCK_SIZE_M"],
                            config["BLOCK_SIZE_N"],
                            config["BLOCK_SIZE_K"],
                            config["GROUP_SIZE_M"],
                            config["num_warps"],
                            config["num_stages"],
                        )
                    ] = kernel_time
                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        if return_all_timings:
            return best_config, all_timings
        return best_config


# ---------------------------------------------------------------------------
# MXFP4-specific post-processing: BLOCK_SIZE_M harmonization
# ---------------------------------------------------------------------------
#
# ``try_get_optimal_moe_config`` asserts that the gate-up and down configs
# returned for the same M share BLOCK_SIZE_M, because ``moe_align_block_size``
# is called once per forward pass with one BLOCK_M value. An untuned sweep
# can pick different local winners and violate that invariant; the
# ``_harmonize_block_m`` pass rewrites both configs to share the BLOCK_M
# that minimizes the *combined* (gate-up + down) latency at each M.
#
# This is MXFP4-specific because it's the only dtype that currently tunes
# gate-up and down as two separate kernels; other quant modes use one
# kernel for both and get a single config per M automatically.


def _lookup_best_with_fixed_block_m(
    timings: Dict[Tuple[int, int, int, int, int, int], float],
    required_block_m: int,
) -> Optional[Dict[str, int]]:
    """Return the fastest config in ``timings`` whose BLOCK_SIZE_M equals
    ``required_block_m``, or ``None`` if the grid contained no such entry."""
    best: Optional[Dict[str, int]] = None
    best_time = float("inf")
    best_t = 0.0
    for (bm, bn, bk, gm, nw, ns), us in timings.items():
        if bm != required_block_m:
            continue
        if us < best_time:
            best_time = us
            best = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm,
                "num_warps": nw,
                "num_stages": ns,
            }
            best_t = us
    return best


def _harmonize_block_m(
    gate_up_best: Dict[int, Dict[str, int]],
    down_best: Dict[int, Dict[str, int]],
    gate_up_timings: Dict[int, Dict[Tuple[int, int, int, int, int, int], float]],
    down_timings: Dict[int, Dict[Tuple[int, int, int, int, int, int], float]],
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:
    for M in sorted(gate_up_best.keys()):
        up = gate_up_best[M]
        dn = down_best.get(M)
        if dn is None or up["BLOCK_SIZE_M"] == dn["BLOCK_SIZE_M"]:
            continue

        candidate_bms = {k[0] for k in gate_up_timings[M].keys()} & {
            k[0] for k in down_timings[M].keys()
        }
        chosen_bm: Optional[int] = None
        chosen_total = float("inf")
        for bm in candidate_bms:
            u_cfg = _lookup_best_with_fixed_block_m(gate_up_timings[M], bm)
            d_cfg = _lookup_best_with_fixed_block_m(down_timings[M], bm)
            if u_cfg is None or d_cfg is None:
                continue
            u_key = (
                u_cfg["BLOCK_SIZE_M"], u_cfg["BLOCK_SIZE_N"], u_cfg["BLOCK_SIZE_K"],
                u_cfg["GROUP_SIZE_M"], u_cfg["num_warps"], u_cfg["num_stages"],
            )
            d_key = (
                d_cfg["BLOCK_SIZE_M"], d_cfg["BLOCK_SIZE_N"], d_cfg["BLOCK_SIZE_K"],
                d_cfg["GROUP_SIZE_M"], d_cfg["num_warps"], d_cfg["num_stages"],
            )
            total = gate_up_timings[M][u_key] + down_timings[M][d_key]
            if total < chosen_total:
                chosen_total = total
                chosen_bm = bm
        assert (
            chosen_bm is not None
        ), f"No shared BLOCK_M available for M={M}; grid too narrow"
        new_up = _lookup_best_with_fixed_block_m(gate_up_timings[M], chosen_bm)
        new_dn = _lookup_best_with_fixed_block_m(down_timings[M], chosen_bm)
        assert new_up is not None and new_dn is not None
        print(
            f"[harmonize] M={M}: "
            f"BLOCK_SIZE_M {up['BLOCK_SIZE_M']}(up)/{dn['BLOCK_SIZE_M']}(down) "
            f"→ {chosen_bm} (combined {chosen_total:.2f} us)"
        )
        gate_up_best[M] = new_up
        down_best[M] = new_dn
    return gate_up_best, down_best


def main(args: argparse.Namespace):
    server_args = ServerArgs(
        model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size
    )

    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int4_w4a16 = args.dtype == "int4_w4a16"
    use_mxfp4_w4a16 = args.dtype == "mxfp4_w4a16"
    per_channel_quant = args.per_channel_quant

    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
    else:
        batch_sizes = [args.batch_size]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed, server_args) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: List[Any]) -> List[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_n, block_k = block_shape[0], block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]
        if use_mxfp4_w4a16:
            # BLOCK_SIZE_K must be a multiple of 32 (MXFP block); 64 also
            # divides K=2880 exactly (gpt-oss) so the fast no-mask load
            # path kicks in. BLOCK_K=128 triggers the partial-tail path
            # and was consistently slower in empirical sweeps.
            search_space = [
                cfg for cfg in search_space if cfg["BLOCK_SIZE_K"] == 64
            ]

        if use_mxfp4_w4a16:
            # Tune gate-up and down independently, then harmonize
            # BLOCK_SIZE_M across the two so moe_align_block_size sees a
            # single compatible value. Two JSONs get written — the
            # ``_down`` suffix is what ``get_moe_configs(down_moe=True)``
            # looks up at runtime.
            up_filename = get_config_filename(
                E, shard_intermediate_size, hidden_size, topk, dtype,
                use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                per_channel_quant, block_shape,
                use_mxfp4_w4a16=True, down_moe=False,
            )
            down_filename = get_config_filename(
                E, shard_intermediate_size, hidden_size, topk, dtype,
                use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                per_channel_quant, block_shape,
                use_mxfp4_w4a16=True, down_moe=True,
            )
            print(
                f"Start MXFP4 tuning over {len(search_space)} configurations "
                f"to create {up_filename} + {down_filename}..."
            )

            start = time.perf_counter()

            # Gate-up sweep: per-batch M is the user token count.
            up_outputs = _distribute(
                "tune",
                [
                    (
                        batch_size, E, shard_intermediate_size, hidden_size,
                        topk, dtype, use_fp8_w8a8, use_int8_w8a8,
                        use_int8_w8a16, use_int4_w4a16, per_channel_quant,
                        block_shape, search_space, True, "gate_up", True,
                    )
                    for batch_size in batch_sizes
                ],
            )
            # Down sweep: kernel M is batch * topk (post-alignment rows),
            # kernel top_k is 1. The *JSON key* is still batch_size so the
            # gate-up and down lookups match on M.
            down_outputs = _distribute(
                "tune",
                [
                    (
                        batch_size * topk, E, shard_intermediate_size, hidden_size,
                        topk, dtype, use_fp8_w8a8, use_int8_w8a8,
                        use_int8_w8a16, use_int4_w4a16, per_channel_quant,
                        block_shape, search_space, True, "down", True,
                    )
                    for batch_size in batch_sizes
                ],
            )

            gate_up_best = {
                M: cfg for M, (cfg, _) in zip(batch_sizes, up_outputs)
            }
            down_best = {
                M: cfg for M, (cfg, _) in zip(batch_sizes, down_outputs)
            }
            gate_up_times = {
                M: timings for M, (_, timings) in zip(batch_sizes, up_outputs)
            }
            down_times = {
                M: timings for M, (_, timings) in zip(batch_sizes, down_outputs)
            }

            gate_up_best, down_best = _harmonize_block_m(
                gate_up_best, down_best, gate_up_times, down_times
            )

            save_configs(
                {M: sort_config(cfg) for M, cfg in sorted(gate_up_best.items())},
                up_filename,
            )
            save_configs(
                {M: sort_config(cfg) for M, cfg in sorted(down_best.items())},
                down_filename,
            )
            end = time.perf_counter()
            print(f"MXFP4 tuning took {end - start:.2f} seconds")
            return

        filename = get_config_filename(
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
        )
        print(
            f"Start tuning over {len(search_space)} configurations to create {filename}..."
        )

        start = time.perf_counter()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                    search_space,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            filename,
        )
        end = time.perf_counter()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    E,
                    shard_intermediate_size,
                    hidden_size,
                    topk,
                    dtype,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                    use_int4_w4a16,
                    per_channel_quant,
                    block_shape,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=[
            "auto",
            "fp8_w8a8",
            "int8_w8a16",
            "int8_w8a8",
            "int4_w4a16",
            "mxfp4_w4a16",
        ],
        default="auto",
        help="Quant mode to tune. mxfp4_w4a16 runs a two-pass sweep "
        "(gate-up + down) and writes two JSONs with BLOCK_SIZE_M "
        "harmonized across them.",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    main(args)
