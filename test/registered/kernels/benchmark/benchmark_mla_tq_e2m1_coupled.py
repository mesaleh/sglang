"""Bracket the full N10 decode layer against the dense FP8 TokenSpeed path.

This benchmark includes the format-specific cache write and query preparation,
the attention reader, and the N10 inverse WHT.  It intentionally uses batch 1
and a 10,240-token logical cache to match the primary endpoint score shape.
"""

import math
import os
import statistics

import torch
from tokenspeed_mla import tokenspeed_mla_decode, tokenspeed_mla_decode_tq_e2m1
from tokenspeed_mla.utils import get_num_sm

# Importing the TQ config first avoids the inherited partial-overlay cycle
# between fp8_kernel and quantization.__init__.
# isort: off
from sglang.srt.layers.quantization.kv_turboquant import NativeE2M1MLAConfig

# isort: on

from sglang.kernels.jit.tq_mla_frontend_n10_native import (
    tq_mla_n10_native_frontend_out,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.quantization.hadamard import (
    hadamard_transform_with_signs,
)

PAGE_SIZE = 32
HEADS = 8
LATENT = 512
ROPE = 64
VALUE = 128


def _capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def _measure(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / replays


def _paired_ratio(candidate, dense, *, windows: int, replays: int):
    candidate_samples = []
    dense_samples = []
    for window in range(windows):
        order = (
            ((candidate, candidate_samples), (dense, dense_samples))
            if window % 2 == 0
            else ((dense, dense_samples), (candidate, candidate_samples))
        )
        for graph, samples in order:
            samples.append(_measure(graph, replays))

    log_ratios = [
        math.log(candidate_us / dense_us)
        for candidate_us, dense_us in zip(candidate_samples, dense_samples)
    ]
    critical = 2.045229642 if windows == 30 else 2.262157163
    mean_log = statistics.fmean(log_ratios)
    standard_error = statistics.stdev(log_ratios) / math.sqrt(windows)
    return (
        statistics.fmean(candidate_samples),
        statistics.fmean(dense_samples),
        math.exp(mean_log),
        math.exp(mean_log - critical * standard_error),
        math.exp(mean_log + critical * standard_error),
    )


def _run(query_len: int, seq_len: int, max_seq_len: int, windows: int, replays: int):
    if seq_len % PAGE_SIZE:
        raise ValueError("seq_len must be page aligned")
    if max_seq_len < seq_len or max_seq_len % PAGE_SIZE:
        raise ValueError("max_seq_len must be page aligned and cover seq_len")

    device = torch.device("cuda")
    torch.manual_seed(20260816 + query_len)
    page_count = seq_len // PAGE_SIZE
    max_page_count = max_seq_len // PAGE_SIZE
    locations = torch.arange(
        seq_len - query_len,
        seq_len,
        dtype=torch.int64,
        device=device,
    )
    block_tables = torch.full(
        (1, max_page_count),
        page_count - 1,
        dtype=torch.int32,
        device=device,
    )
    block_tables[0, :page_count] = torch.arange(
        page_count, dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    query_latent = torch.randn(
        query_len, HEADS, LATENT, dtype=torch.bfloat16, device=device
    )
    query_rope = torch.randn(
        query_len, HEADS, ROPE, dtype=torch.bfloat16, device=device
    )
    cache_latent = torch.randn(
        query_len, 1, LATENT, dtype=torch.bfloat16, device=device
    )
    cache_rope = torch.randn(query_len, 1, ROPE, dtype=torch.bfloat16, device=device)

    packed = torch.zeros(
        page_count, PAGE_SIZE, LATENT // 2, dtype=torch.uint8, device=device
    )
    scales = torch.ones(page_count, PAGE_SIZE, dtype=torch.bfloat16, device=device)
    reciprocal_rope = torch.zeros(
        page_count, PAGE_SIZE, ROPE, dtype=torch.bfloat16, device=device
    )
    dense_cache = torch.zeros(
        page_count,
        PAGE_SIZE,
        LATENT + ROPE,
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    config = NativeE2M1MLAConfig(device=device)
    candidate_query_latent = torch.empty(
        query_len, HEADS, LATENT, dtype=torch.float8_e4m3fn, device=device
    )
    candidate_query_rope = torch.empty(
        query_len, HEADS, ROPE, dtype=torch.bfloat16, device=device
    )
    candidate_rotated_out = torch.empty(
        1, query_len, HEADS, LATENT, dtype=torch.bfloat16, device=device
    )
    candidate_out = torch.empty_like(candidate_rotated_out)
    dense_out = torch.empty_like(candidate_rotated_out)
    fault_status = torch.zeros((), dtype=torch.int32, device=device)
    zero_count = torch.zeros((), dtype=torch.int64, device=device)

    workspace_nbytes = get_num_sm(device) * HEADS * query_len * (LATENT + 1) * 4
    candidate_workspace = torch.empty(workspace_nbytes, dtype=torch.int8, device=device)
    dense_workspace = torch.empty_like(candidate_workspace)
    softmax_scale = 1.0 / math.sqrt(LATENT + ROPE)

    true_weight = (
        torch.randn(HEADS, VALUE, LATENT, dtype=torch.bfloat16, device=device)
        / math.sqrt(LATENT)
    ).transpose(1, 2)
    folded_bf16_weight = config.fuse_inverse_rotation_into_mla_v_weight(true_weight)
    candidate_value_out = torch.empty(
        HEADS, query_len, VALUE, dtype=torch.bfloat16, device=device
    )
    folded_value_out = torch.empty_like(candidate_value_out)
    dense_value_out = torch.empty_like(candidate_value_out)

    def candidate_frontend():
        tq_mla_n10_native_frontend_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            locations,
            config.signs1,
            config.signs2,
            candidate_query_latent,
            candidate_query_rope,
            packed.view(-1, 1, LATENT // 2),
            scales.view(-1, 1),
            reciprocal_rope.view(-1, 1, ROPE),
            fault_status,
            zero_count,
            grid=config.grid,
            rotation_fused=False,
        )

    def candidate_reader():
        tokenspeed_mla_decode_tq_e2m1(
            query_latent=candidate_query_latent.view(1, query_len, HEADS, LATENT),
            query_rope=candidate_query_rope.view(1, query_len, HEADS, ROPE),
            packed_latent=packed,
            reconstruction_scale=scales,
            reciprocal_rope=reciprocal_rope,
            workspace_buffer=candidate_workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            softmax_scale=softmax_scale,
            out=candidate_rotated_out,
            enable_pdl=True,
        )

    def inverse_wht():
        hadamard_transform_with_signs(
            candidate_rotated_out,
            config.signs2,
            config.signs1,
            scale=1.0 / math.sqrt(LATENT),
            out=candidate_out,
        )

    dense_query = None

    def dense_frontend():
        nonlocal dense_query
        dense_query = set_mla_kv_concat_q_fp8(
            dense_cache.view(-1, LATENT + ROPE),
            locations,
            cache_latent,
            cache_rope,
            query_latent,
            query_rope,
        )

    def dense_reader():
        tokenspeed_mla_decode(
            query=dense_query.view(1, query_len, HEADS, LATENT + ROPE),
            kv_cache=dense_cache,
            workspace_buffer=dense_workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            softmax_scale=softmax_scale,
            out=dense_out,
            enable_pdl=True,
        )

    def candidate_full():
        candidate_frontend()
        candidate_reader()
        inverse_wht()

    def candidate_folded_output():
        candidate_frontend()
        candidate_reader()

    def dense_full():
        dense_frontend()
        dense_reader()

    def candidate_current_layer():
        candidate_frontend()
        candidate_reader()
        inverse_wht()
        torch.bmm(
            candidate_out.view(query_len, HEADS, LATENT).transpose(0, 1),
            true_weight,
            out=candidate_value_out,
        )

    def candidate_folded_bf16_layer():
        candidate_frontend()
        candidate_reader()
        torch.bmm(
            candidate_rotated_out.view(query_len, HEADS, LATENT).transpose(0, 1),
            folded_bf16_weight,
            out=folded_value_out,
        )

    def dense_layer():
        dense_frontend()
        dense_reader()
        torch.bmm(
            dense_out.view(query_len, HEADS, LATENT).transpose(0, 1),
            true_weight,
            out=dense_value_out,
        )

    candidate_frontend()
    dense_frontend()
    component_graphs = {
        "candidate_frontend": _capture(candidate_frontend),
        "candidate_reader": _capture(candidate_reader),
        "candidate_inverse": _capture(inverse_wht),
        "dense_frontend": _capture(dense_frontend),
        "dense_reader": _capture(dense_reader),
    }
    candidate_graph = _capture(candidate_full)
    candidate_folded_output_graph = _capture(candidate_folded_output)
    dense_graph = _capture(dense_full)
    candidate_current_layer_graph = _capture(candidate_current_layer)
    candidate_folded_bf16_layer_graph = _capture(candidate_folded_bf16_layer)
    dense_layer_graph = _capture(dense_layer)
    for graph in (
        *component_graphs.values(),
        candidate_graph,
        candidate_folded_output_graph,
        dense_graph,
        candidate_current_layer_graph,
        candidate_folded_bf16_layer_graph,
        dense_layer_graph,
    ):
        _measure(graph, replays)

    candidate_us, dense_us, ratio, lower, upper = _paired_ratio(
        candidate_graph,
        dense_graph,
        windows=windows,
        replays=replays,
    )
    components = {
        name: _measure(graph, replays) for name, graph in component_graphs.items()
    }
    folded_us, folded_dense_us, folded_ratio, folded_lower, folded_upper = (
        _paired_ratio(
            candidate_folded_output_graph,
            dense_graph,
            windows=windows,
            replays=replays,
        )
    )
    print(
        "TQ_E2M1_COUPLED_TIMING "
        f"q_len={query_len} batch=1 seq_len={seq_len} max_seq_len={max_seq_len} "
        f"dense_us={dense_us:.6f} candidate_us={candidate_us:.6f} "
        f"candidate_over_dense={ratio:.6f} ci95=[{lower:.6f},{upper:.6f}] "
        f"folded_dense_us={folded_dense_us:.6f} folded_candidate_us={folded_us:.6f} "
        f"folded_over_dense={folded_ratio:.6f} "
        f"folded_ci95=[{folded_lower:.6f},{folded_upper:.6f}] "
        + " ".join(f"{name}_us={value:.6f}" for name, value in components.items())
        + f" windows={windows} replays={replays}",
        flush=True,
    )
    current_layer = _paired_ratio(
        candidate_current_layer_graph,
        dense_layer_graph,
        windows=windows,
        replays=replays,
    )
    folded_layer = _paired_ratio(
        candidate_folded_bf16_layer_graph,
        dense_layer_graph,
        windows=windows,
        replays=replays,
    )
    folded_over_current = _paired_ratio(
        candidate_folded_bf16_layer_graph,
        candidate_current_layer_graph,
        windows=windows,
        replays=replays,
    )
    print(
        "TQ_E2M1_COUPLED_VALUE_TIMING "
        f"q_len={query_len} batch=1 seq_len={seq_len} max_seq_len={max_seq_len} "
        f"dense_us={current_layer[1]:.6f} "
        f"current_candidate_us={current_layer[0]:.6f} "
        f"current_over_dense={current_layer[2]:.6f} "
        f"current_ci95=[{current_layer[3]:.6f},{current_layer[4]:.6f}] "
        f"folded_dense_us={folded_layer[1]:.6f} "
        f"folded_candidate_us={folded_layer[0]:.6f} "
        f"folded_over_dense={folded_layer[2]:.6f} "
        f"folded_ci95=[{folded_layer[3]:.6f},{folded_layer[4]:.6f}] "
        f"folded_current_us={folded_over_current[1]:.6f} "
        f"folded_over_current={folded_over_current[2]:.6f} "
        f"folded_over_current_ci95="
        f"[{folded_over_current[3]:.6f},{folded_over_current[4]:.6f}] "
        f"windows={windows} replays={replays}",
        flush=True,
    )
    if fault_status.item() != 0:
        raise RuntimeError(f"N10 writer fault status is {fault_status.item()}")


def main():
    seq_len = int(os.environ.get("TQ_COUPLED_SEQ_LEN", "10240"))
    max_seq_len = int(os.environ.get("TQ_COUPLED_MAX_SEQ_LEN", str(seq_len)))
    windows = int(os.environ.get("TQ_COUPLED_WINDOWS", "30"))
    replays = int(os.environ.get("TQ_COUPLED_REPLAYS", "500"))
    if windows not in (10, 30):
        raise ValueError("TQ_COUPLED_WINDOWS must be 10 or 30")
    for query_len in (1, 5):
        _run(query_len, seq_len, max_seq_len, windows, replays)


if __name__ == "__main__":
    main()
