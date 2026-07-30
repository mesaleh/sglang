"""Paired isolated timing and profile gate for the H43 native codebook write."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any

import torch

from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

from test_h41_w2_frontend import (
    HEADS,
    LATENT,
    ROPE,
    allocate_guarded,
    assert_guards,
    launch,
    make_inputs,
    raw_fp8,
    writer_reference,
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
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
        "stdev": statistics.stdev(values),
    }


def capture(fn) -> torch.cuda.CUDAGraph:
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def timed_replays(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 5, 40), default=5)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument(
        "--profile-arm", choices=("codebook", "no-codebook"), default=None
    )
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.pairs < 20 or args.replays < 100:
        raise ValueError("H43 I2 requires at least 20 pairs of 100 replays")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H43 I2 writer delta requires SM100")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    config = TurboQuantConfig(
        bit_width=4,
        head_dim=LATENT,
        device=device,
        k_bit_width=4,
        v_bit_width=4,
        uniform=False,
        e2m1=True,
    )
    inputs = make_inputs(args.tokens, device, generator, "realistic")
    pool_size = args.tokens + 17
    locations = torch.randperm(pool_size, device=device)[: args.tokens].to(torch.int64)
    no_codebook = allocate_guarded(args.tokens, pool_size, device)
    codebook = allocate_guarded(args.tokens, pool_size, device)

    def no_codebook_call() -> None:
        launch(
            inputs,
            locations,
            config,
            no_codebook,
            True,
            8,
            write_codebook=False,
        )

    def codebook_call() -> None:
        launch(inputs, locations, config, codebook, True, 8, write_codebook=True)

    no_codebook_call()
    codebook_call()
    torch.cuda.synchronize()
    for left, right in (
        (raw_fp8(no_codebook.query), raw_fp8(codebook.query)),
        (no_codebook.packed, codebook.packed),
        (no_codebook.scale.view(torch.uint16), codebook.scale.view(torch.uint16)),
        (raw_fp8(no_codebook.rope), raw_fp8(codebook.rope)),
    ):
        if not torch.equal(left, right):
            raise AssertionError("codebook specialization changed an existing output")
    _, _, _, expected_codebook = writer_reference(
        inputs[2], inputs[3], locations, config, pool_size
    )
    if not torch.equal(codebook.codebook[locations], expected_codebook[locations]):
        raise AssertionError("native codebook bytes differ from the independent writer")
    assert_guards(no_codebook)
    assert_guards(codebook)

    graphs = {
        "no-codebook": capture(no_codebook_call),
        "codebook": capture(codebook_call),
    }
    if args.profile_arm is not None:
        torch.cuda.nvtx.range_push(f"H43_I2_WRITER_{args.profile_arm.upper()}")
        graphs[args.profile_arm].replay()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(json.dumps({"status": "PASS", "profile_arm": args.profile_arm}))
        return

    pairs: list[dict[str, Any]] = []
    for pair_index in range(args.pairs):
        no_first = (pair_index % 2 == 0) == (args.sequence % 2 == 1)
        order = ("no-codebook", "codebook") if no_first else ("codebook", "no-codebook")
        durations = {arm: timed_replays(graphs[arm], args.replays) for arm in order}
        pairs.append(
            {
                "pair_index": pair_index,
                "order": list(order),
                "durations_us": durations,
                "delta_us": durations["codebook"] - durations["no-codebook"],
            }
        )
    deltas = [pair["delta_us"] for pair in pairs]
    result = {
        "status": "TIMING_ONLY",
        "experiment": "H43_I2_NATIVE_WRITER_CODEBOOK_DELTA",
        "tokens": args.tokens,
        "warps": 8,
        "pairs": pairs,
        "delta_us": summarize(deltas),
        "selected_row_bytes": {"no_codebook": 322, "codebook": 338},
        "correctness": "BYTE_EXACT",
        "interpretation": "Isolated writer delta; integrated graph is authoritative.",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
