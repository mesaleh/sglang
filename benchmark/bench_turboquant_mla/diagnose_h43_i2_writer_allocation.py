"""Instrument H43 I2 writer allocations without making a qualification claim."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import torch

from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
from test_h41_w2_frontend import (
    LATENT,
    allocate_guarded,
    assert_case,
    assert_guards,
    cpu_codebook_reference,
    launch,
    make_inputs,
    query_reference,
    raw_fp8,
    writer_reference,
)

STAT_KEYS = (
    "allocated_bytes.all.current",
    "active_bytes.all.current",
    "reserved_bytes.all.current",
    "allocation.all.current",
    "active.all.current",
    "segment.all.current",
    "num_alloc_retries",
    "num_ooms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 5), required=True)
    parser.add_argument("--write-codebook", type=int, choices=(0, 1), required=True)
    parser.add_argument("--prelude", choices=("none", "correctness"), required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--prior-q1-graph", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.iterations != 100:
        raise ValueError(
            "the H43 allocation diagnostic requires exactly 100 iterations"
        )
    prefix = args.output_prefix
    if (
        not prefix.is_absolute()
        or prefix.parent != Path("/results")
        or not re.fullmatch(r"[a-z0-9-]+", prefix.name)
    ):
        raise ValueError("output-prefix must be /results/<lowercase-safe-name>")
    for suffix in (
        "initial-before.pickle",
        "eager-before.pickle",
        "eager-after.pickle",
        "graph-after.pickle",
    ):
        if Path(f"{prefix}-{suffix}").exists():
            raise ValueError(f"diagnostic snapshot already exists: {prefix}-{suffix}")
    return args


def allocator_sample(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {key: int(stats.get(key, 0)) for key in STAT_KEYS}


def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in STAT_KEYS}


def run_prelude(
    config: TurboQuantConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[int, tuple[Any, ...]]:
    compile_inputs = make_inputs(1, device, generator, "random")
    compile_buffers = allocate_guarded(1, 4, device)
    compile_locations = torch.tensor([1], dtype=torch.int64, device=device)
    launch(
        compile_inputs,
        compile_locations,
        config,
        compile_buffers,
        True,
        8,
    )
    torch.cuda.synchronize(device)

    cases: list[dict[str, Any]] = []
    for tokens in (1, 2, 5, 7, 10, 15, 20, 25, 30, 35, 40):
        pool_size = tokens + 11
        locations = torch.randperm(pool_size, device=device)[:tokens].to(torch.int64)
        for kind in ("random", "zero", "impulse", "repeated"):
            inputs = make_inputs(tokens, device, generator, kind)
            for rotation_fused in (True, False):
                for warps in (1, 2, 4, 8):
                    metrics = assert_case(
                        inputs,
                        locations,
                        config,
                        rotation_fused,
                        warps,
                        pool_size,
                    )
                    cases.append(
                        {
                            "tokens": tokens,
                            "kind": kind,
                            "rotation_fused": rotation_fused,
                            "warps": warps,
                            **metrics,
                        }
                    )
    count = len(cases)
    if count != 352:
        raise AssertionError(f"correctness prelude covered {count} cases")
    torch.cuda.synchronize(device)
    retained = (
        compile_inputs,
        compile_buffers,
        compile_locations,
        inputs,
        locations,
        cases,
    )
    return count, retained


def run_prior_q1_graph(
    config: TurboQuantConfig,
    device: torch.device,
    generator: torch.Generator,
    iterations: int,
) -> tuple[Any, ...]:
    inputs = make_inputs(1, device, generator, "random")
    locations = torch.arange(1, dtype=torch.int64, device=device) + 3
    buffers = allocate_guarded(1, 8, device)

    def invoke() -> None:
        launch(inputs, locations, config, buffers, True, 8)

    invoke()
    torch.cuda.synchronize(device)
    for _ in range(iterations):
        invoke()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    graph.replay()
    torch.cuda.synchronize(device)
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize(device)
    expected = query_reference(inputs[0], inputs[1], config, True)
    packed_expected, scale_expected, rope_expected, codebook_expected = (
        writer_reference(inputs[2], inputs[3], locations, config, 8)
    )
    if not torch.equal(raw_fp8(buffers.query), raw_fp8(expected)):
        raise AssertionError("prior q1 graph query mismatch")
    if not torch.equal(buffers.codebook[locations], codebook_expected[locations]):
        raise AssertionError("prior q1 graph codebook mismatch")
    if int(buffers.status.item()) != 0:
        raise AssertionError("prior q1 graph sticky status is nonzero")
    assert_guards(buffers)
    del packed_expected, scale_expected
    return graph, expected, rope_expected, codebook_expected


def run_replays(
    *,
    label: str,
    iterations: int,
    invoke: Any,
    device: torch.device,
    synchronize_each: bool,
) -> dict[str, Any]:
    before = allocator_sample(device)
    previous = before
    transitions: list[dict[str, Any]] = []
    for iteration in range(1, iterations + 1):
        invoke()
        if synchronize_each:
            torch.cuda.synchronize(device)
            current = allocator_sample(device)
            if current != previous:
                transitions.append(
                    {
                        "iteration": iteration,
                        "before": previous,
                        "after": current,
                        "delta": delta(current, previous),
                    }
                )
            previous = current
    torch.cuda.synchronize(device)
    after = allocator_sample(device)
    return {
        "label": label,
        "iterations": iterations,
        "before": before,
        "after": after,
        "delta": delta(after, before),
        "transitions": transitions,
        "synchronize_each": synchronize_each,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H43 I2 allocation diagnostic requires SM100")
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
    if args.prelude == "correctness":
        prelude_cases, retained_prelude = run_prelude(config, device, generator)
    else:
        prelude_cases, retained_prelude = 0, None
    retained_prior_q1 = (
        run_prior_q1_graph(config, device, generator, args.iterations)
        if args.prior_q1_graph
        else None
    )

    inputs = make_inputs(args.tokens, device, generator, "random")
    pool_size = args.tokens + 7
    locations = torch.arange(args.tokens, dtype=torch.int64, device=device) + 3
    buffers = allocate_guarded(args.tokens, pool_size, device)

    def invoke() -> None:
        launch(
            inputs,
            locations,
            config,
            buffers,
            True,
            8,
            write_codebook=bool(args.write_codebook),
        )

    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=50_000,
        device=device,
        clear_history=True,
    )
    prefix = str(args.output_prefix)
    torch.cuda.memory._dump_snapshot(f"{prefix}-initial-before.pickle")
    initial_before = allocator_sample(device)
    invoke()
    torch.cuda.synchronize(device)
    initial_after = allocator_sample(device)
    torch.cuda.memory._dump_snapshot(f"{prefix}-eager-before.pickle")
    eager = run_replays(
        label="eager-batched",
        iterations=args.iterations,
        invoke=invoke,
        device=device,
        synchronize_each=False,
    )
    torch.cuda.memory._dump_snapshot(f"{prefix}-eager-after.pickle")

    graph = torch.cuda.CUDAGraph()
    capture_before = allocator_sample(device)
    with torch.cuda.graph(graph):
        invoke()
    torch.cuda.synchronize(device)
    capture_after = allocator_sample(device)
    graph.replay()
    torch.cuda.synchronize(device)
    graph_replay = run_replays(
        label="graph-replay-batched",
        iterations=args.iterations,
        invoke=graph.replay,
        device=device,
        synchronize_each=False,
    )
    eager_stepped = run_replays(
        label="eager-stepped",
        iterations=args.iterations,
        invoke=invoke,
        device=device,
        synchronize_each=True,
    )
    graph_replay_stepped = run_replays(
        label="graph-replay-stepped",
        iterations=args.iterations,
        invoke=graph.replay,
        device=device,
        synchronize_each=True,
    )
    torch.cuda.memory._dump_snapshot(f"{prefix}-graph-after.pickle")
    torch.cuda.memory._record_memory_history(enabled=None, device=device)

    assert int(buffers.status.item()) == 0
    assert_guards(buffers)
    if args.write_codebook:
        expected = cpu_codebook_reference(buffers.scale[locations], config.k_centroids)
        codebook_mismatches = int(
            (buffers.codebook[locations].detach().cpu() != expected).sum().item()
        )
        if codebook_mismatches:
            raise AssertionError(
                f"diagnostic codebook mismatches: {codebook_mismatches}"
            )
    else:
        codebook_mismatches = int((buffers.codebook != 0x3C).sum().item())
        if codebook_mismatches:
            raise AssertionError("no-codebook diagnostic modified the codebook buffer")

    result = {
        "status": "PASS",
        "experiment": "H43_I2_WRITER_ALLOCATION_DIAGNOSTIC_ARM",
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "UNSET"),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "device": torch.cuda.get_device_name(device),
        "tokens": args.tokens,
        "write_codebook": bool(args.write_codebook),
        "prelude": args.prelude,
        "prelude_cases": prelude_cases,
        "prior_q1_graph": bool(args.prior_q1_graph),
        "iterations": args.iterations,
        "seed": args.seed,
        "warmups_before_eager": 1,
        "initial_launch": {
            "before": initial_before,
            "after": initial_after,
            "delta": delta(initial_after, initial_before),
        },
        "eager": eager,
        "eager_invariant": eager["before"]["allocated_bytes.all.current"]
        == eager["after"]["allocated_bytes.all.current"],
        "eager_stepped": eager_stepped,
        "graph_capture": {
            "before": capture_before,
            "after": capture_after,
            "delta": delta(capture_after, capture_before),
        },
        "graph_replay": graph_replay,
        "graph_replay_invariant": graph_replay["before"]["allocated_bytes.all.current"]
        == graph_replay["after"]["allocated_bytes.all.current"],
        "graph_replay_stepped": graph_replay_stepped,
        "sticky_status": int(buffers.status.item()),
        "codebook_mismatches": codebook_mismatches,
        "snapshot_prefix": prefix,
        "qualification_claim": False,
    }
    _ = retained_prelude, retained_prior_q1
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
