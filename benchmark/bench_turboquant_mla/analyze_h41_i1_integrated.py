"""Analyze H41 I1 independent-process integrated graph evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

ALLOWANCE_95_LOWER_US = {10219: 27.0292, 37932: 27.6973}
EXPECTED_SPLIT = {10219: 64, 37932: 40}
CACHE_ROWS = 256000
CONTROL_CACHE_BYTES = 61 * CACHE_ROWS * 576
CANDIDATE_CACHE_BYTES = 47 * CACHE_ROWS * 576 + 14 * CACHE_ROWS * 322


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], *, draws: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choices(values, k=len(values))) for _ in range(draws)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def validate_trace(value: dict[str, Any]) -> None:
    traces = value["operation_traces"]
    control = traces["control"]
    candidate = traces["candidate"]
    assert [item["layer"] for item in control] == list(range(61))
    assert [item["layer"] for item in candidate] == list(range(61))
    assert all(item["representation"] == "dense_fp8" for item in control)
    selected_ids = [
        item["layer"]
        for item in candidate
        if item["representation"] == "tq4_e2m1_fp8_rope"
    ]
    assert selected_ids == list(range(24, 38))
    assert sum(item["representation"] == "dense_fp8" for item in candidate) == 47
    dense_operations = [
        "flashinfer_rope_quantize_fp8",
        "dense_fp8_cache_scatter",
        "tokenspeed_dense_attention",
    ]
    selected_operations = [
        "selected_bf16_rope",
        "h41_combined_query_material_writer",
        "tokenspeed_tq4_attention",
    ]
    assert all(item["operations"] == dense_operations for item in control)
    assert all(
        item["operations"]
        == (selected_operations if 24 <= item["layer"] <= 37 else dense_operations)
        for item in candidate
    )
    assert value["inter_layer_dependency"] == "attention_output_to_next_query_latent"


def validate_gpu_covariates(value: dict[str, Any]) -> None:
    for arm in value["sequence"]:
        for side in ("gpu_before", "gpu_after"):
            covariates = arm[side]
            assert "error" not in covariates
            assert covariates["pstate"] == "P0"
            assert covariates["ecc.errors.uncorrected.volatile.total"] == "0"
            assert covariates["gpu_recovery_action"].lower() in ("none", "n/a")
            assert int(covariates["clocks.current.graphics"]) == int(
                covariates["clocks.max.graphics"]
            )


def load_sequence(root: Path, context: int, sequence: int) -> dict[str, Any]:
    path = root / f"context{context}" / f"seq{sequence:02d}" / "result.json"
    value = json.loads(path.read_text())
    assert value["status"] == "TIMING_ONLY"
    assert value["experiment"] == "H41_I1_INTEGRATED_FOUR_FAMILY"
    assert value["context"] == context
    assert value["split_kv"] == EXPECTED_SPLIT[context]
    assert value["total_layers"] == 61
    assert value["selected_layers"] == 14
    assert value["selected_layer_ids"] == list(range(24, 38))
    assert value["selected_warps"] == 8
    assert value["selected_row_bytes"] == 322
    assert value["cache_rows"] == CACHE_ROWS
    assert value["selected_fp8_rope"] is True
    assert value["selected_codebook_materialized"] is False
    assert value["candidate_tq_codebook_bytes"] == 0
    assert value["control_persistent_cache_bytes"] == CONTROL_CACHE_BYTES
    assert value["candidate_persistent_cache_bytes"] == CANDIDATE_CACHE_BYTES
    assert value["gross_persistent_cache_savings_bytes"] == (
        CONTROL_CACHE_BYTES - CANDIDATE_CACHE_BYTES
    )
    assert value["warmups_per_graph"] >= 100
    assert value["samples_per_arm"] * value["replays_per_sample"] >= 2000
    assert [arm["arm"] for arm in value["sequence"]] == [
        "control_1",
        "candidate",
        "control_2",
    ]
    assert [arm["kind"] for arm in value["sequence"]] == [
        "control",
        "candidate",
        "control",
    ]
    assert all(
        arm["replays"] >= 2000 and math.isfinite(arm["output_checksum"])
        for arm in value["sequence"]
    )
    expected_order = "candidate-first" if sequence == 2 else "control-first"
    assert value["allocation_order"] == expected_order
    assert abs(value["control_flank_drift_fraction"]) <= 0.02
    replay = value["graph_replay_allocation_bytes"]
    assert replay["before"] == replay["after"]
    assert value["one_selected_layer_max_abs_diff"] <= value["correctness_atol"]
    assert value["writer_reader_max_abs_diff"] <= value["correctness_atol"]
    assert all(
        arm["sticky_status"] in (None, 0) for arm in value["sequence"]
    )
    validate_trace(value)
    validate_gpu_covariates(value)
    return value


def load_roundtrip(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for context in (10219, 37932):
        for q_len in (1, 5):
            path = root / f"context{context}" / f"q{q_len}" / "result.json"
            value = json.loads(path.read_text())
            assert value["status"] == "PASS"
            assert value["experiment"] == "H41_I1_WRITER_READER_COMPOSITION"
            assert value["context"] == context
            assert value["q_len"] == q_len
            assert value["split_kv"] == EXPECTED_SPLIT[context]
            assert value["unique_locations"] is True
            assert value["partial_final_page"] is True
            assert value["shuffled_page_table"] is True
            assert value["selected_warps"] == 8
            assert value["selected_row_bytes"] == 322
            assert value["fp8_rope"] is True
            assert value["codebook_materialized"] is False
            assert value["sticky_status"] == 0
            assert value["idle_proof"]["rows"] == []
            assert len(value["checks"]) == 2
            assert [item["label"] for item in value["checks"]] == [
                "eager",
                "graph_replay",
            ]
            assert max(item["max_abs_diff"] for item in value["checks"]) <= value[
                "correctness_atol"
            ]
            replay = value["graph_replay_allocation_bytes"]
            assert replay["before"] == replay["after"]
            results.append(value)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--roundtrip-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    roundtrip = load_roundtrip(args.roundtrip_root)

    analyses: list[dict[str, Any]] = []
    for context in (10219, 37932):
        sequences = [
            load_sequence(args.root, context, sequence) for sequence in (1, 2, 3)
        ]
        deltas = [
            value["integrated_delta_per_selected_layer_us"] for value in sequences
        ]
        lower, upper = bootstrap_mean_ci(
            deltas,
            draws=args.draws,
            seed=args.seed + context,
        )
        allowance = ALLOWANCE_95_LOWER_US[context]
        analyses.append(
            {
                "context": context,
                "split_kv": EXPECTED_SPLIT[context],
                "allocation_orders": [
                    value["allocation_order"] for value in sequences
                ],
                "control_mean_graph_us": statistics.fmean(
                    value["control_mean_graph_us"] for value in sequences
                ),
                "candidate_mean_graph_us": statistics.fmean(
                    value["candidate_mean_graph_us"] for value in sequences
                ),
                "sequence_delta_per_selected_layer_us": deltas,
                "delta_mean_per_selected_layer_us": statistics.fmean(deltas),
                "delta_bootstrap_95_us": [lower, upper],
                "full_allowance_95_lower_us": allowance,
                "headroom_at_delta_upper_us": allowance - upper,
                "max_abs_control_flank_drift_fraction": max(
                    abs(value["control_flank_drift_fraction"])
                    for value in sequences
                ),
                "max_writer_reader_abs_diff": max(
                    value["writer_reader_max_abs_diff"] for value in sequences
                ),
                "gate": "PASS" if upper <= allowance else "FAIL",
            }
        )

    result = {
        "status": (
            "PASS" if all(item["gate"] == "PASS" for item in analyses) else "FAIL"
        ),
        "method": (
            "Independent-process cluster bootstrap of three C/T/C complete-graph "
            "per-selected-layer mean deltas."
        ),
        "draws": args.draws,
        "seed": args.seed,
        "writer_reader_composition": {
            "gate": "PASS",
            "cases": len(roundtrip),
            "contexts": sorted({item["context"] for item in roundtrip}),
            "q_lens": sorted({item["q_len"] for item in roundtrip}),
            "max_abs_diff": max(
                check["max_abs_diff"]
                for item in roundtrip
                for check in item["checks"]
            ),
        },
        "analyses": analyses,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
