"""Analyze H43 I2 independent-process integrated codebook graph evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

ALLOWANCE_95_UPPER_US = {10219: 27.0292, 37932: 27.6973}
EXPECTED_SPLIT = {10219: 64, 37932: 40}
CACHE_ROWS = 256000
CONTROL_CACHE_BYTES = 61 * CACHE_ROWS * 576
CANDIDATE_CACHE_BYTES = 47 * CACHE_ROWS * 576 + 14 * CACHE_ROWS * 338
GPU_UUID = "GPU-9f90e004-9332-4d9d-fa34-018fb9f07fca"
PAIRS_PER_PROCESS = 20
REPLAYS_PER_SAMPLE = 100
PROCESSES_PER_CONTEXT = 10
MINIMUM_RETAINED_PER_ORDER = 8
SENTINEL_MAX_DRIFT_FRACTION = 0.005


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_upper(values: list[float], *, draws: int, seed: int) -> float:
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choices(values, k=len(values))) for _ in range(draws)
    ]
    return percentile(means, 0.95)


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
        "h43_combined_query_material_codebook_writer",
        "tokenspeed_tq4_codebook_attention",
    ]
    assert all(item["operations"] == dense_operations for item in control)
    assert all(
        item["operations"]
        == (selected_operations if 24 <= item["layer"] <= 37 else dense_operations)
        for item in candidate
    )
    assert value["inter_layer_dependency"] == "attention_output_to_next_query_latent"


def validate_hard_gpu_health(sample: dict[str, Any]) -> None:
    assert "error" not in sample
    assert sample["index"] == "0"
    assert sample["uuid"] == GPU_UUID
    assert sample["name"] == "NVIDIA GB200"
    assert sample["ecc.errors.uncorrected.volatile.total"] == "0"
    assert sample["gpu_recovery_action"] == "None"
    assert sample["fabric.state"] == "state  Completed"
    assert sample["fabric.status"] == "status Success"


def expected_telemetry_reasons(sample: dict[str, Any]) -> list[str]:
    expected = {
        "index": "0",
        "uuid": GPU_UUID,
        "pstate": "P0",
        "clocks.sm": "1965",
        "clocks.max.sm": "1965",
        "clocks_event_reasons.hw_slowdown": "Not Active",
        "clocks_event_reasons.sw_thermal_slowdown": "Not Active",
        "ecc.errors.uncorrected.volatile.total": "0",
        "gpu_recovery_action": "None",
        "fabric.state": "state  Completed",
        "fabric.status": "status Success",
    }
    return [
        f"{key}={sample.get(key)!r}, expected {expected_value!r}"
        for key, expected_value in expected.items()
        if sample.get(key) != expected_value
    ]


def validate_gpu_covariates(value: dict[str, Any]) -> None:
    records = [value["sentinel_before"], *value["pairs"], value["sentinel_after"]]
    aggregate_ecc_values: set[str] = set()
    for record in records:
        expected_reasons: list[str] = []
        for side in ("telemetry_before", "telemetry_after"):
            sample = record[side]
            validate_hard_gpu_health(sample)
            aggregate_ecc_values.add(sample["ecc.errors.uncorrected.aggregate.total"])
            expected_reasons.extend(expected_telemetry_reasons(sample))
        assert record["telemetry_reasons"] == expected_reasons
        assert record["telemetry_valid"] == (not expected_reasons)
    assert len(aggregate_ecc_values) == 1


def balanced_process_delta(pairs: list[dict[str, Any]]) -> tuple[float, dict[str, int]]:
    by_order: dict[str, list[float]] = {"AB": [], "BA": []}
    for pair in pairs:
        if not pair["telemetry_valid"]:
            continue
        order = "AB" if pair["order"] == ["control", "candidate"] else "BA"
        durations = pair["durations_us"]
        by_order[order].append((durations["candidate"] - durations["control"]) / 14)
    retained = {order: len(values) for order, values in by_order.items()}
    assert all(count >= MINIMUM_RETAINED_PER_ORDER for count in retained.values())
    return (
        statistics.fmean(statistics.fmean(by_order[order]) for order in ("AB", "BA")),
        retained,
    )


def load_sequence(root: Path, context: int, sequence: int) -> dict[str, Any]:
    path = root / f"context{context}" / f"seq{sequence:02d}" / "result.json"
    value = json.loads(path.read_text())
    assert value["status"] == "TIMING_ONLY"
    assert value["experiment"] == "H43_I2_INTEGRATED_CODEBOOK_FOUR_FAMILY"
    assert value["context"] == context
    assert value["split_kv"] == EXPECTED_SPLIT[context]
    assert value["total_layers"] == 61
    assert value["selected_layers"] == 14
    assert value["selected_layer_ids"] == list(range(24, 38))
    assert value["selected_warps"] == 8
    assert value["selected_row_bytes"] == 338
    assert value["cache_rows"] == CACHE_ROWS
    assert value["selected_fp8_rope"] is True
    assert value["selected_codebook_materialized"] is True
    assert value["candidate_tq_codebook_bytes"] == 14 * CACHE_ROWS * 16
    assert value["control_persistent_cache_bytes"] == CONTROL_CACHE_BYTES
    assert value["candidate_persistent_cache_bytes"] == CANDIDATE_CACHE_BYTES
    assert value["gross_persistent_cache_savings_bytes"] == (
        CONTROL_CACHE_BYTES - CANDIDATE_CACHE_BYTES
    )
    assert value["warmups_per_graph"] >= 100
    assert value["samples_per_arm"] == PAIRS_PER_PROCESS
    assert value["replays_per_sample"] == REPLAYS_PER_SAMPLE
    assert value["sequence_number"] == sequence
    expected_capture = (
        ["control", "candidate"] if sequence % 2 else ["candidate", "control"]
    )
    assert value["capture_order"] == expected_capture
    assert len(value["pairs"]) == PAIRS_PER_PROCESS
    for pair_index, pair in enumerate(value["pairs"]):
        assert pair["pair_index"] == pair_index
        control_first = (pair_index % 2 == 0) == (sequence % 2 == 1)
        expected_order = (
            ["control", "candidate"] if control_first else ["candidate", "control"]
        )
        assert pair["order"] == expected_order
        assert set(pair["durations_us"]) == {"control", "candidate"}
        assert all(
            math.isfinite(duration) and duration > 0
            for duration in pair["durations_us"].values()
        )
    assert math.isfinite(value["control_output_checksum"])
    assert math.isfinite(value["candidate_output_checksum"])
    assert value["sticky_status"] == 0
    expected_order = "candidate-first" if sequence == 2 else "control-first"
    assert value["allocation_order"] == expected_order
    sentinels = (value["sentinel_before"], value["sentinel_after"])
    assert value["sentinel_max_drift_fraction"] == SENTINEL_MAX_DRIFT_FRACTION
    assert all(sentinel["telemetry_valid"] for sentinel in sentinels)
    assert all(
        math.isfinite(sentinel["graph_us"]) and sentinel["graph_us"] > 0
        for sentinel in sentinels
    )
    sentinel_drift = abs(sentinels[1]["graph_us"] - sentinels[0]["graph_us"]) / (
        (sentinels[1]["graph_us"] + sentinels[0]["graph_us"]) / 2.0
    )
    assert sentinel_drift <= SENTINEL_MAX_DRIFT_FRACTION
    replay = value["graph_replay_allocation_bytes"]
    assert replay["before"] == replay["after"]
    assert value["one_selected_layer_max_abs_diff"] <= value["correctness_atol"]
    assert value["writer_reader_max_abs_diff"] <= value["correctness_atol"]
    validate_trace(value)
    validate_gpu_covariates(value)
    balanced_process_delta(value["pairs"])
    return value


def load_roundtrip(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for context in (10219, 37932):
        for q_len in (1, 5):
            path = root / f"context{context}" / f"q{q_len}" / "result.json"
            value = json.loads(path.read_text())
            assert value["status"] == "PASS"
            assert value["experiment"] == "H43_I2_WRITER_CODEBOOK_READER_COMPOSITION"
            assert value["context"] == context
            assert value["q_len"] == q_len
            assert value["split_kv"] == EXPECTED_SPLIT[context]
            assert value["unique_locations"] is True
            assert value["partial_final_page"] is True
            assert value["shuffled_page_table"] is True
            assert value["selected_warps"] == 8
            assert value["selected_row_bytes"] == 338
            assert value["fp8_rope"] is True
            assert value["codebook_materialized"] is True
            assert value["sticky_status"] == 0
            assert value["idle_proof"]["rows"] == []
            assert len(value["checks"]) == 2
            assert [item["label"] for item in value["checks"]] == [
                "eager",
                "graph_replay",
            ]
            assert (
                max(item["max_abs_diff"] for item in value["checks"])
                <= value["correctness_atol"]
            )
            replay = value["graph_replay_allocation_bytes"]
            assert replay["before"] == replay["after"]
            results.append(value)
    return results


def validate_result_inventory(root: Path, roundtrip_root: Path) -> None:
    expected_decision = {
        root / f"context{context}" / f"seq{sequence:02d}" / "result.json"
        for context in (10219, 37932)
        for sequence in range(1, PROCESSES_PER_CONTEXT + 1)
    }
    observed_decision = set(root.rglob("result.json"))
    assert observed_decision == expected_decision
    expected_roundtrip = {
        roundtrip_root / f"context{context}" / f"q{q_len}" / "result.json"
        for context in (10219, 37932)
        for q_len in (1, 5)
    }
    observed_roundtrip = set(roundtrip_root.rglob("result.json"))
    assert observed_roundtrip == expected_roundtrip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--roundtrip-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    validate_result_inventory(args.root, args.roundtrip_root)
    roundtrip = load_roundtrip(args.roundtrip_root)

    analyses: list[dict[str, Any]] = []
    for context in (10219, 37932):
        sequences = [
            load_sequence(args.root, context, sequence)
            for sequence in range(1, PROCESSES_PER_CONTEXT + 1)
        ]
        process_values = [balanced_process_delta(value["pairs"]) for value in sequences]
        deltas = [value[0] for value in process_values]
        retained = [value[1] for value in process_values]
        upper = bootstrap_mean_upper(
            deltas,
            draws=args.draws,
            seed=args.seed + context,
        )
        allowance = ALLOWANCE_95_UPPER_US[context]
        analyses.append(
            {
                "context": context,
                "split_kv": EXPECTED_SPLIT[context],
                "allocation_orders": [value["allocation_order"] for value in sequences],
                "process_delta_per_selected_layer_us": deltas,
                "retained_pairs_by_order": retained,
                "delta_mean_per_selected_layer_us": statistics.fmean(deltas),
                "delta_bootstrap_one_sided_95_upper_us": upper,
                "full_allowance_95_upper_us": allowance,
                "headroom_at_delta_upper_us": allowance - upper,
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
            "Independent-process bootstrap of ten order-balanced paired complete-"
            "graph per-selected-layer mean deltas with a one-sided 95% upper bound."
        ),
        "draws": args.draws,
        "seed": args.seed,
        "writer_reader_composition": {
            "gate": "PASS",
            "cases": len(roundtrip),
            "contexts": sorted({item["context"] for item in roundtrip}),
            "q_lens": sorted({item["q_len"] for item in roundtrip}),
            "max_abs_diff": max(
                check["max_abs_diff"] for item in roundtrip for check in item["checks"]
            ),
        },
        "analyses": analyses,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
