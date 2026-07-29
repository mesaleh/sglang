"""Validate and summarize the five frozen H42 D1 localization arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

VARIANTS = (
    "f-chain",
    "s-chain",
    "s-static",
    "s-phased",
    "s-chain-noapdl",
)
MISSING_LONG_RECOVERY_US = 6.122896
F_CHAIN_REPLICATION_RANGE_US = (30.274, 37.202)
CACHE_ROWS = 256000
CONTROL_CACHE_BYTES = 61 * CACHE_ROWS * 576
CANDIDATE_CACHE_BYTES = 47 * CACHE_ROWS * 576 + 14 * CACHE_ROWS * 322
DENSE_OPERATIONS = [
    "flashinfer_rope_quantize_fp8",
    "dense_fp8_cache_scatter",
    "tokenspeed_dense_attention",
]
SELECTED_OPERATIONS = [
    "selected_bf16_rope",
    "h41_combined_query_material_writer",
    "tokenspeed_tq4_attention",
]


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


def validate_traces(value: dict[str, Any]) -> None:
    variant = value["variant"]
    control = value["operation_traces"]["control"]
    candidate = value["operation_traces"]["candidate"]
    expected_layers = list(range(61)) if variant == "f-chain" else list(range(24, 38))
    assert [item["layer"] for item in control] == expected_layers
    assert [item["layer"] for item in candidate] == expected_layers
    assert all(item["representation"] == "dense_fp8" for item in control)
    assert all(item["operations"] == DENSE_OPERATIONS for item in control)
    assert all(
        item["operations"]
        == (SELECTED_OPERATIONS if 24 <= item["layer"] <= 37 else DENSE_OPERATIONS)
        for item in candidate
    )
    assert [
        item["layer"]
        for item in candidate
        if item["representation"] == "tq4_e2m1_fp8_rope"
    ] == list(range(24, 38))


def load_result(root: Path, variant: str) -> dict[str, Any]:
    path = root / variant / "result.json"
    value = json.loads(path.read_text())
    assert value["status"] == "DIAGNOSTIC_ONLY"
    assert value["experiment"] == "H42_D1_LONG_DEPENDENCY_LOCALIZATION"
    assert value["variant"] == variant
    assert value["trace_only"] is False
    assert value["hostname"].endswith("ct13")
    assert value["context"] == 37932
    assert value["split_kv"] == 40
    assert value["allocation_order"] == "control-first"
    assert value["seed"] == 20260729
    assert value["total_layers"] == 61
    assert value["timed_layers"] == (61 if variant == "f-chain" else 14)
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
    assert value["allocation_check_replays"] == 100
    assert value["attention_pdl"] == (variant != "s-chain-noapdl")
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
    assert abs(value["control_flank_drift_fraction"]) <= 0.02
    control_1 = value["sequence"][0]["graph_us"]["mean"]
    candidate = value["sequence"][1]["graph_us"]["mean"]
    control_2 = value["sequence"][2]["graph_us"]["mean"]
    control_mean = (control_1 + control_2) / 2.0
    delta = candidate - control_mean
    assert math.isclose(value["control_mean_graph_us"], control_mean, abs_tol=1e-9)
    assert math.isclose(value["candidate_mean_graph_us"], candidate, abs_tol=1e-9)
    assert math.isclose(
        value["candidate_minus_control_graph_us"], delta, abs_tol=1e-9
    )
    assert math.isclose(
        value["integrated_delta_per_selected_layer_us"],
        delta / 14,
        abs_tol=1e-9,
    )
    assert math.isclose(
        value["control_flank_drift_fraction"],
        (control_2 - control_1) / control_mean,
        abs_tol=1e-12,
    )
    replay = value["graph_replay_allocation_bytes"]
    assert replay["before"] == replay["after"]
    assert value["one_selected_layer_max_abs_diff"] <= value["correctness_atol"]
    assert value["writer_reader_max_abs_diff"] <= value["correctness_atol"]
    assert all(arm["sticky_status"] in (None, 0) for arm in value["sequence"])
    assert value["idle_proof"]["rows"] == []
    assert len(value["current_locations"]) == 5
    assert len(set(value["current_locations"])) == 5
    expected_topology = (
        "fixed_independent_input"
        if variant in ("s-static", "s-phased")
        else "attention_output_to_next_query_latent"
    )
    expected_schedule = (
        "all_frontends_then_all_readers"
        if variant == "s-phased"
        else "frontend_attention_interleaved"
    )
    assert value["inter_layer_dependency"] == expected_topology
    assert value["query_input_topology"] == expected_topology
    assert value["stream_dependency_topology"] == "single_stream_programmatic_edges"
    assert value["scheduling"] == expected_schedule
    logical = value["logical_operation_counts"]
    assert logical["control_dense_layers"] == (61 if variant == "f-chain" else 14)
    assert logical["candidate_dense_layers"] == (47 if variant == "f-chain" else 0)
    assert logical["candidate_selected_layers"] == 14
    assert logical["operations_per_layer"] == 3
    validate_traces(value)
    validate_gpu_covariates(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    results = {variant: load_result(args.root, variant) for variant in VARIANTS}
    deltas = {
        variant: value["integrated_delta_per_selected_layer_us"]
        for variant, value in results.items()
    }
    f_chain_replicates = (
        F_CHAIN_REPLICATION_RANGE_US[0]
        <= deltas["f-chain"]
        <= F_CHAIN_REPLICATION_RANGE_US[1]
    )
    contributions = {
        "surroundings_and_transitions_us": deltas["f-chain"] - deltas["s-chain"],
        "query_pointer_value_buffer_sensitivity_us": (
            deltas["s-chain"] - deltas["s-static"]
        ),
        "interleaving_us": deltas["s-static"] - deltas["s-phased"],
        "reader_pdl_sensitivity_us": (
            deltas["s-chain-noapdl"] - deltas["s-chain"]
        ),
    }
    implementation_eligible = [
        name
        for name in ("interleaving_us",)
        if contributions[name] >= MISSING_LONG_RECOVERY_US
    ]
    status = "VALID" if f_chain_replicates else "HARNESS_REJECT"
    result = {
        "status": status,
        "method": (
            "Five predeclared fresh-process paired C/T/C source-only graph arms; "
            "component differences are diagnostic and are not added into an endpoint claim."
        ),
        "f_chain_replication": {
            "observed_us_per_selected_layer": deltas["f-chain"],
            "required_range_us": list(F_CHAIN_REPLICATION_RANGE_US),
            "gate": "PASS" if f_chain_replicates else "FAIL",
        },
        "variant_delta_per_selected_layer_us": deltas,
        "contributions": contributions,
        "missing_long_recovery_us": MISSING_LONG_RECOVERY_US,
        "implementation_eligible_mechanisms": implementation_eligible,
        "reader_pdl_followup_triggered": (
            abs(contributions["reader_pdl_sensitivity_us"])
            >= MISSING_LONG_RECOVERY_US
        ),
        "max_control_flank_drift_fraction": max(
            abs(value["control_flank_drift_fraction"])
            for value in results.values()
        ),
        "max_writer_reader_abs_diff": max(
            value["writer_reader_max_abs_diff"] for value in results.values()
        ),
        "interpretation": (
            "Localization only. S-static preserves same-stream PDL edges and is only a "
            "query pointer/value/buffer negative control. S-static and S-phased own identical "
            "per-layer buffers, so their difference is scheduling-only. Phased scheduling is "
            "an illegal deployment variant; a source hypothesis still requires a separately "
            "frozen gate."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
