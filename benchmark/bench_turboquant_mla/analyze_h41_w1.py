#!/usr/bin/env python3
"""Analyze the pre-registered H41 W1 feasibility measurements.

This script intentionally consumes only the frozen all-FP8 endpoint calibration,
the three fresh-process reader brackets, and the three fresh-process front-end
floor brackets.  It does not tune a candidate or estimate endpoint speed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


ENDPOINT_RUNS = {
    10219: "h41-w1-fp8-exact10219-c1-o512-r50",
    37932: "h41-w1-fp8-long37932-c1-o512-r50",
}
PROMPT_TARGETS = {10219: 16000, 37932: 60000}
SPLIT_KV = {10219: 64, 37932: 40}
SELECTED_LAYERS = (14, 19)
SEQUENCES = (1, 2, 3)
ALLOCATION_ORDERS = {
    1: "candidate-first",
    2: "control-first",
    3: "candidate-first",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.bootstrap_draws < 10_000:
        raise ValueError("H41 W1 requires at least 10,000 bootstrap draws")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * (position - lower)


def interval(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "lower_95": quantile(ordered, 0.025),
        "median": quantile(ordered, 0.5),
        "upper_95": quantile(ordered, 0.975),
    }


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator


def internal_accept_length(payload: dict[str, Any]) -> float:
    server = payload.get("sglang_server_info", payload)
    return float(server["internal_states"][0]["avg_spec_accept_length"])


def require_fields(
    observed: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} contract drift: {mismatches}")


def response_accept_length(requests: list[dict[str, Any]]) -> float:
    """Derive accepted tokens/forward from SGLang's grouped stream events.

    The OpenAI stream contains one non-generation event, hence the subtraction.
    The aggregate is checked against SGLang's lifetime accepted-token/forward
    counters before it is used in the bootstrap.
    """

    completion_tokens = sum(float(item["completion_tokens"]) for item in requests)
    forward_events = sum(float(item["response_chunks"] - 1) for item in requests)
    if forward_events <= 0:
        raise ValueError("response stream did not expose positive forward events")
    return completion_tokens / forward_events


def validate_run_config(config: dict[str, Any], context: int) -> None:
    expected = {
        "model": "moonshotai/Kimi-K2.6",
        "prompt_sizes": str(PROMPT_TARGETS[context]),
        "output_tokens": "512",
        "concurrency": "1",
        "cache_mode": "unique-prefix",
        "thinking": "unset",
        "ignore_eos": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeats": 50,
        "warmup_requests": 2,
    }
    require_fields(config, expected, f"context {context} endpoint")


def validate_requests(requests: list[dict[str, Any]], context: int) -> None:
    measured = [item for item in requests if not item["is_warmup"]]
    warmups = [item for item in requests if item["is_warmup"]]
    if len(measured) != 50 or len(warmups) != 2 or len(requests) != 52:
        raise ValueError(f"context {context} requires 2 warmups plus 50 requests")
    for item in requests:
        if not item["ok"] or item["error"] is not None:
            raise ValueError(f"context {context} contains a failed request")
        if item["completion_tokens"] != 512 or item["finish_reason"] != "length":
            raise ValueError(f"context {context} request did not reach the 512-token cap")
        if item["response_chunks"] <= 1:
            raise ValueError(f"context {context} request lacks grouped stream events")
        request_contract = {
            "model": "moonshotai/Kimi-K2.6",
            "prompt_tokens_target": PROMPT_TARGETS[context],
            "output_tokens_target": 512,
            "concurrency": 1,
            "cache_mode": "unique-prefix",
            "thinking": "unset",
            "ignore_eos": False,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        require_fields(item, request_contract, f"context {context} raw request")


def validate_reader_result(
    result: dict[str, Any], *, context: int, layers: int, sequence: int
) -> None:
    expected = {
        "status": "TIMING_ONLY",
        "experiment": "H41_W1_MATERIAL_READER",
        "device": "NVIDIA GB200",
        "context": context,
        "max_context": 256000,
        "batch": 1,
        "q_len": 5,
        "heads": 8,
        "selected_layers": layers,
        "dense_before": {14: 24, 19: 21}[layers],
        "fp8_rope": layers == 14,
        "split_kv": SPLIT_KV[context],
        "allocation_order": ALLOCATION_ORDERS[sequence],
    }
    label = f"N{layers} context {context} reader seq{sequence:02}"
    require_fields(result, expected, label)
    if result["warmups_per_graph"] < 100:
        raise ValueError(f"{label} has too few warmups")
    if result["samples_per_arm"] * result["replays_per_sample"] < 2000:
        raise ValueError(f"{label} has too few timed replays")
    if result["max_abs_diff_one_selected_layer"] > result["correctness_atol"]:
        raise ValueError(f"{label} failed correctness")
    if abs(result["control_flank_drift_fraction"]) > 0.02:
        raise ValueError(f"{label} exceeded the 2% control-drift guard")


def validate_frontend_result(
    result: dict[str, Any], *, context: int, sequence: int
) -> None:
    expected = {
        "status": "TIMING_ONLY",
        "experiment": "H41_W1_FRONTEND_FLOORS",
        "device": "NVIDIA GB200",
        "context": context,
        "max_context": 256000,
        "tokens": 5,
        "heads": 8,
        "pdl_enabled": True,
    }
    label = f"context {context} frontend seq{sequence:02}"
    require_fields(result, expected, label)
    if result["warmups_per_graph"] < 100:
        raise ValueError(f"{label} has too few warmups")
    if result["samples_per_arm"] * result["replays_per_sample"] < 2000:
        raise ValueError(f"{label} has too few timed replays")
    if abs(result["normal_combined_flank_drift_fraction"]) > 0.02:
        raise ValueError(f"{label} exceeded the 2% control-drift guard")


def bootstrap_window(
    *,
    requests: list[dict[str, Any]],
    reader_points: dict[int, list[float]],
    one_launch_points: list[float],
    two_launch_points: list[float],
    draws: int,
    rng: random.Random,
) -> dict[str, Any]:
    measured = [item for item in requests if not item["is_warmup"]]
    tpot = [float(item["tpot_ms"]) for item in measured]
    tokens = [float(item["completion_tokens"]) for item in measured]
    forwards = [float(item["response_chunks"] - 1) for item in measured]
    per_request_accept = [a / b for a, b in zip(tokens, forwards)]

    allowances = {layers: [] for layers in SELECTED_LAYERS}
    readers = {layers: [] for layers in SELECTED_LAYERS}
    one_launch: list[float] = []
    two_launch: list[float] = []

    for _ in range(draws):
        request_indices = [rng.randrange(len(measured)) for _ in measured]
        mean_tpot = sum(tpot[index] for index in request_indices) / len(measured)
        accept_length = sum(tokens[index] for index in request_indices) / sum(
            forwards[index] for index in request_indices
        )

        for layers in SELECTED_LAYERS:
            # The 0.8 factor reserves 20% of the endpoint regression budget.
            allowances[layers].append(
                0.8 * 0.05 * mean_tpot * accept_length * 1000.0 / layers
            )
            readers[layers].append(
                sum(
                    reader_points[layers][rng.randrange(len(SEQUENCES))]
                    for _ in SEQUENCES
                )
                / len(SEQUENCES)
            )

        one_launch.append(
            sum(one_launch_points[rng.randrange(len(SEQUENCES))] for _ in SEQUENCES)
            / len(SEQUENCES)
        )
        two_launch.append(
            sum(two_launch_points[rng.randrange(len(SEQUENCES))] for _ in SEQUENCES)
            / len(SEQUENCES)
        )

    result: dict[str, Any] = {
        "measured_requests": len(measured),
        "mean_tpot_ms": statistics.mean(tpot),
        "response_derived_accept_length": response_accept_length(measured),
        "tpot_accept_length_correlation": correlation(tpot, per_request_accept),
        "one_launch_floor_delta_us": interval(one_launch),
        "two_launch_floor_delta_us": interval(two_launch),
        "arms": {},
    }
    for layers in SELECTED_LAYERS:
        remainder = [
            allowance - reader
            for allowance, reader in zip(allowances[layers], readers[layers])
        ]
        one_total = [
            reader + floor for reader, floor in zip(readers[layers], one_launch)
        ]
        two_total = [
            reader + floor for reader, floor in zip(readers[layers], two_launch)
        ]
        result["arms"][f"n{layers}"] = {
            "allowance_us_per_selected_layer": interval(allowances[layers]),
            "reader_delta_us_per_selected_layer": interval(readers[layers]),
            "frontend_remainder_us_per_selected_layer": interval(remainder),
            "optimistic_one_launch_total_delta_us_per_selected_layer": interval(
                one_total
            ),
            "optimistic_two_launch_total_delta_us_per_selected_layer": interval(
                two_total
            ),
            "one_launch_floor_fits": (
                interval(one_launch)["upper_95"] <= interval(remainder)["lower_95"]
            ),
            "two_launch_floor_fits": (
                interval(two_launch)["upper_95"] <= interval(remainder)["lower_95"]
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    root = args.evidence_root.resolve()
    endpoint_requests: dict[int, list[dict[str, Any]]] = {}
    for context, run_name in ENDPOINT_RUNS.items():
        validate_run_config(
            load_json(root / "all-fp8" / run_name / "run_config.json"), context
        )
        requests = load_jsonl(root / "all-fp8" / run_name / "raw.jsonl")
        validate_requests(requests, context)
        endpoint_requests[context] = requests

    before_long = internal_accept_length(
        load_json(
            root
            / "all-fp8"
            / ENDPOINT_RUNS[37932]
            / "server_info.json"
        )
    )
    after_both = internal_accept_length(
        load_json(root / "all-fp8" / "runtime-proof" / "ct13" / "server-info.json")
    )
    exact_stream = response_accept_length(endpoint_requests[10219])
    cumulative_stream = response_accept_length(
        endpoint_requests[10219] + endpoint_requests[37932]
    )
    exact_relative_error = abs(exact_stream / before_long - 1.0)
    cumulative_relative_error = abs(cumulative_stream / after_both - 1.0)
    if exact_relative_error > 0.01 or cumulative_relative_error > 0.01:
        raise ValueError(
            "response-derived acceptance failed the 1% server-counter validation"
        )

    result: dict[str, Any] = {
        "status": "PASS",
        "experiment": "H41_W1_JOINT_FEASIBILITY",
        "evidence_root": str(root),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "formula": (
            "R=0.8*0.05*mean_TPOT_ms*accepted_tokens_per_forward*1000/layers; "
            "B=R-reader_delta"
        ),
        "acceptance_validation": {
            "exact_stream_aggregate": exact_stream,
            "server_before_long": before_long,
            "exact_relative_error": exact_relative_error,
            "cumulative_stream_aggregate": cumulative_stream,
            "server_after_both": after_both,
            "cumulative_relative_error": cumulative_relative_error,
        },
        "windows": {},
    }

    rng = random.Random(args.seed)
    for context in ENDPOINT_RUNS:
        reader_points: dict[int, list[float]] = {}
        for layers in SELECTED_LAYERS:
            reader_points[layers] = []
            for sequence in SEQUENCES:
                reader_result = load_json(
                    root
                    / "isolated"
                    / "reader"
                    / "decision"
                    / f"n{layers}"
                    / f"context{context}"
                    / f"seq{sequence:02}"
                    / "result.json"
                )
                validate_reader_result(
                    reader_result,
                    context=context,
                    layers=layers,
                    sequence=sequence,
                )
                reader_points[layers].append(
                    float(reader_result["candidate_minus_control_per_layer_us"])
                )

        frontend = []
        for sequence in SEQUENCES:
            frontend_result = load_json(
                root
                / "isolated"
                / "frontend"
                / f"context{context}"
                / f"seq{sequence:02}"
                / "result.json"
            )
            validate_frontend_result(
                frontend_result, context=context, sequence=sequence
            )
            frontend.append(frontend_result)
        one_launch_points = [
            float(item["one_launch_floor_mean_us"])
            - float(item["normal_combined_mean_us"])
            for item in frontend
        ]
        two_launch_points = [
            float(item["two_launch_floor_mean_us"])
            - float(item["normal_combined_mean_us"])
            for item in frontend
        ]
        result["windows"][str(context)] = bootstrap_window(
            requests=endpoint_requests[context],
            reader_points=reader_points,
            one_launch_points=one_launch_points,
            two_launch_points=two_launch_points,
            draws=args.bootstrap_draws,
            rng=rng,
        )

    n14_pass = all(
        window["arms"]["n14"]["one_launch_floor_fits"]
        for window in result["windows"].values()
    )
    n19_optimistic_reject = all(
        window["arms"]["n19"][
            "optimistic_one_launch_total_delta_us_per_selected_layer"
        ]["lower_95"]
        > window["arms"]["n19"]["allowance_us_per_selected_layer"]["upper_95"]
        for window in result["windows"].values()
    )
    result["decision"] = {
        "n14_native_one_launch_is_physically_credible": n14_pass,
        "n19_rejected_even_at_optimistic_one_launch_floor": n19_optimistic_reject,
        "authorize_w2_native_sm100_implementation": n14_pass,
        "interpretation": (
            "A W1 pass authorizes implementation only; it is not an endpoint speed, "
            "quality, or production result."
        ),
    }
    # N19 surviving would keep it as an equal-material comparison arm; it must
    # not veto a physically credible N14 implementation.  In the present data
    # N19 is nevertheless rejected even against the optimistic launch floor.
    if not n14_pass:
        result["status"] = "FAIL"

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
