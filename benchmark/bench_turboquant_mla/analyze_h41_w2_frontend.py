"""Analyze the H41 W2 independent-process timing gate."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

REMAINDER_LOWER_US = {10219: 22.9462, 37932: 13.5240}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], *, seed: int, draws: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choices(values, k=len(values))) for _ in range(draws)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def load_sequence(root: Path, context: int, warps: int, sequence: int) -> dict[str, Any]:
    path = root / f"context{context}" / f"warps{warps}" / f"seq{sequence:02d}" / "result.json"
    value = json.loads(path.read_text())
    assert value["status"] == "TIMING_ONLY"
    assert value["experiment"] == "H41_W2_REAL_COMBINED_FRONTEND"
    assert value["context"] == context
    assert value["tokens"] == 5
    assert value["warps"] == warps
    assert value["warmups_per_graph"] >= 100
    assert value["samples_per_arm"] * value["replays_per_sample"] >= 2000
    expected_order = "candidate-first" if sequence == 2 else "control-first"
    assert value["allocation_order"] == expected_order
    assert abs(value["control_flank_drift_fraction"]) <= 0.02
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    analyses: list[dict[str, Any]] = []
    for context in (10219, 37932):
        for warps in (4, 8):
            sequences = [
                load_sequence(args.root, context, warps, sequence)
                for sequence in (1, 2, 3)
            ]
            deltas = [value["candidate_minus_control_us"] for value in sequences]
            lower, upper = bootstrap_mean_ci(
                deltas,
                seed=args.seed + context + warps,
                draws=args.draws,
            )
            remainder = REMAINDER_LOWER_US[context]
            analyses.append(
                {
                    "context": context,
                    "warps": warps,
                    "allocation_orders": [
                        value["allocation_order"] for value in sequences
                    ],
                    "control_mean_us": statistics.fmean(
                        value["control_mean_us"] for value in sequences
                    ),
                    "candidate_mean_us": statistics.fmean(
                        value["candidate_mean_us"] for value in sequences
                    ),
                    "sequence_delta_us": deltas,
                    "delta_mean_us": statistics.fmean(deltas),
                    "delta_bootstrap_95_us": [lower, upper],
                    "w1_remainder_95_lower_us": remainder,
                    "headroom_at_delta_upper_us": remainder - upper,
                    "max_abs_control_flank_drift_fraction": max(
                        abs(value["control_flank_drift_fraction"])
                        for value in sequences
                    ),
                    "gate": "PASS" if upper < remainder else "FAIL",
                }
            )

    by_warps = {
        warps: {
            "all_contexts_pass": all(
                item["gate"] == "PASS" for item in analyses if item["warps"] == warps
            ),
            "mean_delta_across_contexts_us": statistics.fmean(
                item["delta_mean_us"] for item in analyses if item["warps"] == warps
            ),
            "worst_headroom_us": min(
                item["headroom_at_delta_upper_us"]
                for item in analyses
                if item["warps"] == warps
            ),
        }
        for warps in (4, 8)
    }
    result = {
        "status": "PASS"
        if all(item["gate"] == "PASS" for item in analyses)
        else "FAIL",
        "method": (
            "Independent-process cluster bootstrap of three C/T/C paired mean deltas; "
            "50,000 deterministic percentile draws by default."
        ),
        "draws": args.draws,
        "seed": args.seed,
        "analyses": analyses,
        "warp_summary": by_warps,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
