#!/usr/bin/env python3
"""Analyze the sealed H43 I3 FP8/E2M1/H43/FP8 endpoint bracket."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import random
import statistics
import sys


ROLES = ("fp8_pre", "e2m1", "h43", "fp8_post")
MEASURED_REQUESTS = 10
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20260730
# FP8-pre first measurement through FP8-post last measurement: 122 minutes of
# predeclared timed-role budget plus the full 13-minute shared reserve.
MAX_TIMED_BRACKET_S = 135 * 60
EXPECTED_REQUEST_FIELDS = {
    "cache_mode": "unique-prefix",
    "completion_tokens_source": "usage",
    "concurrency": 1,
    "ignore_eos": False,
    "model": "moonshotai/Kimi-K2.6",
    "output_tokens_target": 512,
    "prompt_tokens_target": 16000,
    "scenario_id": "s0001-synthetic-synthetic-p16000-o512-c1",
    "schema_version": 1,
    "sglang_rid": "",
    "task": "synthetic",
    "temperature": 0.0,
    "thinking": "unset",
    "top_p": 1.0,
    "workload": "synthetic",
}


def percentile_type7(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_timestamp(value: str) -> float:
    return dt.datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=dt.timezone.utc
    ).timestamp()


def find_raw(root: Path, role: str) -> Path:
    matches = sorted((root / role).glob("**/raw.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"{role}: expected one raw.jsonl, found {len(matches)}")
    return matches[0]


def load_role(root: Path, role: str, prompt_hashes: list[str]) -> dict[str, object]:
    raw_path = find_raw(root, role)
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != MEASURED_REQUESTS + 2:
        raise ValueError(f"{role}: expected 12 records, found {len(rows)}")
    if [row["prompt_sha256"] for row in rows] != prompt_hashes:
        raise ValueError(f"{role}: frozen prompt hashes differ")
    if [bool(row["is_warmup"]) for row in rows] != [True, True] + [False] * 10:
        raise ValueError(f"{role}: warmup/measured order differs")
    if [int(row["request_index"]) for row in rows] != [-1, -2] + list(range(10)):
        raise ValueError(f"{role}: request indices differ")
    for field, expected in EXPECTED_REQUEST_FIELDS.items():
        if any(
            type(row.get(field)) is not type(expected) or row.get(field) != expected
            for row in rows
        ):
            raise ValueError(f"{role}: frozen request field {field!r} differs")
    measured = [row for row in rows if not row["is_warmup"]]

    required_finite = ("tpot_ms", "ttft_ms", "e2e_ms", "decode_tok_s")
    correctness = all(
        row.get("ok") is True
        and row.get("http_status") == 200
        and row.get("error") is None
        and row.get("finish_reason") == "length"
        and row.get("completion_tokens") == 512
        and row.get("response_chars", 0) > 0
        and row.get("response_chunks", 0) > 0
        and all(
            math.isfinite(float(row[field])) and float(row[field]) > 0.0
            for field in required_finite
        )
        for row in rows
    )
    if not correctness:
        raise ValueError(f"{role}: request correctness/coherence contract failed")
    try:
        recorded_epochs = [parse_timestamp(row["recorded_at_utc"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{role}: request timestamp contract failed") from error
    if any(left >= right for left, right in zip(recorded_epochs, recorded_epochs[1:])):
        raise ValueError(f"{role}: request timestamps are not strictly increasing")

    first_start = parse_timestamp(measured[0]["recorded_at_utc"]) - (
        float(measured[0]["e2e_ms"]) / 1000.0
    )
    last_end = parse_timestamp(measured[-1]["recorded_at_utc"])
    midpoint = (first_start + last_end) / 2.0
    acceptance = [
        float(row["completion_tokens"]) / float(row["response_chunks"])
        for row in measured
    ]
    target_verify_ms = [
        float(row["tpot_ms"])
        * (float(row["completion_tokens"]) - 1.0)
        / float(row["response_chunks"])
        for row in measured
    ]
    result: dict[str, object] = {
        "raw_path": str(raw_path),
        "midpoint_epoch": midpoint,
        "midpoint_utc": dt.datetime.fromtimestamp(
            midpoint, tz=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "first_start_epoch": first_start,
        "last_end_epoch": last_end,
        "measured": measured,
        "tpot_ms": [float(row["tpot_ms"]) for row in measured],
        "ttft_ms": [float(row["ttft_ms"]) for row in measured],
        "decode_tok_s": [float(row["decode_tok_s"]) for row in measured],
        "acceptance": acceptance,
        "target_verify_ms": target_verify_ms,
        "prompt_tokens": [int(row["prompt_tokens"]) for row in measured],
        "completion_tokens": [int(row["completion_tokens"]) for row in measured],
        "response_sha256": [str(row["response_sha256"]) for row in measured],
        "aggregate_e2e_tok_s": (
            sum(int(row["completion_tokens"]) for row in measured)
            * 1000.0
            / sum(float(row["e2e_ms"]) for row in measured)
        ),
    }
    return result


def series_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def interpolate(
    before: list[float], after: list[float], weight_after: float
) -> list[float]:
    if len(before) != len(after):
        raise ValueError("control series length mismatch")
    return [
        left * (1.0 - weight_after) + right * weight_after
        for left, right in zip(before, after, strict=True)
    ]


def ratio_delta(candidate: list[float], control: list[float]) -> float:
    return statistics.mean(candidate) / statistics.mean(control) - 1.0


def bootstrap_interval(
    candidate: list[float], control: list[float]
) -> tuple[float, float]:
    if len(candidate) != len(control):
        raise ValueError("bootstrap series length mismatch")
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(candidate)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = [rng.randrange(count) for _ in range(count)]
        candidate_mean = statistics.mean(candidate[index] for index in indices)
        control_mean = statistics.mean(control[index] for index in indices)
        draws.append(candidate_mean / control_mean - 1.0)
    return percentile_type7(draws, 0.025), percentile_type7(draws, 0.975)


def band(value: float, pass_edge: float, fail_edge: float) -> str:
    if value <= pass_edge:
        return "PASS"
    if value > fail_edge:
        return "FAIL_PERF"
    return "NO_DECISION"


def analyze(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_hashes = list(manifest["request_prompt_sha256"])
    if len(prompt_hashes) != MEASURED_REQUESTS + 2:
        raise ValueError("frozen manifest must contain exactly 12 prompt hashes")
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("frozen manifest prompt hashes must be unique")
    roles = {role: load_role(root, role, prompt_hashes) for role in ROLES}
    before = roles["fp8_pre"]
    after = roles["fp8_post"]
    before_mid = float(before["midpoint_epoch"])
    after_mid = float(after["midpoint_epoch"])
    role_midpoints = [float(roles[role]["midpoint_epoch"]) for role in ROLES]
    if not all(
        left < right for left, right in zip(role_midpoints, role_midpoints[1:])
    ):
        raise ValueError("timed role order is invalid")
    reference_prompt_tokens = roles["fp8_pre"]["prompt_tokens"]
    if any(
        roles[role]["prompt_tokens"] != reference_prompt_tokens for role in ROLES[1:]
    ):
        raise ValueError("actual prompt-token counts differ across roles")
    timed_bracket_span_s = float(roles["fp8_post"]["last_end_epoch"]) - float(
        roles["fp8_pre"]["first_start_epoch"]
    )
    timed_bracket_valid = timed_bracket_span_s <= MAX_TIMED_BRACKET_S

    control_metrics = ("tpot_ms", "ttft_ms", "acceptance", "target_verify_ms")
    drift = {
        metric: statistics.mean(after[metric]) / statistics.mean(before[metric]) - 1.0
        for metric in control_metrics
    }
    drift_valid = (
        abs(drift["tpot_ms"]) <= 0.01
        and abs(drift["ttft_ms"]) <= 0.02
        and abs(drift["acceptance"]) <= 0.05
        and abs(drift["target_verify_ms"]) <= 0.02
    )

    comparisons: dict[str, object] = {}
    for role in ("e2m1", "h43"):
        candidate = roles[role]
        weight_after = (
            float(candidate["midpoint_epoch"]) - before_mid
        ) / (after_mid - before_mid)
        if not 0.0 <= weight_after <= 1.0:
            raise ValueError(f"{role}: midpoint falls outside FP8 bracket")
        controls = {
            metric: interpolate(before[metric], after[metric], weight_after)
            for metric in control_metrics
        }
        tpot_delta = ratio_delta(candidate["tpot_ms"], controls["tpot_ms"])
        ttft_delta = ratio_delta(candidate["ttft_ms"], controls["ttft_ms"])
        ratios = [
            observed / expected
            for observed, expected in zip(
                candidate["tpot_ms"], controls["tpot_ms"], strict=True
            )
        ]
        bootstrap_low, bootstrap_high = bootstrap_interval(
            candidate["tpot_ms"], controls["tpot_ms"]
        )
        comparisons[role] = {
            "fp8_post_weight": weight_after,
            "tpot_delta": tpot_delta,
            "ttft_delta": ttft_delta,
            "response_acceptance_delta": ratio_delta(
                candidate["acceptance"], controls["acceptance"]
            ),
            "target_verify_ms_delta": ratio_delta(
                candidate["target_verify_ms"], controls["target_verify_ms"]
            ),
            "per_index_tpot_ratio": ratios,
            "per_index_tpot_ratio_p95_type7": percentile_type7(ratios, 0.95),
            "per_index_tpot_ratio_max": max(ratios),
            "tpot_delta_bootstrap_95": [bootstrap_low, bootstrap_high],
            "interpolated_fp8": {
                metric: series_summary(controls[metric]) for metric in control_metrics
            },
        }

    h43 = comparisons["h43"]
    tpot_band = band(float(h43["tpot_delta"]), 0.045, 0.055)
    ttft_band = band(float(h43["ttft_delta"]), 0.09, 0.11)
    tail_pass = (
        float(h43["per_index_tpot_ratio_p95_type7"]) <= 1.10
        and float(h43["per_index_tpot_ratio_max"]) <= 1.15
    )
    if not timed_bracket_valid:
        verdict = "NO_DECISION"
        reason = "timed-bracket-span"
    elif not drift_valid:
        verdict = "NO_DECISION"
        reason = "fp8-flank-drift"
    elif tpot_band == "FAIL_PERF" or ttft_band == "FAIL_PERF" or not tail_pass:
        verdict = "FAIL_PERF"
        reason = "performance-gate"
    elif tpot_band == "NO_DECISION" or ttft_band == "NO_DECISION":
        verdict = "NO_DECISION"
        reason = "threshold-band"
    else:
        verdict = "PASS_PERF"
        reason = "all-automated-matched-performance-gates"

    role_summary = {}
    for role, values in roles.items():
        role_summary[role] = {
            "raw_path": values["raw_path"],
            "midpoint_utc": values["midpoint_utc"],
            "tpot_ms": series_summary(values["tpot_ms"]),
            "ttft_ms": series_summary(values["ttft_ms"]),
            "decode_tok_s": series_summary(values["decode_tok_s"]),
            "response_acceptance": series_summary(values["acceptance"]),
            "target_verify_ms": series_summary(values["target_verify_ms"]),
            "aggregate_e2e_tok_s": values["aggregate_e2e_tok_s"],
            "prompt_tokens": values["prompt_tokens"],
            "completion_tokens": values["completion_tokens"],
            "response_sha256": values["response_sha256"],
        }

    return {
        "schema": 1,
        "verdict": verdict,
        "reason": reason,
        "verdict_scope": "automated-matched-performance-only",
        "external_gates_required_for_endpoint_pass": [
            "candidate-response-acceptance-and-target-verification-explanation",
            "source-and-memory-attestation",
            "stage1-quality-and-graph-gates",
            "coordinator-status-and-machine-health",
        ],
        "frozen_prompt_hashes_match": True,
        "measured_requests_per_role": MEASURED_REQUESTS,
        "role_order": list(ROLES),
        "role_summary": role_summary,
        "fp8_flank_drift": drift,
        "fp8_flank_valid": drift_valid,
        "timed_bracket_span_s": timed_bracket_span_s,
        "timed_bracket_valid": timed_bracket_valid,
        "timed_bracket_limit_s": MAX_TIMED_BRACKET_S,
        "comparisons": comparisons,
        "h43_tpot_band": tpot_band,
        "h43_ttft_band": ttft_band,
        "h43_tail_pass": tail_pass,
        "h43_attribution_delta_vs_e2m1": (
            float(comparisons["h43"]["tpot_delta"])
            - float(comparisons["e2m1"]["tpot_delta"])
        ),
        "contractual_memory_claims": {
            "source": "predeclared-contract-not-measured-by-this-analyzer",
            "net_saved_vs_fp8_gib_per_rank": 0.778770447,
            "net_saved_vs_fp8_gib_tp8": 6.230163574,
            "persistent_saved_vs_e2m1_gib_per_rank": 0.160217285,
            "persistent_saved_vs_e2m1_gib_tp8": 1.281738281,
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "matched-request-index",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.manifest)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
