#!/usr/bin/env python3
"""Analyze H44 Window A and emit its immutable recovery-allowance manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Any

ROLES = ("fp8_pre", "e2m1_n14", "h43_n14", "fp8_post")
MEASURED_REQUESTS = 30
WARMUP_REQUESTS = 2
MIN_RETAINED = 27
ENDPOINT_INDICES = tuple(range(10))
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20260731
EXPECTED_PROMPT_TOKENS = 10_218
EXPECTED_COMPLETION_TOKENS = 512
RFC3339_NANO = re.compile(
    rb"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z(?:\s|$)"
)

DRIFT_LIMITS = {
    "tpot_ms": 0.01,
    "ttft_ms": 0.02,
    "emitted_interval_yield": 0.05,
    "decode_interval_per_verify_ms": 0.02,
}


class AnalysisError(ValueError):
    """Window-A evidence cannot support the frozen analysis contract."""


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisError(f"{label} is not a nonnegative integer")
    return value


def exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} is not an integer")
    return value


def finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} is not a nonnegative finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise AnalysisError(f"{label} is not a nonnegative finite number")
    return result


def require_close(
    observed: Any, expected: float, label: str, *, allow_zero: bool = False
) -> float:
    value = (
        finite_nonnegative(observed, label)
        if allow_zero
        else finite_positive(observed, label)
    )
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AnalysisError(
            f"{label} disagrees with exact inputs: expected {expected}, observed {value}"
        )
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def serialized_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def parse_log_realtime_ns(line: bytes) -> int:
    match = RFC3339_NANO.match(line)
    if match is None:
        raise AnalysisError("instrumented log line lacks an RFC3339 UTC timestamp")
    seconds = int(
        dt.datetime.strptime(match.group(1).decode(), "%Y-%m-%dT%H:%M:%S")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )
    fractional = (match.group(2) or b"").ljust(9, b"0")
    return seconds * 1_000_000_000 + int(fractional or b"0")


def load_hashed_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    observed = sha256_bytes(raw)
    if observed != expected_sha256:
        raise AnalysisError(
            f"{label} hash mismatch: expected {expected_sha256}, observed {observed}"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} is not a JSON object")
    return value


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} is not a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise AnalysisError(f"{label} is not a positive finite number")
    return result


def percentile_type7(values: list[float], probability: float) -> float:
    if not values:
        raise AnalysisError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ratio_of_means(candidate: list[float], control: list[float]) -> float:
    if not candidate or len(candidate) != len(control):
        raise AnalysisError("ratio-of-means series length mismatch")
    return statistics.mean(candidate) / statistics.mean(control)


def interpolate(left: float, right: float, weight_right: float) -> float:
    return left * (1.0 - weight_right) + right * weight_right


def classify_band(ratio: float, pass_edge: float, fail_edge: float) -> str:
    if ratio <= pass_edge:
        return "PASS"
    if ratio > fail_edge:
        return "FAIL"
    return "NO_DECISION"


def top3_gap_details(events: list[dict[str, Any]]) -> list[dict[str, int]]:
    text_events = []
    for event in events:
        if event.get("has_text"):
            monotonic_ns = event.get("monotonic_ns")
            realtime_ns = event.get("realtime_ns")
            if (
                isinstance(monotonic_ns, bool)
                or not isinstance(monotonic_ns, int)
                or isinstance(realtime_ns, bool)
                or not isinstance(realtime_ns, int)
            ):
                raise AnalysisError("text SSE event lacks exact clock timestamps")
            text_events.append((monotonic_ns, realtime_ns))
    if len(text_events) < 2:
        raise AnalysisError("tail classifier needs at least two text SSE events")
    if any(left[0] > right[0] for left, right in zip(text_events, text_events[1:])):
        raise AnalysisError("text SSE event timestamps regress")
    interval = text_events[-1][0] - text_events[0][0]
    if interval <= 0:
        raise AnalysisError("tail classifier decode interval is not positive")
    gaps = [
        {
            "gap_monotonic_ns": right[0] - left[0],
            "start_realtime_ns": left[1],
            "end_realtime_ns": right[1],
            "decode_interval_ns": interval,
        }
        for left, right in zip(text_events, text_events[1:])
    ]
    return sorted(gaps, key=lambda gap: gap["gap_monotonic_ns"], reverse=True)[:3]


def top3_gap_share(events: list[dict[str, Any]]) -> float:
    gaps = top3_gap_details(events)
    return sum(gap["gap_monotonic_ns"] for gap in gaps) / gaps[0]["decode_interval_ns"]


def classify_tail_event(
    candidate_share: float, fp8_share: float, discrete_owner: str | None
) -> dict[str, Any]:
    delta = candidate_share - fp8_share
    if candidate_share <= 0.10:
        classification = "uniform"
        admissible = True
    elif candidate_share >= 0.25 and delta >= 0.10:
        if discrete_owner:
            classification = "discrete-owned"
            admissible = True
        else:
            classification = "discrete-unowned"
            admissible = False
    else:
        classification = "no-decision"
        admissible = False
    return {
        "candidate_top3_gap_share": candidate_share,
        "interpolated_fp8_top3_gap_share": fp8_share,
        "share_delta": delta,
        "classification": classification,
        "discrete_owner": discrete_owner,
        "admissible_for_window_b": admissible,
    }


def align_discrete_stall_owner(
    *,
    owner: str,
    evidence: Any,
    role_data: dict[str, Any],
    row: dict[str, Any],
    required_owned_gap_share: float,
) -> tuple[str | None, dict[str, Any]]:
    result = {"passed": False, "reason": "missing-or-malformed-owner-evidence"}
    if not isinstance(evidence, dict) or evidence.get("owner") != owner:
        return None, result
    node = evidence.get("node")
    if node not in ("ct13", "ct14"):
        result["reason"] = "invalid-owner-node"
        return None, result
    event_kind = evidence.get("event_kind")
    if not isinstance(event_kind, str) or not event_kind:
        result["reason"] = "missing-owner-event-kind"
        return None, result
    log_line_text = evidence.get("log_line_utf8")
    if (
        not isinstance(log_line_text, str)
        or not log_line_text
        or "\n" in log_line_text
        or "\r" in log_line_text
    ):
        result["reason"] = "missing-or-multiline-owner-log-line"
        return None, result
    log_line = log_line_text.encode("utf-8")
    if (
        evidence.get("log_line_sha256") != sha256_bytes(log_line)
        or log_line not in role_data["log_lines_by_node"][node]
    ):
        result["reason"] = "owner log line is absent or hash-mismatched"
        return None, result
    if event_kind.encode() not in log_line or owner.encode() not in log_line:
        result["reason"] = "owner log line lacks the predeclared owner/event kind"
        return None, result
    try:
        event_realtime_ns = exact_nonnegative_int(
            evidence.get("event_realtime_ns"), "owner event realtime"
        )
        gap_rank = exact_nonnegative_int(
            evidence.get("gap_rank"), "owner event gap rank"
        )
    except AnalysisError as exc:
        result["reason"] = str(exc)
        return None, result
    try:
        parsed_log_realtime_ns = parse_log_realtime_ns(log_line)
    except AnalysisError as exc:
        result["reason"] = str(exc)
        return None, result
    if parsed_log_realtime_ns != event_realtime_ns:
        result["reason"] = "owner event timestamp differs from retained log line"
        return None, result
    gaps = top3_gap_details(row["stream"]["events"])
    if not 1 <= gap_rank <= len(gaps):
        result["reason"] = "owner gap rank is outside top-three gaps"
        return None, result
    log_capture = role_data["log_capture_by_node"][node]
    if evidence.get("source_log_sha256") != log_capture["log_sha256"]:
        result["reason"] = "owner event log hash differs"
        return None, result
    if not (
        log_capture["requested_since_realtime_ns"]
        <= event_realtime_ns
        <= log_capture["requested_until_realtime_ns"]
    ):
        result["reason"] = "owner event lies outside retained log range"
        return None, result
    gap = gaps[gap_rank - 1]
    owned_gap_share = gap["gap_monotonic_ns"] / gap["decode_interval_ns"]
    if owned_gap_share < required_owned_gap_share:
        result["reason"] = "selected SSE gap does not explain discrete-stall excess"
        result["owned_gap_share"] = owned_gap_share
        result["required_owned_gap_share"] = required_owned_gap_share
        return None, result
    offset = role_data["clock_offset_by_node"][node]
    node_gap_start = (
        gap["start_realtime_ns"] + offset["maximum_node_minus_client_realtime_ns"]
    )
    node_gap_end = (
        gap["end_realtime_ns"] + offset["minimum_node_minus_client_realtime_ns"]
    )
    if not node_gap_start <= event_realtime_ns <= node_gap_end:
        result["reason"] = "owner event is not aligned to selected SSE gap"
        return None, result
    result = {
        "passed": True,
        "reason": "timestamp-and-log-hash-aligned",
        "node": node,
        "event_kind": event_kind,
        "event_realtime_ns": event_realtime_ns,
        "gap_rank": gap_rank,
        "node_gap_start_realtime_ns": node_gap_start,
        "node_gap_end_realtime_ns": node_gap_end,
        "source_log_sha256": evidence["source_log_sha256"],
        "log_line_sha256": evidence["log_line_sha256"],
        "owned_gap_share": owned_gap_share,
        "required_owned_gap_share": required_owned_gap_share,
    }
    return owner, result


def find_single(root: Path, role: str, name: str) -> Path:
    matches = sorted((root / role).glob(f"**/{name}"))
    if len(matches) != 1:
        raise AnalysisError(
            f"{role}: expected exactly one {name}, observed {len(matches)}"
        )
    return matches[0]


def load_role_logs(
    role_root: Path, role: str, captures: dict[str, Any]
) -> dict[str, list[bytes]]:
    resolved_root = role_root.resolve()
    result = {}
    for node in ("ct13", "ct14"):
        capture = captures[node]
        relative = capture.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise AnalysisError(f"{role}: {node} log path is unsafe")
        candidate = role_root / relative
        path = candidate.resolve()
        if not path.is_relative_to(resolved_root):
            raise AnalysisError(f"{role}: {node} log path escapes role evidence")
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise AnalysisError(f"{role}: cannot inspect {node} log") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise AnalysisError(f"{role}: {node} log is not a regular sealed file")
        raw = path.read_bytes()
        if sha256_bytes(raw) != capture.get("log_sha256"):
            raise AnalysisError(f"{role}: {node} retained log hash differs")
        result[node] = raw.splitlines()
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise AnalysisError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def validate_role_manifest(role: str, role_manifest: dict[str, Any]) -> None:
    required_true = (
        "qualification_passed",
        "health_passed",
        "provenance_passed",
        "log_coverage_passed",
        "clock_offset_passed",
        "container_identity_passed",
        "config_passed",
        "role_deadline_passed",
    )
    for field in required_true:
        if role_manifest.get(field) is not True:
            raise AnalysisError(f"{role}: run-manifest gate {field} did not pass")
    if role == "h43_n14":
        for field in ("h43_attestation_passed", "memory_accounting_passed"):
            if role_manifest.get(field) is not True:
                raise AnalysisError(f"{role}: run-manifest gate {field} did not pass")
    for field in ("raw_jsonl_sha256", "qualification_sha256"):
        value = role_manifest.get(field)
        if not is_sha256(value):
            raise AnalysisError(f"{role}: run manifest lacks {field}")


def validate_log_capture(
    role: str,
    role_manifest: dict[str, Any],
    first_start_realtime_ns: int,
    last_end_realtime_ns: int,
) -> dict[str, dict[str, int]]:
    captures = role_manifest.get("log_capture_by_node")
    if not isinstance(captures, dict) or set(captures) != {"ct13", "ct14"}:
        raise AnalysisError(f"{role}: log-capture node set differs")
    offsets = role_manifest.get("clock_offset_by_node")
    if not isinstance(offsets, dict) or set(offsets) != {"ct13", "ct14"}:
        raise AnalysisError(f"{role}: clock-offset node set differs")
    normalized_offsets = {}
    for node, capture in captures.items():
        if not isinstance(capture, dict):
            raise AnalysisError(f"{role}: {node} log capture is not an object")
        offset = offsets[node]
        if not isinstance(offset, dict):
            raise AnalysisError(f"{role}: {node} clock offset is not an object")
        minimum_offset_ns = exact_int(
            offset.get("minimum_node_minus_client_realtime_ns"),
            f"{role}: {node} minimum clock offset",
        )
        maximum_offset_ns = exact_int(
            offset.get("maximum_node_minus_client_realtime_ns"),
            f"{role}: {node} maximum clock offset",
        )
        if minimum_offset_ns > maximum_offset_ns:
            raise AnalysisError(f"{role}: {node} clock-offset bounds are inverted")
        sample_count = exact_nonnegative_int(
            offset.get("sample_count"), f"{role}: {node} clock sample count"
        )
        before_sample_ns = exact_nonnegative_int(
            offset.get("before_sample_client_realtime_ns"),
            f"{role}: {node} before clock sample",
        )
        after_sample_ns = exact_nonnegative_int(
            offset.get("after_sample_client_realtime_ns"),
            f"{role}: {node} after clock sample",
        )
        if (
            sample_count < 2
            or before_sample_ns > first_start_realtime_ns
            or after_sample_ns < last_end_realtime_ns
        ):
            raise AnalysisError(f"{role}: {node} clock samples do not bracket role")
        if node == "ct13" and (minimum_offset_ns != 0 or maximum_offset_ns != 0):
            raise AnalysisError(f"{role}: CT13/client clock offset must be zero")
        required_since = first_start_realtime_ns - 1_000_000_000 + minimum_offset_ns
        required_until = last_end_realtime_ns + 1_000_000_000 + maximum_offset_ns
        since_ns = exact_nonnegative_int(
            capture.get("requested_since_realtime_ns"),
            f"{role}: {node} requested log start",
        )
        until_ns = exact_nonnegative_int(
            capture.get("requested_until_realtime_ns"),
            f"{role}: {node} requested log end",
        )
        if since_ns > required_since or until_ns < required_until:
            raise AnalysisError(f"{role}: {node} log range does not enclose role")
        if since_ns >= until_ns:
            raise AnalysisError(f"{role}: {node} log range is invalid")
        if capture.get("retrieval_error") is not None:
            raise AnalysisError(f"{role}: {node} log retrieval reported an error")
        if capture.get("truncated") is not False:
            raise AnalysisError(f"{role}: {node} log capture is truncated or unknown")
        if capture.get("log_driver_unchanged") is not True:
            raise AnalysisError(f"{role}: {node} log driver changed or is unknown")
        if capture.get("container_identity_unchanged") is not True:
            raise AnalysisError(f"{role}: {node} log container identity differs")
        if not is_sha256(capture.get("log_sha256")):
            raise AnalysisError(f"{role}: {node} log hash is invalid")
        normalized_offsets[node] = {
            "minimum_node_minus_client_realtime_ns": minimum_offset_ns,
            "maximum_node_minus_client_realtime_ns": maximum_offset_ns,
        }
    return normalized_offsets


def validate_row_identity(
    row: dict[str, Any], expected_identity: dict[str, str], role: str
) -> None:
    identity = row.get("identity")
    if not isinstance(identity, dict):
        raise AnalysisError(f"{role}: row lacks identity")
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise AnalysisError(f"{role}: row identity {field} differs")


def validate_exact_row_counters(row: dict[str, Any], role: str, index: int) -> None:
    stream = row["stream"]
    delta = row["counter_delta"]
    reconciliation = row["counter_reconciliation"]
    completion = exact_nonnegative_int(
        stream.get("completion_tokens"), f"{role}: row {index} completion tokens"
    )
    forwards = exact_nonnegative_int(
        delta.get("forward_ct"), f"{role}: row {index} verification forwards"
    )
    internal = exact_nonnegative_int(
        delta.get("internal_accept_tokens"),
        f"{role}: row {index} internal accept tokens",
    )
    if completion == 0 or forwards == 0 or internal < forwards:
        raise AnalysisError(f"{role}: row {index} has impossible speculative counters")

    before = row.get("counter_before")
    after = row.get("counter_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise AnalysisError(f"{role}: row {index} lacks counter snapshots")
    draft_tokens = exact_nonnegative_int(
        before.get("speculative_num_draft_tokens"),
        f"{role}: row {index} configured draft tokens",
    )
    if draft_tokens < 2 or after.get("speculative_num_draft_tokens") != draft_tokens:
        raise AnalysisError(f"{role}: row {index} draft-token configuration changed")
    boundary = internal - completion
    if not -1 <= boundary <= draft_tokens - 2:
        raise AnalysisError(f"{role}: row {index} boundary adjustment is invalid")

    exact_nonnegative_int(
        reconciliation.get("internal_accept_tokens"),
        f"{role}: row {index} reconciled internal tokens",
    )
    if reconciliation["internal_accept_tokens"] != internal:
        raise AnalysisError(
            f"{role}: row {index} internal counter reconciliation differs"
        )
    integer_expectations = {
        "emitted_completion_tokens": completion,
        "emitted_intervals": completion - 1,
        "boundary_adjustment_tokens": boundary,
        "boundary_adjustment_min": -1,
        "boundary_adjustment_max": draft_tokens - 2,
    }
    for field, expected in integer_expectations.items():
        if reconciliation.get(field) != expected:
            raise AnalysisError(
                f"{role}: row {index} reconciliation field {field} differs"
            )
    require_close(
        reconciliation.get("internal_accept_length"),
        internal / forwards,
        f"{role}: row {index} internal accept length",
    )
    require_close(
        reconciliation.get("response_accept_length"),
        completion / forwards,
        f"{role}: row {index} response accept length",
    )
    require_close(
        reconciliation.get("emitted_interval_yield"),
        (completion - 1) / forwards,
        f"{role}: row {index} emitted interval yield",
        allow_zero=True,
    )

    events = stream.get("events")
    if not isinstance(events, list):
        raise AnalysisError(f"{role}: row {index} lacks ordered SSE events")
    started_mono = exact_nonnegative_int(
        stream.get("started_monotonic_ns"),
        f"{role}: row {index} stream start monotonic time",
    )
    ended_mono = exact_nonnegative_int(
        stream.get("ended_monotonic_ns"),
        f"{role}: row {index} stream end monotonic time",
    )
    if started_mono >= ended_mono:
        raise AnalysisError(f"{role}: row {index} stream monotonic bounds differ")
    started_real = exact_nonnegative_int(
        stream.get("started_realtime_ns"),
        f"{role}: row {index} stream start realtime",
    )
    ended_real = exact_nonnegative_int(
        stream.get("ended_realtime_ns"),
        f"{role}: row {index} stream end realtime",
    )
    if started_real >= ended_real:
        raise AnalysisError(f"{role}: row {index} stream realtime bounds differ")
    finish_reasons = []
    for sequence, event in enumerate(events):
        if not isinstance(event, dict) or event.get("sequence") != sequence:
            raise AnalysisError(f"{role}: row {index} SSE sequence differs")
        timestamp = event.get("monotonic_ns")
        realtime_ns = event.get("realtime_ns")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or not started_mono <= timestamp <= ended_mono
        ):
            raise AnalysisError(f"{role}: row {index} SSE event lies outside stream")
        if (
            isinstance(realtime_ns, bool)
            or not isinstance(realtime_ns, int)
            or not started_real <= realtime_ns <= ended_real
        ):
            raise AnalysisError(f"{role}: row {index} SSE realtime lies outside stream")
        if event.get("finish_reason") is not None:
            finish_reasons.append(event["finish_reason"])
    if finish_reasons != [stream.get("finish_reason")]:
        raise AnalysisError(f"{role}: row {index} SSE finish marker differs")
    text_timestamps = [
        event.get("monotonic_ns")
        for event in events
        if isinstance(event, dict) and event.get("has_text")
    ]
    if not text_timestamps or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in text_timestamps
    ):
        raise AnalysisError(f"{role}: row {index} has insufficient text SSE timing")
    if any(left > right for left, right in zip(text_timestamps, text_timestamps[1:])):
        raise AnalysisError(f"{role}: row {index} text SSE timing is not ordered")
    event_decode_ms = (text_timestamps[-1] - text_timestamps[0]) / 1_000_000
    event_ttft_ms = (text_timestamps[0] - started_mono) / 1_000_000
    require_close(
        stream.get("ttft_ms"),
        event_ttft_ms,
        f"{role}: row {index} TTFT",
    )
    require_close(
        stream.get("decode_interval_ms"),
        event_decode_ms,
        f"{role}: row {index} decode interval",
        allow_zero=True,
    )
    if completion == 1:
        if stream.get("tpot_ms") is not None:
            raise AnalysisError(f"{role}: row {index} one-token TPOT must be null")
    else:
        require_close(
            stream.get("tpot_ms"),
            event_decode_ms / (completion - 1),
            f"{role}: row {index} TPOT",
        )
    require_close(
        reconciliation.get("decode_interval_per_verify_ms"),
        event_decode_ms / forwards,
        f"{role}: row {index} decode interval per verification forward",
        allow_zero=True,
    )


def load_role(
    *,
    root: Path,
    role: str,
    role_manifest: dict[str, Any],
    prompt_hashes: list[str],
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    validate_role_manifest(role, role_manifest)
    raw_path = find_single(root, role, "raw.jsonl")
    qualification_path = find_single(root, role, "exactness_qualification.json")
    if sha256_bytes(raw_path.read_bytes()) != role_manifest["raw_jsonl_sha256"]:
        raise AnalysisError(f"{role}: raw JSONL hash mismatch")
    if (
        sha256_bytes(qualification_path.read_bytes())
        != role_manifest["qualification_sha256"]
    ):
        raise AnalysisError(f"{role}: qualification hash mismatch")
    try:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{role}: qualification is not JSON") from exc
    if not isinstance(qualification, dict):
        raise AnalysisError(f"{role}: qualification is not an object")
    if qualification.get("passed") is not True:
        raise AnalysisError(f"{role}: exactness qualification did not pass")
    if qualification.get("identity") != expected_identity:
        raise AnalysisError(f"{role}: qualification identity differs")

    rows = read_jsonl(raw_path)
    if len(rows) != MEASURED_REQUESTS + WARMUP_REQUESTS:
        raise AnalysisError(f"{role}: expected 32 rows, observed {len(rows)}")
    expected_indices = [-1, -2] + list(range(MEASURED_REQUESTS))
    if [row.get("request_index") for row in rows] != expected_indices:
        raise AnalysisError(f"{role}: request index/order contract differs")
    if [row.get("prompt_sha256") for row in rows] != prompt_hashes:
        raise AnalysisError(f"{role}: prompt hash/order contract differs")
    realtime_bounds = []
    for index, row in enumerate(rows):
        expected_warmup = index < WARMUP_REQUESTS
        if row.get("schema_version") != 2:
            raise AnalysisError(f"{role}: row {index} has unsupported schema")
        if row.get("role") != role:
            raise AnalysisError(f"{role}: row {index} carries another role")
        if row.get("is_warmup") is not expected_warmup:
            raise AnalysisError(f"{role}: row {index} warmup marker differs")
        if row.get("telemetry_valid") is not True:
            raise AnalysisError(f"{role}: row {index} is not telemetry-valid")
        if row.get("validation_errors") != []:
            raise AnalysisError(f"{role}: row {index} has validation errors")
        validate_row_identity(row, expected_identity, role)
        stream = row.get("stream")
        reconciliation = row.get("counter_reconciliation")
        delta = row.get("counter_delta")
        if not isinstance(stream, dict) or stream.get("ok") is not True:
            raise AnalysisError(f"{role}: row {index} stream failed")
        if not isinstance(reconciliation, dict) or not isinstance(delta, dict):
            raise AnalysisError(f"{role}: row {index} lacks exact counters")
        if stream.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS:
            raise AnalysisError(f"{role}: row {index} is not exact-10K")
        validate_exact_row_counters(row, role, index)
        started_real = stream.get("started_realtime_ns")
        ended_real = stream.get("ended_realtime_ns")
        if (
            isinstance(started_real, bool)
            or not isinstance(started_real, int)
            or isinstance(ended_real, bool)
            or not isinstance(ended_real, int)
            or started_real >= ended_real
        ):
            raise AnalysisError(f"{role}: row {index} realtime bounds differ")
        realtime_bounds.append((started_real, ended_real))
    if any(
        previous_end >= current_start
        for (_, previous_end), (current_start, _) in zip(
            realtime_bounds, realtime_bounds[1:]
        )
    ):
        raise AnalysisError(f"{role}: request realtime windows overlap or regress")

    measured = rows[WARMUP_REQUESTS:]
    first_start = measured[0]["stream"].get("started_realtime_ns")
    last_end = measured[-1]["stream"].get("ended_realtime_ns")
    if (
        isinstance(first_start, bool)
        or not isinstance(first_start, int)
        or isinstance(last_end, bool)
        or not isinstance(last_end, int)
    ):
        raise AnalysisError(f"{role}: role boundary timestamps are missing")
    if first_start >= last_end:
        raise AnalysisError(f"{role}: role boundary timestamps are invalid")
    clock_offsets = validate_log_capture(role, role_manifest, first_start, last_end)
    log_lines = load_role_logs(root / role, role, role_manifest["log_capture_by_node"])
    midpoint_ns = (first_start + last_end) / 2.0
    return {
        "raw_path": str(raw_path),
        "qualification_path": str(qualification_path),
        "rows": {row["request_index"]: row for row in measured},
        "midpoint_realtime_ns": midpoint_ns,
        "first_start_realtime_ns": first_start,
        "last_end_realtime_ns": last_end,
        "clock_offset_by_node": clock_offsets,
        "log_capture_by_node": role_manifest["log_capture_by_node"],
        "log_lines_by_node": log_lines,
    }


def row_metric(row: dict[str, Any], metric: str) -> float:
    if metric in ("tpot_ms", "ttft_ms", "decode_interval_ms"):
        return finite_positive(row["stream"].get(metric), metric)
    if metric in (
        "emitted_interval_yield",
        "internal_accept_length",
        "response_accept_length",
        "decode_interval_per_verify_ms",
    ):
        return finite_positive(row["counter_reconciliation"].get(metric), metric)
    raise AnalysisError(f"unknown metric {metric}")


def full_length(row: dict[str, Any]) -> bool:
    stream = row["stream"]
    expected = (
        stream.get("completion_tokens") == EXPECTED_COMPLETION_TOKENS
        and stream.get("finish_reason") == "length"
    )
    if row.get("performance_eligible") is not expected:
        raise AnalysisError("row performance_eligible disagrees with output metadata")
    quality_signal = row.get("quality_signal")
    if expected and quality_signal is not None:
        raise AnalysisError("full-length row unexpectedly carries a quality signal")
    if not expected and not isinstance(quality_signal, dict):
        raise AnalysisError("short row lacks its candidate quality signal")
    return expected


def interpolated_controls(
    roles: dict[str, dict[str, Any]],
    candidate_role: str,
    indices: list[int],
    metric: str,
) -> tuple[list[float], float]:
    pre = roles["fp8_pre"]
    post = roles["fp8_post"]
    candidate = roles[candidate_role]
    denominator = post["midpoint_realtime_ns"] - pre["midpoint_realtime_ns"]
    if denominator <= 0:
        raise AnalysisError("FP8 role midpoints are not ordered")
    weight = (
        candidate["midpoint_realtime_ns"] - pre["midpoint_realtime_ns"]
    ) / denominator
    if not 0 <= weight <= 1:
        raise AnalysisError(f"{candidate_role}: midpoint lies outside FP8 bracket")
    values = [
        interpolate(
            row_metric(pre["rows"][index], metric),
            row_metric(post["rows"][index], metric),
            weight,
        )
        for index in indices
    ]
    return values, weight


def bootstrap_intervals(
    *,
    candidate_rows: list[dict[str, Any]],
    controls: dict[str, list[float]],
    draws: int,
) -> dict[str, list[float]]:
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(candidate_rows)
    samples = {name: [] for name in ("r_tpot", "r_accept", "r_decode_wall", "r_ttft")}
    for _ in range(draws):
        selected = [rng.randrange(count) for _ in range(count)]
        cand_tpot = [row_metric(candidate_rows[i], "tpot_ms") for i in selected]
        cand_accept = [
            row_metric(candidate_rows[i], "emitted_interval_yield") for i in selected
        ]
        cand_ttft = [row_metric(candidate_rows[i], "ttft_ms") for i in selected]
        control_tpot = [controls["tpot_ms"][i] for i in selected]
        control_accept = [controls["emitted_interval_yield"][i] for i in selected]
        control_ttft = [controls["ttft_ms"][i] for i in selected]
        r_tpot = ratio_of_means(cand_tpot, control_tpot)
        r_accept = ratio_of_means(cand_accept, control_accept)
        samples["r_tpot"].append(r_tpot)
        samples["r_accept"].append(r_accept)
        samples["r_decode_wall"].append(r_tpot * r_accept)
        samples["r_ttft"].append(ratio_of_means(cand_ttft, control_ttft))
    return {
        name: [percentile_type7(values, 0.025), percentile_type7(values, 0.975)]
        for name, values in samples.items()
    }


def compare_candidate(
    *,
    roles: dict[str, dict[str, Any]],
    role: str,
    indices: list[int],
    draws: int,
    role_manifest: dict[str, Any],
) -> dict[str, Any]:
    metrics = (
        "tpot_ms",
        "ttft_ms",
        "emitted_interval_yield",
        "decode_interval_per_verify_ms",
        "internal_accept_length",
        "response_accept_length",
    )
    controls = {}
    weight = None
    for metric in metrics:
        controls[metric], observed_weight = interpolated_controls(
            roles, role, indices, metric
        )
        if weight is None:
            weight = observed_weight
        elif not math.isclose(weight, observed_weight, rel_tol=0, abs_tol=1e-15):
            raise AnalysisError("interpolation weights differ by metric")
    candidate_rows = [roles[role]["rows"][index] for index in indices]
    candidate = {
        metric: [row_metric(row, metric) for row in candidate_rows]
        for metric in metrics
    }
    r_tpot = ratio_of_means(candidate["tpot_ms"], controls["tpot_ms"])
    r_accept = ratio_of_means(
        candidate["emitted_interval_yield"], controls["emitted_interval_yield"]
    )
    r_decode = r_tpot * r_accept
    r_ttft = ratio_of_means(candidate["ttft_ms"], controls["ttft_ms"])
    r_decode_per_verify = ratio_of_means(
        candidate["decode_interval_per_verify_ms"],
        controls["decode_interval_per_verify_ms"],
    )
    per_index = []
    for offset, index in enumerate(indices):
        item_tpot = candidate["tpot_ms"][offset] / controls["tpot_ms"][offset]
        item_accept = (
            candidate["emitted_interval_yield"][offset]
            / controls["emitted_interval_yield"][offset]
        )
        per_index.append(
            {
                "request_index": index,
                "r_tpot": item_tpot,
                "r_accept": item_accept,
                "r_decode_wall": item_tpot * item_accept,
                "r_decode_per_verify_crosscheck": (
                    candidate["decode_interval_per_verify_ms"][offset]
                    / controls["decode_interval_per_verify_ms"][offset]
                ),
                "candidate_emitted_interval_yield": candidate["emitted_interval_yield"][
                    offset
                ],
                "interpolated_fp8_emitted_interval_yield": controls[
                    "emitted_interval_yield"
                ][offset],
                "candidate_forward_ct": candidate_rows[offset]["counter_delta"][
                    "forward_ct"
                ],
                "candidate_internal_accept_tokens": candidate_rows[offset][
                    "counter_delta"
                ]["internal_accept_tokens"],
            }
        )
    tpot_ratios = [item["r_tpot"] for item in per_index]
    decode_ratios = [item["r_decode_wall"] for item in per_index]
    result = {
        "role": role,
        "retained_indices": indices,
        "retained_count": len(indices),
        "fp8_post_weight": weight,
        "r_tpot": r_tpot,
        "r_accept": r_accept,
        "r_decode_wall": r_decode,
        "r_ttft": r_ttft,
        "r_decode_per_verify_crosscheck": r_decode_per_verify,
        "d_accept": r_tpot - r_decode,
        "tpot_band": classify_band(r_tpot, 1.045, 1.055),
        "decode_wall_band": classify_band(r_decode, 1.045, 1.055),
        "ttft_band": classify_band(r_ttft, 1.09, 1.11),
        "per_index": per_index,
        "r_tpot_p95_type7": percentile_type7(tpot_ratios, 0.95),
        "r_tpot_max": max(tpot_ratios),
        "r_decode_wall_p95_type7": percentile_type7(decode_ratios, 0.95),
        "r_decode_wall_max": max(decode_ratios),
        "bootstrap_95": bootstrap_intervals(
            candidate_rows=candidate_rows, controls=controls, draws=draws
        ),
        "metric_means": {
            metric: {
                "candidate": statistics.mean(candidate[metric]),
                "interpolated_fp8": statistics.mean(controls[metric]),
                "candidate_to_fp8_ratio": ratio_of_means(
                    candidate[metric], controls[metric]
                ),
            }
            for metric in metrics
        },
    }

    if role == "h43_n14":
        p95_position = 0.95 * (len(per_index) - 1)
        tail_count = len(per_index) - math.floor(p95_position)
        ordered = sorted(
            per_index,
            key=lambda item: item["r_decode_wall"],
            reverse=True,
        )[:tail_count]
        owners = role_manifest.get("stall_owner_by_index") or {}
        if not isinstance(owners, dict):
            raise AnalysisError("h43_n14: stall_owner_by_index is not an object")
        predeclared_owners = role_manifest.get("predeclared_stall_owners") or []
        if (
            not isinstance(predeclared_owners, list)
            or any(
                not isinstance(owner, str) or not owner for owner in predeclared_owners
            )
            or len(set(predeclared_owners)) != len(predeclared_owners)
        ):
            raise AnalysisError("h43_n14: predeclared stall-owner list is invalid")
        owner_evidence = role_manifest.get("stall_owner_evidence_by_index") or {}
        if not isinstance(owner_evidence, dict):
            raise AnalysisError("h43_n14: stall-owner evidence is not an object")
        tail_classification = []
        for item in ordered:
            index = item["request_index"]
            candidate_share = top3_gap_share(
                roles[role]["rows"][index]["stream"]["events"]
            )
            pre_share = top3_gap_share(
                roles["fp8_pre"]["rows"][index]["stream"]["events"]
            )
            post_share = top3_gap_share(
                roles["fp8_post"]["rows"][index]["stream"]["events"]
            )
            fp8_share = interpolate(pre_share, post_share, weight)
            owner = owners.get(str(index))
            if owner is not None and owner not in predeclared_owners:
                raise AnalysisError(
                    f"h43_n14: tail owner for request {index} was not predeclared"
                )
            aligned_owner = None
            alignment = None
            if owner is not None:
                aligned_owner, alignment = align_discrete_stall_owner(
                    owner=owner,
                    evidence=owner_evidence.get(str(index)),
                    role_data=roles[role],
                    row=roles[role]["rows"][index],
                    required_owned_gap_share=max(0.10, candidate_share - fp8_share),
                )
            classified = classify_tail_event(
                candidate_share,
                fp8_share,
                aligned_owner,
            )
            classified["request_index"] = index
            classified["owner_alignment"] = alignment
            tail_classification.append(classified)
        result["tail_contributor_count"] = tail_count
        result["tail_event_classification"] = tail_classification
    return result


def order_stat_contributors(
    per_index: list[dict[str, Any]], metric: str, *, maximum: bool
) -> list[dict[str, Any]]:
    ordered = sorted(per_index, key=lambda item: (item[metric], item["request_index"]))
    if maximum:
        selected = [(ordered[-1], 1.0)]
    else:
        position = 0.95 * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        selected = (
            [(ordered[lower], 1.0)]
            if lower == upper
            else [
                (ordered[lower], upper - position),
                (ordered[upper], position - lower),
            ]
        )
    return [
        {
            "request_index": item["request_index"],
            "weight": weight,
            "r_tpot": item["r_tpot"],
            "r_accept": item["r_accept"],
            "r_decode_wall": item["r_decode_wall"],
            "acceptance_reduction": item["r_tpot"] - item["r_decode_wall"],
        }
        for item, weight in selected
    ]


def tail_ownership(comparison: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "p95": {
            "cap": 1.10,
            "r_tpot": comparison["r_tpot_p95_type7"],
            "r_decode_wall": comparison["r_decode_wall_p95_type7"],
        },
        "max": {
            "cap": 1.15,
            "r_tpot": comparison["r_tpot_max"],
            "r_decode_wall": comparison["r_decode_wall_max"],
        },
    }
    acceptance_required = False
    direct_decode_required = False
    for name, gate in gates.items():
        tpot_failed = gate["r_tpot"] > gate["cap"]
        decode_failed = gate["r_decode_wall"] > gate["cap"]
        tpot_contributors = order_stat_contributors(
            comparison["per_index"], "r_tpot", maximum=name == "max"
        )
        decode_contributors = order_stat_contributors(
            comparison["per_index"], "r_decode_wall", maximum=name == "max"
        )
        statistic_reduction = gate["r_tpot"] - gate["r_decode_wall"]
        material_indices = sorted(
            {
                item["request_index"]
                for item in tpot_contributors + decode_contributors
                if item["acceptance_reduction"] >= 0.01
            }
        )
        acceptance_owned = (
            tpot_failed and statistic_reduction >= 0.01 and bool(material_indices)
        )
        gate["tpot_failed"] = tpot_failed
        gate["decode_wall_failed"] = decode_failed
        gate["acceptance_statistic_reduction"] = statistic_reduction
        gate["tpot_contributors"] = tpot_contributors
        gate["decode_wall_contributors"] = decode_contributors
        gate["material_acceptance_indices"] = material_indices
        gate["owners"] = [
            owner
            for owner, required in (
                ("A_ACCEPTANCE", acceptance_owned),
                ("C_DIRECT_DECODE", decode_failed),
            )
            if required
        ]
        acceptance_required |= acceptance_owned
        direct_decode_required |= decode_failed
    return {
        "gates": gates,
        "acceptance_required": acceptance_required,
        "direct_decode_required": direct_decode_required,
        "tpot_tail_passed": not any(gate["tpot_failed"] for gate in gates.values()),
        "decode_wall_tail_passed": not any(
            gate["decode_wall_failed"] for gate in gates.values()
        ),
    }


def recovery_allowances(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": {
            "tpot_to_4p5pp": max(0.0, comparison["r_tpot"] - 1.045),
            "decode_wall_to_4p5pp": max(0.0, comparison["r_decode_wall"] - 1.045),
            "ttft_to_9pp": max(0.0, comparison["r_ttft"] - 1.09),
        },
        "tail": {
            "tpot_p95_excess": max(0.0, comparison["r_tpot_p95_type7"] - 1.10),
            "tpot_max_excess": max(0.0, comparison["r_tpot_max"] - 1.15),
            "decode_p95_excess": max(0.0, comparison["r_decode_wall_p95_type7"] - 1.10),
            "decode_max_excess": max(0.0, comparison["r_decode_wall_max"] - 1.15),
        },
    }


def comparison_ratios(comparison: dict[str, Any]) -> dict[str, float]:
    return {
        key: comparison[key]
        for key in (
            "r_tpot",
            "r_accept",
            "r_decode_wall",
            "r_ttft",
            "d_accept",
            "r_tpot_p95_type7",
            "r_tpot_max",
            "r_decode_wall_p95_type7",
            "r_decode_wall_max",
        )
    }


def analyze(
    *,
    root: Path,
    prompt_manifest: dict[str, Any],
    prompt_manifest_sha256: str,
    run_manifest: dict[str, Any],
    run_manifest_sha256: str,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise AnalysisError("bootstrap draw count must be a positive integer")
    if not is_sha256(prompt_manifest_sha256) or not is_sha256(run_manifest_sha256):
        raise AnalysisError("manifest hashes must be lowercase SHA-256 values")
    if prompt_manifest.get("schema_version") != 1:
        raise AnalysisError("unsupported prompt manifest schema")
    if run_manifest.get("schema_version") != 1:
        raise AnalysisError("unsupported run manifest schema")
    if run_manifest.get("role_order") != list(ROLES):
        raise AnalysisError("run manifest role order differs")
    if run_manifest.get("prompt_manifest_sha256") != prompt_manifest_sha256:
        raise AnalysisError("run manifest prompt hash differs")
    role_manifests = run_manifest.get("roles")
    if not isinstance(role_manifests, dict) or set(role_manifests) != set(ROLES):
        raise AnalysisError("run manifest role set differs")
    if any(not isinstance(role_manifests[role], dict) for role in ROLES):
        raise AnalysisError("run manifest role entries must be objects")
    if (
        not isinstance(run_manifest.get("window_id"), str)
        or not run_manifest["window_id"]
    ):
        raise AnalysisError("run manifest window ID is missing")
    measured_prompts = prompt_manifest.get("measured_prompts")
    warmup_prompts = prompt_manifest.get("warmup_prompts")
    if not isinstance(measured_prompts, list) or len(measured_prompts) != 30:
        raise AnalysisError("prompt manifest must contain thirty measured prompts")
    if not isinstance(warmup_prompts, list) or len(warmup_prompts) != 2:
        raise AnalysisError("prompt manifest must contain two warmup prompts")
    endpoint_indices = tuple(prompt_manifest.get("endpoint_subset_indices") or ())
    if endpoint_indices != ENDPOINT_INDICES:
        raise AnalysisError("endpoint subset is not the frozen first ten")
    entries = warmup_prompts + measured_prompts
    if any(not isinstance(entry, dict) for entry in entries):
        raise AnalysisError("prompt manifest entries must be objects")
    prompt_hashes = [entry.get("prompt_sha256") for entry in entries]
    if any(not is_sha256(value) for value in prompt_hashes):
        raise AnalysisError("prompt manifest contains an invalid prompt hash")
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise AnalysisError("prompt hashes are not unique")
    expected_identity = {
        "source_commit": run_manifest.get("source_commit"),
        "image_digest": run_manifest.get("image_digest"),
        "package_sha256": run_manifest.get("package_sha256"),
        "prompt_corpus_id": prompt_manifest.get("corpus_id"),
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "h43_canonical_sha256": run_manifest.get("h43_canonical_sha256"),
    }
    if any(
        not isinstance(value, str) or not value for value in expected_identity.values()
    ):
        raise AnalysisError("run manifest identity is incomplete")
    if not is_sha256(expected_identity["package_sha256"]):
        raise AnalysisError("run manifest package hash is invalid")
    if not is_sha256(expected_identity["h43_canonical_sha256"]):
        raise AnalysisError("run manifest H43 canonical hash is invalid")
    source_commit = expected_identity["source_commit"]
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise AnalysisError("run manifest source commit is invalid")
    image_digest = expected_identity["image_digest"]
    if not image_digest.startswith("sha256:") or not is_sha256(image_digest[7:]):
        raise AnalysisError("run manifest image digest is invalid")
    roles = {
        role: load_role(
            root=root,
            role=role,
            role_manifest=role_manifests[role],
            prompt_hashes=prompt_hashes,
            expected_identity=expected_identity,
        )
        for role in ROLES
    }
    midpoints = [roles[role]["midpoint_realtime_ns"] for role in ROLES]
    if any(left >= right for left, right in zip(midpoints, midpoints[1:])):
        raise AnalysisError("role midpoint order differs")
    if any(
        roles[left]["last_end_realtime_ns"] >= roles[right]["first_start_realtime_ns"]
        for left, right in zip(ROLES, ROLES[1:])
    ):
        raise AnalysisError("role timing windows overlap or are out of order")

    all_indices = list(range(MEASURED_REQUESTS))
    full_length_by_role = {
        role: {index: full_length(roles[role]["rows"][index]) for index in all_indices}
        for role in ROLES
    }
    fp8_short_indices = [
        index
        for index in all_indices
        if not full_length_by_role["fp8_pre"][index]
        or not full_length_by_role["fp8_post"][index]
    ]
    fp8_indices = [index for index in all_indices if index not in fp8_short_indices]
    h43_short_indices = [
        index for index in fp8_indices if not full_length_by_role["h43_n14"][index]
    ]
    e2m1_short_indices = [
        index for index in fp8_indices if not full_length_by_role["e2m1_n14"][index]
    ]
    h43_indices = [index for index in fp8_indices if index not in h43_short_indices]
    e2m1_indices = [index for index in fp8_indices if index not in e2m1_short_indices]

    drift = {}
    for metric, limit in DRIFT_LIMITS.items():
        pre = [row_metric(roles["fp8_pre"]["rows"][i], metric) for i in fp8_indices]
        post = [row_metric(roles["fp8_post"]["rows"][i], metric) for i in fp8_indices]
        delta = ratio_of_means(post, pre) - 1.0
        drift[metric] = {"delta": delta, "limit": limit, "passed": abs(delta) <= limit}
    flank_valid = len(fp8_indices) >= MIN_RETAINED and all(
        item["passed"] for item in drift.values()
    )
    acceptance_flank_report = {}
    for metric in ("internal_accept_length", "response_accept_length"):
        pre = [row_metric(roles["fp8_pre"]["rows"][i], metric) for i in fp8_indices]
        post = [row_metric(roles["fp8_post"]["rows"][i], metric) for i in fp8_indices]
        acceptance_flank_report[metric] = {
            "delta": ratio_of_means(post, pre) - 1.0,
            "fp8_pre_mean": statistics.mean(pre),
            "fp8_post_mean": statistics.mean(post),
            "gate": "report-only",
        }

    comparisons = {}
    if len(e2m1_indices) >= MIN_RETAINED:
        comparisons["e2m1_n14"] = compare_candidate(
            roles=roles,
            role="e2m1_n14",
            indices=e2m1_indices,
            draws=draws,
            role_manifest=role_manifests["e2m1_n14"],
        )
    h43_comparison = None
    endpoint_comparison = None
    if len(h43_indices) >= MIN_RETAINED:
        h43_comparison = compare_candidate(
            roles=roles,
            role="h43_n14",
            indices=h43_indices,
            draws=draws,
            role_manifest=role_manifests["h43_n14"],
        )
        comparisons["h43_n14"] = h43_comparison

    endpoint_complete = all(index in h43_indices for index in ENDPOINT_INDICES)
    if endpoint_complete:
        endpoint_comparison = compare_candidate(
            roles=roles,
            role="h43_n14",
            indices=list(ENDPOINT_INDICES),
            draws=draws,
            role_manifest=role_manifests["h43_n14"],
        )
    reasons = []
    if not flank_valid:
        reasons.append("fp8-flank-invalid")
    if h43_comparison is None:
        reasons.append("fewer-than-27-h43-matched-indices")
    if not endpoint_complete:
        reasons.append("endpoint-subset-incomplete")

    allowance = None
    if not reasons and h43_comparison is not None and endpoint_comparison is not None:
        workstreams = []
        workstream_reasons: dict[str, list[str]] = {
            "A_ACCEPTANCE": [],
            "B_PREFILL": [],
            "C_DIRECT_DECODE": [],
        }
        if h43_comparison["r_tpot"] > 1.055 and h43_comparison["d_accept"] >= 0.01:
            workstreams.append("A_ACCEPTANCE")
            workstream_reasons["A_ACCEPTANCE"].append("diagnostic-mean-tpot")
        if h43_comparison["r_decode_wall"] > 1.055:
            workstreams.append("C_DIRECT_DECODE")
            workstream_reasons["C_DIRECT_DECODE"].append("diagnostic-mean-decode-wall")
        if h43_comparison["r_ttft"] > 1.11:
            workstreams.append("B_PREFILL")
            workstream_reasons["B_PREFILL"].append("diagnostic-mean-ttft")
        tail_ownership_by_scope = {
            "diagnostic": tail_ownership(h43_comparison),
            "endpoint_subset": tail_ownership(endpoint_comparison),
        }
        comparisons_by_scope = {
            "diagnostic": h43_comparison,
            "endpoint_subset": endpoint_comparison,
        }
        for scope, ownership in tail_ownership_by_scope.items():
            if ownership["acceptance_required"]:
                if "A_ACCEPTANCE" not in workstreams:
                    workstreams.append("A_ACCEPTANCE")
                workstream_reasons["A_ACCEPTANCE"].append(f"{scope}-tail")
            if ownership["direct_decode_required"]:
                if "C_DIRECT_DECODE" not in workstreams:
                    workstreams.append("C_DIRECT_DECODE")
                workstream_reasons["C_DIRECT_DECODE"].append(f"{scope}-tail")
                if not all(
                    item["admissible_for_window_b"]
                    for item in comparisons_by_scope[scope]["tail_event_classification"]
                ):
                    reasons.append(f"{scope}-residual-tail-uninstrumentable")
        if not reasons:
            workstreams = [
                name
                for name in ("A_ACCEPTANCE", "B_PREFILL", "C_DIRECT_DECODE")
                if name in workstreams
            ]
            allowance = {
                "schema_version": 1,
                "window_id": run_manifest.get("window_id"),
                "run_manifest_sha256": run_manifest_sha256,
                "window_a_artifact_sha256": run_manifest_sha256,
                "prompt_manifest_sha256": prompt_manifest_sha256,
                "source_commit": expected_identity["source_commit"],
                "image_digest": expected_identity["image_digest"],
                "package_sha256": expected_identity["package_sha256"],
                "retained_h43_indices": h43_indices,
                "endpoint_subset_indices": list(ENDPOINT_INDICES),
                "workstreams": workstreams,
                "workstream_reasons": {
                    name: values
                    for name, values in workstream_reasons.items()
                    if values
                },
                "window_b_required": any(
                    item in workstreams for item in ("B_PREFILL", "C_DIRECT_DECODE")
                ),
                "pass_edges": {
                    "mean_tpot": 1.045,
                    "mean_decode_wall": 1.045,
                    "mean_ttft": 1.09,
                    "tail_p95": 1.10,
                    "tail_max": 1.15,
                },
                "scope_ratios": {
                    scope: comparison_ratios(comparison)
                    for scope, comparison in comparisons_by_scope.items()
                },
                "scope_allowances": {
                    "diagnostic": recovery_allowances(h43_comparison),
                    "endpoint_subset": recovery_allowances(endpoint_comparison),
                },
                "tail_ownership": tail_ownership_by_scope,
                "tail_event_classification": {
                    scope: comparison["tail_event_classification"]
                    for scope, comparison in comparisons_by_scope.items()
                },
                "paired_per_index": {
                    scope: comparison["per_index"]
                    for scope, comparison in comparisons_by_scope.items()
                },
                "fp8_flank_drift": drift,
                "fp8_acceptance_flank_report": acceptance_flank_report,
            }

    result = {
        "schema_version": 1,
        "verdict": "ALLOWANCE_READY" if allowance is not None else "NO_DECISION",
        "reasons": reasons,
        "run_manifest_sha256": run_manifest_sha256,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "role_order": list(ROLES),
        "fp8_flank_drift": drift,
        "fp8_acceptance_flank_report": acceptance_flank_report,
        "fp8_flank_valid": flank_valid,
        "fp8_short_indices": fp8_short_indices,
        "h43_short_quality_signal_indices": h43_short_indices,
        "e2m1_short_indices": e2m1_short_indices,
        "h43_retained_indices": h43_indices,
        "e2m1_retained_indices": e2m1_indices,
        "endpoint_subset_complete": endpoint_complete,
        "endpoint_subset_comparison": endpoint_comparison,
        "comparisons": comparisons,
        "bootstrap": {
            "draws": draws,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "matched-request-index",
        },
    }
    if allowance is not None:
        allowance["analysis_result_sha256"] = sha256_bytes(
            serialized_json(result).encode("utf-8")
        )
    return result, allowance


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized_json(value))
    except FileExistsError as exc:
        raise AnalysisError(f"refusing to overwrite existing output: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--prompt-manifest-sha256", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowance-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.allowance_output.exists():
            raise AnalysisError("refusing to overwrite an analyzer output")
        prompt_manifest = load_hashed_json(
            args.prompt_manifest, args.prompt_manifest_sha256, "prompt manifest"
        )
        run_manifest = load_hashed_json(
            args.run_manifest, args.run_manifest_sha256, "run manifest"
        )
        result, allowance = analyze(
            root=args.root,
            prompt_manifest=prompt_manifest,
            prompt_manifest_sha256=args.prompt_manifest_sha256,
            run_manifest=run_manifest,
            run_manifest_sha256=args.run_manifest_sha256,
        )
        write_new_json(args.output, result)
        if allowance is not None:
            write_new_json(args.allowance_output, allowance)
        return 0 if allowance is not None else 2
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "verdict": "ANALYZER_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not args.output.exists():
            write_new_json(args.output, failure)
        print(f"H44 Window-A analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
