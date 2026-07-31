#!/usr/bin/env python3
"""Exact c1 DFlash telemetry harness for the frozen H44 Kimi K2.6 workload.

This module deliberately reuses the sealed H43 prompt and OpenAI payload
generator. It adds process-lifetime speculative-counter snapshots and retains
the complete ordered SSE receive series without assigning timestamps to tokens
inside a multi-token chunk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

H43_CANONICAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "h43-i3-20260730"
    / "endpoint-c2d3d098"
    / "pre-outage"
    / "bench_canonical.py"
)
H43_CANONICAL_SHA256 = (
    "f18e1e9fc0ea9cc2101b1d7c0d80c79f5db3a98b54531cb1be4f16f1b218bf6f"
)

MODEL = "moonshotai/Kimi-K2.6"
PROMPT_TARGET = 16_000
EXPECTED_PROMPT_TOKENS = 10_218
OUTPUT_TOKENS = 512
WARMUP_REQUESTS = 2
QUIET_INTERVAL_S = 0.250
COUNTER_KEYS = (
    "spec_cumulative_num_accept_tokens",
    "spec_cumulative_num_forward_ct",
)
COUNTER_DP_RANK_KEY = "spec_counter_dp_rank"


class TelemetryError(ValueError):
    """The observed telemetry cannot support exact request attribution."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def load_h43_canonical():
    if not H43_CANONICAL_PATH.is_file():
        raise TelemetryError(f"sealed H43 harness is missing: {H43_CANONICAL_PATH}")
    observed_hash = sha256_bytes(H43_CANONICAL_PATH.read_bytes())
    if observed_hash != H43_CANONICAL_SHA256:
        raise TelemetryError(
            "sealed H43 harness hash mismatch: "
            f"expected {H43_CANONICAL_SHA256}, observed {observed_hash}"
        )
    spec = importlib.util.spec_from_file_location(
        "h43_sealed_bench_canonical", H43_CANONICAL_PATH
    )
    if spec is None or spec.loader is None:
        raise TelemetryError("could not load the sealed H43 harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_frozen_kimi_chat(messages: list[dict[str, str]]) -> str:
    """Render the only message shape used by the sealed H43 Kimi workload.

    This is the exact text emitted by the pinned Kimi K2.6 chat template for a
    string system message, a string user message, thinking unset, and
    add_generation_prompt=True. Keeping this deliberately narrow avoids a
    second general-purpose chat-template implementation in the harness.
    """
    if len(messages) != 2 or [message.get("role") for message in messages] != [
        "system",
        "user",
    ]:
        raise TelemetryError("frozen Kimi renderer requires system,user messages")
    system_content = messages[0].get("content")
    user_content = messages[1].get("content")
    if not isinstance(system_content, str) or not isinstance(user_content, str):
        raise TelemetryError("frozen Kimi renderer requires string content")
    return (
        "<|im_system|>system<|im_middle|>"
        + system_content
        + "<|im_end|><|im_user|>user<|im_middle|>"
        + user_content
        + "<|im_end|><|im_assistant|>assistant<|im_middle|><think>"
    )


def load_prompt_manifest(
    path: Path, expected_sha256: str, measured_requests: int
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    observed_sha256 = sha256_bytes(raw)
    if observed_sha256 != expected_sha256:
        raise TelemetryError(
            f"prompt manifest hash mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TelemetryError("prompt manifest is not JSON") from exc
    if manifest.get("schema_version") != 1:
        raise TelemetryError("unsupported prompt manifest schema")
    if not isinstance(manifest.get("corpus_id"), str) or not manifest["corpus_id"]:
        raise TelemetryError("prompt manifest lacks corpus_id")
    qualification = manifest.get("qualification_prompt")
    warmups = manifest.get("warmup_prompts")
    measured = manifest.get("measured_prompts")
    if not isinstance(qualification, dict):
        raise TelemetryError("prompt manifest lacks a qualification prompt")
    if not isinstance(warmups, list) or len(warmups) != WARMUP_REQUESTS:
        raise TelemetryError("prompt manifest must contain exactly two warmups")
    if not isinstance(measured, list) or len(measured) != 30:
        raise TelemetryError("prompt manifest must contain exactly 30 measured prompts")
    qualification_messages = qualification.get("messages")
    qualification_hash = qualification.get("prompt_sha256")
    if (
        not isinstance(qualification_messages, list)
        or not isinstance(qualification.get("salt"), str)
        or qualification_hash != sha256_json(qualification_messages)
        or qualification.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS
        or qualification.get("native_completion_tokens") != OUTPUT_TOKENS
    ):
        raise TelemetryError("qualification prompt is not prevalidated exact/full")
    rendered_hash = qualification.get("rendered_prompt_sha256")
    output_hash = qualification.get("native_output_ids_sha256")
    if (
        not isinstance(rendered_hash, str)
        or len(rendered_hash) != 64
        or not isinstance(output_hash, str)
        or len(output_hash) != 64
    ):
        raise TelemetryError("qualification prompt lacks native request hashes")

    selected = warmups + measured
    seen_hashes = {qualification_hash}
    for expected_index, entry in enumerate(selected):
        if not isinstance(entry, dict):
            raise TelemetryError("prompt manifest entry is not an object")
        messages = entry.get("messages")
        salt = entry.get("salt")
        prompt_hash = entry.get("prompt_sha256")
        if not isinstance(messages, list) or not isinstance(salt, str):
            raise TelemetryError("prompt manifest entry lacks messages or salt")
        if prompt_hash != sha256_json(messages):
            raise TelemetryError("prompt manifest entry hash mismatch")
        if prompt_hash in seen_hashes:
            raise TelemetryError("prompt manifest repeats a prompt hash")
        seen_hashes.add(prompt_hash)
        if entry.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS:
            raise TelemetryError("prompt manifest entry is not exact-10K")
        if expected_index >= WARMUP_REQUESTS:
            measured_index = expected_index - WARMUP_REQUESTS
            if entry.get("selected_index") != measured_index:
                raise TelemetryError("measured prompt indices are not contiguous")
            if entry.get("fp8_completion_tokens") != OUTPUT_TOKENS:
                raise TelemetryError("measured prompt lacks a full FP8 prevalidation")
            output_hash = entry.get("fp8_output_sha256")
            if not isinstance(output_hash, str) or len(output_hash) != 64:
                raise TelemetryError("measured prompt lacks an FP8 output hash")
            if output_hash != sha256_json(entry.get("fp8_output")):
                raise TelemetryError("measured prompt FP8 output hash mismatch")
    return manifest, observed_sha256


def _require_counter(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TelemetryError(f"{label} must be a nonnegative integer, got {value!r}")
    return value


def extract_counter_states(server_info: dict[str, Any]) -> list[dict[str, int]]:
    states = server_info.get("internal_states")
    if not isinstance(states, list) or not states:
        raise TelemetryError("/server_info has no internal_states")

    extracted = []
    seen_dp_ranks = set()
    for index, state in enumerate(states):
        if not isinstance(state, dict):
            raise TelemetryError(f"internal state {index} is not an object")
        dp_rank = _require_counter(
            state.get(COUNTER_DP_RANK_KEY),
            f"state {index} {COUNTER_DP_RANK_KEY}",
        )
        if dp_rank in seen_dp_ranks:
            raise TelemetryError(f"duplicate speculative counter DP rank {dp_rank}")
        seen_dp_ranks.add(dp_rank)
        extracted.append(
            {
                COUNTER_DP_RANK_KEY: dp_rank,
                COUNTER_KEYS[0]: _require_counter(
                    state.get(COUNTER_KEYS[0]), f"state {index} {COUNTER_KEYS[0]}"
                ),
                COUNTER_KEYS[1]: _require_counter(
                    state.get(COUNTER_KEYS[1]), f"state {index} {COUNTER_KEYS[1]}"
                ),
            }
        )
    return sorted(extracted, key=lambda state: state[COUNTER_DP_RANK_KEY])


def fetch_counter_snapshot(
    base_url: str, headers: dict[str, str], timeout_s: int
) -> dict[str, Any]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    mono_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    real_start = time.clock_gettime_ns(time.CLOCK_REALTIME)
    request = urllib.request.Request(
        root + "/server_info", headers=headers, method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
        status = response.status
    mono_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    real_end = time.clock_gettime_ns(time.CLOCK_REALTIME)
    if status != 200:
        raise TelemetryError(f"/server_info returned HTTP {status}")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TelemetryError("/server_info did not return JSON") from exc
    states = extract_counter_states(info)
    draft_tokens = info.get("speculative_num_draft_tokens")
    if draft_tokens is not None:
        draft_tokens = _require_counter(draft_tokens, "speculative_num_draft_tokens")
    return {
        "request_monotonic_ns": mono_start,
        "request_realtime_ns": real_start,
        "response_monotonic_ns": mono_end,
        "response_realtime_ns": real_end,
        "response_sha256": sha256_bytes(raw),
        "speculative_num_draft_tokens": draft_tokens,
        "states": states,
    }


def compute_counter_delta(
    before_states: list[dict[str, int]], after_states: list[dict[str, int]]
) -> dict[str, Any]:
    if not before_states or len(before_states) != len(after_states):
        raise TelemetryError("counter snapshot state count changed or is empty")

    deltas = []
    active = []
    for index, (before, after) in enumerate(zip(before_states, after_states)):
        before_dp_rank = _require_counter(
            before.get(COUNTER_DP_RANK_KEY), "before DP rank"
        )
        after_dp_rank = _require_counter(
            after.get(COUNTER_DP_RANK_KEY), "after DP rank"
        )
        if before_dp_rank != after_dp_rank:
            raise TelemetryError(
                f"counter DP rank changed at sorted index {index}: "
                f"{before_dp_rank} != {after_dp_rank}"
            )
        accept_delta = _require_counter(after.get(COUNTER_KEYS[0]), "after accept") - (
            _require_counter(before.get(COUNTER_KEYS[0]), "before accept")
        )
        forward_delta = _require_counter(
            after.get(COUNTER_KEYS[1]), "after forward"
        ) - (_require_counter(before.get(COUNTER_KEYS[1]), "before forward"))
        if accept_delta < 0 or forward_delta < 0:
            raise TelemetryError(f"counter regression in DP state {index}")
        if forward_delta == 0 and accept_delta != 0:
            raise TelemetryError(
                f"DP state {index} accepted tokens without a verification forward"
            )
        if forward_delta > 0:
            if accept_delta < forward_delta:
                raise TelemetryError(
                    f"DP state {index} accept delta is smaller than forward delta"
                )
            active.append(index)
        deltas.append(
            {
                "dp_state_index": index,
                "dp_rank": before_dp_rank,
                "accept_tokens": accept_delta,
                "forward_ct": forward_delta,
            }
        )

    if len(active) != 1:
        raise TelemetryError(
            f"expected exactly one active DP state, observed {len(active)}: {active}"
        )
    active_index = active[0]
    active_delta = deltas[active_index]
    return {
        "active_dp_state_index": active_index,
        "active_dp_rank": active_delta["dp_rank"],
        "states": deltas,
        "internal_accept_tokens": active_delta["accept_tokens"],
        "forward_ct": active_delta["forward_ct"],
        "internal_accept_length": (
            active_delta["accept_tokens"] / active_delta["forward_ct"]
        ),
    }


def validate_spec_request_identity(
    *,
    counter_delta: dict[str, Any],
    completion_tokens: int,
    verify_ct: int,
    correct_drafts: int,
    proposed_drafts: int,
    reported_accept_length: float,
    configured_draft_tokens: int,
) -> dict[str, Any]:
    """Reconcile SGLang's distinct speculative-decoding count conventions.

    The scheduler's cumulative accept counter is an internal work counter:
    correct drafts plus one target/bonus token for every verification forward.
    Final response metadata instead defines ``spec_accept_length`` as externally
    emitted completion tokens divided by verification forwards. The counts can
    differ at a bounded request boundary because the initial prefill token is
    not a verification result and the final verified run can overshoot the
    requested output limit before output truncation.
    """
    completion_tokens = _require_counter(completion_tokens, "completion_tokens")
    verify_ct = _require_counter(verify_ct, "spec_verify_ct")
    correct_drafts = _require_counter(correct_drafts, "spec_num_correct_drafts")
    proposed_drafts = _require_counter(proposed_drafts, "spec_num_proposed_drafts")
    configured_draft_tokens = _require_counter(
        configured_draft_tokens, "speculative_num_draft_tokens"
    )
    if completion_tokens == 0 or verify_ct == 0:
        raise TelemetryError("spec request must emit tokens and run verification")
    if configured_draft_tokens < 2:
        raise TelemetryError("speculative draft budget must be at least two tokens")

    forward_delta = _require_counter(counter_delta.get("forward_ct"), "forward delta")
    internal_accept_tokens = _require_counter(
        counter_delta.get("internal_accept_tokens"), "internal accept delta"
    )
    if forward_delta != verify_ct:
        raise TelemetryError("cumulative forward delta does not equal spec_verify_ct")
    expected_internal_accept = correct_drafts + verify_ct
    if internal_accept_tokens != expected_internal_accept:
        raise TelemetryError(
            "cumulative internal accept delta fails drafts-plus-bonus identity"
        )

    expected_proposed = verify_ct * (configured_draft_tokens - 1)
    if proposed_drafts != expected_proposed:
        raise TelemetryError(
            f"proposed draft count mismatch: expected {expected_proposed}, "
            f"observed {proposed_drafts}"
        )
    response_accept_length = completion_tokens / verify_ct
    if not math.isclose(
        reported_accept_length,
        response_accept_length,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TelemetryError(
            "reported spec_accept_length fails emitted-completion identity"
        )

    boundary_adjustment = internal_accept_tokens - completion_tokens
    minimum_adjustment = -1
    maximum_adjustment = configured_draft_tokens - 2
    if not minimum_adjustment <= boundary_adjustment <= maximum_adjustment:
        raise TelemetryError(
            "internal/emitted boundary adjustment is outside the exact "
            f"[{minimum_adjustment}, {maximum_adjustment}] bound: "
            f"{boundary_adjustment}"
        )
    emitted_interval_yield = (completion_tokens - 1) / verify_ct
    return {
        "internal_accept_tokens": internal_accept_tokens,
        "internal_accept_length": internal_accept_tokens / verify_ct,
        "response_accept_length": response_accept_length,
        "emitted_interval_yield": emitted_interval_yield,
        "boundary_adjustment_tokens": boundary_adjustment,
        "boundary_adjustment_min": minimum_adjustment,
        "boundary_adjustment_max": maximum_adjustment,
    }


def validate_stream_counter_boundary(
    *,
    counter_delta: dict[str, Any],
    completion_tokens: int,
    configured_draft_tokens: int,
) -> dict[str, Any]:
    """Validate the observable request-boundary relation for an OpenAI stream."""
    completion_tokens = _require_counter(completion_tokens, "completion_tokens")
    configured_draft_tokens = _require_counter(
        configured_draft_tokens, "speculative_num_draft_tokens"
    )
    if completion_tokens == 0:
        raise TelemetryError("stream must emit at least one completion token")
    if configured_draft_tokens < 2:
        raise TelemetryError("speculative draft budget must be at least two tokens")
    verify_ct = _require_counter(counter_delta.get("forward_ct"), "forward delta")
    internal_accept_tokens = _require_counter(
        counter_delta.get("internal_accept_tokens"), "internal accept delta"
    )
    if verify_ct == 0:
        raise TelemetryError("stream has no speculative verification forward")
    boundary_adjustment = internal_accept_tokens - completion_tokens
    minimum_adjustment = -1
    maximum_adjustment = configured_draft_tokens - 2
    if not minimum_adjustment <= boundary_adjustment <= maximum_adjustment:
        raise TelemetryError(
            "stream internal/emitted boundary adjustment is outside the exact "
            f"[{minimum_adjustment}, {maximum_adjustment}] bound: "
            f"{boundary_adjustment}"
        )
    return {
        "internal_accept_tokens": internal_accept_tokens,
        "internal_accept_length": internal_accept_tokens / verify_ct,
        "emitted_completion_tokens": completion_tokens,
        "response_accept_length": completion_tokens / verify_ct,
        "emitted_intervals": completion_tokens - 1,
        "emitted_interval_yield": (completion_tokens - 1) / verify_ct,
        "decode_interval_per_verify_ms": None,
        "boundary_adjustment_tokens": boundary_adjustment,
        "boundary_adjustment_min": minimum_adjustment,
        "boundary_adjustment_max": maximum_adjustment,
    }


def require_counter_continuity(
    previous_after: list[dict[str, int]], current_before: list[dict[str, int]]
) -> None:
    if len(previous_after) != len(current_before):
        raise TelemetryError("DP state count changed between requests")
    for index, (previous, current) in enumerate(zip(previous_after, current_before)):
        previous_dp_rank = _require_counter(
            previous.get(COUNTER_DP_RANK_KEY), "previous DP rank"
        )
        current_dp_rank = _require_counter(
            current.get(COUNTER_DP_RANK_KEY), "current DP rank"
        )
        if previous_dp_rank != current_dp_rank:
            raise TelemetryError(
                f"DP rank changed at sorted index {index}: "
                f"{previous_dp_rank} != {current_dp_rank}"
            )
        for key in COUNTER_KEYS:
            old = _require_counter(previous.get(key), f"previous state {index} {key}")
            new = _require_counter(current.get(key), f"current state {index} {key}")
            if new != old:
                direction = "regression/restart" if new < old else "contamination"
                raise TelemetryError(
                    f"between-request {direction} in DP state {index} {key}: "
                    f"previous={old}, current={new}"
                )


def _event_completion_tokens(chunk: dict[str, Any]) -> tuple[Optional[int], str]:
    candidates = (
        (chunk.get("meta_info"), "meta_info"),
        (chunk.get("usage"), "usage"),
        (chunk, "chunk"),
    )
    for container, source in candidates:
        if isinstance(container, dict) and "completion_tokens" in container:
            return (
                _require_counter(
                    container["completion_tokens"], f"{source}.completion_tokens"
                ),
                source,
            )
    return None, ""


def make_sse_event(
    *,
    raw_line: bytes,
    sequence: int,
    monotonic_ns: int,
    realtime_ns: int,
    previous_known_tokens: int,
) -> tuple[dict[str, Any], str, int]:
    line = raw_line.strip()
    if not line.startswith(b"data:"):
        raise TelemetryError("SSE event does not start with data:")
    body = line[len(b"data:") :].strip()
    if body == b"[DONE]":
        return (
            {
                "sequence": sequence,
                "monotonic_ns": monotonic_ns,
                "realtime_ns": realtime_ns,
                "sse_line_bytes": len(line),
                "body_sha256": sha256_bytes(body),
                "has_text": False,
                "text_bytes": 0,
                "completion_tokens_cumulative": None,
                "completion_tokens_delta": None,
                "completion_tokens_source": "",
                "finish_marker": True,
                "done_marker": True,
            },
            "",
            previous_known_tokens,
        )
    try:
        chunk = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TelemetryError("malformed JSON in SSE data event") from exc

    choices = chunk.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    text_fragment = (
        delta.get("reasoning_content")
        or delta.get("reasoning")
        or delta.get("content")
        or ""
    )
    if not isinstance(text_fragment, str):
        raise TelemetryError("SSE text delta is not a string")
    cumulative, token_source = _event_completion_tokens(chunk)
    token_delta = None
    new_known_tokens = previous_known_tokens
    if cumulative is not None:
        token_delta = cumulative - previous_known_tokens
        if token_delta < 0:
            raise TelemetryError("SSE completion token count regressed")
        new_known_tokens = cumulative
    finish_reason = choice.get("finish_reason")
    event = {
        "sequence": sequence,
        "monotonic_ns": monotonic_ns,
        "realtime_ns": realtime_ns,
        "sse_line_bytes": len(line),
        "body_sha256": sha256_bytes(body),
        "has_text": bool(text_fragment),
        "text_bytes": len(text_fragment.encode("utf-8")),
        "completion_tokens_cumulative": cumulative,
        "completion_tokens_delta": token_delta,
        "completion_tokens_source": token_source,
        "finish_marker": finish_reason is not None,
        "finish_reason": finish_reason,
        "done_marker": False,
    }
    return event, text_fragment, new_known_tokens


def reconstruct_stream_metrics(
    *,
    events: list[dict[str, Any]],
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    final_completion_tokens: int,
) -> dict[str, Any]:
    final_completion_tokens = _require_counter(
        final_completion_tokens, "final completion tokens"
    )
    if final_completion_tokens == 0:
        raise TelemetryError("stream completed with zero completion tokens")
    if ended_monotonic_ns < started_monotonic_ns:
        raise TelemetryError("stream end precedes stream start")
    previous_timestamp = started_monotonic_ns
    text_events = []
    known_cumulative = []
    finish_reasons = []
    for expected_sequence, event in enumerate(events):
        if event.get("sequence") != expected_sequence:
            raise TelemetryError("SSE sequence is not contiguous")
        timestamp = _require_counter(event.get("monotonic_ns"), "event timestamp")
        if timestamp < previous_timestamp:
            raise TelemetryError("SSE monotonic clock regressed")
        if timestamp > ended_monotonic_ns:
            raise TelemetryError("SSE event occurs after stream end")
        previous_timestamp = timestamp
        if event.get("has_text"):
            text_events.append(event)
        cumulative = event.get("completion_tokens_cumulative")
        if cumulative is not None:
            known_cumulative.append(_require_counter(cumulative, "event cumulative"))
        if event.get("finish_reason") is not None:
            finish_reasons.append(event["finish_reason"])
    if not text_events:
        raise TelemetryError("stream has no text-bearing event")
    if known_cumulative and known_cumulative[-1] != final_completion_tokens:
        raise TelemetryError(
            "final usage does not reconcile with the last SSE completion count"
        )
    if len(finish_reasons) != 1:
        raise TelemetryError(
            f"stream must contain one finish reason, observed {finish_reasons!r}"
        )
    first_token_ns = text_events[0]["monotonic_ns"]
    last_token_ns = text_events[-1]["monotonic_ns"]
    decode_interval_ns = last_token_ns - first_token_ns
    denominator = final_completion_tokens - 1
    return {
        "first_token_monotonic_ns": first_token_ns,
        "last_token_monotonic_ns": last_token_ns,
        "ttft_ms": (first_token_ns - started_monotonic_ns) / 1_000_000,
        "decode_interval_ms": decode_interval_ns / 1_000_000,
        "tpot_ms": (
            decode_interval_ns / denominator / 1_000_000 if denominator > 0 else None
        ),
        "tpot_convention": (
            "first-to-last-text-SSE divided by completion_tokens-minus-one; "
            "speculative multi-token chunks are not assigned per-token timestamps"
        ),
        "e2e_ms": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000,
        "decode_tok_s": (
            denominator * 1_000_000_000 / decode_interval_ns
            if denominator > 0 and decode_interval_ns > 0
            else None
        ),
        "completion_tokens": final_completion_tokens,
        "finish_reason": finish_reasons[0],
    }


def _iter_sse_lines(response: Any) -> Iterable[bytes]:
    for raw_line in response:
        line = raw_line.strip()
        if line:
            yield line


def stream_openai_chat(
    *,
    chat_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    encoded = json.dumps(payload).encode()
    request_headers = dict(headers)
    request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        chat_url, data=encoded, headers=request_headers, method="POST"
    )
    start_mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    start_real = time.clock_gettime_ns(time.CLOCK_REALTIME)
    events = []
    text_parts = []
    known_tokens = 0
    usage = None
    http_status = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            http_status = response.status
            for raw_line in _iter_sse_lines(response):
                if not raw_line.startswith(b"data:"):
                    continue
                event, fragment, known_tokens = make_sse_event(
                    raw_line=raw_line,
                    sequence=len(events),
                    monotonic_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC),
                    realtime_ns=time.clock_gettime_ns(time.CLOCK_REALTIME),
                    previous_known_tokens=known_tokens,
                )
                events.append(event)
                if fragment:
                    text_parts.append(fragment)
                body = raw_line[len(b"data:") :].strip()
                if body != b"[DONE]":
                    parsed = json.loads(body)
                    if parsed.get("usage"):
                        usage = parsed["usage"]
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")[:2000]
        http_status = exc.code
    except Exception as exc:  # Preserve partial event evidence for diagnosis.
        error = f"{type(exc).__name__}: {exc}"
    end_mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    end_real = time.clock_gettime_ns(time.CLOCK_REALTIME)
    result = {
        "ok": False,
        "error": error,
        "http_status": http_status,
        "started_monotonic_ns": start_mono,
        "started_realtime_ns": start_real,
        "ended_monotonic_ns": end_mono,
        "ended_realtime_ns": end_real,
        "events": events,
        "response_sha256": sha256_bytes("".join(text_parts).encode()),
        "response_chars": sum(len(part) for part in text_parts),
        "response_bytes": sum(len(part.encode()) for part in text_parts),
    }
    if error is not None:
        return result
    if not isinstance(usage, dict):
        result["error"] = "stream did not return final usage"
        return result
    try:
        metrics = reconstruct_stream_metrics(
            events=events,
            started_monotonic_ns=start_mono,
            ended_monotonic_ns=end_mono,
            final_completion_tokens=usage.get("completion_tokens"),
        )
    except TelemetryError as exc:
        result["error"] = str(exc)
        return result
    result.update(metrics)
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["total_tokens"] = usage.get("total_tokens")
    result["ok"] = http_status == 200
    return result


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: int
) -> dict[str, Any]:
    encoded = json.dumps(payload).encode()
    request_headers = dict(headers)
    request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=encoded, headers=request_headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            if response.status != 200:
                raise TelemetryError(f"{url} returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise TelemetryError(f"{url} returned HTTP {exc.code}: {body}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TelemetryError(f"{url} did not return JSON") from exc


def run_exactness_qualification(
    *,
    base_url: str,
    headers: dict[str, str],
    messages: list[dict[str, str]],
    rid: str,
    timeout_s: int,
) -> dict[str, Any]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    rendered_prompt = render_frozen_kimi_chat(messages)

    time.sleep(QUIET_INTERVAL_S)
    before = fetch_counter_snapshot(root, headers, timeout_s)
    response = _post_json(
        root + "/generate",
        {
            "text": rendered_prompt,
            "sampling_params": {
                "max_new_tokens": OUTPUT_TOKENS,
                "temperature": 0,
                "top_p": 1,
            },
            "stream": False,
            "rid": rid,
        },
        headers,
        timeout_s,
    )
    after = fetch_counter_snapshot(root, headers, timeout_s)
    delta = compute_counter_delta(before["states"], after["states"])
    meta = response.get("meta_info")
    output_ids = response.get("output_ids")
    if not isinstance(meta, dict) or not isinstance(output_ids, list):
        raise TelemetryError("native qualification response lacks final metadata")
    completion_tokens = _require_counter(
        meta.get("completion_tokens"), "meta_info.completion_tokens"
    )
    prompt_tokens = _require_counter(
        meta.get("prompt_tokens"), "meta_info.prompt_tokens"
    )
    verify_ct = _require_counter(meta.get("spec_verify_ct"), "meta_info.spec_verify_ct")
    correct_drafts = _require_counter(
        meta.get("spec_num_correct_drafts"), "meta_info.spec_num_correct_drafts"
    )
    proposed_drafts = _require_counter(
        meta.get("spec_num_proposed_drafts"), "meta_info.spec_num_proposed_drafts"
    )
    if prompt_tokens != EXPECTED_PROMPT_TOKENS:
        raise TelemetryError(
            f"qualification prompt has {prompt_tokens} tokens; "
            f"expected {EXPECTED_PROMPT_TOKENS}"
        )
    if completion_tokens != OUTPUT_TOKENS or len(output_ids) != completion_tokens:
        raise TelemetryError("qualification did not emit the full 512-token contract")
    meta_accept_length = float(meta.get("spec_accept_length"))
    configured = before.get("speculative_num_draft_tokens")
    if configured is None:
        raise TelemetryError(
            "server_info lacks the configured speculative draft budget"
        )
    if after.get("speculative_num_draft_tokens") != configured:
        raise TelemetryError("speculative draft budget changed during qualification")
    reconciled = validate_spec_request_identity(
        counter_delta=delta,
        completion_tokens=completion_tokens,
        verify_ct=verify_ct,
        correct_drafts=correct_drafts,
        proposed_drafts=proposed_drafts,
        reported_accept_length=meta_accept_length,
        configured_draft_tokens=configured,
    )
    return {
        "schema_version": 1,
        "qualified_at_utc": now_utc(),
        "rid": rid,
        "prompt_sha256": sha256_json(messages),
        "rendered_prompt_sha256": sha256_bytes(rendered_prompt.encode()),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "counter_before": before,
        "counter_after": after,
        "counter_delta": delta,
        "spec_verify_ct": verify_ct,
        "spec_num_correct_drafts": correct_drafts,
        "spec_num_proposed_drafts": proposed_drafts,
        "spec_accept_length": meta_accept_length,
        "counter_reconciliation": reconciled,
        "output_sha256": sha256_json(output_ids),
        "passed": True,
    }


def run_timed_request(
    *,
    canonical: Any,
    base_url: str,
    headers: dict[str, str],
    role: str,
    request_index: int,
    is_warmup: bool,
    prompt_entry: dict[str, Any],
    previous_after: list[dict[str, int]],
    timeout_s: int,
    identity: dict[str, str],
) -> dict[str, Any]:
    salt = prompt_entry["salt"]
    messages = prompt_entry["messages"]
    payload = canonical.make_payload(
        model=MODEL,
        messages=messages,
        output_tokens=OUTPUT_TOKENS,
        temperature=0,
        top_p=1,
        include_usage=True,
        ignore_eos=False,
        thinking="unset",
    )
    payload["rid"] = f"h44-{role}-r{request_index}"
    time.sleep(QUIET_INTERVAL_S)
    before = fetch_counter_snapshot(base_url, headers, timeout_s)
    require_counter_continuity(previous_after, before["states"])
    root = base_url.rstrip("/")
    chat_url = (
        root + "/chat/completions"
        if root.endswith("/v1")
        else root + "/v1/chat/completions"
    )
    stream = stream_openai_chat(
        chat_url=chat_url,
        headers=headers,
        payload=payload,
        timeout_s=timeout_s,
    )
    errors = []
    after = None
    counter_delta = None
    counter_reconciliation = None
    if not stream.get("ok"):
        errors.append(stream.get("error") or "stream failed")
    try:
        after = fetch_counter_snapshot(base_url, headers, timeout_s)
    except Exception as exc:
        errors.append(f"after counter snapshot failed: {type(exc).__name__}: {exc}")
    if after is not None:
        try:
            counter_delta = compute_counter_delta(before["states"], after["states"])
        except TelemetryError as exc:
            errors.append(f"counter delta unavailable: {exc}")
    if stream.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS:
        errors.append(
            f"prompt token mismatch: {stream.get('prompt_tokens')} != {EXPECTED_PROMPT_TOKENS}"
        )
    configured = before.get("speculative_num_draft_tokens")
    if configured is None:
        errors.append("server_info lacks the configured speculative draft budget")
    elif after is not None and after.get("speculative_num_draft_tokens") != configured:
        errors.append("speculative draft budget changed during request")
    elif counter_delta is not None and stream.get("completion_tokens") is not None:
        try:
            counter_reconciliation = validate_stream_counter_boundary(
                counter_delta=counter_delta,
                completion_tokens=stream["completion_tokens"],
                configured_draft_tokens=configured,
            )
            decode_interval_ms = stream.get("decode_interval_ms")
            if decode_interval_ms is not None:
                counter_reconciliation["decode_interval_per_verify_ms"] = (
                    decode_interval_ms / counter_delta["forward_ct"]
                )
        except TelemetryError as exc:
            errors.append(str(exc))
    full_length = stream.get("completion_tokens") == OUTPUT_TOKENS
    length_finished = stream.get("finish_reason") == "length"
    telemetry_valid = not errors
    return {
        "schema_version": 2,
        "recorded_at_utc": now_utc(),
        "role": role,
        "request_index": request_index,
        "is_warmup": is_warmup,
        "salt": salt,
        "rid": payload["rid"],
        "prompt_sha256": sha256_json(messages),
        "payload_sha256": sha256_json(payload),
        "prompt_tokens_target": PROMPT_TARGET,
        "expected_prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "output_tokens_target": OUTPUT_TOKENS,
        "temperature": 0,
        "top_p": 1,
        "thinking": "unset",
        "ignore_eos": False,
        "cache_mode": "unique-prefix",
        "counter_before": before,
        "counter_after": after,
        "counter_delta": counter_delta,
        "counter_reconciliation": counter_reconciliation,
        "stream": stream,
        "identity": identity,
        "telemetry_valid": telemetry_valid,
        "performance_eligible": telemetry_valid and full_length and length_finished,
        "quality_signal": (
            None
            if full_length and length_finished
            else {
                "kind": "short-or-nonlength-output",
                "completion_tokens": stream.get("completion_tokens"),
                "finish_reason": stream.get("finish_reason"),
            }
        ),
        "validation_errors": errors,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--measured-requests", required=True, type=int, choices=(10, 30)
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--prompt-manifest-sha256", required=True)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("SGLANG_API_KEY"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    canonical = load_h43_canonical()
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    prompt_manifest, prompt_manifest_sha256 = load_prompt_manifest(
        Path(args.prompt_manifest),
        args.prompt_manifest_sha256,
        args.measured_requests,
    )
    identity = {
        "source_commit": args.source_commit,
        "image_digest": args.image_digest,
        "package_sha256": args.package_sha256,
        "prompt_corpus_id": prompt_manifest["corpus_id"],
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "h43_canonical_sha256": H43_CANONICAL_SHA256,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        qualification_messages = prompt_manifest["qualification_prompt"]["messages"]
        qualification_rendered_hash = sha256_bytes(
            render_frozen_kimi_chat(qualification_messages).encode()
        )
        if qualification_rendered_hash != prompt_manifest["qualification_prompt"].get(
            "rendered_prompt_sha256"
        ):
            raise TelemetryError(
                "qualification native rendered prompt differs from prevalidation"
            )
        qualification = run_exactness_qualification(
            base_url=args.base_url,
            headers=headers,
            messages=qualification_messages,
            rid=f"h44-{args.role}-qualification",
            timeout_s=args.timeout_s,
        )
        qualification["identity"] = identity
        (output_dir / "exactness_qualification.json").write_text(
            json.dumps(qualification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        previous_after = qualification["counter_after"]["states"]
        rows = []
        request_entries = [
            (-(index + 1), entry)
            for index, entry in enumerate(prompt_manifest["warmup_prompts"])
        ] + list(
            enumerate(prompt_manifest["measured_prompts"][: args.measured_requests])
        )
        for request_index, prompt_entry in request_entries:
            row = run_timed_request(
                canonical=canonical,
                base_url=args.base_url,
                headers=headers,
                role=args.role,
                request_index=request_index,
                is_warmup=request_index < 0,
                prompt_entry=prompt_entry,
                previous_after=previous_after,
                timeout_s=args.timeout_s,
                identity=identity,
            )
            rows.append(row)
            with (output_dir / "raw.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if not row["telemetry_valid"]:
                raise TelemetryError(
                    f"request {request_index} failed exact telemetry: "
                    + "; ".join(row["validation_errors"])
                )
            previous_after = row["counter_after"]["states"]
        summary = {
            "schema_version": 1,
            "completed_at_utc": now_utc(),
            "role": args.role,
            "identity": identity,
            "warmup_requests": WARMUP_REQUESTS,
            "measured_requests": args.measured_requests,
            "all_rows_telemetry_valid": all(row["telemetry_valid"] for row in rows),
            "performance_eligible_measured_indices": [
                row["request_index"]
                for row in rows
                if not row["is_warmup"] and row["performance_eligible"]
            ],
            "raw_jsonl_sha256": sha256_bytes((output_dir / "raw.jsonl").read_bytes()),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    except Exception as exc:
        failure = {
            "failed_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "identity": identity,
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"H44 exact telemetry failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
