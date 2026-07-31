#!/usr/bin/env python3
"""Create or replay the frozen exact-10K H44 FP8 prompt corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

TELEMETRY_PATH = Path(__file__).with_name("h44_exact_telemetry.py")
SPEC = importlib.util.spec_from_file_location("h44_exact_telemetry", TELEMETRY_PATH)
TELEMETRY = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("could not load H44 telemetry module")
SPEC.loader.exec_module(TELEMETRY)

INITIAL_CANDIDATES = 40
QUALIFICATION_CANDIDATES = 10
WARMUP_CANDIDATES = 10
SELECTED_PROMPTS = 30
ENDPOINT_PROMPTS = 10


def build_messages(canonical: Any, salt: str) -> list[dict[str, str]]:
    return canonical.build_messages(
        "synthetic",
        "synthetic",
        TELEMETRY.PROMPT_TARGET,
        TELEMETRY.OUTPUT_TOKENS,
        salt,
        "unique-prefix",
    )


def build_openai_payload(
    messages: list[dict[str, str]], max_tokens: int, rid: str
) -> dict[str, Any]:
    return {
        "model": TELEMETRY.MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "rid": rid,
    }


def extract_output(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise TELEMETRY.TelemetryError("OpenAI response does not have one choice")
    choice = choices[0]
    message = choice.get("message")
    usage = response.get("usage")
    if not isinstance(message, dict) or not isinstance(usage, dict):
        raise TELEMETRY.TelemetryError("OpenAI response lacks message or usage")
    prompt_tokens = TELEMETRY._require_counter(
        usage.get("prompt_tokens"), "usage.prompt_tokens"
    )
    completion_tokens = TELEMETRY._require_counter(
        usage.get("completion_tokens"), "usage.completion_tokens"
    )
    output = {
        "reasoning_content": message.get("reasoning_content"),
        "content": message.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "matched_stop": choice.get("matched_stop"),
    }
    if output["reasoning_content"] is not None and not isinstance(
        output["reasoning_content"], str
    ):
        raise TELEMETRY.TelemetryError("reasoning_content is not a string or null")
    if output["content"] is not None and not isinstance(output["content"], str):
        raise TELEMETRY.TelemetryError("content is not a string or null")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output": output,
        "output_sha256": TELEMETRY.sha256_json(output),
    }


def make_candidate_record(
    *,
    canonical: Any,
    chat_url: str,
    headers: dict[str, str],
    corpus_id: str,
    candidate_index: int,
    timeout_s: int,
    warmup_probe: bool,
) -> dict[str, Any]:
    kind = "warmup" if warmup_probe else "candidate"
    salt_index = candidate_index + 100 if warmup_probe else candidate_index
    salt = f"{corpus_id}-s0001-synthetic-synthetic-p16000-o512-c1-" f"r{salt_index:03d}"
    messages = build_messages(canonical, salt)
    max_tokens = 1 if warmup_probe else TELEMETRY.OUTPUT_TOKENS
    payload = build_openai_payload(
        messages, max_tokens, f"h44-prevalidate-{kind}-r{candidate_index:03d}"
    )
    response = TELEMETRY._post_json(chat_url, payload, headers, timeout_s)
    observed = extract_output(response)
    return {
        "schema_version": 1,
        "recorded_at_utc": TELEMETRY.now_utc(),
        "kind": kind,
        "candidate_index": candidate_index,
        "salt_index": salt_index,
        "salt": salt,
        "messages": messages,
        "prompt_sha256": TELEMETRY.sha256_json(messages),
        "payload_sha256": TELEMETRY.sha256_json(payload),
        **observed,
    }


def make_native_qualification_record(
    *,
    canonical: Any,
    native_url: str,
    headers: dict[str, str],
    corpus_id: str,
    candidate_index: int,
    timeout_s: int,
) -> dict[str, Any]:
    salt = (
        f"{corpus_id}-qualification-s0001-synthetic-synthetic-"
        f"p16000-o512-c1-r{candidate_index:03d}"
    )
    messages = build_messages(canonical, salt)
    rendered_prompt = TELEMETRY.render_frozen_kimi_chat(messages)
    payload = {
        "text": rendered_prompt,
        "sampling_params": {
            "max_new_tokens": TELEMETRY.OUTPUT_TOKENS,
            "temperature": 0,
            "top_p": 1,
        },
        "stream": False,
        "rid": f"h44-prevalidate-qualification-r{candidate_index:03d}",
    }
    response = TELEMETRY._post_json(native_url, payload, headers, timeout_s)
    meta = response.get("meta_info")
    output_ids = response.get("output_ids")
    if not isinstance(meta, dict) or not isinstance(output_ids, list):
        raise TELEMETRY.TelemetryError("native qualification lacks metadata or IDs")
    prompt_tokens = TELEMETRY._require_counter(
        meta.get("prompt_tokens"), "meta_info.prompt_tokens"
    )
    completion_tokens = TELEMETRY._require_counter(
        meta.get("completion_tokens"), "meta_info.completion_tokens"
    )
    if len(output_ids) != completion_tokens:
        raise TELEMETRY.TelemetryError(
            "native qualification output IDs do not reconcile"
        )
    return {
        "schema_version": 1,
        "recorded_at_utc": TELEMETRY.now_utc(),
        "kind": "qualification",
        "candidate_index": candidate_index,
        "salt": salt,
        "messages": messages,
        "prompt_sha256": TELEMETRY.sha256_json(messages),
        "rendered_prompt_sha256": TELEMETRY.sha256_bytes(rendered_prompt.encode()),
        "payload_sha256": TELEMETRY.sha256_json(payload),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "native_output_ids_sha256": TELEMETRY.sha256_json(output_ids),
    }


def select_initial_manifest(
    *,
    corpus_id: str,
    qualification_records: list[dict[str, Any]],
    warmup_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    identity: dict[str, str],
) -> dict[str, Any]:
    full_qualifications = [
        record
        for record in qualification_records
        if record["prompt_tokens"] == TELEMETRY.EXPECTED_PROMPT_TOKENS
        and record["completion_tokens"] == TELEMETRY.OUTPUT_TOKENS
    ]
    if not full_qualifications:
        raise TELEMETRY.TelemetryError("no exact/full qualification prompt")
    qualification_record = full_qualifications[0]
    exact_warmups = [
        record
        for record in warmup_records
        if record["prompt_tokens"] == TELEMETRY.EXPECTED_PROMPT_TOKENS
    ]
    if len(exact_warmups) < TELEMETRY.WARMUP_REQUESTS:
        raise TELEMETRY.TelemetryError("fewer than two exact-10K warmup prompts")
    full_candidates = [
        record
        for record in candidate_records
        if record["prompt_tokens"] == TELEMETRY.EXPECTED_PROMPT_TOKENS
        and record["completion_tokens"] == TELEMETRY.OUTPUT_TOKENS
    ]
    if len(full_candidates) < SELECTED_PROMPTS:
        raise TELEMETRY.TelemetryError(
            f"only {len(full_candidates)} exact/full candidates; need {SELECTED_PROMPTS}"
        )

    warmups = []
    for record in exact_warmups[: TELEMETRY.WARMUP_REQUESTS]:
        warmups.append(
            {
                "candidate_index": record["candidate_index"],
                "salt": record["salt"],
                "messages": record["messages"],
                "prompt_sha256": record["prompt_sha256"],
                "prompt_tokens": record["prompt_tokens"],
            }
        )
    measured = []
    for selected_index, record in enumerate(full_candidates[:SELECTED_PROMPTS]):
        measured.append(
            {
                "selected_index": selected_index,
                "candidate_index": record["candidate_index"],
                "salt": record["salt"],
                "messages": record["messages"],
                "prompt_sha256": record["prompt_sha256"],
                "prompt_tokens": record["prompt_tokens"],
                "fp8_completion_tokens": record["completion_tokens"],
                "fp8_output": record["output"],
                "fp8_output_sha256": record["output_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "created_at_utc": TELEMETRY.now_utc(),
        "corpus_id": corpus_id,
        "generator_contract": {
            "h43_canonical_sha256": TELEMETRY.H43_CANONICAL_SHA256,
            "prompt_target": TELEMETRY.PROMPT_TARGET,
            "prompt_tokens": TELEMETRY.EXPECTED_PROMPT_TOKENS,
            "output_tokens": TELEMETRY.OUTPUT_TOKENS,
            "temperature": 0,
            "top_p": 1,
            "thinking": "unset",
            "ignore_eos": False,
            "cache_mode": "unique-prefix",
        },
        "selection_rule": (
            "first 30 of 40 candidate indices returning exactly 10218 prompt "
            "tokens and exactly 512 completion tokens"
        ),
        "endpoint_subset_rule": "first 10 selected_index values before Window A",
        "endpoint_subset_indices": list(range(ENDPOINT_PROMPTS)),
        "identity": identity,
        "qualification_prompt": {
            "candidate_index": qualification_record["candidate_index"],
            "salt": qualification_record["salt"],
            "messages": qualification_record["messages"],
            "prompt_sha256": qualification_record["prompt_sha256"],
            "rendered_prompt_sha256": qualification_record["rendered_prompt_sha256"],
            "prompt_tokens": qualification_record["prompt_tokens"],
            "native_completion_tokens": qualification_record["completion_tokens"],
            "native_output_ids_sha256": qualification_record[
                "native_output_ids_sha256"
            ],
        },
        "warmup_prompts": warmups,
        "measured_prompts": measured,
    }


def verify_endpoint_replay(
    expected_manifest: dict[str, Any], replay_records: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = expected_manifest["measured_prompts"][:ENDPOINT_PROMPTS]
    if len(replay_records) != ENDPOINT_PROMPTS:
        raise TELEMETRY.TelemetryError("endpoint replay did not contain ten prompts")
    comparisons = []
    for expected_entry, replay in zip(expected, replay_records):
        errors = []
        if replay["prompt_sha256"] != expected_entry["prompt_sha256"]:
            errors.append("prompt hash mismatch")
        if replay["prompt_tokens"] != TELEMETRY.EXPECTED_PROMPT_TOKENS:
            errors.append("prompt is not exact-10K")
        if replay["completion_tokens"] != TELEMETRY.OUTPUT_TOKENS:
            errors.append("completion is not full-length")
        if replay["output"] != expected_entry["fp8_output"]:
            errors.append("FP8 output bytes differ")
        if replay["output_sha256"] != expected_entry["fp8_output_sha256"]:
            errors.append("FP8 output hash differs")
        comparisons.append(
            {
                "selected_index": expected_entry["selected_index"],
                "candidate_index": expected_entry["candidate_index"],
                "expected_output_sha256": expected_entry["fp8_output_sha256"],
                "observed_output_sha256": replay["output_sha256"],
                "passed": not errors,
                "errors": errors,
            }
        )
    if not all(comparison["passed"] for comparison in comparisons):
        raise TELEMETRY.TelemetryError("endpoint FP8 byte replay failed")
    return {"passed": True, "comparisons": comparisons}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("initial", "endpoint-replay"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--expected-manifest")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("SGLANG_API_KEY"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = {
        "source_commit": args.source_commit,
        "image_digest": args.image_digest,
        "package_sha256": args.package_sha256,
    }
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    root = args.base_url.rstrip("/")
    chat_url = (
        root + "/chat/completions"
        if root.endswith("/v1")
        else root + "/v1/chat/completions"
    )
    native_root = root[:-3].rstrip("/") if root.endswith("/v1") else root
    native_url = native_root + "/generate"
    canonical = TELEMETRY.load_h43_canonical()
    try:
        if args.mode == "initial":
            if args.expected_manifest or args.expected_manifest_sha256:
                raise TELEMETRY.TelemetryError(
                    "initial mode does not accept an expected manifest"
                )
            qualifications = [
                make_native_qualification_record(
                    canonical=canonical,
                    native_url=native_url,
                    headers=headers,
                    corpus_id=args.corpus_id,
                    candidate_index=index,
                    timeout_s=args.timeout_s,
                )
                for index in range(QUALIFICATION_CANDIDATES)
            ]
            warmups = [
                make_candidate_record(
                    canonical=canonical,
                    chat_url=chat_url,
                    headers=headers,
                    corpus_id=args.corpus_id,
                    candidate_index=index,
                    timeout_s=args.timeout_s,
                    warmup_probe=True,
                )
                for index in range(WARMUP_CANDIDATES)
            ]
            candidates = [
                make_candidate_record(
                    canonical=canonical,
                    chat_url=chat_url,
                    headers=headers,
                    corpus_id=args.corpus_id,
                    candidate_index=index,
                    timeout_s=args.timeout_s,
                    warmup_probe=False,
                )
                for index in range(INITIAL_CANDIDATES)
            ]
            _write_jsonl(output_dir / "qualification_candidates.jsonl", qualifications)
            _write_jsonl(output_dir / "warmup_candidates.jsonl", warmups)
            _write_jsonl(output_dir / "candidates.jsonl", candidates)
            manifest = select_initial_manifest(
                corpus_id=args.corpus_id,
                qualification_records=qualifications,
                warmup_records=warmups,
                candidate_records=candidates,
                identity=identity,
            )
            manifest_path = output_dir / "prompt_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = {
                "mode": args.mode,
                "passed": True,
                "manifest_sha256": TELEMETRY.sha256_bytes(manifest_path.read_bytes()),
                "selected_candidate_indices": [
                    entry["candidate_index"] for entry in manifest["measured_prompts"]
                ],
                "endpoint_subset_indices": manifest["endpoint_subset_indices"],
                "identity": identity,
            }
        else:
            if not args.expected_manifest or not args.expected_manifest_sha256:
                raise TELEMETRY.TelemetryError(
                    "endpoint-replay requires the expected manifest and hash"
                )
            manifest, observed_manifest_sha256 = TELEMETRY.load_prompt_manifest(
                Path(args.expected_manifest),
                args.expected_manifest_sha256,
                ENDPOINT_PROMPTS,
            )
            if manifest["corpus_id"] != args.corpus_id:
                raise TELEMETRY.TelemetryError("endpoint replay corpus_id mismatch")
            replay_records = []
            for entry in manifest["measured_prompts"][:ENDPOINT_PROMPTS]:
                payload = build_openai_payload(
                    entry["messages"],
                    TELEMETRY.OUTPUT_TOKENS,
                    f"h44-endpoint-replay-r{entry['selected_index']:03d}",
                )
                response = TELEMETRY._post_json(
                    chat_url, payload, headers, args.timeout_s
                )
                replay_records.append(
                    {
                        "selected_index": entry["selected_index"],
                        "candidate_index": entry["candidate_index"],
                        "prompt_sha256": TELEMETRY.sha256_json(entry["messages"]),
                        **extract_output(response),
                    }
                )
            _write_jsonl(output_dir / "endpoint_replay.jsonl", replay_records)
            comparison = verify_endpoint_replay(manifest, replay_records)
            result = {
                "mode": args.mode,
                "passed": True,
                "expected_manifest_sha256": observed_manifest_sha256,
                "comparison": comparison,
                "identity": identity,
            }
        (output_dir / "PREVALIDATION_RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    except Exception as exc:
        failure = {
            "mode": args.mode,
            "failed_at_utc": TELEMETRY.now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "identity": identity,
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"H44 corpus prevalidation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
