#!/usr/bin/env python3
"""OpenAI-compatible streaming benchmark harness for Kimi speed baselines.

The harness is intentionally dependency-free so it can run from a debug pod,
inside an inference container, or from a workstation that can reach the
endpoint. It measures streaming TTFT, TPOT, end-to-end latency, and aggregate
decode throughput across synthetic and coding-agent-shaped prompts.

CLI defaults follow the public Artificial Analysis 10k language-model API
performance workload as closely as this local synthetic harness can: one
streaming request at a time, approximately 10k input tokens, at least 1500
answer tokens, no deliberate prompt-cache hits, and P50 over 24 measured
samples after warmup.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .paths import DEFAULT_RESULTS_ROOT
except ImportError:  # Allows direct `python bench_openai_endpoint.py` execution.
    DEFAULT_RESULTS_ROOT = Path(
        os.environ.get("OMNIVA_INFERENCE_BENCH_RESULTS", "/mnt/d/omniva/inference/inference-benchmarking")
    )


FILLER_PARAGRAPH = (
    "This repository contains services for request routing, authentication, "
    "policy evaluation, telemetry export, background workers, and deployment "
    "automation. Each module has tests, configuration files, structured logs, "
    "and migration notes. The code favors explicit error handling, observable "
    "state transitions, and small functions with narrow ownership boundaries. "
)


CODING_TASKS = ("repo_qa", "multi_file_edit", "traceback_debug", "long_context_recovery")

AA_DEFAULT_PROMPT_SIZES = "10k"
AA_DEFAULT_OUTPUT_TOKENS = "1500"
AA_DEFAULT_CONCURRENCY = "1"
AA_DEFAULT_REPEATS = 24
AA_DEFAULT_WARMUP_REQUESTS = 2
AA_DEFAULT_CACHE_MODE = "unique-prefix"
AA_DEFAULT_TEMPERATURE = 0.6
AA_DEFAULT_TOP_P = 1.0


def now_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_token_sizes(raw: str, max_prompt_tokens: Optional[int]) -> List[int]:
    sizes: List[int] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item == "max":
            if max_prompt_tokens is None:
                raise SystemExit("--prompt-sizes includes 'max' but --max-prompt-tokens is unset")
            sizes.append(max_prompt_tokens)
            continue
        multiplier = 1
        if item.endswith("k"):
            multiplier = 1000
            item = item[:-1]
        sizes.append(int(float(item) * multiplier))
    if not sizes:
        raise SystemExit("No prompt sizes parsed")
    return sizes


def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise SystemExit("Expected at least one integer")
    return values


def percentile(values: List[float], pct: float) -> Optional[float]:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    rank = max(1, math.ceil((pct / 100.0) * len(clean)))
    return clean[min(rank - 1, len(clean) - 1)]


def mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def normalize_base_url(base_url: str) -> Tuple[str, str, str]:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        v1 = base
        root = base[:-3].rstrip("/")
    else:
        root = base
        v1 = base + "/v1"
    return root, v1, v1 + "/chat/completions"


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]],
    headers: Dict[str, str],
    timeout_s: int,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req_headers = dict(headers)
    if payload is not None:
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if not body:
            return {"status": resp.status, "body": ""}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"body": body}
        parsed["_http_status"] = resp.status
        return parsed


def best_effort_get(url: str, headers: Dict[str, str], timeout_s: int = 30) -> Dict[str, Any]:
    try:
        return http_json("GET", url, None, headers, timeout_s)
    except Exception as exc:
        return {"error": str(exc)}


SENSITIVE_SERVER_INFO_KEYS = {
    "admin_api_key",
    "api_key",
    "ssl_keyfile_password",
}


def redact_server_info(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key in SENSITIVE_SERVER_INFO_KEYS else redact_server_info(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_server_info(item) for item in value]
    return value


def collect_server_info(base_url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    root, v1, _ = normalize_base_url(base_url)
    return redact_server_info({
        "base_url": base_url,
        "captured_at_utc": now_slug(),
        "health": best_effort_get(root + "/health", headers),
        "models": best_effort_get(v1 + "/models", headers),
        "sglang_server_info": best_effort_get(root + "/get_server_info", headers),
    })


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_spec_decode_metrics(
    base_url: str,
    headers: Dict[str, str],
    timeout_s: int = 5,
) -> Dict[str, Any]:
    """Collect best-effort SGLang speculative-decoding metrics.

    SGLang exposes avg_spec_accept_length under /server_info internal_states.
    The value is server-side cumulative since the scheduler counters were last
    reset, so benchmark runs should use a fresh server per candidate when they
    need candidate-level accept length.
    """
    root, _, _ = normalize_base_url(base_url)
    for endpoint in ("/server_info", "/get_server_info"):
        info = best_effort_get(root + endpoint, headers, timeout_s=timeout_s)
        internal_states = info.get("internal_states")
        if not isinstance(internal_states, list):
            continue

        accept_lengths: List[float] = []
        accept_rates: List[float] = []
        for state in internal_states:
            if not isinstance(state, dict):
                continue
            accept_length = as_float(state.get("avg_spec_accept_length"))
            if accept_length is not None:
                accept_lengths.append(accept_length)
            accept_rate = None
            for key in ("avg_spec_accept_rate", "spec_accept_rate", "accept_rate"):
                if key in state:
                    accept_rate = as_float(state.get(key))
                    break
            if accept_rate is not None:
                accept_rates.append(accept_rate)

        return {
            "server_spec_metric_source": endpoint,
            "server_spec_metric_state_count": len(internal_states),
            "server_avg_spec_accept_length": mean(accept_lengths),
            "server_avg_spec_accept_rate": mean(accept_rates),
        }

    return {
        "server_spec_metric_source": "",
        "server_spec_metric_state_count": 0,
        "server_avg_spec_accept_length": None,
        "server_avg_spec_accept_rate": None,
    }


def approx_tokens_to_chars(tokens: int) -> int:
    return max(256, tokens * 4)


def repeat_to_chars(text: str, target_chars: int) -> str:
    repeats = target_chars // len(text) + 1
    return (text * repeats)[:target_chars]


def random_identifier(rng: random.Random, prefix: str) -> str:
    suffix = "".join(rng.choice(string.ascii_lowercase) for _ in range(8))
    return f"{prefix}_{suffix}"


def make_code_block(rng: random.Random, target_chars: int, language: str = "python") -> str:
    lines: List[str] = []
    total = 0
    idx = 0
    while total < target_chars:
        fn = random_identifier(rng, "handle")
        arg1 = random_identifier(rng, "request")
        arg2 = random_identifier(rng, "context")
        chunk = [
            f"def {fn}({arg1}, {arg2}):",
            f"    result = []",
            f"    for item in {arg1}.items:",
            f"        if item.enabled and not item.deleted:",
            f"            value = normalize_value(item.value, {arg2}.policy)",
            f"            result.append({{'id': item.id, 'value': value}})",
            f"    audit_event({arg2}.trace_id, '{fn}', len(result))",
            f"    return result",
            "",
            f"class {fn.title().replace('_', '')}Worker:",
            f"    def run(self, queue, {arg2}):",
            f"        while queue:",
            f"            batch = queue.pop(0)",
            f"            yield {fn}(batch, {arg2})",
            "",
        ]
        lines.extend(chunk)
        total += sum(len(line) + 1 for line in chunk)
        idx += 1
    code = "\n".join(lines)[:target_chars]
    if language == "go":
        return "// Synthetic Go-like service excerpt\n" + code
    return "# Synthetic Python service excerpt\n" + code


def make_synthetic_prompt(target_tokens: int, salt: str, cache_mode: str, output_tokens: int) -> str:
    target_chars = approx_tokens_to_chars(target_tokens)
    body = repeat_to_chars(FILLER_PARAGRAPH, target_chars)
    head_salt = f"[unique-prefix:{salt}]\n" if cache_mode == "unique-prefix" else ""
    tail_salt = f"\n[request-salt:{salt}]\n" if cache_mode == "shared-prefix" else ""
    return (
        head_salt
        + "You are benchmarking a coding model on a long-context inference server.\n"
        + "Read the full document and then answer the final instruction.\n\n"
        + "=== DOCUMENT START ===\n"
        + body
        + "\n=== DOCUMENT END ===\n"
        + tail_salt
        + "\nFinal instruction: produce a dense technical answer about the document. "
        + f"Continue until the server reaches the {output_tokens} token output cap. "
        + "Do not end early unless forced by the server."
    )


def make_coding_prompt(task: str, target_tokens: int, salt: str, cache_mode: str, output_tokens: int) -> str:
    rng = random.Random(f"{task}:{target_tokens}:{salt}")
    target_chars = approx_tokens_to_chars(target_tokens)
    code_chars = max(1024, int(target_chars * 0.78))
    notes_chars = max(512, target_chars - code_chars)
    code = make_code_block(rng, code_chars)
    notes = repeat_to_chars(
        "Operational note: preserve API compatibility, avoid broad refactors, "
        "keep tests focused, and explain risk clearly. ",
        notes_chars,
    )
    head_salt = f"[unique-prefix:{salt}]\n" if cache_mode == "unique-prefix" else ""
    tail_salt = f"\n[request-salt:{salt}]\n" if cache_mode == "shared-prefix" else ""

    if task == "repo_qa":
        instruction = (
            "Answer the developer's architecture question. Identify the likely "
            "ownership boundaries, name the files you would inspect first, and "
            "explain the risk of the proposed change."
        )
    elif task == "multi_file_edit":
        instruction = (
            "Design a minimal multi-file patch. List the files to modify, the "
            "exact behavioral change, and focused tests. Include code snippets."
        )
    elif task == "traceback_debug":
        instruction = (
            "Debug this failure from the traceback and code context. Give the "
            "most likely root cause, a patch direction, and a verification plan."
        )
        notes += "\nTraceback:\nRuntimeError: CUDA graph replay used an uncaptured shape bucket\n"
        notes += "  at scheduler.py:817 in run_batch\n  at model_runner.py:1443 in forward\n"
    elif task == "long_context_recovery":
        instruction = (
            "Recover the buried requirements from the full context, then produce "
            "an implementation plan that preserves those requirements."
        )
        notes = (
            "Buried requirement: do not change public API names; all kubectl "
            "commands in docs must include explicit --context.\n" + notes
        )
    else:
        raise ValueError(f"Unknown coding task: {task}")

    return (
        head_salt
        + "You are a senior coding agent working inside a real repository.\n"
        + "Use the context below as if it came from checked-out files.\n\n"
        + "=== REPOSITORY CONTEXT ===\n"
        + code
        + "\n\n=== NOTES AND LOGS ===\n"
        + notes
        + "\n=== END CONTEXT ===\n"
        + tail_salt
        + "\nTask: "
        + instruction
        + f" Target a response near {output_tokens} tokens and avoid ending early."
    )


def build_messages(
    workload: str,
    task: str,
    prompt_tokens: int,
    output_tokens: int,
    salt: str,
    cache_mode: str,
) -> List[Dict[str, str]]:
    if workload == "synthetic":
        user = make_synthetic_prompt(prompt_tokens, salt, cache_mode, output_tokens)
    elif workload == "coding":
        user = make_coding_prompt(task, prompt_tokens, salt, cache_mode, output_tokens)
    else:
        raise ValueError(f"Unknown workload: {workload}")
    system = (
        "You are Kimi, a precise coding assistant. Be concrete, technical, and "
        "avoid filler. When asked for code, include only relevant snippets."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def iter_sse_lines(resp: Any) -> Iterable[str]:
    for raw in resp:
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            yield line


def stream_chat_completion(
    chat_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: int,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = dict(headers)
    req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(chat_url, data=data, headers=req_headers, method="POST")

    started = time.perf_counter()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    response_chunks = 0
    text_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    http_status: Optional[int] = None

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            http_status = resp.status
            for line in iter_sse_lines(resp):
                if not line.startswith("data:"):
                    continue
                body = line[len("data:") :].strip()
                if body == "[DONE]":
                    continue
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("content")
                    or ""
                )
                if piece:
                    response_chunks += 1
                    text_parts.append(piece)
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        ended = time.perf_counter()
        return {
            "ok": False,
            "http_status": exc.code,
            "error": body[:2000],
            "started_perf": started,
            "ended_perf": ended,
            "e2e_ms": (ended - started) * 1000.0,
        }
    except Exception as exc:
        ended = time.perf_counter()
        return {
            "ok": False,
            "http_status": http_status,
            "error": str(exc),
            "started_perf": started,
            "ended_perf": ended,
            "e2e_ms": (ended - started) * 1000.0,
        }

    ended = time.perf_counter()
    text = "".join(text_parts)
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None
    token_source = "usage" if completion_tokens is not None else "chunk_count"
    if completion_tokens is None:
        completion_tokens = response_chunks
    total_tokens = usage.get("total_tokens") if usage else None

    ttft_ms = None if first_token_at is None else (first_token_at - started) * 1000.0
    e2e_ms = (ended - started) * 1000.0
    if first_token_at is not None and last_token_at is not None and completion_tokens:
        denom = max(int(completion_tokens) - 1, 1)
        tpot_ms = ((last_token_at - first_token_at) * 1000.0) / denom
    else:
        tpot_ms = None
    decode_tok_s = None
    if tpot_ms and tpot_ms > 0:
        decode_tok_s = 1000.0 / tpot_ms

    ok = response_chunks > 0
    empty_stream_error = None if ok else "stream ended without content or reasoning deltas"

    return {
        "ok": ok,
        "http_status": http_status,
        "error": empty_stream_error,
        "started_perf": started,
        "ended_perf": ended,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "decode_tok_s": decode_tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_source": token_source,
        "response_chunks": response_chunks,
        "finish_reason": finish_reason,
        "response_sha256": sha256_text(text),
        "response_chars": len(text),
        "response_preview": text[:240].replace("\n", "\\n"),
    }


def make_payload(
    model: str,
    messages: List[Dict[str, str]],
    output_tokens: int,
    temperature: float,
    top_p: float,
    include_usage: bool,
    ignore_eos: bool,
    thinking: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    if ignore_eos:
        payload["ignore_eos"] = True
    if thinking == "off":
        payload["chat_template_kwargs"] = {"thinking": False}
    elif thinking == "on":
        payload["chat_template_kwargs"] = {"thinking": True}
    return payload


def run_one_request(
    *,
    chat_url: str,
    headers: Dict[str, str],
    model: str,
    workload: str,
    task: str,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    scenario_id: str,
    request_index: int,
    cache_mode: str,
    timeout_s: int,
    temperature: float,
    top_p: float,
    include_usage: bool,
    ignore_eos: bool,
    thinking: str,
    is_warmup: bool,
    salt_prefix: str,
    sglang_rid_prefix: str,
) -> Dict[str, Any]:
    salt = f"{salt_prefix}{scenario_id}-r{request_index}"
    request_id = (
        f"{sglang_rid_prefix}{scenario_id}-r{request_index}"
        if sglang_rid_prefix
        else ""
    )
    messages = build_messages(workload, task, prompt_tokens, output_tokens, salt, cache_mode)
    payload = make_payload(
        model=model,
        messages=messages,
        output_tokens=output_tokens,
        temperature=temperature,
        top_p=top_p,
        include_usage=include_usage,
        ignore_eos=ignore_eos,
        thinking=thinking,
    )
    if request_id:
        payload["rid"] = request_id
    result = stream_chat_completion(chat_url, headers, payload, timeout_s)
    record: Dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": now_slug(),
        "scenario_id": scenario_id,
        "request_index": request_index,
        "sglang_rid": request_id,
        "is_warmup": is_warmup,
        "model": model,
        "workload": workload,
        "task": task,
        "prompt_tokens_target": prompt_tokens,
        "output_tokens_target": output_tokens,
        "concurrency": concurrency,
        "cache_mode": cache_mode,
        "thinking": thinking,
        "ignore_eos": ignore_eos,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_sha256": sha256_text(json.dumps(messages, sort_keys=True)),
    }
    record.update(result)
    return record


def scenario_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    measured = [r for r in records if not r.get("is_warmup")]
    ok = [r for r in measured if r.get("ok")]
    errors = [r for r in measured if not r.get("ok")]
    if measured:
        scenario_wall_s = (
            max(r["ended_perf"] for r in measured if "ended_perf" in r)
            - min(r["started_perf"] for r in measured if "started_perf" in r)
        )
    else:
        scenario_wall_s = 0.0

    completion_total = sum(int(r.get("completion_tokens") or 0) for r in ok)
    aggregate_e2e_tok_s = completion_total / scenario_wall_s if scenario_wall_s > 0 else None

    return {
        "scenario_id": measured[0]["scenario_id"] if measured else "",
        "model": measured[0]["model"] if measured else "",
        "workload": measured[0]["workload"] if measured else "",
        "task": measured[0]["task"] if measured else "",
        "prompt_tokens_target": measured[0]["prompt_tokens_target"] if measured else "",
        "output_tokens_target": measured[0]["output_tokens_target"] if measured else "",
        "concurrency": measured[0]["concurrency"] if measured else "",
        "cache_mode": measured[0]["cache_mode"] if measured else "",
        "thinking": measured[0]["thinking"] if measured else "",
        "ignore_eos": measured[0]["ignore_eos"] if measured else "",
        "requests": len(measured),
        "ok": len(ok),
        "errors": len(errors),
        "scenario_wall_s": scenario_wall_s,
        "prompt_tokens_p50": percentile([float(r["prompt_tokens"]) for r in ok if r.get("prompt_tokens")], 50),
        "completion_tokens_p50": percentile(
            [float(r["completion_tokens"]) for r in ok if r.get("completion_tokens")],
            50,
        ),
        "ttft_ms_p50": percentile([float(r["ttft_ms"]) for r in ok if r.get("ttft_ms")], 50),
        "ttft_ms_p95": percentile([float(r["ttft_ms"]) for r in ok if r.get("ttft_ms")], 95),
        "tpot_ms_p50": percentile([float(r["tpot_ms"]) for r in ok if r.get("tpot_ms")], 50),
        "tpot_ms_p95": percentile([float(r["tpot_ms"]) for r in ok if r.get("tpot_ms")], 95),
        "e2e_ms_p50": percentile([float(r["e2e_ms"]) for r in ok if r.get("e2e_ms")], 50),
        "e2e_ms_p95": percentile([float(r["e2e_ms"]) for r in ok if r.get("e2e_ms")], 95),
        "per_request_decode_tok_s_mean": mean(
            [float(r["decode_tok_s"]) for r in ok if r.get("decode_tok_s")]
        ),
        "aggregate_e2e_tok_s": aggregate_e2e_tok_s,
        "first_error": errors[0].get("error", "")[:500] if errors else "",
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            scrubbed = {
                key: value
                for key, value in row.items()
                if key not in {"started_perf", "ended_perf"}
            }
            fh.write(json.dumps(scrubbed, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_id",
        "model",
        "workload",
        "task",
        "prompt_tokens_target",
        "output_tokens_target",
        "concurrency",
        "cache_mode",
        "thinking",
        "ignore_eos",
        "requests",
        "ok",
        "errors",
        "scenario_wall_s",
        "prompt_tokens_p50",
        "completion_tokens_p50",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "tpot_ms_p50",
        "tpot_ms_p95",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "per_request_decode_tok_s_mean",
        "aggregate_e2e_tok_s",
        "server_avg_spec_accept_length",
        "server_avg_spec_accept_rate",
        "server_avg_spec_accept_length_before",
        "server_avg_spec_accept_rate_before",
        "server_spec_metric_source",
        "server_spec_metric_state_count",
        "first_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = fmt_float(value)
            writer.writerow(out)


def run_scenario(
    args: argparse.Namespace,
    base_url: str,
    chat_url: str,
    headers: Dict[str, str],
    scenario_id: str,
    workload: str,
    task: str,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    raw_path: Path,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    spec_metrics_before = (
        collect_spec_decode_metrics(base_url, headers)
        if not args.no_server_spec_metrics
        else {}
    )

    for warmup_index in range(args.warmup_requests):
        rec = run_one_request(
            chat_url=chat_url,
            headers=headers,
            model=args.model,
            workload=workload,
            task=task,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            concurrency=concurrency,
            scenario_id=scenario_id,
            request_index=-(warmup_index + 1),
            cache_mode=args.cache_mode,
            timeout_s=args.timeout_s,
            temperature=args.temperature,
            top_p=args.top_p,
            include_usage=not args.no_stream_usage,
            ignore_eos=args.ignore_eos,
            thinking=args.thinking,
            is_warmup=True,
            salt_prefix=args.salt_prefix,
            sglang_rid_prefix=args.sglang_rid_prefix,
        )
        records.append(rec)
        write_jsonl(raw_path, [rec])

    total_requests = concurrency * args.repeats
    print(
        f"[scenario] {scenario_id} workload={workload}/{task} "
        f"prompt~{prompt_tokens} out={output_tokens} c={concurrency} n={total_requests}",
        flush=True,
    )
    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        submitted = [
            pool.submit(
                run_one_request,
                chat_url=chat_url,
                headers=headers,
                model=args.model,
                workload=workload,
                task=task,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                concurrency=concurrency,
                scenario_id=scenario_id,
                request_index=i,
                cache_mode=args.cache_mode,
                timeout_s=args.timeout_s,
                temperature=args.temperature,
                top_p=args.top_p,
                include_usage=not args.no_stream_usage,
                ignore_eos=args.ignore_eos,
                thinking=args.thinking,
                is_warmup=False,
                salt_prefix=args.salt_prefix,
                sglang_rid_prefix=args.sglang_rid_prefix,
            )
            for i in range(total_requests)
        ]
        for fut in futures.as_completed(submitted):
            rec = fut.result()
            records.append(rec)
            write_jsonl(raw_path, [rec])
            status = "ok" if rec.get("ok") else "err"
            print(
                f"  {status} r={rec['request_index']} "
                f"ttft={fmt_float(rec.get('ttft_ms'), 1)}ms "
                f"tpot={fmt_float(rec.get('tpot_ms'), 2)}ms "
                f"ct={rec.get('completion_tokens')}",
                flush=True,
            )

    summary = scenario_records(records)
    if not args.no_server_spec_metrics:
        spec_metrics_after = collect_spec_decode_metrics(base_url, headers)
        summary.update(spec_metrics_after)
        summary["server_avg_spec_accept_length_before"] = spec_metrics_before.get(
            "server_avg_spec_accept_length"
        )
        summary["server_avg_spec_accept_rate_before"] = spec_metrics_before.get(
            "server_avg_spec_accept_rate"
        )
    print(
        f"[summary] {scenario_id} ok={summary['ok']}/{summary['requests']} "
        f"ttft_p50={fmt_float(summary.get('ttft_ms_p50'), 1)}ms "
        f"tpot_p50={fmt_float(summary.get('tpot_ms_p50'), 2)}ms "
        f"agg={fmt_float(summary.get('aggregate_e2e_tok_s'), 2)} tok/s "
        f"accept_len={fmt_float(summary.get('server_avg_spec_accept_length'), 3) or 'n/a'}",
        flush=True,
    )
    if args.settle_s:
        time.sleep(args.settle_s)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Endpoint root or /v1 URL")
    parser.add_argument("--model", default="moonshotai/Kimi-K2.6")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("SGLANG_API_KEY"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--workloads", default="synthetic", help="Comma list: synthetic,coding")
    parser.add_argument("--coding-tasks", default="repo_qa,multi_file_edit,traceback_debug,long_context_recovery")
    parser.add_argument(
        "--prompt-sizes",
        default=AA_DEFAULT_PROMPT_SIZES,
        help="Comma list of target prompt sizes. Default is AA-style 10k.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--output-tokens",
        default=AA_DEFAULT_OUTPUT_TOKENS,
        help="Comma list of max completion tokens. Default is AA-style 1500.",
    )
    parser.add_argument(
        "--concurrency",
        default=AA_DEFAULT_CONCURRENCY,
        help="Comma list of concurrent requests. Default is AA-style c1.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=AA_DEFAULT_REPEATS,
        help="Measured samples per scenario. Default 24 approximates AA's 72h P50 window.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=AA_DEFAULT_WARMUP_REQUESTS,
        help=(
            "Excluded warmup requests before measured samples. Default 2 keeps local cold-start "
            "work out of AA-style speed numbers; set 0 for cold/no-warmup tests."
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("unique-prefix", "shared-prefix"),
        default=AA_DEFAULT_CACHE_MODE,
        help=(
            "unique-prefix avoids deliberate prompt-cache hits and is the AA-style default; "
            "shared-prefix intentionally measures prefix-cache reuse."
        ),
    )
    parser.add_argument("--salt-prefix", default="", help="Extra prefix for request salts to isolate cache state across repeated runs")
    parser.add_argument("--sglang-rid-prefix", default="", help="Optional rid prefix for SGLang request/log correlation")
    parser.add_argument("--thinking", choices=("unset", "on", "off"), default="unset")
    parser.add_argument(
        "--temperature",
        type=float,
        default=AA_DEFAULT_TEMPERATURE,
        help="Sampling temperature. Default 0.6 matches AA's reasoning-model setting.",
    )
    parser.add_argument("--top-p", type=float, default=AA_DEFAULT_TOP_P)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--allow-kimi-generation-overrides",
        action="store_true",
        help=(
            "Allow Kimi K2.6 runs to override production benchmark defaults "
            "thinking=unset and ignore_eos=False. Use only for targeted experiments."
        ),
    )
    parser.add_argument("--no-stream-usage", action="store_true")
    parser.add_argument(
        "--no-server-spec-metrics",
        action="store_true",
        help=(
            "Skip best-effort SGLang /server_info speculative-decoding metrics. "
            "By default, summary.csv includes server_avg_spec_accept_length when available."
        ),
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--settle-s", type=float, default=2.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    is_kimi_k26 = "kimi-k2.6" in args.model.lower()
    if is_kimi_k26 and not args.allow_kimi_generation_overrides:
        if args.thinking != "unset":
            raise SystemExit(
                "Kimi K2.6 production benchmarks must use thinking=unset. "
                "Omit --thinking, or pass --allow-kimi-generation-overrides for a targeted experiment."
            )
        if args.ignore_eos:
            raise SystemExit(
                "Kimi K2.6 production benchmarks must use ignore_eos=False. "
                "Omit --ignore-eos, or pass --allow-kimi-generation-overrides for a targeted experiment."
            )

    root, v1, chat_url = normalize_base_url(args.base_url)
    headers: Dict[str, str] = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    output_root = Path(args.output_dir or DEFAULT_RESULTS_ROOT)
    run_name = args.run_name or f"openai-bench-{now_slug()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    summary_path = run_dir / "summary.csv"
    server_info_path = run_dir / "server_info.json"
    config_path = run_dir / "run_config.json"

    config = vars(args).copy()
    config["base_url_root"] = root
    config["base_url_v1"] = v1
    config["chat_url"] = chat_url
    config["api_key"] = "<set>" if args.api_key else "<unset>"
    config["started_at_utc"] = now_slug()
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    server_info_path.write_text(
        json.dumps(collect_server_info(args.base_url, headers), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    workloads = [x.strip() for x in args.workloads.split(",") if x.strip()]
    prompt_sizes = parse_token_sizes(args.prompt_sizes, args.max_prompt_tokens)
    output_tokens_values = parse_int_list(args.output_tokens)
    concurrency_values = parse_int_list(args.concurrency)
    coding_tasks = [x.strip() for x in args.coding_tasks.split(",") if x.strip()]
    for task in coding_tasks:
        if task not in CODING_TASKS:
            raise SystemExit(f"Unknown coding task {task}; valid: {', '.join(CODING_TASKS)}")

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Chat URL: {chat_url}", flush=True)
    print(f"Model: {args.model}", flush=True)

    summaries: List[Dict[str, Any]] = []
    scenario_counter = 0
    for workload in workloads:
        tasks = ["synthetic"] if workload == "synthetic" else coding_tasks
        for task in tasks:
            for prompt_tokens in prompt_sizes:
                for output_tokens in output_tokens_values:
                    for concurrency in concurrency_values:
                        scenario_counter += 1
                        scenario_id = (
                            f"s{scenario_counter:04d}-{workload}-{task}-"
                            f"p{prompt_tokens}-o{output_tokens}-c{concurrency}"
                        )
                        summary = run_scenario(
                            args=args,
                            base_url=args.base_url,
                            chat_url=chat_url,
                            headers=headers,
                            scenario_id=scenario_id,
                            workload=workload,
                            task=task,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            concurrency=concurrency,
                            raw_path=raw_path,
                        )
                        summaries.append(summary)
                        write_summary_csv(summary_path, summaries)

    print(f"Wrote raw JSONL: {raw_path}", flush=True)
    print(f"Wrote summary CSV: {summary_path}", flush=True)
    print(f"Wrote server info: {server_info_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
