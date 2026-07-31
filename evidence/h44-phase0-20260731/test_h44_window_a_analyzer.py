#!/usr/bin/env python3
"""Adversarial tests for the frozen H44 Window-A analyzer."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("h44_window_a_analyzer.py")
SPEC = importlib.util.spec_from_file_location("h44_window_a_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SHA_A = "a" * 64
SHA_B = "b" * 64
SOURCE_COMMIT = "c" * 40
IMAGE_DIGEST = "sha256:" + "d" * 64
IDENTITY_TEMPLATE = {
    "source_commit": SOURCE_COMMIT,
    "image_digest": IMAGE_DIGEST,
    "package_sha256": SHA_A,
    "prompt_corpus_id": "h44-fixture",
    "prompt_manifest_sha256": "",
    "h43_canonical_sha256": SHA_B,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def rfc3339_nano(value_ns: int) -> str:
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    base = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    return f"{base}.{nanoseconds:09d}Z"


def timed_row(
    *,
    role: str,
    request_index: int,
    ordinal: int,
    role_ordinal: int,
    prompt_sha256: str,
    identity: dict[str, str],
    tpot_ms: float = 4.0,
    ttft_ms: float = 200.0,
    completion_tokens: int = 512,
    forwards: int = 300,
    concentrated_tail: bool = False,
) -> dict[str, object]:
    decode_ns = round(tpot_ms * max(0, completion_tokens - 1) * 1_000_000)
    start_mono = 1_000_000_000_000 + ordinal * 10_000_000_000
    first_mono = start_mono + round(ttft_ms * 1_000_000)
    role_base = 2_000_000_000_000 + role_ordinal * 200_000_000_000
    started_real = role_base + ordinal * 5_000_000_000
    event_count = 41 if completion_tokens > 1 else 1
    if event_count == 1:
        fractions = [0.0]
    elif concentrated_tail:
        fractions = [
            0.65 * index / (event_count - 2) for index in range(event_count - 1)
        ]
        fractions += [1.0]
    else:
        fractions = [index / (event_count - 1) for index in range(event_count)]
    event_times = [first_mono + round(decode_ns * fraction) for fraction in fractions]
    events = [
        {
            "sequence": index,
            "monotonic_ns": timestamp,
            "realtime_ns": started_real + timestamp - start_mono,
            "has_text": True,
            "finish_reason": "length" if index == event_count - 1 else None,
        }
        for index, timestamp in enumerate(event_times)
    ]
    ended_real = started_real + round(
        (ttft_ms + tpot_ms * (completion_tokens - 1) + 10) * 1_000_000
    )
    internal = completion_tokens
    boundary = internal - completion_tokens
    decode_ms = decode_ns / 1_000_000
    eligible = completion_tokens == 512
    return {
        "schema_version": 2,
        "role": role,
        "request_index": request_index,
        "is_warmup": request_index < 0,
        "prompt_sha256": prompt_sha256,
        "identity": copy.deepcopy(identity),
        "telemetry_valid": True,
        "performance_eligible": eligible,
        "quality_signal": None if eligible else {"kind": "short-or-nonlength-output"},
        "validation_errors": [],
        "counter_before": {"speculative_num_draft_tokens": 5},
        "counter_after": {"speculative_num_draft_tokens": 5},
        "counter_delta": {
            "internal_accept_tokens": internal,
            "forward_ct": forwards,
        },
        "counter_reconciliation": {
            "internal_accept_tokens": internal,
            "internal_accept_length": internal / forwards,
            "emitted_completion_tokens": completion_tokens,
            "response_accept_length": completion_tokens / forwards,
            "emitted_intervals": completion_tokens - 1,
            "emitted_interval_yield": (completion_tokens - 1) / forwards,
            "decode_interval_per_verify_ms": decode_ms / forwards,
            "boundary_adjustment_tokens": boundary,
            "boundary_adjustment_min": -1,
            "boundary_adjustment_max": 3,
        },
        "stream": {
            "ok": True,
            "prompt_tokens": 10_218,
            "completion_tokens": completion_tokens,
            "finish_reason": "length",
            "started_monotonic_ns": start_mono,
            "ended_monotonic_ns": first_mono + decode_ns + 10_000_000,
            "started_realtime_ns": started_real,
            "ended_realtime_ns": ended_real,
            "decode_interval_ms": decode_ms,
            "tpot_ms": (
                decode_ms / (completion_tokens - 1) if completion_tokens > 1 else None
            ),
            "ttft_ms": ttft_ms,
            "events": events,
        },
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prompt_manifest = {
            "schema_version": 1,
            "corpus_id": "h44-fixture",
            "endpoint_subset_indices": list(range(10)),
            "warmup_prompts": [
                {
                    "prompt_sha256": hashlib.sha256(
                        f"warmup-{index}".encode()
                    ).hexdigest()
                }
                for index in range(2)
            ],
            "measured_prompts": [
                {
                    "prompt_sha256": hashlib.sha256(
                        f"measured-{index}".encode()
                    ).hexdigest()
                }
                for index in range(30)
            ],
        }
        self.prompt_path = root / "prompt-manifest.json"
        dump_json(self.prompt_path, self.prompt_manifest)
        self.prompt_sha256 = sha256_file(self.prompt_path)
        self.identity = copy.deepcopy(IDENTITY_TEMPLATE)
        self.identity["prompt_manifest_sha256"] = self.prompt_sha256
        self.predeclared_stall_owners: list[str] = []
        self.stall_owner_by_index: dict[str, str] = {}
        self.stall_owner_evidence_by_index: dict[str, dict[str, object]] = {}
        self.clock_offset_bounds = {"ct13": (0, 0), "ct14": (0, 0)}
        self.rows: dict[str, list[dict[str, object]]] = {}
        hashes = [
            entry["prompt_sha256"]
            for entry in self.prompt_manifest["warmup_prompts"]
            + self.prompt_manifest["measured_prompts"]
        ]
        request_indices = [-1, -2] + list(range(30))
        for role_ordinal, role in enumerate(MODULE.ROLES):
            self.rows[role] = [
                timed_row(
                    role=role,
                    request_index=request_index,
                    ordinal=ordinal,
                    role_ordinal=role_ordinal,
                    prompt_sha256=hashes[ordinal],
                    identity=self.identity,
                )
                for ordinal, request_index in enumerate(request_indices)
            ]

    def rewrite_row(
        self,
        role: str,
        request_index: int,
        **changes: object,
    ) -> None:
        ordinal = request_index + 2 if request_index >= 0 else -request_index - 1
        prompt_hash = self.rows[role][ordinal]["prompt_sha256"]
        self.rows[role][ordinal] = timed_row(
            role=role,
            request_index=request_index,
            ordinal=ordinal,
            role_ordinal=MODULE.ROLES.index(role),
            prompt_sha256=prompt_hash,
            identity=self.identity,
            **changes,
        )

    def seal(self) -> tuple[dict[str, object], str]:
        role_manifests = {}
        for role in MODULE.ROLES:
            role_dir = self.root / role
            role_dir.mkdir(parents=True, exist_ok=True)
            raw_path = role_dir / "raw.jsonl"
            raw_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in self.rows[role]
                ),
                encoding="utf-8",
            )
            qualification_path = role_dir / "exactness_qualification.json"
            dump_json(
                qualification_path,
                {"schema_version": 1, "passed": True, "identity": self.identity},
            )
            first_start = self.rows[role][2]["stream"]["started_realtime_ns"]
            last_end = self.rows[role][-1]["stream"]["ended_realtime_ns"]
            logs_dir = role_dir / "logs"
            logs_dir.mkdir()
            log_capture = {}
            for node in ("ct13", "ct14"):
                lines = [
                    f"{rfc3339_nano(first_start - 1_000_000_000)} {role} {node} start",
                    f"{rfc3339_nano(last_end + 1_000_000_000)} {role} {node} end",
                ]
                if role == "h43_n14":
                    lines.extend(
                        evidence["log_line_utf8"]
                        for evidence in self.stall_owner_evidence_by_index.values()
                        if evidence.get("node") == node
                        and isinstance(evidence.get("log_line_utf8"), str)
                    )
                log_path = logs_dir / f"{node}.log"
                log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                log_sha256 = sha256_file(log_path)
                if role == "h43_n14":
                    for evidence in self.stall_owner_evidence_by_index.values():
                        if evidence.get("node") == node:
                            evidence["source_log_sha256"] = log_sha256
                log_capture[node] = {
                    "requested_since_realtime_ns": (
                        first_start - 1_000_000_000 + self.clock_offset_bounds[node][0]
                    ),
                    "requested_until_realtime_ns": (
                        last_end + 1_000_000_000 + self.clock_offset_bounds[node][1]
                    ),
                    "retrieval_error": None,
                    "truncated": False,
                    "log_driver_unchanged": True,
                    "container_identity_unchanged": True,
                    "relative_path": f"logs/{node}.log",
                    "log_sha256": log_sha256,
                }
            clock_offsets = {
                node: {
                    "minimum_node_minus_client_realtime_ns": self.clock_offset_bounds[
                        node
                    ][0],
                    "maximum_node_minus_client_realtime_ns": self.clock_offset_bounds[
                        node
                    ][1],
                    "sample_count": 2,
                    "before_sample_client_realtime_ns": first_start - 2_000_000_000,
                    "after_sample_client_realtime_ns": last_end + 2_000_000_000,
                }
                for node in ("ct13", "ct14")
            }
            role_manifests[role] = {
                "qualification_passed": True,
                "health_passed": True,
                "provenance_passed": True,
                "log_coverage_passed": True,
                "clock_offset_passed": True,
                "container_identity_passed": True,
                "config_passed": True,
                "role_deadline_passed": True,
                "h43_attestation_passed": role == "h43_n14",
                "memory_accounting_passed": role == "h43_n14",
                "raw_jsonl_sha256": sha256_file(raw_path),
                "qualification_sha256": sha256_file(qualification_path),
                "log_capture_by_node": log_capture,
                "clock_offset_by_node": clock_offsets,
                "stall_owner_by_index": copy.deepcopy(self.stall_owner_by_index),
                "stall_owner_evidence_by_index": copy.deepcopy(
                    self.stall_owner_evidence_by_index
                ),
                "predeclared_stall_owners": list(self.predeclared_stall_owners),
            }
        manifest = {
            "schema_version": 1,
            "window_id": "h44-fixture-window-a",
            "role_order": list(MODULE.ROLES),
            "prompt_manifest_sha256": self.prompt_sha256,
            "source_commit": SOURCE_COMMIT,
            "image_digest": IMAGE_DIGEST,
            "package_sha256": SHA_A,
            "h43_canonical_sha256": SHA_B,
            "roles": role_manifests,
        }
        run_path = self.root / "run-manifest.json"
        dump_json(run_path, manifest)
        return manifest, sha256_file(run_path)

    def analyze(
        self, draws: int = 100
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        manifest, manifest_hash = self.seal()
        return MODULE.analyze(
            root=self.root,
            prompt_manifest=self.prompt_manifest,
            prompt_manifest_sha256=self.prompt_sha256,
            run_manifest=manifest,
            run_manifest_sha256=manifest_hash,
            draws=draws,
        )


class WindowAAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_equal_window_emits_empty_recovery_allowance(self) -> None:
        result, allowance = self.fixture.analyze()
        self.assertEqual(result["verdict"], "ALLOWANCE_READY")
        self.assertIsNotNone(allowance)
        self.assertEqual(allowance["workstreams"], [])
        self.assertEqual(result["comparisons"]["h43_n14"]["r_tpot"], 1.0)
        self.assertEqual(result["comparisons"]["h43_n14"]["tail_contributor_count"], 3)
        self.assertEqual(
            result["endpoint_subset_comparison"]["tail_contributor_count"], 2
        )
        self.assertIn("endpoint_subset", allowance["scope_allowances"])
        self.assertEqual(
            allowance["analysis_result_sha256"],
            hashlib.sha256(MODULE.serialized_json(result).encode()).hexdigest(),
        )
        self.assertEqual(len(allowance["paired_per_index"]["endpoint_subset"]), 10)
        self.assertIn("r_accept", allowance["scope_ratios"]["endpoint_subset"])

    def test_counter_tamper_is_rejected_even_when_row_claims_valid(self) -> None:
        row = self.fixture.rows["h43_n14"][5]
        row["counter_reconciliation"]["emitted_interval_yield"] = 99
        with self.assertRaisesRegex(MODULE.AnalysisError, "emitted interval yield"):
            self.fixture.analyze()

    def test_ttft_tamper_is_rejected_against_raw_sse_timing(self) -> None:
        self.fixture.rows["h43_n14"][5]["stream"]["ttft_ms"] = 999
        with self.assertRaisesRegex(MODULE.AnalysisError, "TTFT"):
            self.fixture.analyze()

    def test_log_range_must_enclose_measured_role(self) -> None:
        manifest, _ = self.fixture.seal()
        manifest["roles"]["h43_n14"]["log_capture_by_node"]["ct13"][
            "requested_since_realtime_ns"
        ] += 2_000_000_000
        run_path = self.fixture.root / "run-manifest-bad-log-range.json"
        dump_json(run_path, manifest)
        with self.assertRaisesRegex(MODULE.AnalysisError, "log range"):
            MODULE.analyze(
                root=self.fixture.root,
                prompt_manifest=self.fixture.prompt_manifest,
                prompt_manifest_sha256=self.fixture.prompt_sha256,
                run_manifest=manifest,
                run_manifest_sha256=sha256_file(run_path),
                draws=10,
            )

    def test_ct14_log_enclosure_uses_measured_clock_offset_bounds(self) -> None:
        manifest, _ = self.fixture.seal()
        role = manifest["roles"]["h43_n14"]
        role["clock_offset_by_node"]["ct14"].update(
            {
                "minimum_node_minus_client_realtime_ns": 2_000_000_000,
                "maximum_node_minus_client_realtime_ns": 3_000_000_000,
            }
        )
        capture = role["log_capture_by_node"]["ct14"]
        capture["requested_since_realtime_ns"] += 2_000_000_000
        capture["requested_until_realtime_ns"] += 3_000_000_000
        run_path = self.fixture.root / "run-manifest-ct14-offset.json"
        dump_json(run_path, manifest)
        result, allowance = MODULE.analyze(
            root=self.fixture.root,
            prompt_manifest=self.fixture.prompt_manifest,
            prompt_manifest_sha256=self.fixture.prompt_sha256,
            run_manifest=manifest,
            run_manifest_sha256=sha256_file(run_path),
            draws=10,
        )
        self.assertEqual(result["verdict"], "ALLOWANCE_READY")
        self.assertIsNotNone(allowance)

    def test_fp8_tpot_drift_blocks_allowance(self) -> None:
        for index in range(30):
            self.fixture.rewrite_row("fp8_post", index, tpot_ms=4.08)
        result, allowance = self.fixture.analyze()
        self.assertIsNone(allowance)
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertIn("fp8-flank-invalid", result["reasons"])

    def test_fp8_short_is_symmetric_and_blocks_endpoint_subset(self) -> None:
        self.fixture.rewrite_row("fp8_pre", 0, completion_tokens=1, forwards=1)
        result, allowance = self.fixture.analyze()
        self.assertIsNone(allowance)
        self.assertEqual(result["fp8_short_indices"], [0])
        self.assertNotIn(0, result["h43_retained_indices"])
        self.assertIn("endpoint-subset-incomplete", result["reasons"])

    def test_h43_nonendpoint_short_is_quality_signal_not_phase1_failure(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, completion_tokens=1, forwards=1)
        result, allowance = self.fixture.analyze()
        self.assertIsNotNone(allowance)
        self.assertEqual(result["h43_short_quality_signal_indices"], [20])
        self.assertNotIn(20, result["h43_retained_indices"])

    def test_mean_acceptance_gap_selects_a_without_window_b(self) -> None:
        for index in range(30):
            self.fixture.rewrite_row("h43_n14", index, tpot_ms=4.32, forwards=316)
        result, allowance = self.fixture.analyze()
        self.assertEqual(result["verdict"], "ALLOWANCE_READY")
        self.assertEqual(allowance["workstreams"], ["A_ACCEPTANCE"])
        self.assertFalse(allowance["window_b_required"])

    def test_mean_direct_decode_gap_selects_c_and_window_b(self) -> None:
        for index in range(30):
            self.fixture.rewrite_row("h43_n14", index, tpot_ms=4.32)
        _, allowance = self.fixture.analyze()
        self.assertEqual(allowance["workstreams"], ["C_DIRECT_DECODE"])
        self.assertTrue(allowance["window_b_required"])

    def test_mean_ttft_gap_selects_b_and_window_b(self) -> None:
        for index in range(30):
            self.fixture.rewrite_row("h43_n14", index, ttft_ms=240.0)
        _, allowance = self.fixture.analyze()
        self.assertEqual(allowance["workstreams"], ["B_PREFILL"])
        self.assertTrue(allowance["window_b_required"])

    def test_direct_tail_gap_requires_instrumentable_classification(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0, concentrated_tail=True)
        result, allowance = self.fixture.analyze()
        self.assertIsNone(allowance)
        self.assertIn("diagnostic-residual-tail-uninstrumentable", result["reasons"])

    def test_uniform_direct_tail_gap_selects_c(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0)
        _, allowance = self.fixture.analyze()
        self.assertIsNotNone(allowance)
        self.assertEqual(allowance["workstreams"], ["C_DIRECT_DECODE"])
        self.assertIn(
            "diagnostic-tail", allowance["workstream_reasons"]["C_DIRECT_DECODE"]
        )

    def test_partially_acceptance_restored_tail_selects_a_and_c(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=5.6, forwards=360)
        _, allowance = self.fixture.analyze()
        self.assertEqual(allowance["workstreams"], ["A_ACCEPTANCE", "C_DIRECT_DECODE"])
        max_gate = allowance["tail_ownership"]["diagnostic"]["gates"]["max"]
        self.assertEqual(max_gate["owners"], ["A_ACCEPTANCE", "C_DIRECT_DECODE"])
        self.assertIn(20, max_gate["material_acceptance_indices"])

    def test_immaterial_tail_acceptance_delta_does_not_select_a(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=4.8, forwards=301)
        _, allowance = self.fixture.analyze()
        self.assertEqual(allowance["workstreams"], ["C_DIRECT_DECODE"])
        max_gate = allowance["tail_ownership"]["diagnostic"]["gates"]["max"]
        self.assertEqual(max_gate["material_acceptance_indices"], [])

    def test_discrete_owned_tail_unblocks_allowance(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0, concentrated_tail=True)
        self.fixture.predeclared_stall_owners = ["rank0-scheduler"]
        self.fixture.stall_owner_by_index = {"20": "rank0-scheduler"}
        gap = MODULE.top3_gap_details(
            self.fixture.rows["h43_n14"][22]["stream"]["events"]
        )[0]
        event_realtime_ns = (gap["start_realtime_ns"] + gap["end_realtime_ns"]) // 2
        log_line = (
            f"{rfc3339_nano(event_realtime_ns)} " "rank0-scheduler scheduler-stall"
        )
        self.fixture.stall_owner_evidence_by_index = {
            "20": {
                "owner": "rank0-scheduler",
                "node": "ct13",
                "event_kind": "scheduler-stall",
                "event_realtime_ns": event_realtime_ns,
                "gap_rank": 1,
                "log_line_utf8": log_line,
                "log_line_sha256": hashlib.sha256(log_line.encode()).hexdigest(),
            }
        }
        _, allowance = self.fixture.analyze()
        self.assertIsNotNone(allowance)
        classified = allowance["tail_event_classification"]["diagnostic"]
        owned = [item for item in classified if item["request_index"] == 20][0]
        self.assertEqual(owned["classification"], "discrete-owned")
        self.assertTrue(owned["owner_alignment"]["passed"])

    def test_misaligned_discrete_owner_remains_terminal(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0, concentrated_tail=True)
        self.fixture.predeclared_stall_owners = ["rank0-scheduler"]
        self.fixture.stall_owner_by_index = {"20": "rank0-scheduler"}
        log_line = f"{rfc3339_nano(1)} rank0-scheduler scheduler-stall"
        self.fixture.stall_owner_evidence_by_index = {
            "20": {
                "owner": "rank0-scheduler",
                "node": "ct13",
                "event_kind": "scheduler-stall",
                "event_realtime_ns": 1,
                "gap_rank": 1,
                "log_line_utf8": log_line,
                "log_line_sha256": hashlib.sha256(log_line.encode()).hexdigest(),
            }
        }
        result, allowance = self.fixture.analyze()
        self.assertIsNone(allowance)
        self.assertIn("diagnostic-residual-tail-uninstrumentable", result["reasons"])

    def test_non_dominant_gap_cannot_own_discrete_stall_excess(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0, concentrated_tail=True)
        gaps = MODULE.top3_gap_details(
            self.fixture.rows["h43_n14"][22]["stream"]["events"]
        )
        gap = gaps[1]
        event_realtime_ns = (gap["start_realtime_ns"] + gap["end_realtime_ns"]) // 2
        log_line = (
            f"{rfc3339_nano(event_realtime_ns)} " "rank0-scheduler scheduler-stall"
        )
        self.fixture.predeclared_stall_owners = ["rank0-scheduler"]
        self.fixture.stall_owner_by_index = {"20": "rank0-scheduler"}
        self.fixture.stall_owner_evidence_by_index = {
            "20": {
                "owner": "rank0-scheduler",
                "node": "ct13",
                "event_kind": "scheduler-stall",
                "event_realtime_ns": event_realtime_ns,
                "gap_rank": 2,
                "log_line_utf8": log_line,
                "log_line_sha256": hashlib.sha256(log_line.encode()).hexdigest(),
            }
        }
        result, allowance = self.fixture.analyze()
        self.assertIsNone(allowance)
        self.assertIn("diagnostic-residual-tail-uninstrumentable", result["reasons"])

    def test_ct14_alignment_uses_offset_intersection_not_union(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0, concentrated_tail=True)
        self.fixture.clock_offset_bounds["ct14"] = (0, 50_000_000)
        gap = MODULE.top3_gap_details(
            self.fixture.rows["h43_n14"][22]["stream"]["events"]
        )[0]
        event_realtime_ns = gap["start_realtime_ns"] + 10_000_000
        log_line = (
            f"{rfc3339_nano(event_realtime_ns)} " "rank0-scheduler scheduler-stall"
        )
        self.fixture.predeclared_stall_owners = ["rank0-scheduler"]
        self.fixture.stall_owner_by_index = {"20": "rank0-scheduler"}
        self.fixture.stall_owner_evidence_by_index = {
            "20": {
                "owner": "rank0-scheduler",
                "node": "ct14",
                "event_kind": "scheduler-stall",
                "event_realtime_ns": event_realtime_ns,
                "gap_rank": 1,
                "log_line_utf8": log_line,
                "log_line_sha256": hashlib.sha256(log_line.encode()).hexdigest(),
            }
        }
        result, allowance = self.fixture.analyze()
        self.assertIsNone(allowance)
        self.assertIn("diagnostic-residual-tail-uninstrumentable", result["reasons"])

    def test_ct14_alignment_inside_offset_intersection_is_admissible(self) -> None:
        self.fixture.rewrite_row("h43_n14", 20, tpot_ms=8.0, concentrated_tail=True)
        self.fixture.clock_offset_bounds["ct14"] = (0, 50_000_000)
        gap = MODULE.top3_gap_details(
            self.fixture.rows["h43_n14"][22]["stream"]["events"]
        )[0]
        event_realtime_ns = (
            gap["start_realtime_ns"] + 50_000_000 + gap["end_realtime_ns"]
        ) // 2
        log_line = (
            f"{rfc3339_nano(event_realtime_ns)} " "rank0-scheduler scheduler-stall"
        )
        self.fixture.predeclared_stall_owners = ["rank0-scheduler"]
        self.fixture.stall_owner_by_index = {"20": "rank0-scheduler"}
        self.fixture.stall_owner_evidence_by_index = {
            "20": {
                "owner": "rank0-scheduler",
                "node": "ct14",
                "event_kind": "scheduler-stall",
                "event_realtime_ns": event_realtime_ns,
                "gap_rank": 1,
                "log_line_utf8": log_line,
                "log_line_sha256": hashlib.sha256(log_line.encode()).hexdigest(),
            }
        }
        _, allowance = self.fixture.analyze()
        self.assertIsNotNone(allowance)

    def test_endpoint_subset_tail_is_carried_into_allowance(self) -> None:
        self.fixture.rewrite_row("h43_n14", 5, tpot_ms=8.0)
        _, allowance = self.fixture.analyze()
        self.assertIn(
            "endpoint_subset-tail",
            allowance["workstream_reasons"]["C_DIRECT_DECODE"],
        )
        self.assertGreater(
            allowance["scope_allowances"]["endpoint_subset"]["tail"]["tpot_max_excess"],
            0,
        )

    def test_failed_role_gate_is_rejected(self) -> None:
        manifest, _ = self.fixture.seal()
        manifest["roles"]["h43_n14"]["log_coverage_passed"] = False
        run_path = self.fixture.root / "run-manifest-invalid.json"
        dump_json(run_path, manifest)
        with self.assertRaisesRegex(MODULE.AnalysisError, "log_coverage_passed"):
            MODULE.analyze(
                root=self.fixture.root,
                prompt_manifest=self.fixture.prompt_manifest,
                prompt_manifest_sha256=self.fixture.prompt_sha256,
                run_manifest=manifest,
                run_manifest_sha256=sha256_file(run_path),
                draws=10,
            )

    def test_e2m1_shortage_does_not_invalidate_h43_attribution(self) -> None:
        for index in range(4):
            self.fixture.rewrite_row("e2m1_n14", index + 20, completion_tokens=511)
        result, allowance = self.fixture.analyze()
        self.assertIsNotNone(allowance)
        self.assertNotIn("e2m1_n14", result["comparisons"])
        self.assertEqual(len(result["e2m1_retained_indices"]), 26)


if __name__ == "__main__":
    unittest.main()
