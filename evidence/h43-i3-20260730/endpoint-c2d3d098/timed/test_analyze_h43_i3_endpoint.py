#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("analyze_h43_i3_endpoint.py")
SPEC = importlib.util.spec_from_file_location("analyze_h43_i3_endpoint", MODULE_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.hashes = [f"prompt-{index}" for index in range(12)]
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps({"request_prompt_sha256": self.hashes}), encoding="utf-8"
        )
        analyzer.BOOTSTRAP_DRAWS = 200

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_role(
        self,
        role: str,
        *,
        base_epoch: int,
        tpot_ms: float,
        ttft_ms: float = 100.0,
        response_chunks: int = 200,
        measured_tpots: list[float] | None = None,
    ) -> None:
        if measured_tpots is not None and len(measured_tpots) != 10:
            raise ValueError("measured_tpots must contain ten values")
        role_dir = self.root / role
        role_dir.mkdir(exist_ok=True)
        rows = []
        for index in range(12):
            measured_index = index - 2
            row_tpot_ms = (
                measured_tpots[measured_index]
                if measured_tpots is not None and measured_index >= 0
                else tpot_ms
            )
            recorded = dt.datetime.fromtimestamp(
                base_epoch + index * 3, tz=dt.timezone.utc
            ).strftime("%Y%m%dT%H%M%SZ")
            rows.append(
                {
                    "cache_mode": "unique-prefix",
                    "prompt_sha256": self.hashes[index],
                    "completion_tokens_source": "usage",
                    "concurrency": 1,
                    "is_warmup": index < 2,
                    "request_index": -index - 1 if index < 2 else measured_index,
                    "ok": True,
                    "http_status": 200,
                    "error": None,
                    "finish_reason": "length",
                    "completion_tokens": 512,
                    "ignore_eos": False,
                    "model": "moonshotai/Kimi-K2.6",
                    "output_tokens_target": 512,
                    "prompt_tokens_target": 16000,
                    "response_chars": 2048,
                    "response_chunks": response_chunks,
                    "scenario_id": "s0001-synthetic-synthetic-p16000-o512-c1",
                    "schema_version": 1,
                    "sglang_rid": "",
                    "task": "synthetic",
                    "temperature": 0.0,
                    "thinking": "unset",
                    "top_p": 1.0,
                    "tpot_ms": row_tpot_ms,
                    "ttft_ms": ttft_ms,
                    "e2e_ms": 1000.0,
                    "decode_tok_s": 1000.0 / row_tpot_ms,
                    "recorded_at_utc": recorded,
                    "prompt_tokens": 10218,
                    "response_sha256": f"{role}-{index}",
                    "workload": "synthetic",
                }
            )
        (role_dir / "raw.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def write_pass_fixture(self) -> None:
        self.write_role("fp8_pre", base_epoch=1000, tpot_ms=3.0)
        self.write_role("e2m1", base_epoch=2000, tpot_ms=2.9)
        self.write_role("h43", base_epoch=3000, tpot_ms=3.1)
        self.write_role("fp8_post", base_epoch=4000, tpot_ms=3.0)

    def test_type7_p95_uses_position_8_55_for_ten_values(self) -> None:
        self.assertTrue(
            math.isclose(
                analyzer.percentile_type7(list(range(1, 11)), 0.95),
                9.55,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_full_pass_fixture(self) -> None:
        self.write_pass_fixture()
        result = analyzer.analyze(self.root, self.manifest)
        self.assertEqual(result["verdict"], "PASS_PERF")
        self.assertTrue(result["fp8_flank_valid"])
        self.assertTrue(result["timed_bracket_valid"])
        self.assertAlmostEqual(
            result["comparisons"]["h43"]["tpot_delta"], 3.1 / 3.0 - 1.0
        )
        self.assertAlmostEqual(
            result["comparisons"]["h43"]["per_index_tpot_ratio_p95_type7"],
            3.1 / 3.0,
        )

    def test_fp8_tpot_drift_forces_no_decision(self) -> None:
        self.write_pass_fixture()
        self.write_role("fp8_post", base_epoch=4000, tpot_ms=3.06)
        result = analyzer.analyze(self.root, self.manifest)
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertEqual(result["reason"], "fp8-flank-drift")

    def test_invalid_flank_precedes_failing_candidate(self) -> None:
        self.write_pass_fixture()
        self.write_role(
            "h43",
            base_epoch=3000,
            tpot_ms=3.3,
            measured_tpots=[3.3] * 9 + [3.8],
        )
        self.write_role("fp8_post", base_epoch=4000, tpot_ms=3.06)
        result = analyzer.analyze(self.root, self.manifest)
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertEqual(result["reason"], "fp8-flank-drift")
        self.assertEqual(result["h43_tpot_band"], "FAIL_PERF")
        self.assertFalse(result["h43_tail_pass"])

    def test_valid_flank_tail_only_failure_forces_fail_perf(self) -> None:
        self.write_pass_fixture()
        self.write_role(
            "h43",
            base_epoch=3000,
            tpot_ms=3.0,
            measured_tpots=[3.0] * 9 + [3.5],
        )
        result = analyzer.analyze(self.root, self.manifest)
        self.assertEqual(result["h43_tpot_band"], "PASS")
        self.assertEqual(result["h43_ttft_band"], "PASS")
        self.assertFalse(result["h43_tail_pass"])
        self.assertEqual(result["verdict"], "FAIL_PERF")
        self.assertEqual(result["reason"], "performance-gate")

    def test_ttft_failure_forces_fail_perf(self) -> None:
        self.write_pass_fixture()
        self.write_role("h43", base_epoch=3000, tpot_ms=3.1, ttft_ms=112.0)
        result = analyzer.analyze(self.root, self.manifest)
        self.assertEqual(result["h43_tpot_band"], "PASS")
        self.assertEqual(result["h43_ttft_band"], "FAIL_PERF")
        self.assertTrue(result["h43_tail_pass"])
        self.assertEqual(result["verdict"], "FAIL_PERF")

    def test_fp8_ttft_drift_cap_forces_no_decision(self) -> None:
        self.write_pass_fixture()
        self.write_role(
            "fp8_post", base_epoch=4000, tpot_ms=3.0, ttft_ms=102.1
        )
        result = analyzer.analyze(self.root, self.manifest)
        self.assertGreater(abs(result["fp8_flank_drift"]["ttft_ms"]), 0.02)
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertEqual(result["reason"], "fp8-flank-drift")

    def test_fp8_acceptance_drift_cap_forces_no_decision(self) -> None:
        self.write_pass_fixture()
        self.write_role(
            "fp8_post", base_epoch=4000, tpot_ms=3.0, response_chunks=211
        )
        result = analyzer.analyze(self.root, self.manifest)
        self.assertGreater(abs(result["fp8_flank_drift"]["acceptance"]), 0.05)
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertEqual(result["reason"], "fp8-flank-drift")

    def test_fp8_target_verify_drift_cap_forces_no_decision(self) -> None:
        self.write_pass_fixture()
        self.write_role(
            "fp8_post", base_epoch=4000, tpot_ms=3.0, response_chunks=205
        )
        result = analyzer.analyze(self.root, self.manifest)
        self.assertLess(abs(result["fp8_flank_drift"]["acceptance"]), 0.05)
        self.assertGreater(
            abs(result["fp8_flank_drift"]["target_verify_ms"]), 0.02
        )
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertEqual(result["reason"], "fp8-flank-drift")

    def test_swapped_attribution_roles_are_rejected(self) -> None:
        self.write_pass_fixture()
        self.write_role("e2m1", base_epoch=3000, tpot_ms=2.9)
        self.write_role("h43", base_epoch=2000, tpot_ms=3.1)
        with self.assertRaisesRegex(ValueError, "timed role order is invalid"):
            analyzer.analyze(self.root, self.manifest)

    def test_duplicate_frozen_prompt_is_rejected(self) -> None:
        self.hashes[-1] = self.hashes[-2]
        self.manifest.write_text(
            json.dumps({"request_prompt_sha256": self.hashes}), encoding="utf-8"
        )
        self.write_pass_fixture()
        with self.assertRaisesRegex(ValueError, "prompt hashes must be unique"):
            analyzer.analyze(self.root, self.manifest)

    def test_overlong_timed_bracket_forces_no_decision(self) -> None:
        self.write_pass_fixture()
        self.write_role("fp8_post", base_epoch=20_000, tpot_ms=3.0)
        result = analyzer.analyze(self.root, self.manifest)
        self.assertEqual(result["verdict"], "NO_DECISION")
        self.assertEqual(result["reason"], "timed-bracket-span")

    def test_threshold_band_edges(self) -> None:
        self.assertEqual(analyzer.band(0.045, 0.045, 0.055), "PASS")
        self.assertEqual(analyzer.band(0.05, 0.045, 0.055), "NO_DECISION")
        self.assertEqual(analyzer.band(0.055, 0.045, 0.055), "NO_DECISION")
        self.assertEqual(analyzer.band(0.055001, 0.045, 0.055), "FAIL_PERF")


if __name__ == "__main__":
    unittest.main(verbosity=2)
