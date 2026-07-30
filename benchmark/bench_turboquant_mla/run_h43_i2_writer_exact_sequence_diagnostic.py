#!/usr/bin/env python3
"""Run the literal H43 I2 q1/q5 writer sequence and restore service."""

from __future__ import annotations

import sys
from typing import Any

from run_h43_i2_qualification import I2Qualification, parse_args
from run_h43_i2_writer_allocation_diagnostic import (
    DIAGNOSTIC_PYTHONPATH,
    DIAGNOSTIC_SCRIPT,
    WriterAllocationDiagnostic,
)


class WriterExactSequenceDiagnostic(WriterAllocationDiagnostic):
    def timeout_command_contract(self) -> dict[str, dict[str, Any]]:
        common = I2Qualification.timeout_command_contract(self)
        return {
            key: common[key]
            for key in (
                "gpu_identity_sample",
                "idle_compute_check",
                "tokenspeed_source_manifest_each",
                "native_prebuilt_load",
                "h40_contract",
                "pdl_source_order",
                "gpu_stage_surface",
                "writer_test_cli",
            )
        } | {
            "diagnostic_cli": {
                "timeout": 120,
                "processes": 1,
                "phase": "pre-outage",
                "gpu_access": False,
            },
            "exact_q1_q5_sequence": {"timeout": 600, "processes": 1},
        }

    def qualification(self) -> None:
        self._assert_no_compute_process("idle-after-exact-sequence-start")
        self._sample_idle_gpu("exact-sequence-gpu-start")
        name = "exact-correctness-q1-q5"
        value = self._candidate(
            name,
            [
                "env",
                "-u",
                "CUDA_MODULE_LOADING",
                f"PYTHONPATH={DIAGNOSTIC_PYTHONPATH}",
                "python3",
                f"/results/{DIAGNOSTIC_SCRIPT}",
                "--mode",
                "exact-sequence",
                "--tokens",
                "1",
                "--write-codebook",
                "1",
                "--prelude",
                "correctness",
                "--iterations",
                "100",
                "--output-prefix",
                f"/results/{name}",
                "--prior-q1-graph",
                "0",
            ],
            timeout=600,
            expected_json_status="PASS",
        )
        expected = {
            "mode": "exact-sequence",
            "tokens": [1, 5],
            "write_codebook": True,
            "prelude": "correctness",
            "prelude_cases": 352,
            "prior_q1_graph": False,
            "iterations": 100,
            "seed": 20260729,
            "cuda_module_loading": "UNSET",
            "qualification_claim": False,
        }
        if value is None or any(
            value.get(key) != expected_value for key, expected_value in expected.items()
        ):
            raise RuntimeError("exact q1/q5 diagnostic identity failed")
        sequence = value.get("sequence", {})
        for arm in ("q1", "q5"):
            if (
                sequence.get(arm, {}).get("eager", {}).get("iterations") != 100
                or sequence.get(arm, {}).get("graph_replay", {}).get("iterations")
                != 100
            ):
                raise RuntimeError(f"exact diagnostic sequence is incomplete: {arm}")
        self.diagnostic_results[name] = value
        self._assert_no_compute_process("idle-before-exact-sequence-end")
        self._sample_idle_gpu("exact-sequence-gpu-end")


def main() -> None:
    args = parse_args()
    campaign: WriterExactSequenceDiagnostic | None = None
    try:
        campaign = WriterExactSequenceDiagnostic(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial exact-sequence evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
