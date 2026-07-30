#!/usr/bin/env python3
"""Run the isolated H43 I2 writer-allocation diagnostic and restore service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import run_h43_maintenance as h43
from run_h43_i2_aot_preparation import I2_COMMIT, I2_NATIVE_SHA256
from run_h43_i2_qualification import (
    I2Qualification,
    MEMORY_CONTRACT,
    canonical_digest,
    parse_args,
    sha256_bytes,
)

DIAGNOSTIC_SCRIPT = "diagnose_h43_i2_writer_allocation.py"
DIAGNOSTIC_PYTHONPATH = "/i2/python:/i2:/work:/i2/benchmark/bench_turboquant_mla"


class WriterAllocationDiagnostic(I2Qualification):
    def __init__(self, args: Any):
        super().__init__(args)
        self.diagnostic_results: dict[str, dict[str, Any]] = {}
        self.diagnostic_script_sha256 = ""

    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        return (
            {
                "experiment": 1800,
                "failsafe": 1860,
                "alert": 2460,
                "terminal": 3060,
            },
            "h43_i2_writer_allocation_diagnostic",
        )

    def timeout_command_contract(self) -> dict[str, dict[str, Any]]:
        common = super().timeout_command_contract()
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
            "fresh_diagnostic_arm": {"timeout": 300, "processes": 5},
            "post_correctness_diagnostic_arm": {
                "timeout": 600,
                "processes": 2,
            },
        }

    def setup(self) -> None:
        super().setup()
        script = Path(__file__).resolve().with_name(DIAGNOSTIC_SCRIPT)
        payload = script.read_bytes()
        self.diagnostic_script_sha256 = sha256_bytes(payload)
        remote_script = f"{self.results}/{DIAGNOSTIC_SCRIPT}"
        container_script = f"/results/{DIAGNOSTIC_SCRIPT}"
        h43.write_remote_root_file(
            self.host0, remote_script, payload.decode("utf-8"), "0444"
        )
        value = self._preoutage_run(
            self.candidate,
            "allocation-diagnostic-cli",
            [
                "env",
                f"PYTHONPATH={DIAGNOSTIC_PYTHONPATH}",
                "python3",
                container_script,
                "--help",
            ],
            timeout=120,
        )
        if value is not None:
            raise RuntimeError("diagnostic CLI unexpectedly emitted a JSON result")
        help_text = self._read_remote_text("allocation-diagnostic-cli.stdout.log")
        required_options = (
            "--mode",
            "--tokens",
            "--write-codebook",
            "--prelude",
            "--iterations",
            "--output-prefix",
            "--prior-q1-graph",
        )
        if any(option not in help_text for option in required_options):
            raise RuntimeError("diagnostic CLI is missing a required option")
        evidence = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_WRITER_ALLOCATION_DIAGNOSTIC_PREFLIGHT",
            "tooling_commit": self.tooling_commit,
            "diagnostic_script": remote_script,
            "diagnostic_script_sha256": self.diagnostic_script_sha256,
            "record": self.records["allocation-diagnostic-cli"],
            "qualification_claim": False,
        }
        text = json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n"
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/H43_I2_WRITER_ALLOCATION_DIAGNOSTIC_PREFLIGHT.json",
            text,
            "0644",
        )
        (
            self.local_output / "H43_I2_WRITER_ALLOCATION_DIAGNOSTIC_PREFLIGHT.json"
        ).write_text(text, encoding="utf-8")

    def qualification(self) -> None:
        self._assert_no_compute_process("idle-after-diagnostic-start")
        self._sample_idle_gpu("diagnostic-gpu-start")
        script = f"/results/{DIAGNOSTIC_SCRIPT}"
        arms = (
            ("fresh-nocodebook-q1", 1, 0, "none", 0, "UNSET", 300),
            ("fresh-codebook-q1", 1, 1, "none", 0, "UNSET", 300),
            ("fresh-codebook-q1-eager", 1, 1, "none", 0, "EAGER", 300),
            ("fresh-nocodebook-q5", 5, 0, "none", 0, "UNSET", 300),
            ("fresh-codebook-q5", 5, 1, "none", 0, "UNSET", 300),
            (
                "post-correctness-codebook-q1",
                1,
                1,
                "correctness",
                0,
                "UNSET",
                600,
            ),
            (
                "post-correctness-q1-sequence-codebook-q5",
                5,
                1,
                "correctness",
                1,
                "UNSET",
                600,
            ),
        )
        for (
            name,
            tokens,
            codebook,
            prelude,
            prior_q1_graph,
            module_loading,
            timeout,
        ) in arms:
            environment = [
                "env",
                "-u",
                "CUDA_MODULE_LOADING",
                f"PYTHONPATH={DIAGNOSTIC_PYTHONPATH}",
            ]
            if module_loading != "UNSET":
                environment.append(f"CUDA_MODULE_LOADING={module_loading}")
            command = [
                *environment,
                "python3",
                script,
                "--mode",
                "single",
                "--tokens",
                str(tokens),
                "--write-codebook",
                str(codebook),
                "--prelude",
                prelude,
                "--iterations",
                "100",
                "--output-prefix",
                f"/results/{name}",
                "--prior-q1-graph",
                str(prior_q1_graph),
            ]
            value = self._candidate(
                name,
                command,
                timeout=timeout,
                expected_json_status="PASS",
            )
            expected = {
                "mode": "single",
                "tokens": tokens,
                "write_codebook": bool(codebook),
                "prelude": prelude,
                "prelude_cases": 352 if prelude == "correctness" else 0,
                "prior_q1_graph": bool(prior_q1_graph),
                "iterations": 100,
                "seed": 20260729,
                "cuda_module_loading": module_loading,
                "qualification_claim": False,
            }
            if value is None or any(
                value.get(key) != expected_value
                for key, expected_value in expected.items()
            ):
                raise RuntimeError(f"allocation diagnostic arm identity failed: {name}")
            for phase in (
                "eager",
                "eager_stepped",
                "graph_replay",
                "graph_replay_stepped",
            ):
                if value.get(phase, {}).get("iterations") != 100:
                    raise RuntimeError(
                        f"allocation diagnostic phase is incomplete: {name}"
                    )
            self.diagnostic_results[name] = value
        self._assert_no_compute_process("idle-before-diagnostic-end")
        self._sample_idle_gpu("diagnostic-gpu-end")

    def finalize_qualification(self) -> None:
        xid_evidence: dict[str, str] = {}
        for host in (self.host0, self.host1):
            local_path = self.local_output / f"window-journal-{host}.txt"
            content = local_path.read_text(encoding="utf-8")
            remote_name = f"window-journal-{host}.txt"
            h43.write_remote_root_file(
                self.host0, f"{self.results}/{remote_name}", content, "0644"
            )
            xid_evidence[host] = sha256_bytes(content.encode())
        result = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_WRITER_ALLOCATION_DIAGNOSTIC",
            "campaign": self.campaign,
            "machine_scope": ["ct13", "ct14"],
            "gpu_under_test": "ct13:GPU0",
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "sglang_source_manifest_digest": self.source_manifest_digest,
            "tokenspeed_candidate_commit": self.candidate["commit"],
            "candidate_image": self.candidate["image_id"],
            "native_extension_sha256": I2_NATIVE_SHA256,
            "qualification_frontend_test_sha256": (
                self.qualification_frontend_test_sha256
            ),
            "diagnostic_script_sha256": self.diagnostic_script_sha256,
            "memory_contract": MEMORY_CONTRACT,
            "arms": self.diagnostic_results,
            "arms_digest": canonical_digest(self.diagnostic_results),
            "records": self.records,
            "restoration": {
                "validated": self.validation_passed,
                "health": "PASS",
                "model": "PASS",
                "completion": "PASS",
                "gpu_health": "PASS",
                "xid_scan": "PASS",
                "window_journal_sha256": xid_evidence,
            },
            "qualification_claim": False,
            "performance_decision_authorized": False,
        }
        payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/H43_I2_WRITER_ALLOCATION_DIAGNOSTIC.json",
            payload,
            "0644",
        )
        (self.local_output / "H43_I2_WRITER_ALLOCATION_DIAGNOSTIC.json").write_text(
            payload, encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    campaign: WriterAllocationDiagnostic | None = None
    try:
        campaign = WriterAllocationDiagnostic(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial H43 I2 allocation evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
