#!/usr/bin/env python3
"""Run one H43 I2 synthetic PDL sensitivity pair and restore service."""

from __future__ import annotations

import json
import sys
from typing import Any

import run_h43_maintenance as h43
from run_h43_i2_aot_preparation import I2_COMMIT, I2_NATIVE_SHA256
from run_h43_i2_qualification import (
    I2Qualification,
    MEMORY_CONTRACT,
    QUALIFICATION_PDL_PROBE,
    canonical_digest,
    parse_args,
    sha256_bytes,
)


class PdlSensitivityDiagnostic(I2Qualification):
    def __init__(self, args: Any):
        super().__init__(args)
        self.diagnostic_results: dict[str, dict[str, Any]] = {}

    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        return (
            {
                "experiment": 1800,
                "failsafe": 1860,
                "alert": 2460,
                "terminal": 3060,
            },
            "h43_i2_pdl_sensitivity_diagnostic",
        )

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
        } | {"pdl_sensitivity_each": {"timeout": 300, "processes": 2}}

    def qualification(self) -> None:
        self._assert_no_compute_process("idle-after-pdl-diagnostic-start")
        self._sample_idle_gpu("pdl-diagnostic-gpu-start")
        common = [
            "python3",
            f"/results/{QUALIFICATION_PDL_PROBE}",
            "--context",
            "10219",
            "--q-len",
            "1",
            "--split-kv",
            "64",
            "--steps",
            "1000",
        ]
        reference = self._reference(
            "pdl-sensitivity-reference",
            [*common, "--reader", "pre-move"],
            timeout=300,
            expected_json_status="PASS",
        )
        candidate = self._candidate(
            "pdl-sensitivity-candidate",
            [*common, "--reader", "post-wait"],
            timeout=300,
            expected_json_status="PASS",
        )
        if (
            reference is None
            or reference.get("mismatched_steps", 0) <= 0
            or reference.get("ordered_control_mismatches") != 0
        ):
            raise RuntimeError("reference did not establish positive PDL sensitivity")
        if (
            candidate is None
            or candidate.get("mismatched_steps") != 0
            or candidate.get("ordered_control_mismatches") != 0
        ):
            raise RuntimeError("candidate did not establish post-wait ordering")
        self.diagnostic_results = {
            "reference": reference,
            "candidate": candidate,
        }
        self._assert_no_compute_process("idle-before-pdl-diagnostic-end")
        self._sample_idle_gpu("pdl-diagnostic-gpu-end")

    def finalize_qualification(self) -> None:
        xid_evidence: dict[str, str] = {}
        for host in (self.host0, self.host1):
            path = self.local_output / f"window-journal-{host}.txt"
            content = path.read_text(encoding="utf-8")
            h43.write_remote_root_file(
                self.host0,
                f"{self.results}/window-journal-{host}.txt",
                content,
                "0644",
            )
            xid_evidence[host] = sha256_bytes(content.encode())
        result = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_PDL_SENSITIVITY_DIAGNOSTIC",
            "campaign": self.campaign,
            "machine_scope": ["ct13", "ct14"],
            "gpu_under_test": "ct13:GPU0",
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "tokenspeed_candidate_commit": self.candidate["commit"],
            "tokenspeed_reference_commit": self.reference["commit"],
            "native_extension_sha256": I2_NATIVE_SHA256,
            "qualification_pdl_probe_sha256": self.qualification_pdl_probe_sha256,
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
        name = "H43_I2_PDL_SENSITIVITY_DIAGNOSTIC.json"
        h43.write_remote_root_file(
            self.host0, f"{self.results}/{name}", payload, "0644"
        )
        (self.local_output / name).write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    campaign: PdlSensitivityDiagnostic | None = None
    try:
        campaign = PdlSensitivityDiagnostic(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial PDL evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
