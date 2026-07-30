#!/usr/bin/env python3
"""Run the focused H43 I2 writer sanitizers and restore service."""

from __future__ import annotations

import json
import sys
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


class WriterSanitizerDiagnostic(I2Qualification):
    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        return (
            {
                "experiment": 1800,
                "failsafe": 1860,
                "alert": 2460,
                "terminal": 3060,
            },
            "h43_i2_writer_sanitizer_diagnostic",
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
                "pdl_producer_source",
                "gpu_stage_surface",
                "writer_test_cli",
                "sanitizer_each",
            )
        }

    def qualification(self) -> None:
        self._assert_no_compute_process("idle-after-sanitizer-diagnostic-start")
        self._sample_idle_gpu("sanitizer-diagnostic-gpu-start")
        self._run_writer_sanitizers()
        self._assert_no_compute_process("idle-before-sanitizer-diagnostic-end")
        self._sample_idle_gpu("sanitizer-diagnostic-gpu-end")

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
        sanitizer_records = {
            name: value
            for name, value in self.records.items()
            if name.startswith("writer-sanitizer-")
        }
        result = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_WRITER_SANITIZER_DIAGNOSTIC",
            "campaign": self.campaign,
            "machine_scope": ["ct13", "ct14"],
            "gpu_under_test": "ct13:GPU0",
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "tokenspeed_candidate_commit": self.candidate["commit"],
            "native_extension_sha256": I2_NATIVE_SHA256,
            "qualification_frontend_test_sha256": (
                self.qualification_frontend_test_sha256
            ),
            "memory_contract": MEMORY_CONTRACT,
            "sanitizer_records": sanitizer_records,
            "sanitizer_records_digest": canonical_digest(sanitizer_records),
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
        name = "H43_I2_WRITER_SANITIZER_DIAGNOSTIC.json"
        h43.write_remote_root_file(
            self.host0, f"{self.results}/{name}", payload, "0644"
        )
        (self.local_output / name).write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    campaign: WriterSanitizerDiagnostic | None = None
    try:
        campaign = WriterSanitizerDiagnostic(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial sanitizer evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
