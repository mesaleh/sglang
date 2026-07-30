#!/usr/bin/env python3
"""Run the sealed H43 complete-I1 graph gate with automatic N2 restoration."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import shlex
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import run_h43_maintenance as h43
from run_h43_i2_qualification import (
    I2_COMMIT,
    I2_NATIVE_SHA256,
    I2Qualification,
    MEMORY_CONTRACT,
    canonical_digest,
    last_json_value,
    sha256_bytes,
    sha256_path,
)

I2_TOOLING_COMMIT = "914e76c904824133883f260c3fc6f71abd53745a"
CONTEXT_SPLITS = {10219: 64, 37932: 40}
SEQUENCES = tuple(range(1, 11))
WARMUPS = 100
PAIRS = 20
REPLAYS = 100
BOOTSTRAP_DRAWS = 50_000
SEED = 20260729
SENTINEL_MAX_DRIFT_FRACTION = 0.005
SEQUENCE_PHASE_SECONDS = 600
ANALYSIS_PHASE_SECONDS = 900
# The archived longest fresh process took 26.698303 seconds. Twenty required
# sequences plus one replacement take 560.665 seconds at that conservative
# bound. Further per-sequence replacements remain subject to the same frozen
# 600-second global phase; 120 seconds is a fail-closed per-attempt timeout.
SEQUENCE_TIMEOUT_SECONDS = 120
ROUNDTRIP_MEMBERS = {
    (context, q_len): f"results/roundtrip-c{context}-q{q_len}.json"
    for context in CONTEXT_SPLITS
    for q_len in (1, 5)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preparation-state", type=Path, required=True)
    parser.add_argument("--i2-preparation-result", type=Path, required=True)
    parser.add_argument("--reader-qualification-tar", type=Path, required=True)
    parser.add_argument("--i2-qualification-result", type=Path, required=True)
    parser.add_argument("--i2-qualification-result-sha256", required=True)
    parser.add_argument("--i2-qualification-tar", type=Path, required=True)
    parser.add_argument("--i2-qualification-tar-sha256", required=True)
    parser.add_argument("--tooling-commit", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    args = parser.parse_args()
    args.mode = "qualification"
    args.decision_contract = None
    return args


class I1Qualification(I2Qualification):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.i2_qualification_result = args.i2_qualification_result.resolve()
        self.i2_qualification_tar = args.i2_qualification_tar.resolve()
        self.i2_result_sha256 = args.i2_qualification_result_sha256
        self.i2_tar_sha256 = args.i2_qualification_tar_sha256
        self.i2_summary: dict[str, Any] = {}
        self.roundtrips: dict[tuple[int, int], dict[str, Any]] = {}
        self.attempts: list[dict[str, Any]] = []
        self.component_result: dict[str, Any] | None = None

    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        # 65 minutes through analysis plus 25 minutes for restore/validation.
        return (
            {
                "experiment": 3900,
                "failsafe": 3960,
                "alert": 4800,
                "terminal": 5400,
            },
            "h43_i1_complete_graph",
        )

    def _validate_i2_qualification(self) -> None:
        for label, expected in (
            ("I2 result", self.i2_result_sha256),
            ("I2 archive", self.i2_tar_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"{label} SHA-256 must be a lowercase full digest")
        if sha256_path(self.i2_qualification_result) != self.i2_result_sha256:
            raise ValueError("H43 I2 qualification result digest mismatch")
        if sha256_path(self.i2_qualification_tar) != self.i2_tar_sha256:
            raise ValueError("H43 I2 qualification archive digest mismatch")

        summary_bytes = self.i2_qualification_result.read_bytes()
        summary = json.loads(
            summary_bytes,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        restoration = summary.get("restoration", {})
        if (
            summary.get("status") != "PASS"
            or summary.get("experiment")
            != "H43_I2_CODEBOOK_INTEGRATION_QUALIFICATION"
            or summary.get("tooling_commit") != I2_TOOLING_COMMIT
            or summary.get("sglang_source_commit") != I2_COMMIT
            or summary.get("native_extension_sha256") != I2_NATIVE_SHA256
            or summary.get("sglang_source_manifest_digest")
            != self.source_manifest_digest
            or summary.get("tokenspeed_candidate_commit")
            != self.candidate["commit"]
            or summary.get("memory_contract") != MEMORY_CONTRACT
            or restoration.get("validated") is not True
            or any(
                restoration.get(key) != "PASS"
                for key in ("health", "model", "completion", "gpu_health", "xid_scan")
            )
        ):
            raise ValueError("H43 I2 qualification is not the required sealed PASS")

        with tarfile.open(self.i2_qualification_tar, "r") as archive:
            summary_member = archive.getmember("results/H43_I2_QUALIFICATION.json")
            if not summary_member.isfile():
                raise ValueError("archived I2 qualification result is not a file")
            archived_summary = archive.extractfile(summary_member)
            if archived_summary is None or archived_summary.read() != summary_bytes:
                raise ValueError(
                    "I2 archive and standalone qualification result differ"
                )
            roundtrips: dict[tuple[int, int], dict[str, Any]] = {}
            for key, member_name in ROUNDTRIP_MEMBERS.items():
                member = archive.getmember(member_name)
                if not member.isfile():
                    raise ValueError(f"I2 roundtrip is not a file: {member_name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"cannot read I2 roundtrip: {member_name}")
                value = json.loads(
                    handle.read(),
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(token)
                    ),
                )
                context, q_len = key
                if (
                    value.get("status") != "PASS"
                    or value.get("experiment")
                    != "H43_I2_WRITER_CODEBOOK_READER_COMPOSITION"
                    or value.get("context") != context
                    or value.get("q_len") != q_len
                    or value.get("split_kv") != CONTEXT_SPLITS[context]
                    or value.get("selected_row_bytes") != 338
                    or value.get("codebook_materialized") is not True
                    or value.get("sticky_status") != 0
                ):
                    raise ValueError(f"I2 roundtrip contract mismatch: {member_name}")
                roundtrips[key] = value
        self.i2_summary = summary
        self.roundtrips = roundtrips

    def _validate_i1_source_lineage(self) -> None:
        repository = Path(
            h43.run(
                [
                    "git",
                    "-C",
                    str(Path(__file__).parent),
                    "rev-parse",
                    "--show-toplevel",
                ],
                timeout=30,
            ).stdout.strip()
        )
        for filename in (
            "bench_h41_i1_integrated.py",
            "analyze_h41_i1_integrated.py",
        ):
            path = Path(__file__).resolve().with_name(filename)
            relative = path.relative_to(repository).as_posix()
            sealed = h43.run(
                ["git", "-C", str(repository), "show", f"{I2_COMMIT}:{relative}"],
                timeout=60,
            ).stdout.encode()
            if path.read_bytes() != sealed:
                raise ValueError(f"I1 tool differs from sealed source: {relative}")

    def validate_inputs(self) -> None:
        super().validate_inputs()
        self._validate_i1_source_lineage()
        self._validate_i2_qualification()

    def setup(self) -> None:
        super().setup()
        i1_root = f"{self.results}/i1"
        h43.remote(
            self.host0,
            [
                "install",
                "-d",
                "-m",
                "0755",
                f"{i1_root}/raw",
                f"{i1_root}/selected",
                f"{i1_root}/roundtrip",
            ],
            timeout=60,
        )
        for (context, q_len), value in self.roundtrips.items():
            directory = f"{i1_root}/roundtrip/context{context}/q{q_len}"
            h43.remote(
                self.host0, ["install", "-d", "-m", "0755", directory], timeout=30
            )
            h43.write_remote_root_file(
                self.host0,
                f"{directory}/result.json",
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
                "0444",
            )

        script = Path(__file__).resolve()
        benchmark = script.with_name("bench_h41_i1_integrated.py")
        analyzer = script.with_name("analyze_h41_i1_integrated.py")
        timeout_budget = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I1_TIMEOUT_BUDGET",
            "maintenance_seconds": self.maintenance_contract()[0],
            "phase_seconds": {
                "stop_and_idle": 300,
                "sequences_including_one_replacement": SEQUENCE_PHASE_SECONDS,
                "analysis_and_archive": ANALYSIS_PHASE_SECONDS,
                "restore_and_validation": 1500,
                "shared_reserve": 2100,
                "absolute": 5400,
            },
            "per_sequence_timeout_seconds": SEQUENCE_TIMEOUT_SECONDS,
        }
        preflight = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I1_COMPLETE_GRAPH_PREFLIGHT",
            "campaign": self.campaign,
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "native_extension_sha256": I2_NATIVE_SHA256,
            "i1_runner_sha256": sha256_path(script),
            "i1_benchmark_sha256": sha256_path(benchmark),
            "i1_analyzer_sha256": sha256_path(analyzer),
            "i2_qualification": {
                "result_sha256": self.i2_result_sha256,
                "archive_sha256": self.i2_tar_sha256,
                "campaign": self.i2_summary["campaign"],
                "status": self.i2_summary["status"],
            },
            "contract": {
                "contexts_and_splits": CONTEXT_SPLITS,
                "sequences": list(SEQUENCES),
                "warmups": WARMUPS,
                "pairs": PAIRS,
                "replays": REPLAYS,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "seed": SEED,
                "q_len": 5,
                "reader_pdl": True,
                "selected_warps": 8,
                "selected_row_bytes": 338,
                "sentinel_max_drift_fraction": SENTINEL_MAX_DRIFT_FRACTION,
                "maximum_replacements_per_sequence": 1,
            },
        }
        for filename, value in (
            ("H43_I1_TIMEOUT_BUDGET.json", timeout_budget),
            ("H43_I1_PREFLIGHT.json", preflight),
        ):
            payload = (
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
            (self.local_output / filename).write_text(payload, encoding="utf-8")
            h43.write_remote_root_file(
                self.host0, f"{self.results}/{filename}", payload, "0644"
            )

    @staticmethod
    def _sentinel_reasons(
        value: dict[str, Any],
    ) -> tuple[list[str], float | None]:
        reasons: list[str] = []
        before = value.get("sentinel_before", {})
        after = value.get("sentinel_after", {})
        try:
            before_us = float(before["graph_us"])
            after_us = float(after["graph_us"])
        except (KeyError, TypeError, ValueError):
            return [
                "sentinel timings are absent, non-finite, or non-positive"
            ], None
        if not all(math.isfinite(item) and item > 0 for item in (before_us, after_us)):
            return ["sentinel timings are non-finite or non-positive"], None
        drift = abs(after_us - before_us) / ((after_us + before_us) / 2.0)
        if before.get("telemetry_valid") is not True:
            reasons.append("sentinel-before telemetry is invalid")
        if after.get("telemetry_valid") is not True:
            reasons.append("sentinel-after telemetry is invalid")
        if drift > SENTINEL_MAX_DRIFT_FRACTION:
            reasons.append(
                f"sentinel drift {drift:.9f} exceeds {SENTINEL_MAX_DRIFT_FRACTION:.9f}"
            )
        return reasons, drift

    def _write_json(self, remote_path: str, value: dict[str, Any], mode: str) -> None:
        h43.write_remote_root_file(
            self.host0,
            remote_path,
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            mode,
        )

    def _write_attempt_bundle(
        self,
        name: str,
        stdout: str,
        stderr: str,
        value: dict[str, Any] | None,
        attempt_record: dict[str, Any] | None,
        raw_directory: str | None,
        selected_directory: str | None,
    ) -> None:
        files: list[tuple[str, bytes, int]] = [
            (f"{self.results}/{name}.stdout.log", stdout.encode(), 0o644),
            (f"{self.results}/{name}.stderr.log", stderr.encode(), 0o644),
        ]
        if value is not None:
            assert attempt_record is not None and raw_directory is not None
            value_bytes = (
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode()
            validity_bytes = (
                json.dumps(
                    attempt_record,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            files.extend(
                (
                    (f"{self.results}/{name}.json", value_bytes, 0o644),
                    (f"{raw_directory}/result.json", value_bytes, 0o444),
                    (f"{raw_directory}/validity.json", validity_bytes, 0o444),
                )
            )
            if selected_directory is not None:
                files.append(
                    (f"{selected_directory}/result.json", value_bytes, 0o444)
                )

        encoded = [
            (path, base64.b64encode(payload).decode(), mode)
            for path, payload, mode in files
        ]
        directories = [path for path in (raw_directory, selected_directory) if path]
        script = f"""set -euo pipefail
python3 - <<'PY'
import base64
import os

directories = {directories!r}
files = {encoded!r}
for directory in directories:
    os.makedirs(directory, mode=0o755, exist_ok=False)
for path, payload, mode in files:
    data = base64.b64decode(payload, validate=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
"""
        h43.remote_script(self.host0, script, [], timeout=120)

    def _run_attempt(
        self, context: int, sequence: int, attempt: int, timeout: int
    ) -> tuple[dict[str, Any], list[str], float | None]:
        allocation_order = "candidate-first" if sequence == 2 else "control-first"
        name = f"i1-c{context}-seq{sequence:02d}-attempt{attempt:02d}"
        command = [
            "python3",
            "/i2/benchmark/bench_turboquant_mla/bench_h41_i1_integrated.py",
            "--context",
            str(context),
            "--split-kv",
            str(CONTEXT_SPLITS[context]),
            "--allocation-order",
            allocation_order,
            "--warmups",
            str(WARMUPS),
            "--samples",
            str(PAIRS),
            "--replays-per-sample",
            str(REPLAYS),
            "--seed",
            str(SEED),
            "--sequence",
            str(sequence),
        ]
        wrapped = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
            *command,
        ]
        completed = self.exec_candidate(wrapped, timeout=timeout + 30, check=False)
        record: dict[str, Any] = {
            "returncode": completed.returncode,
            "command": shlex.join(wrapped),
            "stdout": f"{name}.stdout.log",
            "stderr": f"{name}.stderr.log",
        }
        self.records[name] = record
        if completed.returncode:
            self._write_attempt_bundle(
                name,
                completed.stdout,
                completed.stderr,
                None,
                None,
                None,
                None,
            )
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
        value = last_json_value(completed.stdout, name)
        record["json_status"] = value.get("status")
        if value.get("status") != "TIMING_ONLY":
            self._write_attempt_bundle(
                name,
                completed.stdout,
                completed.stderr,
                None,
                None,
                None,
                None,
            )
            raise RuntimeError(
                f"{name} status {value.get('status')!r} != 'TIMING_ONLY'"
            )
        reasons, drift = self._sentinel_reasons(value)
        raw_directory = (
            f"{self.results}/i1/raw/context{context}/seq{sequence:02d}/"
            f"attempt{attempt:02d}"
        )
        attempt_record = {
            "context": context,
            "sequence": sequence,
            "attempt": attempt,
            "allocation_order": allocation_order,
            "capture_order": value.get("capture_order"),
            "sentinel_before_us": value.get("sentinel_before", {}).get("graph_us"),
            "sentinel_after_us": value.get("sentinel_after", {}).get("graph_us"),
            "sentinel_drift_fraction": drift,
            "valid": not reasons,
            "reasons": reasons,
            "raw_result": (
                f"i1/raw/context{context}/seq{sequence:02d}/"
                f"attempt{attempt:02d}/result.json"
            ),
        }
        self.attempts.append(attempt_record)
        selected_directory = (
            f"{self.results}/i1/selected/context{context}/seq{sequence:02d}"
            if not reasons
            else None
        )
        self._write_attempt_bundle(
            name,
            completed.stdout,
            completed.stderr,
            value,
            attempt_record,
            raw_directory,
            selected_directory,
        )
        return value, reasons, drift

    def _write_component_result(
        self,
        status: str,
        analysis: dict[str, Any] | None,
        failure: dict[str, str] | None = None,
    ) -> None:
        result = {
            "schema_version": 1,
            "status": status,
            "experiment": "H43_I1_COMPLETE_GRAPH_COMPONENT_GATE",
            "campaign": self.campaign,
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "native_extension_sha256": I2_NATIVE_SHA256,
            "contexts_and_splits": CONTEXT_SPLITS,
            "valid_sequences_required_per_context": len(SEQUENCES),
            "attempts": self.attempts,
            "analysis": analysis,
            "failure": failure,
            "memory_contract": MEMORY_CONTRACT,
            "authorization": (
                "SUCCESSOR_IMAGE_BUILD_PERMITTED; "
                "ENDPOINT_AND_PRODUCTION_REMAIN_SEPARATE"
                if status == "PASS"
                else "ENDPOINT_WORK_BLOCKED"
            ),
        }
        self.component_result = result
        self._write_json(
            f"{self.results}/H43_I1_COMPONENT_RESULT.json", result, "0644"
        )

    def qualification(self) -> None:
        self._assert_no_compute_process("idle-after-candidate-start")
        self._sample_idle_gpu("i1-gpu-start")
        sequence_deadline = min(
            self.experiment_deadline, time.monotonic() + SEQUENCE_PHASE_SECONDS
        )
        for context in CONTEXT_SPLITS:
            for sequence in SEQUENCES:
                selected: dict[str, Any] | None = None
                for attempt in (1, 2):
                    remaining = int(sequence_deadline - time.monotonic())
                    if remaining < 1:
                        raise TimeoutError("H43 I1 sequence phase exceeded 600 seconds")
                    value, reasons, _ = self._run_attempt(
                        context,
                        sequence,
                        attempt,
                        min(SEQUENCE_TIMEOUT_SECONDS, remaining),
                    )
                    if not reasons:
                        selected = value
                        break
                    if attempt == 2:
                        self._assert_no_compute_process(
                            "idle-after-invalid-replacement"
                        )
                        self._sample_idle_gpu("i1-gpu-no-decision")
                        self._write_component_result("NO_DECISION", None)
                        return
                assert selected is not None

        analysis_deadline = min(
            self.experiment_deadline, time.monotonic() + ANALYSIS_PHASE_SECONDS
        )
        remaining = int(analysis_deadline - time.monotonic())
        if remaining < 1:
            raise TimeoutError("H43 I1 analysis phase has no remaining budget")
        command = [
            "python3",
            "/i2/benchmark/bench_turboquant_mla/analyze_h41_i1_integrated.py",
            "/results/i1/selected",
            "--roundtrip-root",
            "/results/i1/roundtrip",
            "--draws",
            str(BOOTSTRAP_DRAWS),
            "--seed",
            str(SEED),
        ]
        wrapped = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{remaining}s",
            *command,
        ]
        completed = self.exec_candidate(wrapped, timeout=remaining + 30, check=False)
        self._record_result("i1-analysis", completed, command=wrapped)
        analysis = last_json_value(completed.stdout, "i1-analysis")
        if analysis.get("status") not in {"PASS", "FAIL"}:
            raise RuntimeError("H43 I1 analyzer returned neither PASS nor FAIL")
        if [item.get("context") for item in analysis.get("analyses", [])] != list(
            CONTEXT_SPLITS
        ):
            raise RuntimeError("H43 I1 analyzer omitted or reordered a context")
        self._write_json(f"{self.results}/H43_I1_ANALYSIS.json", analysis, "0644")
        self._assert_no_compute_process("idle-before-i1-end")
        self._sample_idle_gpu("i1-gpu-end")
        self._write_component_result(analysis["status"], analysis)

    def finalize_qualification(self) -> None:
        if self.component_result is None:
            raise RuntimeError("H43 I1 component result was not produced")
        xid_evidence: dict[str, str] = {}
        for host in (self.host0, self.host1):
            local_path = self.local_output / f"window-journal-{host}.txt"
            content = local_path.read_text(encoding="utf-8")
            remote_name = f"window-journal-{host}.txt"
            h43.write_remote_root_file(
                self.host0, f"{self.results}/{remote_name}", content, "0644"
            )
            xid_evidence[host] = sha256_bytes(content.encode())

        inventory_code = r"""import hashlib,json,os,sys
root=sys.argv[1]
rows=[]
for current,_,files in os.walk(root):
    for name in files:
        path=os.path.join(current,name)
        relative=os.path.relpath(path,root).replace(os.sep,"/")
        digest=hashlib.sha256()
        with open(path,"rb") as handle:
            for block in iter(lambda:handle.read(1048576),b""):
                digest.update(block)
        rows.append({"path":relative,"sha256":digest.hexdigest(),
                     "bytes":os.path.getsize(path)})
rows.sort(key=lambda row:row["path"])
print(json.dumps(rows,separators=(",",":"),sort_keys=True))
"""
        inventory = json.loads(
            h43.remote(
                self.host0,
                ["python3", "-c", inventory_code, self.results],
                timeout=300,
            ).stdout
        )
        status = self.component_result["status"]
        summary = {
            **self.component_result,
            "experiment": "H43_I1_COMPLETE_GRAPH_QUALIFICATION",
            "machine_scope": ["ct13", "ct14"],
            "gpu_under_test": "ct13:GPU0",
            "i2_qualification": {
                "result_sha256": self.i2_result_sha256,
                "archive_sha256": self.i2_tar_sha256,
                "campaign": self.i2_summary["campaign"],
                "status": "PASS",
            },
            "restoration": {
                "validated": self.validation_passed,
                "health": "PASS",
                "model": "PASS",
                "completion": "PASS",
                "gpu_health": "PASS",
                "xid_scan": "PASS",
                "window_journal_sha256": xid_evidence,
            },
            "evidence_inventory_before_summary": inventory,
            "evidence_inventory_digest": canonical_digest(inventory),
            "decision_authorization": (
                "COMPONENT_PASS_ONLY; SUCCESSOR_IMAGE_BUILD_PERMITTED; "
                "ENDPOINT_AND_PRODUCTION_REMAIN_SEPARATE"
                if status == "PASS"
                else "ENDPOINT_WORK_BLOCKED"
            ),
        }
        payload = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        h43.write_remote_root_file(
            self.host0, f"{self.results}/H43_I1_QUALIFICATION.json", payload, "0644"
        )
        (self.local_output / "H43_I1_QUALIFICATION.json").write_text(
            payload, encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    campaign: I1Qualification | None = None
    try:
        campaign = I1Qualification(args)
        campaign.execute()
    except BaseException as error:
        if campaign is not None and campaign.local_output.exists():
            if campaign.service_stopped and campaign.component_result is None:
                try:
                    campaign._write_component_result(
                        "NO_DECISION",
                        None,
                        {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    )
                except BaseException as result_error:
                    print(
                        f"failed to record H43 I1 NO_DECISION: {result_error}",
                        file=sys.stderr,
                    )
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial H43 I1 evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
