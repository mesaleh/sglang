#!/usr/bin/env python3
"""Safely prepare the two sealed 14-key AOT caches for H43 I2."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_h43_maintenance as h43

I2_PREPARATION = (
    "/var/lib/h43/h43-i3-postreview-20260730-1de12b79/preparation"
)
I2_SOURCE = f"{I2_PREPARATION}/source/sglang"
I2_NATIVE_DIR = f"{I2_PREPARATION}/native"
I2_NATIVE_SO = f"{I2_NATIVE_DIR}/sglang_tq_mla_frontend_sm100_h43_i3_v2.so"
I2_CACHE_PARENT = "/var/lib/h43-i2-codebook-cache"
I2_COMMIT = "1de12b79a5e0272917c0ae88af789772f173728d"
I2_NATIVE_SHA256 = "990fecac6a5dfc02178aebd808d11fa50964457b88702dc123800b25b4db46d8"
I2_KEY_FILES = {
    "python/sglang/jit_kernel/tq_mla_frontend.py": (
        "b71e967f1132e8c2492d29aa2262398662a5080fa9397772b6b409a2cf753af1"
    ),
    "python/sglang/jit_kernel/csrc/tq_mla_frontend/tq_mla_frontend_sm100.cu": (
        "819ed01e06b1181ae02a346402d105e484ab0927574710e207770e4b70d8083b"
    ),
    "benchmark/bench_turboquant_mla/prebuild_h43_i2_aot_cache.py": (
        "fbaa8628f8ef1003595e165440a1641a28fcaba4fd28fdcf8dbc8ed84efb972f"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preparation-state", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    args = parser.parse_args()
    args.mode = "qualification"
    args.decision_contract = None
    return args


class I2AOTPreparation(h43.Campaign):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.i2_cache_roots = {
            identity["role"]: (
                f"{I2_CACHE_PARENT}/{identity['role']}/"
                f"{identity['source_manifest_digest']}"
            )
            for identity in (self.candidate, self.reference)
        }
        self.reference_container_name = f"ct13-h43-i2-aot-reference-{self.campaign}"

    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        return (
            {
                "experiment": 600,
                "failsafe": 660,
                "alert": 1260,
                "terminal": 1800,
            },
            "i2_aot_preparation",
        )

    def validate_inputs(self) -> None:
        super().validate_inputs()
        for path in (I2_SOURCE, I2_NATIVE_SO):
            h43.remote(self.host0, ["test", "!", "-L", path], timeout=30)
        observed_native = h43.remote(
            self.host0, ["sha256sum", I2_NATIVE_SO], timeout=60
        ).stdout.split()[0]
        if observed_native != I2_NATIVE_SHA256:
            raise RuntimeError("H43 I2 native extension digest mismatch")
        writable = h43.remote(
            self.host0,
            [
                "find",
                I2_SOURCE,
                I2_NATIVE_DIR,
                "-not",
                "-type",
                "l",
                "-perm",
                "/222",
                "-print",
                "-quit",
            ],
            timeout=60,
        ).stdout.strip()
        if writable:
            raise RuntimeError(f"H43 I2 sealed preparation is writable: {writable}")
        for relative, expected in I2_KEY_FILES.items():
            path = f"{I2_SOURCE}/{relative}"
            h43.remote(self.host0, ["test", "!", "-L", path], timeout=30)
            observed = h43.remote(
                self.host0, ["sha256sum", path], timeout=60
            ).stdout.split()[0]
            if observed != expected:
                raise RuntimeError(f"H43 I2 sealed source changed: {relative}")
        for cache_root in self.i2_cache_roots.values():
            if not re.fullmatch(
                rf"{re.escape(I2_CACHE_PARENT)}/(candidate|reference)/[0-9a-f]{{64}}",
                cache_root,
            ):
                raise RuntimeError("unsafe H43 I2 cache path")
            exists = h43.remote(
                self.host0, ["test", "-e", cache_root], timeout=30, check=False
            )
            if exists.returncode == 0:
                raise RuntimeError(f"H43 I2 cache already exists: {cache_root}")

    def setup(self) -> None:
        super().setup()
        for role in ("candidate", "reference"):
            h43.remote(
                self.host0,
                ["install", "-d", "-m", "0755", f"{I2_CACHE_PARENT}/{role}"],
                timeout=60,
            )

    def start_candidate(self) -> None:
        if (
            h43.remote(
                self.host0,
                ["docker", "container", "inspect", self.container_name],
                timeout=30,
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError("candidate experiment container name already exists")
        candidate_cache = self.i2_cache_roots["candidate"]
        command = [
            "docker",
            "create",
            "--name",
            self.container_name,
            "--hostname",
            self.contract["machine"]["container_hostname"],
            "--gpus",
            f'"device={self.gpu_index}"',
            "--cap-add",
            "SYS_ADMIN",
            "--ipc",
            "host",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONPATH=/i2/python:/work",
            "--env",
            f"CUTE_DSL_CACHE_DIR={candidate_cache}",
            "--env",
            f"H43_AOT_MANIFEST={candidate_cache}/h43-aot-manifest.json",
            "--env",
            "H43_AOT_EXPECTED_ENTRIES=14",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={self.candidate['source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={self.candidate['installed_mla_sha256']}",
            "--env",
            "SGLANG_TQ_MLA_FRONTEND_SO=/native/sglang_tq_mla_frontend_sm100_h43_i3_v2.so",
            "--env",
            f"SGLANG_TQ_MLA_FRONTEND_SO_SHA256={I2_NATIVE_SHA256}",
            "--volume",
            f"{self.work}:/work:ro",
            "--volume",
            f"{I2_SOURCE}:/i2:ro",
            "--volume",
            f"{I2_NATIVE_DIR}:/native:ro",
            "--volume",
            f"{self.candidate['cache_root']}:{self.candidate['cache_root']}:ro",
            "--volume",
            f"{I2_CACHE_PARENT}:{I2_CACHE_PARENT}",
            "--volume",
            f"{self.results}:/results",
            self.candidate["image_id"],
            "sleep",
            "infinity",
        ]
        self.container_id = h43.remote(self.host0, command, timeout=120).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", self.container_id):
            raise RuntimeError("docker did not return an exact I2 container ID")
        self.restore_config_values.update(
            {
                "H43_EXPERIMENT_CONTAINER_NAME": self.container_name,
                "H43_EXPERIMENT_CONTAINER_ID": self.container_id,
            }
        )
        config = h43.shell_config(self.restore_config_values)
        for host in (self.host0, self.host1):
            h43.write_remote_root_file(host, f"{self.remote_root}/restore.conf", config)
        started = h43.remote(
            self.host0, ["docker", "start", self.container_id], timeout=120
        ).stdout.strip()
        if started != self.container_id:
            raise RuntimeError("docker did not start the exact I2 container")

    def _prebuild_arguments(self, identity: dict[str, Any], *, phase: str) -> list[str]:
        arguments = [
            "python3",
            "/i2/benchmark/bench_turboquant_mla/prebuild_h43_i2_aot_cache.py",
            "--contract",
            "/work/h43_codebook_ab_contract.json",
            "--cache-root",
            self.i2_cache_roots[identity["role"]],
            "--source-manifest-digest",
            identity["source_manifest_digest"],
            "--installed-mla-sha256",
            identity["installed_mla_sha256"],
            "--phase",
            phase,
        ]
        if phase == "extend":
            arguments.extend(["--base-cache-root", identity["cache_root"]])
        return arguments

    def _reference_run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        cache_root = self.i2_cache_roots["reference"]
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            "210s",
            "docker",
            "run",
            "--rm",
            "--name",
            self.reference_container_name,
            "--gpus",
            f'"device={self.gpu_index}"',
            "--cap-add",
            "SYS_ADMIN",
            "--ipc",
            "host",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONPATH=/i2/python:/work",
            "--env",
            f"CUTE_DSL_CACHE_DIR={cache_root}",
            "--env",
            f"H43_AOT_MANIFEST={cache_root}/h43-aot-manifest.json",
            "--env",
            "H43_AOT_EXPECTED_ENTRIES=14",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={self.reference['source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={self.reference['installed_mla_sha256']}",
            "--volume",
            f"{self.reference_work}:/work:ro",
            "--volume",
            f"{I2_SOURCE}:/i2:ro",
            "--volume",
            f"{self.reference['cache_root']}:{self.reference['cache_root']}:ro",
            "--volume",
            f"{I2_CACHE_PARENT}:{I2_CACHE_PARENT}",
            self.reference["image_id"],
            *arguments,
        ]
        try:
            return h43.remote(self.host0, command, timeout=240)
        finally:
            h43.remote(
                self.host0,
                ["docker", "rm", "--force", self.reference_container_name],
                timeout=60,
                check=False,
            )

    def _record_prebuild(
        self,
        role: str,
        phase: str,
        result: subprocess.CompletedProcess[str],
    ) -> None:
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"{role} {phase} prebuild produced no stdout")
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{role} {phase} prebuild did not end with JSON"
            ) from error
        if value.get("status") != "PASS" or value.get("phase") != phase:
            raise RuntimeError(f"{role} {phase} prebuild result is not PASS")
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/{role}-aot-{phase}.json",
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n",
            "0644",
        )
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/{role}-aot-{phase}.stdout.log",
            result.stdout,
            "0644",
        )
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/{role}-aot-{phase}.stderr.log",
            result.stderr,
            "0644",
        )

    def qualification(self) -> None:
        created: list[str] = []
        try:
            candidate_cache = self.i2_cache_roots["candidate"]
            extend = self.exec_candidate(
                self._prebuild_arguments(self.candidate, phase="extend"),
                timeout=240,
            )
            created.append(candidate_cache)
            self._record_prebuild("candidate", "extend", extend)
            warm = self.exec_candidate(
                self._prebuild_arguments(self.candidate, phase="warm"),
                timeout=180,
            )
            self._record_prebuild("candidate", "warm", warm)

            reference_cache = self.i2_cache_roots["reference"]
            reference_extend = self._reference_run(
                self._prebuild_arguments(self.reference, phase="extend")
            )
            created.append(reference_cache)
            self._record_prebuild("reference", "extend", reference_extend)
            reference_warm = self._reference_run(
                self._prebuild_arguments(self.reference, phase="warm")
            )
            self._record_prebuild("reference", "warm", reference_warm)
            for cache_root in created:
                h43.remote(self.host0, ["chmod", "-R", "a-w", cache_root], timeout=60)
        except BaseException:
            for cache_root in created:
                quarantine = f"{cache_root}.rejected-{self.campaign}"
                h43.remote(
                    self.host0,
                    ["mv", cache_root, quarantine],
                    timeout=60,
                    check=False,
                )
            raise

    def finalize_qualification(self) -> None:
        summary: dict[str, Any] = {
            "status": "PASS",
            "experiment": "H43_I2_AOT_PREPARATION",
            "sglang_commit": I2_COMMIT,
            "native_extension_sha256": I2_NATIVE_SHA256,
            "caches": {},
        }
        try:
            for identity in (self.candidate, self.reference):
                role = identity["role"]
                cache_root = self.i2_cache_roots[role]
                writable = h43.remote(
                    self.host0,
                    ["find", cache_root, "-perm", "/222", "-print", "-quit"],
                    timeout=60,
                ).stdout.strip()
                if writable:
                    raise RuntimeError(
                        f"sealed {role} cache remains writable: {writable}"
                    )
                manifest = h43.remote(
                    self.host0,
                    ["sha256sum", f"{cache_root}/h43-aot-manifest.json"],
                    timeout=60,
                ).stdout.split()[0]
                extension = json.loads(
                    h43.remote(
                        self.host0,
                        ["cat", f"{self.results}/{role}-aot-extend.json"],
                        timeout=60,
                    ).stdout
                )
                warm = json.loads(
                    h43.remote(
                        self.host0,
                        ["cat", f"{self.results}/{role}-aot-warm.json"],
                        timeout=60,
                    ).stdout
                )
                if (
                    extension.get("status") != "PASS"
                    or warm.get("status") != "PASS"
                    or len(extension["cache_after"]["files"]) != 29
                    or extension["cache_after"]["digest"]
                    != warm["cache_after"]["digest"]
                ):
                    raise RuntimeError(f"sealed {role} cache evidence is invalid")
                summary["caches"][role] = {
                    "root": cache_root,
                    "source_manifest_digest": identity["source_manifest_digest"],
                    "installed_mla_sha256": identity["installed_mla_sha256"],
                    "artifact_digest": extension["cache_after"]["digest"],
                    "manifest_sha256": manifest,
                    "entry_count": 14,
                    "file_count": 29,
                    "read_only": True,
                }
        except BaseException:
            for cache_root in self.i2_cache_roots.values():
                quarantine = f"{cache_root}.rejected-final-{self.campaign}"
                h43.remote(
                    self.host0,
                    ["mv", cache_root, quarantine],
                    timeout=60,
                    check=False,
                )
            raise
        (self.local_output / "H43_I2_AOT_PREPARATION.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    campaign: I2AOTPreparation | None = None
    try:
        campaign = I2AOTPreparation(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial I2 AOT evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
