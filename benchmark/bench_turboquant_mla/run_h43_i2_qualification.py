#!/usr/bin/env python3
"""Single-shot correctness and resource qualification for H43 I2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import run_h43_maintenance as h43
from analyze_h43_i2_writer_ncu import analyze_paths
from run_h43_i2_aot_preparation import (
    I2_CACHE_PARENT,
    I2_COMMIT,
    I2_NATIVE_DIR,
    I2_NATIVE_SHA256,
    I2_NATIVE_SO,
    I2_SOURCE,
)

I2_AOT_PREPARATION_SHA256 = (
    "dc1dd2b108b47953ceba170298f7cce4f1cbd3c8e05aa7162cc8a2e7d663dfb3"
)
NATIVE_CACHE_FILES = {
    ".ninja_deps": "2b6e7434fd370d2de4940da464106382a92f68c1ecffeb41ee2dc878bb4f944e",
    ".ninja_log": "b2595e97b0b69461403e1fc3622486f7cc066fe5c9d38d14b0fe5b9e2826ffc7",
    "build.ninja": "c0e0a23779f07d6046ad490577ec9e20017e1175530bd5d2c1715e11eec2d76a",
    "tq_mla_frontend_sm100.cuda.o": (
        "fc8586d1730e3e4f1f4dacc7c8b89a89c3f50248072eea16e255332a13089c59"
    ),
    "sglang_tq_mla_frontend_sm100_h43_i2_v1.so": I2_NATIVE_SHA256,
}
PINNED_READER_QUALIFICATION_SHA256 = (
    "1680b09493db8b066f37a1455e4076c47e5132b525bedf83f611c7af796de5e5"
)
PINNED_READER_NCU_SHA256 = (
    "866ee07d3ef6348e7e804fcfe529416027a9c1343ee5ac3492bd291aec0ad411"
)
PINNED_READER_GATES_SHA256 = (
    "67ec28cab9783b5e599ac3fa2ef4b9340e2722dce7c4b1770d3df589e4b3143d"
)
EXPECTED_AOT_LABELS = {
    f"tq-tiles{tiles}-q{q_len}-codebook{codebook}{suffix}"
    for tiles in (32, 50)
    for q_len in (1, 5)
    for codebook, suffix in ((0, ""), (1, ""), (1, "-pdl0"))
} | {"dense-q1", "dense-q5"}
NCU_METRICS = (
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "smsp__inst_executed_op_local_ld.sum",
    "smsp__inst_executed_op_local_st.sum",
)
MEMORY_CONTRACT = {
    "rows": 256_000,
    "selected_layers": 14,
    "selected_global_layers": list(range(24, 38)),
    "dense_fp8_control_bytes_per_rank": 8_994_816_000,
    "integrated_codebook_bytes_per_rank": 8_141_824_000,
    "persistent_saving_bytes_per_rank": 852_992_000,
    "persistent_saving_gib_per_rank": 0.7944107055664062,
    "writer_workspace_bytes_per_rank": 16_793_600,
    "net_saving_bytes_per_rank": 836_198_400,
    "net_saving_gib_per_rank": 0.7787704467773438,
    "tp8_net_saving_gib": 6.23016357421875,
    "retained_fraction_of_no_codebook_gross_saving": 0.937008,
    "meets_old_ten_percent_target_kv_bar": False,
}
QUALIFICATION_FRONTEND_TEST = "test_h41_w2_frontend.py"
QUALIFICATION_PDL_PROBE = "probe_h43_i2_pdl_ordering.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preparation-state", type=Path, required=True)
    parser.add_argument("--i2-preparation-result", type=Path, required=True)
    parser.add_argument("--reader-qualification-tar", type=Path, required=True)
    parser.add_argument("--tooling-commit", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    args = parser.parse_args()
    args.mode = "qualification"
    args.decision_contract = None
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return sha256_bytes(payload)


def last_json_value(stdout: str, label: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} produced no stdout")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[Any] = []
        for index, character in enumerate(stdout):
            if character != "{":
                continue
            try:
                candidate, end = decoder.raw_decode(stdout, index)
            except json.JSONDecodeError:
                continue
            if not stdout[end:].strip():
                candidates.append(candidate)
        if len(candidates) != 1:
            raise RuntimeError(f"{label} does not contain one final JSON value")
        value = candidates[0]
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} final JSON is not an object")
    return value


class I2Qualification(h43.Campaign):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.i2_preparation_path = args.i2_preparation_result.resolve()
        self.reader_qualification_tar = args.reader_qualification_tar.resolve()
        self.tooling_commit = args.tooling_commit
        self.i2_preparation: dict[str, Any] = {}
        self.reader_ncu_proof: dict[str, Any] = {}
        self.source_manifest: list[dict[str, str]] = []
        self.source_manifest_digest = ""
        self.native_build_ninja = ""
        self.qualification_frontend_test_sha256 = ""
        self.qualification_pdl_probe_sha256 = ""
        self.records: dict[str, dict[str, Any]] = {}
        self.reference_container_name = f"ct13-h43-i2-reference-{self.campaign}"

    def maintenance_contract(self) -> tuple[dict[str, int], str]:
        return (
            {
                "experiment": 3000,
                "failsafe": 3060,
                "alert": 3660,
                "terminal": 4200,
            },
            "h43_i2_qualification",
        )

    def timeout_command_contract(self) -> dict[str, dict[str, Any]]:
        return {
            "gpu_identity_sample": {"timeout": 60, "processes": 2},
            "idle_compute_check": {"timeout": 60, "processes": 1},
            "tokenspeed_source_manifest_each": {
                "timeout": 180,
                "processes": 2,
                "phase": "pre-outage",
            },
            "native_prebuilt_load": {
                "timeout": 120,
                "processes": 1,
                "phase": "pre-outage",
            },
            "h40_contract": {
                "timeout": 180,
                "phase": "pre-outage",
                "basis": "pinned 31-test static contract suite",
            },
            "pdl_source_order": {
                "timeout": 120,
                "processes": 1,
                "phase": "pre-outage",
            },
            "pdl_producer_source": {
                "timeout": 120,
                "processes": 1,
                "phase": "pre-outage",
                "gpu_access": False,
            },
            "gpu_stage_surface": {
                "timeout": 180,
                "processes": 1,
                "phase": "pre-outage",
                "gpu_access": False,
            },
            "writer_test_cli": {
                "timeout": 120,
                "processes": 1,
                "phase": "pre-outage",
                "gpu_access": False,
            },
            "writer_correctness": {"timeout": 300, "observed_seconds": 54},
            "lifecycle": {"timeout": 300, "basis": "six focused unit methods"},
            "roundtrip_each": {"timeout": 300, "processes": 5},
            "integrated_smoke_each": {"timeout": 900, "processes": 2},
            "pdl_each": {"timeout": 300, "processes": 8},
            "writer_delta_each": {"timeout": 180, "processes": 3},
            "sanitizer_each": {
                "timeout": 300,
                "processes": 2,
                "observed_seconds": 10,
            },
            "racecheck_each": {"timeout": 300, "processes": 2},
            "ncu_each": {"timeout": 240, "processes": 2},
            "resource_usage": {"timeout": 60, "processes": 1},
        }

    def _validate_tooling_repository(self) -> Path:
        script = Path(__file__).resolve()
        repository = Path(
            h43.run(
                ["git", "-C", str(script.parent), "rev-parse", "--show-toplevel"],
                timeout=30,
            ).stdout.strip()
        )
        head = h43.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], timeout=30
        ).stdout.strip()
        dirty = h43.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            timeout=30,
        ).stdout
        if not re.fullmatch(r"[0-9a-f]{40}", self.tooling_commit):
            raise ValueError("tooling commit must be a full Git object ID")
        if head != self.tooling_commit or dirty:
            raise ValueError(
                f"qualification tooling must be clean at {self.tooling_commit}; "
                f"observed head={head} dirty={bool(dirty)}"
            )
        return repository

    def _validate_i2_preparation(self) -> None:
        if sha256_path(self.i2_preparation_path) != I2_AOT_PREPARATION_SHA256:
            raise ValueError("H43 I2 AOT preparation result digest mismatch")
        value = json.loads(
            self.i2_preparation_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if (
            value.get("status") != "PASS"
            or value.get("experiment") != "H43_I2_AOT_PREPARATION"
            or value.get("sglang_commit") != I2_COMMIT
            or value.get("native_extension_sha256") != I2_NATIVE_SHA256
        ):
            raise ValueError("H43 I2 AOT preparation result is not the sealed PASS")
        for identity in (self.candidate, self.reference):
            role = identity["role"]
            cache = value.get("caches", {}).get(role, {})
            expected_root = (
                f"{I2_CACHE_PARENT}/{role}/{identity['source_manifest_digest']}"
            )
            expected = {
                "root": expected_root,
                "source_manifest_digest": identity["source_manifest_digest"],
                "installed_mla_sha256": identity["installed_mla_sha256"],
                "entry_count": 14,
                "file_count": 29,
                "read_only": True,
            }
            if any(
                cache.get(key) != expected_value
                for key, expected_value in expected.items()
            ):
                raise ValueError(f"sealed {role} I2 cache identity mismatch")
            for field in ("artifact_digest", "manifest_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", cache.get(field, "")):
                    raise ValueError(f"sealed {role} cache has invalid {field}")
        self.i2_preparation = value

    def _validate_reader_ncu_pin(self) -> None:
        if (
            sha256_path(self.reader_qualification_tar)
            != PINNED_READER_QUALIFICATION_SHA256
        ):
            raise ValueError("pinned H43 reader qualification archive changed")
        proof: dict[str, Any] = {
            "archive": str(self.reader_qualification_tar),
            "archive_sha256": PINNED_READER_QUALIFICATION_SHA256,
        }
        with tarfile.open(self.reader_qualification_tar, "r") as archive:
            for member_name, expected_digest, key in (
                (
                    "results/ncu-comparison.json",
                    PINNED_READER_NCU_SHA256,
                    "ncu_comparison",
                ),
                (
                    "results/qualification-gates.json",
                    PINNED_READER_GATES_SHA256,
                    "qualification_gates",
                ),
            ):
                member = archive.getmember(member_name)
                if not member.isfile():
                    raise ValueError(
                        f"pinned reader evidence is not a file: {member_name}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(
                        f"cannot read pinned reader evidence: {member_name}"
                    )
                payload = handle.read()
                if sha256_bytes(payload) != expected_digest:
                    raise ValueError(f"pinned reader evidence changed: {member_name}")
                value = json.loads(payload)
                if value.get("status") != "PASS":
                    raise ValueError(
                        f"pinned reader evidence is not PASS: {member_name}"
                    )
                proof[key] = {
                    "member": member_name,
                    "sha256": expected_digest,
                    "status": "PASS",
                    "result_digest": value.get("result_digest")
                    or value.get("qualification_gates_digest"),
                }
        self.reader_ncu_proof = proof

    def _validate_full_i2_source(self, repository: Path) -> None:
        raw = h43.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                I2_COMMIT,
            ],
            timeout=120,
        ).stdout
        expected: list[dict[str, str]] = []
        for record in raw.split("\0"):
            if not record:
                continue
            identity, path = record.split("\t", 1)
            mode, kind, object_id = identity.split()
            if kind != "blob" or mode not in {"100644", "100755", "120000"}:
                raise ValueError(f"unsupported sealed source entry: {record}")
            expected.append({"mode": mode, "object": object_id, "path": path})
        source_scan = r"""import hashlib,json,os,stat,sys
root=os.path.realpath(sys.argv[1])
rows=[]
for current,dirs,files in os.walk(root,topdown=True,followlinks=False):
    for name in list(dirs):
        path=os.path.join(current,name)
        if os.path.islink(path):
            files.append(name)
            dirs.remove(name)
    for name in files:
        path=os.path.join(current,name)
        relative=os.path.relpath(path,root).replace(os.sep,"/")
        info=os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            data=os.readlink(path).encode()
            mode="120000"
        elif stat.S_ISREG(info.st_mode):
            with open(path,"rb") as handle:
                data=handle.read()
            mode="100755" if info.st_mode & stat.S_IXUSR else "100644"
        else:
            raise RuntimeError(f"unsupported source entry: {relative}")
        header=f"blob {len(data)}\0".encode()
        rows.append({"mode":mode,"object":hashlib.sha1(header+data).hexdigest(),"path":relative})
rows.sort(key=lambda row:row["path"])
print(json.dumps(rows,separators=(",",":"),sort_keys=True))
"""
        observed = json.loads(
            h43.remote(
                self.host0, ["python3", "-c", source_scan, I2_SOURCE], timeout=300
            ).stdout
        )
        expected.sort(key=lambda row: row["path"])
        if observed != expected:
            expected_by_path = {row["path"]: row for row in expected}
            observed_by_path = {row["path"]: row for row in observed}
            changed = sorted(
                path
                for path in set(expected_by_path) | set(observed_by_path)
                if expected_by_path.get(path) != observed_by_path.get(path)
            )
            raise ValueError(f"sealed I2 source tree differs from Git: {changed[:20]}")
        self.source_manifest = expected
        self.source_manifest_digest = canonical_digest(expected)

    def _remote_cache_manifest(self, root: str) -> dict[str, Any]:
        code = (
            "import json,sys;sys.path.insert(0,sys.argv[1]);"
            "from pathlib import Path;"
            "from h43_codebook_ab_common import compiled_artifact_manifest,load_contract;"
            "c=load_contract(Path(sys.argv[1])/'h43_codebook_ab_contract.json');"
            "print(json.dumps(compiled_artifact_manifest(Path(sys.argv[2]),c['cache'])))"
        )
        return json.loads(
            h43.remote(
                self.host0, ["python3", "-c", code, self.work, root], timeout=180
            ).stdout
        )

    def _validate_remote_i2_artifacts(self) -> None:
        for path in (I2_SOURCE, I2_NATIVE_SO):
            h43.remote(self.host0, ["test", "!", "-L", path], timeout=30)
        native_digest = h43.remote(
            self.host0, ["sha256sum", I2_NATIVE_SO], timeout=60
        ).stdout.split()[0]
        if native_digest != I2_NATIVE_SHA256:
            raise ValueError("sealed H43 I2 native extension changed")
        observed_native_files = {
            line.split("  ", 1)[1]: line.split("  ", 1)[0]
            for line in h43.remote(
                self.host0,
                [
                    "bash",
                    "-lc",
                    f"cd {shlex.quote(I2_NATIVE_DIR)} && sha256sum -- * .ninja_deps .ninja_log",
                ],
                timeout=120,
            ).stdout.splitlines()
            if line.strip()
        }
        if observed_native_files != NATIVE_CACHE_FILES:
            raise ValueError("sealed H43 I2 native build-cache inventory changed")
        self.native_build_ninja = h43.remote(
            self.host0, ["cat", f"{I2_NATIVE_DIR}/build.ninja"], timeout=60
        ).stdout
        required_build_fragments = (
            "nvcc = /usr/local/cuda/bin/nvcc",
            "-O3 -lineinfo -gencode=arch=compute_100,code=sm_100",
            "build tq_mla_frontend_sm100.cuda.o: cuda_compile",
            "build sglang_tq_mla_frontend_sm100_h43_i2_v1.so: link",
        )
        if any(
            fragment not in self.native_build_ninja
            for fragment in required_build_fragments
        ):
            raise ValueError("sealed native compiler command is incomplete")
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
            timeout=120,
        ).stdout.strip()
        if writable:
            raise ValueError(
                f"sealed H43 I2 source/native tree is writable: {writable}"
            )
        for identity in (self.candidate, self.reference):
            role = identity["role"]
            expected = self.i2_preparation["caches"][role]
            root = expected["root"]
            writable = h43.remote(
                self.host0,
                ["find", root, "-perm", "/222", "-print", "-quit"],
                timeout=60,
            ).stdout.strip()
            symlink = h43.remote(
                self.host0,
                ["find", root, "-type", "l", "-print", "-quit"],
                timeout=60,
            ).stdout.strip()
            if writable or symlink:
                raise ValueError(
                    f"sealed {role} I2 cache is writable or contains a symlink"
                )
            manifest_path = f"{root}/h43-aot-manifest.json"
            manifest_digest = h43.remote(
                self.host0, ["sha256sum", manifest_path], timeout=60
            ).stdout.split()[0]
            manifest = json.loads(
                h43.remote(self.host0, ["cat", manifest_path], timeout=60).stdout
            )
            labels = {
                entry.get("label") for entry in manifest.get("entries", {}).values()
            }
            artifact = self._remote_cache_manifest(root)
            if (
                manifest_digest != expected["manifest_sha256"]
                or artifact.get("digest") != expected["artifact_digest"]
                or len(artifact.get("files", [])) != 29
                or manifest.get("expected_entries") != 14
                or manifest.get("source_manifest_digest")
                != identity["source_manifest_digest"]
                or manifest.get("installed_mla_sha256")
                != identity["installed_mla_sha256"]
                or labels != EXPECTED_AOT_LABELS
            ):
                raise ValueError(f"sealed {role} I2 cache inventory changed")

    def validate_inputs(self) -> None:
        super().validate_inputs()
        if (
            h43.remote(
                self.host0,
                ["docker", "container", "inspect", self.reference_container_name],
                timeout=30,
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError("reference qualification container already exists")
        repository = self._validate_tooling_repository()
        self._validate_i2_preparation()
        self._validate_reader_ncu_pin()
        self._validate_full_i2_source(repository)
        self._validate_remote_i2_artifacts()

    def setup(self) -> None:
        super().setup()
        qualification_files = (
            (
                QUALIFICATION_FRONTEND_TEST,
                "qualification_frontend_test_sha256",
            ),
            (QUALIFICATION_PDL_PROBE, "qualification_pdl_probe_sha256"),
        )
        for filename, digest_attribute in qualification_files:
            path = Path(__file__).resolve().with_name(filename)
            payload = path.read_bytes()
            setattr(self, digest_attribute, sha256_bytes(payload))
            h43.write_remote_root_file(
                self.host0,
                f"{self.results}/{filename}",
                payload.decode("utf-8"),
                "0444",
            )
        self._run_preoutage_checks()
        timeout_evidence = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_QUALIFICATION_TIMEOUT_BUDGET",
            "maintenance_seconds": self.maintenance_contract()[0],
            "restore_lower_bound_seconds": 1185,
            "commands": self.timeout_command_contract(),
            "note": (
                "Per-command limits are fail-closed; the "
                f"{self.maintenance_contract()[0]['experiment']:,}-second experiment "
                "alarm remains authoritative."
            ),
        }
        preflight = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_QUALIFICATION_PREFLIGHT",
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "sglang_source_manifest_digest": self.source_manifest_digest,
            "sglang_source_entries": len(self.source_manifest),
            "native_extension_sha256": I2_NATIVE_SHA256,
            "qualification_frontend_test_sha256": (
                self.qualification_frontend_test_sha256
            ),
            "qualification_pdl_probe_sha256": self.qualification_pdl_probe_sha256,
            "pdl_conformance_contract": {
                "producer_explicit_trigger": False,
                "reference_overlap_is_opportunistic": True,
                "candidate_mismatched_steps_required": 0,
                "ordered_control_mismatches_required": 0,
            },
            "native_build_cache_files": NATIVE_CACHE_FILES,
            "native_build_ninja": self.native_build_ninja,
            "aot_preparation_sha256": I2_AOT_PREPARATION_SHA256,
            "aot_preparation": self.i2_preparation,
            "pinned_reader_ncu": self.reader_ncu_proof,
            "memory_contract": MEMORY_CONTRACT,
            "preoutage_checks": {
                name: self.records[name]
                for name in (
                    "tokenspeed-candidate-source-manifest",
                    "tokenspeed-reference-source-manifest",
                    "native-prebuilt-load",
                    "h40-contract",
                    "pdl-source-order",
                    "pdl-producer-source",
                    "gpu-stage-surface",
                    "writer-test-cli",
                )
            },
        }
        files = {
            "H43_I2_TIMEOUT_BUDGET.json": timeout_evidence,
            "H43_I2_QUALIFICATION_PREFLIGHT.json": preflight,
            "H43_I2_SGLANG_SOURCE_MANIFEST.json": {
                "commit": I2_COMMIT,
                "digest": self.source_manifest_digest,
                "entries": self.source_manifest,
            },
        }
        for filename, value in files.items():
            payload = (
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
            (self.local_output / filename).write_text(payload, encoding="utf-8")
            h43.write_remote_root_file(
                self.host0, f"{self.results}/{filename}", payload, "0644"
            )

    def _container_environment(self, identity: dict[str, Any]) -> list[str]:
        cache = self.i2_preparation["caches"][identity["role"]]
        return [
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONPATH=/i2/python:/i2:/work",
            "--env",
            f"CUTE_DSL_CACHE_DIR={cache['root']}",
            "--env",
            f"H43_AOT_MANIFEST={cache['root']}/h43-aot-manifest.json",
            "--env",
            "H43_AOT_EXPECTED_ENTRIES=14",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={identity['source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={identity['installed_mla_sha256']}",
            "--env",
            "SGLANG_TQ_MLA_FRONTEND_SO=/native/sglang_tq_mla_frontend_sm100_h43_i2_v1.so",
            "--env",
            f"SGLANG_TQ_MLA_FRONTEND_SO_SHA256={I2_NATIVE_SHA256}",
        ]

    def _container_volumes(self, identity: dict[str, Any]) -> list[str]:
        cache = self.i2_preparation["caches"][identity["role"]]
        work = self.work if identity["role"] == "candidate" else self.reference_work
        tokenspeed_source = f"{self.prep_root}/{identity['role']}/source"
        return [
            "--volume",
            f"{work}:/work:ro",
            "--volume",
            f"{tokenspeed_source}:/tokenspeed-source:ro",
            "--volume",
            f"{I2_SOURCE}:/i2:ro",
            "--volume",
            f"{I2_NATIVE_DIR}:/native:ro",
            "--volume",
            f"{cache['root']}:{cache['root']}:ro",
            "--volume",
            f"{self.results}:/results",
        ]

    def _preoutage_run(
        self,
        identity: dict[str, Any],
        name: str,
        arguments: list[str],
        *,
        timeout: int,
        expected_json_status: str | None = None,
    ) -> dict[str, Any] | None:
        suffix = sha256_bytes(f"{self.campaign}:{identity['role']}:{name}".encode())[
            :12
        ]
        container_name = f"ct13-h43-i2pf-{identity['role']}-{suffix}"
        if (
            h43.remote(
                self.host0,
                ["docker", "container", "inspect", container_name],
                timeout=30,
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError(f"pre-outage container already exists: {container_name}")
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            *self._container_environment(identity),
            *self._container_volumes(identity),
            identity["image_id"],
            *arguments,
        ]
        try:
            result = h43.remote(self.host0, command, timeout=timeout + 30, check=False)
        finally:
            h43.remote(
                self.host0,
                ["docker", "rm", "--force", container_name],
                timeout=60,
                check=False,
            )
        try:
            return self._record_result(
                name,
                result,
                command=command,
                expected_json_status=expected_json_status,
            )
        finally:
            if name in self.records:
                self.records[name].update(
                    {"phase": "pre-outage", "gpu_access": False, "network": "none"}
                )

    def _run_preoutage_checks(self) -> None:
        source_check = [
            "bash",
            "-lc",
            "sha256sum --check --strict /work/H43_D1_SOURCE_MANIFEST.sha256 "
            "&& cat /work/H43_D1_SOURCE_MANIFEST.sha256",
        ]
        self._preoutage_run(
            self.candidate,
            "tokenspeed-candidate-source-manifest",
            source_check,
            timeout=180,
        )
        self._preoutage_run(
            self.reference,
            "tokenspeed-reference-source-manifest",
            source_check,
            timeout=180,
        )
        prebuilt_probe = [
            "python3",
            "-c",
            (
                "import hashlib,json;"
                "import sglang.jit_kernel.tq_mla_frontend as frontend;"
                "path=frontend._get_module().__file__;"
                "digest=hashlib.sha256(open(path,'rb').read()).hexdigest();"
                f"expected='{I2_NATIVE_SHA256}';"
                "print(json.dumps({'status':'PASS' if digest==expected else 'FAIL',"
                "'module_path':path,'module_sha256':digest}))"
            ),
        ]
        value = self._preoutage_run(
            self.candidate,
            "native-prebuilt-load",
            prebuilt_probe,
            timeout=120,
            expected_json_status="PASS",
        )
        if (
            value is None
            or value.get("module_path")
            != "/native/sglang_tq_mla_frontend_sm100_h43_i2_v1.so"
            or value.get("module_sha256") != I2_NATIVE_SHA256
        ):
            raise RuntimeError(
                "native prebuilt loader did not resolve the sealed module"
            )
        self._preoutage_run(
            self.candidate,
            "h40-contract",
            [
                "python3",
                "-m",
                "pytest",
                "/tokenspeed-source/tokenspeed-mla/test/test_tq4_contract.py",
                "-q",
            ],
            timeout=180,
        )
        contract_stdout = h43.remote(
            self.host0, ["cat", f"{self.results}/h40-contract.stdout.log"], timeout=30
        ).stdout
        if not re.search(r"\b31 passed\b", contract_stdout):
            raise RuntimeError("H40 contract did not report exactly 31 passing tests")
        pdl_source = self._preoutage_run(
            self.candidate,
            "pdl-source-order",
            ["python3", "/work/check_h43_pdl_source.py"],
            timeout=120,
            expected_json_status="PASS",
        )
        if pdl_source is None:
            raise RuntimeError("PDL source-order gate produced no result")
        producer_source = self._preoutage_run(
            self.candidate,
            "pdl-producer-source",
            [
                "python3",
                "-c",
                (
                    "import json,pathlib;"
                    "p=pathlib.Path('/i2/python/sglang/jit_kernel/csrc/"
                    "tq_mla_frontend/tq_mla_frontend_sm100.cu');"
                    "s=p.read_text();"
                    "assert 'cudaTriggerProgrammaticLaunchCompletion' not in s;"
                    "assert '<<<grid, block, 0, stream>>>' in s;"
                    "print(json.dumps({'status':'PASS',"
                    "'explicit_trigger':False,'launch':'standard-stream'}))"
                ),
            ],
            timeout=120,
            expected_json_status="PASS",
        )
        if (
            producer_source is None
            or producer_source.get("explicit_trigger") is not False
            or producer_source.get("launch") != "standard-stream"
        ):
            raise RuntimeError("PDL producer source contract is invalid")
        surface_script = r"""set -euo pipefail
help=$(compute-sanitizer --help)
for option in --tool --error-exitcode --target-processes --report-api-errors --kernel-name --log-file; do grep -Fq -- "$option" <<<"$help"; done
help=$(ncu --help)
for option in --nvtx --metrics --csv --page --print-units --force-overwrite --nvtx-include --log-file --export; do grep -Fq -- "$option" <<<"$help"; done
cuobjdump --help | grep -Fq -- --dump-resource-usage
test -f /i2/benchmark/bench_turboquant_mla/test_h41_w2_frontend.py
help=$(python3 /i2/benchmark/bench_turboquant_mla/test_h41_w2_frontend.py --help)
grep -Fq -- '--mode' <<<"$help"
test -f /i2/benchmark/bench_turboquant_mla/test_h41_i1_roundtrip.py
help=$(python3 /i2/benchmark/bench_turboquant_mla/test_h41_i1_roundtrip.py --help)
for option in --context --q-len --split-kv; do grep -Fq -- "$option" <<<"$help"; done
test -f /results/probe_h43_i2_pdl_ordering.py
help=$(PYTHONPATH=/i2/python:/i2:/work:/results python3 /results/probe_h43_i2_pdl_ordering.py --help)
for option in --context --q-len --split-kv --steps --reader; do grep -Fq -- "$option" <<<"$help"; done
test -f /i2/benchmark/bench_turboquant_mla/bench_h43_i2_writer_delta.py
help=$(python3 /i2/benchmark/bench_turboquant_mla/bench_h43_i2_writer_delta.py --help)
for option in --tokens --pairs --replays --profile-arm; do grep -Fq -- "$option" <<<"$help"; done
test -f /i2/benchmark/bench_turboquant_mla/bench_h41_i1_integrated.py
help=$(python3 /i2/benchmark/bench_turboquant_mla/bench_h41_i1_integrated.py --help)
for option in --context --split-kv --allocation-order --sequence --warmups --samples --replays-per-sample; do grep -Fq -- "$option" <<<"$help"; done
test -f /work/probe_h43_tq_racecheck.py
help=$(python3 /work/probe_h43_tq_racecheck.py --help)
for option in --contract --context --sequence; do grep -Fq -- "$option" <<<"$help"; done
test -f /work/check_h43_racecheck.py
help=$(python3 /work/check_h43_racecheck.py --help)
for option in --contract --context --reference-log --candidate-log --reference-target --candidate-target --reference-exit-code --candidate-exit-code; do grep -Fq -- "$option" <<<"$help"; done
PYTHONPATH=/i2/python:/i2/test:/i2:/work python3 -c "from srt.test_turboquant import TestTurboQuantGPU as T; from registered.unit.mem_cache.test_mem_pool_host import TestMLATurboQuantHostKVCache as H; names=('test_mla_fused_kv_write_matches_legacy_quantize_store','test_mla_e2m1_fused_and_fallback_writers_match','test_mla_e2m1_chunked_fused_writer_matches_full_workspace','test_mla_layerwise_pool_has_one_representation_per_layer','test_mla_layerwise_codebook_allocates_only_selected_slots'); assert all(hasattr(T,n) for n in names); assert hasattr(H,'test_codebook_allocation_and_transfer_contract')"
printf '%s\n' '{"binaries":3,"lifecycle_methods":6,"scripts":7,"status":"PASS"}'
"""
        self._preoutage_run(
            self.candidate,
            "gpu-stage-surface",
            ["bash", "-lc", surface_script],
            timeout=180,
            expected_json_status="PASS",
        )
        self._preoutage_run(
            self.candidate,
            "writer-test-cli",
            [
                "env",
                "PYTHONPATH=/i2/python:/i2:/work:/results",
                "python3",
                f"/results/{QUALIFICATION_FRONTEND_TEST}",
                "--help",
            ],
            timeout=120,
        )
        writer_test_help = self._read_remote_text("writer-test-cli.stdout.log")
        if any(option not in writer_test_help for option in ("--mode", "--seed")):
            raise RuntimeError("qualification writer test CLI is incomplete")

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
            raise RuntimeError("candidate qualification container already exists")
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
            *self._container_environment(self.candidate),
            *self._container_volumes(self.candidate),
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

    def _reference_run(
        self, arguments: list[str], *, timeout: int, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
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
            *self._container_environment(self.reference),
            *self._container_volumes(self.reference),
            self.reference["image_id"],
            *arguments,
        ]
        try:
            return h43.remote(self.host0, command, timeout=timeout + 30, check=check)
        finally:
            h43.remote(
                self.host0,
                ["docker", "rm", "--force", self.reference_container_name],
                timeout=60,
                check=False,
            )

    def _record_result(
        self,
        name: str,
        result: subprocess.CompletedProcess[str],
        *,
        command: list[str],
        expected_json_status: str | None = None,
    ) -> dict[str, Any] | None:
        stdout_name = f"{name}.stdout.log"
        stderr_name = f"{name}.stderr.log"
        h43.write_remote_root_file(
            self.host0, f"{self.results}/{stdout_name}", result.stdout, "0644"
        )
        h43.write_remote_root_file(
            self.host0, f"{self.results}/{stderr_name}", result.stderr, "0644"
        )
        record: dict[str, Any] = {
            "returncode": result.returncode,
            "command": shlex.join(command),
            "stdout": stdout_name,
            "stderr": stderr_name,
        }
        value: dict[str, Any] | None = None
        parse_error: BaseException | None = None
        if expected_json_status is not None:
            try:
                value = last_json_value(result.stdout, name)
                record["json_status"] = value.get("status")
                payload = (
                    json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
                )
                h43.write_remote_root_file(
                    self.host0, f"{self.results}/{name}.json", payload, "0644"
                )
            except BaseException as error:
                parse_error = error
        self.records[name] = record
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")
        if parse_error is not None:
            raise parse_error
        if expected_json_status is not None and value is not None:
            if value.get("status") != expected_json_status:
                raise RuntimeError(
                    f"{name} status {value.get('status')!r} != {expected_json_status!r}"
                )
        return value

    def _candidate(
        self,
        name: str,
        command: list[str],
        *,
        timeout: int,
        expected_json_status: str | None = None,
    ) -> dict[str, Any] | None:
        wrapped = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
            *command,
        ]
        result = self.exec_candidate(wrapped, timeout=timeout + 30, check=False)
        return self._record_result(
            name,
            result,
            command=wrapped,
            expected_json_status=expected_json_status,
        )

    def _reference(
        self,
        name: str,
        command: list[str],
        *,
        timeout: int,
        expected_json_status: str | None = None,
    ) -> dict[str, Any] | None:
        result = self._reference_run(command, timeout=timeout, check=False)
        return self._record_result(
            name,
            result,
            command=command,
            expected_json_status=expected_json_status,
        )

    def _sample_idle_gpu(self, name: str) -> None:
        query = ",".join(self.contract["telemetry"]["query_fields"])
        result = h43.remote(
            self.host0,
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            timeout=60,
        )
        row = dict(
            zip(
                self.contract["telemetry"]["query_fields"],
                [part.strip() for part in result.stdout.strip().split(",")],
                strict=True,
            )
        )
        mismatches = h43.gpu_health_mismatches(row, self.contract, self.aggregate_ecc)
        current_sm_clock = mismatches.pop("clocks.sm", None)
        if current_sm_clock is not None:
            try:
                observed_clock = float(row["clocks.sm"])
                maximum_clock = float(row["clocks.max.sm"])
            except (KeyError, ValueError):
                mismatches["clocks.sm"] = current_sm_clock
            else:
                if not 0 < observed_clock <= maximum_clock:
                    mismatches["clocks.sm"] = current_sm_clock
        h43.write_remote_root_file(
            self.host0, f"{self.results}/{name}.csv", result.stdout, "0644"
        )
        if mismatches:
            raise RuntimeError(f"{name} GPU health mismatch: {mismatches}")

    def _assert_no_compute_process(self, name: str) -> None:
        result = h43.remote(
            self.host0,
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=60,
        )
        h43.write_remote_root_file(
            self.host0, f"{self.results}/{name}.txt", result.stdout, "0644"
        )
        if result.stdout.strip():
            raise RuntimeError(f"{name} found an unexpected GPU compute process")

    def _run_writer_correctness_and_lifecycle(self) -> None:
        correctness = self._candidate(
            "writer-correctness",
            [
                "python3",
                f"/results/{QUALIFICATION_FRONTEND_TEST}",
                "--mode",
                "correctness",
            ],
            timeout=300,
            expected_json_status="PASS",
        )
        if correctness is None or correctness.get("case_count") != 352:
            raise RuntimeError("writer correctness did not cover all 352 cases")

        tests = [
            "srt.test_turboquant.TestTurboQuantGPU.test_mla_fused_kv_write_matches_legacy_quantize_store",
            "srt.test_turboquant.TestTurboQuantGPU.test_mla_e2m1_fused_and_fallback_writers_match",
            "srt.test_turboquant.TestTurboQuantGPU.test_mla_e2m1_chunked_fused_writer_matches_full_workspace",
            "srt.test_turboquant.TestTurboQuantGPU.test_mla_layerwise_pool_has_one_representation_per_layer",
            "srt.test_turboquant.TestTurboQuantGPU.test_mla_layerwise_codebook_allocates_only_selected_slots",
            "registered.unit.mem_cache.test_mem_pool_host.TestMLATurboQuantHostKVCache.test_codebook_allocation_and_transfer_contract",
        ]
        lifecycle_command = [
            "env",
            "PYTHONPATH=/i2/python:/i2/test:/i2:/work",
            "python3",
            "-m",
            "unittest",
            "-v",
            *tests,
        ]
        self._candidate("lifecycle", lifecycle_command, timeout=300)
        lifecycle = h43.remote(
            self.host0,
            [
                "bash",
                "-lc",
                f"cat {shlex.quote(self.results)}/lifecycle.stdout.log "
                f"{shlex.quote(self.results)}/lifecycle.stderr.log",
            ],
            timeout=30,
        ).stdout
        if "Ran 6 tests" not in lifecycle or not re.search(r"(?m)^OK$", lifecycle):
            raise RuntimeError("lifecycle suite did not report six passing tests")

    def _run_roundtrip_and_pdl(self) -> None:
        for context, split in ((10219, 64), (37932, 40)):
            for q_len in (1, 5):
                command = [
                    "python3",
                    "/i2/benchmark/bench_turboquant_mla/test_h41_i1_roundtrip.py",
                    "--context",
                    str(context),
                    "--q-len",
                    str(q_len),
                    "--split-kv",
                    str(split),
                ]
                value = self._candidate(
                    f"roundtrip-c{context}-q{q_len}",
                    command,
                    timeout=300,
                    expected_json_status="PASS",
                )
                if value is None or value.get("graph_replay_allocation_bytes", {}).get(
                    "before"
                ) != value.get("graph_replay_allocation_bytes", {}).get("after"):
                    raise RuntimeError("roundtrip graph replay allocation changed")

        for context, split in ((10219, 64), (37932, 40)):
            for q_len in (1, 5):
                common = [
                    "python3",
                    f"/results/{QUALIFICATION_PDL_PROBE}",
                    "--context",
                    str(context),
                    "--q-len",
                    str(q_len),
                    "--split-kv",
                    str(split),
                    "--steps",
                    "1000",
                ]
                reference = self._reference(
                    f"pdl-pre-move-c{context}-q{q_len}",
                    [*common, "--reader", "pre-move"],
                    timeout=300,
                    expected_json_status="PASS",
                )
                candidate = self._candidate(
                    f"pdl-post-wait-c{context}-q{q_len}",
                    [*common, "--reader", "post-wait"],
                    timeout=300,
                    expected_json_status="PASS",
                )
                # PDL overlap is opportunistic, and this production writer has
                # no explicit early trigger.  Preserve the pre-move result as
                # exploratory evidence without requiring an observable race.
                if (
                    reference is None
                    or reference.get("ordered_control_mismatches") != 0
                    or reference.get("sensitivity_required") is not False
                    or reference.get("producer_explicit_trigger") is not False
                    or reference.get("ordering_gate") is not False
                ):
                    raise RuntimeError("pre-move PDL conformance run is invalid")
                if (
                    candidate is None
                    or candidate.get("mismatched_steps") != 0
                    or candidate.get("ordered_control_mismatches") != 0
                    or candidate.get("sensitivity_required") is not False
                    or candidate.get("producer_explicit_trigger") is not False
                    or candidate.get("ordering_gate") is not True
                ):
                    raise RuntimeError("post-wait PDL run did not establish ordering")

    def _run_writer_delta(self) -> None:
        for tokens in (1, 5, 40):
            value = self._candidate(
                f"writer-delta-q{tokens}",
                [
                    "python3",
                    "/i2/benchmark/bench_turboquant_mla/bench_h43_i2_writer_delta.py",
                    "--tokens",
                    str(tokens),
                    "--pairs",
                    "20",
                    "--replays",
                    "100",
                ],
                timeout=180,
                expected_json_status="TIMING_ONLY",
            )
            if value is None or value.get("correctness") != "BYTE_EXACT":
                raise RuntimeError("writer delta lost its byte-exact prerequisite")

    def _run_integrated_smoke(self) -> None:
        for context, split in ((10219, 64), (37932, 40)):
            value = self._candidate(
                f"integrated-smoke-c{context}",
                [
                    "python3",
                    "/i2/benchmark/bench_turboquant_mla/bench_h41_i1_integrated.py",
                    "--context",
                    str(context),
                    "--split-kv",
                    str(split),
                    "--allocation-order",
                    "control-first",
                    "--sequence",
                    "1",
                    "--warmups",
                    "100",
                    "--samples",
                    "20",
                    "--replays-per-sample",
                    "100",
                ],
                timeout=900,
                expected_json_status="TIMING_ONLY",
            )
            if value is None:
                raise RuntimeError("integrated smoke produced no result")
            expected = {
                "context": context,
                "split_kv": split,
                "total_layers": 61,
                "selected_layers": 14,
                "selected_layer_ids": list(range(24, 38)),
                "selected_row_bytes": 338,
                "selected_codebook_materialized": True,
                "control_persistent_cache_bytes": 8_994_816_000,
                "candidate_persistent_cache_bytes": 8_141_824_000,
                "gross_persistent_cache_savings_bytes": 852_992_000,
                "sticky_status": 0,
            }
            mismatches = {
                key: (value.get(key), expected_value)
                for key, expected_value in expected.items()
                if value.get(key) != expected_value
            }
            allocation = value.get("graph_replay_allocation_bytes", {})
            if (
                mismatches
                or allocation.get("before") != allocation.get("after")
                or value.get("one_selected_layer_max_abs_diff", 1.0)
                > value.get("correctness_atol", 0.0)
                or value.get("writer_reader_max_abs_diff", 1.0)
                > value.get("correctness_atol", 0.0)
            ):
                raise RuntimeError(
                    f"integrated smoke structural/correctness gate failed: {mismatches}"
                )

    def _run_writer_sanitizers(self) -> None:
        for tool in ("memcheck", "initcheck"):
            log = f"writer-sanitizer-{tool}.log"
            command = [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                str(self.contract["sanitizers"]["error_exit_code"]),
                "--target-processes",
                "all",
                "--report-api-errors",
                "no",
                "--kernel-name",
                "kns=tq_mla_frontend_kernel",
                "--log-file",
                f"/results/{log}",
                "python3",
                f"/results/{QUALIFICATION_FRONTEND_TEST}",
                "--mode",
                "sanitizer",
            ]
            value = self._candidate(
                f"writer-sanitizer-{tool}-target",
                command,
                timeout=300,
                expected_json_status="PASS",
            )
            sanitizer_log = h43.remote(
                self.host0, ["cat", f"{self.results}/{log}"], timeout=60
            ).stdout
            if value is None or not re.search(
                r"ERROR SUMMARY:\s+0 errors", sanitizer_log
            ):
                raise RuntimeError(f"writer {tool} did not report zero errors")

    def _run_reader_racecheck(self) -> None:
        timeout = 300
        common = [
            "compute-sanitizer",
            "--tool",
            "racecheck",
            "--error-exitcode",
            str(self.contract["sanitizers"]["error_exit_code"]),
            "--target-processes",
            "all",
            "--report-api-errors",
            "no",
        ]
        probe = [
            "python3",
            "/work/probe_h43_tq_racecheck.py",
            "--contract",
            "/work/h43_codebook_ab_contract.json",
            "--context",
            "37932",
            "--sequence",
            "1",
        ]
        reference_log = "/results/sanitizer-racecheck-reference.log"
        candidate_log = "/results/sanitizer-racecheck-candidate.log"
        reference_command = [*common, "--log-file", reference_log, *probe]
        candidate_command = [*common, "--log-file", candidate_log, *probe]
        reference = self._reference_run(reference_command, timeout=timeout, check=False)
        self._record_result(
            "racecheck-reference-target",
            reference,
            command=reference_command,
        )
        candidate = self.exec_candidate(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=15s",
                f"{timeout}s",
                *candidate_command,
            ],
            timeout=timeout + 30,
            check=False,
        )
        self._record_result(
            "racecheck-candidate-target", candidate, command=candidate_command
        )
        compare = [
            "python3",
            "/work/check_h43_racecheck.py",
            "--contract",
            "/work/h43_codebook_ab_contract.json",
            "--context",
            "37932",
            "--reference-log",
            reference_log,
            "--candidate-log",
            candidate_log,
            "--reference-target",
            "/results/racecheck-reference-target.stdout.log",
            "--candidate-target",
            "/results/racecheck-candidate-target.stdout.log",
            "--reference-exit-code",
            str(reference.returncode),
            "--candidate-exit-code",
            str(candidate.returncode),
        ]
        self._candidate(
            "racecheck-comparison",
            compare,
            timeout=120,
            expected_json_status="PASS",
        )

    def _read_remote_text(self, name: str) -> str:
        return h43.remote(
            self.host0, ["cat", f"{self.results}/{name}"], timeout=120
        ).stdout

    def _run_writer_ncu(self) -> None:
        common = [
            "ncu",
            "--nvtx",
            "--metrics",
            ",".join(NCU_METRICS),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
            "--force-overwrite",
        ]
        for arm, nvtx in (
            ("no-codebook", "H43_I2_WRITER_NO-CODEBOOK/"),
            ("codebook", "H43_I2_WRITER_CODEBOOK/"),
        ):
            command = [
                *common,
                "--nvtx-include",
                nvtx,
                "--log-file",
                f"/results/writer-ncu-{arm}.csv",
                "--export",
                f"/results/writer-ncu-{arm}",
                "python3",
                "/i2/benchmark/bench_turboquant_mla/bench_h43_i2_writer_delta.py",
                "--tokens",
                "5",
                "--profile-arm",
                arm,
            ]
            self._candidate(
                f"writer-ncu-{arm}-target",
                command,
                timeout=240,
                expected_json_status="PASS",
            )
        resource = self.exec_candidate(
            [
                "cuobjdump",
                "--dump-resource-usage",
                "/native/sglang_tq_mla_frontend_sm100_h43_i2_v1.so",
            ],
            timeout=60,
            check=False,
        )
        self._record_result(
            "writer-resource-usage", resource, command=list(resource.args)
        )
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/writer-resource-usage.txt",
            resource.stdout + resource.stderr,
            "0644",
        )

        local_no_codebook = self.local_output / "writer-ncu-no-codebook.csv"
        local_codebook = self.local_output / "writer-ncu-codebook.csv"
        local_resource = self.local_output / "writer-resource-usage.txt"
        local_no_codebook.write_text(
            self._read_remote_text("writer-ncu-no-codebook.csv"), encoding="utf-8"
        )
        local_codebook.write_text(
            self._read_remote_text("writer-ncu-codebook.csv"), encoding="utf-8"
        )
        local_resource.write_text(
            self._read_remote_text("writer-resource-usage.txt"), encoding="utf-8"
        )
        analysis = analyze_paths(local_no_codebook, local_codebook, local_resource)
        if analysis["status"] != "PASS":
            raise RuntimeError(f"writer NCU resource gate failed: {analysis['gates']}")
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/writer-ncu-analysis.json",
            json.dumps(analysis, allow_nan=False, indent=2, sort_keys=True) + "\n",
            "0644",
        )
        self.records["writer-ncu-analysis"] = {
            "status": analysis["status"],
            "gates": analysis["gates"],
        }

    def _cache_manifest_in_candidate(self) -> dict[str, Any]:
        cache_root = self.i2_preparation["caches"]["candidate"]["root"]
        code = (
            "import json,sys;sys.path.insert(0,'/work');"
            "from pathlib import Path;"
            "from h43_codebook_ab_common import compiled_artifact_manifest,load_contract;"
            "c=load_contract(Path('/work/h43_codebook_ab_contract.json'));"
            f"print(json.dumps(compiled_artifact_manifest(Path('{cache_root}'),c['cache'])))"
        )
        return json.loads(
            self.exec_candidate(["python3", "-c", code], timeout=120).stdout
        )

    def _cache_persistence(self) -> None:
        if self.container_id is None:
            raise RuntimeError("candidate container is absent")
        before = self._cache_manifest_in_candidate()
        native_before = self.exec_candidate(
            ["sha256sum", "/native/sglang_tq_mla_frontend_sm100_h43_i2_v1.so"],
            timeout=60,
        ).stdout.split()[0]
        observed = h43.remote(
            self.host0,
            ["docker", "inspect", "--format", "{{.Id}}", self.container_id],
            timeout=60,
        ).stdout.strip()
        h43.remote(
            self.host0,
            ["docker", "stop", "--time", "10", self.container_id],
            timeout=60,
        )
        restarted = h43.remote(
            self.host0, ["docker", "start", self.container_id], timeout=60
        ).stdout.strip()
        if observed != self.container_id or restarted != self.container_id:
            raise RuntimeError("cache persistence restarted a different container")
        smoke = self._candidate(
            "cache-persistence-roundtrip",
            [
                "python3",
                "/i2/benchmark/bench_turboquant_mla/test_h41_i1_roundtrip.py",
                "--context",
                "10219",
                "--q-len",
                "1",
                "--split-kv",
                "64",
            ],
            timeout=300,
            expected_json_status="PASS",
        )
        after = self._cache_manifest_in_candidate()
        native_after = self.exec_candidate(
            ["sha256sum", "/native/sglang_tq_mla_frontend_sm100_h43_i2_v1.so"],
            timeout=60,
        ).stdout.split()[0]
        expected = self.i2_preparation["caches"]["candidate"]["artifact_digest"]
        if (
            smoke is None
            or before.get("digest") != expected
            or after.get("digest") != expected
            or native_before != I2_NATIVE_SHA256
            or native_after != I2_NATIVE_SHA256
        ):
            raise RuntimeError("sealed cache/native artifact changed across restart")
        evidence = {
            "status": "PASS",
            "container_id": self.container_id,
            "cache_before": before,
            "cache_after": after,
            "native_before_sha256": native_before,
            "native_after_sha256": native_after,
            "restart_smoke": smoke,
        }
        h43.write_remote_root_file(
            self.host0,
            f"{self.results}/cache-persistence.json",
            json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
            "0644",
        )
        self.records["cache-persistence"] = {"status": "PASS"}

    def qualification(self) -> None:
        self._assert_no_compute_process("idle-after-candidate-start")
        self._sample_idle_gpu("qualification-gpu-start")
        self._run_writer_correctness_and_lifecycle()
        self._run_roundtrip_and_pdl()
        self._run_writer_delta()
        self._run_integrated_smoke()
        self._run_writer_sanitizers()
        self._run_reader_racecheck()
        self._run_writer_ncu()
        self._cache_persistence()
        self._assert_no_compute_process("idle-before-qualification-end")
        self._sample_idle_gpu("qualification-gpu-end")

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
        writer_ncu = json.loads(self._read_remote_text("writer-ncu-analysis.json"))
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
        rows.append({"path":relative,"sha256":digest.hexdigest(),"bytes":os.path.getsize(path)})
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
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "experiment": "H43_I2_CODEBOOK_INTEGRATION_QUALIFICATION",
            "campaign": self.campaign,
            "machine_scope": ["ct13", "ct14"],
            "gpu_under_test": "ct13:GPU0",
            "tooling_commit": self.tooling_commit,
            "sglang_source_commit": I2_COMMIT,
            "sglang_source_manifest_digest": self.source_manifest_digest,
            "tokenspeed_candidate_commit": self.candidate["commit"],
            "tokenspeed_reference_commit": self.reference["commit"],
            "tokenspeed": {
                identity["role"]: {
                    "commit": identity["commit"],
                    "source_manifest_digest": identity["source_manifest_digest"],
                    "installed_mla_sha256": identity["installed_mla_sha256"],
                    "base_cache_artifact_digest": identity["cache_artifact_digest"],
                }
                for identity in (self.candidate, self.reference)
            },
            "candidate_image": self.candidate["image_id"],
            "reference_image": self.reference["image_id"],
            "native_extension_sha256": I2_NATIVE_SHA256,
            "qualification_frontend_test_sha256": (
                self.qualification_frontend_test_sha256
            ),
            "qualification_pdl_probe_sha256": self.qualification_pdl_probe_sha256,
            "native_build_cache_files": NATIVE_CACHE_FILES,
            "native_build_ninja": self.native_build_ninja,
            "aot_preparation": self.i2_preparation,
            "pinned_reader_ncu": self.reader_ncu_proof,
            "writer_ncu": writer_ncu,
            "memory_contract": MEMORY_CONTRACT,
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
            "evidence_inventory_before_summary": inventory,
            "evidence_inventory_digest": canonical_digest(inventory),
            "decision_authorization": "QUALIFICATION_PASS_ONLY; PERFORMANCE_DECISION_REMAINS_SEPARATE",
        }
        payload = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        h43.write_remote_root_file(
            self.host0, f"{self.results}/H43_I2_QUALIFICATION.json", payload, "0644"
        )
        (self.local_output / "H43_I2_QUALIFICATION.json").write_text(
            payload, encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    campaign: I2Qualification | None = None
    try:
        campaign = I2Qualification(args)
        campaign.execute()
    except BaseException:
        if campaign is not None and campaign.local_output.exists():
            try:
                campaign.collect()
            except BaseException as collect_error:
                print(
                    f"failed to collect partial H43 I2 qualification evidence: {collect_error}",
                    file=sys.stderr,
                )
        raise


if __name__ == "__main__":
    main()
