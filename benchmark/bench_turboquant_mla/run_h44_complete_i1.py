#!/usr/bin/env python3
"""Run the sealed H44 complete-I1 component gate using isolated CT13 GPU0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any


H44_ROOT = Path("/var/lib/h44/kimi-k26-tq-recovery-20260731")
CAMPAIGN_EVIDENCE_ROOT = H44_ROOT / "campaign" / "evidence"
SEALED_GATE_ROOT = H44_ROOT / "sealed-inputs" / "gates"
SUPPORTED_PHASES = {
    "phase0_complete_i1",
    "lloyd_complete_i1",
    "combined_complete_i1",
}
HOSTS = {"beacon-cr-ct13": 0, "beacon-cr-ct14": 1}
CONTEXTS = {
    10219: {"split_kv": 64, "allowance_us": 27.0292},
    37932: {"split_kv": 40, "allowance_us": 27.6973},
}
SEQUENCES = tuple(range(1, 11))
WARMUPS = 100
PAIRS = 20
REPLAYS = 100
BOOTSTRAP_DRAWS = 50_000
SEED = 20260729
SENTINEL_MAX_DRIFT_FRACTION = 0.005
MAXIMUM_REPLACEMENTS_TOTAL = 1
MAXIMUM_REPLACEMENTS_PER_SEQUENCE = 1
SEQUENCE_PHASE_SECONDS = 600
EXECUTABLE_PHASE_SECONDS = 1500
POSTFLIGHT_RESERVE_SECONDS = 60
SEQUENCE_TIMEOUT_SECONDS = 120
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SAFE_PHASE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
SEALED_OWNER_UID = 0
SOURCE_COMMIT_OBJECT_PATH = "provenance/source-commit-object"


class I1Error(RuntimeError):
    pass


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
    return hashlib.sha256(payload).hexdigest()


def _git_object_oid(kind: str, payload: bytes) -> bytes:
    header = f"{kind} {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).digest()


def git_tree_oid(root: Path) -> str:
    """Recompute a Git SHA-1 tree ID from an exported, sealed worktree."""

    root = root.resolve(strict=True)

    def directory_oid(directory: Path) -> bytes:
        entries: list[tuple[bytes, bytes]] = []
        for path in directory.iterdir():
            name = os.fsencode(path.name)
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode) and not path.is_symlink():
                mode = b"40000"
                oid = directory_oid(path)
                sort_key = name + b"/"
            elif stat.S_ISLNK(metadata.st_mode):
                mode = b"120000"
                oid = _git_object_oid("blob", os.fsencode(os.readlink(path)))
                sort_key = name
            elif stat.S_ISREG(metadata.st_mode):
                mode = b"100755" if metadata.st_mode & 0o111 else b"100644"
                oid = _git_object_oid("blob", path.read_bytes())
                sort_key = name
            else:
                raise I1Error(f"source tree contains a special file: {path}")
            entries.append((sort_key, mode + b" " + name + b"\0" + oid))
        payload = b"".join(entry for _, entry in sorted(entries))
        return _git_object_oid("tree", payload)

    return directory_oid(root).hex()


def verify_source_identity(
    source_root: Path,
    commit_object_path: Path,
    *,
    source_commit: str,
    source_tree: str,
) -> None:
    payload = commit_object_path.read_bytes()
    if _git_object_oid("commit", payload).hex() != source_commit:
        raise I1Error("complete-I1 source commit object hash differs")
    first_line = payload.splitlines()[0] if payload else b""
    if first_line != f"tree {source_tree}".encode():
        raise I1Error("complete-I1 source commit does not reference the charged tree")
    if git_tree_oid(source_root) != source_tree:
        raise I1Error("complete-I1 source tree differs from the charged Git tree")


def run(
    command: list[str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise I1Error(
            f"command failed to execute: {shlex.join(command)}: {exc}"
        ) from exc
    if check and completed.returncode:
        raise I1Error(
            f"command failed rc={completed.returncode}: {shlex.join(command)}: "
            f"{completed.stderr.strip()}"
        )
    return completed


def docker(
    *arguments: str, timeout: int, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return run(["sudo", "-n", "docker", *arguments], timeout=timeout, check=check)


def write_new(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise I1Error(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path: Path, value: Any, mode: int = 0o600) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    write_new(path, payload, mode)


def last_json_value(stdout: str, label: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise I1Error(f"{label} produced no stdout")
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
            raise I1Error(f"{label} does not contain one final JSON value")
        value = candidates[0]
    if not isinstance(value, dict):
        raise I1Error(f"{label} final JSON is not an object")
    return value


def _safe_relative(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise I1Error(f"unsafe input-manifest path: {relative!r}")
    resolved = (root / path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise I1Error(f"input-manifest path escapes root: {relative!r}") from exc
    return resolved


def inventory_tree(root: Path, *, manifest_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        current_info = current_path.lstat()
        if not stat.S_ISDIR(current_info.st_mode):
            raise I1Error(f"sealed input directory is not a directory: {current_path}")
        if (
            current_info.st_uid != SEALED_OWNER_UID
            or stat.S_IMODE(current_info.st_mode) & 0o022
        ):
            raise I1Error(
                f"sealed input directory is not root-owned read-only: {current_path}"
            )
        for name in sorted(directories):
            path = current_path / name
            if path.is_symlink():
                directories.remove(name)
                relative = path.relative_to(root).as_posix()
                target = os.readlink(path)
                _safe_relative(root, str(Path(relative).parent / target))
                rows.append({"kind": "symlink", "path": relative, "target": target})
        for name in sorted(filenames):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative == manifest_name:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                _safe_relative(root, str(Path(relative).parent / target))
                rows.append({"kind": "symlink", "path": relative, "target": target})
                continue
            if not stat.S_ISREG(info.st_mode):
                raise I1Error(f"sealed input contains a special file: {path}")
            mode = stat.S_IMODE(info.st_mode)
            if info.st_uid != SEALED_OWNER_UID or mode & 0o022 or info.st_nlink != 1:
                raise I1Error(f"sealed input file is not root-owned read-only: {path}")
            rows.append(
                {
                    "bytes": info.st_size,
                    "kind": "file",
                    "mode": mode,
                    "path": relative,
                    "sha256": sha256_path(path),
                }
            )
    rows.sort(key=lambda row: row["path"].encode())
    return rows


def load_input_manifest(
    path: Path,
    *,
    expected_sha256: str,
    phase: str,
    source_commit: str,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise I1Error(f"cannot read complete-I1 input manifest: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != SEALED_OWNER_UID
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or metadata.st_nlink != 1
        or path.is_symlink()
    ):
        raise I1Error("complete-I1 input manifest is not a root-owned read-only file")
    if (
        SHA256.fullmatch(expected_sha256) is None
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise I1Error("complete-I1 input manifest hash mismatch")
    root = path.parent.resolve(strict=True)
    try:
        root.relative_to(SEALED_GATE_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise I1Error(
            "complete-I1 input manifest is outside the sealed gate root"
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise I1Error("complete-I1 input manifest is not JSON") from exc
    expected_fields = {
        "schema_version",
        "gate",
        "phase",
        "source_commit",
        "source_tree",
        "allowed_nodes",
        "image_ref",
        "image_id",
        "source_root",
        "source_commit_object_path",
        "aot_root",
        "loader_path",
        "roundtrip_root",
        "native_extension_path",
        "native_extension_sha256",
        "aot_source_manifest_digest",
        "aot_installed_mla_sha256",
        "aot_expected_entries",
        "files",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise I1Error("complete-I1 input-manifest fields differ from schema")
    expected = {
        "schema_version": 1,
        "gate": "complete_i1_inputs",
        "phase": phase,
        "source_commit": source_commit,
        "allowed_nodes": ["ct13", "ct14"],
        "aot_expected_entries": 14,
    }
    if any(value.get(field) != wanted for field, wanted in expected.items()):
        raise I1Error(
            "complete-I1 input-manifest identity differs from the charged gate"
        )
    for field in ("source_tree", "image_id"):
        candidate = str(value.get(field, ""))
        if field == "source_tree":
            valid = COMMIT.fullmatch(candidate) is not None
        else:
            valid = re.fullmatch(r"sha256:[0-9a-f]{64}", candidate) is not None
        if not valid:
            raise I1Error(f"complete-I1 input manifest has invalid {field}")
    image_ref = str(value.get("image_ref", ""))
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image_ref) is None:
        raise I1Error("complete-I1 image reference is not digest-pinned")
    for field in (
        "native_extension_sha256",
        "aot_source_manifest_digest",
        "aot_installed_mla_sha256",
    ):
        if SHA256.fullmatch(str(value.get(field, ""))) is None:
            raise I1Error(f"complete-I1 input manifest has invalid {field}")
    resolved_inputs = {
        field: _safe_relative(root, str(value.get(field, "")))
        for field in (
            "source_root",
            "source_commit_object_path",
            "aot_root",
            "loader_path",
            "roundtrip_root",
        )
    }
    if value["source_commit_object_path"] != SOURCE_COMMIT_OBJECT_PATH:
        raise I1Error("complete-I1 source commit-object path differs")
    verify_source_identity(
        resolved_inputs["source_root"],
        resolved_inputs["source_commit_object_path"],
        source_commit=source_commit,
        source_tree=str(value["source_tree"]),
    )
    native_path = str(value.get("native_extension_path", ""))
    if native_path != "/opt/omniva/h43/sglang_tq_mla_frontend_sm100_h43_i3_v2.so":
        raise I1Error("complete-I1 native extension path differs")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise I1Error("complete-I1 input file inventory is empty")
    observed = inventory_tree(root, manifest_name=path.name)
    if files != observed:
        raise I1Error("complete-I1 input file inventory differs from sealed manifest")
    image = (
        docker(
            "image",
            "inspect",
            image_ref,
            "--format",
            '{{.Id}} {{index .Config.Labels "com.omniva.inference.h43-native-sha256"}}',
            timeout=30,
        )
        .stdout.strip()
        .split()
    )
    if image != [value["image_id"], value["native_extension_sha256"]]:
        raise I1Error("complete-I1 image ID/native label differs from manifest")
    return value


def sentinel_reasons(value: dict[str, Any]) -> tuple[list[str], float | None]:
    reasons: list[str] = []
    before = value.get("sentinel_before", {})
    after = value.get("sentinel_after", {})
    try:
        before_us = float(before["graph_us"])
        after_us = float(after["graph_us"])
    except (KeyError, TypeError, ValueError):
        return ["sentinel timings are absent, non-finite, or non-positive"], None
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


def load_analyzer(source_root: Path) -> Any:
    path = source_root / "benchmark/bench_turboquant_mla/analyze_h41_i1_integrated.py"
    namespace: dict[str, Any] = {
        "__file__": str(path),
        "__name__": "h44_complete_i1_analyzer",
    }
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    except (OSError, SyntaxError) as exc:
        raise I1Error(f"cannot load the sealed complete-I1 analyzer: {exc}") from exc
    process_environment_reasons = namespace.get("process_environment_reasons")
    if not callable(process_environment_reasons):
        raise I1Error("sealed complete-I1 analyzer omits process validity checks")
    return types.SimpleNamespace(
        process_environment_reasons=process_environment_reasons
    )


class CompleteI1:
    def __init__(self, args: argparse.Namespace):
        self.phase = args.phase
        self.source_commit = args.source_commit
        self.input_manifest_path = args.input_manifest.resolve()
        self.input_manifest_sha256 = args.input_manifest_sha256
        self.result_path = args.result.resolve() if args.result is not None else None
        self.verify_only = args.verify_inputs_only
        self.host = platform.node().split(".", 1)[0]
        self.rank = HOSTS.get(self.host)
        self.manifest: dict[str, Any] = {}
        self.analyzer: Any = None
        self.container_id: str | None = None
        self.attempts: list[dict[str, Any]] = []
        self.replacements = 0
        self.evidence_root: Path | None = None
        self.started_monotonic = time.monotonic()
        self.result_write_authorized = False

    def validate_arguments(self) -> None:
        if (
            self.phase not in SUPPORTED_PHASES
            or SAFE_PHASE.fullmatch(self.phase) is None
        ):
            raise I1Error("unsupported complete-I1 phase")
        if COMMIT.fullmatch(self.source_commit) is None:
            raise I1Error("complete-I1 source commit is invalid")
        if self.rank is None:
            raise I1Error(f"complete-I1 refuses unsupported host: {self.host}")
        if SHA256.fullmatch(self.input_manifest_sha256) is None:
            raise I1Error("complete-I1 input-manifest hash is invalid")
        if self.verify_only:
            if self.result_path is not None:
                raise I1Error("verify-inputs-only refuses a result path")
            return
        if self.host != "beacon-cr-ct13" or self.rank != 0:
            raise I1Error("complete-I1 GPU work is pinned to CT13 rank 0")
        if (
            self.result_path is None
            or self.result_path.name != "complete-i1-result.json"
        ):
            raise I1Error("complete-I1 result filename differs from contract")
        if self.result_path.exists() or self.result_path.is_symlink():
            raise I1Error("complete-I1 result path already exists")
        parent = self.result_path.parent.resolve(strict=True)
        try:
            parent.relative_to(CAMPAIGN_EVIDENCE_ROOT.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise I1Error("complete-I1 result is outside campaign evidence") from exc
        if parent.is_symlink():
            raise I1Error("complete-I1 evidence directory is symlinked")
        self.result_write_authorized = True

    @property
    def container_name(self) -> str:
        return f"tq-h44-{self.phase}-r{self.rank}"

    def verify_inputs(self, *, require_container_absent: bool = True) -> dict[str, Any]:
        self.manifest = load_input_manifest(
            self.input_manifest_path,
            expected_sha256=self.input_manifest_sha256,
            phase=self.phase,
            source_commit=self.source_commit,
        )
        source_root = _safe_relative(
            self.input_manifest_path.parent, str(self.manifest["source_root"])
        )
        self.analyzer = load_analyzer(source_root)
        if require_container_absent:
            stale = docker(
                "container", "inspect", self.container_name, timeout=30, check=False
            )
            if stale.returncode == 0:
                raise I1Error(
                    f"complete-I1 exact container already exists: {self.container_name}"
                )
            if stale.returncode != 1 or not re.search(
                r"No such (container|object)", stale.stderr, re.IGNORECASE
            ):
                raise I1Error("complete-I1 container-existence probe was inconclusive")
        return {
            "host": self.host,
            "input_manifest_sha256": self.input_manifest_sha256,
            "inventory_digest": canonical_digest(self.manifest["files"]),
            "phase": self.phase,
            "source_commit": self.source_commit,
            "status": "PASS",
        }

    def prepare_evidence(self) -> None:
        assert self.result_path is not None
        root = self.result_path.parent / "complete-i1-evidence"
        root.mkdir(mode=0o700)
        for relative in ("raw", "selected", "roundtrip"):
            (root / relative).mkdir(mode=0o700)
        self.evidence_root = root
        manifest_root = self.input_manifest_path.parent
        roundtrip_root = _safe_relative(
            manifest_root, str(self.manifest["roundtrip_root"])
        )
        for context in CONTEXTS:
            raw_context = root / "raw" / f"context{context}"
            selected_context = root / "selected" / f"context{context}"
            roundtrip_context = root / "roundtrip" / f"context{context}"
            for directory in (raw_context, selected_context, roundtrip_context):
                directory.mkdir(mode=0o700)
            for sequence in SEQUENCES:
                (raw_context / f"seq{sequence:02d}").mkdir(mode=0o700)
            for q_len in (1, 5):
                source = (
                    roundtrip_root / f"context{context}" / f"q{q_len}" / "result.json"
                )
                destination = roundtrip_context / f"q{q_len}"
                destination.mkdir(mode=0o700)
                write_new(destination / "result.json", source.read_bytes(), 0o400)
        write_json_new(
            root / "input-verification.json",
            {
                "input_manifest_sha256": self.input_manifest_sha256,
                "inventory_digest": canonical_digest(self.manifest["files"]),
                "phase": self.phase,
                "source_commit": self.source_commit,
                "verified_unix": time.time(),
            },
            0o400,
        )

    def create_container(self) -> None:
        assert self.evidence_root is not None and self.result_path is not None
        root = self.input_manifest_path.parent
        source_root = _safe_relative(root, str(self.manifest["source_root"]))
        aot_root = _safe_relative(root, str(self.manifest["aot_root"]))
        loader = _safe_relative(root, str(self.manifest["loader_path"]))
        command = [
            "create",
            "--name",
            self.container_name,
            "--hostname",
            f"{self.container_name}-isolated",
            "--gpus",
            "device=0",
            "--network",
            "none",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONPATH=/i2/python:/i2:/work",
            "--env",
            "CUTE_DSL_CACHE_DIR=/h44-aot",
            "--env",
            "H43_AOT_MANIFEST=/h44-aot/h43-aot-manifest.json",
            "--env",
            f"H43_AOT_EXPECTED_ENTRIES={self.manifest['aot_expected_entries']}",
            "--env",
            f"H43_SOURCE_MANIFEST_DIGEST={self.manifest['aot_source_manifest_digest']}",
            "--env",
            f"H43_INSTALLED_MLA_SHA256={self.manifest['aot_installed_mla_sha256']}",
            "--env",
            f"SGLANG_TQ_MLA_FRONTEND_SO={self.manifest['native_extension_path']}",
            "--env",
            f"SGLANG_TQ_MLA_FRONTEND_SO_SHA256={self.manifest['native_extension_sha256']}",
            "--volume",
            f"{source_root}:/i2:ro",
            "--volume",
            f"{loader.parent}:/work:ro",
            "--volume",
            f"{aot_root}:/h44-aot:ro",
            "--volume",
            f"{self.result_path.parent}:/results:ro",
            str(self.manifest["image_ref"]),
            "sleep",
            "infinity",
        ]
        created = docker(*command, timeout=120)
        container_id = created.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            raise I1Error("Docker did not return an exact complete-I1 container ID")
        self.container_id = container_id
        inspect = (
            docker(
                "container",
                "inspect",
                container_id,
                "--format",
                "{{.Name}} {{.Image}} {{.Config.Image}}",
                timeout=30,
            )
            .stdout.strip()
            .split()
        )
        expected = [
            f"/{self.container_name}",
            str(self.manifest["image_id"]),
            str(self.manifest["image_ref"]),
        ]
        if inspect != expected:
            raise I1Error(f"complete-I1 container identity mismatch: {inspect!r}")
        write_new(
            self.evidence_root / "container-id.txt",
            (container_id + "\n").encode(),
            0o400,
        )
        started = docker("start", container_id, timeout=120).stdout.strip()
        if started != container_id:
            raise I1Error("Docker did not start the exact complete-I1 container")

    def run_attempt(
        self, context: int, sequence: int, attempt: int, timeout: int
    ) -> tuple[dict[str, Any], list[str], float | None]:
        assert self.container_id is not None and self.evidence_root is not None
        allocation_order = "candidate-first" if sequence == 2 else "control-first"
        name = f"i1-c{context}-seq{sequence:02d}-attempt{attempt:02d}"
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{timeout}s",
            "python3",
            "/i2/benchmark/bench_turboquant_mla/bench_h41_i1_integrated.py",
            "--context",
            str(context),
            "--split-kv",
            str(CONTEXTS[context]["split_kv"]),
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
        completed = docker(
            "exec", self.container_id, *command, timeout=timeout + 30, check=False
        )
        write_new(self.evidence_root / f"{name}.stdout.log", completed.stdout.encode())
        write_new(self.evidence_root / f"{name}.stderr.log", completed.stderr.encode())
        if completed.returncode:
            if completed.returncode == 124 and timeout < SEQUENCE_TIMEOUT_SECONDS:
                raise I1Error(
                    "complete-I1 sequence phase exceeded 600 seconds during its "
                    "truncated final attempt"
                )
            raise I1Error(f"{name} failed with exit code {completed.returncode}")
        value = last_json_value(completed.stdout, name)
        if value.get("status") != "TIMING_ONLY":
            raise I1Error(f"{name} status differs from TIMING_ONLY")
        reasons, drift = sentinel_reasons(value)
        if self.analyzer is None:
            raise I1Error("complete-I1 analyzer validity checks are not loaded")
        environment_reasons = self.analyzer.process_environment_reasons(value)
        if not isinstance(environment_reasons, list) or not all(
            isinstance(reason, str) and reason for reason in environment_reasons
        ):
            raise I1Error("complete-I1 analyzer returned invalid process reasons")
        reasons.extend(environment_reasons)
        raw = (
            self.evidence_root
            / "raw"
            / f"context{context}"
            / f"seq{sequence:02d}"
            / f"attempt{attempt:02d}"
        )
        raw.mkdir(mode=0o700)
        record = {
            "allocation_order": allocation_order,
            "attempt": attempt,
            "capture_order": value.get("capture_order"),
            "command": shlex.join(command),
            "context": context,
            "raw_result": str(raw.relative_to(self.evidence_root) / "result.json"),
            "reasons": reasons,
            "sentinel_after_us": value.get("sentinel_after", {}).get("graph_us"),
            "sentinel_before_us": value.get("sentinel_before", {}).get("graph_us"),
            "sentinel_drift_fraction": drift,
            "sequence": sequence,
            "valid": not reasons,
        }
        self.attempts.append(record)
        write_json_new(raw / "result.json", value, 0o400)
        write_json_new(raw / "validity.json", record, 0o400)
        if not reasons:
            selected = (
                self.evidence_root
                / "selected"
                / f"context{context}"
                / f"seq{sequence:02d}"
            )
            selected.mkdir(mode=0o700)
            write_json_new(selected / "result.json", value, 0o400)
        return value, reasons, drift

    def run_sequences(self, deadline: float) -> None:
        for context in CONTEXTS:
            for sequence in SEQUENCES:
                for attempt in (1, 2):
                    remaining = int(deadline - time.monotonic())
                    if remaining < 1:
                        raise I1Error("complete-I1 sequence phase exceeded 600 seconds")
                    _, reasons, _ = self.run_attempt(
                        context,
                        sequence,
                        attempt,
                        min(SEQUENCE_TIMEOUT_SECONDS, remaining),
                    )
                    if not reasons:
                        break
                    if attempt == 1:
                        if self.replacements >= MAXIMUM_REPLACEMENTS_TOTAL:
                            raise I1Error(
                                "complete-I1 exhausted its one total validity replacement"
                            )
                        self.replacements += 1
                        continue
                    raise I1Error("complete-I1 replacement missed the sentinel gate")

    def analyze(self, deadline: float) -> dict[str, Any]:
        assert self.container_id is not None and self.evidence_root is not None
        remaining = int(deadline - time.monotonic() - POSTFLIGHT_RESERVE_SECONDS)
        if remaining < 1:
            raise I1Error("complete-I1 analysis phase has no remaining budget")
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=15s",
            f"{remaining}s",
            "python3",
            "/i2/benchmark/bench_turboquant_mla/analyze_h41_i1_integrated.py",
            "/results/complete-i1-evidence/selected",
            "--roundtrip-root",
            "/results/complete-i1-evidence/roundtrip",
            "--draws",
            str(BOOTSTRAP_DRAWS),
            "--seed",
            str(SEED),
        ]
        completed = docker(
            "exec", self.container_id, *command, timeout=remaining + 30, check=False
        )
        write_new(self.evidence_root / "analysis.stdout.log", completed.stdout.encode())
        write_new(self.evidence_root / "analysis.stderr.log", completed.stderr.encode())
        if completed.returncode:
            raise I1Error(f"complete-I1 analyzer failed rc={completed.returncode}")
        analysis = last_json_value(completed.stdout, "complete-I1 analyzer")
        if analysis.get("status") not in {"PASS", "FAIL"}:
            raise I1Error("complete-I1 analyzer returned neither PASS nor FAIL")
        observed_contexts = [row.get("context") for row in analysis.get("analyses", [])]
        if observed_contexts != list(CONTEXTS):
            raise I1Error("complete-I1 analyzer omitted or reordered a context")
        write_json_new(self.evidence_root / "analysis.json", analysis, 0o400)
        return analysis

    def result(
        self,
        *,
        status: str,
        analysis: dict[str, Any] | None,
        failure: BaseException | None = None,
    ) -> dict[str, Any]:
        contexts: dict[str, Any] = {}
        if analysis is not None:
            by_context = {row["context"]: row for row in analysis["analyses"]}
            for context, contract in CONTEXTS.items():
                selected_attempts = [
                    row
                    for row in self.attempts
                    if row["context"] == context and row["valid"]
                ]
                drifts = [
                    float(row["sentinel_drift_fraction"]) for row in selected_attempts
                ]
                contexts[str(context)] = {
                    "one_sided_95_upper_us": by_context[context][
                        "delta_bootstrap_one_sided_95_upper_us"
                    ],
                    "replacement_count": sum(
                        row["context"] == context and row["attempt"] == 2
                        for row in self.attempts
                    ),
                    "sentinel_max_observed_drift_fraction": max(drifts),
                    "split_kv": contract["split_kv"],
                    "valid_processes": len(selected_attempts),
                }
        return {
            "analysis": analysis,
            "attempts": self.attempts,
            "container_id": self.container_id,
            "failure": (
                None
                if failure is None
                else {"message": str(failure), "type": type(failure).__name__}
            ),
            "gate": "complete_i1",
            "image_id": self.manifest.get("image_id"),
            "image_ref": self.manifest.get("image_ref"),
            "input_inventory_digest": (
                canonical_digest(self.manifest["files"]) if self.manifest else None
            ),
            "input_manifest_sha256": self.input_manifest_sha256,
            "maximum_replacements_per_sequence": MAXIMUM_REPLACEMENTS_PER_SEQUENCE,
            "maximum_replacements_total": MAXIMUM_REPLACEMENTS_TOTAL,
            "pairs": PAIRS,
            "phase": self.phase,
            "q_len": 5,
            "reader_pdl": True,
            "replays_per_pair": REPLAYS,
            "schema_version": 1,
            "seed": SEED,
            "selected_warps": 8,
            "sentinel_max_drift_fraction": SENTINEL_MAX_DRIFT_FRACTION,
            "sequences": len(SEQUENCES),
            "source_commit": self.source_commit,
            "source_tree": self.manifest.get("source_tree"),
            "status": status,
            "contexts": contexts,
            "warmups": WARMUPS,
        }

    def execute(self) -> dict[str, Any]:
        self.validate_arguments()
        verification = self.verify_inputs()
        if self.verify_only:
            return verification
        assert self.result_path is not None
        overall_deadline = self.started_monotonic + EXECUTABLE_PHASE_SECONDS
        sequence_deadline = min(
            overall_deadline, time.monotonic() + SEQUENCE_PHASE_SECONDS
        )
        self.prepare_evidence()
        self.create_container()
        self.run_sequences(sequence_deadline)
        analysis = self.analyze(overall_deadline)
        post = self.verify_inputs(require_container_absent=False)
        write_json_new(self.evidence_root / "post-input-verification.json", post, 0o400)
        status = str(analysis["status"])
        value = self.result(status=status, analysis=analysis)
        write_json_new(self.result_path, value, 0o400)
        return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(SUPPORTED_PHASES), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--verify-inputs-only", action="store_true")
    parser.add_argument("result", type=Path, nargs="?")
    return parser.parse_args(argv)


def install_signal_guards() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"complete-I1 interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    return previous


def restore_signal_guards(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    previous = install_signal_guards()
    campaign: CompleteI1 | None = None
    try:
        campaign = CompleteI1(args)
        value = campaign.execute()
        print(json.dumps(value, allow_nan=False, sort_keys=True))
        return 0 if value.get("status") == "PASS" else 2
    except BaseException as exc:
        if (
            campaign is not None
            and campaign.result_path is not None
            and campaign.result_write_authorized
        ):
            try:
                if (
                    not campaign.result_path.exists()
                    and not campaign.result_path.is_symlink()
                ):
                    write_json_new(
                        campaign.result_path,
                        campaign.result(
                            status="NO_DECISION", analysis=None, failure=exc
                        ),
                        0o400,
                    )
            except BaseException as evidence_error:
                print(
                    f"failed to persist complete-I1 NO_DECISION: {evidence_error}",
                    file=sys.stderr,
                )
        print(f"complete-I1: {exc}", file=sys.stderr)
        return 2
    finally:
        restore_signal_guards(previous)


if __name__ == "__main__":
    raise SystemExit(main())
