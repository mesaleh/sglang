#!/usr/bin/env python3
"""Seal a staged H44 complete-I1 input tree and emit its immutable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any

_prior_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    import run_h44_complete_i1 as runner
finally:
    sys.dont_write_bytecode = _prior_dont_write_bytecode


NATIVE_PATH = "/opt/omniva/h43/sglang_tq_mla_frontend_sm100_h43_i3_v2.so"
IMAGE_REF = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
STAGING_NAME = re.compile(r"^\.staging-[a-z0-9][a-z0-9._-]{0,100}$")
ROUNDTRIP_PATHS = {
    f"context{context}/q{q_len}/result.json"
    for context in runner.CONTEXTS
    for q_len in (1, 5)
}


class SealError(RuntimeError):
    pass


def docker_image_identity(image_ref: str) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image_ref,
                "--format",
                '{{.Id}} {{index .Config.Labels "com.omniva.inference.h43-native-sha256"}}',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealError(f"cannot inspect complete-I1 image: {exc}") from exc
    fields = completed.stdout.strip().split()
    if completed.returncode or len(fields) != 2:
        raise SealError(
            f"complete-I1 image inspection failed rc={completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    image_id, native_sha256 = fields
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        or runner.SHA256.fullmatch(native_sha256) is None
    ):
        raise SealError("complete-I1 image identity or native label is invalid")
    return image_id, native_sha256


def validate_layout(
    root: Path, *, source_commit: str, source_tree: str
) -> dict[str, Any]:
    required_directories = {
        "source": root / "source",
        "aot": root / "aot",
        "work": root / "work",
        "roundtrip": root / "roundtrip",
        "provenance": root / "provenance",
    }
    if set(path.name for path in root.iterdir()) != set(required_directories):
        raise SealError("staged complete-I1 root has unexpected top-level entries")
    for label, path in required_directories.items():
        if path.is_symlink() or not path.is_dir():
            raise SealError(f"staged complete-I1 {label} is not a real directory")
    commit_object = required_directories["provenance"] / "source-commit-object"
    try:
        runner.verify_source_identity(
            required_directories["source"],
            commit_object,
            source_commit=source_commit,
            source_tree=source_tree,
        )
    except (OSError, runner.I1Error) as exc:
        raise SealError(f"staged source identity is invalid: {exc}") from exc
    for relative in (
        "benchmark/bench_turboquant_mla/run_h44_complete_i1.py",
        "benchmark/bench_turboquant_mla/bench_h41_i1_integrated.py",
        "benchmark/bench_turboquant_mla/analyze_h41_i1_integrated.py",
    ):
        path = required_directories["source"] / relative
        if path.is_symlink() or not path.is_file():
            raise SealError(f"staged source omits complete-I1 tool: {relative}")
    loader = required_directories["work"] / "h43_aot_loader.py"
    if loader.is_symlink() or not loader.is_file():
        raise SealError("staged complete-I1 loader is absent")
    observed_roundtrips = {
        path.relative_to(required_directories["roundtrip"]).as_posix()
        for path in required_directories["roundtrip"].rglob("*")
        if path.is_file()
    }
    if observed_roundtrips != ROUNDTRIP_PATHS:
        raise SealError("staged complete-I1 roundtrip set differs from four results")
    aot_manifest_path = required_directories["aot"] / "h43-aot-manifest.json"
    try:
        aot_manifest = json.loads(aot_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealError(f"cannot read staged AOT manifest: {exc}") from exc
    if (
        not isinstance(aot_manifest, dict)
        or aot_manifest.get("expected_entries") != 14
        or len(aot_manifest.get("entries", {})) != 14
        or runner.SHA256.fullmatch(str(aot_manifest.get("source_manifest_digest", "")))
        is None
        or runner.SHA256.fullmatch(str(aot_manifest.get("installed_mla_sha256", "")))
        is None
    ):
        raise SealError("staged complete-I1 AOT manifest differs from contract")
    return aot_manifest


def seal_tree(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    directories: list[Path] = []
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            try:
                runner._safe_symlink_target(
                    root,
                    path.relative_to(root).as_posix(),
                    os.readlink(path),
                )
            except runner.I1Error as exc:
                raise SealError(f"unsafe staged complete-I1 symlink: {path}") from exc
            os.chown(path, 0, 0, follow_symlinks=False)
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise SealError(f"staged complete-I1 file is hard-linked: {path}")
            os.chown(path, 0, 0)
            path.chmod(0o555 if metadata.st_mode & 0o111 else 0o444)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise SealError(f"staged complete-I1 tree contains a special file: {path}")
        else:
            directories.append(path)
    for path in sorted(
        directories,
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        os.chown(path, 0, 0)
        path.chmod(0o555)


def seal(args: argparse.Namespace) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise SealError("complete-I1 input sealing requires root")
    if args.phase not in runner.SUPPORTED_PHASES:
        raise SealError("unsupported complete-I1 phase")
    if runner.COMMIT.fullmatch(args.source_commit) is None:
        raise SealError("complete-I1 source commit is invalid")
    if runner.COMMIT.fullmatch(args.source_tree) is None:
        raise SealError("complete-I1 source tree is invalid")
    if IMAGE_REF.fullmatch(args.image_ref) is None:
        raise SealError("complete-I1 image reference is not digest-pinned")
    root = args.root.resolve(strict=True)
    gate_root = runner.SEALED_GATE_ROOT.resolve(strict=True)
    if root.parent != gate_root or STAGING_NAME.fullmatch(root.name) is None:
        raise SealError(
            "complete-I1 staging root is outside the exact sealed gate root"
        )
    if not root.name.startswith(f".staging-{args.phase}-"):
        raise SealError("complete-I1 staging root is not bound to its phase")
    if root.is_symlink():
        raise SealError("complete-I1 staging root is symlinked")
    final_root = gate_root / args.phase
    if final_root.exists() or final_root.is_symlink():
        raise SealError("complete-I1 final input root already exists")
    manifest_path = root / "input-manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise SealError("complete-I1 input manifest already exists")

    aot_manifest = validate_layout(
        root, source_commit=args.source_commit, source_tree=args.source_tree
    )
    image_id, native_sha256 = docker_image_identity(args.image_ref)
    executable_dir = root / "executable"
    executable_dir.mkdir(mode=0o700)
    source_runner = (
        root
        / "source"
        / "benchmark"
        / "bench_turboquant_mla"
        / "run_h44_complete_i1.py"
    )
    executable = executable_dir / "run_h44_complete_i1.py"
    shutil.copyfile(source_runner, executable)
    if runner.sha256_path(executable) != runner.sha256_path(source_runner):
        raise SealError(
            "complete-I1 executable copy differs from the charged source tree"
        )
    executable.chmod(0o700)
    descriptor = os.open(
        manifest_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        seal_tree(root)
        files = runner.inventory_tree(root, manifest_name=manifest_path.name)
        manifest = {
            "allowed_nodes": ["ct13", "ct14"],
            "aot_expected_entries": 14,
            "aot_installed_mla_sha256": aot_manifest["installed_mla_sha256"],
            "aot_root": "aot",
            "aot_source_manifest_digest": aot_manifest["source_manifest_digest"],
            "files": files,
            "gate": "complete_i1_inputs",
            "image_id": image_id,
            "image_ref": args.image_ref,
            "loader_path": "work/h43_aot_loader.py",
            "native_extension_path": NATIVE_PATH,
            "native_extension_sha256": native_sha256,
            "phase": args.phase,
            "roundtrip_root": "roundtrip",
            "schema_version": 1,
            "source_commit": args.source_commit,
            "source_commit_object_path": runner.SOURCE_COMMIT_OBJECT_PATH,
            "source_root": "source",
            "source_tree": args.source_tree,
        }
        payload = (
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise SealError("short write while creating complete-I1 manifest")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    manifest_sha256 = runner.sha256_path(manifest_path)
    observed = runner.load_input_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        phase=args.phase,
        source_commit=args.source_commit,
    )
    if observed != manifest:
        raise SealError("complete-I1 manifest self-verification changed the value")
    return {
        "executable_path": str(final_root / "executable" / "run_h44_complete_i1.py"),
        "executable_sha256": runner.sha256_path(
            executable_dir / "run_h44_complete_i1.py"
        ),
        "input_manifest_path": str(final_root / manifest_path.name),
        "input_manifest_sha256": manifest_sha256,
        "inventory_digest": runner.canonical_digest(files),
        "status": "PASS",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=sorted(runner.SUPPORTED_PHASES), required=True
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        value = seal(parse_args(argv))
    except (OSError, SealError, runner.I1Error) as exc:
        print(f"seal-complete-I1: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
