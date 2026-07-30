#!/usr/bin/env python3
"""Extend a sealed H43 reader cache with the I2 no-PDL control keys."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import tokenspeed_mla.mla_decode as mla_decode
import torch
from h43_aot_loader import (
    AOT_MANIFEST_NAME,
    dispatch_key,
    install_h43_aot_from_environment,
    load_aot_manifest,
    write_aot_manifest,
)
from h43_codebook_ab_common import (
    canonical_json_digest,
    compiled_artifact_manifest,
    load_contract,
    sha256_file,
)
from prebuild_h43_codebook_cache import (
    export_aot_kernel,
    key_arguments,
    serializable,
)

BASE_ENTRIES = 10
I2_ENTRIES = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-cache-root", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest-digest", required=True)
    parser.add_argument("--installed-mla-sha256", required=True)
    parser.add_argument("--phase", choices=("extend", "warm"), required=True)
    return parser.parse_args()


def all_keys(contract: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    keys: list[tuple[str, dict[str, Any]]] = []
    query_lengths = contract["geometry"]["correctness_query_lengths"]
    tiles_values = sorted(
        {value["tq4_tiles_per_split"] for value in contract["contexts"].values()}
    )
    for q_len in query_lengths:
        keys.append((f"dense-q{q_len}", key_arguments(q_len=q_len, tq4=False)))
    for tiles in tiles_values:
        for q_len in query_lengths:
            for codebook in (False, True):
                keys.append(
                    (
                        f"tq-tiles{tiles}-q{q_len}-codebook{int(codebook)}",
                        key_arguments(
                            q_len=q_len,
                            tq4=True,
                            tiles_per_split=tiles,
                            codebook=codebook,
                        ),
                    )
                )
            no_pdl = key_arguments(
                q_len=q_len,
                tq4=True,
                tiles_per_split=tiles,
                codebook=True,
            )
            no_pdl["use_pdl"] = False
            keys.append((f"tq-tiles{tiles}-q{q_len}-codebook1-pdl0", no_pdl))
    if len(keys) != I2_ENTRIES:
        raise RuntimeError(f"H43 I2 AOT key count {len(keys)} != {I2_ENTRIES}")
    return keys


def validate_environment(
    *, cache_root: Path, source_digest: str, installed_digest: str
) -> Path:
    if cache_root.name != source_digest:
        raise RuntimeError("cache namespace is not keyed by source-manifest digest")
    configured = Path(os.environ.get("CUTE_DSL_CACHE_DIR", "")).resolve()
    if configured != cache_root:
        raise RuntimeError(f"CUTE_DSL_CACHE_DIR {configured} != {cache_root}")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")
    imported = Path(
        __import__("tokenspeed_mla.mla_decode_fp8", fromlist=["x"]).__file__
    ).resolve()
    if sha256_file(imported) != installed_digest:
        raise RuntimeError("installed mla_decode_fp8.py digest mismatch")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("H43 I2 AOT preparation requires one visible CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.init()
    return imported


def copy_base_artifacts(
    *, base_root: Path, cache_root: Path, base_manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if cache_root.exists():
        raise RuntimeError("extend requires a new cache namespace")
    cache_root.mkdir(parents=True, mode=0o755)
    entries: dict[str, dict[str, Any]] = {}
    for key, entry in base_manifest["entries"].items():
        copied = dict(entry)
        for field, digest_field in (
            ("object", "object_sha256"),
            ("library", "library_sha256"),
        ):
            source = (base_root / entry[field]).resolve()
            destination = cache_root / entry[field]
            destination.parent.mkdir(exist_ok=True)
            with source.open("rb") as source_handle, destination.open(
                "xb"
            ) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)
            if sha256_file(destination) != entry[digest_field]:
                raise RuntimeError(f"copied H43 artifact digest mismatch: {field}")
        entries[key] = copied
    return entries


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract.resolve())
    cache_root = args.cache_root.resolve()
    imported = validate_environment(
        cache_root=cache_root,
        source_digest=args.source_manifest_digest,
        installed_digest=args.installed_mla_sha256,
    )
    keys = all_keys(contract)
    expected_dispatch = {
        dispatch_key(mla_decode._get_compiled_mla_kernel, **arguments): label
        for label, arguments in keys
    }
    if len(expected_dispatch) != I2_ENTRIES:
        raise RuntimeError("H43 I2 expected dispatch keys are not unique")
    if args.phase == "extend":
        before = {
            "root": str(cache_root),
            "files": [],
            "digest": canonical_json_digest([]),
        }
    else:
        before = compiled_artifact_manifest(cache_root, contract["cache"])
    evidence: list[dict[str, Any]] = []
    base_provenance: dict[str, Any] | None = None

    if args.phase == "extend":
        if args.base_cache_root is None:
            raise RuntimeError("extend requires --base-cache-root")
        base_root = args.base_cache_root.resolve()
        if base_root == cache_root:
            raise RuntimeError("base and output cache namespaces must differ")
        base_manifest = load_aot_manifest(
            base_root / AOT_MANIFEST_NAME,
            expected_source_manifest_digest=args.source_manifest_digest,
            expected_installed_mla_sha256=args.installed_mla_sha256,
            expected_entries=BASE_ENTRIES,
        )
        base_before = compiled_artifact_manifest(base_root, contract["cache"])
        if len(base_before["files"]) != 2 * BASE_ENTRIES + 1:
            raise RuntimeError("sealed base AOT cache is not exactly 21 files")
        expected_base_dispatch = {
            key: label
            for key, label in expected_dispatch.items()
            if not label.endswith("-pdl0")
        }
        observed_base_dispatch = {
            key: entry["label"] for key, entry in base_manifest["entries"].items()
        }
        if observed_base_dispatch != expected_base_dispatch:
            raise RuntimeError("sealed base AOT dispatch inventory mismatch")
        base_provenance = {
            "root": str(base_root),
            "manifest_digest": base_manifest["manifest_digest"],
            "artifact_manifest": base_before,
        }
        try:
            entries = copy_base_artifacts(
                base_root=base_root,
                cache_root=cache_root,
                base_manifest=base_manifest,
            )
            for label, arguments in keys:
                if not label.endswith("-pdl0"):
                    continue
                started = time.monotonic()
                compiled = mla_decode._get_compiled_mla_kernel(**arguments)
                elapsed = time.monotonic() - started
                key = dispatch_key(mla_decode._get_compiled_mla_kernel, **arguments)
                if key in entries:
                    raise RuntimeError(f"duplicate H43 I2 AOT key for {label}")
                entries[key] = export_aot_kernel(
                    compiled=compiled,
                    cache_root=cache_root,
                    label=label,
                    key=key,
                )
                evidence.append(
                    {
                        "label": label,
                        "arguments": serializable(arguments),
                        "wall_time_seconds": elapsed,
                    }
                )
            if len(entries) != I2_ENTRIES:
                raise RuntimeError(
                    f"extended entry count {len(entries)} != {I2_ENTRIES}"
                )
            write_aot_manifest(
                cache_root / AOT_MANIFEST_NAME,
                experiment="H43_I2_CODEBOOK_INTEGRATION",
                source_manifest_digest=args.source_manifest_digest,
                installed_mla_sha256=args.installed_mla_sha256,
                entries=entries,
            )
            base_after = compiled_artifact_manifest(base_root, contract["cache"])
            if base_before["digest"] != base_after["digest"]:
                raise RuntimeError("extension changed the sealed base AOT cache")
        except BaseException:
            if cache_root.exists():
                quarantine = cache_root.with_name(
                    f"{cache_root.name}.incomplete-{time.time_ns()}"
                )
                try:
                    os.replace(cache_root, quarantine)
                except OSError as quarantine_error:
                    raise RuntimeError(
                        "failed to quarantine the incomplete H43 I2 AOT cache"
                    ) from quarantine_error
            raise
    else:
        if args.base_cache_root is not None:
            raise RuntimeError("warm verification does not accept --base-cache-root")
        getter = install_h43_aot_from_environment(I2_ENTRIES)
        for label, arguments in keys:
            cache_before = getter.cache_info()
            started = time.monotonic()
            getter(**arguments)
            elapsed = time.monotonic() - started
            cache_after = getter.cache_info()
            if cache_after.misses - cache_before.misses != 1:
                raise RuntimeError(f"{label} was not one fresh AOT dispatch")
            evidence.append(
                {
                    "label": label,
                    "arguments": serializable(arguments),
                    "wall_time_seconds": elapsed,
                }
            )

    manifest = load_aot_manifest(
        cache_root / AOT_MANIFEST_NAME,
        expected_source_manifest_digest=args.source_manifest_digest,
        expected_installed_mla_sha256=args.installed_mla_sha256,
        expected_entries=I2_ENTRIES,
    )
    observed_dispatch = {
        key: entry["label"] for key, entry in manifest["entries"].items()
    }
    if observed_dispatch != expected_dispatch:
        raise RuntimeError("H43 I2 AOT manifest dispatch inventory mismatch")
    after = compiled_artifact_manifest(cache_root, contract["cache"])
    if len(after["files"]) != 2 * I2_ENTRIES + 1:
        raise RuntimeError("H43 I2 AOT artifact inventory is not exactly 29 files")
    if args.phase == "warm" and before["digest"] != after["digest"]:
        raise RuntimeError("warm verification changed the sealed cache")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": "H43_I2_CODEBOOK_INTEGRATION",
        "phase": args.phase,
        "contract_digest": canonical_json_digest(contract),
        "source_manifest_digest": args.source_manifest_digest,
        "installed_mla_path": str(imported),
        "installed_mla_sha256": args.installed_mla_sha256,
        "cache_before": before,
        "cache_after": after,
        "base_cache": base_provenance,
        "keys": evidence,
        "reader_launches": 0,
        "ring_allocations": 0,
    }
    result["result_digest"] = canonical_json_digest(result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
