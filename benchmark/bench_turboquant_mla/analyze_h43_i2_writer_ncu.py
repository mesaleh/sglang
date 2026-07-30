#!/usr/bin/env python3
"""Fail-closed resource gate for the H43 I2 native writer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

METRICS = (
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
# The accepted W2 eight-warp explicit-rotation specialization used 48
# registers/thread, the same upper class allowed here: 65,536/(48*256) floors
# to five resident blocks before any looser block/warp limit is considered.
W2_MIN_RESIDENT_BLOCKS_PER_SM = 5
RESOURCE = re.compile(
    r"Function (?P<name>[^\n]*tq_mla_frontend_kernel[^\n]*):\n"
    r"\s+REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_number(value: str) -> float:
    parsed = float(value.replace(",", "").strip())
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite NCU value: {value!r}")
    return parsed


def load_one_launch(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    required_columns = {"ID", "Kernel Name", *METRICS}
    header_index = next(
        (index for index, row in enumerate(rows) if required_columns.issubset(row)),
        None,
    )
    if header_index is None:
        raise ValueError(f"{path} has no complete NCU raw header")
    header = rows[header_index]
    if len(header) != len(set(header)):
        raise ValueError(f"{path} has duplicate NCU columns")
    indices = {column: header.index(column) for column in required_columns}
    launches: list[dict[str, Any]] = []
    for row in rows[header_index + 1 :]:
        if len(row) != len(header):
            continue
        raw_id = row[indices["ID"]].strip()
        if not raw_id:
            continue
        try:
            launch_id = int(raw_id)
        except ValueError:
            continue
        launches.append(
            {
                "id": launch_id,
                "kernel_name": row[indices["Kernel Name"]],
                "metrics": {
                    metric: parse_number(row[indices[metric]]) for metric in METRICS
                },
            }
        )
    if len(launches) != 1:
        raise ValueError(f"{path} contains {len(launches)} launches, expected one")
    launch = launches[0]
    if "tq_mla_frontend_kernel" not in launch["kernel_name"]:
        raise ValueError(f"{path} captured the wrong kernel: {launch['kernel_name']}")
    return launch


def summarize(launch: dict[str, Any]) -> dict[str, Any]:
    metrics = launch["metrics"]
    block_limits = (
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    )
    return {
        "launch_id": launch["id"],
        "kernel_name": launch["kernel_name"],
        "registers_per_thread": metrics["launch__registers_per_thread"],
        "static_smem_per_block": metrics["launch__shared_mem_per_block_static"],
        "dynamic_smem_per_block": metrics["launch__shared_mem_per_block_dynamic"],
        "resident_blocks_per_sm": min(metrics[name] for name in block_limits),
        "local_load_instructions": metrics["smsp__inst_executed_op_local_ld.sum"],
        "local_store_instructions": metrics["smsp__inst_executed_op_local_st.sum"],
    }


def load_resources(path: Path) -> dict[str, Any]:
    matches = list(RESOURCE.finditer(path.read_text(encoding="utf-8")))
    if len(matches) != 32:
        raise ValueError(
            f"native module has {len(matches)} writer resource rows, expected 32"
        )
    rows = [
        {
            "name": match.group("name"),
            "registers": int(match.group("registers")),
            "stack": int(match.group("stack")),
            "shared": int(match.group("shared")),
            "local": int(match.group("local")),
        }
        for match in matches
    ]
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("native module repeats a writer specialization")
    return {
        "specialization_count": len(rows),
        "registers_max": max(row["registers"] for row in rows),
        "stack_max": max(row["stack"] for row in rows),
        "shared_max": max(row["shared"] for row in rows),
        "local_max": max(row["local"] for row in rows),
    }


def analyze_paths(
    no_codebook_path: Path, codebook_path: Path, resource_path: Path
) -> dict[str, Any]:
    no_codebook_path = no_codebook_path.resolve()
    codebook_path = codebook_path.resolve()
    resource_path = resource_path.resolve()
    no_codebook = summarize(load_one_launch(no_codebook_path))
    codebook = summarize(load_one_launch(codebook_path))
    resources = load_resources(resource_path)
    gates = {
        "no_codebook_registers_at_most_48": no_codebook["registers_per_thread"] <= 48,
        "codebook_registers_at_most_48": codebook["registers_per_thread"] <= 48,
        "no_static_smem_growth": codebook["static_smem_per_block"]
        <= no_codebook["static_smem_per_block"],
        "no_dynamic_smem_growth": codebook["dynamic_smem_per_block"]
        <= no_codebook["dynamic_smem_per_block"],
        "no_codebook_w2_resident_block_class": no_codebook["resident_blocks_per_sm"]
        >= W2_MIN_RESIDENT_BLOCKS_PER_SM,
        "codebook_w2_resident_block_class": codebook["resident_blocks_per_sm"]
        >= W2_MIN_RESIDENT_BLOCKS_PER_SM,
        "codebook_no_resident_block_regression": codebook["resident_blocks_per_sm"]
        >= no_codebook["resident_blocks_per_sm"],
        "no_codebook_no_local_instructions": no_codebook["local_load_instructions"] == 0
        and no_codebook["local_store_instructions"] == 0,
        "codebook_no_local_instructions": codebook["local_load_instructions"] == 0
        and codebook["local_store_instructions"] == 0,
        "all_specializations_registers_at_most_48": resources["registers_max"] <= 48,
        "all_specializations_zero_stack": resources["stack_max"] == 0,
        "all_specializations_zero_shared": resources["shared_max"] == 0,
        "all_specializations_zero_local": resources["local_max"] == 0,
    }
    value = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "experiment": "H43_I2_NATIVE_WRITER_RESOURCE_GATE",
        "no_codebook_csv_sha256": sha256_file(no_codebook_path),
        "codebook_csv_sha256": sha256_file(codebook_path),
        "resource_usage_sha256": sha256_file(resource_path),
        "no_codebook": no_codebook,
        "codebook": codebook,
        "resources": resources,
        "w2_min_resident_blocks_per_sm": W2_MIN_RESIDENT_BLOCKS_PER_SM,
        "gates": gates,
    }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-codebook", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--resource-usage", type=Path, required=True)
    args = parser.parse_args()
    value = analyze_paths(args.no_codebook, args.codebook, args.resource_usage)
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
    if value["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
