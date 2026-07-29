#!/usr/bin/env python3
"""Validate H42 D1 CUDA-graph node counts and operation ordering.

Nsight timing is intentionally ignored.  The reports were captured only to
prove that each diagnostic graph contains the declared kernel families in the
declared order.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


VARIANTS = (
    "f-chain",
    "s-chain",
    "s-static",
    "s-phased",
    "s-chain-noapdl",
)


def family(kernel: str) -> str:
    if kernel == "RopeQuantizeKernel":
        return "dense_rope_fp8"
    if kernel == "set_mla_kv_buffer_kernel":
        return "dense_scatter"
    if kernel == "BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel":
        return "bf16_rope"
    if kernel == "tq_mla_frontend_kernel":
        return "tq_frontend"
    if kernel.startswith("kernel_cutlass_split_kv_kernel_tokenspeed_mla"):
        return "reader_split"
    if kernel.startswith("kernel_cutlass_reduction_kernel_tokenspeed_mla"):
        return "reader_reduce"
    raise ValueError(f"unexpected graph kernel: {kernel}")


def expected_order(variant: str, arm: str) -> list[str]:
    dense = [
        "dense_rope_fp8",
        "dense_scatter",
        "reader_split",
        "reader_reduce",
    ]
    selected = ["bf16_rope", "tq_frontend", "reader_split", "reader_reduce"]
    if variant == "f-chain":
        return dense * 61 if arm == "control" else dense * 24 + selected * 14 + dense * 23
    if variant == "s-phased":
        frontend = (
            ["dense_rope_fp8", "dense_scatter"]
            if arm == "control"
            else ["bf16_rope", "tq_frontend"]
        )
        return frontend * 14 + ["reader_split", "reader_reduce"] * 14
    return (dense if arm == "control" else selected) * 14


def load_graphs(path: Path) -> dict[str, dict[str, Any]]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT
              (k.graphNodeId >> 32) AS generation,
              (k.graphNodeId & 4294967295) AS node_index,
              s.value AS kernel,
              COUNT(*) AS executions
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            WHERE k.graphNodeId IS NOT NULL
            GROUP BY generation, node_index, kernel
            ORDER BY generation, node_index
            """
        ).fetchall()

    generations: dict[int, list[tuple[int, str, int]]] = {}
    for generation, node_index, kernel, executions in rows:
        generations.setdefault(generation, []).append(
            (node_index, family(kernel), executions)
        )
    if len(generations) != 2:
        raise ValueError(f"{path.name}: expected two graph generations, got {generations.keys()}")

    classified: dict[str, dict[str, Any]] = {}
    for generation, nodes in generations.items():
        families = [item[1] for item in nodes]
        arm = "candidate" if "tq_frontend" in families else "control"
        if arm in classified:
            raise ValueError(f"{path.name}: duplicate {arm} graph")
        indices = [item[0] for item in nodes]
        if indices != list(range(len(nodes))):
            raise ValueError(f"{path.name}: {arm} node indices are not contiguous")
        execution_counts = sorted({item[2] for item in nodes})
        classified[arm] = {
            "generation": generation,
            "nodes": len(nodes),
            "executions_per_node": execution_counts,
            "families": families,
        }
    if set(classified) != {"control", "candidate"}:
        raise ValueError(f"{path.name}: could not classify both graph arms")
    return classified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "status": "VALID",
        "interpretation": (
            "Physical graph structure only; instrumented kernel durations are excluded "
            "from every performance decision."
        ),
        "variants": {},
    }
    for variant in VARIANTS:
        path = args.root / f"{variant}.sqlite"
        graphs = load_graphs(path)
        expected_nodes = 244 if variant == "f-chain" else 56
        for arm, expected_replays in (("control", 12), ("candidate", 11)):
            graph = graphs[arm]
            if graph["nodes"] != expected_nodes:
                raise ValueError(
                    f"{variant} {arm}: expected {expected_nodes} nodes, got {graph['nodes']}"
                )
            if graph["executions_per_node"] != [expected_replays]:
                raise ValueError(
                    f"{variant} {arm}: replay count drift {graph['executions_per_node']}"
                )
            declared = expected_order(variant, arm)
            if graph["families"] != declared:
                mismatch = next(
                    index
                    for index, (observed, expected) in enumerate(
                        zip(graph["families"], declared, strict=True)
                    )
                    if observed != expected
                )
                raise ValueError(
                    f"{variant} {arm}: order mismatch at node {mismatch}: "
                    f"{graph['families'][mismatch]} != {declared[mismatch]}"
                )
            del graph["families"]
            graph["order_gate"] = "PASS"
        report["variants"][variant] = graphs

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
