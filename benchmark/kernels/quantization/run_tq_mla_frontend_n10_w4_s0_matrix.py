"""Run and summarize the preregistered W4-N0 crossed component pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


SHAPES = (
    ("t10219", 10219, 10),
    ("t4096", 4096, 10),
    ("q1", 1, 100),
)
BLOCKS = (
    ("control-first", 0, 0),
    ("control-first", 1, 1),
    ("control-first", 2, 2),
    ("candidates-first", 3, 3),
    ("candidates-first", 4, 4),
    ("candidates-first", 5, 5),
)
TIMING_ORDERS = (
    ("r3b", "s0f", "n0f"),
    ("s0f", "n0f", "r3b"),
    ("n0f", "r3b", "s0f"),
    ("r3b", "n0f", "s0f"),
    ("n0f", "s0f", "r3b"),
    ("s0f", "r3b", "n0f"),
)
ALLOCATION_ORDERS = (
    ("r3b", "n0f", "s0f"),
    ("n0f", "s0f", "r3b"),
    ("s0f", "r3b", "n0f"),
    ("r3b", "s0f", "n0f"),
    ("s0f", "n0f", "r3b"),
    ("n0f", "r3b", "s0f"),
)
EXPECTED_EXPERIMENT = "A17_N10_W4_N0_NATIVE_SM100F_E2M1"
WRITER_DEBIT_BUDGET_MS = 5.206480
MAX_DRIFT = 0.01
MAX_REPLACEMENTS = 3


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, raw: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = raw.encode()
        pending = memoryview(data)
        while pending:
            written = os.write(descriptor, pending)
            if written <= 0:
                raise OSError("short evidence write")
            pending = pending[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path: Path, value: Any) -> None:
    write_new(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", default="workload")
    parser.add_argument("--pod", required=True)
    parser.add_argument("--container", default="server")
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monitor-stop-file", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.kubeconfig.is_file()
        or not args.runner_path.is_absolute()
        or len(args.runner_sha256) != 64
        or not args.output_dir.is_absolute()
        or not args.monitor_stop_file.is_absolute()
    ):
        parser.error("matrix path or identity contract differs")
    return args


def command_for(
    args: argparse.Namespace,
    *,
    tokens: int,
    replays: int,
    allocation_order: str,
    timing_index: int,
    allocation_index: int,
) -> list[str]:
    return [
        "kubectl",
        f"--kubeconfig={args.kubeconfig}",
        f"--context={args.context}",
        "--namespace",
        args.namespace,
        "exec",
        args.pod,
        "-c",
        args.container,
        "--",
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "TORCH_EXTENSIONS_DIR=/tmp/n10w4n0full/torch_extensions",
        "PYTHONPATH=/sgl-workspace/sglang/python:/opt/sglang/python",
        "python3",
        str(args.runner_path),
        "--context",
        "10219",
        "--tokens",
        str(tokens),
        "--allocation-order",
        allocation_order,
        "--timing-order-index",
        str(timing_index),
        "--candidate-allocation-index",
        str(allocation_index),
        "--warmups",
        "100",
        "--samples",
        "20",
        "--replays-per-sample",
        str(replays),
    ]


def validate_result(
    value: dict[str, Any],
    *,
    tokens: int,
    replays: int,
    allocation_order: str,
    timing_index: int,
    allocation_index: int,
) -> None:
    expected_timing_order = list(TIMING_ORDERS[timing_index])
    expected_allocation_order = list(ALLOCATION_ORDERS[allocation_index])
    observed_drift = float(
        value.get("control_flank_abs_drift_fraction", float("inf"))
    )
    try:
        arm_means = value["arm_mean_us"]
        numeric_means = (
            float(value["control_mean_us"]),
            float(arm_means["r3b"]),
            float(arm_means["s0f"]),
            float(arm_means["n0f"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("timing mean fields differ") from exc
    if (
        value.get("status") != "TIMING_ONLY"
        or value.get("experiment") != EXPECTED_EXPERIMENT
        or value.get("hostname") != "r1"
        or value.get("device") != "NVIDIA GB200"
        or value.get("context") != 10219
        or value.get("tokens") != tokens
        or value.get("allocation_order") != allocation_order
        or value.get("timing_order_index") != timing_index
        or value.get("candidate_allocation_index") != allocation_index
        or value.get("timing_candidate_order") != expected_timing_order
        or value.get("candidate_allocation_order") != expected_allocation_order
        or value.get("warmups_per_graph") != 100
        or value.get("samples_per_arm") != 20
        or value.get("replays_per_sample") != replays
        or value.get("candidate_physical_launches")
        != {"r3b": 1, "s0f": 1, "n0f": 1}
        or value.get("control_physical_launches") != 2
        or value.get("passes_control_flank_drift_gate")
        != (observed_drift <= MAX_DRIFT)
        or not math.isfinite(observed_drift)
        or observed_drift < 0
        or any(not math.isfinite(mean) or mean <= 0 for mean in numeric_means)
    ):
        raise RuntimeError("timing result contract differs")
    sequence = value.get("sequence")
    if (
        not isinstance(sequence, list)
        or len(sequence) != 5
        or [entry.get("kind") for entry in sequence]
        != ["control", *expected_timing_order, "control"]
        or any(entry.get("replays") != 20 * replays for entry in sequence)
    ):
        raise RuntimeError("timing sequence contract differs")


def verify_remote_runner(args: argparse.Namespace) -> None:
    command = [
        "kubectl",
        f"--kubeconfig={args.kubeconfig}",
        f"--context={args.context}",
        "--namespace",
        args.namespace,
        "exec",
        args.pod,
        "-c",
        args.container,
        "--",
        "sha256sum",
        str(args.runner_path),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    fields = completed.stdout.split()
    if (
        completed.returncode != 0
        or len(fields) != 2
        or fields[0] != args.runner_sha256
    ):
        raise RuntimeError("remote timing runner digest differs")


def aggregate(valid: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    shape_results: dict[str, Any] = {}
    for shape, values in valid.items():
        means = {
            arm: sum(float(value["arm_mean_us"][arm]) for value in values)
            / len(values)
            for arm in ("r3b", "s0f", "n0f")
        }
        control = sum(float(value["control_mean_us"]) for value in values) / len(
            values
        )
        shape_results[shape] = {
            "valid_blocks": len(values),
            "control_mean_us": control,
            "r3b_mean_us": means["r3b"],
            "s0f_mean_us": means["s0f"],
            "n0f_mean_us": means["n0f"],
            "s0f_minus_r3b_us": means["s0f"] - means["r3b"],
            "s0f_over_r3b": means["s0f"] / means["r3b"],
            "n0f_minus_s0f_us": means["n0f"] - means["s0f"],
            "n0f_over_s0f": means["n0f"] / means["s0f"],
            "max_control_flank_abs_drift_fraction": max(
                float(value["control_flank_abs_drift_fraction"]) for value in values
            ),
        }
    debit = 61.0 * (
        shape_results["t10219"]["n0f_mean_us"]
        - shape_results["t10219"]["control_mean_us"]
    ) / 1000.0
    gates = {
        "t10219_writer_debit": debit <= WRITER_DEBIT_BUDGET_MS,
        "t10219_improves_s0f": shape_results["t10219"]["n0f_over_s0f"] < 1.0,
        "t4096_improves_s0f": shape_results["t4096"]["n0f_over_s0f"] < 1.0,
        "t4096_no_gt_5pct_regression": shape_results["t4096"]["n0f_over_s0f"]
        <= 1.05,
        "q1_no_gt_5pct_regression": shape_results["q1"]["n0f_over_s0f"] <= 1.05,
    }
    return {
        "shape_results": shape_results,
        "projected_61_layer_debit_ms": debit,
        "writer_debit_budget_ms": WRITER_DEBIT_BUDGET_MS,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    if args.monitor_stop_file.exists():
        raise FileExistsError(args.monitor_stop_file)
    valid: dict[str, list[dict[str, Any]]] = {shape: [] for shape, _, _ in SHAPES}
    raw_files: list[Path] = []
    attempts: list[dict[str, Any]] = []
    pids: set[int] = set()
    identities: dict[str, Any] | None = None
    failure: str | None = None
    aggregate_result: dict[str, Any] = {"passed": False}
    started_at = time.time()
    try:
        verify_remote_runner(args)
        for shape, tokens, replays in SHAPES:
            for block, (allocation_order, timing_index, allocation_index) in enumerate(
                BLOCKS
            ):
                accepted = False
                for attempt in range(MAX_REPLACEMENTS + 1):
                    command = command_for(
                        args,
                        tokens=tokens,
                        replays=replays,
                        allocation_order=allocation_order,
                        timing_index=timing_index,
                        allocation_index=allocation_index,
                    )
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    stem = f"{shape}-block{block}-attempt{attempt}"
                    if completed.stderr:
                        write_new(args.output_dir / f"{stem}.stderr", completed.stderr)
                    if completed.returncode != 0:
                        write_new(args.output_dir / f"{stem}.stdout", completed.stdout)
                        raise RuntimeError(
                            f"{shape} block {block} attempt {attempt} returned "
                            f"{completed.returncode}"
                        )
                    try:
                        value = json.loads(completed.stdout)
                    except json.JSONDecodeError as exc:
                        write_new(args.output_dir / f"{stem}.stdout", completed.stdout)
                        raise RuntimeError("timing runner emitted non-JSON stdout") from exc
                    path = args.output_dir / f"{stem}.json"
                    write_new(path, completed.stdout)
                    raw_files.append(path)
                    if not isinstance(value, dict):
                        raise RuntimeError("timing runner JSON root differs")
                    validate_result(
                        value,
                        tokens=tokens,
                        replays=replays,
                        allocation_order=allocation_order,
                        timing_index=timing_index,
                        allocation_index=allocation_index,
                    )
                    pid = int(value["pid"])
                    if pid in pids:
                        raise RuntimeError("timing block did not use a fresh process")
                    pids.add(pid)
                    current_identities = {
                        "artifacts": value["artifact_identities"],
                        "config": value["config_identity"],
                    }
                    if identities is None:
                        identities = current_identities
                    elif current_identities != identities:
                        raise RuntimeError("timing artifact identity drifted")
                    drift = float(value["control_flank_abs_drift_fraction"])
                    attempts.append(
                        {
                            "shape": shape,
                            "block": block,
                            "attempt": attempt,
                            "path": path.name,
                            "control_flank_abs_drift_fraction": drift,
                            "accepted": drift <= MAX_DRIFT,
                        }
                    )
                    if drift <= MAX_DRIFT:
                        valid[shape].append(value)
                        accepted = True
                        break
                if not accepted:
                    raise RuntimeError(
                        f"{shape} block {block} exhausted drift replacements"
                    )
        if any(len(values) != 6 for values in valid.values()):
            raise RuntimeError("valid timing matrix cardinality differs")
        verify_remote_runner(args)
        aggregate_result = aggregate(valid)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    summary = {
        "schema_version": 1,
        "experiment": EXPECTED_EXPERIMENT,
        "started_unix": started_at,
        "finished_unix": time.time(),
        "failure": failure,
        "fresh_process_count": len(pids),
        "runner_path": str(args.runner_path),
        "runner_sha256": args.runner_sha256,
        "artifact_identities": identities,
        "attempts": attempts,
        "raw_sha256": {
            str(path.relative_to(args.output_dir)): sha256_path(path)
            for path in raw_files
        },
        **aggregate_result,
    }
    summary_path = args.output_dir / "summary.json"
    write_new_json(summary_path, summary)
    write_new_json(
        args.monitor_stop_file,
        {
            "schema_version": 1,
            "event": "component-timing-complete",
            "runner_path": str(args.runner_path),
            "runner_sha256": args.runner_sha256,
            "summary_path": str(summary_path),
            "summary_sha256": sha256_path(summary_path),
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
