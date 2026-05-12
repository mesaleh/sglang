"""Microbenchmark Kimi/MLA TurboQuant decode attention components.

This benchmark isolates the local, single-rank TurboQuant MLA decode path from
the HTTP server, NCCL all-reduce, and MoE work. It answers whether the fused
TurboQuant MLA attention code is itself worth kernel work before we attempt a
CUDA rewrite.

Example:
    python3 python/sglang/jit_kernel/benchmark/bench_tq_mla_decode_attention.py \
      --seq-lens 1024 4096 16000 32000 --batch-sizes 1 3 --out-dir /tmp/tq-mla
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import triton
import triton.testing

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_softmax_reducev_fwd,
)
from sglang.srt.layers.attention.triton_ops.turboquant_mla_decode_attention import (
    _MIN_BLOCK_KV,
    _fwd_tq_mla_decode_stage1,
    tq_mla_decode_attention_fwd,
)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig


GRAPH_UNSAFE_OPS = {
    # The production server fills this metadata outside the captured CUDA graph.
    # Capturing it directly is useful once as a bug reproducer, but it poisons
    # the process after the expected "operation not permitted when stream is
    # capturing" error.
    "metadata_full",
    "local_decode_with_metadata",
}


@dataclass
class Shape:
    batch_size: int
    seq_len: int
    q_heads: int
    lora_rank: int
    rope_dim: int
    max_kv_splits: int


@dataclass
class BenchResult:
    op: str
    batch_size: int
    seq_len: int
    q_heads: int
    block_n: int
    block_h: int
    num_warps: int
    num_stages: int
    eager_p50_us: float | None
    eager_p20_us: float | None
    eager_p80_us: float | None
    graph_p50_us: float | None
    graph_p20_us: float | None
    graph_p80_us: float | None
    status: str
    error: str


class Inputs:
    def __init__(self, shape: Shape, seed: int, device: str = "cuda") -> None:
        torch.manual_seed(seed + shape.batch_size * 100000 + shape.seq_len)
        self.shape = shape
        self.device = device
        self.pool_size = shape.batch_size * shape.seq_len
        self.dtype = torch.bfloat16

        self.q_nope = torch.randn(
            (shape.batch_size, shape.q_heads, shape.lora_rank),
            dtype=self.dtype,
            device=device,
        )
        self.q_rope = torch.randn(
            (shape.batch_size, shape.q_heads, shape.rope_dim),
            dtype=self.dtype,
            device=device,
        )
        self.q_nope_rotated = torch.empty_like(self.q_nope)
        self.o_rotated = torch.empty_like(self.q_nope)
        self.o = torch.empty_like(self.q_nope)

        self.k_nope_packed = torch.randint(
            0,
            256,
            (self.pool_size, 1, shape.lora_rank // 2),
            dtype=torch.uint8,
            device=device,
        )
        self.k_scale = (
            0.75
            + 0.5
            * torch.rand((self.pool_size, 1), dtype=torch.float32, device=device)
        ).to(self.dtype)
        self.k_rope = torch.randn(
            (self.pool_size, 1, shape.rope_dim), dtype=self.dtype, device=device
        )

        self.tq_config = TurboQuantConfig(
            bit_width=4,
            head_dim=shape.lora_rank,
            device=device,
            k_bit_width=4,
            v_bit_width=4,
            uniform=False,
        )
        self.k_centroids = self.tq_config.k_centroids

        self.req_to_token = torch.arange(
            self.pool_size, dtype=torch.int32, device=device
        ).reshape(shape.batch_size, shape.seq_len)
        self.req_pool_indices = torch.arange(
            shape.batch_size, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.full(
            (shape.batch_size,), shape.seq_len, dtype=torch.int32, device=device
        )
        self.kv_indptr = torch.empty(
            shape.batch_size + 1, dtype=torch.int32, device=device
        )
        self.kv_indices = torch.empty(
            self.pool_size, dtype=torch.int32, device=device
        )
        self.num_kv_splits = torch.full(
            (shape.batch_size,),
            shape.max_kv_splits,
            dtype=torch.int32,
            device=device,
        )
        self.att_logits = torch.empty(
            (
                shape.batch_size,
                shape.q_heads,
                shape.max_kv_splits,
                shape.lora_rank,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.att_lse = torch.empty(
            (shape.batch_size, shape.q_heads, shape.max_kv_splits),
            dtype=torch.float32,
            device=device,
        )

        self.fill_metadata()
        self.q_nope_rotated.copy_(self.tq_config.rotate_query(self.q_nope))
        tq_mla_decode_attention_fwd(
            q_nope_rotated=self.q_nope_rotated,
            q_rope=self.q_rope,
            k_nope_packed=self.k_nope_packed,
            k_scale=self.k_scale,
            k_rope=self.k_rope,
            k_centroids=self.k_centroids,
            o=self.o_rotated,
            kv_indptr=self.kv_indptr,
            kv_indices=self.kv_indices,
            att_logits=self.att_logits,
            att_lse=self.att_lse,
            num_kv_splits=self.num_kv_splits,
            max_kv_splits=shape.max_kv_splits,
            sm_scale=1.0 / math.sqrt(shape.lora_rank + shape.rope_dim),
            logit_cap=0.0,
            uniform=False,
        )
        torch.cuda.synchronize()

    def fill_metadata(self) -> None:
        self.kv_indptr[0] = 0
        self.kv_indptr[1 : self.shape.batch_size + 1] = torch.cumsum(
            self.seq_lens, dim=0
        )
        create_flashinfer_kv_indices_triton[(self.shape.batch_size,)](
            self.req_to_token,
            self.req_pool_indices,
            self.seq_lens,
            self.kv_indptr,
            None,
            self.kv_indices,
            self.req_to_token.stride(0),
        )

    def fill_indices_only(self) -> None:
        create_flashinfer_kv_indices_triton[(self.shape.batch_size,)](
            self.req_to_token,
            self.req_pool_indices,
            self.seq_lens,
            self.kv_indptr,
            None,
            self.kv_indices,
            self.req_to_token.stride(0),
        )


def launch_stage1(
    x: Inputs,
    block_n: int,
    block_h: int,
    num_warps: int,
    num_stages: int,
) -> None:
    shape = x.shape
    block_h = min(block_h, shape.q_heads)
    grid = (
        shape.batch_size,
        triton.cdiv(shape.q_heads, block_h),
        shape.max_kv_splits,
    )
    _fwd_tq_mla_decode_stage1[grid](
        x.q_nope_rotated,
        x.q_rope,
        x.k_nope_packed,
        x.k_scale,
        x.k_rope,
        x.k_centroids,
        x.kv_indptr,
        x.kv_indices,
        x.num_kv_splits,
        x.att_logits,
        x.att_lse,
        x.q_nope_rotated.stride(0),
        x.q_nope_rotated.stride(1),
        x.q_rope.stride(0),
        x.q_rope.stride(1),
        x.k_nope_packed.stride(0),
        x.k_nope_packed.stride(1),
        x.k_scale.stride(0),
        x.k_rope.stride(0),
        x.k_rope.stride(1),
        x.att_logits.stride(0),
        x.att_logits.stride(1),
        x.att_logits.stride(2),
        1.0 / math.sqrt(shape.lora_rank + shape.rope_dim),
        q_head_num=shape.q_heads,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        LORA_RANK=shape.lora_rank,
        ROPE_DIM=shape.rope_dim,
        LORA_PACKED=shape.lora_rank // 2,
        BLOCK_LORA=triton.next_power_of_2(shape.lora_rank),
        BLOCK_LORA_PACKED=triton.next_power_of_2(shape.lora_rank // 2),
        BLOCK_ROPE=triton.next_power_of_2(shape.rope_dim),
        logit_cap=0.0,
        UNIFORM=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def launch_stage2(x: Inputs) -> None:
    _decode_softmax_reducev_fwd(
        x.att_logits,
        x.att_lse,
        x.q_nope_rotated,
        x.o_rotated,
        1.0,
        x.o_rotated,
        x.kv_indptr,
        x.num_kv_splits,
        x.shape.max_kv_splits,
        None,
    )


def launch_full_attention(x: Inputs) -> None:
    tq_mla_decode_attention_fwd(
        q_nope_rotated=x.q_nope_rotated,
        q_rope=x.q_rope,
        k_nope_packed=x.k_nope_packed,
        k_scale=x.k_scale,
        k_rope=x.k_rope,
        k_centroids=x.k_centroids,
        o=x.o_rotated,
        kv_indptr=x.kv_indptr,
        kv_indices=x.kv_indices,
        att_logits=x.att_logits,
        att_lse=x.att_lse,
        num_kv_splits=x.num_kv_splits,
        max_kv_splits=x.shape.max_kv_splits,
        sm_scale=1.0 / math.sqrt(x.shape.lora_rank + x.shape.rope_dim),
        logit_cap=0.0,
        uniform=False,
    )


def launch_local_decode(x: Inputs) -> None:
    x.q_nope_rotated.copy_(x.tq_config.rotate_query(x.q_nope))
    launch_full_attention(x)
    x.o.copy_(x.tq_config.inverse_rotate_output(x.o_rotated))


def launch_local_decode_with_metadata(x: Inputs) -> None:
    x.fill_metadata()
    launch_local_decode(x)


def measure(
    fn: Callable[[], None], warmup: int, rep: int, use_graph: bool
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if use_graph:
        ms, p20_ms, p80_ms = triton.testing.do_bench_cudagraph(
            fn, rep=rep, quantiles=[0.5, 0.2, 0.8]
        )
    else:
        ms, p20_ms, p80_ms = triton.testing.do_bench(
            fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
        )
    return 1000.0 * ms, 1000.0 * p20_ms, 1000.0 * p80_ms


def safe_measure(
    fn: Callable[[], None], warmup: int, rep: int, use_graph: bool
) -> tuple[float | None, float | None, float | None, str, str]:
    try:
        p50, p20, p80 = measure(fn, warmup=warmup, rep=rep, use_graph=use_graph)
        return p50, p20, p80, "ok", ""
    except Exception as exc:  # pragma: no cover - used for cluster diagnostics
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
        return None, None, None, "error", error


def skipped_graph_measure(reason: str) -> tuple[None, None, None, str, str]:
    return None, None, None, "skipped", reason


def op_fns(
    x: Inputs,
    block_n: int,
    block_h: int,
    num_warps: int,
    num_stages: int,
) -> dict[str, Callable[[], None]]:
    return {
        "metadata_full": x.fill_metadata,
        "metadata_indices_only": x.fill_indices_only,
        "rotate_query": lambda: x.q_nope_rotated.copy_(x.tq_config.rotate_query(x.q_nope)),
        "stage1": lambda: launch_stage1(x, block_n, block_h, num_warps, num_stages),
        "stage2": lambda: launch_stage2(x),
        "full_attention": lambda: launch_full_attention(x),
        "inverse_rotate": lambda: x.o.copy_(x.tq_config.inverse_rotate_output(x.o_rotated)),
        "local_decode": lambda: launch_local_decode(x),
        "local_decode_with_metadata": lambda: launch_local_decode_with_metadata(x),
    }


def result_to_row(result: BenchResult) -> dict[str, object]:
    return {
        "op": result.op,
        "batch_size": result.batch_size,
        "seq_len": result.seq_len,
        "q_heads": result.q_heads,
        "block_n": result.block_n,
        "block_h": result.block_h,
        "num_warps": result.num_warps,
        "num_stages": result.num_stages,
        "eager_p50_us": result.eager_p50_us,
        "eager_p20_us": result.eager_p20_us,
        "eager_p80_us": result.eager_p80_us,
        "graph_p50_us": result.graph_p50_us,
        "graph_p20_us": result.graph_p20_us,
        "graph_p80_us": result.graph_p80_us,
        "status": result.status,
        "error": result.error.replace("\n", "\\n") if result.error else "",
    }


def write_outputs(results: list[BenchResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [result_to_row(result) for result in results]
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        empty_result = BenchResult(
            "", 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, "", ""
        )
        fieldnames = list(result_to_row(empty_result).keys())
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_ints(values: Iterable[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        out.extend(int(part) for part in str(value).split(",") if part)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/tq-mla-microbench"))
    parser.add_argument("--seq-lens", nargs="+", default=["1024", "4096", "16000", "32000"])
    parser.add_argument("--batch-sizes", nargs="+", default=["1", "3"])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--max-kv-splits", type=int, default=8)
    parser.add_argument("--block-ns", nargs="+", default=["64"])
    parser.add_argument("--block-hs", nargs="+", default=["16"])
    parser.add_argument("--num-warps", nargs="+", default=["8"])
    parser.add_argument("--num-stages", nargs="+", default=["2"])
    parser.add_argument("--ops", nargs="+", default=["all"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    requested_ops = set(args.ops)
    seq_lens = parse_ints(args.seq_lens)
    batch_sizes = parse_ints(args.batch_sizes)
    block_ns = parse_ints(args.block_ns)
    block_hs = parse_ints(args.block_hs)
    num_warps_values = parse_ints(args.num_warps)
    num_stages_values = parse_ints(args.num_stages)
    all_results: list[BenchResult] = []

    env = {
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "pid": os.getpid(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            shape = Shape(
                batch_size=batch_size,
                seq_len=seq_len,
                q_heads=args.q_heads,
                lora_rank=args.lora_rank,
                rope_dim=args.rope_dim,
                max_kv_splits=args.max_kv_splits,
            )
            x = Inputs(shape, seed=args.seed)
            for block_n in block_ns:
                for block_h in block_hs:
                    for num_warps in num_warps_values:
                        for num_stages in num_stages_values:
                            fns = op_fns(x, block_n, block_h, num_warps, num_stages)
                            for op_name, fn in fns.items():
                                if "all" not in requested_ops and op_name not in requested_ops:
                                    continue
                                eager = safe_measure(fn, args.warmup, args.rep, use_graph=False)
                                if op_name in GRAPH_UNSAFE_OPS:
                                    graph = skipped_graph_measure(
                                        "graph capture skipped: op includes CUDA-graph-unsafe "
                                        "metadata assignment/cumsum"
                                    )
                                else:
                                    graph = safe_measure(
                                        fn, args.warmup, args.rep, use_graph=True
                                    )
                                status = (
                                    "ok"
                                    if eager[3] == "ok" and graph[3] in {"ok", "skipped"}
                                    else "error"
                                )
                                error = "\n".join(err for err in (eager[4], graph[4]) if err)
                                result = BenchResult(
                                    op=op_name,
                                    batch_size=batch_size,
                                    seq_len=seq_len,
                                    q_heads=args.q_heads,
                                    block_n=block_n,
                                    block_h=min(block_h, args.q_heads),
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                    eager_p50_us=eager[0],
                                    eager_p20_us=eager[1],
                                    eager_p80_us=eager[2],
                                    graph_p50_us=graph[0],
                                    graph_p20_us=graph[1],
                                    graph_p80_us=graph[2],
                                    status=status,
                                    error=error,
                                )
                                all_results.append(result)
                                print(json.dumps(result_to_row(result), sort_keys=True), flush=True)

    write_outputs(all_results, args.out_dir)
    print(f"Wrote {len(all_results)} rows to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
