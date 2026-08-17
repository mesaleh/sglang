"""Bounded SM100f prefix reconstruction for the N10 MLA cache ABI.

The caller owns one flat planar BF16 output scratch: contiguous latent rows
followed by contiguous RoPE rows. This module gathers only the requested rows;
it never exposes or materializes a dense full-pool view.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_MODULE: ModuleType | None = None
_MODULE_LOCK = threading.Lock()
_MODULE_NAME = "sglang_tq_mla_prefix_sm100f_a17_n10_e0_p0_v7"


def _get_module() -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    with _MODULE_LOCK:
        if _MODULE is None:
            source = (
                Path(__file__).resolve().parent
                / "csrc"
                / "tq_mla_frontend"
                / "tq_mla_prefix_n10_native_sm100f.cu"
            )
            _MODULE = load(
                name=_MODULE_NAME,
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "-lineinfo",
                    "-gencode=arch=compute_100f,code=sm_100f",
                ],
                verbose=False,
            )
    return _MODULE


def tq_mla_n10_prefix_gather_out(
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    reciprocal_rope_cache: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    output: torch.Tensor,
    validation_status: torch.Tensor,
    sticky_fault_status: torch.Tensor,
) -> None:
    """Reconstruct requested N10 rows into a caller-owned BF16 scratch.

    ``output`` is a flat ``capacity * 576`` workspace. Its first
    ``capacity * 512`` elements hold contiguous latent rows and the remainder
    hold contiguous RoPE rows. Only the first ``locations.numel()`` rows in
    each plane are eligible for writes. All locations are validated in a
    preceding same-stream kernel; if any is invalid, neither plane is mutated
    and both status words receive the invalid-location bit.
    """

    _get_module().tq_mla_n10_prefix_gather_out(
        packed_cache,
        scale_cache,
        reciprocal_rope_cache,
        locations,
        signs1,
        signs2,
        output,
        validation_status,
        sticky_fault_status,
    )


__all__ = ["tq_mla_n10_prefix_gather_out"]
