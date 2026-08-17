"""Fixed-codebook native SM100f writer for the N10 reciprocal-RoPE MLA ABI.

The persistent cache row is 256 packed E2M1 bytes, one BF16 scale, and 64
BF16 reciprocal-RoPE values. All public entries are allocation-free after the
extension is loaded and launch on PyTorch's current stream.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_MODULE: ModuleType | None = None
_MODULE_LOCK = threading.Lock()
_MODULE_NAME = "sglang_tq_mla_frontend_sm100f_a17_n10_w4_n0f_v2"


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
                / "tq_mla_frontend_n10_native_sm100f.cu"
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


def tq_mla_n10_native_cache_writer_out(
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    reciprocal_rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    zero_count: torch.Tensor,
    *,
    grid: float,
    strict: bool = False,
) -> None:
    """Write post-RoPE BF16 keys into the no-shadow N10 cache ABI."""

    _get_module().tq_mla_n10_native_cache_writer_out(
        cache_latent,
        cache_rope,
        locations,
        signs1,
        signs2,
        packed_cache,
        scale_cache,
        reciprocal_rope_cache,
        fault_status,
        zero_count,
        float(grid),
        bool(strict),
    )


def tq_mla_n10_native_frontend_out(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    query_latent_out: torch.Tensor,
    query_rope_out: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    reciprocal_rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    zero_count: torch.Tensor,
    *,
    grid: float,
    rotation_fused: bool,
    strict: bool = False,
) -> None:
    """Emit separate query tensors and a post-RoPE N10 row in one launch."""

    _get_module().tq_mla_n10_native_frontend_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        locations,
        signs1,
        signs2,
        query_latent_out,
        query_rope_out,
        packed_cache,
        scale_cache,
        reciprocal_rope_cache,
        fault_status,
        zero_count,
        float(grid),
        bool(rotation_fused),
        bool(strict),
    )


def tq_mla_n10_native_frontend_rope_out(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    query_latent_out: torch.Tensor,
    query_rope_out: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    reciprocal_rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    zero_count: torch.Tensor,
    *,
    grid: float,
    rotation_fused: bool,
    strict: bool = False,
) -> None:
    """Fuse NeoX RoPE, separate query emission, and the N10 cache write."""

    _get_module().tq_mla_n10_native_frontend_rope_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        cos_sin_cache,
        positions,
        locations,
        signs1,
        signs2,
        query_latent_out,
        query_rope_out,
        packed_cache,
        scale_cache,
        reciprocal_rope_cache,
        fault_status,
        zero_count,
        float(grid),
        bool(rotation_fused),
        bool(strict),
    )


__all__ = [
    "tq_mla_n10_native_cache_writer_out",
    "tq_mla_n10_native_frontend_out",
    "tq_mla_n10_native_frontend_rope_out",
]
