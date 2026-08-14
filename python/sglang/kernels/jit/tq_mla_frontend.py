"""Native SM100 writer/front end for the N8 E2M1 MLA cache ABI.

All ``*_out`` entries are allocation-free after the extension is loaded and
launch on PyTorch's current stream.  The cache-only entry is owned by the KV
pool; the combined entry additionally emits the FP8 attention query and is
reserved for a format-aware native reader backend.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_MODULE: ModuleType | None = None
_MODULE_LOCK = threading.Lock()
_MODULE_NAME = "sglang_tq_mla_frontend_sm100_s5_n8_v4"


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
                / "tq_mla_frontend_sm100.cu"
            )
            _MODULE = load(
                name=_MODULE_NAME,
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "-lineinfo",
                    "-gencode=arch=compute_100,code=sm_100",
                ],
                verbose=False,
            )
    return _MODULE


def tq_mla_cache_writer_out(
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    boundaries: torch.Tensor,
    levels: torch.Tensor,
    storage_codes: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    grid: float,
    strict: bool = False,
) -> None:
    """Write exact N8 cache rows with one SM100 warp per source token.

    ``fault_status`` is process-lifetime sticky state.  Initialize it once and
    never clear it between graph replays.  Invalid locations set bit zero and
    return before any destination pointer is formed.  ``strict`` additionally
    traps and is intended only for isolated throwaway validation processes.
    """

    _get_module().tq_mla_cache_writer_out(
        cache_latent,
        cache_rope,
        locations,
        signs1,
        signs2,
        boundaries,
        levels,
        storage_codes,
        packed_cache,
        scale_cache,
        rope_cache,
        fault_status,
        float(grid),
        bool(strict),
    )


def tq_mla_frontend_out(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    boundaries: torch.Tensor,
    levels: torch.Tensor,
    storage_codes: torch.Tensor,
    query_out: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    grid: float,
    rotation_fused: bool,
    num_warps: int = 9,
    strict: bool = False,
) -> None:
    """Emit the FP8 query and exact N8 cache row in one SM100 launch."""

    _get_module().tq_mla_frontend_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        locations,
        signs1,
        signs2,
        boundaries,
        levels,
        storage_codes,
        query_out,
        packed_cache,
        scale_cache,
        rope_cache,
        fault_status,
        float(grid),
        bool(rotation_fused),
        int(num_warps),
        bool(strict),
    )


def tq_mla_frontend_rope_out(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    boundaries: torch.Tensor,
    levels: torch.Tensor,
    storage_codes: torch.Tensor,
    query_out: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    grid: float,
    rotation_fused: bool,
    num_warps: int = 9,
    strict: bool = False,
) -> None:
    """Fuse NeoX BF16 RoPE, FP8 query emission, and the exact N8 write.

    The internal RoPE result is explicitly rounded to BF16 before SATFINITE
    E4M3 conversion, matching the selected two-launch oracle without
    materializing either BF16 RoPE output tensor.
    """

    _get_module().tq_mla_frontend_rope_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        cos_sin_cache,
        positions,
        locations,
        signs1,
        signs2,
        boundaries,
        levels,
        storage_codes,
        query_out,
        packed_cache,
        scale_cache,
        rope_cache,
        fault_status,
        float(grid),
        bool(rotation_fused),
        int(num_warps),
        bool(strict),
    )


def tq_mla_frontend(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    boundaries: torch.Tensor,
    levels: torch.Tensor,
    storage_codes: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    grid: float,
    rotation_fused: bool,
    num_warps: int = 9,
    strict: bool = False,
) -> torch.Tensor:
    """Eager convenience wrapper that allocates only the FP8 query output."""

    query_out = torch.empty(
        (*query_latent.shape[:-1], 576),
        dtype=torch.float8_e4m3fn,
        device=query_latent.device,
    )
    tq_mla_frontend_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        locations,
        signs1,
        signs2,
        boundaries,
        levels,
        storage_codes,
        query_out,
        packed_cache,
        scale_cache,
        rope_cache,
        fault_status,
        grid=grid,
        rotation_fused=rotation_fused,
        num_warps=num_warps,
        strict=strict,
    )
    return query_out


__all__ = [
    "tq_mla_cache_writer_out",
    "tq_mla_frontend",
    "tq_mla_frontend_out",
    "tq_mla_frontend_rope_out",
]
