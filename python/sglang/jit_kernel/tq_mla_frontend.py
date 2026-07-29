"""Native SM100 TurboQuant MLA query and cache-writer front end.

This is an experimental, default-off H41 operator. The capture-safe ``out``
entry point owns no allocations and writes only caller-provided tensors.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_MODULE: ModuleType | None = None
_MODULE_LOCK = threading.Lock()
_MODULE_NAME = "sglang_tq_mla_frontend_sm100_h41_v1"


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


def tq_mla_frontend_out(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    boundaries: torch.Tensor,
    quant_centroids: torch.Tensor,
    storage_code_lut: torch.Tensor,
    query_out: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    scale_multiplier: float,
    rotation_fused: bool,
    num_warps: int = 8,
    strict: bool = False,
) -> None:
    """Launch the allocation-free combined front end on the current stream.

    ``locations`` must be unique, matching SGLang's token-slot allocator
    contract. ``fault_status`` is process-lifetime sticky state: initialize it
    once before graph capture and never clear it per replay.
    """

    _get_module().tq_mla_frontend_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        locations,
        signs1,
        signs2,
        boundaries,
        quant_centroids,
        storage_code_lut,
        query_out,
        packed_cache,
        scale_cache,
        rope_cache,
        fault_status,
        float(scale_multiplier),
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
    quant_centroids: torch.Tensor,
    storage_code_lut: torch.Tensor,
    packed_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    fault_status: torch.Tensor,
    *,
    scale_multiplier: float,
    rotation_fused: bool,
    num_warps: int = 8,
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
        quant_centroids,
        storage_code_lut,
        query_out,
        packed_cache,
        scale_cache,
        rope_cache,
        fault_status,
        scale_multiplier=scale_multiplier,
        rotation_fused=rotation_fused,
        num_warps=num_warps,
        strict=strict,
    )
    return query_out


__all__ = ["tq_mla_frontend", "tq_mla_frontend_out"]
