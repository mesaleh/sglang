"""Native SM100 TurboQuant MLA query and cache-writer front end.

This is an experimental, default-off H43 operator. The capture-safe ``out``
entry point owns no allocations and writes only caller-provided tensors. Its
optional raw-FP8 codebook output is compile-time specialized in the CUDA
extension and requires the storage-order decode centroids.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import threading
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_MODULE: ModuleType | None = None
_MODULE_PREBUILT = False
_MODULE_LOCK = threading.Lock()
_MODULE_NAME = "sglang_tq_mla_frontend_sm100_h43_i3_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prebuilt_module() -> ModuleType | None:
    value = os.environ.get("SGLANG_TQ_MLA_FRONTEND_SO")
    expected = os.environ.get("SGLANG_TQ_MLA_FRONTEND_SO_SHA256")
    if value is None and expected is None:
        return None
    if (
        not value
        or not expected
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise RuntimeError("H43 native frontend prebuilt environment is incomplete")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve() != path
        or not path.is_file()
    ):
        raise RuntimeError("H43 native frontend prebuilt path is unsafe or missing")
    if _sha256(path) != expected:
        raise RuntimeError("H43 native frontend prebuilt digest mismatch")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("H43 native frontend prebuilt import spec is invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_module() -> ModuleType:
    global _MODULE, _MODULE_PREBUILT
    if _MODULE is not None:
        return _MODULE
    with _MODULE_LOCK:
        if _MODULE is None:
            _MODULE = _load_prebuilt_module()
            _MODULE_PREBUILT = _MODULE is not None
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


def preload_tq_mla_frontend_prebuilt() -> None:
    """Load the digest-pinned extension without permitting a runtime build."""

    global _MODULE, _MODULE_PREBUILT
    if _MODULE is not None:
        if not _MODULE_PREBUILT:
            raise RuntimeError(
                "H43 serving refuses an extension loaded by runtime compilation"
            )
        return
    with _MODULE_LOCK:
        if _MODULE is None:
            module = _load_prebuilt_module()
            if module is None:
                raise RuntimeError(
                    "H43 serving requires SGLANG_TQ_MLA_FRONTEND_SO and "
                    "SGLANG_TQ_MLA_FRONTEND_SO_SHA256"
                )
            _MODULE = module
            _MODULE_PREBUILT = True


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
    decode_centroids: torch.Tensor | None = None,
    codebook_cache: torch.Tensor | None = None,
    scale_multiplier: float,
    rotation_fused: bool,
    num_warps: int = 8,
    strict: bool = False,
) -> None:
    """Launch the allocation-free combined front end on the current stream.

    ``locations`` must be unique, matching SGLang's token-slot allocator
    contract. ``fault_status`` is process-lifetime sticky state: initialize it
    once before graph capture and never clear it per replay. The optional
    ``decode_centroids`` and ``codebook_cache`` arguments are all-or-nothing.
    """

    if (decode_centroids is None) != (codebook_cache is None):
        raise ValueError(
            "decode_centroids and codebook_cache must be provided together"
        )

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
        decode_centroids,
        codebook_cache,
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
    decode_centroids: torch.Tensor | None = None,
    codebook_cache: torch.Tensor | None = None,
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
        decode_centroids=decode_centroids,
        codebook_cache=codebook_cache,
        scale_multiplier=scale_multiplier,
        rotation_fused=rotation_fused,
        num_warps=num_warps,
        strict=strict,
    )
    return query_out


__all__ = [
    "preload_tq_mla_frontend_prebuilt",
    "tq_mla_frontend",
    "tq_mla_frontend_out",
]
