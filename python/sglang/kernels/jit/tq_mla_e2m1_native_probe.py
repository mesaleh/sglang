"""Research-only SM100f native E2M1 encode/decode falsifier."""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

_MODULE: ModuleType | None = None
_MODULE_LOCK = threading.Lock()
_MODULE_NAME = "sglang_tq_mla_e2m1_native_probe_sm100f_a17_n10_w4_n0_v1"


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
                / "tq_mla_e2m1_native_probe_sm100f.cu"
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


def e2m1_native_probe(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw bytes, corrected N10 bytes, and decoded FP32 pairs."""

    raw, corrected, decoded = _get_module().e2m1_native_probe(values)
    return raw, corrected, decoded


__all__ = ["e2m1_native_probe"]
