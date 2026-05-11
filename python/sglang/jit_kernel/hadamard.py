from __future__ import annotations

import logging
import os
import traceback
from typing import TYPE_CHECKING, Callable

import torch

from sglang.jit_kernel.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)
_HADAMARD_DEBUG = os.environ.get("SGLANG_TQ_HADAMARD_DEBUG", "0") == "1"


def _rank_context() -> str:
    keys = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "SGLANG_TP_RANK",
        "SGLANG_TP_SIZE",
        "CUDA_VISIBLE_DEVICES",
        "SGLANG_TQ_MLA_FUSED_DECODE",
    )
    return ", ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in keys)


@cache_once
def _jit_hadamard_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    hadamard_include_dir = (KERNEL_PATH / "csrc" / "fast-hadamard-transform").resolve()
    return load_jit(
        "hadamard",
        *args,
        cuda_files=["fast-hadamard-transform/hadamard_jit.cuh"],
        cuda_wrappers=[
            ("hadamard_transform", f"HadamardKernel<{args}>::run"),
            ("hadamard_transform_with_signs", f"HadamardWithSignsKernel<{args}>::run"),
            ("hadamard_transform_12n", f"Hadamard12NKernel<{args}>::run"),
            ("hadamard_transform_20n", f"Hadamard20NKernel<{args}>::run"),
            ("hadamard_transform_28n", f"Hadamard28NKernel<{args}>::run"),
            ("hadamard_transform_40n", f"Hadamard40NKernel<{args}>::run"),
        ],
        extra_include_paths=[str(hadamard_include_dir)],
    )


def _hadamard_transform_impl(
    x: torch.Tensor,
    scale: float,
    pad_multiple: int,
    kernel_fn: Callable,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError(f"{kernel_fn.__name__} only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    needs_pad = dim_og % pad_multiple != 0
    if needs_pad:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    out = torch.empty_like(x)
    kernel_fn(x, out, scale)

    if needs_pad:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 8, module.hadamard_transform)


def hadamard_transform_with_signs(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused WHT rotation: out = signs2 * H(signs1 * x) * scale.

    Fuses signs1 multiply, Hadamard transform, and signs2 multiply into a
    single CUDA kernel launch, eliminating 2 elementwise kernel launches.

    Args:
        x: (..., dim) tensor, any dtype. Will be cast to float32 internally.
        signs1: (dim,) float32 sign vector applied before Hadamard.
        signs2: (dim,) float32 sign vector applied after Hadamard.
        scale: scalar multiplier (typically 1/sqrt(dim)).

    Returns:
        out: same shape as x, float32.
    """
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_with_signs only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)

    signs1_shape = tuple(signs1.shape)
    signs2_shape = tuple(signs2.shape)
    if _HADAMARD_DEBUG:
        logger.warning(
            "hadamard_transform_with_signs call: x_shape=%s x_dtype=%s "
            "x_device=%s signs1_shape=%s signs2_shape=%s scale=%s %s",
            tuple(shapes_og),
            x.dtype,
            x.device,
            signs1_shape,
            signs2_shape,
            scale,
            _rank_context(),
        )

    signs_mismatch = (
        signs1.dim() != 1
        or signs2.dim() != 1
        or signs1.numel() != dim_og
        or signs2.numel() != dim_og
    )
    if signs_mismatch:
        stack = "".join(traceback.format_stack(limit=12)[:-1])
        msg = (
            "hadamard_transform_with_signs sign/input dim mismatch: "
            f"x_shape={tuple(shapes_og)} x_dtype={x.dtype} x_device={x.device} "
            f"x_last_dim={dim_og} signs1_shape={signs1_shape} "
            f"signs2_shape={signs2_shape} scale={scale} {_rank_context()}\n"
            f"Python stack:\n{stack}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    # Use x's native dtype — the CUDA kernel handles bf16/fp16 I/O
    # with float32 computation internally (load converts to float,
    # store converts back to input_t)
    out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_with_signs(x, out, signs1, signs2, scale)

    return out.reshape(shapes_og)


def hadamard_transform_12n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 12, module.hadamard_transform_12n)


def hadamard_transform_20n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 20, module.hadamard_transform_20n)


def hadamard_transform_28n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 28, module.hadamard_transform_28n)


def hadamard_transform_40n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 40, module.hadamard_transform_40n)
