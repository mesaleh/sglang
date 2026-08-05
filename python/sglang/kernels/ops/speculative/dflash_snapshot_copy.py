from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

_BLOCK = 512


@triton.jit(
    do_not_specialize=[
        "snapshot_base",
        "ring_base",
        "ring_start_page",
        "valid_rows",
    ]
)
def _copy_dflash_snapshot_single_kernel(
    buffer_ptrs,
    snapshot_base,
    ring_base,
    ring_start_page,
    valid_rows,
    row_elements: tl.constexpr,
    page_size: tl.constexpr,
    ring_pages: tl.constexpr,
    snapshot_to_ring: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    buffer_id = tl.program_id(1)
    mask = element_offset < valid_rows * row_elements
    token_offset = element_offset // row_elements
    column = element_offset - token_offset * row_elements

    base = tl.load(buffer_ptrs + buffer_id).to(tl.pointer_type(tl.bfloat16))
    snapshot_token = snapshot_base + token_offset
    ring_token = (
        ring_base
        + ((ring_start_page + token_offset // page_size) % ring_pages) * page_size
        + token_offset % page_size
    )
    snapshot_offset = snapshot_token * row_elements + column
    ring_offset = ring_token * row_elements + column
    if snapshot_to_ring:
        value = tl.load(base + snapshot_offset, mask=mask)
        tl.store(base + ring_offset, value, mask=mask)
    else:
        value = tl.load(base + ring_offset, mask=mask)
        tl.store(base + snapshot_offset, value, mask=mask)


@triton.jit(
    do_not_specialize=[
        "packed0",
        "packed1",
        "packed2",
        "packed3",
        "packed4",
        "packed5",
        "packed6",
        "packed7",
    ]
)
def _copy_dflash_snapshot_packed_kernel(
    buffer_ptrs,
    packed0,
    packed1,
    packed2,
    packed3,
    packed4,
    packed5,
    packed6,
    packed7,
    snapshot_base: tl.constexpr,
    snapshot_rows: tl.constexpr,
    row_stride: tl.constexpr,
    row_elements: tl.constexpr,
    page_size: tl.constexpr,
    ring_pages: tl.constexpr,
    snapshot_to_ring: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    buffer_id = tl.program_id(1)
    request_id = tl.program_id(2)

    packed = packed0
    packed = tl.where(request_id == 1, packed1, packed)
    packed = tl.where(request_id == 2, packed2, packed)
    packed = tl.where(request_id == 3, packed3, packed)
    packed = tl.where(request_id == 4, packed4, packed)
    packed = tl.where(request_id == 5, packed5, packed)
    packed = tl.where(request_id == 6, packed6, packed)
    packed = tl.where(request_id == 7, packed7, packed)
    slot = packed & 31
    request_pool_index = (packed >> 5) & 15
    ring_start_page = (packed >> 9) & 127
    valid_rows = (packed >> 16) & 4095

    mask = element_offset < valid_rows * row_elements
    token_offset = element_offset // row_elements
    column = element_offset - token_offset * row_elements
    base = tl.load(buffer_ptrs + buffer_id).to(tl.pointer_type(tl.bfloat16))
    snapshot_token = snapshot_base + slot * snapshot_rows + token_offset
    ring_token = (
        page_size
        + request_pool_index * row_stride
        + ((ring_start_page + token_offset // page_size) % ring_pages) * page_size
        + token_offset % page_size
    )
    snapshot_offset = snapshot_token * row_elements + column
    ring_offset = ring_token * row_elements + column
    if snapshot_to_ring:
        value = tl.load(base + snapshot_offset, mask=mask)
        tl.store(base + ring_offset, value, mask=mask)
    else:
        value = tl.load(base + ring_offset, mask=mask)
        tl.store(base + snapshot_offset, value, mask=mask)


def copy_dflash_snapshot(
    *,
    buffer_ptrs: torch.Tensor,
    snapshot_base: int,
    ring_base: int,
    ring_start_page: int,
    valid_rows: int,
    row_elements: int,
    page_size: int,
    ring_pages: int,
    snapshot_to_ring: bool,
) -> None:
    """Copy one snapshot without allocating per-call device metadata tensors."""

    if buffer_ptrs.ndim != 1 or buffer_ptrs.numel() == 0:
        raise ValueError("buffer_ptrs must be a non-empty 1D tensor")
    if buffer_ptrs.dtype not in (torch.int64, torch.uint64):
        raise ValueError("buffer_ptrs must use int64 or uint64 addresses")
    if valid_rows <= 0:
        return
    if row_elements <= 0 or page_size <= 0 or ring_pages <= 0:
        raise ValueError("row_elements, page_size, and ring_pages must be positive")

    grid = (triton.cdiv(valid_rows * row_elements, _BLOCK), buffer_ptrs.numel())
    _copy_dflash_snapshot_single_kernel[grid](
        buffer_ptrs,
        int(snapshot_base),
        int(ring_base),
        int(ring_start_page),
        int(valid_rows),
        row_elements=row_elements,
        page_size=page_size,
        ring_pages=ring_pages,
        snapshot_to_ring=snapshot_to_ring,
        BLOCK=_BLOCK,
        num_warps=8,
    )


def copy_dflash_snapshot_batch_packed(
    *,
    buffer_ptrs: torch.Tensor,
    packed_metadata: Sequence[int],
    max_rows: int,
    snapshot_base: int,
    snapshot_rows: int,
    row_stride: int,
    row_elements: int,
    page_size: int,
    ring_pages: int,
    snapshot_to_ring: bool,
) -> None:
    """Copy two to eight snapshots with metadata passed as launch scalars."""

    batch_size = len(packed_metadata)
    if batch_size < 2 or batch_size > 8:
        raise ValueError("packed snapshot batches require two to eight requests")
    if buffer_ptrs.ndim != 1 or buffer_ptrs.numel() == 0:
        raise ValueError("buffer_ptrs must be a non-empty 1D tensor")
    if buffer_ptrs.dtype not in (torch.int64, torch.uint64):
        raise ValueError("buffer_ptrs must use int64 or uint64 addresses")
    if (
        min(
            max_rows,
            snapshot_rows,
            row_stride,
            row_elements,
            page_size,
            ring_pages,
        )
        <= 0
    ):
        raise ValueError("snapshot copy dimensions must be positive")
    padded = [int(value) for value in packed_metadata] + [0] * (8 - batch_size)
    grid = (
        triton.cdiv(max_rows * row_elements, _BLOCK),
        buffer_ptrs.numel(),
        batch_size,
    )
    _copy_dflash_snapshot_packed_kernel[grid](
        buffer_ptrs,
        *padded,
        snapshot_base=snapshot_base,
        snapshot_rows=snapshot_rows,
        row_stride=row_stride,
        row_elements=row_elements,
        page_size=page_size,
        ring_pages=ring_pages,
        snapshot_to_ring=snapshot_to_ring,
        BLOCK=_BLOCK,
        num_warps=8,
    )
