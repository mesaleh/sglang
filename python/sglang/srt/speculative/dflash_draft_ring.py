from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def compact_draft_seq_len(seq_len: int, window_size: int, page_size: int) -> int:
    """Exact page-preserving suffix length used by compact DFlash attention."""
    seq_len = int(seq_len)
    window_size = int(window_size)
    page_size = int(page_size)
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}.")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")

    visible_len = min(seq_len, window_size)
    visible_start = seq_len - visible_len
    aligned_start = visible_start - visible_start % page_size
    return seq_len - aligned_start


def max_compact_draft_seq_len(window_size: int, page_size: int) -> int:
    window_size = int(window_size)
    page_size = int(page_size)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")
    return window_size + page_size - 1


@dataclass(frozen=True)
class DFlashDraftRingConfig:
    window_size: int
    page_size: int
    block_size: int
    alloc_reserve: int
    request_rows: int
    max_compact_len: int
    max_live_pages: int
    ring_pages: int
    row_stride: int
    physical_tokens: int
    padded_tokens: int


def build_dflash_draft_ring_config(
    *,
    window_size: int,
    page_size: int,
    block_size: int,
    alloc_reserve: int,
    request_rows: int,
) -> DFlashDraftRingConfig:
    window_size = int(window_size)
    page_size = int(page_size)
    block_size = int(block_size)
    alloc_reserve = int(alloc_reserve)
    request_rows = int(request_rows)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if alloc_reserve < block_size:
        raise ValueError(
            "alloc_reserve must cover one DFlash block: "
            f"reserve={alloc_reserve}, block_size={block_size}."
        )
    if request_rows <= 0:
        raise ValueError(f"request_rows must be positive, got {request_rows}.")

    max_compact_len = max_compact_draft_seq_len(window_size, page_size)
    max_live_pages = _ceil_div(max_compact_len + alloc_reserve, page_size)
    ring_pages = max_live_pages + 1
    row_stride = ring_pages * page_size
    physical_tokens = request_rows * row_stride
    return DFlashDraftRingConfig(
        window_size=window_size,
        page_size=page_size,
        block_size=block_size,
        alloc_reserve=alloc_reserve,
        request_rows=request_rows,
        max_compact_len=max_compact_len,
        max_live_pages=max_live_pages,
        ring_pages=ring_pages,
        row_stride=row_stride,
        physical_tokens=physical_tokens,
        padded_tokens=physical_tokens + page_size,
    )


def draft_ring_cache_locs(
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    config: DFlashDraftRingConfig,
) -> torch.Tensor:
    """Map request-local absolute positions into the private physical ring.

    Slot zero's padded page is deliberately skipped. Each addressable request
    row owns a disjoint page-aligned segment; absolute page modulo preserves
    in-page offsets across wrap.
    """
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    if req_pool_indices.dtype != torch.int64:
        req_pool_indices = req_pool_indices.to(torch.int64)
    if positions.ndim == req_pool_indices.ndim + 1:
        req_pool_indices = req_pool_indices.unsqueeze(-1)
    try:
        req_pool_indices, positions = torch.broadcast_tensors(
            req_pool_indices, positions
        )
    except RuntimeError as exc:
        raise ValueError(
            "req_pool_indices and positions are not broadcast-compatible: "
            f"{tuple(req_pool_indices.shape)} vs {tuple(positions.shape)}."
        ) from exc

    if positions.device.type == "cpu":
        if bool(torch.any(positions < 0)):
            raise ValueError("draft ring positions must be non-negative.")
        if bool(torch.any(req_pool_indices < 0)) or bool(
            torch.any(req_pool_indices >= config.request_rows)
        ):
            raise ValueError(
                "draft ring request index is outside the configured rows: "
                f"request_rows={config.request_rows}."
            )

    page = torch.div(positions, config.page_size, rounding_mode="floor")
    in_page = torch.remainder(positions, config.page_size)
    ring_page = torch.remainder(page, config.ring_pages)
    return (
        config.page_size
        + req_pool_indices * config.row_stride
        + ring_page * config.page_size
        + in_page
    ).to(torch.int64)


@dataclass(frozen=True)
class DFlashPrefillSlice:
    flat_start: int
    flat_end: int
    absolute_start: int
    absolute_end: int

    @property
    def length(self) -> int:
        return self.flat_end - self.flat_start


def select_dflash_ring_prefill_slices(
    *,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    window_size: int,
    page_size: int,
) -> list[DFlashPrefillSlice]:
    """Select only newly-produced hidden rows still visible after prefill.

    A selected range is always shorter than one physical ring stride, so one
    indexed store can never contain duplicate modulo destinations.
    """
    if len(prefix_lens) != len(extend_lens):
        raise ValueError(
            "prefix_lens and extend_lens must have equal length: "
            f"{len(prefix_lens)} vs {len(extend_lens)}."
        )

    out: list[DFlashPrefillSlice] = []
    flat_cursor = 0
    max_compact_len = max_compact_draft_seq_len(window_size, page_size)
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        prefix_len = int(prefix_len)
        extend_len = int(extend_len)
        if prefix_len < 0 or extend_len < 0:
            raise ValueError(
                "prefix/extend lengths must be non-negative: "
                f"prefix={prefix_len}, extend={extend_len}."
            )
        seq_end = prefix_len + extend_len
        visible_len = compact_draft_seq_len(seq_end, window_size, page_size)
        visible_start = seq_end - visible_len
        absolute_start = max(prefix_len, visible_start)
        local_start = absolute_start - prefix_len
        selected_len = extend_len - local_start
        if selected_len > max_compact_len:
            raise AssertionError(
                "selected DFlash prefill suffix exceeds the compact bound: "
                f"selected={selected_len}, max={max_compact_len}."
            )
        out.append(
            DFlashPrefillSlice(
                flat_start=flat_cursor + local_start,
                flat_end=flat_cursor + extend_len,
                absolute_start=absolute_start,
                absolute_end=seq_end,
            )
        )
        flat_cursor += extend_len
    return out


def dflash_draft_ring_reprefill_tail_tokens(window_size: int, page_size: int) -> int:
    return max_compact_draft_seq_len(window_size, page_size)


def configured_dflash_draft_ring_reprefill_tail_tokens(server_args) -> int:
    """Return the radix-hit holdback required by the enabled draft ring."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_OMNIVA_DFLASH_DRAFT_RING.get():
        return 0

    algorithm = getattr(server_args, "speculative_algorithm", None)
    algorithm_name = getattr(algorithm, "value", algorithm)
    if str(algorithm_name).upper() != "DFLASH":
        return 0

    window_size = getattr(server_args, "speculative_draft_window_size", None)
    if window_size is None:
        raise RuntimeError(
            "SGLANG_OMNIVA_DFLASH_DRAFT_RING requires "
            "--speculative-draft-window-size."
        )
    return dflash_draft_ring_reprefill_tail_tokens(
        int(window_size), int(server_args.page_size)
    )


def actual_dflash_ring_live_page_span(
    *, prefix_len: int, window_size: int, page_size: int, block_size: int
) -> int:
    """Independent scalar oracle over actual committed+verify positions."""
    prefix_len = int(prefix_len)
    block_size = int(block_size)
    draft_len = compact_draft_seq_len(prefix_len, window_size, page_size)
    first_page = (prefix_len - draft_len) // page_size
    last_page = (prefix_len + block_size - 1) // page_size
    return last_page - first_page + 1
