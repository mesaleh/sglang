import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    verified_id_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    verified_id_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    verified_id = tl.load(verified_id_ptr + row * verified_id_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, verified_id.to(tl.int64), block_ids)

    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols,
        block_ids,
        mask=row_mask,
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _prepare_dflash_draft_block_unchecked(
    *,
    verified_id: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(verified_id.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError("DFLASH prepare_block requires contiguous block_ids_out.")
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError("DFLASH prepare_block requires contiguous positions_out.")
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError("DFLASH prepare_block requires contiguous cache_loc_out.")

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        verified_id,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        verified_id.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=_pick_num_warps(block),
    )


@triton.jit
def _prepare_dflash_compact_draft_block_kernel(
    verified_id_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    target_req_to_token_ptr,
    draft_req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    draft_seq_lens_out_ptr,
    block_end_out_ptr,
    verified_id_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    target_req_to_token_row_stride,
    draft_req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    target_req_to_token_width: tl.constexpr,
    draft_req_to_token_width: tl.constexpr,
    window_size: tl.constexpr,
    page_size: tl.constexpr,
    block_size: tl.constexpr,
    max_compact_len: tl.constexpr,
    mask_token_id: tl.constexpr,
    BLOCK_COPY: tl.constexpr,
):
    row = tl.program_id(0)
    copy_block = tl.program_id(1)
    offsets = copy_block * BLOCK_COPY + tl.arange(0, BLOCK_COPY)

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    verified_id = tl.load(verified_id_ptr + row * verified_id_stride)

    visible_len = tl.minimum(prefix_len, window_size)
    if page_size > 1:
        visible_start = prefix_len - visible_len
        aligned_start = visible_start - (visible_start % page_size)
        draft_len = prefix_len - aligned_start
    else:
        aligned_start = prefix_len - visible_len
        draft_len = visible_len
    block_end = draft_len + block_size

    target_row = target_req_to_token_ptr + req_idx * target_req_to_token_row_stride
    draft_row = draft_req_to_token_ptr + req_idx * draft_req_to_token_row_stride

    in_draft_prefix = offsets < draft_len
    target_pos = tl.where(
        in_draft_prefix,
        aligned_start + offsets,
        prefix_len + (offsets - draft_len),
    )
    copy_mask = (
        (offsets < block_end)
        & (offsets < (max_compact_len + block_size))
        & (offsets < draft_req_to_token_width)
        & (target_pos >= 0)
        & (target_pos < target_req_to_token_width)
    )
    slot_ids = tl.load(target_row + target_pos, mask=copy_mask, other=0)
    tl.store(draft_row + offsets, slot_ids, mask=copy_mask)

    meta_mask = copy_block == 0
    tl.store(draft_seq_lens_out_ptr + row, draft_len.to(tl.int32), mask=meta_mask)
    tl.store(block_end_out_ptr + row, block_end.to(tl.int32), mask=meta_mask)

    block_mask = meta_mask & (offsets < block_size)
    block_pos = prefix_len + offsets
    block_slot_ids = tl.load(
        target_row + block_pos,
        mask=block_mask & (block_pos < target_req_to_token_width),
        other=0,
    )
    block_ids = tl.full((BLOCK_COPY,), mask_token_id, tl.int64)
    block_ids = tl.where(offsets == 0, verified_id.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + offsets,
        block_ids,
        mask=block_mask,
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + offsets,
        block_pos,
        mask=block_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + offsets,
        block_slot_ids.to(tl.int64),
        mask=block_mask,
    )


def _prepare_dflash_compact_draft_block_unchecked(
    *,
    verified_id: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    target_req_to_token: torch.Tensor,
    draft_req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    draft_seq_lens_out: torch.Tensor,
    block_end_out: torch.Tensor,
    window_size: int,
    page_size: int,
    max_compact_len: int,
    mask_token_id: int,
) -> None:
    batch_size = int(verified_id.numel())
    if batch_size == 0:
        return

    if target_req_to_token.ndim != 2 or target_req_to_token.stride(1) != 1:
        raise ValueError("DFLASH compact prepare requires row-major target req_to_token.")
    if draft_req_to_token.ndim != 2 or draft_req_to_token.stride(1) != 1:
        raise ValueError("DFLASH compact prepare requires row-major draft req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError("DFLASH compact prepare requires contiguous block_ids_out.")
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError("DFLASH compact prepare requires contiguous positions_out.")
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError("DFLASH compact prepare requires contiguous cache_loc_out.")
    if int(window_size) <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if int(page_size) <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")
    if int(max_compact_len) <= 0:
        raise ValueError(f"max_compact_len must be positive, got {max_compact_len}.")

    block_size = int(block_ids_out.shape[1])
    block_copy = 256
    copy_len = int(max_compact_len) + block_size
    _prepare_dflash_compact_draft_block_kernel[
        (batch_size, triton.cdiv(copy_len, block_copy))
    ](
        verified_id,
        prefix_lens,
        req_pool_indices,
        target_req_to_token,
        draft_req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        draft_seq_lens_out,
        block_end_out,
        verified_id.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        target_req_to_token.stride(0),
        draft_req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(target_req_to_token.shape[1]),
        int(draft_req_to_token.shape[1]),
        int(window_size),
        int(page_size),
        block_size,
        int(max_compact_len),
        int(mask_token_id),
        BLOCK_COPY=block_copy,
        num_warps=4,
    )
