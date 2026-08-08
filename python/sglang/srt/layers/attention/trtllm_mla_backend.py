"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.pad import (
    pad_draft_extend_query as pad_draft_extend_query_triton,
)
from sglang.kernels.ops.attention.pad import (
    unpad_draft_extend_output as unpad_draft_extend_output_triton,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q,
    can_use_set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    covered as set_mla_kv_concat_q_covered,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    covered_fp8 as set_mla_kv_concat_q_fp8_covered,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    set_mla_kv_concat_q,
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.unified_mem_hooks import unified_mla_hooks
from sglang.srt.layers.attention.verify_mask import VerifyMask, maybe_create_verify_mask
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_buffer,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

# Constants
DEFAULT_WORKSPACE_SIZE_MB = 150  # Memory workspace size in MB

# Block constraint from flashinfer requirements
# From flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0
# This imposes that the total number of blocks must be divisible by
# (128 / block_size). We capture the 128 constant here so we can
# compute the LCM with other padding constraints.
TRTLLM_BLOCK_CONSTRAINT = 128

TRTLLM_MLA_MAX_BATCH_SIZE = 8192


def _multi_ctas_kv_counter_bytes(
    device: torch.device, num_q_heads: int, batch_size: int
) -> int:
    sm_count = flashinfer.utils.get_device_sm_count(device)
    return flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch_size, num_q_heads, sm_count
    )


def make_persistent_multi_ctas_kv_counter_buffer(
    device: torch.device, num_q_heads: int, max_batch_size: int
) -> torch.Tensor:
    num_bytes = _multi_ctas_kv_counter_bytes(
        device, num_q_heads, max(TRTLLM_MLA_MAX_BATCH_SIZE, max_batch_size)
    )
    return torch.zeros(num_bytes, dtype=torch.uint8, device=device)


def grow_multi_ctas_kv_counter_buffer_if_needed(
    buffer: torch.Tensor, device: torch.device, num_q_heads: int, batch_size: int
) -> torch.Tensor:
    required_bytes = _multi_ctas_kv_counter_bytes(device, num_q_heads, batch_size)
    if buffer.numel() >= required_bytes:
        return buffer
    return torch.zeros(required_bytes, dtype=torch.uint8, device=device)


@triton.jit
def pad_draft_extend_query_kernel(
    q_ptr,  # Input query tensor [total_seq_len, num_heads, head_dim]
    padded_q_ptr,  # Output padded query tensor [batch_size, max_seq_len, num_heads, head_dim]
    seq_lens_q_ptr,  # Sequence lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of sequence lengths [batch_size + 1]
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing."""
    # Use 3D program IDs: (batch_seq, head_block, dim_block)
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    # Load sequence length for this batch
    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return

    # Load cumulative sum to get start position in input tensor
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset
    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    # Calculate output offset
    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  # Input raw output tensor (batch_size, token_per_batch, tp_q_head_num, v_head_dim)
    output_ptr,  # Output tensor (-1, tp_q_head_num, v_head_dim)
    num_accept_tokens_ptr,  # Accept lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]
    batch_size,
    token_per_batch,
    tp_q_head_num,
    v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing."""
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    accept_len = tl.load(num_accept_tokens_ptr + batch_id)

    if seq_pos >= accept_len:
        return

    # Load cumulative sum to get start position in output tensor
    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset: (batch_id, seq_pos, head_id, dim_id)
    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def build_draft_frontier_page_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    kv_indptr_ptr,
    block_tables_ptr,
    seq_lens_out_ptr,
    req_to_token_stride: tl.constexpr,
    block_table_stride: tl.constexpr,
    page_size: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_table_width: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    cand_id = tl.program_id(0)
    page_block_id = tl.program_id(1)

    req_id = cand_id // topk
    topk_id = cand_id - req_id * topk
    req_pool_id = tl.load(req_pool_indices_ptr + req_id)
    token_pool = req_to_token_ptr + req_pool_id * req_to_token_stride

    prefix_len = tl.load(prefix_lens_ptr + cand_id).to(tl.int64)
    kv_start = tl.load(kv_indptr_ptr + cand_id).to(tl.int64)
    kv_end = tl.load(kv_indptr_ptr + cand_id + 1).to(tl.int64)
    token_len = kv_end - kv_start
    draft_len = token_len - prefix_len

    full_prefix_pages = prefix_len // page_size
    last_page_len = prefix_len - full_prefix_pages * page_size
    branch_pages = tl.cdiv(last_page_len + draft_len, page_size)

    num_new_pages_per_topk = tl.cdiv(last_page_len + speculative_num_steps, page_size)
    prefix_base = full_prefix_pages * page_size
    branch_base = prefix_base + topk_id * num_new_pages_per_topk * page_size

    slots = page_block_id * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    in_width = slots < page_table_width
    table_ptr = block_tables_ptr + cand_id * block_table_stride + slots

    page_ids = tl.full((BLOCK_PAGES,), -1, tl.int32)
    prefix_mask = slots < full_prefix_pages
    branch_slot = slots - full_prefix_pages
    branch_mask = (branch_slot >= 0) & (branch_slot < branch_pages)

    prefix_token_pos = slots * page_size
    prefix_token_ids = tl.load(
        token_pool + prefix_token_pos,
        mask=in_width & prefix_mask,
        other=0,
    )
    prefix_page_ids = (prefix_token_ids // page_size).to(tl.int32)

    branch_token_pos = branch_base + branch_slot * page_size
    branch_token_ids = tl.load(
        token_pool + branch_token_pos,
        mask=in_width & branch_mask,
        other=0,
    )
    branch_page_ids = (branch_token_ids // page_size).to(tl.int32)

    page_ids = tl.where(prefix_mask, prefix_page_ids, page_ids)
    page_ids = tl.where(branch_mask, branch_page_ids, page_ids)
    tl.store(table_ptr, page_ids, mask=in_width)

    if page_block_id == 0:
        tl.store(seq_lens_out_ptr + cand_id, token_len.to(tl.int32))


def _quantize_fp8_qkv(q, k, v, layer):
    q = q.to(torch.float8_e4m3fn)

    k_scale = getattr(layer, "k_scale_float", None)
    if k_scale is None:
        k_scale = 1.0
    if k_scale != 1.0:
        assert hasattr(layer, "k_scale"), "k_scale is not set"
        k_2d, _ = scaled_fp8_quant(
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale
        )
        k = k_2d.reshape(k.shape)
    else:
        k = k.to(torch.float8_e4m3fn)

    v_scale = getattr(layer, "v_scale_float", None)
    if v_scale is None:
        v_scale = 1.0
    if v_scale != 1.0:
        assert hasattr(layer, "v_scale"), "v_scale is not set"
        v_2d, _ = scaled_fp8_quant(
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale
        )
        v = v_2d.reshape(v.shape)
    else:
        v = v.to(torch.float8_e4m3fn)

    return q, k, v, k_scale, v_scale


# cute-dsl needs its own workspace: it overwrites the buffer with split-KV
# partials, which corrupts the trtllm-gen multiCtasKv counters that rely on the
# zero-init buffer (they share it under attention-backend=cutedsl_mla, where
# draft-extend falls back to trtllm-gen) and deadlocks the reduction.
global_cute_dsl_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations."""

    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    fallback_to_flashinfer_impl: bool = False


from sglang.kernels.jit.utils import is_arch_support_pdl

# Arm PDL on the trtllm-gen decode launch so its prolog overlaps the tail of
# the query-prep kernels (which already trigger their PDL secondary).
_ENABLE_PDL = is_arch_support_pdl()


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""

    block_kv_indices: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    max_seq_len_q: Optional[int] = None
    sum_seq_lens_q: Optional[int] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    seq_lens_q: Optional[torch.Tensor] = None
    seq_lens_k: Optional[torch.Tensor] = None
    custom_mask: Optional[torch.Tensor] = None
    custom_mask_offsets: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None
    is_draft_frontier: bool = False
    global_seq_lens_k: Optional[torch.Tensor] = None


def _target_verify_max_seq_len(metadata: TRTLLMMLADecodeMetadata) -> int:
    """Return the final KV bound recorded by target-verify metadata.

    Both eager and CUDA-graph metadata store ``seq_lens_k`` after draft tokens
    have been appended. The matching max bound must therefore be consumed
    directly rather than adding the draft width again.
    """

    if metadata.max_seq_len_k is None:
        raise RuntimeError("target-verify MLA metadata is missing max_seq_len_k")
    return int(metadata.max_seq_len_k)


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""

    # trtllm-gen kernels rebuild metadata from preallocated buffers and never
    # read seq_lens_cpu / seq_lens_sum; opt out of the D2H sync.
    needs_cpu_seq_lens: bool = False
    supports_custom_decode_mask: bool = False

    # Ragged verify: the packed query is front-aligned into the dense
    # [bs, draft_token_num] layout in forward_extend; metadata stays uniform.
    supports_ragged_verify_graph: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
        backend: str = "trtllm-gen",
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config

        # Model parameters
        self.num_q_heads = config.num_attention_heads // get_parallel().attn_tp_size
        self.num_kv_heads = config.get_num_kv_heads(get_parallel().attn_tp_size)
        self.num_local_heads = config.num_attention_heads // get_parallel().attn_tp_size

        # MLA-specific dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Runtime parameters
        self.backend = backend
        self.scaling = config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Workspace allocation
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        if self.backend == "cute-dsl":
            # Separate buffer from trtllm-gen (see note above); safe to share
            # among cute-dsl instances.
            global global_cute_dsl_workspace_buffer
            if global_cute_dsl_workspace_buffer is None:
                global_cute_dsl_workspace_buffer = torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_cute_dsl_workspace_buffer
        else:
            self.workspace_buffer = get_buffer(
                "trtllm_mla_zero_workspace",
                lambda: torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                ),
            )

        self._multi_ctas_kv_counter_buffer = (
            make_persistent_multi_ctas_kv_counter_buffer(
                torch.device(self.device),
                self.num_q_heads,
                max_batch_size=model_runner.max_running_requests,
            )
        )

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = get_schedule().disable_chunked_prefix_cache

        self.num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.cuda_graph_custom_mask = None
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_topk = get_spec().speculative_eagle_topk
        self._verify_mask = None
        # Tree-mask scratch is fetched from the target backend only.
        self.is_draft_runner = model_runner.is_draft_worker

        # Unified-memory dense-view hooks (None on the static pool). req_to_token
        # holds VIRTUAL token ids; the block table needs DENSE page ids, so the
        # kv-index kernels gather virtual->physical page through `_v2p_page_table`
        # then scale by `_kernel_page_multiplier` (= num MLA layers). See
        # build_dense_mla_views / create_flashmla_kv_indices_triton.
        _hooks = unified_mla_hooks(model_runner.token_to_kv_pool_allocator)
        self._v2p_page_table = _hooks.v2p_page_table
        self._kernel_page_multiplier = _hooks.kernel_page_multiplier
        self._unified_mla = _hooks.enabled
        # virtual token id -> DENSE kernel-facing id, for the KV write loc.
        self._translate_kv_loc_dense = _hooks.translate_kv_loc_dense
        # Per-forward dense write loc ([:n] view of a capture-stable buffer),
        # set by the cuda-graph out-graph hook; None on the eager path (where the
        # write translates through the pool's _full_translate hook instead).
        self._decode_dense_loc: Optional[torch.Tensor] = None
        self.cuda_graph_out_cache_loc_dense: Optional[torch.Tensor] = None
        # Fused KV-scatter + q-concat on the decode dense-loc path (one launch
        # instead of set_mla_kv_buffer + concat_mla_absorb_q). Disabled under
        # async asserts: the fused path writes the pool directly and would
        # skip the pool's OOB probe.
        self._fused_set_kv_concat_q = (
            self.data_type == torch.bfloat16
            and not envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
            and can_use_set_mla_kv_concat_q(
                self.kv_lora_rank * 2, self.qk_rope_head_dim * 2
            )
        )
        # fp8 sibling: quantize + KV scatter + q concat in one launch
        # (replaces mla_quantize_without_rope_for_fp8's concat + three aten
        # casts plus the KV-row write on the fp8 decode path).
        self._fused_set_kv_concat_q_fp8 = (
            self.data_type == torch.float8_e4m3fn
            and not envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and can_use_set_mla_kv_concat_q_fp8()
        )

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.

        Args:
            max_seq_len: Maximum sequence length in tokens

        Returns:
            Number of blocks padded to satisfy all constraints
        """
        blocks = triton.cdiv(max_seq_len, self.page_size)

        # Apply dual constraints (take LCM to satisfy both):
        # 1. TRT-LLM: block_num % (128 / page_size) == 0
        # 2. Triton: number of pages per block
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(self.page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.

        Args:
            batch_size: Batch size
            max_blocks: Maximum number of blocks per sequence
            req_pool_indices: Request pool indices
            seq_lens: Sequence lengths
            device: Target device

        Returns:
            Block KV indices tensor
        """
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        create_flashmla_kv_indices_triton[
            (
                batch_size,
                get_num_kv_index_blocks_flashmla(max_blocks, self.page_size),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=self.page_size,
            v2p_ptr=self._v2p_page_table,
            PAGE_MULT=self._kernel_page_multiplier,
        )

        return block_kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""

        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        # Draft tree frontiers flatten to bs*topk query rows. Allocate enough rows
        # for either regular decode/verify (max_bs) or draft frontier decode
        # (max_num_tokens = max_bs*topk in the EAGLE draft graph runner).
        max_decode_rows = max(max_bs, max_num_tokens)
        self.decode_cuda_graph_kv_indices = torch.full(
            (max_decode_rows, max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_cuda_graph_seq_lens_k = torch.zeros(
            (max_decode_rows,), dtype=torch.int32, device=self.device
        )
        if self.supports_custom_decode_mask:
            self.decode_cuda_graph_custom_mask = torch.zeros(
                (max_decode_rows * max_blocks_per_seq * self.page_size,),
                dtype=torch.bool,
                device=self.device,
            )
            self.decode_cuda_graph_custom_mask_offsets = torch.zeros(
                (max_decode_rows,), dtype=torch.int32, device=self.device
            )
        else:
            self.decode_cuda_graph_custom_mask = None
            self.decode_cuda_graph_custom_mask_offsets = None
        # Unified pool: capture-stable buffer for the DENSE KV write loc, filled
        # out-of-graph in init_forward_metadata_out_graph so the in-graph
        # set_mla_kv_buffer captures no translate.
        if self._unified_mla:
            self.cuda_graph_out_cache_loc_dense = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=self.device
            )
        num_tokens_per_req = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_req // 2, self.num_q_heads, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens // 2, self.num_q_heads, 512),
                dtype=self.store_dtype,
                device=self.device,
            )
        else:
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_req, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )

        # Target verify never reaches the parent's mask read: it is excluded from
        # every super() dispatch (init_forward_metadata, _out_graph, forward_extend)
        # and runs the trtllm-gen kernel, which takes no mask.
        self._verify_mask = maybe_create_verify_mask(
            is_draft_runner=self.is_draft_runner,
            skip_prefill=self.skip_prefill,
            max_bs=max_bs,
            max_context_len=self.max_context_len,
            num_draft_tokens=self.num_draft_tokens,
            device=self.device,
            is_read=False,
        )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self._verify_mask

    @staticmethod
    def _is_draft_frontier_spec(
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        request_bs: int,
    ) -> bool:
        if not forward_mode.is_decode_or_idle() or spec_info is None:
            return False
        kv_indptr = getattr(spec_info, "kv_indptr", None)
        kv_indices = getattr(spec_info, "kv_indices", None)
        return (
            kv_indptr is not None
            and kv_indices is not None
            and (kv_indptr.shape[0] - 1) > request_bs
        )

    def _build_draft_frontier_page_metadata(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        device: torch.device,
        *,
        block_kv_indices_buf: Optional[torch.Tensor] = None,
        seq_lens_k_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        custom_mask_offsets_buf: Optional[torch.Tensor] = None,
        page_table_width: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Translate token-level EAGLE draft paths into page metadata.

        The EAGLE topk draft backend encodes each candidate as an ordered list of
        physical token slots. FlashInfer consumes that with page_size=1. The
        tokenspeed MLA kernel needs page-sized TMA tiles. For page-size > 1 EAGLE
        already duplicates the last partial page per topk branch; use that branch
        page in place of the original partial page so the logical sequence remains
        contiguous and the decode can stay on the fast causal path without a
        whole-history custom mask.
        """

        del kv_indices, custom_mask_buf, custom_mask_offsets_buf
        num_cand = kv_indptr.shape[0] - 1
        if prefix_lens is None or prefix_lens.numel() < num_cand:
            raise RuntimeError(
                "EAGLE draft frontier page metadata requires per-candidate "
                f"prefix positions; got {None if prefix_lens is None else prefix_lens.shape}, "
                f"need at least {num_cand}."
            )
        if page_table_width is None:
            page_table_width = self._calc_padded_blocks(self.max_context_len)

        topk = int(self.speculative_topk or 1)
        if topk <= 1 or num_cand % topk != 0:
            raise RuntimeError(
                "EAGLE draft frontier page metadata expected candidates to be "
                f"a positive multiple of topk; got num_cand={num_cand}, topk={topk}."
            )
        request_bs = num_cand // topk
        if req_pool_indices.numel() < request_bs:
            raise RuntimeError(
                "EAGLE draft frontier req_pool_indices too small: "
                f"need {request_bs}, got {req_pool_indices.numel()}."
            )

        if block_kv_indices_buf is None:
            block_kv_indices = torch.empty(
                (num_cand, page_table_width), dtype=torch.int32, device=device
            )
        else:
            if (
                num_cand > block_kv_indices_buf.shape[0]
                or page_table_width > block_kv_indices_buf.shape[1]
            ):
                raise RuntimeError(
                    "EAGLE draft frontier block table buffer too small: "
                    f"need ({num_cand}, {page_table_width}), "
                    f"got {tuple(block_kv_indices_buf.shape)}"
                )
            block_kv_indices = block_kv_indices_buf[:num_cand, :page_table_width]

        if seq_lens_k_buf is None:
            seq_lens_k = torch.empty((num_cand,), dtype=torch.int32, device=device)
        else:
            if num_cand > seq_lens_k_buf.shape[0]:
                raise RuntimeError(
                    "EAGLE draft frontier seq_lens buffer too small: "
                    f"need {num_cand}, got {seq_lens_k_buf.shape[0]}"
                )
            seq_lens_k = seq_lens_k_buf[:num_cand]

        block_pages = 128
        build_draft_frontier_page_table_kernel[
            (num_cand, triton.cdiv(page_table_width, block_pages))
        ](
            self.req_to_token,
            req_pool_indices[:request_bs],
            prefix_lens[:num_cand],
            kv_indptr,
            block_kv_indices,
            seq_lens_k,
            self.req_to_token.stride(0),
            block_kv_indices.stride(0),
            self.page_size,
            topk,
            int(self.speculative_num_steps),
            page_table_width,
            BLOCK_PAGES=block_pages,
        )

        return (
            block_kv_indices,
            seq_lens_k,
            None,
            None,
            self.max_context_len,
        )

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        device: torch.device,
        req_pool_indices: Optional[torch.Tensor] = None,
        spec_info: Optional[SpecInput] = None,
    ):
        """Allocate persistent metadata buffers for CUDA graph capture."""
        if forward_mode.is_target_verify() and bs in self.decode_cuda_graph_metadata:
            # Token tiers at the same slot count must share one per-bs buffer
            # set (each graph bakes in the tensors it captured).
            self.forward_decode_metadata = self.decode_cuda_graph_metadata[bs]
            return
        metadata = TRTLLMMLADecodeMetadata()

        if forward_mode.is_target_verify():
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)
            metadata.global_seq_lens_k = torch.zeros(
                (bs,), dtype=torch.int32, device=device
            )
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_req + 1,
                num_tokens_per_req,
                dtype=torch.int32,
                device=device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_req, dtype=torch.int32, device=device
            )
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)

        if self._is_draft_frontier_spec(forward_mode, spec_info, bs):
            if req_pool_indices is None:
                raise RuntimeError(
                    "EAGLE draft frontier metadata capture requires req_pool_indices."
                )
            if not self.supports_custom_decode_mask:
                raise RuntimeError(
                    "EAGLE topk>1 MLA draft frontier requires a decode backend "
                    "with custom-mask support."
                )
            max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
            (
                block_kv_indices,
                seq_lens_k,
                custom_mask,
                custom_mask_offsets,
                max_seq_len_k,
            ) = self._build_draft_frontier_page_metadata(
                spec_info.kv_indptr,
                spec_info.kv_indices,
                getattr(spec_info, "positions", None),
                req_pool_indices,
                device,
                block_kv_indices_buf=self.decode_cuda_graph_kv_indices,
                seq_lens_k_buf=self.decode_cuda_graph_seq_lens_k,
                custom_mask_buf=self.decode_cuda_graph_custom_mask,
                custom_mask_offsets_buf=self.decode_cuda_graph_custom_mask_offsets,
                page_table_width=max_blocks_per_seq,
            )
            metadata.block_kv_indices = block_kv_indices
            metadata.seq_lens_k = seq_lens_k
            metadata.custom_mask = custom_mask
            metadata.custom_mask_offsets = custom_mask_offsets
            metadata.max_seq_len_k = max_seq_len_k
            metadata.batch_size = spec_info.kv_indptr.shape[0] - 1
            metadata.is_draft_frontier = True

            self.decode_cuda_graph_metadata[bs] = metadata
            self.forward_decode_metadata = metadata
            return

        if metadata.seq_lens_k is None:
            # Plain decode: static int32 seq_lens buffer, refreshed by the
            # capture+replay body below (same pattern as target-verify).
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Capture with full width so future longer sequences are safe during replay.
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]
        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len
        metadata.batch_size = bs

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput] = None,
    ):
        """Shared decode / target-verify / draft-extend capture+replay body.

        Public entry: :py:meth:`init_forward_metadata_out_graph` (which routes
        the non-decode-family modes to the FlashInferMLA parent).
        """
        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_target_verify():
            # Intentional int64 -> int32 same-kind out= downcast.
            torch.add(
                seq_lens[:bs],
                self.num_draft_tokens,
                out=metadata.global_seq_lens_k,
            )
            metadata.seq_lens_k.copy_(metadata.global_seq_lens_k)
            seq_lens = metadata.seq_lens_k
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_req = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_req
            metadata.sum_seq_lens_q = num_tokens_per_req * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)
        elif metadata.seq_lens_k is not None:
            # Plain decode: int64 -> int32 downcast copy into the static
            # buffer (once per step, replacing the per-layer conversion).
            metadata.seq_lens_k.copy_(seq_lens[:bs])

        if self._is_draft_frontier_spec(forward_mode, spec_info, bs):
            if not self.supports_custom_decode_mask:
                raise RuntimeError(
                    "EAGLE topk>1 MLA draft frontier requires a decode backend "
                    "with custom-mask support."
                )
            max_seq_len_k = self._build_draft_frontier_page_metadata(
                spec_info.kv_indptr,
                spec_info.kv_indices,
                getattr(spec_info, "positions", None),
                req_pool_indices,
                seq_lens.device,
                block_kv_indices_buf=metadata.block_kv_indices,
                seq_lens_k_buf=metadata.seq_lens_k,
                custom_mask_buf=metadata.custom_mask,
                custom_mask_offsets_buf=metadata.custom_mask_offsets,
                page_table_width=metadata.block_kv_indices.shape[1],
            )[-1]
            metadata.max_seq_len_k = max_seq_len_k
            return

        # Update block indices for new sequences.
        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    metadata.block_kv_indices.shape[1], self.page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=self.page_size,
            v2p_ptr=self._v2p_page_table,
            PAGE_MULT=self._kernel_page_multiplier,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ) -> None:
        has_prefix = any(forward_batch.extend_prefix_lens_cpu)
        fallback_to_flashinfer_impl = (
            self.disable_chunked_prefix_cache and has_prefix
        ) or is_in_tc_piecewise_cuda_graph()
        if fallback_to_flashinfer_impl:
            super().init_mha_chunk_metadata(
                forward_batch, disable_flashinfer_ragged=True
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        forward_mode = forward_batch.forward_mode

        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend_v2()
        ):
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

        bs = forward_batch.batch_size
        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._init_cuda_graph_metadata(
                bs,
                num_tokens,
                forward_mode,
                forward_batch.seq_lens,
                forward_batch.seq_lens.device,
                req_pool_indices=forward_batch.req_pool_indices,
                spec_info=forward_batch.spec_info,
            )
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
                spec_info=forward_batch.spec_info,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
                spec_info=forward_batch.spec_info,
            )

        # Unified pool: precompute the DENSE KV write loc into the capture-stable
        # buffer (both capture and each replay-prep run this out of the graph),
        # so the in-graph set_mla_kv_buffer writes a dense loc without capturing a
        # translate. Only decode writes KV under unified (spec is gated off).
        if self._unified_mla and forward_mode.is_decode_or_idle():
            out_cache_loc = forward_batch.out_cache_loc
            n = out_cache_loc.shape[0]
            dst = self.cuda_graph_out_cache_loc_dense[:n]
            self._translate_kv_loc_dense(out_cache_loc, out=dst)
            # Replay-prep receives the RAW (unpadded) out_cache_loc
            # (build_replay_fb_view), but the captured write kernel consumes the
            # full captured tier of this buffer. Zero the tail so pad rows write
            # to the dense sink (row 0) instead of stale dense locs left by
            # earlier larger replays — a stale tail scatters pad-row garbage into
            # live KV pages. Mirrors the runner's PaddingPolicy.ZERO on its own
            # out_cache_loc slot.
            self.cuda_graph_out_cache_loc_dense[n:].zero_()
            self._decode_dense_loc = dst
        else:
            self._decode_dense_loc = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        # Eager path: no capture-stable dense write loc; the pool's _full_translate
        # hook translates the write loc (safe out of a cuda graph).
        self._decode_dense_loc = None
        # Delegate to parent for non-decode modes.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # For extend batch with prefix length > 0, fallback to ragged kernel implemented in flashinfer MLA backend
            # when chunked prefix cache is disabled.
            # Also fallback to flashinfer MLA backend when in piecewise cuda graph, since it only supports MLA forward mode.
            has_prefix = any(forward_batch.extend_prefix_lens_cpu)
            fallback_to_flashinfer_impl = (
                self.disable_chunked_prefix_cache and has_prefix
            ) or is_in_tc_piecewise_cuda_graph()
            if fallback_to_flashinfer_impl:
                super().init_forward_metadata(forward_batch)

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            cum_seq_lens_q = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device
                    ),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
                fallback_to_flashinfer_impl,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            bs = forward_batch.batch_size
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()
            # This is necessary because the backend instance persists across forward passes,
            # and forward_prefill_metadata from a previous regular extend call could still be set.
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                self.forward_prefill_metadata = None
            # Never read max_seq from the GPU tensor (.max().item() blocks the
            # host on the stream backlog); max_seq only sizes the block table /
            # scheduling hint, so the static context bound is a safe fallback.
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = self.max_context_len

            seq_lens = forward_batch.seq_lens

            if forward_batch.forward_mode.is_target_verify():
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
                self.forward_decode_metadata.global_seq_lens_k = (
                    self.forward_decode_metadata.seq_lens_k
                )
            elif forward_batch.forward_mode.is_decode_or_idle():
                # One int32 conversion per step; forward_decode reads it back
                # so the per-layer .to(int32) in _run_decode_kernel stays a
                # no-op (24 elementwise copies/step otherwise).
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
            elif forward_batch.forward_mode.is_draft_extend_v2():
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                # see NOTE(draft_extend seq_len handling)
                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)

            spec = forward_batch.spec_info
            if self._is_draft_frontier_spec(forward_batch.forward_mode, spec, bs):
                if not self.supports_custom_decode_mask:
                    raise RuntimeError(
                        "EAGLE topk>1 MLA draft frontier requires a decode backend "
                        "with custom-mask support."
                    )
                (
                    block_kv_indices,
                    seq_lens_k,
                    custom_mask,
                    custom_mask_offsets,
                    frontier_max_seq_k,
                ) = self._build_draft_frontier_page_metadata(
                    spec.kv_indptr,
                    spec.kv_indices,
                    getattr(spec, "positions", None),
                    forward_batch.req_pool_indices,
                    seq_lens.device,
                )
                bs = spec.kv_indptr.shape[0] - 1
                self.forward_decode_metadata.seq_lens_k = seq_lens_k
                self.forward_decode_metadata.custom_mask = custom_mask
                self.forward_decode_metadata.custom_mask_offsets = custom_mask_offsets
                self.forward_decode_metadata.is_draft_frontier = True
            else:
                max_seqlen_pad = self._calc_padded_blocks(max_seq)
                block_kv_indices = self._create_block_kv_indices(
                    bs,
                    max_seqlen_pad,
                    forward_batch.req_pool_indices,
                    seq_lens,
                    seq_lens.device,
                )
                frontier_max_seq_k = int(max_seq)

            self.forward_decode_metadata.block_kv_indices = block_kv_indices
            self.forward_decode_metadata.max_seq_len_k = frontier_max_seq_k
            self.forward_decode_metadata.batch_size = bs

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel."""
        return pad_draft_extend_query_triton(
            q,
            padded_q,
            seq_lens_q,
            cu_seqlens_q,
        )

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
        zero_uncovered: bool = False,
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel.

        zero_uncovered: ragged verify's clamped rows leave output positions
        unwritten; zero the destination so those discarded rows stay finite
        (draft_extend writes every position and does not need this).
        """
        output_buffer = self.unpad_output_buffer
        if zero_uncovered:
            if output_buffer is not None:
                output_buffer[:sum_seq_lens_q].zero_()
            else:
                # No persistent buffer without cuda graph state; the triton
                # wrapper's dynamic fallback is torch.empty.
                output_buffer = torch.zeros(
                    (sum_seq_lens_q, raw_out.shape[2], raw_out.shape[3]),
                    dtype=raw_out.dtype,
                    device=raw_out.device,
                )
        return unpad_draft_extend_output_triton(
            raw_out,
            cu_seqlens_q,
            seq_lens_q,
            sum_seq_lens_q,
            output_buffer,
        )

    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
        """BMM1 scale q_scale * k_scale * softmax_scale. k_scale only
        applies when the KV cache stores FP8."""
        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
        else:
            if getattr(layer, "k_scale_float", None) is not None:
                logger.warning_once(
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                    layer.k_scale_float,
                    self.data_type,
                )
            k_scale = 1.0
        return q_scale * k_scale * layer.scaling

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
        custom_mask: Optional[torch.Tensor] = None,
        custom_mask_offsets: Optional[torch.Tensor] = None,
        *,
        causal_seqs: Optional[torch.Tensor] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        return_lse: bool = False,
    ) -> torch.Tensor:
        """Hook for subclasses to swap the decode/spec-verify kernel.

        The DCP arguments belong to the hook contract because forward_extend
        passes them on the DCP target-verify path. This implementation does not
        forward them to the kernel and returns no LSE, so only the DCP-capable
        subclasses serve them. ``custom_mask`` is accepted for signature parity
        with the TokenSpeed override; TRT-LLM-gen has no tree-mask support, so
        tree drafting still runs only on a custom-mask-capable subclass."""
        if cp_world > 1 or return_lse:
            raise NotImplementedError(
                "trtllm_mla does not forward the cyclic DCP metadata to its "
                "decode kernel and returns no rank-local LSE for the cross-rank "
                "merge; select cutedsl_mla or tokenspeed_mla for a DCP "
                "target-verify run"
            )

        # Scale computation for TRTLLM MLA kernel BMM1 operation:
        # The final BMM1 scale is computed as: q_scale * k_scale * softmax_scale
        # Scale components:
        # - q_scale: Query scaling factor (set to 1.0 for both FP16/FP8 paths)
        # - k_scale: Key scaling factor from model checkpoint. Only applied when KV cache
        #   stores FP8-quantized values, to compensate for the quantization scaling.
        #   For BF16/FP16 KV cache, k_scale must be 1.0 since values are unscaled.
        # - softmax_scale: Attention softmax scaling = 1/sqrt(head_dim), pre-computed as layer.scaling
        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        extra_kwargs = {"backend": self.backend} if self.backend != "trtllm-gen" else {}
        if self.backend == "trtllm-gen":
            extra_kwargs["multi_ctas_kv_counter_buffer"] = (
                self._multi_ctas_kv_counter_buffer
            )
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            enable_pdl=_ENABLE_PDL,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            **extra_kwargs,
        )

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        seq_lens_kv: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
        out_buffer: torch.Tensor,
        o_sf_scale: float = 1.0,
    ):
        """Hook for subclasses to swap the ragged prefill kernel. Q/K/V arrive
        in model-native dtype; subclasses do any kernel-specific quantization.
        Returns the output tensor or (output, lse) if return_lse."""
        q_scale = k_scale = v_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)
        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace_buffer,
            batch_size=batch_size,
            window_left=-1,
            enable_pdl=False,
            max_q_len=max_q_len,
            bmm1_scale=q_scale * k_scale * layer.scaling,
            bmm2_scale=v_scale,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            seq_lens=seq_lens_kv,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
            return_lse=return_lse,
            o_sf_scale=o_sf_scale,
            out=out_buffer,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
        )

    def _set_kv_and_concat_q_fused(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        q: torch.Tensor,
        q_rope: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Decode: scatter the KV row at ``loc`` (already physical — the
        dense-loc buffer on the unified pool, or out_cache_loc on the static
        pool where ``_full_translate`` is identity) and build the
        [q_nope | q_rope] fmha query in one kernel launch (saves one launch
        per MLA layer and keeps the PDL chain intact).

        Returns the concatenated query, or None when the fused kernel does
        not cover the inputs (caller falls back to the two-kernel path).
        """
        k_nope_2d = k.view(k.shape[0], -1)
        k_rope_2d = k_rope.view(k_rope.shape[0], -1)
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope_3d = q_rope.view(
            -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
        )
        # Same raw per-layer buffer the decode kernel reads below (bf16-only
        # gate means store_dtype == dtype, so no view); get_key_buffer applies
        # the hybrid pool's full-attention layer-id mapping.
        kv_raw = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_2d = kv_raw.view(kv_raw.shape[0], -1) if kv_raw.dim() != 2 else kv_raw
        if not set_mla_kv_concat_q_covered(
            kv_buffer=kv_2d,
            loc=loc,
            k_nope=k_nope_2d,
            k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
        ):
            return None
        return set_mla_kv_concat_q(
            kv_buffer=kv_2d,
            loc=loc,
            cache_k_nope=k_nope_2d,
            cache_k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
        )

    def _set_kv_and_concat_q_fp8_fused(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        q: torch.Tensor,
        q_rope: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """fp8-KV decode: quantize + scatter the KV row at ``loc`` (already
        physical) and build the fp8 [q_nope | q_rope] query in one launch.

        Returns the fp8 query, or None when the fused kernel does not cover
        the inputs (caller falls back to the aten quantize chain).
        """
        k_nope_2d = k.view(k.shape[0], -1)
        k_rope_2d = k_rope.view(k_rope.shape[0], -1)
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope_3d = q_rope.view(
            -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
        )
        # fp8 view of the pool's uint8 store; same buffer the decode kernel
        # reads below.
        kv_raw = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_2d = kv_raw.view(kv_raw.shape[0], -1) if kv_raw.dim() != 2 else kv_raw
        if not set_mla_kv_concat_q_fp8_covered(
            kv_buffer=kv_2d,
            loc=loc,
            k_nope=k_nope_2d,
            k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
        ):
            return None
        parallel = get_parallel()
        return set_mla_kv_concat_q_fp8(
            kv_buffer=kv_2d,
            loc=loc,
            cache_k_nope=k_nope_2d,
            cache_k_rope=k_rope_2d,
            q_nope=q_nope,
            q_rope=q_rope_3d,
            # DCP cyclic KV sharding: virtual loc -> owner mask + loc//world
            # (identity when attn_dcp_size == 1).
            dcp_world_size=parallel.attn_dcp_size,
            dcp_rank=parallel.attn_dcp_rank,
        )

    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope
        k: torch.Tensor,  # k_nope
        v: torch.Tensor,  # not used in this backend
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""
        merge_query = q_rope is not None
        fused_fp8_query = None
        if self.data_type == torch.float8_e4m3fn:
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                if save_kv_cache and self._fused_set_kv_concat_q_fp8:
                    loc = (
                        self._decode_dense_loc
                        if self._decode_dense_loc is not None
                        else (
                            None if self._unified_mla else forward_batch.out_cache_loc
                        )
                    )
                    if loc is not None:
                        # Fused: bf16->fp8 quantize + KV scatter + q concat
                        # in one launch; None when not covered.
                        fused_fp8_query = self._set_kv_and_concat_q_fp8_fused(
                            layer=layer,
                            loc=loc,
                            q=q,
                            q_rope=q_rope,
                            k=k,
                            k_rope=k_rope,
                        )
                if fused_fp8_query is None:
                    q, k, k_rope = mla_quantize_without_rope_for_fp8(
                        q, q_rope, k.squeeze(1), k_rope.squeeze(1)
                    )
            else:
                q, k, k_rope = mla_quantize_and_rope_for_fp8(
                    q,
                    q_rope,
                    k.squeeze(1),
                    k_rope.squeeze(1),
                    forward_batch.positions,
                    cos_sin_cache,
                    is_neox,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                )
            merge_query = False

        # Save KV cache if requested (the fused fp8 path already wrote it)
        query = fused_fp8_query
        if query is None and save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            if self._decode_dense_loc is not None:
                # cuda-graph path: dense write loc precomputed out-of-graph, so
                # the in-graph write captures no translate allocation.
                if merge_query and self._fused_set_kv_concat_q:
                    # Fused: KV scatter + [q_nope | q_rope] concat in one
                    # launch; None when the inputs are not covered.
                    query = self._set_kv_and_concat_q_fused(
                        layer=layer,
                        loc=self._decode_dense_loc,
                        k=k,
                        k_rope=k_rope,
                        q=q,
                        q_rope=q_rope,
                    )
                if query is None:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer, self._decode_dense_loc, k, k_rope, loc_is_dense=True
                    )
            else:
                # eager (or static pool): the pool's _full_translate handles it.
                if (
                    merge_query
                    and self._fused_set_kv_concat_q
                    and not self._unified_mla
                ):
                    # Static pool: _full_translate is identity, so
                    # out_cache_loc is already the physical write loc.
                    query = self._set_kv_and_concat_q_fused(
                        layer=layer,
                        loc=forward_batch.out_cache_loc,
                        k=k,
                        k_rope=k_rope,
                        q=q,
                        q_rope=q_rope,
                    )
                if query is None:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer, forward_batch.out_cache_loc, k, k_rope
                    )

        # Prepare query tensor inline (already built when the fused save-KV
        # path ran)
        if query is None:
            if merge_query:
                # For FP16 path, we merge the query and rope parts into a single tensor
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope_reshaped = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
                query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
            else:
                # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            query = query.to(self.q_data_type) * llama_4_scaling
            query = query.to(self.data_type)

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1
        if query.dim() == 3:
            query = query.unsqueeze(1)

        # Omniva MLA tree (topk>1): the draft frontier is flattened to bs*topk
        # candidates. Its source KV path is token-level, but metadata translates it
        # to page-size block tables plus a per-candidate mask so the kernel can stay
        # on the supported paged MLA path.
        draft_frontier = (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.positions is not None
            and forward_batch.positions.shape[0] == query.shape[0]
            and query.shape[0] != forward_batch.seq_lens.shape[0]
        )

        # Prepare KV cache inline
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        # Backstop: metadata was built pre-pad (marked) and DP padding then
        # grew the batch. The marker path deliberately does not re-plan
        # post-pad (DSA can't rebuild on a padded batch, see #27091), so this
        # local re-plan catches the size mismatch.
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        # Draft frontier metadata carries per-candidate page-rounded KV lengths.
        # Regular decode keeps the request-level seq_lens.
        decode_seq_lens = forward_batch.seq_lens
        if draft_frontier:
            decode_seq_lens = metadata.seq_lens_k
            if decode_seq_lens is None:
                raise RuntimeError(
                    "EAGLE draft frontier decode is missing per-candidate "
                    "page-rounded seq_lens metadata."
                )
        raw_out = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=decode_seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
            custom_mask=getattr(metadata, "custom_mask", None),
            custom_mask_offsets=getattr(metadata, "custom_mask_offsets", None),
        )

        # Reshape output directly without slicing
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # The fallback belongs to genuine extend forwards only. Target-verify /
        # draft-extend must never honor it: `forward_prefill_metadata` is a
        # stale leftover from the last prefill there (eager init clears it,
        # but decode-graph capture does not), and capturing verify through the
        # flashinfer path binds the graph to prefill-planned wrapper buffers,
        # which fault (illegal address) at replay.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl
        ):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
            )

        # TODO refactor to avoid code duplication
        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
            assert q_rope is not None and k_rope is not None
            if cos_sin_cache is None:
                q, k, k_rope = mla_quantize_without_rope_for_fp8(
                    q, q_rope, k.squeeze(1), k_rope.squeeze(1)
                )
            else:
                q, k, k_rope = mla_quantize_and_rope_for_fp8(
                    q,
                    q_rope,
                    k.squeeze(1),
                    k_rope.squeeze(1),
                    forward_batch.positions,
                    cos_sin_cache,
                    is_neox,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                )
            merge_query = False

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # TODO refactor to avoid code duplication
        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q = q.to(self.q_data_type) * llama_4_scaling
            q = q.to(self.data_type)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )

            # Backstop: metadata was built pre-pad (marked) and DP padding
            # then grew the batch. The marker path deliberately does not
            # re-plan post-pad (DSA can't rebuild on a padded batch, see
            # #27091), so this local re-plan catches the size mismatch.
            batch_size = getattr(metadata, "batch_size", None)
            if batch_size is not None and batch_size < forward_batch.batch_size:
                self.init_forward_metadata(forward_batch)
                metadata = forward_batch.decode_trtllm_mla_metadata

            # Ensure query has shape [bs, num_draft_tokens, num_q_heads, head_dim]
            bs = forward_batch.batch_size

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q = q.to(self.data_type)

            if forward_batch.forward_mode.is_target_verify():
                max_seq_len = _target_verify_max_seq_len(metadata)
                draft_token_num = forward_batch.spec_info.draft_token_num
                dcp_enabled = get_parallel().dcp_enabled
                ragged_layout = forward_batch.spec_info.ragged_verify_layout
                if ragged_layout is None or dcp_enabled:
                    q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    if ragged_layout.bs != bs:
                        # Capped variant: the dense [bs, draft_token_num] q
                        # buffer below cannot take a row the full-coverage pad
                        # may inflate past the verify window, and it keeps
                        # qo_indptr consistent with the clamped lens (same
                        # contract as the KDA dense path).
                        ragged_layout = ragged_layout.padded_to_bucket(
                            padded_bs=bs, cap=draft_token_num
                        )
                    total_tokens = q.shape[0]
                    seq_lens_q = torch.clamp(
                        ragged_layout.verify_lens, max=draft_token_num
                    )
                    cu_seqlens_q = ragged_layout.qo_indptr_device
                    padded_q = torch.zeros(
                        (bs, draft_token_num, layer.tp_q_head_num, layer.head_dim),
                        dtype=q.dtype,
                        device=q.device,
                    )
                    q = self.pad_draft_extend_query(
                        q, padded_q, seq_lens_q, cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_zero_uncovered = True
                    unpad_seq_lens_q = seq_lens_q
                    unpad_cu_seqlens_q = cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens
            else:
                # draft_extend: handle varying num_correct_drafts_per_req. If total_tokens % bs == 0,
                # we can directly reshape q; otherwise, pad to max_seq_len_q.
                total_tokens = q.shape[0]
                tokens_per_seq = total_tokens // bs if bs > 0 else 0
                can_direct_view = bs > 0 and (total_tokens % bs == 0)

                if can_direct_view:
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    # Varying lengths: pad q to (bs, max_seq_len_q, ...)
                    actual_seq_lens_q = forward_batch.extend_seq_lens
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q

                    actual_cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),
                        (1, 0),
                    )

                    if self.padded_q_buffer is not None:
                        padded_q = self.padded_q_buffer[
                            :bs, :actual_max_seq_len_q, :, :
                        ].to(dtype=q.dtype)
                        padded_q.zero_()
                    else:
                        padded_q = torch.zeros(
                            (
                                bs,
                                actual_max_seq_len_q,
                                layer.tp_q_head_num,
                                layer.head_dim,
                            ),
                            dtype=q.dtype,
                            device=q.device,
                        )

                    q = self.pad_draft_extend_query(
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_zero_uncovered = False
                    unpad_seq_lens_q = actual_seq_lens_q
                    unpad_cu_seqlens_q = actual_cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens

            assert kv_cache.dtype == self.data_type

            # Omniva MLA tree-spec: pass SGLang's tree attention mask to the verify
            # kernel only for tree drafting (topk>1). For topk=1 the tree is a chain
            # (== causal), so we leave custom_mask=None to keep the fast causal path.
            # The tokenspeed_mla override consumes it; the TRT-LLM-gen base ignores it.
            cmask = (
                getattr(forward_batch.spec_info, "custom_mask", None)
                if (
                    forward_batch.forward_mode.is_target_verify()
                    and getattr(forward_batch.spec_info, "topk", 1) > 1
                )
                else None
            )
            if (
                forward_batch.forward_mode.is_target_verify()
                and get_parallel().dcp_enabled
            ):
                raw_out, lse = self._run_decode_kernel(
                    query=q,
                    kv_cache=kv_cache,
                    block_tables=metadata.block_kv_indices,
                    seq_lens=metadata.seq_lens_k,
                    max_seq_len=max_seq_len,
                    layer=layer,
                    custom_mask=cmask,
                    custom_mask_offsets=getattr(metadata, "custom_mask_offsets", None),
                    causal_seqs=metadata.global_seq_lens_k,
                    cp_world=get_parallel().dcp_size,
                    cp_rank=get_parallel().dcp_rank,
                    return_lse=True,
                )
                output = raw_out.view(
                    bs * draft_token_num,
                    layer.tp_q_head_num,
                    layer.v_head_dim,
                )
                lse = lse.view(bs * draft_token_num, layer.tp_q_head_num)
                dense_q_indptr = torch.arange(
                    0,
                    (bs + 1) * draft_token_num,
                    draft_token_num,
                    dtype=torch.int32,
                    device=q.device,
                )
                fixup_zero_kv_rows(
                    output,
                    lse,
                    metadata.seq_lens_k,
                    dense_q_indptr,
                    draft_token_num,
                )
                return output.flatten(1), lse

            raw_out = self._run_decode_kernel(
                query=q,
                kv_cache=kv_cache,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.seq_lens_k,
                max_seq_len=max_seq_len,
                layer=layer,
                custom_mask=cmask,
            )

            if needs_unpad:
                # Unpad the output for draft_extend mode with varying lengths
                # Use the actual values computed during padding, not from metadata
                output = self.unpad_draft_extend_output(
                    raw_out,
                    unpad_cu_seqlens_q,
                    unpad_seq_lens_q,
                    unpad_sum_seq_lens_q,
                    zero_uncovered=unpad_zero_uncovered,
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            else:
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # When chunked prefix cache is enabled, dispatch to different path for ragged attention.
        if forward_batch.attn_attend_prefix_cache:
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            out = torch.empty(
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=self.q_data_type,
                device=q.device,
            )
            result = self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                is_causal=False,
                return_lse=True,
                out_buffer=out,
                o_sf_scale=-1.0,
            )

            # The TRT-LLM ragged attention cubin kernel does not correctly
            # handle rows with kv_len == 0: it leaves stale data in the
            # workspace softmaxStats buffer and may produce non-zero output
            # for those rows.  Fix up by forcing out=0 and lse=-inf for
            # zero-KV rows so that downstream merge_state ignores them.
            # Skip entirely when this chunk has no zero-KV rows (pure CPU
            # check, precomputed in prepare_chunked_prefix_cache_info).
            if forward_batch.prefix_chunk_has_zero_kv[chunk_idx]:
                out_tensor, lse_tensor = result
                fixup_zero_kv_rows(
                    out_tensor,
                    lse_tensor,
                    forward_batch.prefix_chunk_seq_lens[chunk_idx],
                    self.forward_prefill_metadata.cum_seq_lens,
                    self.forward_prefill_metadata.max_seq_len,
                )

            return result
        else:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self.q_data_type,
            )
            return self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=self.forward_prefill_metadata.seq_lens,
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
                max_kv_len=self.forward_prefill_metadata.max_seq_len,
                is_causal=True,
                return_lse=forward_batch.mha_return_lse,
                out_buffer=out,
                o_sf_scale=1.0,
            )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""

    # Per-step draft decode never reads seq_lens_cpu / seq_lens_sum; opt out so
    # decide_needs_cpu_seq_lens' OR over the backends stays False.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        backend: str = "trtllm-gen",
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
                backend=backend,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if self.topk > 1 and getattr(
            self.attn_backends[0], "supports_custom_decode_mask", False
        ):
            kv_indices = torch.zeros(
                (self.speculative_num_steps, 1),
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )

            def call_fn(i, _forward_batch):
                self.attn_backends[i].init_forward_metadata(_forward_batch)

            return self.common_template(forward_batch, kv_indices, call_fn)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if self.topk > 1 and getattr(
            self.attn_backends[0], "supports_custom_decode_mask", False
        ):

            def call_fn(i, _forward_batch):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    _forward_batch, in_capture=in_capture
                )

            return self.common_template(
                forward_batch, self.cuda_graph_kv_indices, call_fn
            )

        if in_capture:
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(inner_fb)

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn,
    ):
        if self.topk <= 1 or not getattr(
            self.attn_backends[0], "supports_custom_decode_mask", False
        ):
            return super().common_template(forward_batch, kv_indices_buffer, call_fn)

        num_seqs = forward_batch.batch_size
        num_cand = self.topk * num_seqs

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()
        forward_batch.spec_info.positions = forward_batch.positions

        # The tokenspeed page-table path does not read the full token-level
        # kv_indices buffer. Keep only the tiny frontier indptr that carries each
        # candidate's token length; the page-table kernel derives pages directly
        # from req_to_token and the branch layout.
        base_lens = forward_batch.seq_lens.to(torch.int32).repeat_interleave(self.topk)
        for i in range(self.speculative_num_steps - 1):
            kv_indptr = self.kv_indptr[i, : num_cand + 1]
            kv_indptr[:1].zero_()
            torch.cumsum(base_lens + (i + 1), dim=0, out=kv_indptr[1:])
            forward_batch.spec_info.kv_indptr = kv_indptr
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i, :1]
            forward_batch.spec_info.draft_step = i + 1
            call_fn(i, forward_batch)
