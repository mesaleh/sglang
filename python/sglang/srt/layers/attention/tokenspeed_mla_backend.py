# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

"""Attention backend for the tokenspeed-mla CuTe DSL kernels on Blackwell.

Subclasses :class:`TRTLLMMLABackend` to share its MLA data preparation and
prefill plumbing. Decode-context parallelism is implemented here because the
TokenSpeed decode kernel natively accepts CP rank/world metadata and returns
the partial log-sum-exp needed by the cross-rank merge.
"""

import logging
import math
from inspect import signature
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.dcp_kernels import (
    create_mla_kv_page_table_for_dcp,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8 import (
    mla_kv_pack_quantize_fp8,
)
from sglang.kernels.ops.attention.utils import (
    mla_quantize_and_rope_for_fp8,
    mla_quantize_without_rope_for_fp8,
)
from sglang.kernels.ops.kvcache.kv_indices import (
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.kernels.ops.quantization.fp8_quantize import fp8_quantize
from sglang.kernels.ops.quantization.hadamard import hadamard_transform_with_signs
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available

if is_flashinfer_available():
    import flashinfer.rope as _flashinfer_rope

if is_tokenspeed_mla_available():
    import tokenspeed_mla

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

logger = logging.getLogger(__name__)


def _supports_custom_decode_mask(decode_fn) -> bool:
    parameters = signature(decode_fn).parameters
    return "custom_mask" in parameters and "cmask_off" in parameters


def _custom_decode_mask_kwargs(
    *,
    supports_custom_decode_mask: bool,
    custom_mask,
    custom_mask_offsets,
) -> dict:
    if custom_mask is not None and not supports_custom_decode_mask:
        raise RuntimeError(
            "tokenspeed_mla target verification requires a side-library build "
            "whose decode API accepts custom_mask and cmask_off."
        )
    if not supports_custom_decode_mask:
        return {}
    return {
        "custom_mask": custom_mask,
        "cmask_off": custom_mask_offsets,
    }


# Workspace upper bound for tokenspeed_mla_decode:
#   num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * sizeof(float32)
# MAX_Q_LEN=8 covers EAGLE3 num_draft_tokens=4 plus headroom. Larger
# speculative widths grow the bound at backend initialization.
_TOKENSPEED_MAX_Q_LEN = 8
_N10_KV_CACHE_DTYPE = "turboquant_4bit_e2m1_recip_bf16"
_N10_QUERY_HEADS = 8
_N10_LATENT_DIM = 512
_N10_ROPE_DIM = 64
_N10_PAGE_SIZE = 32
_N10_QUERY_LENGTHS = (1, 5)
_N10_PAGE_TABLE_TILE_PAGES = 4


@triton.jit
def _repeat_last_valid_mla_page_kernel(
    block_tables,
    seq_lens,
    table_width: tl.constexpr,
    table_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens + row).to(tl.int32)
    num_pages = tl.minimum(tl.maximum(tl.cdiv(seq_len, PAGE_SIZE), 1), table_width)
    last_page = tl.load(
        block_tables + row * table_stride + num_pages - 1,
        mask=seq_len > 0,
        other=0,
    )
    mask = ((seq_len <= 0) | (offsets >= num_pages)) & (offsets < table_width)
    tl.store(block_tables + row * table_stride + offsets, last_page, mask=mask)


def _repeat_last_valid_mla_pages(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Fill TokenSpeed's 128-token TMA tail with a valid repeated page.

    The logical sequence length still masks these entries in attention.  The
    packed reader nevertheless requires every four-page TMA descriptor lane to
    name valid storage, unlike the dense backend which tolerates ``-1`` tails.
    """

    if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
        raise ValueError("N10 block_tables must be a two-dimensional int32 tensor.")
    if seq_lens.ndim != 1 or seq_lens.dtype != torch.int32:
        raise ValueError("N10 seq_lens must be a one-dimensional int32 tensor.")
    if block_tables.shape[0] != seq_lens.shape[0]:
        raise ValueError(
            "N10 block-table/sequence batch mismatch: "
            f"{block_tables.shape[0]} versus {seq_lens.shape[0]}."
        )
    width = block_tables.shape[1]
    if width <= 0 or width % _N10_PAGE_TABLE_TILE_PAGES:
        raise ValueError(
            "N10 block-table width must be a positive multiple of four pages, "
            f"got {width}."
        )
    block_size = 256
    _repeat_last_valid_mla_page_kernel[
        (block_tables.shape[0], triton.cdiv(width, block_size))
    ](
        block_tables,
        seq_lens,
        table_width=width,
        table_stride=block_tables.stride(0),
        PAGE_SIZE=_N10_PAGE_SIZE,
        BLOCK_SIZE=block_size,
    )


def _tokenspeed_workspace_nbytes(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    max_q_len: int = _TOKENSPEED_MAX_Q_LEN,
) -> int:
    # DCP target verification gathers Q to the full head count before launching
    # TokenSpeed; size for that launch shape, not the rank-local head count.
    num_heads *= get_parallel().attn_dcp_size
    max_q_len = max(max_q_len, _TOKENSPEED_MAX_Q_LEN)
    return (
        tokenspeed_mla.get_num_sm(device)
        * num_heads
        * max_q_len
        * (kv_lora_rank + 1)
        * 4
    )


def _get_tokenspeed_workspace(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    max_q_len: int = _TOKENSPEED_MAX_Q_LEN,
) -> torch.Tensor:
    from sglang.srt.runtime_context import get_resources

    needed = _tokenspeed_workspace_nbytes(device, num_heads, kv_lora_rank, max_q_len)
    buffers = get_resources().buffers
    key = f"tokenspeed_mla_workspace:{device}"
    existing = buffers.get(key)
    if existing is None or existing.numel() < needed:
        buffers[key] = torch.empty(needed, dtype=torch.int8, device=device)
    return buffers[key]


# TODO(Qiaolin-Yu): Merge this attention backend into trtllm_mla_backend.py
# once the same CuteDSL kernels in flashinfer_trtllm are stable
# and there is no performance gap compared to this backend.
class TokenspeedMLABackend(TRTLLMMLABackend):
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100, FP8 KV)."""

    supports_custom_decode_mask: bool = is_tokenspeed_mla_available() and (
        _supports_custom_decode_mask(tokenspeed_mla.tokenspeed_mla_decode)
    )

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        self._validate_cache_contract(model_runner)

        self._tokenspeed_workspace: Optional[torch.Tensor] = None
        if is_tokenspeed_mla_available():
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                self.device,
                self.num_q_heads,
                self.kv_lora_rank,
                max_q_len=(
                    model_runner.server_args.max_speculative_num_draft_tokens or 1
                ),
            )

            # Pre-JIT the prefill kernel variants. Each cute.compile takes 1-2
            # min; without warm-up the first request trips the 300 s scheduler
            # watchdog.
            _compile_prefill_kernel = tokenspeed_mla.mla_prefill._compile_prefill_kernel
            _compiled_kernels = tokenspeed_mla.mla_prefill._compiled_kernels
            head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            enable_ex2_emulation = tokenspeed_mla.mla_prefill._enable_ex2_emulation()
            use_pdl = is_arch_support_pdl()
            for is_causal in (True, False):
                for return_lse in (True, False):
                    # Non-causal is only entered from the chunked-prefix
                    # branch, which always asks for the LSE.
                    if is_causal is False and return_lse is False:
                        continue
                    # Runtime feeds fp8_e4m3fn q/k/v
                    config = (
                        torch.float8_e4m3fn,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl,
                        enable_ex2_emulation,
                    )
                    if config in _compiled_kernels:
                        continue
                    _compiled_kernels[config] = _compile_prefill_kernel(
                        torch.float8_e4m3fn,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl=use_pdl,
                        enable_ex2_emulation=enable_ex2_emulation,
                    )

    def _validate_cache_contract(self, model_runner: ModelRunner) -> None:
        if self.data_type != torch.float8_e4m3fn:
            raise ValueError(
                "tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got data_type={self.data_type}."
            )
        if self.page_size not in (32, 64):
            raise ValueError(
                "tokenspeed_mla backend requires page_size in {32, 64}, "
                f"got page_size={self.page_size}."
            )

    def _fused_rope_fp8_quantize(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        is_neox: bool,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RoPE + FP8 quantize that also packs nope+pe along the last
        dim, so FMHA consumes contig FP8 Q/K without an extra concat or cast.
        """
        num_heads = q_nope.shape[1]
        seq_len = q_nope.shape[0]
        q_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_nope.device,
        )
        k_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=k_nope.device,
        )
        if seq_len == 0:
            return q_fp8, k_fp8

        # Broadcast the shared latent k_pe across heads — RoPE is position-only
        # so per-head outputs are identical, and the cache write below reuses
        # head 0.
        if k_pe.dim() == 3 and k_pe.shape[1] == 1:
            k_pe_expanded = k_pe.expand(-1, num_heads, -1)
        else:
            k_pe_expanded = k_pe

        _flashinfer_rope.mla_rope_quantize_fp8(
            q_rope=q_pe,
            k_rope=k_pe_expanded,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=is_neox,
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_fp8[..., qk_nope_head_dim:],
            k_rope_out=k_fp8[..., qk_nope_head_dim:],
            q_nope_out=q_fp8[..., :qk_nope_head_dim],
            k_nope_out=k_fp8[..., :qk_nope_head_dim],
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=is_arch_support_pdl(),
        )
        return q_fp8, k_fp8

    def prepare_prefill_qkv(
        self,
        *,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        layer: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build FP8 (Q, K, V) for the FMHA kernel and write FP8 KV cache."""
        kv = layer.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, layer.num_local_heads, layer.qk_nope_head_dim + layer.v_head_dim
        )
        k_nope = kv[..., : layer.qk_nope_head_dim]
        v_bf16 = kv[..., layer.qk_nope_head_dim :]
        q_nope = q[..., : layer.qk_nope_head_dim]

        q_fp8, k_fp8 = self._fused_rope_fp8_quantize(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,
            positions=positions,
            is_neox=getattr(layer.rotary_emb, "is_neox_style", True),
            qk_nope_head_dim=layer.qk_nope_head_dim,
            qk_rope_head_dim=layer.qk_rope_head_dim,
        )
        v_fp8 = fp8_quantize(v_bf16, enable_pdl=is_arch_support_pdl())

        # k_pe is shared across heads (RoPE is position-only), so head 0
        # reproduces the original [tokens, 1, qk_rope] latent layout.
        kv_a_fp8 = fp8_quantize(kv_a, enable_pdl=is_arch_support_pdl())
        k_pe_fp8 = k_fp8[:, 0:1, layer.qk_nope_head_dim :]
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            kv_a_fp8.unsqueeze(1),
            k_pe_fp8,
        )
        return q_fp8, k_fp8, v_fp8

    def pack_prefix_chunk_kv(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack strided ``k_nope``+``k_pe`` into contig FP8 K and quantize
        strided ``v`` into contig FP8 V in a single kernel.
        """
        return mla_kv_pack_quantize_fp8(
            k_nope, k_pe, v, enable_pdl=is_arch_support_pdl()
        )

    def _ensure_workspace(self, device: torch.device, q_len: int) -> torch.Tensor:
        if (
            self._tokenspeed_workspace is None
            or self._tokenspeed_workspace.device != device
            or self._tokenspeed_workspace.numel()
            < _tokenspeed_workspace_nbytes(
                device, self.num_q_heads, self.kv_lora_rank, q_len
            )
        ):
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                device, self.num_q_heads, self.kv_lora_rank, max_q_len=q_len
            )
        return self._tokenspeed_workspace

    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return seq_lens
        return get_dcp_lens(seq_lens, parallel.dcp_size, parallel.dcp_rank).to(
            torch.int32
        )

    def _get_dcp_local_max_seq_len(self, max_seq_len: int) -> int:
        parallel = get_parallel()
        if not parallel.dcp_enabled:
            return max_seq_len
        local_max = max_seq_len // parallel.dcp_size + int(
            parallel.dcp_rank < max_seq_len % parallel.dcp_size
        )
        # TokenSpeed requires a positive scheduling bound even when every
        # sequence in a padded graph row is empty on this rank.
        return max(local_max, 1)

    def _fill_dcp_block_kv_indices(
        self,
        block_kv_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        local_seq_lens: torch.Tensor,
    ) -> None:
        parallel = get_parallel()
        pages_per_block = get_num_page_per_block_flashmla(self.page_size)
        create_mla_kv_page_table_for_dcp[
            (
                block_kv_indices.shape[0],
                get_num_kv_index_blocks_flashmla(
                    block_kv_indices.shape[1], self.page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            local_seq_lens,
            block_kv_indices,
            self.req_to_token.stride(0),
            block_kv_indices.stride(0),
            PHYSICAL_PAGE_SIZE=self.page_size,
            DCP_SIZE=parallel.dcp_size,
            DCP_RANK=parallel.dcp_rank,
            PAGES_PER_BLOCK=pages_per_block,
        )

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not get_parallel().dcp_enabled:
            return super()._create_block_kv_indices(
                batch_size,
                max_blocks,
                req_pool_indices,
                seq_lens,
                device,
            )

        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        self._fill_dcp_block_kv_indices(
            block_kv_indices,
            req_pool_indices,
            self._get_dcp_local_seq_lens(seq_lens),
        )
        return block_kv_indices

    def _raise_dcp_draft_extend_unsupported(self):
        raise RuntimeError(
            "tokenspeed_mla DCP does not support draft-extend speculative "
            "decoding: the draft KV pool needs DCP-aware index translation."
        )

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode,
        seq_lens: torch.Tensor,
        device: torch.device,
        req_pool_indices: Optional[torch.Tensor] = None,
        spec_info=None,
    ):
        if get_parallel().dcp_enabled and forward_mode.is_draft_extend_v2():
            self._raise_dcp_draft_extend_unsupported()
        super()._init_cuda_graph_metadata(
            bs,
            num_tokens,
            forward_mode,
            seq_lens,
            device,
            req_pool_indices=req_pool_indices,
            spec_info=spec_info,
        )
        if (
            get_parallel().dcp_enabled
            and not getattr(self.forward_decode_metadata, "is_draft_frontier", False)
            and not forward_mode.is_draft_extend_v2()
        ):
            self.forward_decode_metadata.max_seq_len_k = (
                self._get_dcp_local_max_seq_len(
                    self.max_context_len
                    + (self.num_draft_tokens if forward_mode.is_target_verify() else 0)
                )
            )

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode,
        spec_info=None,
    ):
        if not get_parallel().dcp_enabled:
            return super()._apply_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                forward_mode,
                spec_info=spec_info,
            )
        if forward_mode.is_draft_extend_v2():
            self._raise_dcp_draft_extend_unsupported()
        if self._is_draft_frontier_spec(forward_mode, spec_info, bs):
            raise RuntimeError(
                "tokenspeed_mla DCP CUDA graph metadata does not support "
                "EAGLE topk>1 draft frontier replay."
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_target_verify():
            torch.add(
                seq_lens[:bs],
                self.num_draft_tokens,
                out=metadata.global_seq_lens_k,
            )
            metadata.seq_lens_k.copy_(
                self._get_dcp_local_seq_lens(metadata.global_seq_lens_k)
            )
            local_seq_lens = metadata.seq_lens_k
        else:
            seq_lens = seq_lens[:bs]
            local_seq_lens = self._get_dcp_local_seq_lens(seq_lens)
            metadata.seq_lens_k.copy_(local_seq_lens)
            local_seq_lens = metadata.seq_lens_k

        self._fill_dcp_block_kv_indices(
            metadata.block_kv_indices,
            req_pool_indices[:bs],
            local_seq_lens,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if (
            get_parallel().dcp_enabled
            and forward_batch.forward_mode.is_draft_extend_v2()
        ):
            self._raise_dcp_draft_extend_unsupported()
        super().init_forward_metadata(forward_batch)
        if (
            get_parallel().dcp_enabled
            and self.forward_decode_metadata is not None
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
        ):
            metadata = self.forward_decode_metadata
            if getattr(metadata, "is_draft_frontier", False):
                raise RuntimeError(
                    "tokenspeed_mla DCP metadata does not support "
                    "EAGLE topk>1 draft frontier replay."
                )
            if forward_batch.forward_mode.is_target_verify():
                metadata.global_seq_lens_k = metadata.seq_lens_k
                metadata.seq_lens_k = self._get_dcp_local_seq_lens(
                    metadata.global_seq_lens_k
                )
            metadata.max_seq_len_k = self._get_dcp_local_max_seq_len(
                metadata.max_seq_len_k
            )

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
    ):
        k_scale = getattr(layer, "k_scale_float", None)
        if k_scale is None:
            k_scale = 1.0
        softmax_scale = float(layer.scaling) * float(k_scale)
        output_scale = float(k_scale)

        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        decode_kwargs = dict(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._ensure_workspace(query.device, query.shape[1]),
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=int(max_seq_len),
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            enable_pdl=is_arch_support_pdl(),
            return_lse=return_lse,
            causal_seqs=causal_seqs,
            cp_world=cp_world,
            cp_rank=cp_rank,
        )
        decode_kwargs.update(
            _custom_decode_mask_kwargs(
                supports_custom_decode_mask=self.supports_custom_decode_mask,
                custom_mask=custom_mask,
                custom_mask_offsets=custom_mask_offsets,
            )
        )
        return tokenspeed_mla.tokenspeed_mla_decode(**decode_kwargs)

    def forward_decode(
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
    ):
        parallel = get_parallel()
        # FlashInfer autotunes MoE kernels with a synthetic full-model decode
        # and discards the attention/logits result.  On multi-node GB300, the
        # synthetic full-head DCP metadata can make both the TokenSpeed and
        # TRTLLM decode kernels surface cudaErrorNvlinkUncorrectable.  Skip
        # attention only inside that explicitly scoped dummy pass.  Real
        # requests and CUDA graph capture continue through TokenSpeed below.
        if parallel.dcp_enabled and get_in_autotune_dummy_run():
            output = torch.zeros(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.q_data_type,
                device=q.device,
            )
            lse = torch.zeros(
                (q.shape[0], layer.tp_q_head_num),
                dtype=torch.float32,
                device=q.device,
            )
            return output, lse

        if not parallel.dcp_enabled:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

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

        if save_kv_cache:
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        if llama_4_scaling is not None:
            query = (query.to(self.q_data_type) * llama_4_scaling).to(self.data_type)
        if query.dim() == 3:
            query = query.unsqueeze(1)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )
        metadata_batch_size = getattr(metadata, "batch_size", None)
        if (
            metadata_batch_size is not None
            and metadata_batch_size < forward_batch.batch_size
        ):
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        global_seq_lens = forward_batch.seq_lens[: forward_batch.batch_size]
        local_seq_lens = self._get_dcp_local_seq_lens(global_seq_lens)
        raw_out, lse = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=local_seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
            causal_seqs=global_seq_lens,
            cp_world=parallel.dcp_size,
            cp_rank=parallel.dcp_rank,
            return_lse=True,
        )

        output = raw_out.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        lse = lse.view(-1, layer.tp_q_head_num)
        fixup_zero_kv_rows(
            output,
            lse,
            local_seq_lens,
            self.q_indptr_decode[: forward_batch.batch_size + 1],
            1,
        )
        return output.flatten(1), lse

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
    ):  # Q/K/V arrive already in FP8 via the model-side fused path
        # (prepare_prefill_qkv / pack_prefix_chunk_kv); no quantize here.
        # Hybrid MLA models resolve the model-side hook through the outer
        # HybridLinearAttnBackend, so their fallback MHA path can pass V as a
        # last-dimension slice of kv_b_proj (stride(-2) > size(-1)).  The
        # TokenSpeed prefill kernel requires dense Q/K/V layouts even though
        # the public wrapper accepts arbitrary torch tensors.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        return tokenspeed_mla.tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens_kv,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=int(max_kv_len),
            batch_size=int(batch_size),
            softmax_scale=float(layer.scaling),
            is_causal=is_causal,
            return_lse=return_lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=int(max_q_len),
            enable_pdl=is_arch_support_pdl(),
        )


class TokenspeedTQE2M1MLABackend(TokenspeedMLABackend):
    """No-shadow N10 packed-E2M1 MLA backend for the Kimi TP8 geometry."""

    supports_custom_decode_mask: bool = False
    supports_ragged_verify_graph: bool = False

    def _validate_cache_contract(self, model_runner: ModelRunner) -> None:
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolNativeE2M1RecipBF16,
        )

        if model_runner.kv_cache_dtype_str != _N10_KV_CACHE_DTYPE:
            raise ValueError(
                "N10 TokenSpeed backend requires --kv-cache-dtype="
                f"{_N10_KV_CACHE_DTYPE}, got {model_runner.kv_cache_dtype_str}."
            )
        if not isinstance(
            model_runner.token_to_kv_pool, MLATokenToKVPoolNativeE2M1RecipBF16
        ):
            raise TypeError(
                "N10 TokenSpeed backend requires the reciprocal-BF16-RoPE "
                "native E2M1 pool."
            )
        if self.data_type != torch.bfloat16:
            raise ValueError(
                "N10 TokenSpeed backend requires BF16 logical model dtype, "
                f"got {self.data_type}."
            )
        if (
            self.page_size != _N10_PAGE_SIZE
            or self.num_q_heads != _N10_QUERY_HEADS
            or self.kv_lora_rank != _N10_LATENT_DIM
            or self.qk_rope_head_dim != _N10_ROPE_DIM
        ):
            raise ValueError(
                "N10 TokenSpeed backend requires page32/H8/latent512/RoPE64, "
                f"got page{self.page_size}/H{self.num_q_heads}/"
                f"latent{self.kv_lora_rank}/RoPE{self.qk_rope_head_dim}."
            )
        if get_parallel().dcp_enabled:
            raise ValueError("N10 TokenSpeed backend does not support DCP.")
        if self._unified_mla:
            raise ValueError(
                "N10 TokenSpeed backend requires the format-owning static pool; "
                "unified dense MLA views are unsupported."
            )
        if self.speculative_topk not in (None, 1):
            raise ValueError(
                "N10 TokenSpeed backend supports only chain/top-k=1 target verify."
            )
        if not is_tokenspeed_mla_available() or not hasattr(
            tokenspeed_mla, "tokenspeed_mla_decode_tq_e2m1"
        ):
            raise RuntimeError(
                "The installed tokenspeed_mla package does not expose the "
                "matched tokenspeed_mla_decode_tq_e2m1 reader."
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)
        self._n10_graph_capacity = max_num_tokens
        self._n10_query_latent = torch.empty(
            (max_num_tokens, _N10_QUERY_HEADS, _N10_LATENT_DIM),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        self._n10_query_rope = torch.empty(
            (max_num_tokens, _N10_QUERY_HEADS, _N10_ROPE_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._n10_rotated_output = torch.empty(
            (max_num_tokens, _N10_QUERY_HEADS, _N10_LATENT_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._n10_original_output = torch.empty_like(self._n10_rotated_output)

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        block_tables = super()._create_block_kv_indices(
            batch_size,
            max_blocks,
            req_pool_indices,
            seq_lens,
            device,
        )
        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        _repeat_last_valid_mla_pages(block_tables, seq_lens_i32)
        return block_tables

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode,
        spec_info=None,
    ):
        super()._apply_cuda_graph_metadata(
            bs,
            req_pool_indices,
            seq_lens,
            forward_mode,
            spec_info=spec_info,
        )
        metadata = self.decode_cuda_graph_metadata[bs]
        _repeat_last_valid_mla_pages(
            metadata.block_kv_indices,
            metadata.seq_lens_k,
        )

    def _n10_buffers(self, num_tokens: int):
        graph_capacity = getattr(self, "_n10_graph_capacity", 0)
        if num_tokens <= graph_capacity:
            return (
                self._n10_query_latent[:num_tokens],
                self._n10_query_rope[:num_tokens],
                self._n10_rotated_output[:num_tokens],
                self._n10_original_output[:num_tokens],
            )

        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode():
            raise RuntimeError(
                "N10 decode token count exceeds the graph-stable allocation "
                f"during capture: {num_tokens} > {graph_capacity}."
            )
        eager_capacity = getattr(self, "_n10_eager_capacity", 0)
        if num_tokens > eager_capacity:
            eager_capacity = max(num_tokens, eager_capacity * 2, 5)
            self._n10_eager_query_latent = torch.empty(
                (eager_capacity, _N10_QUERY_HEADS, _N10_LATENT_DIM),
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            self._n10_eager_query_rope = torch.empty(
                (eager_capacity, _N10_QUERY_HEADS, _N10_ROPE_DIM),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._n10_eager_rotated_output = torch.empty(
                (eager_capacity, _N10_QUERY_HEADS, _N10_LATENT_DIM),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._n10_eager_original_output = torch.empty_like(
                self._n10_eager_rotated_output
            )
            self._n10_eager_capacity = eager_capacity
        return (
            self._n10_eager_query_latent[:num_tokens],
            self._n10_eager_query_rope[:num_tokens],
            self._n10_eager_rotated_output[:num_tokens],
            self._n10_eager_original_output[:num_tokens],
        )

    def _write_cache_and_prepare_query(
        self,
        *,
        q: torch.Tensor,
        q_rope: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from sglang.kernels.jit.tq_mla_frontend_n10_native import (
            tq_mla_n10_native_frontend_out,
        )

        num_tokens = q.shape[0]
        query_latent, query_rope, rotated_out, original_out = self._n10_buffers(
            num_tokens
        )
        pool_buffers = self.token_to_kv_pool.get_native_e2m1_recip_bf16_buffers(
            layer.layer_id
        )
        tq_mla_n10_native_frontend_out(
            q.view(num_tokens, _N10_QUERY_HEADS, _N10_LATENT_DIM),
            q_rope.view(num_tokens, _N10_QUERY_HEADS, _N10_ROPE_DIM),
            k.view(num_tokens, 1, _N10_LATENT_DIM),
            k_rope.view(num_tokens, 1, _N10_ROPE_DIM),
            forward_batch.out_cache_loc,
            self.token_to_kv_pool.tq_config.signs1,
            self.token_to_kv_pool.tq_config.signs2,
            query_latent,
            query_rope,
            pool_buffers.packed_nope,
            pool_buffers.nope_scale,
            pool_buffers.reciprocal_rope,
            self.token_to_kv_pool.get_native_e2m1_fault_status(),
            self.token_to_kv_pool.get_native_e2m1_zero_count(),
            grid=self.token_to_kv_pool.tq_config.grid,
            rotation_fused=False,
        )
        return query_latent, query_rope, rotated_out, original_out

    def _run_n10_attention(
        self,
        *,
        query_latent: torch.Tensor,
        query_rope: torch.Tensor,
        rotated_out: torch.Tensor,
        original_out: torch.Tensor,
        query_len: int,
        metadata,
        layer: RadixAttention,
    ) -> torch.Tensor:
        if query_len not in _N10_QUERY_LENGTHS:
            raise RuntimeError(
                "N10 TokenSpeed reader supports only q1 decode or q5 target "
                f"verify, got q{query_len}."
            )
        num_tokens = query_latent.shape[0]
        if num_tokens % query_len:
            raise RuntimeError(
                f"N10 q{query_len} token count {num_tokens} is not batch-uniform."
            )
        batch_size = num_tokens // query_len
        if metadata is None:
            raise RuntimeError("N10 decode metadata is unavailable.")
        if metadata.block_kv_indices.shape[0] != batch_size:
            raise RuntimeError(
                "N10 query/page-table batch mismatch: "
                f"{batch_size} versus {metadata.block_kv_indices.shape[0]}."
            )
        if tuple(metadata.seq_lens_k.shape) != (batch_size,):
            raise RuntimeError(
                "N10 query/sequence-length batch mismatch: "
                f"expected {(batch_size,)}, got {tuple(metadata.seq_lens_k.shape)}."
            )
        pool_buffers = self.token_to_kv_pool.get_native_e2m1_recip_bf16_buffers(
            layer.layer_id
        )
        packed = pool_buffers.packed_nope.view(-1, _N10_PAGE_SIZE, _N10_LATENT_DIM // 2)
        scale = pool_buffers.nope_scale.view(-1, _N10_PAGE_SIZE)
        reciprocal_rope = pool_buffers.reciprocal_rope.view(
            -1, _N10_PAGE_SIZE, _N10_ROPE_DIM
        )
        query_latent_4d = query_latent.view(
            batch_size, query_len, _N10_QUERY_HEADS, _N10_LATENT_DIM
        )
        query_rope_4d = query_rope.view(
            batch_size, query_len, _N10_QUERY_HEADS, _N10_ROPE_DIM
        )
        rotated_out_4d = rotated_out.view(
            batch_size, query_len, _N10_QUERY_HEADS, _N10_LATENT_DIM
        )
        tokenspeed_mla.tokenspeed_mla_decode_tq_e2m1(
            query_latent=query_latent_4d,
            query_rope=query_rope_4d,
            packed_latent=packed,
            reconstruction_scale=scale,
            reciprocal_rope=reciprocal_rope,
            workspace_buffer=self._ensure_workspace(query_latent.device, query_len),
            block_tables=metadata.block_kv_indices,
            seq_lens=metadata.seq_lens_k,
            max_seq_len=int(metadata.max_seq_len_k),
            softmax_scale=float(layer.scaling),
            output_scale=1.0,
            out=rotated_out_4d,
            enable_pdl=is_arch_support_pdl(),
        )
        if getattr(self.token_to_kv_pool.tq_config, "output_rotation_fused", False):
            return rotated_out.view(num_tokens, -1)
        hadamard_transform_with_signs(
            rotated_out,
            self.token_to_kv_pool.tq_config.signs2,
            self.token_to_kv_pool.tq_config.signs1,
            scale=1.0 / math.sqrt(_N10_LATENT_DIM),
            out=original_out,
        )
        return original_out.view(num_tokens, -1)

    def forward_decode(
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
    ):
        if not save_kv_cache:
            raise RuntimeError("N10 decode requires the fused target cache write.")
        if q_rope is None or k_rope is None:
            raise RuntimeError("N10 decode requires post-RoPE BF16 q/k coordinates.")
        if cos_sin_cache is not None:
            raise RuntimeError("N10 outer MLA path must apply RoPE exactly once.")
        if llama_4_scaling is not None:
            raise RuntimeError("N10 does not yet support llama_4 query scaling.")
        prepared = self._write_cache_and_prepare_query(
            q=q,
            q_rope=q_rope,
            k=k,
            k_rope=k_rope,
            layer=layer,
            forward_batch=forward_batch,
        )
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )
        return self._run_n10_attention(
            query_latent=prepared[0],
            query_rope=prepared[1],
            rotated_out=prepared[2],
            original_out=prepared[3],
            query_len=1,
            metadata=metadata,
            layer=layer,
        )

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
    ):
        if not forward_batch.forward_mode.is_target_verify():
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )
        if getattr(forward_batch.spec_info, "topk", 1) > 1:
            raise RuntimeError("N10 target verify supports only DFlash top-k=1.")
        if getattr(forward_batch.spec_info, "ragged_verify_layout", None) is not None:
            raise RuntimeError("N10 ragged target verify is not yet supported.")
        if q_rope is None or k_rope is None or cos_sin_cache is not None:
            raise RuntimeError(
                "N10 target verify requires post-RoPE BF16 q/k and no fused-RoPE input."
            )
        if not save_kv_cache or llama_4_scaling is not None:
            raise RuntimeError(
                "N10 target verify requires fused cache write and no llama_4 scaling."
            )
        prepared = self._write_cache_and_prepare_query(
            q=q,
            q_rope=q_rope,
            k=k,
            k_rope=k_rope,
            layer=layer,
            forward_batch=forward_batch,
        )
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )
        query_len = forward_batch.spec_info.draft_token_num
        return self._run_n10_attention(
            query_latent=prepared[0],
            query_rope=prepared[1],
            rotated_out=prepared[2],
            original_out=prepared[3],
            query_len=query_len,
            metadata=metadata,
            layer=layer,
        )

    def prepare_prefill_qkv(
        self,
        *,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        layer: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv = layer.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, layer.num_local_heads, layer.qk_nope_head_dim + layer.v_head_dim
        )
        k_nope = kv[..., : layer.qk_nope_head_dim]
        v_bf16 = kv[..., layer.qk_nope_head_dim :]
        q_nope = q[..., : layer.qk_nope_head_dim]
        q_fp8, k_fp8 = self._fused_rope_fp8_quantize(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,
            positions=positions,
            is_neox=getattr(layer.rotary_emb, "is_neox_style", True),
            qk_nope_head_dim=layer.qk_nope_head_dim,
            qk_rope_head_dim=layer.qk_rope_head_dim,
        )
        v_fp8 = fp8_quantize(v_bf16, enable_pdl=is_arch_support_pdl())
        _, k_pe_bf16 = layer.rotary_emb(positions, q_pe, k_pe)
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            kv_a.unsqueeze(1),
            k_pe_bf16,
        )
        return q_fp8, k_fp8, v_fp8


class TokenspeedMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for tokenspeed_mla used by EAGLE."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with tokenspeed instances sharing the parent's index buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TokenspeedMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
