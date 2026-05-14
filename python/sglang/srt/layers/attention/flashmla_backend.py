"""
Support attention backend for FlashMLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


PAGE_SIZE = 64
_TQ_MLA_FAST_METADATA = envs.SGLANG_TQ_MLA_FAST_METADATA.get()
_TQ_MLA_FUSED_METADATA_INDICES = envs.SGLANG_TQ_MLA_FUSED_METADATA_INDICES.get()
_TQ_MLA_PROFILE_NVTX = envs.SGLANG_TQ_MLA_PROFILE_NVTX.get()


class _NoOpNVTXRange:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _TorchNVTXRange:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.nvtx.range_pop()
        return False


_NOOP_NVTX_RANGE = _NoOpNVTXRange()


def _tq_mla_nvtx_range(name: str):
    if not _TQ_MLA_PROFILE_NVTX:
        return _NOOP_NVTX_RANGE
    return _TorchNVTXRange(name)


@triton.jit
def _tq_build_indptr_triton(
    seq_lens,
    seq_lens_i32,
    kv_indptr,
    BS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < BS
    vals = tl.load(seq_lens + offs, mask=mask, other=0).to(tl.int32)
    prefix = tl.cumsum(vals, axis=0)

    tl.store(kv_indptr, 0)
    tl.store(seq_lens_i32 + offs, vals, mask=mask)
    tl.store(kv_indptr + offs + 1, prefix, mask=mask)


@triton.jit
def _tq_build_indptr_and_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    seq_lens_ptr,
    kv_indptr,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    bs_offsets = tl.arange(0, BLOCK_BS)
    bs_mask = bs_offsets < BS
    seq_lens = tl.load(seq_lens_ptr + bs_offsets, mask=bs_mask, other=0).to(
        tl.int32
    )
    kv_start_offset = tl.sum(tl.where(bs_offsets < pid, seq_lens, 0), axis=0)
    kv_len = tl.load(seq_lens_ptr + pid).to(tl.int32)

    tl.store(kv_indptr + pid, kv_start_offset)
    tl.store(kv_indptr + BS, kv_start_offset + kv_len, mask=pid == BS - 1)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    num_loop = tl.cdiv(kv_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_start_offset + offset, data, mask=mask)


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices


class FlashMLABackend(FlashInferMLAAttnBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[FlashMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.is_fp8_kvcache = self.data_type in {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.cuda_graph_kv_indices = None
        self.cuda_graph_mla_metadata = None
        self.cuda_graph_num_splits = None
        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(
                forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE
            )
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                forward_batch.seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        elif forward_batch.forward_mode.is_target_verify():
            seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
            seq_lens = forward_batch.seq_lens + self.num_draft_tokens

            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            self.cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self.cuda_graph_kv_indices = block_kv_indices

        device_props = torch.cuda.get_device_properties(self.req_to_token.device)
        max_num_sm_parts = device_props.multi_processor_count

        self.cuda_graph_mla_metadata = torch.empty(
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_num_splits = torch.empty(
            max_bs + 1,
            dtype=torch.int32,
            device="cuda",
        )

        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            num_q_heads = self.num_q_heads

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                :actual_num_sm_parts
            ]
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )

        elif forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                :actual_num_sm_parts
            ]
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
        else:
            super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs]
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            num_q_heads = self.num_q_heads

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]

            if actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"num_sm_parts mismatch in CUDA Graph replay: "
                    f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                    f"replay={actual_num_sm_parts}. "
                    f"This may indicate batch size changed between capture and replay."
                )
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
                self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.forward_metadata.mla_metadata = self.cuda_graph_mla_metadata_view
            self.forward_metadata.num_splits = self.cuda_graph_num_splits_view
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]

        elif forward_mode.is_target_verify():
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            seq_lens_cpu = seq_lens_cpu[:bs] + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]

            if actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]:
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
                self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.forward_metadata.mla_metadata = self.cuda_graph_mla_metadata_view
            self.forward_metadata.num_splits = self.cuda_graph_num_splits_view
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
        else:
            super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        if self.is_fp8_kvcache:
            if layer.k_scale is not None:
                q_scale = layer.k_scale
                descale_q = layer.k_scale.reshape(1)
                descale_k = layer.k_scale.reshape(1)
            else:
                q_scale = torch.ones((1,), dtype=torch.float32, device=reshape_q.device)
                descale_q = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )
                descale_k = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )

            q_shape = reshape_q.shape
            reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
            reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
            reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
            o, _ = flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
                descale_q=descale_q,
                descale_k=descale_k,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            o, _ = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
        ):
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            cache_loc = forward_batch.out_cache_loc

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            bs = forward_batch.batch_size
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            if self.is_fp8_kvcache:
                if layer.k_scale is not None:
                    q_scale = layer.k_scale
                    descale_q = layer.k_scale.reshape(1)
                    descale_k = layer.k_scale.reshape(1)
                else:
                    q_scale = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_q = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_k = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )

                q_shape = reshape_q.shape
                reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
                reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
                reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q_fp8,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=descale_q,
                    descale_k=descale_k,
                )
            else:
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashMLAMultiStepDraftBackend:
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        if topk > 1:
            raise ValueError(
                "Currently FlashMLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashMLABackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=None,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            # EAGLE draft worker uses DECODE mode for draft steps
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            # Create a dummy forward_mode for draft step
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)


class TurboQuantMLABackend(FlashMLABackend):
    """MLA decode backend that reads packed TurboQuant KV directly via a fused
    Triton kernel, avoiding the per-call full-pool dequant that Stage A pays.

    Subclasses FlashMLABackend. Inherits:
      - __init__  (we re-use all the paging / metadata setup)
      - init_forward_metadata (decode + extend + target_verify)
      - init_cuda_graph_state / capture / replay
      - forward_extend (falls through to parent — Stage A dequant-on-read path)

    Overrides only forward_decode. Extend paths keep using the Stage A
    dequant-on-read + flashmla kernel: flashmla's forward_extend calls
    pool.get_key_buffer() which returns the un-rotated bf16 view Stage A
    produces, so it Just Works. Only decode swaps to our fused path.

    Selection: set env SGLANG_TQ_MLA_FUSED_DECODE=1 and the model runner's
    attention-backend dispatch returns this class. See server_args.py
    _handle_attention_backend_selection for the wire-up.

    Design doc: OmniSec/Inference/Performance Optimization/
      Design - Stage C (fused Triton MLA decode on packed KV).md
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        # Assert the pool is TurboQuant-MLA (hard requirement — this backend
        # reads packed buffers directly). Use isinstance per hicache lesson
        # (Kernel Engineering Rule KE-11 style — explicit type check).
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolTurboQuant

        assert isinstance(model_runner.token_to_kv_pool, MLATokenToKVPoolTurboQuant), (
            f"TurboQuantMLABackend requires MLATokenToKVPoolTurboQuant device "
            f"pool; got {type(model_runner.token_to_kv_pool).__name__}. "
            f"Set --kv-cache-dtype turboquant_4bit."
        )

        # Hold refs for convenience in forward_decode.
        self._tq_pool = model_runner.token_to_kv_pool
        self._tq_config = self._tq_pool.tq_config

        # Fixed stage-1 split count. Compile-time constant — avoids a
        # per-forward `.item()` host sync on max_kv_splits, which is one
        # of the two host syncs that blocked CUDA graph capture in v1.
        # Same pattern as triton_backend.py's static_kv_splits fallback.
        self._tq_max_kv_splits = 8

        # Persistent CG-safe buffers. Pre-allocated in init_cuda_graph_state
        # at max_bs and filled at replay time (which runs before the captured
        # graph launches). forward_decode reads these with zero allocation.
        # Non-CG path also uses them via _ensure_fallback_buffers.
        self._tq_kv_indices = None         # (max_bs * max_context_len,) int32
        self._tq_kv_indptr = None          # (max_bs + 1,) int32
        self._tq_seq_lens_i32 = None       # (max_bs,) int32 scratch for metadata fast path
        self._tq_num_kv_splits = None      # (max_bs,) int32 — filled with _tq_max_kv_splits
        self._tq_stage1_logits = None      # (max_bs, q_heads, max_kv_splits, lora_rank) fp32
        self._tq_stage1_lse = None         # (max_bs, q_heads, max_kv_splits) fp32
        self._tq_o_rotated = None          # (max_bs, q_heads, lora_rank) bf16

        # Boot-time visibility: confirms this class (not base FlashMLABackend)
        # is actually instantiated in the live pod. One-line per rank at init.
        import logging
        logging.getLogger(__name__).info(
            "TurboQuantMLABackend active (Stage C fused Triton MLA decode)."
        )

    def _tq_ensure_buffers(self, max_bs: int, device):
        """Allocate all Stage C hot-path buffers sized for max_bs.

        Called from init_cuda_graph_state (when CG enabled) and lazily from
        init_forward_metadata on non-CG path. Buffer addresses are stable
        across forwards so CUDA graph capture sees fixed memory regions.

        Grows on demand: if a larger max_bs arrives (e.g. CG init fires
        after a smaller-bs warmup allocated buffers), we reallocate at the
        larger size. Buffers are never shrunk.
        """
        max_total_tokens = max_bs * self.max_context_len
        if (
            self._tq_kv_indices is not None
            and self._tq_kv_indices.numel() >= max_total_tokens
            and self._tq_kv_indptr is not None
            and self._tq_kv_indptr.numel() >= max_bs + 1
            and self._tq_seq_lens_i32 is not None
            and self._tq_seq_lens_i32.numel() >= max_bs
            and self._tq_num_kv_splits is not None
            and self._tq_num_kv_splits.numel() >= max_bs
            and self._tq_stage1_logits is not None
            and self._tq_stage1_logits.shape[0] >= max_bs
            and self._tq_stage1_lse is not None
            and self._tq_stage1_lse.shape[0] >= max_bs
            and self._tq_o_rotated is not None
            and self._tq_o_rotated.shape[0] >= max_bs
        ):
            return

        q_heads = self.num_q_heads
        lora_rank = self.kv_lora_rank
        max_splits = self._tq_max_kv_splits
        # Max tokens across all batches = max_bs * max_context_len. This is
        # the ceiling kv_indices can ever need (bs=max_bs all at max_seqlen).

        self._tq_kv_indices = torch.empty(
            max_total_tokens, dtype=torch.int32, device=device
        )
        self._tq_kv_indptr = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self._tq_seq_lens_i32 = torch.empty(
            max_bs, dtype=torch.int32, device=device
        )
        # Fixed split count — fill once; kernel reads kv_splits[b] per batch.
        self._tq_num_kv_splits = torch.full(
            (max_bs,), max_splits, dtype=torch.int32, device=device
        )
        self._tq_stage1_logits = torch.empty(
            (max_bs, q_heads, max_splits, lora_rank),
            dtype=torch.float32,
            device=device,
        )
        self._tq_stage1_lse = torch.empty(
            (max_bs, q_heads, max_splits),
            dtype=torch.float32,
            device=device,
        )
        self._tq_o_rotated = torch.empty(
            (max_bs, q_heads, lora_rank),
            dtype=self.q_data_type,
            device=device,
        )

    def _tq_warmup_kernel(self, device):
        """Pre-compile the Stage C Triton kernel so its first launch doesn't
        happen inside CUDA graph capture (JIT during capture = broken graph).

        Runs a bs=1, seq_len=64 decode with dummy inputs. Pool buffers are
        read from the real pool (zero-init is fine at warmup time — nothing
        is actually stored there yet).
        """
        from sglang.srt.layers.attention.triton_ops.turboquant_mla_decode_attention import (
            tq_mla_decode_attention_fwd,
        )

        q_heads = self.num_q_heads
        lora_rank = self.kv_lora_rank
        rope_dim = self.qk_rope_head_dim
        max_splits = self._tq_max_kv_splits

        q_nope_rot = torch.zeros(
            (1, q_heads, lora_rank), dtype=self.q_data_type, device=device
        )
        q_rope = torch.zeros(
            (1, q_heads, rope_dim), dtype=self.q_data_type, device=device
        )
        o = torch.empty(
            (1, q_heads, lora_rank), dtype=self.q_data_type, device=device
        )
        kv_indptr = torch.tensor([0, 64], dtype=torch.int32, device=device)
        kv_indices = torch.zeros(64, dtype=torch.int32, device=device)
        num_kv_splits = torch.full(
            (1,), max_splits, dtype=torch.int32, device=device
        )
        att_logits = torch.empty(
            (1, q_heads, max_splits, lora_rank),
            dtype=torch.float32,
            device=device,
        )
        att_lse = torch.empty(
            (1, q_heads, max_splits), dtype=torch.float32, device=device
        )

        # Read real pool buffers for layer 0 — compile picks up their dtype
        # + strides, so first real forward doesn't recompile.
        layer_id_rel = 0
        k_nope_packed = self._tq_pool.kv_nope_packed_buffer[layer_id_rel]
        k_scale = self._tq_pool.kv_nope_scale_buffer[layer_id_rel]
        k_rope = self._tq_pool.kv_rope_buffer[layer_id_rel]
        k_centroids = self._tq_config.k_centroids
        uniform = getattr(self._tq_config, "uniform", False)

        tq_mla_decode_attention_fwd(
            q_nope_rotated=q_nope_rot,
            q_rope=q_rope,
            k_nope_packed=k_nope_packed,
            k_scale=k_scale,
            k_rope=k_rope,
            k_centroids=k_centroids,
            o=o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            att_logits=att_logits,
            att_lse=att_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=max_splits,
            sm_scale=self.scaling,
            logit_cap=0.0,
            uniform=uniform,
        )
        torch.cuda.synchronize()

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        # Parent builds flashmla's own CG buffers (block_kv_indices, mla
        # metadata, num_splits). We keep that working since forward_extend
        # and the target-verify path still use them.
        with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.parent_state"):
            super().init_cuda_graph_state(max_bs, max_num_tokens, block_kv_indices)

        # Add Stage C's own CG-safe buffers. These are read by forward_decode
        # when SGLANG_TQ_MLA_FUSED_DECODE=1 is set.
        device = self.req_to_token.device
        with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.ensure_buffers"):
            self._tq_ensure_buffers(max_bs, device)

        # Compile the Stage C Triton kernel before graph capture starts.
        # A JIT compile inside capture produces a broken graph with a
        # subtle "kernel missing" symptom at replay.
        with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.warmup"):
            self._tq_warmup_kernel(device)

    def _tq_build_kv_indices(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Fill self._tq_kv_indptr[:bs+1] and self._tq_kv_indices into
        pre-allocated buffers. No new allocations — CG-safe.

        kv_indptr is prefix-sum of seq_lens (exclusive). kv_indices is the
        flat concatenation of per-batch req_to_token rows, written directly
        into the pre-allocated max-sized buffer.
        """
        if _TQ_MLA_FUSED_METADATA_INDICES and seq_lens.is_cuda:
            with _tq_mla_nvtx_range(
                "omniva.tq_mla.metadata.fused_indptr_kv_indices"
            ):
                block_bs = max(1, triton.next_power_of_2(bs))
                _tq_build_indptr_and_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    self._tq_kv_indptr,
                    self._tq_kv_indices,
                    self.req_to_token.stride(0),
                    BS=bs,
                    BLOCK_BS=block_bs,
                )
            return

        if (
            _TQ_MLA_FAST_METADATA
            and seq_lens.is_cuda
            and self._tq_seq_lens_i32 is not None
            and self._tq_seq_lens_i32.numel() >= bs
        ):
            with _tq_mla_nvtx_range("omniva.tq_mla.metadata.fast_indptr"):
                block = max(1, triton.next_power_of_2(bs))
                _tq_build_indptr_triton[(1,)](
                    seq_lens,
                    self._tq_seq_lens_i32,
                    self._tq_kv_indptr,
                    BS=bs,
                    BLOCK=block,
                )
                seq_lens_i32 = self._tq_seq_lens_i32
        else:
            with _tq_mla_nvtx_range("omniva.tq_mla.metadata.seq_lens_i32"):
                seq_lens_i32 = seq_lens[:bs].to(torch.int32)
            with _tq_mla_nvtx_range("omniva.tq_mla.metadata.build_indptr"):
                # Write cumsum into pre-allocated kv_indptr. Implicit int64→int32
                # cast on assignment matches triton_backend.py pattern.
                self._tq_kv_indptr[0] = 0
                self._tq_kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens_i32, dim=0)

        # Fill kv_indices in-place via the flashinfer triton kernel. It
        # writes only sum(seq_lens) entries; the rest of the pre-allocated
        # buffer stays garbage but is never read (bounded by kv_indptr).
        with _tq_mla_nvtx_range("omniva.tq_mla.metadata.build_kv_indices"):
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens_i32,
                self._tq_kv_indptr,
                None,
                self._tq_kv_indices,
                self.req_to_token.stride(0),
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Run parent first to populate flashmla's block_kv_indices, mla
        # metadata, and num_splits — still needed for extend/target_verify.
        with _tq_mla_nvtx_range("omniva.tq_mla.metadata.parent_flashmla"):
            super().init_forward_metadata(forward_batch)

        # Stage C additions: our token-indexed kv_indices + kv_indptr. Built
        # once per forward at init time so forward_decode is alloc-free.
        if forward_batch.forward_mode.is_decode_or_idle():
            bs = forward_batch.batch_size
            # Non-CG path lazy-allocates and grows buffers at runtime bs. This
            # is the path used when --disable-cuda-graph is set.
            with _tq_mla_nvtx_range("omniva.tq_mla.metadata.ensure_buffers"):
                self._tq_ensure_buffers(bs, forward_batch.seq_lens.device)
            with _tq_mla_nvtx_range("omniva.tq_mla.metadata.build_tq_indices"):
                self._tq_build_kv_indices(
                    bs, forward_batch.req_pool_indices, forward_batch.seq_lens
                )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["SpecInput"],
    ):
        # Parent fills block_kv_indices + mla_metadata + num_splits for the
        # forward_metadata object. We re-use it and additionally populate
        # Stage C's own kv_indptr + kv_indices into the pre-allocated pool.
        with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.capture_parent_metadata"):
            super().init_forward_metadata_capture_cuda_graph(
                bs, num_tokens, req_pool_indices, seq_lens,
                encoder_lens, forward_mode, spec_info,
            )
        if forward_mode.is_decode_or_idle():
            with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.capture_tq_indices"):
                self._tq_build_kv_indices(bs, req_pool_indices, seq_lens)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # Replay-time metadata build runs on the CPU before the captured
        # graph launches, so host-sync operations ARE allowed here. Parent
        # already does an .item() on seq_lens_cpu.max() — we inherit that
        # and just add our own in-place buffer refresh.
        with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.replay_parent_metadata"):
            super().init_forward_metadata_replay_cuda_graph(
                bs, req_pool_indices, seq_lens, seq_lens_sum,
                encoder_lens, forward_mode, spec_info, seq_lens_cpu,
            )
        if forward_mode.is_decode_or_idle():
            with _tq_mla_nvtx_range("omniva.tq_mla.cuda_graph.replay_tq_indices"):
                self._tq_build_kv_indices(bs, req_pool_indices, seq_lens[:bs])

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        # Write K to the pool (set_mla_kv_buffer path) — unchanged from parent.
        cache_loc = forward_batch.out_cache_loc
        if k is not None:
            assert v is not None
            if save_kv_cache:
                with _tq_mla_nvtx_range("omniva.tq_mla.decode.save_kv_cache"):
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v
                    )

        from sglang.srt.layers.attention.triton_ops.turboquant_mla_decode_attention import (
            tq_mla_decode_attention_fwd,
        )

        bs = forward_batch.batch_size
        q_heads = layer.tp_q_head_num
        lora_rank = self.kv_lora_rank
        rope_dim = self.qk_rope_head_dim
        full_dim = lora_rank + rope_dim  # = head_dim = 576 for Kimi K2.6
        max_splits = self._tq_max_kv_splits

        # Q comes in shape (bs, q_heads * full_dim) or (bs, 1, q_heads, full_dim).
        # Reshape to (bs, q_heads, full_dim), then split nope | rope.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.q_split_contiguous"):
            reshape_q = q.view(bs, q_heads, full_dim)
            q_nope = reshape_q[:, :, :lora_rank].contiguous()
            q_rope = reshape_q[:, :, lora_rank:].contiguous()

        # --- Fix 2 (Phase 1 findings): rotate Q_nope into WHT domain ---
        # This activates the orthogonality trick: (H·Q)·(H·K) = Q·K lets the
        # kernel dot rotated-Q against rotated-K without per-row inverse WHT.
        # rotate_query returns fp32; cast back to input dtype for the kernel.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.rotate_query"):
            q_nope_rot = self._tq_config.rotate_query(q_nope).to(q_nope.dtype)

        # Pool references (current layer). Kernel reads these directly.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.pool_refs"):
            layer_id_rel = layer.layer_id - self._tq_pool.start_layer
            k_nope_packed = self._tq_pool.kv_nope_packed_buffer[layer_id_rel]
            k_scale = self._tq_pool.kv_nope_scale_buffer[layer_id_rel]
            k_rope = self._tq_pool.kv_rope_buffer[layer_id_rel]
            k_centroids = self._tq_config.k_centroids
            uniform = getattr(self._tq_config, "uniform", False)

        # Read pre-populated CG-safe buffers. kv_indptr is populated for
        # indices [0..bs]; kv_indices first `kv_indptr[bs]` entries are
        # valid. Kernel reads only through kv_indptr so the tail garbage
        # in kv_indices is never touched.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.index_views"):
            kv_indptr = self._tq_kv_indptr[: bs + 1]
            kv_indices = self._tq_kv_indices  # full buffer; kernel bounds by kv_indptr
            num_kv_splits = self._tq_num_kv_splits[:bs]

        # Narrow persistent output buffers to current bs. Slicing returns
        # views with stable base addresses — CG-safe.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.output_views"):
            att_logits = self._tq_stage1_logits[:bs, :q_heads, :, :]
            att_lse = self._tq_stage1_lse[:bs, :q_heads, :]
            o_rotated = self._tq_o_rotated[:bs, :q_heads, :]

        with _tq_mla_nvtx_range("omniva.tq_mla.decode.tq_attention"):
            tq_mla_decode_attention_fwd(
                q_nope_rotated=q_nope_rot,
                q_rope=q_rope,
                k_nope_packed=k_nope_packed,
                k_scale=k_scale,
                k_rope=k_rope,
                k_centroids=k_centroids,
                o=o_rotated,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                att_logits=att_logits,
                att_lse=att_lse,
                num_kv_splits=num_kv_splits,
                max_kv_splits=max_splits,
                sm_scale=layer.scaling,
                logit_cap=getattr(layer, "logit_cap", 0.0) or 0.0,
                uniform=uniform,
            )

        # Kernel output is in rotated nope space. Inverse-rotate once on
        # the small (bs, q_heads, lora_rank) tensor to get original domain.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.inverse_rotate"):
            o = self._tq_config.inverse_rotate_output(o_rotated).to(q_nope.dtype)

        # Flashmla backend contract: (bs * q_heads, v_head_dim). Match that.
        with _tq_mla_nvtx_range("omniva.tq_mla.decode.output_view"):
            return o.view(-1, q_heads * lora_rank)
