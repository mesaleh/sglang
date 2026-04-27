"""
Support attention backend for FlashMLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

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

        # Pre-allocated stage-1 output + LSE. Sized for max_bs * max_q_heads *
        # max_kv_splits * lora_rank. Re-used across forwards; stable address
        # under CUDA graph capture.
        # We allocate lazily on first forward_decode (we don't know max_bs
        # here at init time in a CG-safe way).
        self._stage1_logits = None
        self._stage1_lse = None

    def _ensure_stage1_buffers(self, bs: int, max_kv_splits: int, device, dtype=torch.float32):
        """Lazy-alloc stage-1 partials buffer. Grows if larger bs arrives;
        steady-state address is stable so CUDA graph capture sees fixed size
        (because bs is itself a capture-time constant)."""
        q_heads = self.num_q_heads
        lora_rank = self.kv_lora_rank
        needed_logits_shape = (bs, q_heads, max_kv_splits, lora_rank)
        needed_lse_shape = (bs, q_heads, max_kv_splits)
        if (
            self._stage1_logits is None
            or tuple(self._stage1_logits.shape) != needed_logits_shape
        ):
            self._stage1_logits = torch.empty(
                needed_logits_shape, dtype=dtype, device=device
            )
            self._stage1_lse = torch.empty(
                needed_lse_shape, dtype=dtype, device=device
            )

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

        # Q comes in shape (bs, q_heads * full_dim) or (bs, 1, q_heads, full_dim).
        # Reshape to (bs, q_heads, full_dim), then split nope | rope.
        reshape_q = q.view(bs, q_heads, full_dim)
        q_nope = reshape_q[:, :, :lora_rank].contiguous()
        q_rope = reshape_q[:, :, lora_rank:].contiguous()

        # --- Fix 2 (Phase 1 findings): rotate Q_nope into WHT domain ---
        # This is the move that activates the orthogonality trick, letting the
        # kernel dot rotated Q against rotated K without needing per-row
        # inverse Hadamard.
        # rotate_query returns fp32 regardless of input dtype (CUDA kernel
        # convention). Cast back to bf16 to match kernel's expected dtype
        # and keep register pressure bounded.
        q_nope_rot = self._tq_config.rotate_query(q_nope).to(q_nope.dtype)

        # Pool references (current layer). Kernel reads these directly.
        layer_id_rel = layer.layer_id - self._tq_pool.start_layer
        k_nope_packed = self._tq_pool.kv_nope_packed_buffer[layer_id_rel]
        k_scale = self._tq_pool.kv_nope_scale_buffer[layer_id_rel]
        k_rope = self._tq_pool.kv_rope_buffer[layer_id_rel]
        k_centroids = self._tq_config.k_centroids
        uniform = getattr(self._tq_config, "uniform", False)

        # Build TOKEN-indexed flat kv_indices for the fused kernel.
        #
        # IMPORTANT: cannot reuse self.forward_metadata.block_kv_indices here.
        # That tensor is PAGE-indexed (page_id = pool_row // PAGE_SIZE=64),
        # produced by create_flashmla_kv_indices_triton for flashmla's native
        # page-based reader (`k_cache.view(-1, PAGE_SIZE, ...)` with block_table).
        # Our kernel reads individual pool rows, so we need per-TOKEN pool
        # indices. Same pattern as triton_backend.py init_forward_metadata.
        seq_lens = forward_batch.seq_lens[:bs].to(torch.int32)
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=seq_lens.device)
        # Assignment does implicit int64→int32 cast (pattern from triton_backend.py).
        kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
        # Total tokens across batch. .item() is a host-sync — acceptable for v1
        # since CUDA graphs are off on this backend anyway (Phase 3 cleans this).
        total_kv_tokens = int(kv_indptr[bs].item())
        kv_indices = torch.empty(
            total_kv_tokens, dtype=torch.int32, device=seq_lens.device
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            forward_batch.req_pool_indices[:bs],
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
        )

        # num_kv_splits: reuse flashmla metadata for the split count.
        num_kv_splits = self.forward_metadata.num_splits[:bs].to(torch.int32)
        # max_kv_splits: an upper bound to size the stage-1 output.
        max_kv_splits = int(num_kv_splits.max().item()) if bs > 0 else 1
        # ^ NOTE: .item() here is a host-sync and NOT CG-compatible. For v1 we
        # accept it (CUDA graphs will be off on this backend until Phase 3).
        # Remove before shipping with graphs on.

        # Stage-1 output buffers
        self._ensure_stage1_buffers(bs, max_kv_splits, q_nope.device)

        # Output in rotated nope space (stage-2 writes here)
        o_rotated = torch.empty(
            (bs, q_heads, lora_rank), dtype=q_nope.dtype, device=q_nope.device
        )

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
            att_logits=self._stage1_logits,
            att_lse=self._stage1_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=max_kv_splits,
            sm_scale=layer.scaling,
            logit_cap=getattr(layer, "logit_cap", 0.0) or 0.0,
            uniform=uniform,
        )

        # --- Fix 1 (Phase 1 findings): un-rotate the output ---
        # Kernel output is in rotated nope space. Apply inverse Hadamard once
        # on the small (bs, q_heads, lora_rank) tensor via the CUDA helper.
        o = self._tq_config.inverse_rotate_output(o_rotated).to(q_nope.dtype)

        # Flashmla backend returns (bs * q_heads, v_head_dim). Match that.
        return o.view(-1, q_heads * lora_rank)
