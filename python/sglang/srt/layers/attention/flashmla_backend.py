"""
Support attention backend for FlashMLA.
"""

from __future__ import annotations

import logging
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
    get_num_kv_index_blocks_flashmla,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

PAGE_SIZE = 64
_TQ_MLA_FAST_METADATA = envs.SGLANG_TQ_MLA_FAST_METADATA.get()
_TQ_MLA_FUSED_METADATA_INDICES = envs.SGLANG_TQ_MLA_FUSED_METADATA_INDICES.get()
_TQ_MLA_STAGED_FLASHMLA = envs.SGLANG_TQ_MLA_STAGED_FLASHMLA.get()
_TQ_MLA_STAGED_FLASHMLA_THREADS = envs.SGLANG_TQ_MLA_STAGED_FLASHMLA_THREADS.get()
_TQ_MLA_PROFILE_NVTX = envs.SGLANG_TQ_MLA_PROFILE_NVTX.get()


def _tq_staged_flashmla_supported(device: torch.device) -> bool:
    if not _TQ_MLA_STAGED_FLASHMLA:
        return False
    if not device.type == "cuda" or not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability(device)
    # The staged path materializes BF16 dense KV and calls FlashMLA dense decode.
    # Current FlashMLA supports that dense decode operator on Hopper/SM90 only;
    # Blackwell/SM100 supports different sparse/FP8 MLA paths, so forcing this
    # route on GB200 fails at runtime.
    return major == 9


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

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_decode_or_idle() or forward_mode.is_target_verify():
            self._apply_decode_target_verify_metadata(
                bs=forward_batch.batch_size,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=forward_mode,
            )
        else:
            super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

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
            create_flashmla_kv_indices_triton[
                (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
            ](
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
            create_flashmla_kv_indices_triton[
                (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
            ](
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

    def _apply_decode_target_verify_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
    ):
        """Shared decode/target-verify capture+replay body.

        Public entry: :py:meth:`init_forward_metadata_out_graph` (which routes
        to this helper for decode/target-verify and falls back to the
        FlashInferMLA parent for prefill/draft-extend).
        """
        if True:
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None

            if forward_mode.is_target_verify():
                seq_lens = seq_lens + self.num_draft_tokens
                if seq_lens_cpu is not None:
                    seq_lens_cpu = seq_lens_cpu + self.num_draft_tokens

            seq_max = (
                seq_lens_cpu.max().item()
                if seq_lens_cpu is not None
                else seq_lens.max().item()
            )
            max_seqlen_pad = triton.cdiv(seq_max, PAGE_SIZE)

            create_flashmla_kv_indices_triton[
                (
                    bs,
                    get_num_kv_index_blocks_flashmla(
                        self.cuda_graph_kv_indices.stride(0), PAGE_SIZE
                    ),
                )
            ](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            q_head_mult = (
                self.num_draft_tokens if forward_mode.is_target_verify() else 1
            )
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                q_head_mult * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            if (
                self.cuda_graph_mla_metadata_view is None
                or actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]
            ):
                if self.cuda_graph_mla_metadata_view is not None:
                    logger.warning(
                        f"num_sm_parts mismatch in CUDA Graph replay: "
                        f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                        f"replay={actual_num_sm_parts}. "
                        f"This may indicate batch size changed between capture and replay."
                    )
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
            # num_splits has shape (bs+1,) — always update for the current bs.
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
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
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

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
        if forward_batch.forward_mode in (
            ForwardMode.EXTEND,
            ForwardMode.DRAFT_EXTEND,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            cache_loc = forward_batch.out_cache_loc

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            bs = forward_batch.batch_size
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

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

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardMode,
            build_inner_fb_view,
        )

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, call_fn)
    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


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
        self._tq_staged_k_cache = None     # (pool_tokens, 1, lora_rank + rope_dim) bf16
        self._tq_staged_q = None           # (max_bs, 1, q_heads, lora_rank + rope_dim) bf16
        self._tq_staged_pages_per_req = 1
        self._tq_use_staged_flashmla = _tq_staged_flashmla_supported(
            self.req_to_token.device
        )

        # Boot-time visibility: confirms this class (not base FlashMLABackend)
        # is actually instantiated in the live pod. One-line per rank at init.
        import logging

        if _TQ_MLA_STAGED_FLASHMLA and not self._tq_use_staged_flashmla:
            logging.getLogger(__name__).warning(
                "TurboQuant staged FlashMLA requested but disabled on device "
                "capability %s. Falling back to the non-staged TurboQuant MLA "
                "decode path.",
                torch.cuda.get_device_capability(self.req_to_token.device),
            )
        logging.getLogger(__name__).info(
            "TurboQuantMLABackend active. staged_flashmla_requested=%s "
            "staged_flashmla_active=%s",
            _TQ_MLA_STAGED_FLASHMLA,
            self._tq_use_staged_flashmla,
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
            and (
                not self._tq_use_staged_flashmla
                or (
                    self._tq_staged_k_cache is not None
                    and self._tq_staged_q is not None
                    and self._tq_staged_q.shape[0] >= max_bs
                )
            )
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
        if self._tq_use_staged_flashmla:
            full_dim = self.kv_lora_rank + self.qk_rope_head_dim
            source_tokens = self._tq_pool.kv_nope_packed_buffer[0].shape[0]
            staged_tokens = triton.cdiv(
                source_tokens, self._tq_pool.page_size
            ) * self._tq_pool.page_size
            self._tq_staged_k_cache = torch.empty(
                (
                    staged_tokens,
                    1,
                    full_dim,
                ),
                dtype=torch.bfloat16,
                device=device,
            )
            self._tq_staged_q = torch.empty(
                (max_bs, 1, q_heads, full_dim),
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

    def _tq_warmup_staged_flashmla(self, device):
        """Pre-compile the staged CUDA op and FlashMLA path before graph capture."""
        from sglang.srt.layers.attention.turboquant_mla_staged_flashmla import (
            stage_tq_mla_pages_to_physical,
        )

        q_heads = self.num_q_heads
        lora_rank = self.kv_lora_rank
        rope_dim = self.qk_rope_head_dim
        full_dim = lora_rank + rope_dim

        q_flash = torch.zeros(
            (1, 1, q_heads, full_dim), dtype=self.q_data_type, device=device
        )
        seq_lens = torch.tensor([64], dtype=torch.int32, device=device)
        req_to_token = torch.arange(64, dtype=torch.int32, device=device).view(1, 64)
        req_pool_indices = torch.zeros((1,), dtype=torch.int32, device=device)
        block_kv_indices = torch.zeros((1, 1), dtype=torch.int32, device=device)
        mla_metadata, num_splits = get_mla_metadata(
            seq_lens,
            q_heads,
            1,
            is_fp8_kvcache=False,
        )

        layer_id_rel = 0
        stage_tq_mla_pages_to_physical(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            k_nope_packed=self._tq_pool.kv_nope_packed_buffer[layer_id_rel],
            k_scale=self._tq_pool.kv_nope_scale_buffer[layer_id_rel],
            k_rope=self._tq_pool.kv_rope_buffer[layer_id_rel],
            k_centroids=self._tq_config.k_centroids,
            out_k_cache=self._tq_staged_k_cache,
            req_stride=req_to_token.stride(0),
            pages_per_req=1,
            threads=_TQ_MLA_STAGED_FLASHMLA_THREADS,
        )
        flash_mla_with_kvcache(
            q=q_flash,
            k_cache=self._tq_staged_k_cache.view(-1, PAGE_SIZE, 1, full_dim),
            block_table=block_kv_indices,
            cache_seqlens=seq_lens,
            head_dim_v=lora_rank,
            tile_scheduler_metadata=mla_metadata,
            num_splits=num_splits,
            softmax_scale=self.scaling,
            causal=True,
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
            if self._tq_use_staged_flashmla:
                self._tq_warmup_staged_flashmla(device)
            else:
                self._tq_warmup_kernel(device)

    def _forward_decode_staged_flashmla(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        q_nope_rot: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope_packed: torch.Tensor,
        k_scale: torch.Tensor,
        k_rope: torch.Tensor,
        k_centroids: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.turboquant_mla_staged_flashmla import (
            stage_tq_mla_pages_to_physical,
        )

        bs = forward_batch.batch_size
        q_heads = layer.tp_q_head_num
        lora_rank = self.kv_lora_rank
        rope_dim = self.qk_rope_head_dim
        full_dim = lora_rank + rope_dim

        with _tq_mla_nvtx_range("omniva.tq_mla.decode.staged_q_pack"):
            q_flash = self._tq_staged_q[:bs, :, :q_heads, :]
            q_flash[:, 0, :, :lora_rank].copy_(q_nope_rot)
            q_flash[:, 0, :, lora_rank:].copy_(q_rope)

        seq_lens_i32 = self._tq_seq_lens_i32[:bs]
        pages_per_req = self._tq_staged_pages_per_req

        with _tq_mla_nvtx_range("omniva.tq_mla.decode.stage_tq_pages"):
            stage_tq_mla_pages_to_physical(
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices[:bs],
                seq_lens=seq_lens_i32,
                k_nope_packed=k_nope_packed,
                k_scale=k_scale,
                k_rope=k_rope,
                k_centroids=k_centroids,
                out_k_cache=self._tq_staged_k_cache,
                req_stride=self.req_to_token.stride(0),
                pages_per_req=pages_per_req,
                threads=_TQ_MLA_STAGED_FLASHMLA_THREADS,
            )

        with _tq_mla_nvtx_range("omniva.tq_mla.decode.flashmla_workspace_attention"):
            out, _ = flash_mla_with_kvcache(
                q=q_flash,
                k_cache=self._tq_staged_k_cache.view(-1, PAGE_SIZE, 1, full_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=seq_lens_i32,
                head_dim_v=lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )

        with _tq_mla_nvtx_range("omniva.tq_mla.decode.staged_inverse_rotate"):
            o = self._tq_config.inverse_rotate_output(out[:, 0, :, :]).to(q.dtype)

        return o.view(-1, q_heads * lora_rank)

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
        if not torch.is_inference_mode_enabled():
            # CUDA graph capture can call this outside InferenceMode while the
            # persistent metadata buffers were allocated as inference tensors.
            # Run all metadata writes in inference mode to keep PyTorch 2.11
            # from rejecting in-place updates to those buffers.
            with torch.inference_mode():
                return self._tq_build_kv_indices(bs, req_pool_indices, seq_lens)

        if self._tq_seq_lens_i32 is not None and self._tq_seq_lens_i32.numel() >= bs:
            if seq_lens.dtype == torch.int32:
                self._tq_seq_lens_i32[:bs].copy_(seq_lens[:bs])
            else:
                self._tq_seq_lens_i32[:bs].copy_(seq_lens[:bs].to(torch.int32))

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
            if self._tq_use_staged_flashmla:
                self._tq_staged_pages_per_req = triton.cdiv(
                    forward_batch.seq_lens_cpu[:bs].max().item(), PAGE_SIZE
                )
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
            if self._tq_use_staged_flashmla:
                # CUDA graph capture records the kernel launch grid, so the
                # staged kernel must launch with a static page count. The
                # kernel reads device seq_lens and returns early for invalid
                # pages, preserving correctness on replay with longer prompts.
                self._tq_staged_pages_per_req = self.cuda_graph_kv_indices.shape[1]
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
            if self._tq_use_staged_flashmla:
                self._tq_staged_pages_per_req = self.cuda_graph_kv_indices.shape[1]
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
                    self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

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

        if self._tq_use_staged_flashmla:
            return self._forward_decode_staged_flashmla(
                q,
                layer,
                forward_batch,
                q_nope_rot,
                q_rope,
                k_nope_packed,
                k_scale,
                k_rope,
                k_centroids,
            )

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
