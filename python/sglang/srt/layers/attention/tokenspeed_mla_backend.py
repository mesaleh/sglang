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

Subclasses :class:`TRTLLMMLABackend`, swaps its decode/prefill kernels, and
provides an optional absorbed-MLA frontend hook. Metadata, KV-cache layout,
CUDA-graph plumbing, draft-extend padding, and chunked-prefix dispatch remain
inherited from the parent.
"""

import json
import logging
from inspect import signature
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.fp8_quantize import fp8_quantize
from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8
from sglang.jit_kernel.tq_mla_frontend import (
    get_tq_mla_frontend_module_attestation,
    preload_tq_mla_frontend_prebuilt,
    tq_mla_frontend_out,
)
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.attention.tokenspeed_workspace import (
    tokenspeed_workspace_bytes,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
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

_g_tokenspeed_workspace: dict[torch.device, torch.Tensor] = {}
_TQ4_CODEBOOK_CUDA_GRAPH_MAX_SEQ_LEN = 32_768


def _supports_custom_decode_mask(decode_fn) -> bool:
    parameters = signature(decode_fn).parameters
    return "custom_mask" in parameters and "cmask_off" in parameters


def _supports_decode_lse(decode_fn) -> bool:
    return "return_lse" in signature(decode_fn).parameters


def _find_mla_turboquant_pool(token_to_kv_pool):
    if getattr(token_to_kv_pool, "is_mla_turboquant_pool", False):
        return token_to_kv_pool
    wrapped_pool = getattr(token_to_kv_pool, "full_kv_pool", None)
    if getattr(wrapped_pool, "is_mla_turboquant_pool", False):
        return wrapped_pool
    return None


def _validate_h43_frontend_activation(
    enabled: bool, is_draft_worker: bool, num_q_heads: int
) -> None:
    if not enabled:
        return
    if is_draft_worker:
        raise RuntimeError("H43 frontend is supported only on the target model")
    if num_q_heads != 8:
        raise RuntimeError(
            "H43 frontend requires exactly 8 local query heads; " f"got {num_q_heads}"
        )


def _has_populated_tq_codebook(
    codebook_buffer: Optional[list[Optional[torch.Tensor]]],
) -> bool:
    return codebook_buffer is not None and any(
        value is not None for value in codebook_buffer
    )


def _tq4_split_override(batch_size: int, max_seq_len: int, num_sms: int) -> int:
    # Keep graph-capture shapes on one compact CuTe specialization.  The
    # split count stays fixed across graph batch sizes; packed tile traversal
    # is runtime-rolled when a configured context assigns more than four
    # 128-token tiles to each split. Sixty-four splits is already the c1
    # decode choice on GB200.
    del batch_size, num_sms
    tiles = (max_seq_len + 127) // 128
    return min(tiles, 64)


def _tq4_kernel_max_seq_len(requested: int, configured: int) -> int:
    if configured > 262144:
        raise ValueError(
            f"native TurboQuant MLA requires context_length <= 262144; got {configured}"
        )
    return min(requested, configured)


def _tq4_codebook_cuda_graph_max_seq_len(configured: int, has_codebook: bool) -> int:
    return (
        min(configured, _TQ4_CODEBOOK_CUDA_GRAPH_MAX_SEQ_LEN)
        if has_codebook
        else configured
    )


def _tq4_codebook_cuda_graph_eligible(
    seq_lens_cpu: Optional[torch.Tensor],
    extra_kv_tokens: int,
    graph_max_seq_len: int,
) -> bool:
    if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
        return False
    return int(seq_lens_cpu.max().item()) + extra_kv_tokens <= graph_max_seq_len


def _tq4_workspace_bytes(
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    q_len: int,
    split_kv: int,
) -> int:
    if split_kv == 1:
        return 0
    return batch_size * num_heads * q_len * split_kv * (kv_lora_rank + 1) * 4


@triton.jit
def _fp8_quantize_tq4_query_kernel(
    x_ptr,
    out_ptr,
    m_rows,
    x_row_stride,
    out_row_stride,
    LATENT_N: tl.constexpr,
    ROPE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < m_rows
    latent_idx = tl.arange(0, LATENT_N)
    rope_idx = tl.arange(0, ROPE_N)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    x_row = m_idx[:, None] * x_row_stride
    out_row = m_idx[:, None] * out_row_stride
    latent = tl.load(x_ptr + x_row + latent_idx[None, :], mask=m_mask[:, None])
    rope = tl.load(x_ptr + x_row + LATENT_N + rope_idx[None, :], mask=m_mask[:, None])
    tl.store(
        out_ptr + out_row + latent_idx[None, :],
        latent.to(tl.float8e4nv),
        mask=m_mask[:, None],
    )
    tl.store(
        out_ptr + out_row + LATENT_N + rope_idx[None, :],
        rope.to(tl.float8e4nv),
        mask=m_mask[:, None],
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _quantize_tq4_query(
    query: torch.Tensor, kv_lora_rank: int, enable_pdl: bool
) -> torch.Tensor:
    if kv_lora_rank != 512 or query.shape[-1] != 576:
        raise ValueError(
            "native TurboQuant query quantization requires latent/total "
            f"dimensions 512/576; got {kv_lora_rank}/{query.shape[-1]}"
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "native TurboQuant query quantization requires BF16/FP16 input; "
            f"got {query.dtype}"
        )
    if query.stride(-1) != 1:
        raise ValueError(
            "native TurboQuant query quantization requires a contiguous "
            f"inner dimension; got stride {query.stride()}"
        )
    for dim in range(query.ndim - 2):
        expected_stride = query.shape[dim + 1] * query.stride(dim + 1)
        if query.stride(dim) != expected_stride:
            raise ValueError(
                "native TurboQuant query quantization cannot flatten leading "
                f"dimension {dim}: got stride {query.stride(dim)}, expected "
                f"{expected_stride} for shape {tuple(query.shape)}"
            )
    query_fp8 = torch.empty(query.shape, dtype=torch.float8_e4m3fn, device=query.device)
    rows = query.numel() // query.shape[-1]
    row_stride = query.stride(-2) if query.ndim > 1 else query.shape[-1]
    out_row_stride = query_fp8.stride(-2) if query_fp8.ndim > 1 else query_fp8.shape[-1]
    block_m = 4 if rows <= 2048 else 16 if rows <= 16384 else 32
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _fp8_quantize_tq4_query_kernel[(triton.cdiv(rows, block_m),)](
        query,
        query_fp8,
        rows,
        row_stride,
        out_row_stride,
        LATENT_N=512,
        ROPE_N=64,
        BLOCK_M=block_m,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=2,
        **extra_kwargs,
    )
    return query_fp8


def _get_tokenspeed_workspace(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    q_len: int,
    minimum_bytes: int = 0,
) -> torch.Tensor:
    needed = max(
        minimum_bytes,
        tokenspeed_workspace_bytes(
            tokenspeed_mla.get_num_sm(device), num_heads, kv_lora_rank, q_len
        ),
    )
    existing = _g_tokenspeed_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_tokenspeed_workspace[device] = torch.empty(
            needed, dtype=torch.int8, device=device
        )
    return _g_tokenspeed_workspace[device]


# TODO(Qiaolin-Yu): Merge this attention backend into trtllm_mla_backend.py
# once the same CuteDSL kernels in flashinfer_trtllm are stable
# and there is no performance gap compared to this backend.
class TokenspeedMLABackend(TRTLLMMLABackend):
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100, FP8 KV)."""

    supports_custom_decode_mask: bool = is_tokenspeed_mla_available() and (
        _supports_custom_decode_mask(tokenspeed_mla.tokenspeed_mla_decode)
    )
    supports_decode_lse: bool = is_tokenspeed_mla_available() and (
        _supports_decode_lse(tokenspeed_mla.tokenspeed_mla_decode)
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

        self._tq_pool = _find_mla_turboquant_pool(self.token_to_kv_pool)
        self._tq4_cache = self._tq_pool is not None
        self._tq4_all_layers = bool(
            self._tq4_cache and getattr(self._tq_pool, "all_layers_turboquant", True)
        )
        self._tq4_hotcold_cache = bool(
            self._tq4_cache
            and getattr(self._tq_pool, "is_mla_turboquant_hotcold_pool", False)
        )
        self._tq_config = self._tq_pool.tq_config if self._tq4_cache else None
        local_tq_layers = (
            bool(getattr(self._tq_pool, "_tq_layer_ids_rel", ()))
            if self._tq4_cache
            else False
        )
        if self._tq4_cache and not hasattr(self._tq_pool, "_tq_layer_ids_rel"):
            local_tq_layers = any(
                value is not None
                for value in getattr(self._tq_pool, "kv_rope_buffer", ())
            )
        self._h43_frontend = bool(
            self._tq4_cache
            and local_tq_layers
            and getattr(self._tq_pool, "enable_h43_frontend", False)
        )
        self._h43_frontend_dispatch_count = 0
        self._h43_reader_dispatch_count = 0
        self._h43_frontend_dispatches_by_mode: dict[str, int] = {}
        self._h43_reader_dispatches_by_mode: dict[str, int] = {}
        self._h43_frontend_query_dtypes: dict[str, int] = {}
        self._h43_reader_query_dtypes: dict[str, int] = {}
        self._h43_first_success_events: set[tuple[str, str]] = set()
        self._h43_model_runner = model_runner
        self._h43_is_draft_worker = bool(model_runner.is_draft_worker)

        if (
            self._tq4_cache
            and bool(getattr(self._tq_config, "e2m1", False))
            and _has_populated_tq_codebook(self._tq_pool.kv_nope_codebook_buffer)
            and not self._h43_frontend
        ):
            raise RuntimeError(
                "TokenSpeed E2M1 must not allocate the redundant FP8 lookup row"
            )
        _validate_h43_frontend_activation(
            self._h43_frontend,
            self._h43_is_draft_worker,
            self.num_q_heads,
        )
        if self._h43_frontend:
            if not bool(getattr(self._tq_config, "e2m1", False)):
                raise RuntimeError("H43 frontend requires E2M1 TurboQuant")
            if self._tq_pool.kv_nope_codebook_buffer is None:
                raise RuntimeError("H43 frontend requires the FP8 codebook pool")
            selected_rope = next(
                (value for value in self._tq_pool.kv_rope_buffer if value is not None),
                None,
            )
            if selected_rope is None or selected_rope.dtype != torch.float8_e4m3fn:
                raise RuntimeError("H43 frontend requires an FP8 RoPE pool")
            if self._tq_pool.page_size != 32:
                raise RuntimeError("H43 frontend requires page_size=32")
            if torch.cuda.get_device_capability(selected_rope.device) != (10, 0):
                raise RuntimeError("H43 frontend requires an SM100 GPU")
            tq4_parameters = signature(
                tokenspeed_mla.tokenspeed_mla_decode_tq4
            ).parameters
            if (
                "kv_nope_codebook" not in tq4_parameters
                or "fp8_rope" not in tq4_parameters
            ):
                raise RuntimeError(
                    "installed TokenSpeed lacks the H43 codebook/FP8-RoPE reader"
                )
            preload_tq_mla_frontend_prebuilt()
        if not self._tq4_cache and self.data_type != torch.float8_e4m3fn:
            raise ValueError(
                "tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got data_type={self.data_type}."
            )
        if self._tq4_cache and self.page_size != 32:
            raise ValueError(
                "native TurboQuant tokenspeed_mla requires page_size=32, "
                f"got page_size={self.page_size}."
            )
        if self._tq4_cache:
            _tq4_kernel_max_seq_len(self.max_context_len, self.max_context_len)
        if not self._tq4_cache and self.page_size not in (32, 64):
            raise ValueError(
                "tokenspeed_mla backend requires page_size in {32, 64}, "
                f"got page_size={self.page_size}."
            )
        if self._tq4_cache and not hasattr(tokenspeed_mla, "tokenspeed_mla_decode_tq4"):
            raise RuntimeError(
                "installed tokenspeed_mla does not provide native TQ4 decode"
            )
        if self._tq4_hotcold_cache and not self.supports_custom_decode_mask:
            raise RuntimeError(
                "static hot/cold TurboQuant requires a TokenSpeed decode "
                "kernel with custom_mask and cmask_off support"
            )
        if self._tq4_hotcold_cache and not self.supports_decode_lse:
            raise RuntimeError(
                "static hot/cold TurboQuant requires a TokenSpeed decode "
                "kernel with return_lse support"
            )
        if self._tq4_hotcold_cache and not hasattr(
            tokenspeed_mla, "merge_attention_outputs_base2"
        ):
            raise RuntimeError(
                "static hot/cold TurboQuant requires TokenSpeed "
                "merge_attention_outputs_base2 support"
            )

        self._tokenspeed_workspace: Optional[torch.Tensor] = None
        if is_tokenspeed_mla_available():
            workspace_q_len = max(1, self.num_draft_tokens or 1)
            tq4_minimum_bytes = 0
            if self._tq4_cache:
                decode_graph = model_runner.server_args.cuda_graph_config.decode
                graph_max_batch = max(1, decode_graph.max_bs or 1)
                graph_split_kv = _tq4_split_override(
                    graph_max_batch,
                    self.max_context_len,
                    tokenspeed_mla.get_num_sm(self.device),
                )
                tq4_minimum_bytes = _tq4_workspace_bytes(
                    graph_max_batch,
                    self.num_q_heads,
                    self.kv_lora_rank,
                    workspace_q_len,
                    graph_split_kv,
                )
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                self.device,
                self.num_q_heads,
                self.kv_lora_rank,
                workspace_q_len,
                minimum_bytes=tq4_minimum_bytes,
            )

            # Pre-JIT the prefill kernel variants. Each cute.compile takes 1-2
            # min; without warm-up the first request trips the 300 s scheduler
            # watchdog.
            # Plain TQ4 delegates prefill to the TRT-LLM parent; FP8 and the
            # hot/cold pool both dispatch to TokenSpeed prefill below.
            if not (self._tq4_all_layers and not self._tq4_hotcold_cache):
                _compile_prefill_kernel = (
                    tokenspeed_mla.mla_prefill._compile_prefill_kernel
                )
                _compiled_kernels = tokenspeed_mla.mla_prefill._compiled_kernels
                head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
                enable_ex2_emulation = (
                    tokenspeed_mla.mla_prefill._enable_ex2_emulation()
                )
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

    def _h43_attestation_payload(self, phase: str) -> dict[str, object]:
        pool = getattr(self, "_tq_pool", None)
        tq_enabled = pool is not None
        h43_enabled = bool(getattr(self, "_h43_frontend", False))
        tq_config = getattr(self, "_tq_config", None)
        e2m1 = bool(getattr(tq_config, "e2m1", False))

        if tq_enabled:
            start_layer = int(getattr(pool, "start_layer", 0))
            layer_count = int(
                getattr(
                    pool,
                    "layer_num",
                    len(getattr(pool, "kv_rope_buffer", ())),
                )
            )
            configured_ids = getattr(pool, "turboquant_layer_ids", None)
            if configured_ids is None:
                end_layer = int(
                    getattr(
                        pool,
                        "end_layer",
                        start_layer + layer_count,
                    )
                )
                selected_layer_ids = tuple(range(start_layer, end_layer))
            else:
                end_layer = start_layer + layer_count
                selected_layer_ids = tuple(
                    sorted(
                        int(i)
                        for i in configured_ids
                        if start_layer <= int(i) < end_layer
                    )
                )
            codebook_buffer = getattr(pool, "kv_nope_codebook_buffer", None)
            codebook_layer_ids = (
                tuple(
                    start_layer + i
                    for i, value in enumerate(codebook_buffer)
                    if value is not None
                )
                if codebook_buffer is not None
                else ()
            )
            rope_buffer = getattr(pool, "kv_rope_buffer", ())
            fp8_rope_layer_ids = tuple(
                start_layer + i
                for i, value in enumerate(rope_buffer)
                if value is not None and value.dtype == torch.float8_e4m3fn
            )
            codebook_slots = len(codebook_layer_ids)
            fp8_rope_slots = len(fp8_rope_layer_ids)
            rope_item_bytes = 1 if fp8_rope_layer_ids else 2
            selected_row_bytes = (
                int(self.kv_lora_rank) // 2
                + 2
                + int(self.qk_rope_head_dim) * rope_item_bytes
                + (16 if codebook_slots else 0)
            )
            get_pool_bytes = getattr(pool, "get_per_token_all_layer_bytes", None)
            pool_all_layer_bytes = (
                int(get_pool_bytes()) if callable(get_pool_bytes) else None
            )
            expected_pool_all_layer_bytes = selected_row_bytes * len(
                selected_layer_ids
            ) + (int(self.kv_lora_rank) + int(self.qk_rope_head_dim)) * (
                layer_count - len(selected_layer_ids)
            )
        else:
            selected_layer_ids = ()
            codebook_layer_ids = ()
            fp8_rope_layer_ids = ()
            codebook_slots = 0
            fp8_rope_slots = 0
            selected_row_bytes = int(self.kv_lora_rank) + int(self.qk_rope_head_dim)
            pool_all_layer_bytes = None
            expected_pool_all_layer_bytes = None

        frontend_fault_status = None
        if h43_enabled and phase == "post_cuda_graph":
            status = getattr(pool, "tq_mla_frontend_fault_status", None)
            if status is None or status.numel() != 1:
                raise RuntimeError("H43 attestation requires one sticky fault word")
            frontend_fault_status = int(status.item())

        runner = getattr(self, "_h43_model_runner", None)
        payload: dict[str, object] = {
            "schema": 1,
            "phase": phase,
            "tp_rank": int(getattr(runner, "tp_rank", -1)),
            "pp_rank": int(getattr(runner, "pp_rank", -1)),
            "gpu_id": int(getattr(runner, "gpu_id", -1)),
            "draft_worker": bool(getattr(self, "_h43_is_draft_worker", False)),
            "tq_enabled": tq_enabled,
            "h43_enabled": h43_enabled,
            "e2m1": e2m1,
            "selected_layer_ids": list(selected_layer_ids),
            "selected_layer_count": len(selected_layer_ids),
            "selected_row_bytes": selected_row_bytes,
            "pool_all_layer_bytes": pool_all_layer_bytes,
            "expected_pool_all_layer_bytes": expected_pool_all_layer_bytes,
            "codebook_layer_ids": list(codebook_layer_ids),
            "codebook_slots": codebook_slots,
            "fp8_rope_layer_ids": list(fp8_rope_layer_ids),
            "fp8_rope_slots": fp8_rope_slots,
            "frontend_fault_status": frontend_fault_status,
            "frontend_path": "mla_absorb" if h43_enabled else None,
            "frontend_dispatches": int(
                getattr(self, "_h43_frontend_dispatch_count", 0)
            ),
            "reader_dispatches": int(getattr(self, "_h43_reader_dispatch_count", 0)),
            "frontend_dispatches_by_mode": dict(
                getattr(self, "_h43_frontend_dispatches_by_mode", {})
            ),
            "reader_dispatches_by_mode": dict(
                getattr(self, "_h43_reader_dispatches_by_mode", {})
            ),
            "frontend_query_dtypes": dict(
                getattr(self, "_h43_frontend_query_dtypes", {})
            ),
            "reader_query_dtypes": dict(getattr(self, "_h43_reader_query_dtypes", {})),
        }
        payload.update(get_tq_mla_frontend_module_attestation())
        return payload

    def _validate_h43_attestation(self, payload: dict[str, object]) -> None:
        if payload["phase"] not in {"init", "post_cuda_graph"}:
            raise RuntimeError(f"unknown H43 attestation phase: {payload['phase']}")
        if payload["draft_worker"]:
            return

        h43_enabled = bool(payload["h43_enabled"])
        e2m1 = bool(payload["e2m1"])
        tq_enabled = bool(payload["tq_enabled"])
        if h43_enabled:
            if not e2m1:
                raise RuntimeError("H43 attestation requires E2M1 storage")
            if not (
                payload["native_loaded"]
                and payload["native_prebuilt"]
                and payload["native_path"]
                and payload["native_sha256"]
            ):
                raise RuntimeError("H43 attestation requires a pinned prebuilt module")
            if payload["selected_row_bytes"] != 338:
                raise RuntimeError(
                    "H43 attestation requires a 338-byte selected row; "
                    f"got {payload['selected_row_bytes']}"
                )
            if payload["selected_layer_count"] <= 0:
                raise RuntimeError("H43 attestation requires selected layers")
            if payload["codebook_layer_ids"] != payload["selected_layer_ids"]:
                raise RuntimeError("H43 attestation codebook layers do not match")
            if payload["fp8_rope_layer_ids"] != payload["selected_layer_ids"]:
                raise RuntimeError("H43 attestation FP8 RoPE layers do not match")
            if (
                payload["pool_all_layer_bytes"]
                != payload["expected_pool_all_layer_bytes"]
            ):
                raise RuntimeError("H43 attestation pool byte accounting mismatch")
            if payload["frontend_path"] != "mla_absorb":
                raise RuntimeError("H43 frontend must use the absorbed MLA path")
            if payload["phase"] == "post_cuda_graph" and not (
                payload["frontend_dispatches"] > 0 and payload["reader_dispatches"] > 0
            ):
                raise RuntimeError(
                    "H43 graph setup did not traverse frontend and reader paths"
                )
            if payload["phase"] == "post_cuda_graph":
                if payload["frontend_fault_status"] != 0:
                    raise RuntimeError(
                        "H43 native frontend reported a sticky device fault"
                    )
                frontend_modes = payload["frontend_dispatches_by_mode"]
                reader_modes = payload["reader_dispatches_by_mode"]
                if (
                    not frontend_modes
                    or frontend_modes != reader_modes
                    or not set(frontend_modes).issubset({"decode", "target_verify"})
                ):
                    raise RuntimeError(
                        "H43 per-mode absorbed-MLA frontend and reader launches "
                        "do not match"
                    )
                expected_dtype_counts = {
                    "torch.float8_e4m3fn": payload["frontend_dispatches"]
                }
                if payload["frontend_query_dtypes"] != expected_dtype_counts or payload[
                    "reader_query_dtypes"
                ] != {"torch.float8_e4m3fn": payload["reader_dispatches"]}:
                    raise RuntimeError("H43 attestation requires FP8 query dispatch")
        else:
            if payload["native_loaded"]:
                raise RuntimeError("disabled H43 path loaded the native module")
            if payload["frontend_dispatches"] or payload["reader_dispatches"]:
                raise RuntimeError("disabled H43 path executed an H43 branch")
            if any(
                payload[key]
                for key in (
                    "frontend_dispatches_by_mode",
                    "reader_dispatches_by_mode",
                    "frontend_query_dtypes",
                    "reader_query_dtypes",
                )
            ):
                raise RuntimeError("disabled H43 path recorded H43 dispatch metadata")
            if not tq_enabled and payload["selected_row_bytes"] != 576:
                raise RuntimeError("FP8 MLA attestation requires a 576-byte row")
            if e2m1 and (
                payload["selected_row_bytes"] != 386
                or payload["codebook_slots"]
                or payload["fp8_rope_slots"]
                or payload["pool_all_layer_bytes"]
                != payload["expected_pool_all_layer_bytes"]
            ):
                raise RuntimeError(
                    "E2M1 attestation requires a 386-byte row without H43 buffers"
                )

    def emit_h43_runtime_attestation(self, phase: str) -> dict[str, object]:
        payload = self._h43_attestation_payload(phase)
        self._validate_h43_attestation(payload)
        if not payload["draft_worker"]:
            logger.info("H43_RUNTIME_ATTEST %s", json.dumps(payload, sort_keys=True))
        return payload

    def _record_h43_launch(
        self, event: str, forward_mode: object, query_dtype: torch.dtype
    ) -> None:
        if not getattr(self, "_h43_frontend", False):
            return
        mode = str(getattr(forward_mode, "name", forward_mode)).lower()
        dtype = str(query_dtype)
        if event == "frontend":
            self._h43_frontend_dispatch_count = (
                getattr(self, "_h43_frontend_dispatch_count", 0) + 1
            )
            mode_counts = getattr(self, "_h43_frontend_dispatches_by_mode", {})
            dtype_counts = getattr(self, "_h43_frontend_query_dtypes", {})
            self._h43_frontend_dispatches_by_mode = mode_counts
            self._h43_frontend_query_dtypes = dtype_counts
        elif event == "reader":
            self._h43_reader_dispatch_count = (
                getattr(self, "_h43_reader_dispatch_count", 0) + 1
            )
            mode_counts = getattr(self, "_h43_reader_dispatches_by_mode", {})
            dtype_counts = getattr(self, "_h43_reader_query_dtypes", {})
            self._h43_reader_dispatches_by_mode = mode_counts
            self._h43_reader_query_dtypes = dtype_counts
        else:
            raise ValueError(f"unknown H43 launch event: {event}")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        logged = getattr(self, "_h43_first_success_events", set())
        event_key = (event, mode)
        if event_key not in logged:
            logged.add(event_key)
            self._h43_first_success_events = logged
            runner = getattr(self, "_h43_model_runner", None)
            logger.info(
                "H43_RUNTIME_EVENT %s",
                json.dumps(
                    {
                        "schema": 1,
                        "event": event,
                        "forward_mode": mode,
                        "query_dtype": dtype,
                        "tp_rank": int(getattr(runner, "tp_rank", -1)),
                        "pp_rank": int(getattr(runner, "pp_rank", -1)),
                    },
                    sort_keys=True,
                ),
            )

    def _is_turboquant_layer(self, layer: RadixAttention) -> bool:
        tq4_cache = getattr(
            self, "_tq4_cache", getattr(self, "_tq_pool", None) is not None
        )
        if not tq4_cache:
            return False
        predicate = getattr(
            getattr(self, "_tq_pool", None), "is_turboquant_layer", None
        )
        return predicate(layer.layer_id) if callable(predicate) else True

    def uses_fp8_frontend(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        tq4_cache = getattr(
            self, "_tq4_cache", getattr(self, "_tq_pool", None) is not None
        )
        if not tq4_cache:
            return self.data_type == torch.float8_e4m3fn
        if getattr(self, "_tq4_hotcold_cache", False):
            return self.should_use_hot_fp8_frontend(forward_batch)
        return self.uses_mla_absorb_frontend(
            layer, forward_batch
        ) or not self._is_turboquant_layer(layer)

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

    def uses_mla_absorb_frontend(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        return bool(
            getattr(self, "_h43_frontend", False)
            and self._is_turboquant_layer(layer)
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
        )

    def prepare_mla_absorb_qkv(
        self,
        *,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Write an H43 row and return the combined FP8 absorbed-MLA query."""

        if not self.uses_mla_absorb_frontend(layer, forward_batch):
            return None
        if llama_4_scaling is not None:
            raise RuntimeError("H43 frontend does not support Llama-4 query scaling")
        if not getattr(self._tq_config, "mla_absorb_rotation_fused", False):
            raise RuntimeError("H43 frontend requires fused MLA absorb rotations")
        if q_nope.shape[-1] != self.kv_lora_rank:
            raise RuntimeError(
                "H43 frontend requires the absorbed MLA query; "
                f"got width {q_nope.shape[-1]}"
            )
        if q_rope.shape[-1] != self.qk_rope_head_dim:
            raise RuntimeError("H43 frontend received an invalid query RoPE width")
        if k_nope.shape[-1] != self.kv_lora_rank or k_rope.shape[-1] != (
            self.qk_rope_head_dim
        ):
            raise RuntimeError("H43 frontend received an invalid latent KV shape")

        query_out = torch.empty(
            (*q_nope.shape[:-1], self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_nope.device,
        )
        layer_id_rel = layer.layer_id - self._tq_pool.start_layer
        packed = self._tq_pool.kv_nope_packed_buffer[layer_id_rel]
        scale = self._tq_pool.kv_nope_scale_buffer[layer_id_rel]
        rope = self._tq_pool.kv_rope_buffer[layer_id_rel]
        codebook = self._tq_pool.kv_nope_codebook_buffer[layer_id_rel]
        status = self._tq_pool.tq_mla_frontend_fault_status
        if any(value is None for value in (packed, scale, rope, codebook, status)):
            raise RuntimeError("H43 selected-layer pool is incomplete")
        storage_code_lut = self._tq_config.k_storage_code_lut
        if storage_code_lut is None:
            raise RuntimeError("H43 E2M1 storage-code LUT is missing")

        tq_mla_frontend_out(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            forward_batch.out_cache_loc,
            self._tq_config.signs1,
            self._tq_config.signs2,
            self._tq_config.k_boundaries,
            self._tq_config.k_quant_centroids,
            storage_code_lut,
            query_out,
            packed,
            scale,
            rope,
            status,
            decode_centroids=self._tq_config.k_centroids,
            codebook_cache=codebook,
            scale_multiplier=self._tq_config.k_dequant_scale_multiplier,
            rotation_fused=True,
            num_warps=8,
            strict=False,
        )
        self._record_h43_launch("frontend", forward_batch.forward_mode, query_out.dtype)
        return query_out

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        logical_start = None
        if (
            self._tq4_hotcold_cache
            and forward_batch.batch_size == 1
            and forward_batch.extend_prefix_lens_cpu is not None
        ):
            logical_start = int(forward_batch.extend_prefix_lens_cpu[0])

        # Attention consumes FP8 Q/K/V, but persistent TQ rows must quantize
        # the original BF16 latent rather than an already FP8-rounded copy.
        # Preserve the established direct-FP8 writer for an entirely hot
        # prefill; when any row reaches the cold tier, compute BF16 RoPE and
        # let the disjoint pool quantize its own hot subset.
        writes_cold_tq = self._is_turboquant_layer(layer.attn_mha) and (
            not self._tq4_hotcold_cache
            or logical_start is None
            or logical_start + forward_batch.out_cache_loc.numel()
            > self._tq_pool.hot_capacity_tokens
        )
        if writes_cold_tq:
            _, cache_k_pe = layer.rotary_emb(positions, q_pe, k_pe)
            cache_kv_a = kv_a.unsqueeze(1)
        else:
            # k_pe is shared across heads (RoPE is position-only), so head 0
            # reproduces the original [tokens, 1, qk_rope] latent layout.
            cache_kv_a = fp8_quantize(kv_a, enable_pdl=is_arch_support_pdl()).unsqueeze(
                1
            )
            cache_k_pe = k_fp8[:, 0:1, layer.qk_nope_head_dim :]
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            cache_kv_a,
            cache_k_pe,
            **({"logical_start": logical_start} if self._tq4_hotcold_cache else {}),
        )
        return q_fp8, k_fp8, v_fp8

    def should_use_hot_fp8_frontend(self, forward_batch: ForwardBatch) -> bool:
        """Whether this forward is wholly served by the incumbent FP8 tier."""
        if not self._tq4_hotcold_cache:
            return False
        mode = forward_batch.forward_mode
        if not (mode.is_decode_or_idle() or mode.is_target_verify()):
            return False
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None or len(seq_lens_cpu) == 0:
            return False
        max_seq_len = int(seq_lens_cpu.max())
        if mode.is_target_verify():
            max_seq_len += self.num_draft_tokens
        return max_seq_len <= self._tq_pool.hot_capacity_tokens

    def hotcold_kv_write_logical_start(self, forward_batch: ForwardBatch) -> int:
        """Return the CPU-known logical position of the first KV write."""
        if not self._tq4_hotcold_cache or forward_batch.batch_size != 1:
            raise RuntimeError(
                "static hot/cold TurboQuant KV writes require batch_size=1"
            )
        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
            if seq_lens_cpu is None or len(seq_lens_cpu) != 1:
                raise RuntimeError(
                    "static hot/cold target verification requires one CPU "
                    "sequence length"
                )
            logical_start = int(seq_lens_cpu[0])
        elif mode.is_decode_or_idle():
            seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
            if seq_lens_cpu is None or len(seq_lens_cpu) != 1:
                raise RuntimeError(
                    "static hot/cold decode requires one CPU sequence length"
                )
            logical_start = int(seq_lens_cpu[0]) - int(
                forward_batch.out_cache_loc.numel()
            )
        else:
            prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            if prefix_lens_cpu is None or len(prefix_lens_cpu) != 1:
                raise RuntimeError(
                    "static hot/cold extend requires one CPU prefix length"
                )
            logical_start = int(prefix_lens_cpu[0])
        if logical_start < 0:
            raise RuntimeError(
                "static hot/cold TurboQuant computed a negative KV write "
                f"position: {logical_start}"
            )
        return logical_start

    def _validate_hotcold_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Reject unsupported speculative crossing before any layer writes KV."""
        if not self._tq4_hotcold_cache:
            return
        if forward_batch.batch_size != 1:
            raise RuntimeError("static hot/cold TurboQuant requires batch_size=1")
        mode = forward_batch.forward_mode
        if not (mode.is_target_verify() or mode.is_draft_extend_v2()):
            return
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None or len(seq_lens_cpu) == 0:
            raise RuntimeError(
                "static hot/cold TurboQuant speculative admission requires "
                "CPU sequence lengths"
            )
        max_seq_len = int(seq_lens_cpu.max())
        if mode.is_target_verify():
            max_seq_len += self.num_draft_tokens
        elif forward_batch.extend_seq_lens_cpu:
            max_seq_len = max(
                max_seq_len,
                int(seq_lens_cpu.max())
                - min(forward_batch.extend_seq_lens_cpu)
                + max(forward_batch.extend_seq_lens_cpu),
            )
        if max_seq_len > self._tq_pool.hot_capacity_tokens:
            raise RuntimeError(
                "segmented hot/cold TurboQuant decode is not quality-safe "
                "with speculative verification yet; increase "
                "SGLANG_TQ_MLA_HOT_TOKENS to cover the request or launch "
                "without speculative decoding"
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._validate_hotcold_forward_batch(forward_batch)
        return super().init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        self._validate_hotcold_forward_batch(forward_batch)
        return super().init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

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

    def _ensure_workspace(
        self,
        device: torch.device,
        q_len: int,
        batch_size: int = 1,
        split_kv: Optional[int] = None,
    ) -> torch.Tensor:
        tq4_minimum_bytes = (
            _tq4_workspace_bytes(
                batch_size,
                self.num_q_heads,
                self.kv_lora_rank,
                q_len,
                split_kv,
            )
            if self._tq4_cache and split_kv is not None
            else 0
        )
        needed = max(
            tq4_minimum_bytes,
            tokenspeed_workspace_bytes(
                tokenspeed_mla.get_num_sm(device),
                self.num_q_heads,
                self.kv_lora_rank,
                q_len,
            ),
        )
        if (
            self._tokenspeed_workspace is None
            or self._tokenspeed_workspace.device != device
            or self._tokenspeed_workspace.numel() < needed
        ):
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                device,
                self.num_q_heads,
                self.kv_lora_rank,
                q_len,
                minimum_bytes=tq4_minimum_bytes,
            )
        return self._tokenspeed_workspace

    def _page_table_padding_value(self) -> int:
        # Packed TMA stages load complete 128-token tiles before applying the
        # sequence mask. Page zero is an allocated, zero-initialized safe page.
        return (
            0
            if getattr(self, "_tq4_cache", False)
            else super()._page_table_padding_value()
        )

    def get_cuda_graph_max_seq_len(self) -> int:
        if not self._tq4_cache:
            return super().get_cuda_graph_max_seq_len()
        if self._tq4_hotcold_cache:
            # Captured requests read only the ordinary FP8 hot store. The
            # all-TQ codebook reader's 32K graph limit does not apply here.
            return min(
                self.max_context_len,
                self._tq_pool.hot_capacity_tokens,
            )
        return _tq4_codebook_cuda_graph_max_seq_len(
            self.max_context_len,
            self._tq_pool.kv_nope_codebook_buffer is not None,
        )

    def get_cuda_graph_max_prefix_len(self, forward_mode: ForwardMode) -> int:
        graph_max_seq_len = self.get_cuda_graph_max_seq_len()
        if self._tq4_cache and (
            forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
        ):
            return graph_max_seq_len - self.num_draft_tokens
        return graph_max_seq_len

    def can_run_cuda_graph(self, forward_batch: ForwardBatch) -> bool:
        graph_max_seq_len = self.get_cuda_graph_max_seq_len()
        if graph_max_seq_len >= self.max_context_len:
            return True

        extra_kv_tokens = (
            self.num_draft_tokens
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
            else 0
        )
        return _tq4_codebook_cuda_graph_eligible(
            getattr(forward_batch, "seq_lens_cpu", None),
            extra_kv_tokens,
            graph_max_seq_len,
        )

    def _get_decode_kv_cache(self, layer: RadixAttention) -> torch.Tensor:
        if not self._tq4_cache:
            return super()._get_decode_kv_cache(layer)
        layer_id_rel = layer.layer_id - self._tq_pool.start_layer
        if self._tq4_hotcold_cache:
            hot = self._tq_pool.kv_hot_buffer[layer_id_rel]
            if hot.shape[0] % self.page_size != 0:
                raise RuntimeError(
                    "TurboQuant hot pool token count must be page aligned; "
                    f"got {hot.shape[0]} tokens for page_size={self.page_size}"
                )
            return hot.view(-1, self.page_size, hot.shape[-1]).unsqueeze(1)
        if not self._is_turboquant_layer(layer):
            fp8 = self._tq_pool.kv_fp8_buffer[layer_id_rel]
            assert fp8 is not None
            if fp8.shape[0] % self.page_size != 0:
                raise RuntimeError(
                    "FP8 MLA pool token count must be page aligned; "
                    f"got {fp8.shape[0]} tokens for page_size={self.page_size}"
                )
            return fp8.view(-1, self.page_size, fp8.shape[-1]).unsqueeze(1)
        packed = self._tq_pool.kv_nope_packed_buffer[layer_id_rel]
        assert packed is not None
        if packed.shape[0] % self.page_size != 0:
            raise RuntimeError(
                "TurboQuant pool token count must be page aligned; "
                f"got {packed.shape[0]} tokens for page_size={self.page_size}"
            )
        return packed.view(-1, self.page_size, packed.shape[-1]).unsqueeze(1)

    def _validate_decode_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self._tq4_hotcold_cache:
            assert kv_cache.dtype == torch.float8_e4m3fn
            return
        if self._tq4_cache:
            expected_dtypes = (
                {torch.uint8, torch.float8_e4m3fn}
                if not getattr(self, "_tq4_all_layers", True)
                else {torch.uint8}
            )
            assert kv_cache.dtype in expected_dtypes
            return
        super()._validate_decode_kv_cache(kv_cache)

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
        forward_mode: Optional[ForwardMode] = None,
    ) -> torch.Tensor:
        if self._tq4_hotcold_cache:
            query_fp8 = (
                query
                if query.dtype == torch.float8_e4m3fn
                else _quantize_tq4_query(
                    query,
                    self.kv_lora_rank,
                    is_arch_support_pdl(),
                )
            )
            k_scale = getattr(layer, "k_scale_float", None)
            if k_scale is None:
                k_scale = 1.0
            seq_lens_i32 = (
                seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
            )
            decode_kwargs = dict(
                query=query_fp8,
                kv_cache=kv_cache,
                workspace_buffer=self._ensure_workspace(query.device, query.shape[1]),
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=seq_lens_i32,
                max_seq_len=int(max_seq_len),
                softmax_scale=float(layer.scaling) * float(k_scale),
                output_scale=float(k_scale),
                enable_pdl=is_arch_support_pdl(),
            )
            if self.supports_custom_decode_mask:
                decode_kwargs.update(
                    custom_mask=custom_mask,
                    cmask_off=custom_mask_offsets,
                )
            if int(max_seq_len) <= self._tq_pool.hot_capacity_tokens:
                return tokenspeed_mla.tokenspeed_mla_decode(**decode_kwargs)

            # The first implementation deliberately supports one fresh,
            # statically partitioned request. Exact-10K stays in the fast
            # branch above; this eager path exists to prove that the cold
            # capacity is usable without a full-width shadow copy.
            if query.shape[0] != 1 or seq_lens_i32.numel() != 1:
                raise RuntimeError(
                    "segmented hot/cold TurboQuant decode currently requires "
                    "batch_size=1"
                )
            # Speculative crossing is rejected in metadata initialization,
            # before the inherited forward path writes any layer's KV.  Keep a
            # defensive kernel-level assertion for direct callers.
            if getattr(self, "num_draft_tokens", 0):
                raise RuntimeError(
                    "speculative hot/cold boundary admission guard was bypassed"
                )
            k_scale = float(k_scale)
            if k_scale != 1.0:
                raise RuntimeError(
                    "segmented hot/cold TurboQuant decode currently requires "
                    f"k_scale=1.0; got {k_scale}"
                )

            # max_seq_len is computed from CPU sequence lengths while forward
            # metadata is built. Reuse it instead of synchronizing on the GPU
            # seq_lens tensor once per layer.
            full_seq_len = int(max_seq_len)
            hot_seq_len = self._tq_pool.hot_capacity_tokens
            cold_seq_len = full_seq_len - hot_seq_len
            if cold_seq_len <= 0:
                raise RuntimeError(
                    "segmented hot/cold decode was selected without cold tokens; "
                    f"seq_len={full_seq_len}, hot_capacity={hot_seq_len}"
                )

            hot_page_count = hot_seq_len // self.page_size
            cold_page_count = (cold_seq_len + self.page_size - 1) // self.page_size
            if block_tables.shape[1] < hot_page_count + cold_page_count:
                raise RuntimeError(
                    "hot/cold block table is too short for the segmented request"
                )
            hot_block_tables = block_tables[:, :hot_page_count]
            cold_actual_block_tables = (
                block_tables[:, hot_page_count : hot_page_count + cold_page_count]
                - hot_page_count
            )
            # Static single-request allocation plus disabled radix reuse makes
            # the tier/page mapping deterministic. Unit tests validate the
            # boundary; avoid min/max .item() checks here because each would
            # serialize every model layer during eager long-context decode.
            # The TQ4 reader issues full 128-token tile page-table loads before
            # applying the sequence mask. Pad unused columns with its allocated
            # zero page instead of exposing adjacent logical-page entries.
            cold_table_columns = ((cold_seq_len + 127) // 128) * 4
            cold_block_tables = torch.zeros(
                (1, cold_table_columns),
                dtype=block_tables.dtype,
                device=block_tables.device,
            )
            cold_block_tables[:, :cold_page_count] = cold_actual_block_tables

            q_len = query.shape[1]
            zero_mask_offset = torch.zeros(1, dtype=torch.int32, device=query.device)
            hot_custom_mask = None
            cold_custom_mask = None
            if custom_mask is not None:
                custom_mask_offset = (
                    int(custom_mask_offsets[0].item())
                    if custom_mask_offsets is not None
                    else 0
                )
                full_mask = (
                    custom_mask.view(torch.bool)
                    .view(-1)[
                        custom_mask_offset : custom_mask_offset + q_len * full_seq_len
                    ]
                    .view(q_len, full_seq_len)
                )
                cold_custom_mask = full_mask[:, hot_seq_len:].contiguous()
                if cold_seq_len < q_len:
                    hot_custom_mask = full_mask[:, :hot_seq_len].contiguous()
            elif cold_seq_len < q_len:
                query_positions = torch.arange(
                    full_seq_len - q_len,
                    full_seq_len,
                    dtype=torch.int64,
                    device=query.device,
                )
                key_positions = torch.arange(
                    full_seq_len, dtype=torch.int64, device=query.device
                )
                full_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                hot_custom_mask = full_mask[:, :hot_seq_len].contiguous()
                cold_custom_mask = full_mask[:, hot_seq_len:].contiguous()

            hot_seq_lens = torch.tensor(
                [hot_seq_len], dtype=torch.int32, device=query.device
            )
            hot_decode_kwargs = dict(decode_kwargs)
            hot_decode_kwargs.update(
                block_tables=hot_block_tables,
                seq_lens=hot_seq_lens,
                max_seq_len=hot_seq_len,
                causal_mask=False,
                return_lse=True,
            )
            if self.supports_custom_decode_mask:
                hot_decode_kwargs.update(
                    custom_mask=hot_custom_mask,
                    cmask_off=zero_mask_offset,
                )
            elif hot_custom_mask is not None:
                raise RuntimeError(
                    "segmented hot/cold decode requires TokenSpeed custom-mask "
                    "support for this query shape"
                )
            hot_output, hot_lse = tokenspeed_mla.tokenspeed_mla_decode(
                **hot_decode_kwargs
            )

            rotation_fused = getattr(
                self._tq_config, "mla_absorb_rotation_fused", False
            )
            if rotation_fused:
                cold_query = query
            else:
                cold_query_nope = self._tq_config.rotate_query(
                    query[..., : self.kv_lora_rank]
                )
                cold_query = torch.cat(
                    (cold_query_nope, query[..., self.kv_lora_rank :]), dim=-1
                )
            cold_query_fp8 = _quantize_tq4_query(
                cold_query,
                self.kv_lora_rank,
                is_arch_support_pdl(),
            )
            layer_id_rel = layer.layer_id - self._tq_pool.start_layer
            packed = self._tq_pool.kv_nope_packed_buffer[layer_id_rel].view(
                -1, self.page_size, self.kv_lora_rank // 2
            )
            scale = self._tq_pool.kv_nope_scale_buffer[layer_id_rel].view(
                -1, self.page_size
            )
            rope = self._tq_pool.kv_rope_buffer[layer_id_rel].view(
                -1, self.page_size, self.qk_rope_head_dim
            )
            codebook_buffer = self._tq_pool.kv_nope_codebook_buffer
            e2m1_cache = bool(getattr(self._tq_config, "e2m1", False))
            if not e2m1_cache and codebook_buffer is None:
                raise RuntimeError(
                    "TokenSpeed Lloyd TQ4 decode requires its FP8 codebook buffer"
                )
            codebook = (
                None
                if e2m1_cache
                else codebook_buffer[layer_id_rel].view(-1, self.page_size, 16)
            )
            split_kv = _tq4_split_override(
                1,
                cold_seq_len,
                tokenspeed_mla.get_num_sm(query.device),
            )
            cold_seq_lens = torch.tensor(
                [cold_seq_len], dtype=torch.int32, device=query.device
            )
            cold_output, cold_lse = tokenspeed_mla.tokenspeed_mla_decode_tq4(
                query=cold_query_fp8,
                kv_nope_packed=packed,
                kv_nope_scale=scale,
                kv_rope=rope,
                centroids=self._tq_config.k_centroids,
                kv_nope_codebook=codebook,
                workspace_buffer=self._ensure_workspace(
                    query.device,
                    q_len,
                    batch_size=1,
                    split_kv=split_kv,
                ),
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=cold_block_tables,
                seq_lens=cold_seq_lens,
                max_seq_len=cold_seq_len,
                softmax_scale=float(layer.scaling),
                enable_pdl=is_arch_support_pdl(),
                causal_mask=cold_custom_mask is None,
                custom_mask=cold_custom_mask,
                cmask_off=zero_mask_offset,
                split_kv_override=split_kv,
                return_lse=True,
            )
            if not rotation_fused:
                cold_output = self._tq_config.inverse_rotate_output(cold_output).to(
                    hot_output.dtype
                )
            merged_output, _ = tokenspeed_mla.merge_attention_outputs_base2(
                hot_output,
                hot_lse,
                cold_output,
                cold_lse,
            )
            return merged_output

        if self._is_turboquant_layer(layer):
            kernel_max_seq_len = _tq4_kernel_max_seq_len(
                int(max_seq_len), self.max_context_len
            )
            rotation_fused = getattr(
                self._tq_config, "mla_absorb_rotation_fused", False
            )
            h43_frontend = getattr(self, "_h43_frontend", False)
            if h43_frontend:
                if not rotation_fused:
                    raise RuntimeError(
                        "H43 frontend requires fused MLA absorb rotations"
                    )
                if query.dtype != torch.float8_e4m3fn:
                    raise RuntimeError(
                        "H43 reader requires the native FP8 query output"
                    )
                query_fp8 = query
            elif rotation_fused:
                query_rotated = query
            else:
                query_nope_rotated = self._tq_config.rotate_query(
                    query[..., : self.kv_lora_rank]
                )
                query_rotated = torch.cat(
                    (query_nope_rotated, query[..., self.kv_lora_rank :]), dim=-1
                )
            if not h43_frontend:
                query_fp8 = _quantize_tq4_query(
                    query_rotated,
                    self.kv_lora_rank,
                    is_arch_support_pdl(),
                )
            layer_id_rel = layer.layer_id - self._tq_pool.start_layer
            scale_buffer = self._tq_pool.kv_nope_scale_buffer[layer_id_rel]
            rope_buffer = self._tq_pool.kv_rope_buffer[layer_id_rel]
            assert scale_buffer is not None and rope_buffer is not None
            scale = scale_buffer.view(-1, self.page_size)
            rope = rope_buffer.view(-1, self.page_size, self.qk_rope_head_dim)
            e2m1_cache = bool(getattr(self._tq_config, "e2m1", False))
            codebook_buffer = self._tq_pool.kv_nope_codebook_buffer
            if e2m1_cache and not h43_frontend:
                # Canonical E2M1 codes can be expanded from the packed cache
                # and per-token BF16 scale. Keeping a 16-byte FP8 lookup row
                # would add a redundant shadow representation.
                codebook = None
            else:
                if codebook_buffer is None:
                    raise RuntimeError(
                        "TokenSpeed Lloyd TQ4 decode requires its FP8 codebook buffer"
                    )
                codebook_layer = codebook_buffer[layer_id_rel]
                assert codebook_layer is not None
                codebook = codebook_layer.view(-1, self.page_size, 16)
            seq_lens_i32 = (
                seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
            )
            split_kv = _tq4_split_override(
                query.shape[0],
                kernel_max_seq_len,
                tokenspeed_mla.get_num_sm(query.device),
            )
            decode_kwargs = dict(
                query=query_fp8,
                kv_nope_packed=kv_cache,
                kv_nope_scale=scale,
                kv_rope=rope,
                centroids=self._tq_config.k_centroids,
                kv_nope_codebook=codebook,
                workspace_buffer=self._ensure_workspace(
                    query.device,
                    query.shape[1],
                    batch_size=query.shape[0],
                    split_kv=split_kv,
                ),
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=seq_lens_i32,
                max_seq_len=kernel_max_seq_len,
                softmax_scale=float(layer.scaling),
                enable_pdl=is_arch_support_pdl(),
                custom_mask=custom_mask,
                cmask_off=custom_mask_offsets,
                split_kv_override=split_kv,
            )
            if h43_frontend:
                decode_kwargs["fp8_rope"] = True
            output = tokenspeed_mla.tokenspeed_mla_decode_tq4(**decode_kwargs)
            if h43_frontend:
                if forward_mode is None:
                    raise RuntimeError("H43 reader requires the actual forward mode")
                self._record_h43_launch(
                    "reader",
                    forward_mode,
                    query.dtype,
                )
            if not rotation_fused:
                output = self._tq_config.inverse_rotate_output(output)
            return output

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
        )
        if self.supports_custom_decode_mask:
            decode_kwargs.update(
                custom_mask=custom_mask,
                cmask_off=custom_mask_offsets,
            )
        return tokenspeed_mla.tokenspeed_mla_decode(**decode_kwargs)

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
        # (prepare_prefill_qkv / prepare_mla_absorb_qkv /
        # pack_prefix_chunk_kv); no quantize here.
        if self._is_turboquant_layer(layer) and not self._tq4_hotcold_cache:
            return TRTLLMMLABackend._run_prefill_kernel(
                self,
                q,
                k,
                v,
                layer,
                batch_size,
                cum_seq_lens_q,
                max_q_len,
                seq_lens_kv,
                cum_seq_lens_kv,
                max_kv_len,
                is_causal,
                return_lse,
                out_buffer,
                o_sf_scale,
            )
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
