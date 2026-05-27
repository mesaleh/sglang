from typing import Optional, Tuple

import torch

_FusedLMHeadGemm: Optional[Tuple[object, object]] = None
_FusedLMHeadDecisionCache: dict[tuple, bool] = {}


def get_fused_lm_head_gemm():
    """Return TokenSpeed's tuned small-M lm_head GEMM if it is installed."""
    global _FusedLMHeadGemm
    if _FusedLMHeadGemm is not None:
        return _FusedLMHeadGemm
    try:
        from tokenspeed_kernel.thirdparty.cuda.lm_head_gemm import (
            lm_head_gemm,
            should_use_fused,
        )

        _FusedLMHeadGemm = (should_use_fused, lm_head_gemm)
    except Exception:
        _FusedLMHeadGemm = (None, None)
    return _FusedLMHeadGemm


def maybe_fused_lm_head_matmul(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    should_use_fused, lm_head_gemm = get_fused_lm_head_gemm()
    if should_use_fused is not None:
        hs = (
            hidden_states
            if hidden_states.is_contiguous()
            else hidden_states.contiguous()
        )
        cache_key = (
            tuple(hs.shape),
            tuple(hs.stride()),
            hs.dtype,
            hs.device.type,
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
            weight.device.type,
        )
        use_fused = _FusedLMHeadDecisionCache.get(cache_key)
        if use_fused is None:
            use_fused = bool(should_use_fused(hs, weight))
            _FusedLMHeadDecisionCache[cache_key] = use_fused
        if use_fused:
            return lm_head_gemm(hs, weight, enable_pdl=True)
    return torch.matmul(hidden_states, weight.T)
