"""Omniva MXFP4 MoE runner.

Bespoke MoE runner for MXFP4-quantized experts. Built for gpt-oss-120b at
TP=1 on a single H100 where (a) the OpenAI ``triton_kernels.matmul_ogs``
path crashes with a state-dependent CUDA illegal memory access and
(b) dequantizing weights to bf16 at load time OOMs (~125 GiB > 80 GiB).

This runner uses ``fused_moe_kernel_mxfp4`` which consumes packed MXFP4
weights directly: no swizzle, no upcast. See:

    python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
        fused_moe_kernel_mxfp4 / invoke_fused_moe_kernel_mxfp4

and the design document at
``OmniSec/Inference/GPT-OSS/Design - MXFP4 MoE Runner Kernel.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


# Tile configs come from the autotune lookup in
# ``try_get_optimal_moe_config`` (dtype="mxfp4_w4a16"), which reads tuned
# JSONs out of ``triton_utils/configs/triton_<ver>/E=...,N=...,dtype=
# mxfp4_w4a16[,down=true].json`` and falls back to the MXFP4 branch of
# ``get_default_config`` when no JSON is present.
#
# To retune for a new shape, run ``benchmark/kernels/fused_moe_triton/
# tuning_fused_moe_triton.py --dtype mxfp4_w4a16`` and commit the
# generated JSONs. The default-config fallback is deliberately
# conservative so untuned shapes still work correctly, just not at peak.

# BLOCK_SIZE_K must be a multiple of 32 (MXFP_BLOCK_SIZE). The autotune
# grid enforces this at lookup time; this constant keeps the sanity-check
# readable.
_MXFP_BLOCK = 32


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OmnivaMxfp4RunnerInput(RunnerInput):
    """Input bundle for the Omniva MXFP4 runner core.

    ``topk_ids`` / ``topk_weights`` are used both to build the aligned
    ``sorted_token_ids`` / ``expert_ids`` arrays and to multiply routing
    weights into the second matmul's accumulator.
    """

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.OMNIVA_MXFP4


@dataclass
class OmnivaMxfp4RunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.OMNIVA_MXFP4


@dataclass
class OmnivaMxfp4QuantInfo(MoeQuantInfo):
    """Packed MXFP4 weights as registered by ``Mxfp4MoEMethod.create_weights``.

    w13_weight:       [E, 2*N, K/2] uint8  (K packed, contiguous inner)
    w13_weight_scale: [E, 2*N, K/32] uint8  (E8M0)
    w13_weight_bias:  [E, 2*N] bf16
    w2_weight:        [E, K_hidden, N/2] uint8
    w2_weight_scale:  [E, K_hidden, N/32] uint8
    w2_weight_bias:   [E, K_hidden] bf16
    """

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w13_weight_bias: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    w2_weight_bias: torch.Tensor


# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------


class OmnivaMxfp4RunnerCore(MoeRunnerCore):
    """Execute MoE experts via ``fused_moe_kernel_mxfp4``.

    Pipeline (per batch):
        1. Align topk_ids into ``sorted_token_ids`` / ``expert_ids``.
        2. Gate-up matmul: A @ W13^T -> intermediate_cache1 [M*topk, 2*N].
        3. Swiglu activation (with gpt-oss alpha/clamp if configured).
        4. Down matmul: act @ W2^T -> out [M*topk, K_hidden], accumulating
           routed_weights.
        5. Reduce over topk -> out_hidden_states [M, K_hidden].
    """

    def run(
        self,
        runner_input: OmnivaMxfp4RunnerInput,
        quant_info: OmnivaMxfp4QuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> OmnivaMxfp4RunnerOutput:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            try_get_optimal_moe_config,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            invoke_fused_moe_kernel_mxfp4,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        assert (
            self.config.is_gated
        ), "OmnivaMxfp4RunnerCore only supports gated MoEs (gate+up fused)"
        assert (
            self.config.activation == "silu"
        ), "OmnivaMxfp4RunnerCore only supports silu (swiglu) activation"
        assert hooks is None, "LoRA hooks not supported by OmnivaMxfp4RunnerCore"

        hidden_states = runner_input.hidden_states
        topk_ids = runner_input.topk_ids
        topk_weights = runner_input.topk_weights
        assert hidden_states.dtype == torch.bfloat16

        M, K_hidden = hidden_states.shape
        E, two_N, _ = quant_info.w13_weight.shape
        N = two_N // 2  # intermediate_size
        topk = topk_ids.shape[1]

        # Autotune lookup. ``try_get_optimal_moe_config`` reads
        # ``configs/triton_<ver>/E=E,N=N,device=...,dtype=mxfp4_w4a16[.down].json``
        # if it exists, else falls back to ``get_default_config`` — which
        # has an explicit MXFP4 branch (see fused_moe_triton_config.py).
        # The helper returns (up_config, (down_config, max_block_m)) and
        # internally asserts up["BLOCK_SIZE_M"] == down["BLOCK_SIZE_M"]
        # because moe_align_block_size only takes one BLOCK_M. When
        # shipping new tuned JSONs, harmonize the two BLOCK_M values per
        # M (see the tuning script's post-processing step) so this
        # assertion never trips.
        gate_up_config, (down_config, _max_block_m) = try_get_optimal_moe_config(
            w1_shape=quant_info.w13_weight.shape,
            w2_shape=quant_info.w2_weight.shape,
            top_k=topk,
            dtype="mxfp4_w4a16",
            M=M,
            return_down_config=True,
        )
        if down_config is None:
            # No tuned down-config JSON; reuse the gate-up config.
            # Safe because the two kernels share signature/constraints.
            down_config = dict(gate_up_config)

        assert gate_up_config["BLOCK_SIZE_K"] % _MXFP_BLOCK == 0, (
            f"BLOCK_SIZE_K={gate_up_config['BLOCK_SIZE_K']} must be multiple of "
            f"{_MXFP_BLOCK} (MXFP block size); check tuned config for up matmul"
        )
        assert down_config["BLOCK_SIZE_K"] % _MXFP_BLOCK == 0, (
            f"BLOCK_SIZE_K={down_config['BLOCK_SIZE_K']} must be multiple of "
            f"{_MXFP_BLOCK} (MXFP block size); check tuned config for down matmul"
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, gate_up_config["BLOCK_SIZE_M"], E
        )

        # --- Gate-up matmul: A [M, K_hidden] @ W13 [E, 2N, K_hidden/2] ---
        ic1 = torch.empty(
            (M * topk, two_N), device=hidden_states.device, dtype=torch.bfloat16
        )

        invoke_fused_moe_kernel_mxfp4(
            hidden_states,
            quant_info.w13_weight,
            quant_info.w13_weight_scale,
            ic1,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=self.config.apply_router_weight_on_input,
            top_k=topk,
            config=gate_up_config,
            compute_type=tl.bfloat16,
            filter_expert=False,
        )

        # Add gate-up bias (per-expert, per-output-row). The kernel does
        # not fuse bias because MXFP4 output bias lives at bf16 precision
        # and cleanly composes with the following swiglu.
        if quant_info.w13_weight_bias is not None:
            # Gather bias per-token via expert assignment. Shape:
            # ic1[m*topk + t] += w13_weight_bias[expert_ids_flat[m*topk + t]]
            # However we don't have per-token expert ids directly; topk_ids
            # gives them. Flatten topk_ids into [M*topk] and gather.
            flat_expert = topk_ids.reshape(-1).to(torch.int64)
            # Bias: [E, 2N] -> select by flat_expert -> [M*topk, 2N]
            ic1 = ic1 + quant_info.w13_weight_bias[flat_expert].to(ic1.dtype)

        # --- Swiglu activation ---
        # Reuse the same activation helpers used by TritonRunnerCore so
        # numerics stay bit-identical with the bf16 reference path.
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _swiglu_gpt_oss_sigmoid_alpha,
            _swiglu_silu_clamp_mul,
        )

        if (
            self.config.gemm1_alpha is not None
            and self.config.gemm1_clamp_limit is not None
        ):
            ic2 = _swiglu_gpt_oss_sigmoid_alpha(
                ic1, self.config.gemm1_alpha, self.config.gemm1_clamp_limit
            )
        elif self.config.gemm1_clamp_limit is not None:
            ic2 = _swiglu_silu_clamp_mul(ic1, self.config.gemm1_clamp_limit)
        else:
            ic2 = _swiglu_plain(ic1)

        del ic1

        # --- Down matmul: ic2 [M*topk, N] @ W2 [E, K_hidden, N/2] ---
        # For the down matmul, A = ic2 has per-row tokens (M*topk rows).
        # To match the kernel's addressing (which computes A row index
        # from offs_token // top_k), we need sorted_token_ids to index
        # into M*topk rows. Re-align using top_k=1 semantics: each
        # intermediate-cache row is its own "token", and the kernel
        # divides by top_k=1 to get the A row -> so we use the same
        # sorted_token_ids we already have (expert mapping is unchanged).
        # But the kernel uses `offs_token // top_k` to index A — for the
        # down matmul A is already M*topk rows, so top_k=1 is correct.

        ic3 = torch.empty(
            (M * topk, K_hidden), device=hidden_states.device, dtype=torch.bfloat16
        )

        invoke_fused_moe_kernel_mxfp4(
            ic2,
            quant_info.w2_weight,
            quant_info.w2_weight_scale,
            ic3,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=not self.config.apply_router_weight_on_input,
            top_k=1,
            config=down_config,
            compute_type=tl.bfloat16,
            filter_expert=False,
        )

        if quant_info.w2_weight_bias is not None:
            # Bias addition for down matmul: per-expert [K_hidden]. Apply
            # after routing-weight mul (the kernel already multiplied
            # routed weight into the accumulator); we scale bias by the
            # same routed weight so the final sum reduction balances.
            flat_expert = topk_ids.reshape(-1).to(torch.int64)
            bias = quant_info.w2_weight_bias[flat_expert].to(ic3.dtype)
            if not self.config.apply_router_weight_on_input:
                # ic3 already has routed weight applied by the kernel;
                # scale bias consistently so reduce-sum yields
                # sum_t w_t * (A @ W2 + b) rather than sum_t w_t*A@W2 + b.
                bias = bias * topk_weights.reshape(-1, 1).to(ic3.dtype)
            ic3 = ic3 + bias

        del ic2

        # --- Combine: sum over topk dimension -> [M, K_hidden] ---
        ic3 = ic3.view(M, topk, K_hidden)
        out = ic3.sum(dim=1).to(torch.bfloat16)

        if (
            self.config.routed_scaling_factor is not None
            and self.config.routed_scaling_factor != 1.0
        ):
            out = out * self.config.routed_scaling_factor

        return OmnivaMxfp4RunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.OMNIVA_MXFP4


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


def _swiglu_plain(x: torch.Tensor) -> torch.Tensor:
    """Plain swiglu: silu(gate) * up, with interleaved [gate, up] layout.

    Only used as fallback when neither gemm1_alpha nor gemm1_clamp_limit
    are configured. gpt-oss always sets gemm1_alpha so this branch is
    dormant for our target model.
    """
    gate = x[..., 0::2]
    up = x[..., 1::2]
    return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------
# Permute hooks: standard dispatch -> OmnivaMxfp4 runner input
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "omniva_mxfp4")
def pre_permute_standard_to_omniva_mxfp4(
    dispatch_output: "StandardDispatchOutput",
    quant_info: OmnivaMxfp4QuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> OmnivaMxfp4RunnerInput:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    assert TopKOutputChecker.format_is_standard(
        topk_output
    ), "OmnivaMxfp4 runner expects StandardTopKOutput"

    return OmnivaMxfp4RunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
    )


@register_post_permute("omniva_mxfp4", "standard")
def post_permute_omniva_mxfp4_to_standard(
    runner_output: OmnivaMxfp4RunnerOutput,
    quant_info: OmnivaMxfp4QuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> "StandardCombineInput":
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
