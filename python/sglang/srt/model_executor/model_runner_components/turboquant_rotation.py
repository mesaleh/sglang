from __future__ import annotations

from typing import Any, Sequence

import torch


def fuse_turboquant_output_rotation_weights(
    tq_config: Any,
    candidates: Sequence[tuple[torch.Tensor, int]],
    *,
    skipped_layers: int,
) -> bool:
    """Fuse every o_proj weight or leave every weight in its original domain."""
    if skipped_layers or not candidates:
        return False

    with torch.no_grad():
        for weight, num_heads in candidates:
            transposed = weight.data.t().contiguous()
            fused = tq_config.fuse_inverse_rotation_into_o_proj(transposed, num_heads)
            weight.data.copy_(fused.t().contiguous())
    tq_config.output_rotation_fused = True
    return True


def fuse_turboquant_mla_output_rotation_weights(
    tq_config: Any,
    candidates: Sequence[torch.Tensor],
    *,
    skipped_layers: int,
) -> bool:
    """Atomically fold inverse WHT into every absorbed MLA value weight."""

    if getattr(tq_config, "output_rotation_fused", False):
        return True
    if skipped_layers or not candidates:
        return False

    # Compute every replacement before mutating model state.  A shape, dtype,
    # or kernel failure therefore leaves all layers in the original domain.
    replacements = [
        tq_config.fuse_inverse_rotation_into_mla_v_weight(weight)
        for weight in candidates
    ]
    with torch.no_grad():
        for weight, replacement in zip(candidates, replacements):
            weight.copy_(replacement)
    tq_config.output_rotation_fused = True
    return True
