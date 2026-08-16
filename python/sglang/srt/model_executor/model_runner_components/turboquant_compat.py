from __future__ import annotations

from sglang.srt.layers.quantization.kv_turboquant import (
    is_native_e2m1_mla_kv_cache_dtype,
    is_native_e2m1_recip_bf16_mla_kv_cache_dtype,
)


def validate_turboquant_transfer_compatibility(
    *,
    kv_cache_dtype,
    disaggregation_mode: str,
    enable_hierarchical_cache: bool,
    use_mla_backend: bool,
    enable_deterministic_inference: bool = False,
    prefill_attention_backend: str | None = None,
    decode_attention_backend: str | None = None,
    mla_fused_decode_enabled: bool = False,
) -> None:
    if not (
        isinstance(kv_cache_dtype, str)
        and kv_cache_dtype.startswith("turboquant_")
    ):
        return

    if is_native_e2m1_mla_kv_cache_dtype(kv_cache_dtype):
        raise ValueError(
            "--kv-cache-dtype=turboquant_4bit_e2m1 is reserved for the matched "
            "native-E2M1 MLA writer, pool, and reader, which are not enabled "
            "in this source revision. Use a supported --kv-cache-dtype."
        )

    if is_native_e2m1_recip_bf16_mla_kv_cache_dtype(kv_cache_dtype):
        if disaggregation_mode != "null":
            raise ValueError(
                "N10 native E2M1 MLA does not support PD disaggregation."
            )
        if enable_hierarchical_cache:
            raise ValueError(
                "N10 native E2M1 MLA does not support hierarchical/CPU KV "
                "offload."
            )
        if enable_deterministic_inference:
            raise ValueError(
                "N10 native E2M1 MLA does not yet support deterministic "
                "inference paths."
            )
        if not use_mla_backend:
            raise ValueError("N10 native E2M1 KV cache requires an MLA model.")
        if (
            prefill_attention_backend != "tokenspeed_mla"
            or decode_attention_backend != "tokenspeed_mla"
        ):
            raise ValueError(
                "N10 native E2M1 MLA requires tokenspeed_mla for both "
                "prefill and decode."
            )
        raise ValueError(
            "N10 native E2M1 MLA pool/writer support is present, but its "
            "matched TokenSpeed reader backend is not wired in this source "
            "revision. Use a supported --kv-cache-dtype."
        )

    if disaggregation_mode != "null":
        raise ValueError(
            "TurboQuant KV cache is not supported with PD disaggregation: "
            "the transfer descriptors do not include every packed value, scale, "
            "and RoPE buffer. Use --disaggregation-mode null or a supported "
            "--kv-cache-dtype."
        )

    if enable_hierarchical_cache and not use_mla_backend:
        raise ValueError(
            "MHA TurboQuant KV cache is not supported with "
            "--enable-hierarchical-cache: the generic host pool cannot represent "
            "its packed K/V and scale buffers. Disable hierarchical cache or use "
            "a supported --kv-cache-dtype."
        )

    if enable_deterministic_inference and not use_mla_backend:
        raise ValueError(
            "MHA TurboQuant KV cache is not supported with "
            "--enable-deterministic-inference: the deterministic Triton extend "
            "path does not consume its packed K/V and scale buffers. Disable "
            "deterministic inference or use a supported --kv-cache-dtype."
        )

    if use_mla_backend and not mla_fused_decode_enabled:
        raise ValueError(
            "MLA TurboQuant KV cache requires SGLANG_TQ_MLA_FUSED_DECODE=1: "
            "the generic MLA attention backends reconstruct the entire packed "
            "KV pool on every layer and forward. Enable the fused research path "
            "or use a supported --kv-cache-dtype."
        )

    if use_mla_backend and (
        prefill_attention_backend != "flashmla"
        or decode_attention_backend != "flashmla"
    ):
        raise ValueError(
            "MLA TurboQuant KV cache requires flashmla for both prefill and "
            "decode so SGLANG_TQ_MLA_FUSED_DECODE selects "
            "TurboQuantMLABackend. Set --attention-backend flashmla or use a "
            "supported --kv-cache-dtype."
        )
