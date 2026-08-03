from __future__ import annotations


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

    native_tokenspeed_mla = (
        prefill_attention_backend == "tokenspeed_mla"
        and decode_attention_backend == "tokenspeed_mla"
    )

    if use_mla_backend and native_tokenspeed_mla:
        # The native TokenSpeed backend consumes the packed MLA pool directly
        # and validates its TQ4 side-library ABI during backend construction.
        # It does not use the older env-gated FlashMLA research dispatcher.
        return

    if use_mla_backend and not mla_fused_decode_enabled:
        raise ValueError(
            "MLA TurboQuant KV cache requires SGLANG_TQ_MLA_FUSED_DECODE=1: "
            "use the env-gated flashmla fused research path, select native "
            "tokenspeed_mla for both prefill and decode, or use a supported "
            "--kv-cache-dtype."
        )

    if use_mla_backend and (
        prefill_attention_backend != "flashmla"
        or decode_attention_backend != "flashmla"
    ):
        raise ValueError(
            "MLA TurboQuant KV cache requires either native tokenspeed_mla for "
            "both prefill and decode, or flashmla for both so "
            "SGLANG_TQ_MLA_FUSED_DECODE selects TurboQuantMLABackend. Set one "
            "complete backend pair or use a supported --kv-cache-dtype."
        )
