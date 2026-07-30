#!/usr/bin/env python3
"""Focused H43 I3 production-call-chain integration gate on SM100."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.jit_kernel.tq_mla_frontend import (
    preload_tq_mla_frontend_prebuilt,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolTurboQuant
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from test_h41_w2_frontend import query_reference, raw_fp8, writer_reference


def assert_selected_rows_equal(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    locations: torch.Tensor,
) -> None:
    for actual_buffer, expected_buffer in zip(actual, expected):
        actual_rows = actual_buffer[locations]
        # The independent writer oracle returns packed/scale/codebook as full
        # pool images, but its RoPE oracle is deliberately only the input rows.
        expected_rows = (
            expected_buffer
            if expected_buffer.shape[0] == locations.numel()
            else expected_buffer[locations]
        )
        if actual_rows.dtype == torch.float8_e4m3fn:
            actual_rows = raw_fp8(actual_rows)
        if expected_rows.dtype == torch.float8_e4m3fn:
            expected_rows = raw_fp8(expected_rows)
        torch.testing.assert_close(actual_rows, expected_rows, rtol=0, atol=0)


def main() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H43 I3 integration requires SM100")
    preload_tq_mla_frontend_prebuilt()

    with (
        envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(True),
        envs.SGLANG_TQ_MLA_FUSED_ROPE_WRITE.override(True),
        envs.SGLANG_TQ_MLA_KV_WRITE_WORKSPACE_TOKENS.override(32),
    ):
        pool = MLATokenToKVPoolTurboQuant(
            size=64,
            page_size=32,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            turboquant_e2m1=True,
            enable_fp8_codebook=True,
            enable_fp8_rope=True,
            enable_h43_frontend=True,
        )
        pool.tq_config.mla_absorb_rotation_fused = True

        backend = object.__new__(TokenspeedMLABackend)
        backend._h43_frontend = True
        backend._tq4_cache = True
        backend._tq4_hotcold_cache = False
        backend._tq_pool = pool
        backend._tq_config = pool.tq_config

        kv_b_proj = Mock(side_effect=AssertionError("kv_b_proj must be skipped"))
        layer = SimpleNamespace(
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
            kv_b_proj=kv_b_proj,
            rotary_emb=lambda positions, q_pe, k_pe: (q_pe, k_pe),
            attn_mha=SimpleNamespace(layer_id=0),
        )
        generator = torch.Generator(device=device).manual_seed(20260730)

        for forward_mode, tokens, locations_list in (
            (ForwardMode.DECODE, 1, [7]),
            (ForwardMode.TARGET_VERIFY, 5, [11, 3, 19, 5, 23]),
        ):
            full_query = torch.empty(
                tokens, 8, 576, dtype=torch.bfloat16, device=device
            ).normal_(0.0, 0.125, generator=generator)
            query_latent = full_query[..., :512]
            assert not query_latent.is_contiguous()
            query_rope = torch.empty(
                tokens, 8, 64, dtype=torch.bfloat16, device=device
            ).normal_(0.0, 0.125, generator=generator)
            cache_latent = torch.empty(
                tokens, 512, dtype=torch.bfloat16, device=device
            ).normal_(0.0, 0.125, generator=generator)
            cache_rope = torch.empty(
                tokens, 1, 64, dtype=torch.bfloat16, device=device
            ).normal_(0.0, 0.125, generator=generator)
            locations = torch.tensor(locations_list, dtype=torch.int64, device=device)
            query, key, value = backend.prepare_prefill_qkv(
                q=full_query,
                q_pe=query_rope,
                kv_a=cache_latent,
                k_pe=cache_rope,
                positions=torch.arange(tokens, dtype=torch.int64, device=device),
                layer=layer,
                forward_batch=SimpleNamespace(
                    forward_mode=forward_mode,
                    out_cache_loc=locations,
                ),
            )
            expected_query = query_reference(
                query_latent, query_rope, pool.tq_config, rotation_fused=True
            )
            torch.testing.assert_close(
                raw_fp8(query), raw_fp8(expected_query), rtol=0, atol=0
            )
            expected_cache = writer_reference(
                cache_latent.unsqueeze(1),
                cache_rope,
                locations,
                pool.tq_config,
                pool.size + pool.page_size,
            )
            assert_selected_rows_equal(
                (
                    pool.kv_nope_packed_buffer[0],
                    pool.kv_nope_scale_buffer[0],
                    pool.kv_rope_buffer[0],
                    pool.kv_nope_codebook_buffer[0],
                ),
                expected_cache,
                locations,
            )
            assert key is None and value is None

        prefill_locations = torch.tensor(
            [29, 31, 37, 41, 43, 47, 53], dtype=torch.int64, device=device
        )
        prefill_latent = torch.empty(
            7, 1, 512, dtype=torch.bfloat16, device=device
        ).normal_(0.0, 0.125, generator=generator)
        prefill_rope = torch.empty(
            7, 1, 64, dtype=torch.bfloat16, device=device
        ).normal_(0.0, 0.125, generator=generator)
        pool.set_mla_kv_buffer(
            SimpleNamespace(layer_id=0),
            prefill_locations,
            prefill_latent,
            prefill_rope,
        )
        expected_prefill = writer_reference(
            prefill_latent,
            prefill_rope,
            prefill_locations,
            pool.tq_config,
            pool.size + pool.page_size,
        )
        assert_selected_rows_equal(
            (
                pool.kv_nope_packed_buffer[0],
                pool.kv_nope_scale_buffer[0],
                pool.kv_rope_buffer[0],
                pool.kv_nope_codebook_buffer[0],
            ),
            expected_prefill,
            prefill_locations,
        )

        invalid_location = torch.tensor(
            [pool.size + pool.page_size], dtype=torch.int64, device=device
        )
        invalid_query = torch.zeros(1, 8, 576, dtype=torch.bfloat16, device=device)
        backend.prepare_prefill_qkv(
            q=invalid_query,
            q_pe=torch.zeros(1, 8, 64, dtype=torch.bfloat16, device=device),
            kv_a=torch.zeros(1, 512, dtype=torch.bfloat16, device=device),
            k_pe=torch.zeros(1, 1, 64, dtype=torch.bfloat16, device=device),
            positions=torch.zeros(1, dtype=torch.int64, device=device),
            layer=layer,
            forward_batch=SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                out_cache_loc=invalid_location,
            ),
        )
        torch.cuda.synchronize()
        assert int(pool.tq_mla_frontend_fault_status.item()) == 1
        kv_b_proj.assert_not_called()

    print(
        json.dumps(
            {
                "experiment": "H43_I3_SERVING_INTEGRATION",
                "status": "PASS",
                "sm": "sm100",
                "page_size": 32,
                "selected_row_bytes": 338,
                "decode_q1": "PASS",
                "target_verify_q5": "PASS",
                "noncontiguous_projection_view": "PASS",
                "prefill_writer": "PASS",
                "kv_b_proj_skipped": "PASS",
                "sticky_fault_status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
