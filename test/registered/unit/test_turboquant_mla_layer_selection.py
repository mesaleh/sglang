from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.layers.quantization.kv_turboquant import (
    parse_mla_turboquant_layer_ids,
    should_allocate_mla_tq_fp8_codebook,
    should_use_mla_tq_h43_frontend,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator


def test_parse_mla_turboquant_layer_ids_canonicalizes_ranges():
    assert parse_mla_turboquant_layer_ids(None, 61) is None
    assert parse_mla_turboquant_layer_ids("20, 12-15,13,8", 61) == (
        8,
        12,
        13,
        14,
        15,
        20,
    )


def test_h43_frontend_gate_and_codebook_truth_table():
    assert not should_use_mla_tq_h43_frontend(
        "tokenspeed_mla", "tokenspeed_mla", True, False
    )
    with pytest.raises(ValueError, match="both prefill and decode"):
        should_use_mla_tq_h43_frontend("flashinfer", "tokenspeed_mla", True, True)
    with pytest.raises(ValueError, match="requires E2M1"):
        should_use_mla_tq_h43_frontend("tokenspeed_mla", "tokenspeed_mla", False, True)
    with pytest.raises(ValueError, match="requires CUDA graph capture"):
        should_use_mla_tq_h43_frontend(
            "tokenspeed_mla",
            "tokenspeed_mla",
            True,
            True,
            disable_cuda_graph=True,
        )
    with pytest.raises(ValueError, match="does not support two-batch overlap"):
        should_use_mla_tq_h43_frontend(
            "tokenspeed_mla",
            "tokenspeed_mla",
            True,
            True,
            enable_two_batch_overlap=True,
        )
    with pytest.raises(ValueError, match="does not support PDMux"):
        should_use_mla_tq_h43_frontend(
            "tokenspeed_mla",
            "tokenspeed_mla",
            True,
            True,
            enable_pdmux=True,
        )
    assert should_use_mla_tq_h43_frontend(
        "tokenspeed_mla", "tokenspeed_mla", True, True
    )
    assert not should_use_mla_tq_h43_frontend(
        "fa4",
        "fa4",
        False,
        True,
        is_draft_worker=True,
    )

    assert not should_allocate_mla_tq_fp8_codebook("tokenspeed_mla", True)
    assert should_allocate_mla_tq_fp8_codebook(
        "tokenspeed_mla", True, h43_frontend=True
    )
    assert should_allocate_mla_tq_fp8_codebook("tokenspeed_mla", False)


@pytest.mark.parametrize(
    "spec",
    ("", "1,", ",1", "1,,2", "-1", "a", "2-1", "1-2-3", "61"),
)
def test_parse_mla_turboquant_layer_ids_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        parse_mla_turboquant_layer_ids(spec, 61)


def _fake_model_runner(selected_layer_ids):
    server_args = SimpleNamespace(
        get_attention_backends=lambda: ("tokenspeed_mla", "tokenspeed_mla")
    )
    return SimpleNamespace(
        use_mla_backend=True,
        kv_cache_dtype=torch.bfloat16,
        turboquant_bits=4,
        turboquant_k_bits=4,
        turboquant_e2m1=True,
        turboquant_mla_layer_ids=selected_layer_ids,
        start_layer=10,
        end_layer=20,
        model_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            hf_config=SimpleNamespace(architectures=[]),
        ),
        server_args=server_args,
    )


def test_mla_turboquant_pool_sizing_weights_local_selected_layers():
    configurator = object.__new__(DefaultPoolConfigurator)
    configurator._fixed_size = 0
    runner = _fake_model_runner((8, 12, 15, 25))

    with (
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=8,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.envs.SGLANG_TQ_MLA_HOT_TOKENS.get",
            return_value=0,
        ),
    ):
        cell_size = configurator._compute_cell_size(runner, num_layers=10)

    assert cell_size == 2 * 386 + 8 * 576


def test_mla_turboquant_pool_sizing_preserves_all_layer_default():
    configurator = object.__new__(DefaultPoolConfigurator)
    configurator._fixed_size = 0
    runner = _fake_model_runner(None)

    with (
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=8,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.envs.SGLANG_TQ_MLA_HOT_TOKENS.get",
            return_value=0,
        ),
    ):
        cell_size = configurator._compute_cell_size(runner, num_layers=10)

    assert cell_size == 10 * 386


def test_h43_pool_sizing_uses_338_byte_selected_row():
    configurator = object.__new__(DefaultPoolConfigurator)
    configurator._fixed_size = 0
    runner = _fake_model_runner((8, 12, 15, 25))

    with (
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=8,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.envs.SGLANG_TQ_MLA_HOT_TOKENS.get",
            return_value=0,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.envs.SGLANG_TQ_MLA_H43_FRONTEND.get",
            return_value=True,
        ),
    ):
        cell_size = configurator._compute_cell_size(runner, num_layers=10)

    assert cell_size == 2 * 338 + 8 * 576


def test_tokenspeed_layer_static_frontend_dispatch():
    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        TokenspeedMLABackend,
    )

    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend._tq4_hotcold_cache = False
    backend._tq_pool = SimpleNamespace(
        is_turboquant_layer=lambda layer_id: layer_id == 11
    )
    forward_batch = SimpleNamespace()

    assert backend.uses_fp8_frontend(SimpleNamespace(layer_id=10), forward_batch)
    assert not backend.uses_fp8_frontend(SimpleNamespace(layer_id=11), forward_batch)


@pytest.mark.parametrize(
    ("tokens", "expected_chunk_sizes"),
    (
        (0, ()),
        (2, (2,)),
        (3, (3,)),
        (4, (3, 1)),
        (6, (3, 3)),
        (8, (3, 3, 2)),
    ),
)
def test_mla_turboquant_fused_writer_chunks_aligned_inputs(
    tokens, expected_chunk_sizes
):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolTurboQuant

    pool = object.__new__(MLATokenToKVPoolTurboQuant)
    pool._tq_mla_kv_write_unit = torch.empty(4, 1, 8, dtype=torch.float32)
    pool._tq_mla_kv_write_norms = torch.empty(3, 1, dtype=torch.float32)
    pool._tq_mla_kv_write_y = torch.empty(5, 1, 8, dtype=torch.float32)
    pool._set_mla_kv_buffer_fused = Mock()

    loc = torch.tensor([13, 2, 21, 8, 5, 34, 1, 3])[:tokens]
    nope = (
        torch.arange(8 * 8, dtype=torch.float32)
        .to(torch.bfloat16)
        .view(8, 1, 8)[:tokens]
    )
    rope = (
        torch.arange(8 * 2, dtype=torch.float32)
        .to(torch.bfloat16)
        .view(8, 1, 2)[:tokens]
    )

    pool._set_mla_kv_buffer_fused_chunked(7, loc, nope, rope)

    assert pool._fused_kv_write_chunk_capacity() == 3
    assert pool._set_mla_kv_buffer_fused.call_count == len(expected_chunk_sizes)
    expected_ranges = []
    start = 0
    for chunk_size in expected_chunk_sizes:
        expected_ranges.append((start, start + chunk_size))
        start += chunk_size
    for call, (start, end) in zip(
        pool._set_mla_kv_buffer_fused.call_args_list, expected_ranges, strict=True
    ):
        assert call.args[0] == 7
        torch.testing.assert_close(call.args[1], loc[start:end])
        torch.testing.assert_close(call.args[2], nope[start:end])
        torch.testing.assert_close(call.args[3], rope[start:end])
