from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.layers.quantization.kv_turboquant import (
    parse_mla_turboquant_layer_ids,
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
