from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    _supports_custom_decode_mask,
    _tq4_kernel_max_seq_len,
    _tq4_split_override,
    _tq4_workspace_bytes,
)
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.model_executor.model_runner import ModelRunner


def test_custom_decode_mask_contract_requires_both_parameters():
    def supported(custom_mask=None, cmask_off=None):
        pass

    def missing_offset(custom_mask=None):
        pass

    def stock_decode(query=None):
        pass

    assert _supports_custom_decode_mask(supported)
    assert not _supports_custom_decode_mask(missing_offset)
    assert not _supports_custom_decode_mask(stock_decode)


def test_tq4_split_override_keeps_one_compact_graph_specialization():
    assert _tq4_split_override(batch_size=1, max_seq_len=128, num_sms=148) == 1
    assert _tq4_split_override(batch_size=1, max_seq_len=129, num_sms=148) == 2
    assert _tq4_split_override(batch_size=1, max_seq_len=10_000, num_sms=148) == 64
    assert _tq4_split_override(batch_size=8, max_seq_len=10_000, num_sms=148) == 64


def test_tq4_kernel_max_seq_len_clamps_capture_padding_only():
    assert _tq4_kernel_max_seq_len(32_768, 32_768) == 32_768
    assert _tq4_kernel_max_seq_len(32_773, 32_768) == 32_768

    with pytest.raises(ValueError, match="context_length <= 32768"):
        _tq4_kernel_max_seq_len(32_769, 32_769)


def test_tq4_workspace_tracks_fixed_split_graph_shape():
    assert _tq4_workspace_bytes(8, 8, 512, 5, 64) == 42_024_960
    assert _tq4_workspace_bytes(1, 8, 512, 5, 1) == 0


def test_tq4_absorb_rotation_does_not_override_mha_prefill():
    # The fused rotations live only in the 512-wide MLA absorb weights. MHA
    # prefill uses independent 192-wide Q/K and 128-wide V projections, so the
    # TokenSpeed backend must preserve the parent prefill implementation.
    assert TokenspeedMLABackend.forward_extend is TRTLLMMLABackend.forward_extend


def test_tq4_rotation_fusion_uses_resolved_decode_backend():
    tq_config = SimpleNamespace()

    def make_runner(default_backend, decode_backend):
        runner = object.__new__(ModelRunner)
        runner.turboquant_bits = 4
        runner.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(tq_config=tq_config)
        )
        runner.use_mla_backend = True
        runner.server_args = SimpleNamespace(
            enable_lora=False,
            get_attention_backends=lambda: (default_backend, decode_backend),
        )
        runner._maybe_fuse_tq_mla_absorb_rotations = Mock()
        return runner

    explicit_other_decode = make_runner("tokenspeed_mla", "flashinfer")
    explicit_native_decode = make_runner("fa3", "tokenspeed_mla")

    ModelRunner._maybe_fuse_tq_output_rotation(explicit_other_decode)
    ModelRunner._maybe_fuse_tq_output_rotation(explicit_native_decode)

    explicit_other_decode._maybe_fuse_tq_mla_absorb_rotations.assert_not_called()
    explicit_native_decode._maybe_fuse_tq_mla_absorb_rotations.assert_called_once()


def test_tq4_rotation_fusion_is_idempotent():
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        named_modules=Mock(side_effect=AssertionError("weights revisited"))
    )
    tq_config = SimpleNamespace(mla_absorb_rotation_fused=True)
    logger = Mock()

    ModelRunner._maybe_fuse_tq_mla_absorb_rotations(runner, tq_config, logger)

    runner.model.named_modules.assert_not_called()
