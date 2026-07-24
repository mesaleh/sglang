from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    _supports_custom_decode_mask,
    _tq4_codebook_cuda_graph_eligible,
    _tq4_codebook_cuda_graph_max_seq_len,
    _tq4_kernel_max_seq_len,
    _tq4_split_override,
    _tq4_workspace_bytes,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    _capture_block_table,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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
    assert _tq4_split_override(batch_size=1, max_seq_len=256_000, num_sms=148) == 64
    assert _tq4_split_override(batch_size=8, max_seq_len=262_144, num_sms=148) == 64


def test_tq4_kernel_max_seq_len_clamps_capture_padding_only():
    assert _tq4_kernel_max_seq_len(32_768, 32_768) == 32_768
    assert _tq4_kernel_max_seq_len(32_773, 32_768) == 32_768
    assert _tq4_kernel_max_seq_len(256_005, 256_000) == 256_000
    assert _tq4_kernel_max_seq_len(262_144, 262_144) == 262_144

    with pytest.raises(ValueError, match="context_length <= 262144"):
        _tq4_kernel_max_seq_len(262_145, 262_145)


def test_tq4_workspace_tracks_fixed_split_graph_shape():
    assert _tq4_workspace_bytes(8, 8, 512, 5, 64) == 42_024_960
    assert _tq4_workspace_bytes(1, 8, 512, 5, 1) == 0


def test_tq4_codebook_graph_bound_and_verifier_accounting():
    assert _tq4_codebook_cuda_graph_max_seq_len(256_000, True) == 32_768
    assert _tq4_codebook_cuda_graph_max_seq_len(256_000, False) == 256_000
    assert _tq4_codebook_cuda_graph_max_seq_len(16_384, True) == 16_384

    assert _tq4_codebook_cuda_graph_eligible(
        torch.tensor([32_763], dtype=torch.int32), 5, 32_768
    )
    assert not _tq4_codebook_cuda_graph_eligible(
        torch.tensor([32_764], dtype=torch.int32), 5, 32_768
    )
    assert not _tq4_codebook_cuda_graph_eligible(None, 0, 32_768)
    assert not _tq4_codebook_cuda_graph_eligible(
        torch.empty(0, dtype=torch.int32), 0, 32_768
    )


def test_bounded_graph_block_table_has_contiguous_stable_storage():
    backing = torch.arange(8 * 8000, dtype=torch.int32).reshape(8, 8000)
    captured = _capture_block_table(backing, batch_size=5, width=1024)

    assert captured.shape == (5, 1024)
    assert captured.is_contiguous()
    assert captured.data_ptr() != backing.data_ptr()
    assert torch.equal(captured, backing[:5, :1024])


def test_tq4_codebook_backend_fails_closed_above_graph_bound():
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend._tq_pool = SimpleNamespace(kv_nope_codebook_buffer=object())
    backend.max_context_len = 256_000
    backend.num_draft_tokens = 5

    exact_boundary = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens_cpu=torch.tensor([32_763], dtype=torch.int32),
    )
    one_token_over = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens_cpu=torch.tensor([32_764], dtype=torch.int32),
    )
    missing_host_lengths = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens_cpu=None,
    )

    assert backend.get_cuda_graph_max_seq_len() == 32_768
    assert (
        backend.get_cuda_graph_max_prefix_len(ForwardMode.TARGET_VERIFY) == 32_763
    )
    assert backend.get_cuda_graph_max_prefix_len(ForwardMode.DECODE) == 32_768
    assert backend.can_run_cuda_graph(exact_boundary)
    assert not backend.can_run_cuda_graph(one_token_over)
    assert not backend.can_run_cuda_graph(missing_host_lengths)

    backend._tq_pool.kv_nope_codebook_buffer = None
    assert backend.get_cuda_graph_max_seq_len() == 256_000
    assert backend.can_run_cuda_graph(missing_host_lengths)


def test_attention_graph_policy_defaults_true_and_hybrid_delegates():
    assert AttentionBackend().can_run_cuda_graph(SimpleNamespace())

    hybrid = object.__new__(HybridAttnBackend)
    hybrid.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_attention_mode="decode")
    )
    hybrid.decode_backend = SimpleNamespace(can_run_cuda_graph=lambda batch: False)
    hybrid.prefill_backend = SimpleNamespace(can_run_cuda_graph=lambda batch: True)
    decode_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    assert not hybrid.can_run_cuda_graph(decode_batch)


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
