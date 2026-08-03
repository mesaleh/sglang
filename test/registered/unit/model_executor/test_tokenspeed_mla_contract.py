from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
)
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    TokenspeedMLAMultiStepDraftBackend,
    _custom_decode_mask_kwargs,
    _supports_custom_decode_mask,
    _tq4_cuda_graph_eligible,
    _tq4_kernel_max_seq_len,
    _tq4_split_override,
    _tq4_workspace_bytes,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
    TRTLLMMLAMultiStepDraftBackend,
    _target_verify_max_seq_len,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def test_custom_decode_mask_contract_fails_closed_when_required():
    mask = object()

    with pytest.raises(RuntimeError, match="custom_mask and cmask_off"):
        _custom_decode_mask_kwargs(
            supports_custom_decode_mask=False,
            custom_mask=mask,
            custom_mask_offsets=None,
        )

    assert (
        _custom_decode_mask_kwargs(
            supports_custom_decode_mask=False,
            custom_mask=None,
            custom_mask_offsets=None,
        )
        == {}
    )
    assert _custom_decode_mask_kwargs(
        supports_custom_decode_mask=True,
        custom_mask=mask,
        custom_mask_offsets="offsets",
    ) == {"custom_mask": mask, "cmask_off": "offsets"}


def test_target_verify_max_seq_len_does_not_add_draft_width_twice():
    metadata = TRTLLMMLADecodeMetadata(max_seq_len_k=108)

    assert _target_verify_max_seq_len(metadata) == 108

    with pytest.raises(RuntimeError, match="missing max_seq_len_k"):
        _target_verify_max_seq_len(TRTLLMMLADecodeMetadata())


def test_tq4_split_override_keeps_one_compact_graph_specialization():
    assert _tq4_split_override(128) == 1
    assert _tq4_split_override(129) == 2
    assert _tq4_split_override(10_000) == 64
    assert _tq4_split_override(262_144) == 64


def test_tq4_kernel_context_guard_and_graph_boundary():
    assert _tq4_kernel_max_seq_len(32_768, 256_000) == 32_768
    assert _tq4_kernel_max_seq_len(256_005, 256_000) == 256_000
    assert _tq4_kernel_max_seq_len(262_144, 262_144) == 262_144
    with pytest.raises(ValueError, match="context_length <= 262144"):
        _tq4_kernel_max_seq_len(262_145, 262_145)

    assert _tq4_cuda_graph_eligible(
        torch.tensor([32_763], dtype=torch.int32), 5, 32_768
    )
    assert not _tq4_cuda_graph_eligible(
        torch.tensor([32_764], dtype=torch.int32), 5, 32_768
    )
    assert not _tq4_cuda_graph_eligible(None, 0, 32_768)
    assert not _tq4_cuda_graph_eligible(
        torch.empty(0, dtype=torch.int32), 0, 32_768
    )


def test_tq4_workspace_covers_fixed_split_dflash_batch():
    assert _tq4_workspace_bytes(8, 8, 512, 5, 64) == 42_024_960
    assert _tq4_workspace_bytes(1, 8, 512, 5, 1) == 0


def test_tq4_graph_workspace_is_reserved_before_parent_capture(monkeypatch):
    parent_init = Mock()
    monkeypatch.setattr(TRTLLMMLABackend, "init_cuda_graph_state", parent_init)
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend.max_context_len = 256_000
    backend.device = torch.device("cpu")
    backend._ensure_workspace = Mock(return_value=torch.empty(0))

    backend.init_cuda_graph_state(max_bs=8, max_num_tokens=40)

    backend._ensure_workspace.assert_called_once_with(
        backend.device,
        5,
        batch_size=8,
        split_kv=64,
    )
    parent_init.assert_called_once_with(8, 40, None)


def test_bounded_graph_page_tables_are_contiguous_per_request(monkeypatch):
    parent_init = Mock()
    monkeypatch.setattr(
        FlashInferMLAAttnBackend, "init_cuda_graph_state", parent_init
    )
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend.max_context_len = 256_000
    backend.page_size = 32
    backend.device = torch.device("cpu")
    backend.supports_custom_decode_mask = False
    backend.data_type = torch.bfloat16
    backend.num_q_heads = 8
    backend.kv_cache_dim = 576
    backend.num_draft_tokens = 0

    TRTLLMMLABackend.init_cuda_graph_state(
        backend, max_bs=2, max_num_tokens=10
    )

    bounded_width = backend._calc_padded_blocks(32_768)
    frontier_width = backend._calc_padded_blocks(256_000)
    assert backend.decode_cuda_graph_kv_indices.shape == (10, bounded_width)
    assert backend.decode_cuda_graph_kv_indices.is_contiguous()
    assert backend.decode_cuda_graph_kv_indices[:2].is_contiguous()
    assert backend.decode_cuda_graph_frontier_kv_indices.shape == (
        10,
        frontier_width,
    )
    assert frontier_width > bounded_width
    parent_init.assert_called_once_with(2, 10, None)


def test_bounded_tq_backend_keeps_spec_v2_cpu_lengths(monkeypatch):
    from sglang.srt.layers.attention import tokenspeed_mla_backend as module

    def fake_tq4_decode(custom_mask=None, cmask_off=None):
        pass

    config = SimpleNamespace(k_centroids=torch.empty(16))
    pool = SimpleNamespace(is_mla_turboquant_pool=True, tq_config=config)

    def fake_parent_init(backend, *args, **kwargs):
        backend.token_to_kv_pool = pool
        backend.data_type = torch.bfloat16
        backend.page_size = 32
        backend.max_context_len = 256_000

    monkeypatch.setattr(TRTLLMMLABackend, "__init__", fake_parent_init)
    monkeypatch.setattr(module, "is_tokenspeed_mla_available", lambda: False)
    monkeypatch.delattr(module, "tokenspeed_mla", raising=False)
    with pytest.raises(RuntimeError, match="does not provide native TQ4 decode"):
        TokenspeedMLABackend(SimpleNamespace())

    monkeypatch.setattr(
        module,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode_tq4=fake_tq4_decode),
        raising=False,
    )

    backend = TokenspeedMLABackend(SimpleNamespace())

    assert backend.needs_cpu_seq_lens is True
    assert backend.supports_custom_decode_mask is True


def test_tq_multi_step_wrapper_propagates_cpu_lengths_and_graph_policy(monkeypatch):
    from sglang.srt.layers.attention import tokenspeed_mla_backend as module

    children = [
        SimpleNamespace(
            needs_cpu_seq_lens=True,
            can_run_cuda_graph=lambda forward_batch: forward_batch.safe,
        ),
        SimpleNamespace(
            needs_cpu_seq_lens=False,
            can_run_cuda_graph=lambda forward_batch: True,
        ),
    ]

    def fake_parent_init(backend, model_runner, topk, speculative_num_steps):
        backend.speculative_num_steps = speculative_num_steps
        backend.attn_backends = [None] * (speculative_num_steps - 1)
        backend.kv_indptr = [object()] * (speculative_num_steps - 1)
        backend.q_indptr_decode = object()

    monkeypatch.setattr(TRTLLMMLAMultiStepDraftBackend, "__init__", fake_parent_init)
    monkeypatch.setattr(module, "TokenspeedMLABackend", lambda *args, **kwargs: children.pop(0))

    backend = TokenspeedMLAMultiStepDraftBackend(
        SimpleNamespace(), topk=1, speculative_num_steps=3
    )

    assert backend.needs_cpu_seq_lens is True
    assert backend.can_run_cuda_graph(SimpleNamespace(safe=True))
    assert not backend.can_run_cuda_graph(SimpleNamespace(safe=False))


def test_tq4_backend_fails_closed_above_bounded_graph_metadata():
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
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
    assert backend.can_run_cuda_graph(exact_boundary)
    assert not backend.can_run_cuda_graph(one_token_over)
    assert not backend.can_run_cuda_graph(missing_host_lengths)


def test_attention_graph_policy_defaults_true_and_hybrid_delegates():
    assert AttentionBackend().can_run_cuda_graph(SimpleNamespace())

    hybrid = object.__new__(HybridAttnBackend)
    hybrid._select_backend = lambda forward_mode: SimpleNamespace(
        can_run_cuda_graph=lambda batch: False
    )
    assert not hybrid.can_run_cuda_graph(
        SimpleNamespace(forward_mode=ForwardMode.DECODE)
    )


def test_tq4_page_padding_and_direct_cache_view_never_dequantize():
    transfer_counter = Mock()
    packed = torch.zeros((64, 1, 256), dtype=torch.uint8)
    pool = SimpleNamespace(
        start_layer=3,
        layer_transfer_counter=transfer_counter,
        kv_nope_packed_buffer=[packed],
    )
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend._tq_pool = pool
    backend.page_size = 32
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=Mock(side_effect=AssertionError("dense shadow requested"))
    )

    cache = backend._get_decode_kv_cache(SimpleNamespace(layer_id=3))

    assert backend._page_table_padding_value() == 0
    assert cache.shape == (2, 1, 32, 256)
    assert cache.data_ptr() == packed.data_ptr()
    transfer_counter.wait_until.assert_called_once_with(0)
    backend.token_to_kv_pool.get_key_buffer.assert_not_called()

    dense_backend = object.__new__(TRTLLMMLABackend)
    assert dense_backend._page_table_padding_value() == -1
    constructing_backend = object.__new__(TokenspeedMLABackend)
    assert constructing_backend._page_table_padding_value() == -1


def test_tq4_decode_dispatches_packed_buffers_and_preserves_tree_mask(monkeypatch):
    from sglang.srt.layers.attention import tokenspeed_mla_backend as module

    calls = {}
    query = torch.zeros((1, 5, 8, 576), dtype=torch.bfloat16)
    packed_pages = torch.zeros((4, 1, 32, 256), dtype=torch.uint8)
    scale = torch.ones((128, 1), dtype=torch.bfloat16)
    rope = torch.zeros((128, 1, 64), dtype=torch.bfloat16)
    output_rotated = torch.ones((1, 5, 8, 512), dtype=torch.bfloat16)

    def fake_quantize(value, kv_lora_rank, enable_pdl):
        calls["quantize"] = (value, kv_lora_rank, enable_pdl)
        return value

    def fake_decode(**kwargs):
        calls["decode"] = kwargs
        return output_rotated

    config = SimpleNamespace(
        k_centroids=torch.arange(16, dtype=torch.float32),
        rotate_query=lambda value: value + 1,
        inverse_rotate_output=lambda value: value + 2,
    )
    pool = SimpleNamespace(
        start_layer=7,
        kv_nope_scale_buffer=[scale],
        kv_rope_buffer=[rope],
    )
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend._tq_pool = pool
    backend._tq_config = config
    backend.page_size = 32
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.max_context_len = 256_000
    backend._ensure_workspace = (
        lambda device, q_len, batch_size, split_kv: "workspace"
    )

    monkeypatch.setattr(module, "_quantize_tq4_query", fake_quantize)
    monkeypatch.setattr(module, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(
        module.tokenspeed_mla, "tokenspeed_mla_decode_tq4", fake_decode
    )
    custom_mask = torch.tensor([True, False])
    custom_mask_offsets = torch.tensor([0], dtype=torch.int32)

    actual = backend._run_decode_kernel(
        query=query,
        kv_cache=packed_pages,
        block_tables=torch.zeros((1, 4), dtype=torch.int32),
        seq_lens=torch.tensor([10_000], dtype=torch.int64),
        max_seq_len=10_000,
        layer=SimpleNamespace(layer_id=7, scaling=0.125),
        custom_mask=custom_mask,
        custom_mask_offsets=custom_mask_offsets,
    )

    kwargs = calls["decode"]
    assert kwargs["kv_nope_packed"] is packed_pages
    assert kwargs["kv_nope_scale"].data_ptr() == scale.data_ptr()
    assert kwargs["kv_rope"].data_ptr() == rope.data_ptr()
    assert kwargs["custom_mask"] is custom_mask
    assert kwargs["cmask_off"] is custom_mask_offsets
    assert kwargs["seq_lens"].dtype == torch.int32
    assert kwargs["max_seq_len"] == 10_000
    assert kwargs["split_kv_override"] == 64
    assert calls["quantize"][0][..., :512].eq(1).all()
    torch.testing.assert_close(actual, output_rotated + 2)


def test_tq4_prefill_writer_quantizes_original_bf16_latent(monkeypatch):
    from sglang.srt.layers.attention import tokenspeed_mla_backend as module

    tokens = 2
    heads = 8
    q = torch.randn(tokens, heads, 576, dtype=torch.bfloat16)
    q_pe = q[..., 512:]
    kv_a = torch.randn(tokens, 512, dtype=torch.bfloat16)
    k_pe = torch.randn(tokens, 1, 64, dtype=torch.bfloat16)
    projected = torch.randn(tokens, heads, 640, dtype=torch.bfloat16)
    rotated_k_pe = k_pe + 3
    rotary = Mock(return_value=(q_pe, rotated_k_pe))
    rotary.cos_sin_cache = torch.empty(0)
    rotary.is_neox_style = True
    layer = SimpleNamespace(
        kv_b_proj=Mock(return_value=(projected,)),
        num_local_heads=heads,
        qk_nope_head_dim=512,
        v_head_dim=128,
        qk_rope_head_dim=64,
        rotary_emb=rotary,
        attn_mha="attention-layer",
    )
    pool = SimpleNamespace(set_mla_kv_buffer=Mock())
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend.token_to_kv_pool = pool
    backend._fused_rope_fp8_quantize = Mock(
        return_value=(
            torch.empty_like(q),
            torch.empty(tokens, heads, 576, dtype=torch.bfloat16),
        )
    )
    monkeypatch.setattr(module, "fp8_quantize", lambda value, enable_pdl: value)
    forward_batch = SimpleNamespace(out_cache_loc=torch.tensor([1, 2]))

    backend.prepare_prefill_qkv(
        q=q,
        q_pe=q_pe,
        kv_a=kv_a,
        k_pe=k_pe,
        positions=torch.tensor([5, 6]),
        layer=layer,
        forward_batch=forward_batch,
    )

    write = pool.set_mla_kv_buffer.call_args
    assert write.args[0] == "attention-layer"
    assert write.args[1] is forward_batch.out_cache_loc
    assert write.args[2].dtype == torch.bfloat16
    assert write.args[2].shape == (tokens, 1, 512)
    assert write.args[2].data_ptr() == kv_a.data_ptr()
    assert write.args[3] is rotated_k_pe
