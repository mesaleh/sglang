from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    _has_populated_tq_codebook,
    _supports_custom_decode_mask,
    _tq4_codebook_cuda_graph_eligible,
    _tq4_codebook_cuda_graph_max_seq_len,
    _tq4_kernel_max_seq_len,
    _tq4_split_override,
    _tq4_workspace_bytes,
    _validate_h43_frontend_activation,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    _capture_block_table,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    handle_attention_tokenspeed_mla,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)


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


def test_h43_activation_is_target_only_and_requires_eight_local_heads():
    _validate_h43_frontend_activation(False, True, 1)
    _validate_h43_frontend_activation(True, False, 8)

    with pytest.raises(RuntimeError, match="only on the target"):
        _validate_h43_frontend_activation(True, True, 8)
    with pytest.raises(RuntimeError, match="exactly 8 local query heads"):
        _validate_h43_frontend_activation(True, False, 4)


def test_h43_pp_empty_shard_has_no_populated_codebook_rows():
    assert not _has_populated_tq_codebook(None)
    assert not _has_populated_tq_codebook([None, None, None])
    assert _has_populated_tq_codebook([None, torch.empty(1, dtype=torch.uint8), None])


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
    backend._tq4_hotcold_cache = False
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
    mixed_batch_crossing = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens_cpu=torch.tensor([32_763, 32_764], dtype=torch.int32),
    )
    missing_host_lengths = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens_cpu=None,
    )

    assert backend.get_cuda_graph_max_seq_len() == 32_768
    assert backend.get_cuda_graph_max_prefix_len(ForwardMode.TARGET_VERIFY) == 32_763
    assert backend.get_cuda_graph_max_prefix_len(ForwardMode.DECODE) == 32_768
    assert backend.can_run_cuda_graph(exact_boundary)
    assert not backend.can_run_cuda_graph(one_token_over)
    assert not backend.can_run_cuda_graph(mixed_batch_crossing)
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


def test_hybrid_backend_emits_attestation_for_each_unique_backend():
    hybrid = object.__new__(HybridAttnBackend)
    prefill_emit = Mock(return_value={"role": "prefill"})
    decode_emit = Mock(return_value={"role": "decode"})
    hybrid.prefill_backend = SimpleNamespace(emit_h43_runtime_attestation=prefill_emit)
    hybrid.decode_backend = SimpleNamespace(emit_h43_runtime_attestation=decode_emit)

    payloads = hybrid.emit_h43_runtime_attestation("post_cuda_graph")

    assert payloads == [{"role": "prefill"}, {"role": "decode"}]
    prefill_emit.assert_called_once_with("post_cuda_graph")
    decode_emit.assert_called_once_with("post_cuda_graph")

    hybrid.decode_backend = hybrid.prefill_backend
    payloads = hybrid.emit_h43_runtime_attestation("init")
    assert payloads == [{"role": "prefill"}]


def test_tokenspeed_decode_and_verify_dispatch_to_absorbed_mla():
    attn = SimpleNamespace(disable_chunked_prefix_cache=False)
    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        batch = SimpleNamespace(forward_mode=mode, extend_prefix_lens_cpu=None)
        assert handle_attention_tokenspeed_mla(attn, batch) == AttnForwardMethod.MLA


def test_hybrid_delegates_absorbed_mla_frontend_to_selected_backend():
    hybrid = object.__new__(HybridAttnBackend)
    hybrid.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_attention_mode="decode")
    )
    hybrid.prefill_backend = SimpleNamespace()
    predicate = Mock(return_value=True)
    prepare = Mock(return_value=torch.empty(1, dtype=torch.float8_e4m3fn))
    hybrid.decode_backend = SimpleNamespace(
        uses_mla_absorb_frontend=predicate,
        prepare_mla_absorb_qkv=prepare,
    )
    layer = SimpleNamespace(layer_id=3)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    assert hybrid.uses_mla_absorb_frontend(layer, batch)
    result = hybrid.prepare_mla_absorb_qkv(
        forward_batch=batch,
        layer=layer,
        q_nope=object(),
        q_rope=object(),
        k_nope=object(),
        k_rope=object(),
    )

    assert result.dtype == torch.float8_e4m3fn
    predicate.assert_called_once_with(layer, batch)
    prepare.assert_called_once()


def test_absorbed_mla_core_uses_backend_query_and_disables_duplicate_kv_write():
    attn = object.__new__(DeepseekMLAForwardMixin)
    attn.current_attention_backend = "tokenspeed_mla"
    attn.num_local_heads = 1
    attn.kv_lora_rank = 2
    attn.v_head_dim = 2
    attn.use_deep_gemm_bmm = False
    attn.w_vc = torch.eye(2, dtype=torch.bfloat16).unsqueeze(0)
    attn.o_proj = Mock(side_effect=lambda value: (value, None))
    attn.next_skip_topk = None
    attn._skip_rope_for_dsa_tilelang_fused = lambda: False
    attn.attn_mha = SimpleNamespace(layer_id=3)
    attn.attn_mqa = Mock(return_value=torch.zeros(1, 2, dtype=torch.bfloat16))
    attn.attn_mqa.layer_id = 3
    prepared_query = torch.zeros(1, 1, 4, dtype=torch.float8_e4m3fn)
    backend = SimpleNamespace(
        uses_mla_absorb_frontend=Mock(return_value=True),
        prepare_mla_absorb_qkv=Mock(return_value=prepared_query),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    with (
        patch(
            "sglang.srt.models.deepseek_common.attention_forward_methods."
            "forward_mla.get_attn_backend",
            return_value=backend,
        ),
        patch(
            "sglang.srt.models.deepseek_common.attention_forward_methods."
            "forward_mla.dcp_enabled",
            return_value=False,
        ),
        patch(
            "sglang.srt.models.deepseek_common.attention_forward_methods."
            "forward_mla.is_kv_b_lora_active",
            return_value=False,
        ),
    ):
        output = attn.forward_absorb_core(
            q_pe=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
            k_pe=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
            q_nope_out=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
            k_nope=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
            forward_batch=batch,
            zero_allocator=None,
            positions=torch.zeros(1, dtype=torch.int64),
            topk_indices=None,
            llama_4_scaling=None,
            fusion_plan=None,
        )

    assert output.shape == (1, 2)
    backend.prepare_mla_absorb_qkv.assert_called_once()
    call = attn.attn_mqa.call_args
    assert call.args[0] is prepared_query
    assert call.args[1:3] == (None, None)
    assert call.kwargs["q_rope"] is None
    assert call.kwargs["k_rope"] is None
    assert call.kwargs["save_kv_cache"] is False


def test_tq4_absorb_rotation_does_not_override_mha_prefill():
    # The fused rotations live only in the 512-wide MLA absorb weights. MHA
    # prefill uses independent 192-wide Q/K and 128-wide V projections, so the
    # TokenSpeed backend must preserve the parent prefill implementation.
    assert TokenspeedMLABackend.forward_extend is TRTLLMMLABackend.forward_extend


def test_h43_frontend_consumes_absorbed_mla_query_and_latent_kv():
    backend = object.__new__(TokenspeedMLABackend)
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend._h43_frontend = True
    backend._tq4_cache = True
    backend._tq4_all_layers = True
    backend._tq4_hotcold_cache = False
    config = SimpleNamespace(
        mla_absorb_rotation_fused=True,
        signs1=torch.ones(512),
        signs2=torch.ones(512),
        k_boundaries=torch.zeros(15),
        k_quant_centroids=torch.zeros(16),
        k_storage_code_lut=torch.arange(16, dtype=torch.uint8),
        k_centroids=torch.zeros(16),
        k_dequant_scale_multiplier=1.0,
    )
    pool = SimpleNamespace(
        start_layer=0,
        is_turboquant_layer=lambda layer_id: layer_id == 3,
        kv_nope_packed_buffer=[
            None,
            None,
            None,
            torch.empty(8, 1, 256, dtype=torch.uint8),
        ],
        kv_nope_scale_buffer=[
            None,
            None,
            None,
            torch.empty(8, 1, dtype=torch.bfloat16),
        ],
        kv_rope_buffer=[
            None,
            None,
            None,
            torch.empty(8, 1, 64, dtype=torch.float8_e4m3fn),
        ],
        kv_nope_codebook_buffer=[
            None,
            None,
            None,
            torch.empty(8, 1, 16, dtype=torch.uint8),
        ],
        tq_mla_frontend_fault_status=torch.zeros(1, dtype=torch.int32),
    )
    backend._tq_pool = pool
    backend._tq_config = config

    layer = SimpleNamespace(layer_id=3)
    q_nope = torch.zeros(5, 8, 512, dtype=torch.bfloat16)
    q_rope = torch.zeros(5, 8, 64, dtype=torch.bfloat16)
    k_nope = torch.zeros(5, 1, 512, dtype=torch.bfloat16)
    k_rope = torch.zeros(5, 1, 64, dtype=torch.bfloat16)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        out_cache_loc=torch.arange(5, dtype=torch.int64),
    )

    with patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend.tq_mla_frontend_out"
    ) as frontend:
        query = backend.prepare_mla_absorb_qkv(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            layer=layer,
            forward_batch=forward_batch,
        )

    frontend.assert_called_once()
    assert query.shape == (5, 8, 576)
    assert query.dtype == torch.float8_e4m3fn
    assert backend._h43_frontend_dispatch_count == 1
    assert backend._h43_frontend_dispatches_by_mode == {"decode": 1}


def _make_prepared_h43_parent_backend():
    backend = object.__new__(TokenspeedMLABackend)
    backend.data_type = torch.bfloat16
    backend.q_data_type = torch.bfloat16
    backend._h43_frontend = True
    backend._tq4_cache = True
    backend._tq4_hotcold_cache = False
    backend._tq_pool = SimpleNamespace(is_turboquant_layer=lambda layer_id: True)
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.token_to_kv_pool = Mock()
    backend._get_decode_kv_cache = Mock(return_value=torch.empty(1, dtype=torch.uint8))
    backend._run_decode_kernel = Mock()
    return backend


def test_h43_prepared_decode_query_reaches_reader_without_requantization():
    backend = _make_prepared_h43_parent_backend()
    backend.forward_decode_metadata = SimpleNamespace(
        block_kv_indices=torch.tensor([[1]], dtype=torch.int32),
        max_seq_len_k=64,
        batch_size=1,
    )
    backend._run_decode_kernel.return_value = torch.zeros(1, 1, 1, 512)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.tensor([64], dtype=torch.int32),
        positions=torch.tensor([63], dtype=torch.int64),
        decode_trtllm_mla_metadata=None,
        batch_size=1,
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=1,
        v_head_dim=512,
        head_dim=576,
    )
    query = torch.zeros(1, 1, 576, dtype=torch.float8_e4m3fn)

    with patch(
        "sglang.srt.layers.attention.trtllm_mla_backend."
        "mla_quantize_and_rope_for_fp8"
    ) as ordinary_frontend:
        TRTLLMMLABackend.forward_decode(
            backend,
            query,
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
        )

    ordinary_frontend.assert_not_called()
    call = backend._run_decode_kernel.call_args.kwargs
    assert call["query"].dtype == torch.float8_e4m3fn
    assert call["forward_mode"] == ForwardMode.DECODE


def test_h43_prepared_verify_query_reaches_reader_without_requantization():
    backend = _make_prepared_h43_parent_backend()
    backend.forward_prefill_metadata = None
    metadata = SimpleNamespace(
        block_kv_indices=torch.tensor([[1]], dtype=torch.int32),
        max_seq_len_k=64,
        seq_lens_k=torch.tensor([69], dtype=torch.int32),
        batch_size=1,
    )
    backend.forward_decode_metadata = metadata
    backend._run_decode_kernel.return_value = torch.zeros(1, 5, 1, 512)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        positions=torch.arange(5, dtype=torch.int64),
        decode_trtllm_mla_metadata=metadata,
        spec_info=SimpleNamespace(draft_token_num=5, topk=1),
        batch_size=1,
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=1,
        v_head_dim=512,
        head_dim=576,
    )
    query = torch.zeros(5, 1, 576, dtype=torch.float8_e4m3fn)

    with patch(
        "sglang.srt.layers.attention.trtllm_mla_backend."
        "mla_quantize_and_rope_for_fp8"
    ) as ordinary_frontend:
        TRTLLMMLABackend.forward_extend(
            backend,
            query,
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
        )

    ordinary_frontend.assert_not_called()
    call = backend._run_decode_kernel.call_args.kwargs
    assert call["query"].dtype == torch.float8_e4m3fn
    assert call["forward_mode"] == ForwardMode.TARGET_VERIFY


def _make_attestation_backend(mode: str):
    backend = object.__new__(TokenspeedMLABackend)
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend._h43_model_runner = SimpleNamespace(tp_rank=3, pp_rank=0, gpu_id=3)
    backend._h43_is_draft_worker = False
    backend._h43_frontend_dispatch_count = 0
    backend._h43_reader_dispatch_count = 0
    backend._h43_frontend_dispatches_by_mode = {}
    backend._h43_reader_dispatches_by_mode = {}
    backend._h43_frontend_query_dtypes = {}
    backend._h43_reader_query_dtypes = {}
    backend._h43_first_success_events = set()

    if mode == "fp8":
        backend._tq_pool = None
        backend._tq4_cache = False
        backend._tq_config = None
        backend._h43_frontend = False
        return backend

    selected = frozenset(range(24, 38))
    codebook = None
    rope_dtype = torch.bfloat16
    all_layer_bytes = 32_476
    if mode == "h43":
        codebook = [
            torch.empty(1, dtype=torch.uint8) if i in selected else None
            for i in range(61)
        ]
        rope_dtype = torch.float8_e4m3fn
        all_layer_bytes = 31_804
    rope = [
        torch.empty(1, dtype=rope_dtype) if i in selected else None for i in range(61)
    ]
    backend._tq_pool = SimpleNamespace(
        turboquant_layer_ids=selected,
        start_layer=0,
        end_layer=61,
        layer_num=61,
        kv_nope_codebook_buffer=codebook,
        kv_rope_buffer=rope,
        tq_mla_frontend_fault_status=(
            torch.zeros(1, dtype=torch.int32) if mode == "h43" else None
        ),
        get_per_token_all_layer_bytes=lambda: all_layer_bytes,
    )
    backend._tq4_cache = True
    backend._tq_config = SimpleNamespace(e2m1=True)
    backend._h43_frontend = mode == "h43"
    return backend


def test_h43_runtime_attestation_fails_closed_and_proves_negative_roles():
    unloaded = {
        "native_loaded": False,
        "native_prebuilt": False,
        "native_path": None,
        "native_sha256": None,
    }
    with patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend."
        "get_tq_mla_frontend_module_attestation",
        return_value=unloaded,
    ):
        fp8 = _make_attestation_backend("fp8").emit_h43_runtime_attestation(
            "post_cuda_graph"
        )
        e2m1 = _make_attestation_backend("e2m1").emit_h43_runtime_attestation(
            "post_cuda_graph"
        )

    assert fp8["selected_row_bytes"] == 576
    assert not fp8["tq_enabled"]
    assert e2m1["selected_row_bytes"] == 386
    assert e2m1["selected_layer_ids"] == list(range(24, 38))
    assert e2m1["codebook_slots"] == 0
    assert e2m1["fp8_rope_slots"] == 0

    loaded = {
        "native_loaded": True,
        "native_prebuilt": True,
        "native_path": "/opt/native/h43.so",
        "native_sha256": "a" * 64,
    }
    disabled = _make_attestation_backend("e2m1")
    with (
        patch(
            "sglang.srt.layers.attention.tokenspeed_mla_backend."
            "get_tq_mla_frontend_module_attestation",
            return_value=loaded,
        ),
        pytest.raises(RuntimeError, match="disabled H43 path loaded"),
    ):
        disabled.emit_h43_runtime_attestation("post_cuda_graph")

    h43 = _make_attestation_backend("h43")
    with patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend."
        "get_tq_mla_frontend_module_attestation",
        return_value=loaded,
    ):
        init = h43.emit_h43_runtime_attestation("init")
        assert init["selected_row_bytes"] == 338
        assert init["codebook_slots"] == 14
        assert init["fp8_rope_slots"] == 14
        with pytest.raises(RuntimeError, match="did not traverse"):
            h43.emit_h43_runtime_attestation("post_cuda_graph")
        h43._record_h43_launch(
            "frontend", ForwardMode.TARGET_VERIFY, torch.float8_e4m3fn
        )
        h43._record_h43_launch("reader", ForwardMode.TARGET_VERIFY, torch.float8_e4m3fn)
        post = h43.emit_h43_runtime_attestation("post_cuda_graph")

    assert post["frontend_dispatches"] == 1
    assert post["reader_dispatches"] == 1
    assert post["frontend_fault_status"] == 0
    assert post["frontend_path"] == "mla_absorb"
    assert post["frontend_dispatches_by_mode"] == {"target_verify": 1}
    assert post["reader_dispatches_by_mode"] == {"target_verify": 1}
    assert post["frontend_query_dtypes"] == {"torch.float8_e4m3fn": 1}
    assert post["reader_query_dtypes"] == {"torch.float8_e4m3fn": 1}


def test_h43_runtime_attestation_rejects_mapped_buffer_mode_and_dtype_drift():
    loaded = {
        "native_loaded": True,
        "native_prebuilt": True,
        "native_path": "/opt/native/h43.so",
        "native_sha256": "a" * 64,
    }
    with patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend."
        "get_tq_mla_frontend_module_attestation",
        return_value=loaded,
    ):
        mapped_wrong = _make_attestation_backend("h43")
        mapped_wrong._tq_pool.kv_nope_codebook_buffer[24] = None
        mapped_wrong._tq_pool.kv_nope_codebook_buffer[23] = torch.empty(
            1, dtype=torch.uint8
        )
        with pytest.raises(RuntimeError, match="codebook layers do not match"):
            mapped_wrong.emit_h43_runtime_attestation("init")

        mode_wrong = _make_attestation_backend("h43")
        mode_wrong._record_h43_launch(
            "frontend", ForwardMode.DECODE, torch.float8_e4m3fn
        )
        mode_wrong._record_h43_launch(
            "reader", ForwardMode.TARGET_VERIFY, torch.float8_e4m3fn
        )
        with pytest.raises(RuntimeError, match="per-mode"):
            mode_wrong.emit_h43_runtime_attestation("post_cuda_graph")

        dtype_wrong = _make_attestation_backend("h43")
        dtype_wrong._record_h43_launch(
            "frontend", ForwardMode.DECODE, torch.float8_e4m3fn
        )
        dtype_wrong._record_h43_launch("reader", ForwardMode.DECODE, torch.bfloat16)
        with pytest.raises(RuntimeError, match="FP8 query"):
            dtype_wrong.emit_h43_runtime_attestation("post_cuda_graph")

        sticky_fault = _make_attestation_backend("h43")
        sticky_fault._record_h43_launch(
            "frontend", ForwardMode.DECODE, torch.float8_e4m3fn
        )
        sticky_fault._record_h43_launch(
            "reader", ForwardMode.DECODE, torch.float8_e4m3fn
        )
        sticky_fault._tq_pool.tq_mla_frontend_fault_status.fill_(1)
        with pytest.raises(RuntimeError, match="sticky device fault"):
            sticky_fault.emit_h43_runtime_attestation("post_cuda_graph")

        with pytest.raises(RuntimeError, match="unknown H43 attestation phase"):
            _make_attestation_backend("h43").emit_h43_runtime_attestation("ready")


def test_h43_runtime_attestation_filters_global_layers_for_pipeline_rank():
    backend = _make_attestation_backend("h43")
    backend._tq_pool.start_layer = 30
    backend._tq_pool.end_layer = 40
    backend._tq_pool.layer_num = 10
    backend._tq_pool.kv_nope_codebook_buffer = [
        torch.empty(1, dtype=torch.uint8) if i < 8 else None for i in range(10)
    ]
    backend._tq_pool.kv_rope_buffer = [
        torch.empty(1, dtype=torch.float8_e4m3fn) if i < 8 else None for i in range(10)
    ]
    backend._tq_pool.get_per_token_all_layer_bytes = lambda: 3_856
    loaded = {
        "native_loaded": True,
        "native_prebuilt": True,
        "native_path": "/opt/native/h43.so",
        "native_sha256": "a" * 64,
    }

    with patch(
        "sglang.srt.layers.attention.tokenspeed_mla_backend."
        "get_tq_mla_frontend_module_attestation",
        return_value=loaded,
    ):
        payload = backend.emit_h43_runtime_attestation("init")

    assert payload["selected_layer_ids"] == list(range(30, 38))
    assert payload["codebook_layer_ids"] == list(range(30, 38))
    assert payload["fp8_rope_layer_ids"] == list(range(30, 38))
    assert payload["pool_all_layer_bytes"] == 3_856


def test_h43_native_module_attestation_does_not_load_module(tmp_path):
    from sglang.jit_kernel import tq_mla_frontend

    native = tmp_path / "h43.so"
    native.write_bytes(b"sealed-native")
    with (
        patch.object(tq_mla_frontend, "_MODULE", None),
        patch.object(tq_mla_frontend, "_MODULE_PREBUILT", False),
    ):
        state = tq_mla_frontend.get_tq_mla_frontend_module_attestation()
    assert not state["native_loaded"]

    module = SimpleNamespace(__file__=str(native))
    with (
        patch.object(tq_mla_frontend, "_MODULE", module),
        patch.object(tq_mla_frontend, "_MODULE_PREBUILT", True),
    ):
        state = tq_mla_frontend.get_tq_mla_frontend_module_attestation()
    assert state["native_loaded"]
    assert state["native_prebuilt"]
    assert state["native_path"] == str(native)
    assert len(state["native_sha256"]) == 64


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


@pytest.mark.parametrize(
    "disable_chunked_prefix_cache,prefill_graph_backend",
    [(True, "disabled"), (False, "tc_piecewise")],
)
def test_tq4_rotation_fusion_skips_generic_extend_fallbacks(
    disable_chunked_prefix_cache, prefill_graph_backend
):
    runner = object.__new__(ModelRunner)
    runner.turboquant_bits = 4
    runner.token_to_kv_pool_allocator = SimpleNamespace(
        get_kvcache=lambda: SimpleNamespace(tq_config=SimpleNamespace())
    )
    runner.use_mla_backend = True
    runner.server_args = SimpleNamespace(
        enable_lora=False,
        disable_chunked_prefix_cache=disable_chunked_prefix_cache,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=prefill_graph_backend)
        ),
        get_attention_backends=lambda: ("tokenspeed_mla", "tokenspeed_mla"),
    )
    runner._maybe_fuse_tq_mla_absorb_rotations = Mock()

    ModelRunner._maybe_fuse_tq_output_rotation(runner)

    runner._maybe_fuse_tq_mla_absorb_rotations.assert_not_called()


def test_tq4_rotation_fusion_is_idempotent():
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        named_modules=Mock(side_effect=AssertionError("weights revisited"))
    )
    tq_config = SimpleNamespace(mla_absorb_rotation_fused=True)
    logger = Mock()

    ModelRunner._maybe_fuse_tq_mla_absorb_rotations(runner, tq_config, logger)

    runner.model.named_modules.assert_not_called()


def test_tq4_rotation_fusion_changes_only_selected_layers():
    def make_layer(layer_id):
        return SimpleNamespace(
            layer_id=layer_id,
            w_kc=torch.full((2, 3, 4), float(layer_id), dtype=torch.bfloat16),
            w_vc=torch.full((2, 4, 3), float(layer_id), dtype=torch.bfloat16),
        )

    layer0 = make_layer(0)
    layer1 = make_layer(1)
    layer0_k = layer0.w_kc.clone()
    layer0_v = layer0.w_vc.clone()
    tq_config = SimpleNamespace(
        mla_absorb_rotation_fused=False,
        head_dim=4,
        fuse_mla_absorb_rotations=Mock(
            side_effect=lambda w_kc, w_vc: (w_kc + 10, w_vc + 20)
        ),
    )
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        named_modules=lambda: iter(
            (("model.layers.0.self_attn", layer0), ("model.layers.1.self_attn", layer1))
        )
    )

    ModelRunner._maybe_fuse_tq_mla_absorb_rotations(
        runner, tq_config, Mock(), selected_layer_ids={1}
    )

    torch.testing.assert_close(layer0.w_kc, layer0_k)
    torch.testing.assert_close(layer0.w_vc, layer0_v)
    torch.testing.assert_close(layer1.w_kc, torch.full_like(layer1.w_kc, 11))
    torch.testing.assert_close(layer1.w_vc, torch.full_like(layer1.w_vc, 21))
    assert tq_config.mla_absorb_rotation_fused
    tq_config.fuse_mla_absorb_rotations.assert_called_once()


def test_tq4_rotation_fusion_fails_closed_if_a_selected_layer_is_missing():
    layer = SimpleNamespace(
        layer_id=1,
        w_kc=torch.zeros((2, 3, 4), dtype=torch.bfloat16),
        w_vc=torch.zeros((2, 4, 3), dtype=torch.bfloat16),
    )
    tq_config = SimpleNamespace(
        mla_absorb_rotation_fused=False,
        head_dim=4,
        fuse_mla_absorb_rotations=Mock(
            side_effect=lambda w_kc, w_vc: (w_kc + 1, w_vc + 1)
        ),
    )
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        named_modules=lambda: iter((("model.layers.1.self_attn", layer),))
    )

    ModelRunner._maybe_fuse_tq_mla_absorb_rotations(
        runner, tq_config, Mock(), selected_layer_ids={1, 2}
    )

    assert not tq_config.mla_absorb_rotation_fused
    tq_config.fuse_mla_absorb_rotations.assert_not_called()


def test_tq4_rotation_fusion_fails_closed_for_duplicate_selected_layer():
    def make_layer():
        return SimpleNamespace(
            layer_id=1,
            w_kc=torch.zeros((2, 3, 4), dtype=torch.bfloat16),
            w_vc=torch.zeros((2, 4, 3), dtype=torch.bfloat16),
        )

    tq_config = SimpleNamespace(
        mla_absorb_rotation_fused=False,
        head_dim=4,
        fuse_mla_absorb_rotations=Mock(
            side_effect=lambda w_kc, w_vc: (w_kc + 1, w_vc + 1)
        ),
    )
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        named_modules=lambda: iter(
            (("model.layers.1.self_attn", make_layer()), ("duplicate", make_layer()))
        )
    )

    ModelRunner._maybe_fuse_tq_mla_absorb_rotations(
        runner, tq_config, Mock(), selected_layer_ids={1}
    )

    assert not tq_config.mla_absorb_rotation_fused
    tq_config.fuse_mla_absorb_rotations.assert_not_called()
