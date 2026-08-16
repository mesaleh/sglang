from inspect import Parameter, signature
from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention import cutedsl_mla_backend as cutedsl_module
from sglang.srt.layers.attention import tokenspeed_mla_backend as backend_module
from sglang.srt.layers.attention.attention_registry import (
    create_tokenspeed_mla_backend,
)
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    TokenspeedTQE2M1MLABackend,
    _custom_decode_mask_kwargs,
    _supports_custom_decode_mask,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
    _target_verify_max_seq_len,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_tokenspeed_factory_keeps_dense_and_n10_backends_distinct(monkeypatch):
    dense = object()
    n10 = object()
    monkeypatch.setattr(backend_module, "TokenspeedMLABackend", lambda runner: dense)
    monkeypatch.setattr(
        backend_module, "TokenspeedTQE2M1MLABackend", lambda runner: n10
    )

    runner = SimpleNamespace(use_mla_backend=True, kv_cache_dtype_str="fp8_e4m3")
    assert create_tokenspeed_mla_backend(runner) is dense

    runner.kv_cache_dtype_str = "turboquant_4bit_e2m1_recip_bf16"
    assert create_tokenspeed_mla_backend(runner) is n10


def test_n10_cache_contract_does_not_weaken_dense_fp8_contract(monkeypatch):
    from sglang.srt.mem_cache.memory_pool import (
        MLATokenToKVPoolNativeE2M1RecipBF16,
    )

    pool = MLATokenToKVPoolNativeE2M1RecipBF16(
        size=32,
        page_size=32,
        dtype=backend_module.torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    runner = SimpleNamespace(
        kv_cache_dtype_str="turboquant_4bit_e2m1_recip_bf16",
        token_to_kv_pool=pool,
    )
    backend = object.__new__(TokenspeedTQE2M1MLABackend)
    backend.data_type = backend_module.torch.bfloat16
    backend.page_size = 32
    backend.num_q_heads = 8
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend._unified_mla = False
    backend.speculative_topk = 1

    monkeypatch.setattr(
        backend_module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=False)
    )
    monkeypatch.setattr(backend_module, "is_tokenspeed_mla_available", lambda: True)
    monkeypatch.setattr(
        backend_module,
        "tokenspeed_mla",
        SimpleNamespace(tokenspeed_mla_decode_tq_e2m1=lambda **kwargs: None),
        raising=False,
    )
    backend._validate_cache_contract(runner)

    dense_backend = object.__new__(TokenspeedMLABackend)
    dense_backend.data_type = backend_module.torch.bfloat16
    dense_backend.page_size = 32
    with pytest.raises(ValueError, match="requires --kv-cache-dtype fp8_e4m3"):
        dense_backend._validate_cache_contract(runner)

    backend.num_q_heads = 4
    with pytest.raises(ValueError, match="page32/H8/latent512/RoPE64"):
        backend._validate_cache_contract(runner)


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


def _signature_contract(fn):
    return [
        (
            param.name,
            param.kind,
            param.default if param.default is not Parameter.empty else Parameter.empty,
        )
        for param in signature(fn).parameters.values()
    ]


@pytest.mark.parametrize("backend_cls", [TokenspeedMLABackend, CuteDslMLABackend])
def test_mla_cuda_graph_metadata_overrides_match_trtllm_signature(backend_cls):
    assert _signature_contract(backend_cls._init_cuda_graph_metadata) == (
        _signature_contract(TRTLLMMLABackend._init_cuda_graph_metadata)
    )
    assert _signature_contract(backend_cls._apply_cuda_graph_metadata) == (
        _signature_contract(TRTLLMMLABackend._apply_cuda_graph_metadata)
    )
    assert _signature_contract(backend_cls._run_decode_kernel) == (
        _signature_contract(TRTLLMMLABackend._run_decode_kernel)
    )


def test_tokenspeed_cuda_graph_metadata_forwards_upstream_kwargs(monkeypatch):
    backend = object.__new__(TokenspeedMLABackend)
    req_pool_indices = object()
    spec_info = object()
    calls = []

    def fake_init(
        self,
        bs,
        num_tokens,
        forward_mode,
        seq_lens,
        device,
        req_pool_indices=None,
        spec_info=None,
    ):
        calls.append(("init", bs, req_pool_indices, spec_info))
        self.forward_decode_metadata = TRTLLMMLADecodeMetadata(max_seq_len_k=1)

    def fake_apply(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info=None,
    ):
        calls.append(("apply", bs, req_pool_indices, spec_info))
        return "forwarded"

    monkeypatch.setattr(TRTLLMMLABackend, "_init_cuda_graph_metadata", fake_init)
    monkeypatch.setattr(TRTLLMMLABackend, "_apply_cuda_graph_metadata", fake_apply)
    monkeypatch.setattr(
        backend_module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=False)
    )

    backend._init_cuda_graph_metadata(
        4,
        4,
        object(),
        object(),
        object(),
        req_pool_indices=req_pool_indices,
        spec_info=spec_info,
    )

    assert (
        backend._apply_cuda_graph_metadata(
            4,
            req_pool_indices,
            object(),
            object(),
            spec_info=spec_info,
        )
        == "forwarded"
    )
    assert calls == [
        ("init", 4, req_pool_indices, spec_info),
        ("apply", 4, req_pool_indices, spec_info),
    ]


@pytest.mark.parametrize(
    ("backend_cls", "module"),
    [
        (TokenspeedMLABackend, backend_module),
        (CuteDslMLABackend, cutedsl_module),
    ],
)
def test_mla_dcp_cuda_graph_metadata_fails_closed_for_draft_frontier(
    monkeypatch, backend_cls, module
):
    backend = object.__new__(backend_cls)
    forward_mode = SimpleNamespace(
        is_decode_or_idle=lambda: True,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )
    spec_info = SimpleNamespace(
        kv_indptr=SimpleNamespace(shape=(4,)),
        kv_indices=object(),
    )

    monkeypatch.setattr(
        module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)
    )

    with pytest.raises(RuntimeError, match="DCP CUDA graph metadata"):
        backend._apply_cuda_graph_metadata(
            bs=2,
            req_pool_indices=object(),
            seq_lens=object(),
            forward_mode=forward_mode,
            spec_info=spec_info,
        )


@pytest.mark.parametrize(
    ("backend_cls", "module"),
    [
        (TokenspeedMLABackend, backend_module),
        (CuteDslMLABackend, cutedsl_module),
    ],
)
def test_mla_dcp_eager_metadata_fails_closed_for_draft_frontier(
    monkeypatch, backend_cls, module
):
    backend = object.__new__(backend_cls)
    forward_mode = SimpleNamespace(
        is_decode_or_idle=lambda: True,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )
    forward_batch = SimpleNamespace(forward_mode=forward_mode)

    def fake_init_forward_metadata(self, forward_batch):
        self.forward_decode_metadata = SimpleNamespace(
            is_draft_frontier=True,
            max_seq_len_k=1,
        )

    monkeypatch.setattr(
        TRTLLMMLABackend, "init_forward_metadata", fake_init_forward_metadata
    )
    monkeypatch.setattr(
        module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)
    )

    with pytest.raises(RuntimeError, match="DCP metadata"):
        backend.init_forward_metadata(forward_batch)


@pytest.mark.parametrize(
    ("backend_cls", "module"),
    [
        (TokenspeedMLABackend, backend_module),
        (CuteDslMLABackend, cutedsl_module),
    ],
)
def test_mla_dcp_draft_extend_cuda_graph_metadata_fails_closed(
    monkeypatch, backend_cls, module
):
    backend = object.__new__(backend_cls)
    forward_mode = SimpleNamespace(is_draft_extend_v2=lambda: True)
    monkeypatch.setattr(
        module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)
    )

    with pytest.raises(RuntimeError, match="draft-extend speculative decoding"):
        backend._apply_cuda_graph_metadata(
            bs=2,
            req_pool_indices="req",
            seq_lens="seq",
            forward_mode=forward_mode,
            spec_info="spec",
        )


@pytest.mark.parametrize(
    ("backend_cls", "module"),
    [
        (TokenspeedMLABackend, backend_module),
        (CuteDslMLABackend, cutedsl_module),
    ],
)
def test_mla_dcp_draft_extend_init_metadata_fails_closed(
    monkeypatch, backend_cls, module
):
    backend = object.__new__(backend_cls)
    forward_mode = SimpleNamespace(
        is_draft_extend_v2=lambda: True,
        is_decode_or_idle=lambda: False,
        is_target_verify=lambda: False,
    )
    forward_batch = SimpleNamespace(forward_mode=forward_mode)
    monkeypatch.setattr(
        module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)
    )

    with pytest.raises(RuntimeError, match="draft-extend speculative decoding"):
        backend.init_forward_metadata(forward_batch)


@pytest.mark.parametrize(
    ("backend_cls", "module"),
    [
        (TokenspeedMLABackend, backend_module),
        (CuteDslMLABackend, cutedsl_module),
    ],
)
def test_mla_dcp_decode_cuda_graph_metadata_uses_rank_local_page_table(
    monkeypatch, backend_cls, module
):
    class CopySink:
        def __init__(self):
            self.copied = None

        def copy_(self, value):
            self.copied = value

    backend = object.__new__(backend_cls)
    sink = CopySink()
    global_sink = CopySink()
    metadata = SimpleNamespace(
        block_kv_indices="blocks",
        seq_lens_k=sink,
        global_seq_lens_k=global_sink,
    )
    backend.decode_cuda_graph_metadata = {2: metadata}
    backend._is_draft_frontier_spec = lambda forward_mode, spec_info, bs: False
    backend._get_dcp_local_seq_lens = lambda seq_lens: ("local", tuple(seq_lens))
    calls = []
    backend._fill_dcp_block_kv_indices = lambda blocks, req, local: calls.append(
        (blocks, req, local)
    )
    forward_mode = SimpleNamespace(
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )
    monkeypatch.setattr(
        module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)
    )

    backend._apply_cuda_graph_metadata(
        bs=2,
        req_pool_indices=["r0", "r1", "pad"],
        seq_lens=[11, 12, 13],
        forward_mode=forward_mode,
        spec_info=None,
    )

    assert sink.copied == ("local", (11, 12))
    if backend_cls is CuteDslMLABackend:
        assert global_sink.copied == [11, 12]
    else:
        assert global_sink.copied is None
    assert calls == [("blocks", ["r0", "r1"], sink)]


def test_cutedsl_decode_forwards_custom_mask_on_non_dcp_path(monkeypatch):
    backend = object.__new__(CuteDslMLABackend)
    calls = []

    def fake_run_decode_kernel(
        self,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        layer,
        custom_mask=None,
        custom_mask_offsets=None,
    ):
        calls.append((custom_mask, custom_mask_offsets))
        return "parent-decode"

    monkeypatch.setattr(TRTLLMMLABackend, "_run_decode_kernel", fake_run_decode_kernel)

    assert (
        backend._run_decode_kernel(
            query="q",
            kv_cache="kv",
            block_tables="blocks",
            seq_lens="seq",
            max_seq_len=8,
            layer="layer",
            custom_mask="mask",
            custom_mask_offsets="offsets",
        )
        == "parent-decode"
    )
    assert calls == [("mask", "offsets")]


def test_cutedsl_dcp_decode_fails_closed_for_custom_mask():
    backend = object.__new__(CuteDslMLABackend)

    with pytest.raises(RuntimeError, match="custom-mask"):
        backend._run_decode_kernel(
            query="q",
            kv_cache="kv",
            block_tables="blocks",
            seq_lens="seq",
            max_seq_len=8,
            layer="layer",
            custom_mask="mask",
            custom_mask_offsets="offsets",
            cp_world=2,
        )


def test_cutedsl_cuda_graph_metadata_forwards_upstream_kwargs(monkeypatch):
    backend = object.__new__(CuteDslMLABackend)
    req_pool_indices = object()
    spec_info = object()
    calls = []

    def fake_init(
        self,
        bs,
        num_tokens,
        forward_mode,
        seq_lens,
        device,
        req_pool_indices=None,
        spec_info=None,
    ):
        calls.append(("init", bs, req_pool_indices, spec_info))
        self.forward_decode_metadata = TRTLLMMLADecodeMetadata(max_seq_len_k=1)

    def fake_apply(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info=None,
    ):
        calls.append(("apply", bs, req_pool_indices, spec_info))
        return "forwarded"

    monkeypatch.setattr(TRTLLMMLABackend, "_init_cuda_graph_metadata", fake_init)
    monkeypatch.setattr(TRTLLMMLABackend, "_apply_cuda_graph_metadata", fake_apply)
    monkeypatch.setattr(
        cutedsl_module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=False)
    )

    backend._init_cuda_graph_metadata(
        4,
        4,
        object(),
        object(),
        object(),
        req_pool_indices=req_pool_indices,
        spec_info=spec_info,
    )

    assert (
        backend._apply_cuda_graph_metadata(
            4,
            req_pool_indices,
            object(),
            object(),
            spec_info=spec_info,
        )
        == "forwarded"
    )
    assert calls == [
        ("init", 4, req_pool_indices, spec_info),
        ("apply", 4, req_pool_indices, spec_info),
    ]
