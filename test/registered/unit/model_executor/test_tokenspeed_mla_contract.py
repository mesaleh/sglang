from inspect import Parameter, signature
from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention import cutedsl_mla_backend as cutedsl_module
from sglang.srt.layers.attention import tokenspeed_mla_backend as backend_module
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
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
        (param.name, param.kind, param.default is not Parameter.empty)
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
    forward_mode = SimpleNamespace(is_decode_or_idle=lambda: True)
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
