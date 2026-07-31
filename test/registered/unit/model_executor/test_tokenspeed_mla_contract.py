import pytest

from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    _custom_decode_mask_kwargs,
    _supports_custom_decode_mask,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
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
