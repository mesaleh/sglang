from types import SimpleNamespace

from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration


def _language_model_with_attn_layers(*attn_layers):
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attn) for attn in attn_layers],
            start_layer=0,
            end_layer=len(attn_layers),
        )
    )


def test_missing_mla_absorb_weight_layers_detects_unpacked_layers():
    packed = SimpleNamespace(kv_b_proj=object(), w_kc=object(), w_vc=object())
    missing_kc = SimpleNamespace(kv_b_proj=object(), w_kc=None, w_vc=object())
    missing_vc = SimpleNamespace(kv_b_proj=object(), w_kc=object(), w_vc=None)
    non_mla = SimpleNamespace(w_kc=None, w_vc=None)

    language_model = _language_model_with_attn_layers(
        packed, missing_kc, missing_vc, non_mla
    )

    assert KimiK25ForConditionalGeneration._missing_mla_absorb_weight_layers(
        language_model
    ) == [1, 2]


def test_missing_mla_absorb_weight_layers_respects_layer_bounds():
    missing = SimpleNamespace(kv_b_proj=object(), w_kc=None, w_vc=None)
    packed = SimpleNamespace(kv_b_proj=object(), w_kc=object(), w_vc=object())
    language_model = _language_model_with_attn_layers(missing, packed)
    language_model.model.start_layer = 1

    assert (
        KimiK25ForConditionalGeneration._missing_mla_absorb_weight_layers(
            language_model
        )
        == []
    )


def test_load_weights_forces_full_post_load_when_mla_weights_are_missing():
    attn = SimpleNamespace(kv_b_proj=object(), w_kc=None, w_vc=None)

    class FakeLanguageModel:
        def __init__(self):
            self.model = _language_model_with_attn_layers(attn).model
            self.loaded_weights = None
            self.post_load_calls = []

        def load_weights(self, weights):
            self.loaded_weights = list(weights)

        def post_load_weights(self, weight_names=None):
            self.post_load_calls.append(weight_names)
            attn.w_kc = object()
            attn.w_vc = object()

    kimi = object.__new__(KimiK25ForConditionalGeneration)
    kimi.config = SimpleNamespace(
        encoder_only=False,
        language_only=True,
    )
    kimi.hf_to_sglang_mapper = None
    kimi.language_model = FakeLanguageModel()

    KimiK25ForConditionalGeneration.load_weights(kimi, [("model.layers.0.x", object())])

    assert kimi.language_model.post_load_calls == [None]
    assert len(kimi.language_model.loaded_weights) == 1
