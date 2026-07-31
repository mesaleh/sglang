"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))


class TestResolveLayerIndices(unittest.TestCase):
    def _mtp_target_stage(self):
        model = SimpleNamespace(start_layer=0, end_layer=32)
        model_config = SimpleNamespace(
            num_nextn_predict_layers=1,
            num_hidden_layers=64,
            num_attention_layers=64,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV3ForCausalLM"],
                loop_num=1,
            ),
        )
        return model, model_config

    def test_partial_mtp_target_rejected_without_external_draft_pp(self):
        model, model_config = self._mtp_target_stage()
        with self.assertRaisesRegex(AssertionError, "PP is not compatible"):
            resolve_layer_indices(
                model=model,
                model_config=model_config,
                is_draft_worker=False,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
            )

    def test_external_draft_pp_allows_partial_mtp_target(self):
        model, model_config = self._mtp_target_stage()
        layer_info = resolve_layer_indices(
            model=model,
            model_config=model_config,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            allow_external_draft_pp=True,
        )
        self.assertEqual(layer_info.start_layer, 0)
        self.assertEqual(layer_info.end_layer, 32)
        self.assertEqual(layer_info.num_effective_layers, 32)


if __name__ == "__main__":
    unittest.main()
