"""
Unit tests for TurboQuant KV cache compression.

Tests:
1. Codebook construction (Lloyd's algorithm, closed-form centroids)
2. WHT rotation correctness (self-inverse property)
3. Quantizer roundtrip quality (MSE, cosine similarity) with WHT
4. Bit packing/unpacking correctness
5. Zero vector and batch consistency

Usage:
    python -m pytest test/srt/test_turboquant.py -v -k "not GPU"   # CPU only
    python -m pytest test/srt/test_turboquant.py -v -k "GPU"       # GPU only
    python -m pytest test/srt/test_turboquant.py -v                # All
"""

import argparse
import hashlib
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class TestTurboQuantCLI(unittest.TestCase):

    def test_preserved_kv_cache_dtype_choices_parse(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        for dtype in (
            "turboquant_2bit",
            "turboquant_4bit",
            "turboquant_4bit_e2m1",
            "turboquant_4bit_uniform",
            "turboquant_k4v2",
        ):
            args = parser.parse_args(
                ["--model-path", "test-model", "--kv-cache-dtype", dtype]
            )
            self.assertEqual(args.kv_cache_dtype, dtype)

    def test_dtype_parser(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            parse_turboquant_kv_cache_dtype,
        )

        self.assertEqual(
            parse_turboquant_kv_cache_dtype("turboquant_4bit"), (4, 4, False)
        )
        self.assertEqual(
            parse_turboquant_kv_cache_dtype("turboquant_4bit_uniform"),
            (4, 4, True),
        )
        self.assertEqual(
            parse_turboquant_kv_cache_dtype("turboquant_4bit_e2m1"),
            (4, 4, False),
        )
        self.assertEqual(
            parse_turboquant_kv_cache_dtype("turboquant_k4v2"), (4, 2, False)
        )
        self.assertIsNone(parse_turboquant_kv_cache_dtype("bf16"))

    def test_native_e2m1_dtype_identity_is_explicit(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            is_native_e2m1_mla_kv_cache_dtype,
        )

        self.assertTrue(
            is_native_e2m1_mla_kv_cache_dtype("turboquant_4bit_e2m1")
        )
        self.assertFalse(is_native_e2m1_mla_kv_cache_dtype("turboquant_4bit"))
        self.assertFalse(is_native_e2m1_mla_kv_cache_dtype(None))

    def test_native_e2m1_format_fails_closed_until_pool_is_wired(self):
        from sglang.srt.model_executor.model_runner_components.turboquant_compat import (
            validate_turboquant_transfer_compatibility,
        )

        with self.assertRaisesRegex(ValueError, "matched native-E2M1 MLA writer"):
            validate_turboquant_transfer_compatibility(
                kv_cache_dtype="turboquant_4bit_e2m1",
                disaggregation_mode="null",
                enable_hierarchical_cache=False,
                use_mla_backend=True,
                prefill_attention_backend="tokenspeed_mla",
                decode_attention_backend="tokenspeed_mla",
                mla_fused_decode_enabled=True,
            )

    def test_model_runner_configures_mha_backend_through_override(self):
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        class ServerArgs(SimpleNamespace):
            def override(self, source, **fields):
                self.override_source = source
                for name, value in fields.items():
                    setattr(self, name, value)

        runner = object.__new__(ModelRunner)
        runner.server_args = ServerArgs(
            kv_cache_dtype="turboquant_k4v2",
            speculative_draft_attention_backend=None,
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            disable_cuda_graph=False,
        )
        runner.model = SimpleNamespace(quant_config=None)
        runner.dtype = torch.bfloat16
        runner.is_draft_worker = False
        runner.spec_algorithm = SpeculativeAlgorithm.NONE
        runner.use_mla_backend = False

        runner.configure_kv_cache_dtype()

        self.assertEqual(runner.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(runner.turboquant_k_bits, 4)
        self.assertEqual(runner.turboquant_v_bits, 2)
        self.assertEqual(runner.server_args.prefill_attention_backend, "triton")
        self.assertEqual(runner.server_args.decode_attention_backend, "triton")
        self.assertFalse(runner.server_args.disable_cuda_graph)

    def test_model_runner_keeps_mla_backend(self):
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        class ServerArgs(SimpleNamespace):
            def override(self, source, **fields):
                for name, value in fields.items():
                    setattr(self, name, value)

        runner = object.__new__(ModelRunner)
        runner.server_args = ServerArgs(
            kv_cache_dtype="turboquant_4bit",
            speculative_draft_attention_backend=None,
            attention_backend=None,
            prefill_attention_backend="fa3",
            decode_attention_backend="flashmla",
            disable_cuda_graph=False,
        )
        runner.model = SimpleNamespace(quant_config=None)
        runner.dtype = torch.bfloat16
        runner.is_draft_worker = False
        runner.spec_algorithm = SpeculativeAlgorithm.NONE
        runner.use_mla_backend = True

        runner.configure_kv_cache_dtype()

        self.assertEqual(runner.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(runner.server_args.prefill_attention_backend, "fa3")
        self.assertEqual(runner.server_args.decode_attention_backend, "flashmla")

    def test_dflash_fa4_draft_owns_unpacked_bf16_kv(self):
        from copy import deepcopy

        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        class ServerArgs(SimpleNamespace):
            def override(self, source, **fields):
                self.override_source = source
                for name, value in fields.items():
                    setattr(self, name, value)

        target_args = ServerArgs(
            kv_cache_dtype="turboquant_4bit",
            speculative_draft_attention_backend="fa4",
            attention_backend=None,
            prefill_attention_backend="tokenspeed_mla",
            decode_attention_backend="tokenspeed_mla",
            disable_cuda_graph=False,
        )
        draft_args = deepcopy(target_args)
        draft_args.attention_backend = "fa4"
        draft_args.prefill_attention_backend = None
        draft_args.decode_attention_backend = None

        target = object.__new__(ModelRunner)
        target.server_args = target_args
        target.model = SimpleNamespace(quant_config=None)
        target.dtype = torch.bfloat16
        target.is_draft_worker = False
        target.spec_algorithm = SpeculativeAlgorithm.DFLASH
        target.use_mla_backend = True

        draft = object.__new__(ModelRunner)
        draft.server_args = draft_args
        draft.model = SimpleNamespace(quant_config=None)
        draft.dtype = torch.bfloat16
        draft.is_draft_worker = True
        draft.spec_algorithm = SpeculativeAlgorithm.DFLASH
        draft.use_mla_backend = False

        target.configure_kv_cache_dtype()
        draft.configure_kv_cache_dtype()

        self.assertEqual(target.server_args.kv_cache_dtype, "turboquant_4bit")
        self.assertEqual(target.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(target.turboquant_k_bits, 4)
        self.assertEqual(target.server_args.decode_attention_backend, "tokenspeed_mla")

        # v0.5.17 keeps target and draft process configuration immutable and
        # carries the resolved representation on each runner.
        self.assertEqual(draft.server_args.kv_cache_dtype, "turboquant_4bit")
        self.assertEqual(draft.kv_cache_dtype_str, "bf16")
        self.assertEqual(draft.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(draft.server_args.attention_backend, "fa4")
        self.assertIsNone(draft.server_args.prefill_attention_backend)
        self.assertIsNone(draft.server_args.decode_attention_backend)
        self.assertFalse(hasattr(draft, "turboquant_bits"))
        self.assertFalse(hasattr(draft, "turboquant_k_bits"))
        self.assertFalse(hasattr(draft, "turboquant_v_bits"))

    def test_pool_sizing_uses_runner_owned_kv_representation(self):
        from sglang.srt.model_executor.pool_configurator import _get_turboquant_bits

        target = SimpleNamespace(kv_cache_dtype_str="turboquant_k4v2")
        draft = SimpleNamespace(
            kv_cache_dtype_str="bf16",
            server_args=SimpleNamespace(kv_cache_dtype="turboquant_k4v2"),
        )

        self.assertEqual(_get_turboquant_bits(target), (4, 2))
        self.assertIsNone(_get_turboquant_bits(draft))

    def test_output_rotation_fusion_is_all_or_nothing(self):
        from sglang.srt.model_executor.model_runner_components.turboquant_rotation import (
            fuse_turboquant_output_rotation_weights,
        )

        class Config:
            output_rotation_fused = False

            @staticmethod
            def fuse_inverse_rotation_into_o_proj(weight, _n_heads):
                return weight + 1

        weight = torch.nn.Parameter(
            torch.arange(16, dtype=torch.float32).reshape(4, 4)
        )
        cfg = Config()
        original = weight.detach().clone()

        fused = fuse_turboquant_output_rotation_weights(
            cfg,
            [(weight, 2)],
            skipped_layers=1,
        )

        self.assertFalse(fused)
        torch.testing.assert_close(weight, original)
        self.assertFalse(cfg.output_rotation_fused)

    def test_output_rotation_fuses_when_all_layers_are_eligible(self):
        from sglang.srt.model_executor.model_runner_components.turboquant_rotation import (
            fuse_turboquant_output_rotation_weights,
        )

        weight = torch.nn.Parameter(torch.zeros(4, 4))
        cfg = SimpleNamespace(
            output_rotation_fused=False,
            fuse_inverse_rotation_into_o_proj=lambda weight, _n_heads: weight + 1,
        )
        fused = fuse_turboquant_output_rotation_weights(
            cfg,
            [(weight, 2)],
            skipped_layers=0,
        )

        self.assertTrue(fused)
        torch.testing.assert_close(weight, torch.ones_like(weight))
        self.assertTrue(cfg.output_rotation_fused)

    def test_staged_flashmla_jit_does_not_mutate_arch_environment(self):
        from sglang.srt.layers.attention import turboquant_mla_staged_flashmla

        turboquant_mla_staged_flashmla._load_staging_extension.cache_clear()
        previous = os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        try:
            with patch(
                "torch.utils.cpp_extension.load_inline",
                return_value=object(),
            ):
                turboquant_mla_staged_flashmla._load_staging_extension()
            self.assertNotIn("TORCH_CUDA_ARCH_LIST", os.environ)
        finally:
            turboquant_mla_staged_flashmla._load_staging_extension.cache_clear()
            if previous is not None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous

    def test_transfer_modes_fail_closed_for_unsupported_layouts(self):
        from sglang.srt.model_executor.model_runner_components.turboquant_compat import (
            validate_turboquant_transfer_compatibility,
        )

        with self.assertRaisesRegex(ValueError, "PD disaggregation"):
            validate_turboquant_transfer_compatibility(
                kv_cache_dtype="turboquant_4bit",
                disaggregation_mode="decode",
                enable_hierarchical_cache=False,
                use_mla_backend=True,
            )
        with self.assertRaisesRegex(ValueError, "MHA TurboQuant"):
            validate_turboquant_transfer_compatibility(
                kv_cache_dtype="turboquant_k4v2",
                disaggregation_mode="null",
                enable_hierarchical_cache=True,
                use_mla_backend=False,
            )
        with self.assertRaisesRegex(ValueError, "deterministic Triton"):
            validate_turboquant_transfer_compatibility(
                kv_cache_dtype="turboquant_k4v2",
                disaggregation_mode="null",
                enable_hierarchical_cache=False,
                use_mla_backend=False,
                enable_deterministic_inference=True,
            )

        validate_turboquant_transfer_compatibility(
            kv_cache_dtype="turboquant_4bit",
            disaggregation_mode="null",
            enable_hierarchical_cache=True,
            use_mla_backend=True,
            enable_deterministic_inference=True,
            prefill_attention_backend="flashmla",
            decode_attention_backend="flashmla",
            mla_fused_decode_enabled=True,
        )
        validate_turboquant_transfer_compatibility(
            kv_cache_dtype="bf16",
            disaggregation_mode="decode",
            enable_hierarchical_cache=True,
            use_mla_backend=False,
        )

    def test_mla_turboquant_requires_explicit_fused_backend(self):
        from sglang.srt.model_executor.model_runner_components.turboquant_compat import (
            validate_turboquant_transfer_compatibility,
        )

        common = dict(
            kv_cache_dtype="turboquant_4bit",
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            use_mla_backend=True,
        )
        with self.assertRaisesRegex(ValueError, "SGLANG_TQ_MLA_FUSED_DECODE=1"):
            validate_turboquant_transfer_compatibility(
                **common,
                prefill_attention_backend="flashmla",
                decode_attention_backend="flashmla",
            )
        with self.assertRaisesRegex(ValueError, "requires flashmla"):
            validate_turboquant_transfer_compatibility(
                **common,
                prefill_attention_backend="trtllm_mla",
                decode_attention_backend="flashmla",
                mla_fused_decode_enabled=True,
            )

        validate_turboquant_transfer_compatibility(
            **common,
            prefill_attention_backend="flashmla",
            decode_attention_backend="flashmla",
            mla_fused_decode_enabled=True,
        )


class TestCodebook(unittest.TestCase):

    def test_1bit_centroids(self):
        from sglang.srt.layers.quantization.kv_turboquant import build_codebook
        centroids, boundaries = build_codebook(1, head_dim=128)
        self.assertEqual(len(centroids), 2)
        self.assertAlmostEqual(centroids[0], -centroids[1], places=6)
        expected = math.sqrt(2.0 / (math.pi * 128))
        self.assertAlmostEqual(abs(centroids[1]), expected, places=5)

    def test_2bit_centroids(self):
        from sglang.srt.layers.quantization.kv_turboquant import build_codebook
        centroids, boundaries = build_codebook(2, head_dim=128)
        self.assertEqual(len(centroids), 4)
        for i in range(len(centroids) - 1):
            self.assertLess(centroids[i], centroids[i + 1])

    def test_4bit_lloyds(self):
        from sglang.srt.layers.quantization.kv_turboquant import build_codebook
        centroids, boundaries = build_codebook(4, head_dim=128)
        self.assertEqual(len(centroids), 16)


class TestNativeE2M1Contract(unittest.TestCase):

    def test_frozen_levels_map_to_exact_hardware_codes(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NATIVE_E2M1_CODES,
            NATIVE_E2M1_LEVELS,
            NativeE2M1MLAConfig,
            select_native_e2m1_codes,
        )

        cfg = NativeE2M1MLAConfig(device="cpu")
        rotated = torch.tensor(NATIVE_E2M1_LEVELS, dtype=torch.float32)
        rotated = rotated[:, None].expand(-1, cfg.head_dim) * cfg.grid
        codes, raw_levels = select_native_e2m1_codes(rotated, cfg)

        expected_codes = torch.tensor(NATIVE_E2M1_CODES, dtype=torch.uint8)
        expected_codes = expected_codes[:, None].expand_as(codes)
        expected_levels = torch.tensor(NATIVE_E2M1_LEVELS, dtype=torch.float32)
        expected_levels = expected_levels[:, None].expand_as(raw_levels)
        torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
        torch.testing.assert_close(raw_levels, expected_levels, rtol=0, atol=0)
        self.assertNotIn(8, codes.unique().tolist())

    def test_every_code_packs_in_both_nibble_positions(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NATIVE_E2M1_CODES,
            pack_native_e2m1_codes,
        )

        codes = torch.tensor(NATIVE_E2M1_CODES, dtype=torch.uint8)
        low = torch.stack((codes, torch.zeros_like(codes)), dim=-1)
        high = torch.stack((torch.zeros_like(codes), codes), dim=-1)
        torch.testing.assert_close(
            pack_native_e2m1_codes(low).squeeze(-1), codes, rtol=0, atol=0
        )
        torch.testing.assert_close(
            pack_native_e2m1_codes(high).squeeze(-1), codes << 4, rtol=0, atol=0
        )

    def test_scale_operation_order_and_zero_rule_are_frozen(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NATIVE_E2M1_LEVELS,
            NativeE2M1MLAConfig,
            quantize_native_e2m1_rotated,
        )

        cfg = NativeE2M1MLAConfig(device="cpu")
        levels = torch.tensor(NATIVE_E2M1_LEVELS, dtype=torch.float32)
        raw = levels.repeat((cfg.head_dim + len(levels) - 1) // len(levels))
        raw = raw[: cfg.head_dim]
        rotated = torch.stack((raw * cfg.grid, torch.ones_like(raw)))
        norms = torch.tensor([3.25, 0.0], dtype=torch.float32)

        packed, scales = quantize_native_e2m1_rotated(rotated, norms, cfg)
        expected_scale = (
            (norms[0] / torch.linalg.vector_norm(raw * cfg.grid)) * cfg.grid
        ).to(torch.bfloat16)
        torch.testing.assert_close(scales[0], expected_scale, rtol=0, atol=0)
        self.assertEqual(scales[1].item(), 0.0)
        self.assertEqual(torch.count_nonzero(packed[1]).item(), 0)

    def test_native_signs_match_stock_seed_42_identity(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
            TurboQuantConfig,
        )

        native = NativeE2M1MLAConfig(device="cpu")
        stock = TurboQuantConfig(bit_width=4, head_dim=512, device="cpu")
        torch.testing.assert_close(native.signs1, stock.signs1, rtol=0, atol=0)
        torch.testing.assert_close(native.signs2, stock.signs2, rtol=0, atol=0)
        self.assertEqual(
            hashlib.sha256(native.signs1.numpy().tobytes()).hexdigest(),
            "1b6c7d653224e6f3cedfa54cc12346fc1b92911743620e6108e594e5ba49bdcc",
        )
        self.assertEqual(
            hashlib.sha256(native.signs2.numpy().tobytes()).hexdigest(),
            "7df6518ccbd93f08e23a6fe647819f1109abb753cfa8d0831afa1dd8354c6a4f",
        )

    def test_native_contract_rejects_unsupported_shape_and_codes(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
            pack_native_e2m1_codes,
            select_native_e2m1_codes,
        )

        with self.assertRaisesRegex(ValueError, "head_dim=512"):
            NativeE2M1MLAConfig(device="cpu", head_dim=256)

        cfg = NativeE2M1MLAConfig(device="cpu")
        with self.assertRaisesRegex(ValueError, "final dimension"):
            select_native_e2m1_codes(torch.zeros(1, 256), cfg)
        with self.assertRaisesRegex(ValueError, "even final dimension"):
            pack_native_e2m1_codes(torch.zeros(3, dtype=torch.uint8))
        with self.assertRaisesRegex(TypeError, "must be uint8"):
            pack_native_e2m1_codes(torch.zeros(4, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "fit in four bits"):
            pack_native_e2m1_codes(torch.tensor([0, 16], dtype=torch.uint8))

    def test_native_pool_rejects_wrong_logical_dtype_shape_and_page(self):
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolNativeE2M1,
        )

        common = dict(
            size=32,
            page_size=32,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        with self.assertRaisesRegex(ValueError, "BF16 logical/model dtype"):
            MLATokenToKVPoolNativeE2M1(**(common | {"dtype": torch.float16}))
        with self.assertRaisesRegex(ValueError, "kv_lora_rank=512"):
            MLATokenToKVPoolNativeE2M1(**(common | {"kv_lora_rank": 256}))
        with self.assertRaisesRegex(ValueError, "page_size=32"):
            MLATokenToKVPoolNativeE2M1(**(common | {"page_size": 16}))

    def test_native_pool_routing_is_explicit(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        configurator = object.__new__(KVCacheConfigurator)
        configurator.kv_cache_dtype_str = "turboquant_4bit_e2m1"
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.server_args = SimpleNamespace(
            page_size=32,
            enable_memory_saver=False,
        )
        configurator.model_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        configurator.layer_info = SimpleNamespace(
            num_effective_layers=61,
            start_layer=0,
            end_layer=60,
        )
        configurator.device = "cuda"
        sentinel = object()
        with patch(
            "sglang.srt.mem_cache.kv_cache_configurator."
            "MLATokenToKVPoolNativeE2M1",
            return_value=sentinel,
        ) as constructor:
            result = configurator._build_mla_turboquant_kv_pool(
                max_total_num_tokens=12345
            )

        self.assertIs(result, sentinel)
        constructor.assert_called_once_with(
            12345,
            page_size=32,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=61,
            device="cuda",
            enable_memory_saver=False,
            start_layer=0,
            end_layer=60,
        )


class TestTurboQuantConfig(unittest.TestCase):

    def test_config_creation_cpu(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device="cpu")
        self.assertEqual(cfg.bit_width, 4)
        self.assertEqual(cfg.k_centroids.shape[0], 16)
        self.assertEqual(cfg.signs1.shape[0], 128)
        self.assertEqual(cfg.signs2.shape[0], 128)
        self.assertEqual(cfg.k_packed_dim, 64)

    def test_signs_are_pm1(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device="cpu")
        self.assertTrue((cfg.signs1.abs() == 1.0).all())
        self.assertTrue((cfg.signs2.abs() == 1.0).all())

    def test_packed_dim_2bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=2, head_dim=128, device="cpu")
        self.assertEqual(cfg.k_packed_dim, 32)

    def test_packed_dim_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device="cpu")
        self.assertEqual(cfg.k_packed_dim, 64)

    def test_mha_config_gate_skips_mla_pool(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            get_mha_turboquant_config,
        )

        cfg = object()
        mha_pool = SimpleNamespace(tq_config=cfg)
        mla_pool = SimpleNamespace(tq_config=cfg, is_mla_turboquant_pool=True)
        wrapped_mla_pool = SimpleNamespace(
            tq_config=cfg,
            full_kv_pool=SimpleNamespace(is_mla_turboquant_pool=True),
        )

        self.assertIs(get_mha_turboquant_config(mha_pool), cfg)
        self.assertIsNone(get_mha_turboquant_config(mla_pool))
        self.assertIsNone(get_mha_turboquant_config(wrapped_mla_pool))


try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestTurboQuantGPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
        cls.device = "cuda"
        cls.configs = {}
        for bits in [2, 4]:
            cls.configs[bits] = TurboQuantConfig(
                bit_width=bits, head_dim=128, device=cls.device
            )

    def test_native_e2m1_fused_writer_matches_independent_oracle(self):
        from sglang.kernels.ops.attention.turboquant_quantize import (
            fused_native_e2m1_mla_quantize_and_store,
        )
        from sglang.kernels.ops.quantization.hadamard import (
            hadamard_transform_with_signs,
        )
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        torch.manual_seed(20260814)
        cfg = NativeE2M1MLAConfig(device=self.device)
        tokens = 19
        pool_size = 80
        workspace_tokens = 7

        # Slice only the token axis so the source has non-contiguous token
        # stride while preserving the writer's contiguous-feature contract.
        x_backing = torch.randn(
            tokens * 2,
            1,
            512,
            dtype=torch.bfloat16,
            device=self.device,
        )
        rope_backing = torch.randn(
            tokens * 2,
            1,
            64,
            dtype=torch.bfloat16,
            device=self.device,
        )
        x = x_backing[::2]
        rope = rope_backing[::2]
        x[0].zero_()
        x[1].zero_()
        x[1, 0, 0] = torch.finfo(torch.bfloat16).tiny
        self.assertFalse(x.is_contiguous())
        self.assertEqual(x.stride(-1), 1)

        loc = torch.tensor(
            [
                31,
                1,
                32,
                0,
                63,
                2,
                47,
                16,
                64,
                7,
                48,
                15,
                33,
                3,
                62,
                8,
                49,
                17,
                65,
            ],
            dtype=torch.int32,
            device=self.device,
        )
        packed = torch.full(
            (pool_size, 1, 256),
            0xA5,
            dtype=torch.uint8,
            device=self.device,
        )
        scale = torch.full(
            (pool_size, 1),
            -2.0,
            dtype=torch.bfloat16,
            device=self.device,
        )
        rope_out = torch.full(
            (pool_size, 1, 64),
            3.0,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        pre_unit = torch.empty(
            workspace_tokens, 1, 512, dtype=torch.float32, device=self.device
        )
        pre_norms = torch.empty(
            workspace_tokens, 1, dtype=torch.float32, device=self.device
        )
        pre_y = torch.empty_like(pre_unit)

        fused_native_e2m1_mla_quantize_and_store(
            x,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            cfg.grid,
            packed,
            scale,
            loc,
            rope,
            rope_out,
            pre_unit=pre_unit,
            pre_norms=pre_norms,
            pre_y=pre_y,
        )

        # Independent Torch construction of the frozen N8 operation order.
        x_fp32 = x.float()
        norms = torch.linalg.vector_norm(x_fp32, dim=-1, dtype=torch.float32)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        unit = x_fp32 / safe_norms.unsqueeze(-1)
        rotated = hadamard_transform_with_signs(
            unit,
            cfg.signs1,
            cfg.signs2,
            scale=1.0 / math.sqrt(512),
        )
        indices = torch.searchsorted(cfg.boundaries, rotated / cfg.grid)
        raw = cfg.levels[indices]
        expected_codes = cfg.codes[indices]
        zero = norms <= 0
        expected_codes = torch.where(zero.unsqueeze(-1), 0, expected_codes)
        pairs = expected_codes.reshape(tokens, 1, 256, 2)
        expected_packed = pairs[..., 0] | (pairs[..., 1] << 4)
        quant_norms = torch.linalg.vector_norm(
            raw * cfg.grid, dim=-1, dtype=torch.float32
        )
        safe_quant_norms = torch.where(
            quant_norms > 0, quant_norms, torch.ones_like(quant_norms)
        )
        expected_scale = ((norms / safe_quant_norms) * cfg.grid).to(
            torch.bfloat16
        )
        expected_scale = torch.where(zero, 0, expected_scale)
        expected_rope = rope.to(torch.float8_e4m3fn)

        torch.testing.assert_close(packed[loc], expected_packed, rtol=0, atol=0)
        torch.testing.assert_close(scale[loc], expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(
            rope_out[loc].view(torch.uint8),
            expected_rope.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        self.assertEqual(torch.count_nonzero(packed[loc[0]]).item(), 0)
        self.assertEqual(scale[loc[0]].item(), 0.0)
        self.assertEqual(torch.count_nonzero(packed[loc[1]]).item(), 0)
        self.assertEqual(scale[loc[1]].item(), 0.0)

        untouched = torch.ones(pool_size, dtype=torch.bool, device=self.device)
        untouched[loc] = False
        self.assertTrue(torch.all(packed[untouched] == 0xA5).item())
        self.assertTrue(torch.all(scale[untouched] == -2.0).item())
        self.assertTrue(
            torch.all(
                rope_out[untouched] == torch.tensor(3.0, device=self.device)
            ).item()
        )

    def test_native_e2m1_fused_writer_identical_duplicate_is_stable(self):
        from sglang.kernels.ops.attention.turboquant_quantize import (
            fused_native_e2m1_mla_quantize_and_store,
        )
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        cfg = NativeE2M1MLAConfig(device=self.device)
        x = torch.randn(1, 1, 512, dtype=torch.bfloat16, device=self.device).expand(
            2, -1, -1
        )
        rope = torch.randn(
            1, 1, 64, dtype=torch.bfloat16, device=self.device
        ).expand(2, -1, -1)
        loc = torch.tensor([5, 5], dtype=torch.int64, device=self.device)
        packed = torch.zeros(8, 1, 256, dtype=torch.uint8, device=self.device)
        scale = torch.zeros(8, 1, dtype=torch.bfloat16, device=self.device)
        rope_out = torch.zeros(
            8, 1, 64, dtype=torch.float8_e4m3fn, device=self.device
        )
        pre_unit = torch.empty(2, 1, 512, dtype=torch.float32, device=self.device)
        pre_norms = torch.empty(2, 1, dtype=torch.float32, device=self.device)
        pre_y = torch.empty_like(pre_unit)

        fused_native_e2m1_mla_quantize_and_store(
            x,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            cfg.grid,
            packed,
            scale,
            loc,
            rope,
            rope_out,
            pre_unit=pre_unit,
            pre_norms=pre_norms,
            pre_y=pre_y,
        )
        first = (packed[5].clone(), scale[5].clone(), rope_out[5].clone())
        fused_native_e2m1_mla_quantize_and_store(
            x,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            cfg.grid,
            packed,
            scale,
            loc,
            rope,
            rope_out,
            pre_unit=pre_unit,
            pre_norms=pre_norms,
            pre_y=pre_y,
        )
        torch.testing.assert_close(packed[5], first[0], rtol=0, atol=0)
        torch.testing.assert_close(scale[5], first[1], rtol=0, atol=0)
        torch.testing.assert_close(
            rope_out[5].view(torch.uint8), first[2].view(torch.uint8), rtol=0, atol=0
        )

    def test_native_e2m1_mla_pool_layout_write_and_lifecycle(self):
        from sglang.kernels.ops.quantization.hadamard import (
            hadamard_transform_with_signs,
        )
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.kv_turboquant import (
            quantize_native_e2m1_rotated,
        )
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolNativeE2M1,
        )

        size = 96
        page_size = 32
        layers = 2
        workspace_tokens = 4
        with envs.SGLANG_TQ_MLA_KV_WRITE_WORKSPACE_TOKENS.override(
            workspace_tokens
        ):
            pool = MLATokenToKVPoolNativeE2M1(
                size=size,
                page_size=page_size,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=layers,
                device=self.device,
                enable_memory_saver=False,
            )

        rows = size + page_size
        self.assertEqual(pool.kv_nope_packed_buffer[0].shape, (rows, 1, 256))
        self.assertEqual(pool.kv_nope_scale_buffer[0].shape, (rows, 1))
        self.assertEqual(pool.kv_rope_buffer[0].shape, (rows, 1, 64))
        self.assertEqual(pool.kv_nope_packed_buffer[0].dtype, torch.uint8)
        self.assertEqual(pool.kv_nope_scale_buffer[0].dtype, torch.bfloat16)
        self.assertEqual(pool.kv_rope_buffer[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(pool.get_kv_size_bytes(), rows * layers * 322)
        self.assertEqual(
            pool.get_native_e2m1_workspace_size_bytes(), workspace_tokens * 4100
        )

        descriptors = pool.get_native_e2m1_buffers(0)
        self.assertEqual(descriptors.row_nbytes, 322)
        ptrs, lens, item_lens = pool.get_native_e2m1_contiguous_buf_infos()
        self.assertEqual(len(ptrs), layers * 3)
        self.assertEqual(
            lens,
            layers
            * [rows * 256, rows * 2, rows * 64],
        )
        self.assertEqual(
            item_lens,
            layers
            * [page_size * 256, page_size * 2, page_size * 64],
        )

        torch.manual_seed(17082)
        tokens = 9
        x = torch.randn(
            tokens, 1, 512, dtype=torch.bfloat16, device=self.device
        )
        rope = torch.randn(
            tokens, 1, 64, dtype=torch.bfloat16, device=self.device
        )
        x[0].zero_()
        loc = torch.tensor(
            [127, 0, 31, 32, 95, 96, 64, 63, 1],
            dtype=torch.int32,
            device=self.device,
        )
        layer = SimpleNamespace(layer_id=0)
        pool.set_mla_kv_buffer(layer, loc, x, rope)

        x_fp32 = x.float()
        norms = torch.linalg.vector_norm(x_fp32, dim=-1, dtype=torch.float32)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        unit = x_fp32 / safe_norms.unsqueeze(-1)
        rotated = hadamard_transform_with_signs(
            unit,
            pool.tq_config.signs1,
            pool.tq_config.signs2,
            scale=1.0 / math.sqrt(512),
        )
        expected_packed, expected_scale = quantize_native_e2m1_rotated(
            rotated, norms, pool.tq_config
        )
        torch.testing.assert_close(
            descriptors.packed_nope[loc], expected_packed, rtol=0, atol=0
        )
        torch.testing.assert_close(
            descriptors.nope_scale[loc], expected_scale, rtol=0, atol=0
        )
        torch.testing.assert_close(
            descriptors.rope[loc].view(torch.uint8),
            rope.to(torch.float8_e4m3fn).view(torch.uint8),
            rtol=0,
            atol=0,
        )

        src = torch.tensor([31, 32, 96], dtype=torch.int64, device=self.device)
        dst = torch.tensor([10, 11, 12], dtype=torch.int64, device=self.device)
        before_move = tuple(tensor[src].clone() for tensor in descriptors.tensors)
        pool.move_kv_cache(dst, src)
        for tensor, expected in zip(descriptors.tensors, before_move):
            torch.testing.assert_close(tensor[dst], expected, rtol=0, atol=0)

        overwrite_x = torch.randn(
            1, 1, 512, dtype=torch.bfloat16, device=self.device
        )
        overwrite_rope = torch.randn(
            1, 1, 64, dtype=torch.bfloat16, device=self.device
        )
        overwrite_loc = torch.tensor([11], dtype=torch.int64, device=self.device)
        old_row = tuple(tensor[11].clone() for tensor in descriptors.tensors)
        pool.set_mla_kv_buffer(
            layer, overwrite_loc, overwrite_x, overwrite_rope
        )
        self.assertFalse(torch.equal(descriptors.packed_nope[11], old_row[0]))
        self.assertFalse(torch.equal(descriptors.nope_scale[11], old_row[1]))
        self.assertFalse(
            torch.equal(
                descriptors.rope[11].view(torch.uint8), old_row[2].view(torch.uint8)
            )
        )

        with self.assertRaisesRegex(RuntimeError, "dense full-pool key"):
            pool.get_key_buffer(0)
        with self.assertRaisesRegex(RuntimeError, "dense full-pool value"):
            pool.get_value_buffer(0)
        with self.assertRaisesRegex(RuntimeError, "dense full-pool KV"):
            pool.get_kv_buffer(0)
        with self.assertRaisesRegex(RuntimeError, "row-gathered reconstruction"):
            pool.get_mla_kv_buffer(layer, loc)
        with self.assertRaisesRegex(RuntimeError, "PD/disaggregation"):
            pool.get_contiguous_buf_infos()
        with self.assertRaisesRegex(RuntimeError, "CPU KV offload"):
            pool.get_cpu_copy(loc)

    def test_native_e2m1_mla_pool_write_captures_and_replays(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolNativeE2M1,
        )

        with envs.SGLANG_TQ_MLA_KV_WRITE_WORKSPACE_TOKENS.override(8):
            pool = MLATokenToKVPoolNativeE2M1(
                size=32,
                page_size=32,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=1,
                device=self.device,
                enable_memory_saver=False,
            )

        layer = SimpleNamespace(layer_id=0)
        x = torch.randn(4, 1, 512, dtype=torch.bfloat16, device=self.device)
        rope = torch.randn(4, 1, 64, dtype=torch.bfloat16, device=self.device)
        loc = torch.tensor([1, 31, 32, 63], dtype=torch.int32, device=self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_mla_kv_buffer(layer, loc, x, rope)
        graph.replay()
        torch.cuda.synchronize()
        first = tuple(
            tensor[loc].clone()
            for tensor in pool.get_native_e2m1_buffers(0).tensors
        )

        x.copy_(
            torch.randn(4, 1, 512, dtype=torch.bfloat16, device=self.device)
        )
        rope.copy_(
            torch.randn(4, 1, 64, dtype=torch.bfloat16, device=self.device)
        )
        graph.replay()
        torch.cuda.synchronize()
        second = tuple(
            tensor[loc].clone()
            for tensor in pool.get_native_e2m1_buffers(0).tensors
        )
        self.assertFalse(torch.equal(first[0], second[0]))
        self.assertFalse(torch.equal(first[1], second[1]))
        self.assertFalse(
            torch.equal(first[2].view(torch.uint8), second[2].view(torch.uint8))
        )

    def _roundtrip(self, bits, tokens=64, heads=4):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[bits]
        torch.manual_seed(42)
        x = torch.randn(tokens, heads, 128, device=self.device, dtype=torch.bfloat16)

        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, cfg.bit_width
        )
        # dequantize includes inverse WHT — returns in original domain
        x_hat = batched_dequantize(
            packed, norms, cfg.k_centroids, cfg.bit_width, cfg.signs1, cfg.signs2
        )

        mse = ((x.float() - x_hat.float()) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            x.float().reshape(-1), x_hat.float().reshape(-1), dim=0
        ).item()
        return mse, cos

    def test_2bit_roundtrip(self):
        mse, cos = self._roundtrip(2)
        self.assertGreater(cos, 0.7, f"2-bit cosine too low: {cos:.4f}")

    def test_4bit_roundtrip(self):
        mse, cos = self._roundtrip(4)
        self.assertGreater(cos, 0.95, f"4-bit cosine too low: {cos:.4f}")

    def test_wht_self_inverse(self):
        """Normalized WHT applied twice should return the original."""
        from sglang.kernels.ops.quantization.hadamard import hadamard_transform
        import math
        torch.manual_seed(0)
        x = torch.randn(8, 4, 128, device=self.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(128)
        y = hadamard_transform(x, scale=scale)
        x_back = hadamard_transform(y, scale=scale)
        err = (x - x_back).abs().max().item()
        self.assertLess(err, 1e-4, f"WHT not self-inverse: max error {err}")

    def test_packing_2bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import batched_quantize
        cfg = self.configs[2]
        x = torch.randn(8, 4, 128, device=self.device, dtype=torch.bfloat16)
        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 2
        )
        self.assertEqual(packed.shape, (8, 4, 32))
        self.assertEqual(packed.dtype, torch.uint8)

    def test_packing_4bit(self):
        from sglang.srt.layers.quantization.kv_turboquant import batched_quantize
        cfg = self.configs[4]
        x = torch.randn(8, 4, 128, device=self.device, dtype=torch.bfloat16)
        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        self.assertEqual(packed.shape, (8, 4, 64))

    def test_zero_vector(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[4]
        x = torch.zeros(1, 1, 128, device=self.device, dtype=torch.bfloat16)
        packed, norms, quant_norms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        x_hat = batched_dequantize(
            packed, norms, cfg.k_centroids, 4, cfg.signs1, cfg.signs2
        )
        self.assertAlmostEqual(x_hat.abs().max().item(), 0.0, places=3)

    def test_batch_consistency(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 2, 128, device=self.device, dtype=torch.bfloat16)

        packed_b, norms_b, _ = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        hat_b = batched_dequantize(packed_b, norms_b, cfg.k_centroids, 4, cfg.signs1, cfg.signs2)

        for i in range(4):
            packed_i, norms_i, _ = batched_quantize(
                x[i:i+1], cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
            )
            hat_i = batched_dequantize(packed_i, norms_i, cfg.k_centroids, 4, cfg.signs1, cfg.signs2)
            torch.testing.assert_close(hat_b[i:i+1], hat_i, atol=1e-4, rtol=0)

    def test_attention_score_preservation(self):
        """Verify Q@K^T scores are well-preserved after quantization."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        Q = torch.randn(1, 4, 128, device=self.device, dtype=torch.bfloat16)
        K = torch.randn(32, 4, 128, device=self.device, dtype=torch.bfloat16)

        scores_orig = torch.matmul(Q.float(), K.float().transpose(-2, -1))

        k_packed, k_norms, _ = batched_quantize(
            K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        K_hat = batched_dequantize(k_packed, k_norms, cfg.k_centroids, 4, cfg.signs1, cfg.signs2)
        scores_quant = torch.matmul(Q.float(), K_hat.float().transpose(-2, -1))

        cos = torch.nn.functional.cosine_similarity(
            scores_orig.flatten(), scores_quant.flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.9, f"Attention scores diverged: cos={cos:.4f}")

    def test_query_rotation_equivalence(self):
        """Verify Q_rot @ K_rotspace produces same scores as Q @ K_dequant."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_dequantize_rotspace, batched_quantize,
        )
        for bits in [2, 4]:
            cfg = self.configs[bits]
            torch.manual_seed(42)
            Q = torch.randn(4, 8, 128, device=self.device, dtype=torch.bfloat16)
            K = torch.randn(32, 8, 128, device=self.device, dtype=torch.bfloat16)

            # Quantize K
            k_packed, k_norms, k_quant_norms = batched_quantize(
                K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )

            # Method A: Traditional dequant (with inverse WHT)
            K_hat = batched_dequantize(
                k_packed, k_norms, cfg.k_centroids, bits, cfg.signs1, cfg.signs2
            )
            scores_a = torch.einsum(
                "qhd,khd->hqk", Q.float(), K_hat.float()
            )

            # Method B: Query Rotation + rotspace dequant (no inverse WHT)
            Q_rot = cfg.rotate_query(Q)
            safe_k_qnorms = torch.where(k_quant_norms > 1e-10, k_quant_norms, torch.ones_like(k_quant_norms))
            k_dequant_scale = k_norms / safe_k_qnorms
            K_rotspace = batched_dequantize_rotspace(
                k_packed, k_dequant_scale, cfg.k_centroids, bits, head_dim=128
            )
            scores_b = torch.einsum(
                "qhd,khd->hqk", Q_rot.float(), K_rotspace.float()
            )

            max_diff = (scores_a - scores_b).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                scores_a.flatten(), scores_b.flatten(), dim=0
            ).item()
            self.assertLess(
                max_diff, 0.2,
                f"{bits}-bit: Query rotation scores diverge: max_diff={max_diff:.4f}"
            )
            self.assertGreater(
                cos, 0.99,
                f"{bits}-bit: Query rotation scores diverge: cos={cos:.6f}"
            )

    def test_output_inverse_rotation(self):
        """Verify inverse_rotate(rotate(x)) == x (bf16 precision)."""
        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 8, 128, device=self.device, dtype=torch.bfloat16)
        x_rot = cfg.rotate_query(x)
        x_back = cfg.inverse_rotate_output(x_rot)
        max_err = (x.float() - x_back.float()).abs().max().item()
        self.assertLess(max_err, 0.02, f"Rotation roundtrip error: {max_err}")

    def test_rotspace_dequant_v_output_equivalence(self):
        """Verify attention output equivalence for V side:
        D1@H@D2 @ sum(attn_i * V_rotspace_i) == sum(attn_i * V_dequant_i)"""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize, batched_dequantize_rotspace, batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        V = torch.randn(32, 4, 128, device=self.device, dtype=torch.bfloat16)
        # attn_weights: (1, 4, 32) — 1 query, 4 heads, 32 KV tokens
        attn_weights = torch.softmax(
            torch.randn(1, 4, 32, device=self.device), dim=-1
        )

        v_packed, v_norms, v_quant_norms = batched_quantize(
            V, cfg.signs1, cfg.signs2, cfg.v_centroids, cfg.v_boundaries, 4
        )

        # Method A: traditional dequant
        V_hat = batched_dequantize(
            v_packed, v_norms, cfg.v_centroids, 4, cfg.signs1, cfg.signs2
        )
        # V_hat: (32, 4, 128) → need (heads, kv_tokens, dim) for einsum
        o_a = torch.einsum("qhk,khd->qhd", attn_weights, V_hat.float())

        # Method B: rotspace + inverse rotation on output
        safe_v_qnorms = torch.where(v_quant_norms > 1e-10, v_quant_norms, torch.ones_like(v_quant_norms))
        v_dequant_scale = v_norms / safe_v_qnorms
        V_rotspace = batched_dequantize_rotspace(
            v_packed, v_dequant_scale, cfg.v_centroids, 4, head_dim=128
        )
        o_rotspace = torch.einsum("qhk,khd->qhd", attn_weights, V_rotspace.float())
        o_b = cfg.inverse_rotate_output(o_rotspace.to(torch.bfloat16)).float()

        max_diff = (o_a - o_b).abs().max().item()
        self.assertLess(
            max_diff, 0.1,
            f"V output rotspace equivalence failed: max_diff={max_diff:.4f}"
        )

    def test_fused_decode_kernel_correctness(self):
        """Verify fused TQ decode kernel matches PyTorch reference implementation."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize_rotspace, batched_quantize,
        )
        from sglang.kernels.ops.attention.turboquant_decode_attention import (
            tq_decode_attention_fwd,
        )
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd,
        )

        torch.manual_seed(42)
        batch = 2
        q_heads = 32
        kv_heads = 8
        head_dim = 128
        seq_lens = [64, 48]
        total_kv = sum(seq_lens)
        max_kv_splits = 4

        for bits in [2, 4]:  # fused kernel supports both 2-bit and 4-bit
            cfg = self.configs[bits]

            # Generate Q (already rotated) and raw K/V
            Q = torch.randn(batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            K_raw = torch.randn(total_kv, kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            V_raw = torch.randn(total_kv, kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

            # Quantize K and V
            k_packed, k_norms, k_qnorms = batched_quantize(
                K_raw, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )
            v_packed, v_norms, v_qnorms = batched_quantize(
                V_raw, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )

            # Precompute dequant scales
            safe_k_qn = torch.where(k_qnorms > 1e-10, k_qnorms, torch.ones_like(k_qnorms))
            safe_v_qn = torch.where(v_qnorms > 1e-10, v_qnorms, torch.ones_like(v_qnorms))
            k_dscale = (k_norms / safe_k_qn).to(torch.bfloat16)
            v_dscale = (v_norms / safe_v_qn).to(torch.bfloat16)

            # Build kv_indptr and kv_indices (identity mapping: slot i = position i)
            kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32, device=self.device)
            kv_indices = torch.arange(total_kv, dtype=torch.int64, device=self.device)

            # num_kv_splits
            num_kv_splits = torch.full((batch,), max_kv_splits, dtype=torch.int32, device=self.device)

            # --- Method A: Fused kernel ---
            o_fused = torch.zeros(batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            attn_logits_a = torch.zeros(batch, q_heads, max_kv_splits, head_dim, device=self.device, dtype=torch.float32)
            attn_lse_a = torch.zeros(batch, q_heads, max_kv_splits, device=self.device, dtype=torch.float32)

            tq_decode_attention_fwd(
                Q, k_packed, v_packed,
                k_dscale, v_dscale,
                cfg.k_centroids,
                o_fused, kv_indptr, kv_indices,
                attn_logits_a, attn_lse_a,
                num_kv_splits, max_kv_splits,
                sm_scale=1.0 / (head_dim ** 0.5),
                bit_width=bits,
            )

            # --- Method B: PyTorch reference (rotspace dequant + standard attention) ---
            K_dequant = batched_dequantize_rotspace(
                k_packed, k_dscale, cfg.k_centroids, bits, head_dim=128
            )
            V_dequant = batched_dequantize_rotspace(
                v_packed, v_dscale, cfg.k_centroids, bits, head_dim=128
            )
            o_ref = torch.zeros(batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16)
            attn_logits_b = torch.zeros(batch, q_heads, max_kv_splits, head_dim, device=self.device, dtype=torch.float32)
            attn_lse_b = torch.zeros(batch, q_heads, max_kv_splits, device=self.device, dtype=torch.float32)

            decode_attention_fwd(
                Q, K_dequant, V_dequant, o_ref,
                kv_indptr, kv_indices,
                attn_logits_b, attn_lse_b,
                num_kv_splits, max_kv_splits,
                sm_scale=1.0 / (head_dim ** 0.5),
                k_scale=1.0, v_scale=1.0,
            )

            # Compare
            max_diff = (o_fused.float() - o_ref.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                o_fused.float().flatten(), o_ref.float().flatten(), dim=0
            ).item()
            self.assertLess(
                max_diff, 0.1,
                f"{bits}-bit fused kernel output diverged: max_diff={max_diff:.4f}"
            )
            self.assertGreater(
                cos, 0.999,
                f"{bits}-bit fused kernel output diverged: cos={cos:.6f}"
            )

    def test_asymmetric_k4v2_roundtrip(self):
        """Verify K=4bit V=2bit asymmetric quantization roundtrip."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig, batched_quantize, batched_dequantize,
        )
        cfg = TurboQuantConfig(bit_width=4, head_dim=128, device=self.device,
                               k_bit_width=4, v_bit_width=2)
        self.assertEqual(cfg.k_packed_dim, 64)   # 4-bit: dim//2
        self.assertEqual(cfg.v_packed_dim, 32)    # 2-bit: dim//4
        self.assertEqual(cfg.k_centroids.shape[0], 16)
        self.assertEqual(cfg.v_centroids.shape[0], 4)

        torch.manual_seed(42)
        K = torch.randn(16, 4, 128, device=self.device, dtype=torch.bfloat16)
        V = torch.randn(16, 4, 128, device=self.device, dtype=torch.bfloat16)

        k_packed, k_norms, k_qnorms = batched_quantize(
            K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, cfg.k_bit_width)
        v_packed, v_norms, v_qnorms = batched_quantize(
            V, cfg.signs1, cfg.signs2, cfg.v_centroids, cfg.v_boundaries, cfg.v_bit_width)

        self.assertEqual(k_packed.shape[-1], 64)
        self.assertEqual(v_packed.shape[-1], 32)

        # Use full dequant (with inverse WHT) to compare in original domain
        K_hat = batched_dequantize(
            k_packed, k_norms, cfg.k_centroids, cfg.k_bit_width, cfg.signs1, cfg.signs2)
        V_hat = batched_dequantize(
            v_packed, v_norms, cfg.v_centroids, cfg.v_bit_width, cfg.signs1, cfg.signs2)

        k_cos = torch.nn.functional.cosine_similarity(
            K.float().reshape(-1), K_hat.float().reshape(-1), dim=0).item()
        v_cos = torch.nn.functional.cosine_similarity(
            V.float().reshape(-1), V_hat.float().reshape(-1), dim=0).item()
        self.assertGreater(k_cos, 0.9, f"K 4-bit roundtrip cos too low: {k_cos:.4f}")
        self.assertGreater(v_cos, 0.5, f"V 2-bit roundtrip cos too low: {v_cos:.4f}")

    def test_move_kv_cache_dequant_correctness(self):
        """Verify data is correct after move_kv_cache."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize_rotspace,
            batched_quantize,
        )
        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 8, 128, device=self.device, dtype=torch.bfloat16)

        packed, norms, qnorms = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, cfg.k_bit_width)

        # Simulate pool: write to positions 0-3, move to 10-13
        pool_packed = torch.zeros(20, 8, 64, dtype=torch.uint8, device=self.device)
        safe_qnorms = torch.where(qnorms > 1e-10, qnorms, torch.ones_like(qnorms))
        dequant_scale = (norms / safe_qnorms).to(torch.bfloat16)
        pool_dscale = torch.zeros(20, 8, dtype=torch.bfloat16, device=self.device)

        src = torch.arange(4, device=self.device)
        tgt = torch.arange(10, 14, device=self.device)

        pool_packed[src] = packed
        pool_dscale[src] = dequant_scale

        # Move
        pool_packed[tgt] = pool_packed[src]
        pool_dscale[tgt] = pool_dscale[src]

        # Dequant from moved positions
        hat_src = batched_dequantize_rotspace(
            pool_packed[src], pool_dscale[src], cfg.k_centroids, 4, head_dim=128)
        hat_tgt = batched_dequantize_rotspace(
            pool_packed[tgt], pool_dscale[tgt], cfg.k_centroids, 4, head_dim=128)

        torch.testing.assert_close(hat_src, hat_tgt, atol=1e-6, rtol=0)

    def test_mla_fused_kv_write_matches_legacy_quantize_store(self):
        """Verify the Phase 35A MLA NoPE fused store matches the legacy path."""
        from sglang.kernels.ops.attention.turboquant_quantize import (
            fused_turboquant_quantize_and_store,
        )
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig,
            batched_dequantize_rotspace,
            batched_quantize,
        )

        torch.manual_seed(42)
        tokens = 7
        pool_size = 32
        lora_rank = 512
        rope_dim = 64
        cfg = TurboQuantConfig(
            bit_width=4,
            head_dim=lora_rank,
            device=self.device,
            k_bit_width=4,
            v_bit_width=4,
        )

        loc = torch.tensor([1, 5, 9, 10, 17, 20, 29], device=self.device)
        cache_k = torch.randn(
            tokens,
            1,
            lora_rank + rope_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        cache_k_nope = cache_k[..., :lora_rank]
        cache_k_rope = cache_k[..., lora_rank:]

        packed, norms, quant_norms = batched_quantize(
            cache_k_nope.contiguous(),
            cfg.signs1,
            cfg.signs2,
            cfg.k_centroids,
            cfg.k_boundaries,
            4,
        )
        safe_qnorms = torch.where(
            quant_norms > 1e-10, quant_norms, torch.ones_like(quant_norms)
        )
        dequant_scale = (norms / safe_qnorms).to(torch.bfloat16)

        legacy_packed = torch.zeros(
            pool_size, 1, lora_rank // 2, dtype=torch.uint8, device=self.device
        )
        legacy_scale = torch.zeros(
            pool_size, 1, dtype=torch.bfloat16, device=self.device
        )
        legacy_rope = torch.zeros(
            pool_size, 1, rope_dim, dtype=torch.bfloat16, device=self.device
        )
        legacy_packed[loc] = packed
        legacy_scale[loc] = dequant_scale
        legacy_rope[loc] = cache_k_rope.contiguous()

        fused_packed = torch.zeros_like(legacy_packed)
        fused_scale = torch.zeros_like(legacy_scale)
        fused_rope = torch.zeros_like(legacy_rope)
        pre_unit = torch.empty(
            tokens, 1, lora_rank, dtype=torch.float32, device=self.device
        )
        pre_norms = torch.empty(tokens, 1, dtype=torch.float32, device=self.device)
        pre_y = torch.empty(
            tokens, 1, lora_rank, dtype=torch.float32, device=self.device
        )
        fused_turboquant_quantize_and_store(
            cache_k_nope,
            cfg.signs1,
            cfg.signs2,
            cfg.k_centroids,
            cfg.k_boundaries,
            4,
            fused_packed,
            fused_scale,
            loc,
            pre_unit=pre_unit,
            pre_norms=pre_norms,
            pre_y=pre_y,
            rope_src=cache_k_rope.contiguous(),
            rope_buffer=fused_rope,
        )

        torch.testing.assert_close(fused_packed[loc], legacy_packed[loc])
        torch.testing.assert_close(fused_rope[loc], legacy_rope[loc])
        torch.testing.assert_close(
            fused_scale[loc].float(),
            legacy_scale[loc].float(),
            atol=1e-2,
            rtol=1e-2,
        )

        legacy_rot = batched_dequantize_rotspace(
            legacy_packed[loc],
            legacy_scale[loc],
            cfg.k_centroids,
            4,
            head_dim=lora_rank,
        )
        fused_rot = batched_dequantize_rotspace(
            fused_packed[loc],
            fused_scale[loc],
            cfg.k_centroids,
            4,
            head_dim=lora_rank,
        )
        torch.testing.assert_close(
            fused_rot.float(), legacy_rot.float(), atol=2e-2, rtol=2e-2
        )

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolTurboQuant

        def make_pool():
            return MLATokenToKVPoolTurboQuant(
                size=pool_size,
                page_size=0,
                dtype=torch.bfloat16,
                kv_lora_rank=lora_rank,
                qk_rope_head_dim=rope_dim,
                layer_num=1,
                device=self.device,
                enable_memory_saver=False,
                turboquant_bits=4,
                turboquant_k_bits=4,
                turboquant_v_bits=4,
                start_layer=0,
                end_layer=0,
            )

        legacy_pool = make_pool()
        fused_pool = make_pool()
        fused_rope_pool = make_pool()
        layer = SimpleNamespace(layer_id=0)

        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(False):
            legacy_pool.set_kv_buffer(layer, loc, cache_k, cache_k)
        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(True):
            fused_pool.set_kv_buffer(layer, loc, cache_k, cache_k)
        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(True):
            with envs.SGLANG_TQ_MLA_FUSED_ROPE_WRITE.override(True):
                fused_rope_pool.set_kv_buffer(layer, loc, cache_k, cache_k)

        torch.testing.assert_close(
            fused_pool.kv_nope_packed_buffer[0][loc],
            legacy_pool.kv_nope_packed_buffer[0][loc],
        )
        torch.testing.assert_close(
            fused_rope_pool.kv_nope_packed_buffer[0][loc],
            legacy_pool.kv_nope_packed_buffer[0][loc],
        )
        torch.testing.assert_close(
            fused_pool.kv_rope_buffer[0][loc],
            legacy_pool.kv_rope_buffer[0][loc],
        )
        torch.testing.assert_close(
            fused_rope_pool.kv_rope_buffer[0][loc],
            legacy_pool.kv_rope_buffer[0][loc],
        )
        torch.testing.assert_close(
            fused_pool.kv_nope_scale_buffer[0][loc].float(),
            legacy_pool.kv_nope_scale_buffer[0][loc].float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            fused_rope_pool.kv_nope_scale_buffer[0][loc].float(),
            legacy_pool.kv_nope_scale_buffer[0][loc].float(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_non_128_head_dim(self):
        """Verify TurboQuant works with head_dim=64 and head_dim=256."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig, batched_quantize, batched_dequantize, batched_dequantize_rotspace,
        )
        for dim in [64, 256]:
            for bits in [2, 4]:
                cfg = TurboQuantConfig(bit_width=bits, head_dim=dim, device=self.device)
                torch.manual_seed(42)
                x = torch.randn(8, 4, dim, device=self.device, dtype=torch.bfloat16)

                packed, norms, qnorms = batched_quantize(
                    x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits)

                # Full dequant roundtrip
                x_hat = batched_dequantize(
                    packed, norms, cfg.k_centroids, bits, cfg.signs1, cfg.signs2)
                self.assertEqual(x_hat.shape, x.shape,
                    f"dim={dim} bits={bits}: shape mismatch {x_hat.shape} vs {x.shape}")

                cos = torch.nn.functional.cosine_similarity(
                    x.float().reshape(-1), x_hat.float().reshape(-1), dim=0).item()
                min_cos = 0.5 if bits == 2 else 0.93
                self.assertGreater(cos, min_cos,
                    f"dim={dim} bits={bits}: cos={cos:.4f} < {min_cos}")

                # Rotspace dequant
                safe_qn = torch.where(qnorms > 1e-10, qnorms, torch.ones_like(qnorms))
                ds = (norms / safe_qn).to(torch.bfloat16)
                rs = batched_dequantize_rotspace(
                    packed, ds, cfg.k_centroids, bits, head_dim=dim)
                self.assertEqual(rs.shape[-1], dim)


if __name__ == "__main__":
    unittest.main()
