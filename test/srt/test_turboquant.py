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
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class TestTurboQuantCLI(unittest.TestCase):

    def test_preserved_kv_cache_dtype_choices_parse(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        for dtype in (
            "turboquant_2bit",
            "turboquant_4bit",
            "turboquant_4bit_uniform",
            "turboquant_4bit_e2m1",
            "turboquant_k4v2",
        ):
            args = parser.parse_args(
                ["--model-path", "test-model", "--kv-cache-dtype", dtype]
            )
            self.assertEqual(args.kv_cache_dtype, dtype)

    def test_dflash_fa4_draft_does_not_inherit_target_turboquant(self):
        import torch

        from sglang.srt.model_executor.model_runner import ModelRunner

        server_args = SimpleNamespace(
            kv_cache_dtype="turboquant_4bit",
            speculative_draft_attention_backend="fa4",
            decode_attention_backend=None,
        )
        runner = SimpleNamespace(
            server_args=server_args,
            is_draft_worker=True,
            spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
            dtype=torch.bfloat16,
        )

        ModelRunner.configure_kv_cache_dtype(runner)

        self.assertIs(runner.kv_cache_dtype, torch.bfloat16)
        self.assertFalse(hasattr(runner, "turboquant_bits"))
        self.assertIsNone(server_args.decode_attention_backend)

    def test_hotcold_mla_does_not_globally_rotate_absorb_weights(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        tq_config = SimpleNamespace(mla_absorb_rotation_fused=False)
        pool = SimpleNamespace(
            tq_config=tq_config,
            is_mla_turboquant_hotcold_pool=True,
        )
        fuse = Mock()
        runner = SimpleNamespace(
            turboquant_bits=4,
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: pool
            ),
            use_mla_backend=True,
            server_args=SimpleNamespace(
                get_attention_backends=lambda: (None, "tokenspeed_mla"),
                enable_lora=False,
            ),
            _maybe_fuse_tq_mla_absorb_rotations=fuse,
        )

        ModelRunner._maybe_fuse_tq_output_rotation(runner)

        fuse.assert_not_called()
        self.assertFalse(tq_config.mla_absorb_rotation_fused)

    def test_native_e2m1_cli_resolves_mla_format_contract(self):
        import torch

        from sglang.srt.model_executor.model_runner import ModelRunner

        server_args = SimpleNamespace(
            kv_cache_dtype="turboquant_4bit_e2m1",
            decode_attention_backend="tokenspeed_mla",
            disable_cuda_graph=False,
        )
        runner = SimpleNamespace(
            server_args=server_args,
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_dflash=lambda: False),
            dtype=torch.bfloat16,
            use_mla_backend=True,
        )

        ModelRunner.configure_kv_cache_dtype(runner)

        self.assertTrue(runner.turboquant_e2m1)
        self.assertFalse(runner.turboquant_uniform)
        self.assertEqual(runner.turboquant_bits, 4)
        self.assertEqual(runner.turboquant_k_bits, 4)
        self.assertEqual(runner.turboquant_v_bits, 4)
        self.assertIs(runner.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(server_args.decode_attention_backend, "tokenspeed_mla")

    def test_native_e2m1_cli_rejects_mha(self):
        import torch

        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = SimpleNamespace(
            server_args=SimpleNamespace(
                kv_cache_dtype="turboquant_4bit_e2m1",
                decode_attention_backend=None,
                disable_cuda_graph=False,
            ),
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_dflash=lambda: False),
            dtype=torch.bfloat16,
            use_mla_backend=False,
        )

        with self.assertRaisesRegex(NotImplementedError, "requires an MLA model"):
            ModelRunner.configure_kv_cache_dtype(runner)

    @patch("sglang.srt.arg_groups.overrides.is_blackwell_supported", return_value=True)
    def test_native_e2m1_is_allowed_by_tokenspeed_backend(self, _):
        from sglang.srt.arg_groups.overrides import _mla_kv_cache_dtype_checks

        view = SimpleNamespace(
            attention_backend="tokenspeed_mla",
            decode_attention_backend="tokenspeed_mla",
            kv_cache_dtype="turboquant_4bit_e2m1",
        )
        self.assertEqual(_mla_kv_cache_dtype_checks(view), {})


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

    def test_e2m1_codebook_matches_hardware_encoding(self):
        from sglang.srt.layers.quantization.kv_turboquant import (
            E2M1_GAUSSIAN_SCALE,
            build_e2m1_codebook,
        )

        quant, boundaries, decode, code_lut, scale = build_e2m1_codebook(512)

        np.testing.assert_array_equal(
            code_lut,
            np.array([15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7]),
        )
        self.assertTrue(np.signbit(decode[8]))
        np.testing.assert_array_equal(
            decode,
            np.array(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(quant, decode[code_lut] * scale, rtol=0, atol=0)
        np.testing.assert_allclose(
            boundaries, (quant[:-1] + quant[1:]) / 2, rtol=0, atol=0
        )
        self.assertAlmostEqual(scale, E2M1_GAUSSIAN_SCALE / math.sqrt(512))


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

    def test_native_e2m1_config_separates_quant_and_decode_order(self):
        import torch

        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

        cfg = TurboQuantConfig(bit_width=4, head_dim=512, device="cpu", e2m1=True)

        self.assertTrue(cfg.e2m1)
        self.assertTrue(
            torch.all(cfg.k_quant_centroids[1:] >= cfg.k_quant_centroids[:-1])
        )
        self.assertEqual(cfg.k_centroids[7].item(), 6.0)
        self.assertEqual(cfg.k_centroids[15].item(), -6.0)
        torch.testing.assert_close(
            cfg.k_storage_code_lut,
            torch.tensor(
                [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7],
                dtype=torch.uint8,
            ),
        )
        self.assertAlmostEqual(
            cfg.k_dequant_scale_multiplier,
            0.48707925311412725 / math.sqrt(512),
        )

    def test_native_e2m1_rejects_incompatible_modes(self):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TurboQuantConfig(
                bit_width=4,
                head_dim=512,
                device="cpu",
                uniform=True,
                e2m1=True,
            )
        with self.assertRaisesRegex(ValueError, "symmetric 4-bit"):
            TurboQuantConfig(
                bit_width=4,
                head_dim=512,
                device="cpu",
                k_bit_width=4,
                v_bit_width=2,
                e2m1=True,
            )

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


class TestTurboQuantMLAGraphMetadata(unittest.TestCase):

    def test_hotcold_static_ownership_requires_single_uncached_request(self):
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            _validate_tq_hotcold_server_args,
        )

        valid = SimpleNamespace(
            kv_cache_dtype="turboquant_4bit_e2m1",
            get_attention_backends=lambda: ("tokenspeed_mla", "tokenspeed_mla"),
            page_size=32,
            disable_radix_cache=True,
            max_running_requests=1,
            dcp_size=1,
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            attn_cp_size=1,
            enable_dp_attention=False,
        )
        invalid = SimpleNamespace(
            kv_cache_dtype="turboquant_4bit_e2m1",
            get_attention_backends=lambda: ("tokenspeed_mla", "flashmla"),
            page_size=64,
            disable_radix_cache=False,
            max_running_requests=8,
            dcp_size=2,
            enable_hierarchical_cache=True,
            disaggregation_mode="decode",
            attn_cp_size=2,
            enable_dp_attention=True,
        )
        with envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_384):
            _validate_tq_hotcold_server_args(
                valid,
                use_mla_backend=True,
                turboquant_k_bits=4,
                turboquant_v_bits=4,
            )
            with self.assertRaisesRegex(
                ValueError,
                "tokenspeed_mla.*--page-size 32.*--disable-radix-cache",
            ):
                _validate_tq_hotcold_server_args(invalid)

    def test_hotcold_gate_rejects_non_tq_non_mla_target_at_startup(self):
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            _validate_tq_hotcold_server_args,
        )

        invalid = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            get_attention_backends=lambda: ("flashinfer", "flashmla"),
            page_size=64,
            disable_radix_cache=True,
            max_running_requests=1,
            dcp_size=1,
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            attn_cp_size=1,
            enable_dp_attention=False,
        )
        with envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_384):
            with self.assertRaisesRegex(
                ValueError, "turboquant_4bit.*an MLA target model.*tokenspeed_mla"
            ):
                _validate_tq_hotcold_server_args(
                    invalid, use_mla_backend=False
                )

        # The default-off path must remain a true no-op and must not inspect
        # unrelated ServerArgs fields.
        with envs.SGLANG_TQ_MLA_HOT_TOKENS.override(0):
            _validate_tq_hotcold_server_args(SimpleNamespace())

    def test_hotcold_gate_rejects_conflicting_shadow_and_staging_modes(self):
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            _validate_tq_hotcold_server_args,
        )

        valid = SimpleNamespace(
            kv_cache_dtype="turboquant_4bit",
            get_attention_backends=lambda: ("tokenspeed_mla", "tokenspeed_mla"),
            page_size=32,
            disable_radix_cache=True,
            max_running_requests=1,
            dcp_size=1,
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            attn_cp_size=1,
            enable_dp_attention=False,
        )
        with (
            envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_384),
            envs.SGLANG_TQ_MLA_STAGED_FLASHMLA.override(True),
        ):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_TQ_MLA_STAGED_FLASHMLA disabled"
            ):
                _validate_tq_hotcold_server_args(valid)

        with (
            envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_384),
            patch.dict(
                os.environ,
                {"SGLANG_TQ_MLA_FP8_ACTIVE_SHADOW": "1"},
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_TQ_MLA_FP8_ACTIVE_SHADOW disabled"
            ):
                _validate_tq_hotcold_server_args(valid)

        with (
            envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_384),
            patch.dict(
                os.environ,
                {"SGLANG_TQ_MLA_FP8_ACTIVE_SHADOW": "0"},
            ),
        ):
            _validate_tq_hotcold_server_args(valid)

    def test_hotcold_gate_rejects_unsafe_graph_spec_and_tile_modes(self):
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            _validate_tq_hotcold_server_args,
        )

        base = dict(
            kv_cache_dtype="turboquant_4bit_e2m1",
            get_attention_backends=lambda: ("tokenspeed_mla", "tokenspeed_mla"),
            page_size=32,
            disable_radix_cache=True,
            max_running_requests=1,
            dcp_size=1,
            enable_hierarchical_cache=False,
            disaggregation_mode="null",
            attn_cp_size=1,
            enable_dp_attention=False,
            speculative_algorithm=None,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
        )
        with envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_384):
            _validate_tq_hotcold_server_args(SimpleNamespace(**base))
            _validate_tq_hotcold_server_args(
                SimpleNamespace(**(base | {"speculative_algorithm": "DFLASH"}))
            )
            with self.assertRaisesRegex(
                ValueError, "speculative decoding disabled or DFLASH"
            ):
                _validate_tq_hotcold_server_args(
                    SimpleNamespace(
                        **(base | {"speculative_algorithm": "EAGLE3"})
                    )
                )
            with self.assertRaisesRegex(ValueError, "prefill CUDA graphs disabled"):
                _validate_tq_hotcold_server_args(
                    SimpleNamespace(
                        **(
                            base
                            | {
                                "cuda_graph_config": SimpleNamespace(
                                    prefill=SimpleNamespace(backend="breakable")
                                )
                            }
                        )
                    )
                )
        with envs.SGLANG_TQ_MLA_HOT_TOKENS.override(16_352):
            with self.assertRaisesRegex(ValueError, "aligned to 128 tokens"):
                _validate_tq_hotcold_server_args(SimpleNamespace(**base))

    def test_hotcold_pool_rejects_page_aligned_partial_reader_tile(self):
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolTurboQuantHotCold,
        )

        with self.assertRaisesRegex(ValueError, "128-token native-reader tile"):
            MLATokenToKVPoolTurboQuantHotCold(
                256,
                page_size=32,
                hot_capacity_tokens=160,
            )

    def test_hotcold_pool_sizing_uses_fixed_hot_tier_premium(self):
        import torch

        from sglang.srt.environ import envs
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
        )

        num_layers = 2
        hot_tokens = 128
        configured_tokens = 256
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=SimpleNamespace(),
            ),
            kv_cache_dtype=torch.bfloat16,
            use_mla_backend=True,
            turboquant_bits=4,
            turboquant_k_bits=4,
            turboquant_e2m1=False,
            server_args=SimpleNamespace(
                max_total_tokens=configured_tokens,
                get_attention_backends=lambda: (
                    "tokenspeed_mla",
                    "tokenspeed_mla",
                ),
            ),
        )
        configurator = object.__new__(DefaultPoolConfigurator)
        configurator._fixed_size = 0
        with (
            envs.SGLANG_TQ_MLA_HOT_TOKENS.override(hot_tokens),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
                return_value=8,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.is_deepseek_dsa",
                return_value=False,
            ),
        ):
            configurator._cell_size = configurator._compute_cell_size(
                runner, num_layers
            )

        cold_bytes = 402
        fp8_bytes = 576
        self.assertEqual(configurator._cell_size, cold_bytes * num_layers)
        self.assertEqual(
            configurator._fixed_size,
            hot_tokens * (fp8_bytes - cold_bytes) * num_layers,
        )
        available_bytes = configurator._fixed_size + (
            configured_tokens * configurator._cell_size
        )
        pool_config = configurator.calculate_pool_sizes(
            available_bytes, page_size=32
        )
        self.assertEqual(
            pool_config.max_total_num_tokens,
            configured_tokens,
        )

    def test_hotcold_spec_crossing_rejected_before_parent_metadata(self):
        import torch

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend._tq4_hotcold_cache = True
        backend._tq_pool = SimpleNamespace(hot_capacity_tokens=128)
        backend.num_draft_tokens = 5
        mode = SimpleNamespace(
            is_target_verify=lambda: True,
            is_draft_extend_v2=lambda: False,
        )
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=mode,
            seq_lens_cpu=torch.tensor([126], dtype=torch.int64),
        )
        with patch.object(
            TRTLLMMLABackend, "init_forward_metadata"
        ) as parent_init:
            with self.assertRaisesRegex(RuntimeError, "not quality-safe"):
                backend.init_forward_metadata(batch)
        parent_init.assert_not_called()

    def test_hotcold_fp8_frontend_is_used_only_inside_hot_capacity(self):
        import torch

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend._tq4_hotcold_cache = True
        backend._tq_pool = SimpleNamespace(hot_capacity_tokens=128)
        backend.num_draft_tokens = 5
        mode = SimpleNamespace(
            is_decode_or_idle=lambda: True,
            is_target_verify=lambda: False,
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens_cpu=torch.tensor([128], dtype=torch.int64),
        )
        self.assertTrue(backend.should_use_hot_fp8_frontend(batch))
        batch.seq_lens_cpu[0] = 129
        self.assertFalse(backend.should_use_hot_fp8_frontend(batch))

    def test_tokenspeed_hotcold_graph_is_bounded_by_fp8_capacity(self):
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend._tq4_cache = True
        backend._tq4_hotcold_cache = True
        backend.max_context_len = 256_000
        backend.num_draft_tokens = 5
        backend._tq_pool = SimpleNamespace(
            hot_capacity_tokens=16_384,
            kv_nope_codebook_buffer=[object()],
        )
        self.assertEqual(backend.get_cuda_graph_max_seq_len(), 16_384)
        target_verify = SimpleNamespace(
            is_target_verify=lambda: True,
            is_draft_extend_v2=lambda: False,
        )
        self.assertEqual(
            backend.get_cuda_graph_max_prefix_len(target_verify), 16_379
        )
        backend._tq_pool.hot_capacity_tokens = 65_536
        self.assertEqual(backend.get_cuda_graph_max_seq_len(), 65_536)

    def test_tokenspeed_hotcold_prefill_reuses_incumbent_fp8_kernel(self):
        import torch

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend._tq4_cache = True
        backend._tq4_hotcold_cache = True
        q = torch.empty((1, 1, 576), dtype=torch.float8_e4m3fn)
        k = torch.empty_like(q)
        v = torch.empty((1, 1, 512), dtype=torch.float8_e4m3fn)
        seq_lens = torch.tensor([1], dtype=torch.int32)
        cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        output = object()
        layer = SimpleNamespace(scaling=1.0)

        with (
            patch(
                "sglang.srt.layers.attention.tokenspeed_mla_backend."
                "is_arch_support_pdl",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.tokenspeed_mla_backend."
                "tokenspeed_mla.tokenspeed_mla_prefill",
                return_value=output,
            ) as native_prefill,
            patch.object(
                TRTLLMMLABackend,
                "_run_prefill_kernel",
                side_effect=AssertionError(
                    "hot/cold prefill must not use the carrier-dtype fallback"
                ),
            ),
        ):
            actual = backend._run_prefill_kernel(
                q,
                k,
                v,
                layer,
                batch_size=1,
                cum_seq_lens_q=cu_seq_lens,
                max_q_len=1,
                seq_lens_kv=seq_lens,
                cum_seq_lens_kv=cu_seq_lens,
                max_kv_len=1,
                is_causal=True,
                return_lse=False,
                out_buffer=torch.empty_like(v),
            )

        self.assertIs(actual, output)
        native_prefill.assert_called_once()
        self.assertIs(native_prefill.call_args.kwargs["query"], q)
        self.assertIs(native_prefill.call_args.kwargs["key"], k)
        self.assertIs(native_prefill.call_args.kwargs["value"], v)

    def test_out_graph_refreshes_packed_indices_for_decode(self):
        import torch

        from sglang.srt.layers.attention.flashmla_backend import (
            FlashMLABackend,
            TurboQuantMLABackend,
        )

        backend = object.__new__(TurboQuantMLABackend)
        backend._tq_use_staged_flashmla = False
        backend._tq_build_kv_indices = Mock()

        req_pool_indices = torch.tensor([3, 7], dtype=torch.int32)
        seq_lens = torch.tensor([64, 96], dtype=torch.int32)
        forward_batch = SimpleNamespace(
            batch_size=2,
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        with patch.object(
            FlashMLABackend, "init_forward_metadata_out_graph"
        ) as parent_init:
            backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

        parent_init.assert_called_once_with(forward_batch, in_capture=True)
        backend._tq_build_kv_indices.assert_called_once()
        args = backend._tq_build_kv_indices.call_args.args
        self.assertEqual(args[0], 2)
        self.assertIs(args[1], req_pool_indices)
        torch.testing.assert_close(args[2], seq_lens)


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

    def _roundtrip(self, bits, tokens=64, heads=4):
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize,
            batched_quantize,
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
        import math

        from sglang.jit_kernel.hadamard import hadamard_transform

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
            batched_dequantize,
            batched_quantize,
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
            batched_dequantize,
            batched_quantize,
        )

        cfg = self.configs[4]
        torch.manual_seed(42)
        x = torch.randn(4, 2, 128, device=self.device, dtype=torch.bfloat16)

        packed_b, norms_b, _ = batched_quantize(
            x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        hat_b = batched_dequantize(
            packed_b, norms_b, cfg.k_centroids, 4, cfg.signs1, cfg.signs2
        )

        for i in range(4):
            packed_i, norms_i, _ = batched_quantize(
                x[i : i + 1],
                cfg.signs1,
                cfg.signs2,
                cfg.k_centroids,
                cfg.k_boundaries,
                4,
            )
            hat_i = batched_dequantize(
                packed_i, norms_i, cfg.k_centroids, 4, cfg.signs1, cfg.signs2
            )
            torch.testing.assert_close(hat_b[i : i + 1], hat_i, atol=1e-4, rtol=0)

    def test_attention_score_preservation(self):
        """Verify Q@K^T scores are well-preserved after quantization."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize,
            batched_quantize,
        )

        cfg = self.configs[4]
        torch.manual_seed(42)
        Q = torch.randn(1, 4, 128, device=self.device, dtype=torch.bfloat16)
        K = torch.randn(32, 4, 128, device=self.device, dtype=torch.bfloat16)

        scores_orig = torch.matmul(Q.float(), K.float().transpose(-2, -1))

        k_packed, k_norms, _ = batched_quantize(
            K, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, 4
        )
        K_hat = batched_dequantize(
            k_packed, k_norms, cfg.k_centroids, 4, cfg.signs1, cfg.signs2
        )
        scores_quant = torch.matmul(Q.float(), K_hat.float().transpose(-2, -1))

        cos = torch.nn.functional.cosine_similarity(
            scores_orig.flatten(), scores_quant.flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.9, f"Attention scores diverged: cos={cos:.4f}")

    def test_query_rotation_equivalence(self):
        """Verify Q_rot @ K_rotspace produces same scores as Q @ K_dequant."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize,
            batched_dequantize_rotspace,
            batched_quantize,
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
            scores_a = torch.einsum("qhd,khd->hqk", Q.float(), K_hat.float())

            # Method B: Query Rotation + rotspace dequant (no inverse WHT)
            Q_rot = cfg.rotate_query(Q)
            safe_k_qnorms = torch.where(
                k_quant_norms > 1e-10, k_quant_norms, torch.ones_like(k_quant_norms)
            )
            k_dequant_scale = k_norms / safe_k_qnorms
            K_rotspace = batched_dequantize_rotspace(
                k_packed, k_dequant_scale, cfg.k_centroids, bits, head_dim=128
            )
            scores_b = torch.einsum("qhd,khd->hqk", Q_rot.float(), K_rotspace.float())

            max_diff = (scores_a - scores_b).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                scores_a.flatten(), scores_b.flatten(), dim=0
            ).item()
            self.assertLess(
                max_diff,
                0.2,
                f"{bits}-bit: Query rotation scores diverge: max_diff={max_diff:.4f}",
            )
            self.assertGreater(
                cos, 0.99, f"{bits}-bit: Query rotation scores diverge: cos={cos:.6f}"
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

    def test_mla_absorb_weight_rotation_equivalence(self):
        cfg = self.configs[4]
        torch.manual_seed(42)
        heads, batch, qk_dim, value_dim = 3, 5, 64, 80
        w_kc = torch.randn(heads, qk_dim, 128, device=self.device, dtype=torch.float32)
        w_vc = torch.randn(
            heads, 128, value_dim, device=self.device, dtype=torch.float32
        )
        q = torch.randn(heads, batch, qk_dim, device=self.device, dtype=torch.float32)
        o = torch.randn(heads, batch, 128, device=self.device, dtype=torch.float32)

        w_kc_rotated, w_vc_rotated = cfg.fuse_mla_absorb_rotations(w_kc, w_vc)

        torch.testing.assert_close(
            torch.bmm(q, w_kc_rotated),
            cfg.rotate_query(torch.bmm(q, w_kc)),
            atol=2e-4,
            rtol=2e-4,
        )
        torch.testing.assert_close(
            torch.bmm(cfg.rotate_query(o), w_vc_rotated),
            torch.bmm(o, w_vc),
            atol=2e-4,
            rtol=2e-4,
        )

    def test_rotspace_dequant_v_output_equivalence(self):
        """Verify attention output equivalence for V side:
        D1@H@D2 @ sum(attn_i * V_rotspace_i) == sum(attn_i * V_dequant_i)"""
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize,
            batched_dequantize_rotspace,
            batched_quantize,
        )

        cfg = self.configs[4]
        torch.manual_seed(42)
        V = torch.randn(32, 4, 128, device=self.device, dtype=torch.bfloat16)
        # attn_weights: (1, 4, 32) — 1 query, 4 heads, 32 KV tokens
        attn_weights = torch.softmax(torch.randn(1, 4, 32, device=self.device), dim=-1)

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
        safe_v_qnorms = torch.where(
            v_quant_norms > 1e-10, v_quant_norms, torch.ones_like(v_quant_norms)
        )
        v_dequant_scale = v_norms / safe_v_qnorms
        V_rotspace = batched_dequantize_rotspace(
            v_packed, v_dequant_scale, cfg.v_centroids, 4, head_dim=128
        )
        o_rotspace = torch.einsum("qhk,khd->qhd", attn_weights, V_rotspace.float())
        o_b = cfg.inverse_rotate_output(o_rotspace.to(torch.bfloat16)).float()

        max_diff = (o_a - o_b).abs().max().item()
        self.assertLess(
            max_diff,
            0.1,
            f"V output rotspace equivalence failed: max_diff={max_diff:.4f}",
        )

    def test_fused_decode_kernel_correctness(self):
        """Verify fused TQ decode kernel matches PyTorch reference implementation."""
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
            tq_decode_attention_fwd,
        )
        from sglang.srt.layers.quantization.kv_turboquant import (
            batched_dequantize_rotspace,
            batched_quantize,
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
            Q = torch.randn(
                batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16
            )
            K_raw = torch.randn(
                total_kv, kv_heads, head_dim, device=self.device, dtype=torch.bfloat16
            )
            V_raw = torch.randn(
                total_kv, kv_heads, head_dim, device=self.device, dtype=torch.bfloat16
            )

            # Quantize K and V
            k_packed, k_norms, k_qnorms = batched_quantize(
                K_raw, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )
            v_packed, v_norms, v_qnorms = batched_quantize(
                V_raw, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
            )

            # Precompute dequant scales
            safe_k_qn = torch.where(
                k_qnorms > 1e-10, k_qnorms, torch.ones_like(k_qnorms)
            )
            safe_v_qn = torch.where(
                v_qnorms > 1e-10, v_qnorms, torch.ones_like(v_qnorms)
            )
            k_dscale = (k_norms / safe_k_qn).to(torch.bfloat16)
            v_dscale = (v_norms / safe_v_qn).to(torch.bfloat16)

            # Build kv_indptr and kv_indices (identity mapping: slot i = position i)
            kv_indptr = torch.tensor(
                [0, seq_lens[0], total_kv], dtype=torch.int32, device=self.device
            )
            kv_indices = torch.arange(total_kv, dtype=torch.int64, device=self.device)

            # num_kv_splits
            num_kv_splits = torch.full(
                (batch,), max_kv_splits, dtype=torch.int32, device=self.device
            )

            # --- Method A: Fused kernel ---
            o_fused = torch.zeros(
                batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16
            )
            attn_logits_a = torch.zeros(
                batch,
                q_heads,
                max_kv_splits,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            attn_lse_a = torch.zeros(
                batch, q_heads, max_kv_splits, device=self.device, dtype=torch.float32
            )

            tq_decode_attention_fwd(
                Q,
                k_packed,
                v_packed,
                k_dscale,
                v_dscale,
                cfg.k_centroids,
                cfg.k_centroids,
                o_fused,
                kv_indptr,
                kv_indices,
                attn_logits_a,
                attn_lse_a,
                num_kv_splits,
                max_kv_splits,
                sm_scale=1.0 / (head_dim**0.5),
                k_bit_width=bits,
                v_bit_width=bits,
            )

            # --- Method B: PyTorch reference (rotspace dequant + standard attention) ---
            K_dequant = batched_dequantize_rotspace(
                k_packed, k_dscale, cfg.k_centroids, bits, head_dim=128
            )
            V_dequant = batched_dequantize_rotspace(
                v_packed, v_dscale, cfg.k_centroids, bits, head_dim=128
            )
            o_ref = torch.zeros(
                batch, q_heads, head_dim, device=self.device, dtype=torch.bfloat16
            )
            attn_logits_b = torch.zeros(
                batch,
                q_heads,
                max_kv_splits,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            attn_lse_b = torch.zeros(
                batch, q_heads, max_kv_splits, device=self.device, dtype=torch.float32
            )

            decode_attention_fwd(
                Q,
                K_dequant,
                V_dequant,
                o_ref,
                kv_indptr,
                kv_indices,
                attn_logits_b,
                attn_lse_b,
                num_kv_splits,
                max_kv_splits,
                sm_scale=1.0 / (head_dim**0.5),
                k_scale=1.0,
                v_scale=1.0,
            )

            # Compare
            max_diff = (o_fused.float() - o_ref.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                o_fused.float().flatten(), o_ref.float().flatten(), dim=0
            ).item()
            self.assertLess(
                max_diff,
                0.1,
                f"{bits}-bit fused kernel output diverged: max_diff={max_diff:.4f}",
            )
            self.assertGreater(
                cos, 0.999, f"{bits}-bit fused kernel output diverged: cos={cos:.6f}"
            )

    def test_asymmetric_k4v2_roundtrip(self):
        """Verify K=4bit V=2bit asymmetric quantization roundtrip."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig,
            batched_dequantize,
            batched_quantize,
        )

        cfg = TurboQuantConfig(
            bit_width=4, head_dim=128, device=self.device, k_bit_width=4, v_bit_width=2
        )
        self.assertEqual(cfg.k_packed_dim, 64)  # 4-bit: dim//2
        self.assertEqual(cfg.v_packed_dim, 32)  # 2-bit: dim//4
        self.assertEqual(cfg.k_centroids.shape[0], 16)
        self.assertEqual(cfg.v_centroids.shape[0], 4)

        torch.manual_seed(42)
        K = torch.randn(16, 4, 128, device=self.device, dtype=torch.bfloat16)
        V = torch.randn(16, 4, 128, device=self.device, dtype=torch.bfloat16)

        k_packed, k_norms, k_qnorms = batched_quantize(
            K,
            cfg.signs1,
            cfg.signs2,
            cfg.k_centroids,
            cfg.k_boundaries,
            cfg.k_bit_width,
        )
        v_packed, v_norms, v_qnorms = batched_quantize(
            V,
            cfg.signs1,
            cfg.signs2,
            cfg.v_centroids,
            cfg.v_boundaries,
            cfg.v_bit_width,
        )

        self.assertEqual(k_packed.shape[-1], 64)
        self.assertEqual(v_packed.shape[-1], 32)

        # Use full dequant (with inverse WHT) to compare in original domain
        K_hat = batched_dequantize(
            k_packed, k_norms, cfg.k_centroids, cfg.k_bit_width, cfg.signs1, cfg.signs2
        )
        V_hat = batched_dequantize(
            v_packed, v_norms, cfg.v_centroids, cfg.v_bit_width, cfg.signs1, cfg.signs2
        )

        k_cos = torch.nn.functional.cosine_similarity(
            K.float().reshape(-1), K_hat.float().reshape(-1), dim=0
        ).item()
        v_cos = torch.nn.functional.cosine_similarity(
            V.float().reshape(-1), V_hat.float().reshape(-1), dim=0
        ).item()
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
            x,
            cfg.signs1,
            cfg.signs2,
            cfg.k_centroids,
            cfg.k_boundaries,
            cfg.k_bit_width,
        )

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
            pool_packed[src], pool_dscale[src], cfg.k_centroids, 4, head_dim=128
        )
        hat_tgt = batched_dequantize_rotspace(
            pool_packed[tgt], pool_dscale[tgt], cfg.k_centroids, 4, head_dim=128
        )

        torch.testing.assert_close(hat_src, hat_tgt, atol=1e-6, rtol=0)

    def test_mla_fused_kv_write_matches_legacy_quantize_store(self):
        """Verify the Phase 35A MLA NoPE fused store matches the legacy path."""
        from sglang.srt.layers.attention.triton_ops.turboquant_quantize import (
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
        fused_codebook = torch.zeros(
            pool_size, 1, 16, dtype=torch.uint8, device=self.device
        )
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
            codebook_buffer=fused_codebook.view(torch.float8_e4m3fn),
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
        expected_fused_codebook = (
            cfg.k_centroids.float().view(1, 1, 16)
            * fused_scale[loc].float().unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            fused_codebook[loc], expected_fused_codebook.view(torch.uint8)
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
                enable_fp8_codebook=True,
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
        for pool in (legacy_pool, fused_pool, fused_rope_pool):
            scale = pool.kv_nope_scale_buffer[0][loc]
            expected_codebook = (
                cfg.k_centroids.float().view(1, 1, 16) * scale.float().unsqueeze(-1)
            ).to(torch.float8_e4m3fn)
            torch.testing.assert_close(
                pool.kv_nope_codebook_buffer[0][loc],
                expected_codebook.view(torch.uint8),
            )
            self.assertEqual(pool.get_kv_size_bytes(), pool_size * 402)

        src = torch.tensor([loc[0].item()], device=self.device)
        dst = torch.tensor([pool_size - 1], device=self.device)
        expected_codebook = fused_pool.kv_nope_codebook_buffer[0][src].clone()
        fused_pool.move_kv_cache(dst, src)
        torch.testing.assert_close(
            fused_pool.kv_nope_codebook_buffer[0][dst], expected_codebook
        )

    def test_mla_e2m1_fused_and_fallback_writers_match(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.kv_turboquant import batched_quantize
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolTurboQuant

        torch.manual_seed(20260723)
        tokens = 11
        pool_size = 32
        lora_rank = 512
        rope_dim = 64
        loc = torch.tensor([1, 2, 4, 7, 8, 11, 16, 20, 21, 27, 30], device=self.device)
        cache = torch.randn(
            tokens,
            1,
            lora_rank + rope_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        layer = SimpleNamespace(layer_id=0)

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
                turboquant_e2m1=True,
                enable_fp8_codebook=True,
                start_layer=0,
                end_layer=0,
            )

        fallback_pool = make_pool()
        fused_pool = make_pool()
        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(False):
            fallback_pool.set_kv_buffer(layer, loc, cache, cache)
        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(True):
            with envs.SGLANG_TQ_MLA_FUSED_ROPE_WRITE.override(True):
                fused_pool.set_kv_buffer(layer, loc, cache, cache)

        cfg = fallback_pool.tq_config
        expected_packed, norms, quant_norms = batched_quantize(
            cache[..., :lora_rank].contiguous(),
            cfg.signs1,
            cfg.signs2,
            cfg.k_quant_centroids,
            cfg.k_boundaries,
            4,
            storage_code_lut=cfg.k_storage_code_lut,
        )
        safe_quant_norms = torch.where(
            quant_norms > 1e-10, quant_norms, torch.ones_like(quant_norms)
        )
        expected_scale = (
            (norms / safe_quant_norms) * cfg.k_dequant_scale_multiplier
        ).to(torch.bfloat16)
        expected_codebook = (
            cfg.k_centroids.view(1, 1, 16) * expected_scale.float().unsqueeze(-1)
        ).to(torch.float8_e4m3fn)

        for pool in (fallback_pool, fused_pool):
            torch.testing.assert_close(
                pool.kv_nope_packed_buffer[0][loc], expected_packed
            )
            torch.testing.assert_close(
                pool.kv_nope_scale_buffer[0][loc].float(),
                expected_scale.float(),
                rtol=1e-2,
                atol=1e-2,
            )
            torch.testing.assert_close(
                pool.kv_nope_codebook_buffer[0][loc],
                expected_codebook.view(torch.uint8),
            )
            reconstructed = pool._dequant_nope(
                pool.kv_nope_packed_buffer[0][loc],
                pool.kv_nope_scale_buffer[0][loc],
            )
            cosine = torch.nn.functional.cosine_similarity(
                cache[..., :lora_rank].float().flatten(),
                reconstructed.float().flatten(),
                dim=0,
            ).item()
            self.assertGreater(cosine, 0.99)

    def test_mla_hotcold_pool_has_disjoint_physical_ownership(self):
        from types import SimpleNamespace

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolTurboQuant,
            MLATokenToKVPoolTurboQuantHotCold,
        )

        size = 256
        page_size = 32
        hot_tokens = 128
        layer = SimpleNamespace(layer_id=0)
        hot_loc = torch.tensor([32, 159], device=self.device)
        cold_loc = torch.tensor([160, 287], device=self.device)
        hot_cache = torch.randn(
            2, 1, 576, dtype=torch.bfloat16, device=self.device
        )
        cold_cache = torch.randn(
            2, 1, 576, dtype=torch.bfloat16, device=self.device
        )

        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(False):
            pool = MLATokenToKVPoolTurboQuantHotCold(
                size=size,
                page_size=page_size,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=1,
                device=self.device,
                enable_memory_saver=False,
                turboquant_bits=4,
                enable_fp8_codebook=True,
                hot_capacity_tokens=hot_tokens,
                start_layer=0,
                end_layer=0,
            )
            cold_reference = MLATokenToKVPoolTurboQuant(
                size=size - hot_tokens,
                page_size=page_size,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=1,
                device=self.device,
                enable_memory_saver=False,
                turboquant_bits=4,
                enable_fp8_codebook=True,
                start_layer=0,
                end_layer=0,
            )

            pool.set_kv_buffer(layer, hot_loc, hot_cache, hot_cache)
            self.assertEqual(
                int(torch.count_nonzero(pool.kv_nope_packed_buffer[0]).item()), 0
            )
            self.assertEqual(
                int(torch.count_nonzero(pool.kv_nope_scale_buffer[0]).item()), 0
            )
            self.assertEqual(
                int(torch.count_nonzero(pool.kv_rope_buffer[0]).item()), 0
            )
            expected_hot = hot_cache.to(torch.float8_e4m3fn)
            torch.testing.assert_close(
                pool.kv_hot_buffer[0][hot_loc], expected_hot, rtol=0, atol=0
            )
            hot_snapshot = pool.kv_hot_buffer[0].clone()

            adjusted_cold_loc = cold_loc - hot_tokens
            pool.set_kv_buffer(layer, cold_loc, cold_cache, cold_cache)
            cold_reference.set_kv_buffer(
                layer, adjusted_cold_loc, cold_cache, cold_cache
            )
            torch.testing.assert_close(pool.kv_hot_buffer[0], hot_snapshot)
            torch.testing.assert_close(
                pool.kv_nope_packed_buffer[0][adjusted_cold_loc],
                cold_reference.kv_nope_packed_buffer[0][adjusted_cold_loc],
            )
            torch.testing.assert_close(
                pool.kv_nope_scale_buffer[0][adjusted_cold_loc],
                cold_reference.kv_nope_scale_buffer[0][adjusted_cold_loc],
            )
            torch.testing.assert_close(
                pool.kv_rope_buffer[0][adjusted_cold_loc],
                cold_reference.kv_rope_buffer[0][adjusted_cold_loc],
            )
            torch.testing.assert_close(
                pool.kv_nope_codebook_buffer[0][adjusted_cold_loc],
                cold_reference.kv_nope_codebook_buffer[0][adjusted_cold_loc],
            )

            boundary_cache = torch.stack((hot_cache[1], cold_cache[0]))
            pool.set_mla_kv_buffer(
                layer,
                torch.stack((hot_loc[1], cold_loc[0])),
                boundary_cache[..., :512],
                boundary_cache[..., 512:],
                logical_start=hot_tokens - 1,
            )
            torch.testing.assert_close(
                pool.kv_hot_buffer[0][hot_loc[1]],
                boundary_cache[0].to(torch.float8_e4m3fn),
            )
            boundary_cold_nope, boundary_cold_rope = pool.get_mla_kv_buffer(
                layer,
                cold_loc[:1],
                torch.bfloat16,
                logical_start=hot_tokens,
            )
            reference_boundary_nope, reference_boundary_rope = (
                cold_reference.get_mla_kv_buffer(
                    layer,
                    adjusted_cold_loc[:1],
                    torch.bfloat16,
                )
            )
            torch.testing.assert_close(
                boundary_cold_nope, reference_boundary_nope
            )
            torch.testing.assert_close(
                boundary_cold_rope, reference_boundary_rope
            )

            # Chunked prefill gathers only the requested rows. Verify mixed
            # tier ordering and that the operation creates no persistent
            # full-width shadow allocation.
            hot_ptr = pool.kv_hot_buffer[0].data_ptr()
            packed_ptr = pool.kv_nope_packed_buffer[0].data_ptr()
            scale_ptr = pool.kv_nope_scale_buffer[0].data_ptr()
            rope_ptr = pool.kv_rope_buffer[0].data_ptr()
            codebook_ptr = pool.kv_nope_codebook_buffer[0].data_ptr()
            mixed_loc = torch.stack(
                (cold_loc[1], hot_loc[0], cold_loc[0], hot_loc[1])
            )
            gathered_nope, gathered_rope = pool.get_mla_kv_buffer(
                layer, mixed_loc, torch.bfloat16
            )
            cold_nope, cold_rope = cold_reference.get_mla_kv_buffer(
                layer, adjusted_cold_loc, torch.bfloat16
            )
            expected_nope = torch.stack(
                (
                    cold_nope[1],
                    expected_hot[0, ..., :512].to(torch.bfloat16),
                    cold_nope[0],
                    expected_hot[1, ..., :512].to(torch.bfloat16),
                )
            )
            expected_rope = torch.stack(
                (
                    cold_rope[1],
                    expected_hot[0, ..., 512:].to(torch.bfloat16),
                    cold_rope[0],
                    expected_hot[1, ..., 512:].to(torch.bfloat16),
                )
            )
            torch.testing.assert_close(gathered_nope, expected_nope)
            torch.testing.assert_close(gathered_rope, expected_rope)

            hinted_nope, hinted_rope = pool.get_mla_kv_buffer(
                layer,
                torch.stack((hot_loc[1], cold_loc[0])),
                torch.bfloat16,
                logical_start=hot_tokens - 1,
            )
            torch.testing.assert_close(
                hinted_nope,
                torch.stack(
                    (
                        expected_hot[1, ..., :512].to(torch.bfloat16),
                        cold_nope[0],
                    )
                ),
            )
            torch.testing.assert_close(
                hinted_rope,
                torch.stack(
                    (
                        expected_hot[1, ..., 512:].to(torch.bfloat16),
                        cold_rope[0],
                    )
                ),
            )
            self.assertEqual(pool.kv_hot_buffer[0].data_ptr(), hot_ptr)
            self.assertEqual(pool.kv_nope_packed_buffer[0].data_ptr(), packed_ptr)
            self.assertEqual(pool.kv_nope_scale_buffer[0].data_ptr(), scale_ptr)
            self.assertEqual(pool.kv_rope_buffer[0].data_ptr(), rope_ptr)
            self.assertEqual(
                pool.kv_nope_codebook_buffer[0].data_ptr(), codebook_ptr
            )

        expected_bytes = (hot_tokens + page_size) * 576 + (
            size - hot_tokens + page_size
        ) * 402
        self.assertEqual(pool.get_kv_size_bytes(), expected_bytes)

        pool.move_kv_cache(
            torch.tensor([64], device=self.device),
            torch.tensor([32], device=self.device),
        )
        torch.testing.assert_close(
            pool.kv_hot_buffer[0][64], pool.kv_hot_buffer[0][32]
        )
        pool.move_kv_cache(
            torch.tensor([192], device=self.device),
            torch.tensor([160], device=self.device),
        )
        torch.testing.assert_close(
            pool.kv_nope_packed_buffer[0][64],
            pool.kv_nope_packed_buffer[0][32],
        )
        with self.assertRaisesRegex(RuntimeError, "cross-tier move"):
            pool.move_kv_cache(
                torch.tensor([160], device=self.device),
                torch.tensor([32], device=self.device),
            )

    def test_hot_fp8_mla_scatter_writer_handles_boundary_tile(self):
        """Exercise the real direct-FP8 hot-cache writer used by decode."""
        from sglang.srt.mem_cache.triton_ops.mla_buffer import (
            set_mla_kv_buffer_triton,
        )

        torch.manual_seed(20260724)
        nope_dim = 512
        rope_dim = 64
        # Cover both dispatcher branches: the small Triton fallback used by
        # decode/verify and the bulk TMA store used by exact-10K prefill.
        for batch_size in (3, 1024):
            with self.subTest(batch_size=batch_size):
                cache_size = batch_size + 16
                loc = torch.arange(
                    1,
                    batch_size + 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                cache_k_nope = torch.randn(
                    (batch_size, 1, nope_dim),
                    dtype=torch.bfloat16,
                    device=self.device,
                ).to(torch.float8_e4m3fn)
                cache_k_rope = torch.randn(
                    (batch_size, 1, rope_dim),
                    dtype=torch.bfloat16,
                    device=self.device,
                ).to(torch.float8_e4m3fn)
                kv_buffer = torch.zeros(
                    (cache_size, 1, nope_dim + rope_dim),
                    dtype=torch.float8_e4m3fn,
                    device=self.device,
                )

                set_mla_kv_buffer_triton(
                    kv_buffer,
                    loc,
                    cache_k_nope,
                    cache_k_rope,
                )

                torch.testing.assert_close(
                    kv_buffer[loc, :, :nope_dim],
                    cache_k_nope,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    kv_buffer[loc, :, nope_dim:],
                    cache_k_rope,
                    rtol=0,
                    atol=0,
                )

    def test_hotcold_static_allocator_preserves_tiers_after_reuse(self):
        from sglang.srt.mem_cache.allocator import (
            StaticTieredPagedTokenToKVPoolAllocator,
        )

        allocator = StaticTieredPagedTokenToKVPoolAllocator(
            size=256,
            page_size=32,
            dtype=torch.bfloat16,
            device=self.device,
            kvcache=SimpleNamespace(),
            need_sort=False,
            hot_capacity_tokens=128,
        )

        def alloc_extend(prefix_len, seq_len, last_loc):
            return allocator.alloc_extend(
                torch.tensor([prefix_len], dtype=torch.int64, device=self.device),
                torch.tensor([prefix_len], dtype=torch.int64),
                torch.tensor([seq_len], dtype=torch.int64, device=self.device),
                torch.tensor([seq_len], dtype=torch.int64),
                torch.tensor([last_loc], dtype=torch.int64, device=self.device),
                seq_len - prefix_len,
            )

        first = alloc_extend(0, 160, -1)
        torch.testing.assert_close(
            first,
            torch.arange(32, 192, dtype=torch.int64, device=self.device),
        )
        allocator.free(first)

        # A completed request must restart from the FP8 tier rather than reuse
        # the untouched high pages at the head of the stock allocator queue.
        second = alloc_extend(0, 160, -1)
        torch.testing.assert_close(second, first)

        # Simulate tail retraction/cancellation and growth back across the
        # ownership boundary.  The reclaimed fourth/fifth pages must return in
        # logical order.
        allocator.free(second[96:])
        regrown = alloc_extend(96, 160, int(second[95].item()))
        torch.testing.assert_close(regrown, second[96:])

    def test_hotcold_static_allocator_sorts_only_after_release(self):
        import torch

        from sglang.srt.mem_cache.allocator import (
            StaticTieredPagedTokenToKVPoolAllocator,
        )

        allocator = StaticTieredPagedTokenToKVPoolAllocator(
            size=256,
            page_size=32,
            dtype=torch.bfloat16,
            device=self.device,
            kvcache=SimpleNamespace(),
            need_sort=False,
            hot_capacity_tokens=128,
        )
        prefix_lens = torch.tensor([0], dtype=torch.int64, device=self.device)
        prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([32], dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor([32], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64, device=self.device)

        with patch(
            "sglang.srt.mem_cache.allocator.paged.torch.sort",
            wraps=torch.sort,
        ) as sort_pages:
            first = allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens=32,
            )
            self.assertEqual(sort_pages.call_count, 0)

            allocator.free(first)
            reused = allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens=32,
            )
            self.assertEqual(sort_pages.call_count, 1)

        torch.testing.assert_close(first, reused)

    def test_tokenspeed_hotcold_prefill_passes_cpu_logical_start(self):
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend._tq4_hotcold_cache = True
        backend.token_to_kv_pool = Mock()
        backend._fused_rope_fp8_quantize = Mock(
            return_value=(
                torch.zeros(
                    2, 1, 576, dtype=torch.float8_e4m3fn, device=self.device
                ),
                torch.zeros(
                    2, 1, 576, dtype=torch.float8_e4m3fn, device=self.device
                ),
            )
        )
        layer = SimpleNamespace(
            kv_b_proj=Mock(
                return_value=(
                    torch.zeros(
                        2, 1, 1024, dtype=torch.bfloat16, device=self.device
                    ),
                )
            ),
            num_local_heads=1,
            qk_nope_head_dim=512,
            v_head_dim=512,
            qk_rope_head_dim=64,
            rotary_emb=SimpleNamespace(
                cos_sin_cache=torch.empty(0, device=self.device),
                is_neox_style=True,
            ),
            attn_mha=SimpleNamespace(layer_id=0),
        )
        batch = SimpleNamespace(
            batch_size=1,
            extend_prefix_lens_cpu=[123],
            out_cache_loc=torch.tensor(
                [160, 161], dtype=torch.int64, device=self.device
            ),
        )
        q = torch.zeros(
            2, 1, 576, dtype=torch.bfloat16, device=self.device
        )
        kv_a = torch.zeros(
            2, 512, dtype=torch.bfloat16, device=self.device
        )
        k_pe = torch.zeros(
            2, 1, 64, dtype=torch.bfloat16, device=self.device
        )
        with patch(
            "sglang.srt.layers.attention.tokenspeed_mla_backend.fp8_quantize",
            side_effect=lambda tensor, **_: tensor.to(torch.float8_e4m3fn),
        ):
            backend.prepare_prefill_qkv(
                q=q,
                q_pe=q[..., 512:],
                kv_a=kv_a,
                k_pe=k_pe,
                positions=torch.arange(2, device=self.device),
                layer=layer,
                forward_batch=batch,
            )
        self.assertEqual(
            backend.token_to_kv_pool.set_mla_kv_buffer.call_args.kwargs[
                "logical_start"
            ],
            123,
        )

    def test_hotcold_parent_decode_uses_incumbent_fp8_frontend(self):
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend._tq4_hotcold_cache = True
        backend._tq_pool = SimpleNamespace(hot_capacity_tokens=128)
        backend.num_draft_tokens = 0
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.token_to_kv_pool = Mock()
        backend._get_decode_kv_cache = Mock(
            return_value=torch.empty(1, device=self.device)
        )
        fp8_query = torch.zeros(
            1, 1, 1, 576, dtype=torch.float8_e4m3fn, device=self.device
        )
        fp8_k = torch.zeros(
            1, 1, 512, dtype=torch.float8_e4m3fn, device=self.device
        )
        fp8_rope = torch.zeros(
            1, 1, 64, dtype=torch.float8_e4m3fn, device=self.device
        )
        backend._run_decode_kernel = Mock(
            return_value=torch.zeros(
                1, 1, 1, 512, dtype=torch.bfloat16, device=self.device
            )
        )
        backend.forward_decode_metadata = SimpleNamespace(
            block_kv_indices=torch.tensor(
                [[1]], dtype=torch.int32, device=self.device
            ),
            max_seq_len_k=64,
            batch_size=1,
        )
        mode = SimpleNamespace(
            is_decode_or_idle=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens_cpu=torch.tensor([64], dtype=torch.int64),
            seq_lens=torch.tensor([64], dtype=torch.int32, device=self.device),
            positions=torch.tensor([63], dtype=torch.int64, device=self.device),
            out_cache_loc=torch.tensor(
                [95], dtype=torch.int64, device=self.device
            ),
            decode_trtllm_mla_metadata=None,
            batch_size=1,
        )
        layer = SimpleNamespace(
            tp_q_head_num=1,
            v_head_dim=512,
            head_dim=576,
        )
        q = torch.zeros(
            1, 1, 512, dtype=torch.bfloat16, device=self.device
        )
        q_rope = torch.zeros(
            1, 1, 64, dtype=torch.bfloat16, device=self.device
        )
        k = torch.zeros_like(q)
        k_rope = torch.zeros_like(q_rope)
        with patch(
            "sglang.srt.layers.attention.trtllm_mla_backend.mla_quantize_and_rope_for_fp8",
            return_value=(fp8_query, fp8_k, fp8_rope),
        ) as fused_frontend:
            TRTLLMMLABackend.forward_decode(
                backend,
                q,
                k,
                None,
                layer,
                batch,
                q_rope=q_rope,
                k_rope=k_rope,
                cos_sin_cache=torch.empty(0, device=self.device),
                llama_4_scaling=torch.ones(
                    1, dtype=torch.bfloat16, device=self.device
                ),
            )
        fused_frontend.assert_called_once()
        self.assertEqual(
            backend.token_to_kv_pool.set_mla_kv_buffer.call_args.kwargs[
                "logical_start"
            ],
            63,
        )
        parent_query = backend._run_decode_kernel.call_args.kwargs["query"]
        self.assertIs(parent_query.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            parent_query,
            fp8_query.view(1, 1, 1, 576),
        )

    def test_hotcold_target_verify_keeps_incumbent_fp8_frontend(self):
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
        )
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            TRTLLMMLABackend,
        )

        backend = object.__new__(TokenspeedMLABackend)
        backend.data_type = torch.bfloat16
        backend._tq4_hotcold_cache = True
        backend._tq_pool = SimpleNamespace(hot_capacity_tokens=128)
        backend.num_draft_tokens = 5
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.token_to_kv_pool = Mock()
        backend.forward_prefill_metadata = None
        backend._get_decode_kv_cache = Mock(
            return_value=torch.empty(
                1, dtype=torch.float8_e4m3fn, device=self.device
            )
        )
        fp8_query = torch.zeros(
            5, 1, 576, dtype=torch.float8_e4m3fn, device=self.device
        )
        fp8_k = torch.zeros(
            5, 1, 512, dtype=torch.float8_e4m3fn, device=self.device
        )
        fp8_rope = torch.zeros(
            5, 1, 64, dtype=torch.float8_e4m3fn, device=self.device
        )
        backend._run_decode_kernel = Mock(
            return_value=torch.zeros(
                1, 5, 1, 512, dtype=torch.bfloat16, device=self.device
            )
        )
        metadata = SimpleNamespace(
            block_kv_indices=torch.tensor(
                [[1]], dtype=torch.int32, device=self.device
            ),
            max_seq_len_k=64,
            seq_lens_k=torch.tensor(
                [69], dtype=torch.int32, device=self.device
            ),
            batch_size=1,
        )
        backend.forward_decode_metadata = metadata
        mode = SimpleNamespace(
            is_decode_or_idle=lambda: False,
            is_target_verify=lambda: True,
            is_draft_extend_v2=lambda: False,
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens_cpu=torch.tensor([64], dtype=torch.int64),
            seq_lens=torch.tensor([64], dtype=torch.int32, device=self.device),
            positions=torch.arange(5, dtype=torch.int64, device=self.device),
            out_cache_loc=torch.arange(
                5, dtype=torch.int64, device=self.device
            ),
            decode_trtllm_mla_metadata=metadata,
            spec_info=SimpleNamespace(draft_token_num=5, topk=1),
            batch_size=1,
        )
        layer = SimpleNamespace(
            tp_q_head_num=1,
            v_head_dim=512,
            head_dim=576,
        )
        q = torch.zeros(
            5, 1, 512, dtype=torch.bfloat16, device=self.device
        )
        q_rope = torch.zeros(
            5, 1, 64, dtype=torch.bfloat16, device=self.device
        )
        k = torch.zeros_like(q)
        k_rope = torch.zeros_like(q_rope)
        with patch(
            "sglang.srt.layers.attention.trtllm_mla_backend.mla_quantize_and_rope_for_fp8",
            return_value=(fp8_query, fp8_k, fp8_rope),
        ) as fused_frontend:
            TRTLLMMLABackend.forward_extend(
                backend,
                q,
                k,
                None,
                layer,
                batch,
                q_rope=q_rope,
                k_rope=k_rope,
                cos_sin_cache=torch.empty(0, device=self.device),
            )
        fused_frontend.assert_called_once()
        self.assertEqual(
            backend.token_to_kv_pool.set_mla_kv_buffer.call_args.kwargs[
                "logical_start"
            ],
            64,
        )
        parent_query = backend._run_decode_kernel.call_args.kwargs["query"]
        self.assertIs(parent_query.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(
            parent_query,
            fp8_query.view(1, 5, 1, 576),
        )

    def test_tokenspeed_hotcold_backend_matches_dense_segment_oracle(self):
        import tokenspeed_mla

        from sglang.jit_kernel.utils import is_arch_support_pdl
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedMLABackend,
            _quantize_tq4_query,
        )
        from sglang.srt.mem_cache.memory_pool import (
            MLATokenToKVPoolTurboQuantHotCold,
        )

        page_size = 32
        hot_tokens = int(os.getenv("SGLANG_TQ_HOTCOLD_ORACLE_HOT", "128"))
        size = int(os.getenv("SGLANG_TQ_HOTCOLD_ORACLE_SIZE", "256"))
        q_len = 5
        heads = 8
        layer = SimpleNamespace(layer_id=0, scaling=1 / math.sqrt(576))
        torch.manual_seed(20260724)

        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(False):
            pool = MLATokenToKVPoolTurboQuantHotCold(
                size=size,
                page_size=page_size,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=1,
                device=self.device,
                enable_memory_saver=False,
                turboquant_bits=4,
                enable_fp8_codebook=True,
                hot_capacity_tokens=hot_tokens,
                start_layer=0,
                end_layer=0,
            )
            cache = (
                torch.randn(
                    size,
                    1,
                    576,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
            loc = torch.arange(
                page_size,
                size + page_size,
                dtype=torch.int64,
                device=self.device,
            )
            pool.set_kv_buffer(layer, loc, cache, cache)

        backend = object.__new__(TokenspeedMLABackend)
        backend._tq4_cache = True
        backend._tq4_hotcold_cache = True
        backend._tq_pool = pool
        backend._tq_config = pool.tq_config
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.page_size = page_size
        backend.num_q_heads = heads
        backend.max_context_len = size
        backend.supports_custom_decode_mask = True
        backend._tokenspeed_workspace = None

        cold_nope = pool._dequant_nope(
            pool.kv_nope_packed_buffer[0],
            pool.kv_nope_scale_buffer[0],
        )
        cold_dense = torch.cat(
            (cold_nope, pool.kv_rope_buffer[0]), dim=-1
        ).to(torch.float8_e4m3fn)
        dense_flat = torch.zeros(
            size + page_size,
            576,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        dense_flat[: hot_tokens + page_size] = pool.kv_hot_buffer[0].squeeze(1)
        dense_flat[hot_tokens + page_size :] = cold_dense[page_size:].squeeze(1)
        dense_cache = dense_flat.view(-1, page_size, 576)
        hot_cache = backend._get_decode_kv_cache(layer)
        workspace = torch.empty(64 << 20, dtype=torch.int8, device=self.device)
        query = (
            torch.randn(
                1,
                q_len,
                heads,
                576,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        query_fp8 = _quantize_tq4_query(
            query, 512, is_arch_support_pdl()
        )

        def dense_oracle(seq_len, custom_mask=None):
            pages = (seq_len + page_size - 1) // page_size
            return tokenspeed_mla.tokenspeed_mla_decode(
                query=query_fp8,
                kv_cache=dense_cache,
                workspace_buffer=workspace,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=torch.arange(
                    1, pages + 1, dtype=torch.int32, device=self.device
                ).view(1, -1),
                seq_lens=torch.tensor(
                    [seq_len], dtype=torch.int32, device=self.device
                ),
                max_seq_len=seq_len,
                softmax_scale=layer.scaling,
                causal_mask=True,
                custom_mask=custom_mask,
                cmask_off=torch.zeros(
                    1, dtype=torch.int32, device=self.device
                ),
                enable_pdl=is_arch_support_pdl(),
            )

        def run_backend(seq_len, custom_mask=None):
            pages = (seq_len + page_size - 1) // page_size
            return backend._run_decode_kernel(
                query=query,
                kv_cache=hot_cache,
                block_tables=torch.arange(
                    1, pages + 1, dtype=torch.int32, device=self.device
                ).view(1, -1),
                seq_lens=torch.tensor(
                    [seq_len], dtype=torch.int32, device=self.device
                ),
                max_seq_len=seq_len,
                layer=layer,
                custom_mask=custom_mask,
                custom_mask_offsets=torch.zeros(
                    1, dtype=torch.int32, device=self.device
                ),
            )

        # The winning exact-context branch must remain the ordinary FP8
        # TokenSpeed decode with byte-identical output.
        torch.testing.assert_close(
            run_backend(100), dense_oracle(100), rtol=0, atol=0
        )

        # The boundary case has query tokens on both sides of the ownership
        # split and exercises the synthesized segmented causal masks.
        torch.testing.assert_close(
            run_backend(hot_tokens + 2),
            dense_oracle(hot_tokens + 2),
            rtol=0,
            atol=0.02,
        )

        # A longer tree-verification request keeps all history visible while
        # masking sibling query tokens, as DFlash target verification does.
        full_seq_len = size
        tree_mask = torch.ones(
            q_len, full_seq_len, dtype=torch.bool, device=self.device
        )
        tree_mask[:, full_seq_len - q_len :] = False
        for query_idx in range(q_len):
            tree_mask[query_idx, full_seq_len - q_len + query_idx] = True
        torch.testing.assert_close(
            run_backend(full_seq_len, tree_mask),
            dense_oracle(full_seq_len, tree_mask),
            rtol=0,
            atol=0.02,
        )

        # Live DFlash validation showed that low acceptance after crossing the
        # compressed boundary can emit invalid bonus-token chains even though
        # the isolated q5 attention oracle above is correct. Fail closed until
        # that speculative lifecycle is integrated explicitly.
        backend.num_draft_tokens = q_len
        with self.assertRaisesRegex(RuntimeError, "admission guard was bypassed"):
            run_backend(full_seq_len, tree_mask)

    def test_non_128_head_dim(self):
        """Verify TurboQuant works with head_dim=64 and head_dim=256."""
        from sglang.srt.layers.quantization.kv_turboquant import (
            TurboQuantConfig,
            batched_dequantize,
            batched_dequantize_rotspace,
            batched_quantize,
        )

        for dim in [64, 256]:
            for bits in [2, 4]:
                cfg = TurboQuantConfig(bit_width=bits, head_dim=dim, device=self.device)
                torch.manual_seed(42)
                x = torch.randn(8, 4, dim, device=self.device, dtype=torch.bfloat16)

                packed, norms, qnorms = batched_quantize(
                    x, cfg.signs1, cfg.signs2, cfg.k_centroids, cfg.k_boundaries, bits
                )

                # Full dequant roundtrip
                x_hat = batched_dequantize(
                    packed, norms, cfg.k_centroids, bits, cfg.signs1, cfg.signs2
                )
                self.assertEqual(
                    x_hat.shape,
                    x.shape,
                    f"dim={dim} bits={bits}: shape mismatch {x_hat.shape} vs {x.shape}",
                )

                cos = torch.nn.functional.cosine_similarity(
                    x.float().reshape(-1), x_hat.float().reshape(-1), dim=0
                ).item()
                min_cos = 0.5 if bits == 2 else 0.93
                self.assertGreater(
                    cos, min_cos, f"dim={dim} bits={bits}: cos={cos:.4f} < {min_cos}"
                )

                # Rotspace dequant
                safe_qn = torch.where(qnorms > 1e-10, qnorms, torch.ones_like(qnorms))
                ds = (norms / safe_qn).to(torch.bfloat16)
                rs = batched_dequantize_rotspace(
                    packed, ds, cfg.k_centroids, bits, head_dim=dim
                )
                self.assertEqual(rs.shape[-1], dim)


if __name__ == "__main__":
    unittest.main()
