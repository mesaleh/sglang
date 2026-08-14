import math
import unittest
from types import SimpleNamespace

import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 0)


@unittest.skipUnless(_is_sm100(), "native N8 front end requires SM100")
class TestTQMLAFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        cls.cfg = NativeE2M1MLAConfig(device=cls.device)

    def _reference_write(self, x, rope, locations, pool_size):
        from sglang.kernels.ops.attention.turboquant_quantize import (
            fused_native_e2m1_mla_quantize_and_store,
        )

        packed = torch.full(
            (pool_size, 1, 256), 0xA5, dtype=torch.uint8, device=self.device
        )
        scale = torch.full(
            (pool_size, 1), -2.0, dtype=torch.bfloat16, device=self.device
        )
        rope_out = torch.full(
            (pool_size, 1, 64),
            3.0,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        workspace_tokens = max(1, min(7, x.shape[0]))
        unit = torch.empty(
            (workspace_tokens, 1, 512), dtype=torch.float32, device=self.device
        )
        norms = torch.empty(
            (workspace_tokens, 1), dtype=torch.float32, device=self.device
        )
        rotated = torch.empty_like(unit)
        cfg = self.cfg
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
            locations,
            rope,
            rope_out,
            pre_unit=unit,
            pre_norms=norms,
            pre_y=rotated,
        )
        return packed, scale, rope_out

    def _allocate_outputs(self, pool_size):
        return (
            torch.full(
                (pool_size, 1, 256),
                0xA5,
                dtype=torch.uint8,
                device=self.device,
            ),
            torch.full(
                (pool_size, 1), -2.0, dtype=torch.bfloat16, device=self.device
            ),
            torch.full(
                (pool_size, 1, 64),
                3.0,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            torch.zeros((1,), dtype=torch.int32, device=self.device),
        )

    def test_cache_writer_matches_reference_for_both_location_dtypes(self):
        from sglang.kernels.jit.tq_mla_frontend import tq_mla_cache_writer_out

        torch.manual_seed(20260814)
        tokens = 19
        pool_size = 80
        x = torch.randn(
            (tokens * 2, 1, 512), dtype=torch.bfloat16, device=self.device
        )[::2]
        rope = torch.randn(
            (tokens * 2, 1, 64), dtype=torch.bfloat16, device=self.device
        )[::2]
        x[0].zero_()
        x[1].zero_()
        x[1, 0, 0] = torch.finfo(torch.bfloat16).tiny
        base_locations = [
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
        ]
        for location_dtype in (torch.int32, torch.int64):
            with self.subTest(location_dtype=location_dtype):
                locations = torch.tensor(
                    base_locations, dtype=location_dtype, device=self.device
                )
                expected = self._reference_write(x, rope, locations, pool_size)
                actual = self._allocate_outputs(pool_size)
                cfg = self.cfg
                tq_mla_cache_writer_out(
                    x,
                    rope,
                    locations,
                    cfg.signs1,
                    cfg.signs2,
                    cfg.boundaries,
                    cfg.levels,
                    cfg.codes,
                    actual[0],
                    actual[1],
                    actual[2],
                    actual[3],
                    grid=cfg.grid,
                )
                torch.cuda.synchronize(self.device)
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[2].view(torch.uint8),
                    expected[2].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(actual[3].item(), 0)

    def test_fused_rope_frontend_matches_materialized_bf16_oracle(self):
        import flashinfer.rope

        from sglang.kernels.jit.tq_mla_frontend import tq_mla_frontend_rope_out
        from sglang.kernels.ops.quantization.hadamard import (
            hadamard_transform_with_signs,
        )

        torch.manual_seed(37)
        tokens = 5
        query_latent = torch.randn(
            (tokens, 8, 512), dtype=torch.bfloat16, device=self.device
        )
        query_rope = torch.randn(
            (tokens, 8, 64), dtype=torch.bfloat16, device=self.device
        )
        cache_latent = torch.randn(
            (tokens, 1, 512), dtype=torch.bfloat16, device=self.device
        )
        cache_rope = torch.randn(
            (tokens, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        cos_sin_cache = torch.randn(
            (128, 64), dtype=torch.float32, device=self.device
        )
        positions = torch.tensor(
            [3, 17, 31, 63, 127], dtype=torch.int64, device=self.device
        )
        locations = torch.tensor(
            [11, 3, 19, 7, 23], dtype=torch.int64, device=self.device
        )
        query_rope_ref = torch.empty_like(query_rope)
        cache_rope_ref = torch.empty_like(cache_rope)
        flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
            q=query_rope,
            k=cache_rope,
            q_rope=query_rope_ref,
            k_rope=cache_rope_ref,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            interleave=False,
        )
        expected_cache = self._reference_write(
            cache_latent, cache_rope_ref, locations, 32
        )

        for rotation_fused in (True, False):
            with self.subTest(rotation_fused=rotation_fused):
                query_nope_ref = query_latent
                if not rotation_fused:
                    query_nope_ref = hadamard_transform_with_signs(
                        query_latent,
                        self.cfg.signs1,
                        self.cfg.signs2,
                        scale=1.0 / math.sqrt(512),
                    )
                query_expected = torch.cat(
                    (query_nope_ref, query_rope_ref), dim=-1
                ).to(torch.float8_e4m3fn)
                actual = self._allocate_outputs(32)
                query_actual = torch.empty_like(query_expected)
                cfg = self.cfg
                tq_mla_frontend_rope_out(
                    query_latent,
                    query_rope,
                    cache_latent,
                    cache_rope,
                    cos_sin_cache,
                    positions,
                    locations,
                    cfg.signs1,
                    cfg.signs2,
                    cfg.boundaries,
                    cfg.levels,
                    cfg.codes,
                    query_actual,
                    actual[0],
                    actual[1],
                    actual[2],
                    actual[3],
                    grid=cfg.grid,
                    rotation_fused=rotation_fused,
                    num_warps=9,
                )
                torch.cuda.synchronize(self.device)
                torch.testing.assert_close(
                    query_actual.view(torch.uint8),
                    query_expected.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual[0], expected_cache[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual[1], expected_cache[1], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual[2].view(torch.uint8),
                    expected_cache[2].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(actual[3].item(), 0)

    def test_invalid_inputs_set_sticky_status_before_writing(self):
        from sglang.kernels.jit.tq_mla_frontend import (
            tq_mla_cache_writer_out,
            tq_mla_frontend_rope_out,
        )

        cfg = self.cfg
        x = torch.randn((2, 1, 512), dtype=torch.bfloat16, device=self.device)
        rope = torch.randn((2, 1, 64), dtype=torch.bfloat16, device=self.device)
        outputs = self._allocate_outputs(8)
        invalid_locations = torch.tensor(
            [-1, 8], dtype=torch.int64, device=self.device
        )
        tq_mla_cache_writer_out(
            x,
            rope,
            invalid_locations,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *outputs,
            grid=cfg.grid,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(outputs[3].item(), 1)
        self.assertTrue(torch.all(outputs[0] == 0xA5).item())
        self.assertTrue(torch.all(outputs[1] == -2.0).item())
        self.assertTrue(torch.all(outputs[2] == 3.0).item())

        # A later valid launch cannot clear the process-lifetime status.
        tq_mla_cache_writer_out(
            x[:1],
            rope[:1],
            torch.tensor([3], dtype=torch.int64, device=self.device),
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *outputs,
            grid=cfg.grid,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(outputs[3].item(), 1)

        query = torch.randn((1, 8, 512), dtype=torch.bfloat16, device=self.device)
        query_rope = torch.randn(
            (1, 8, 64), dtype=torch.bfloat16, device=self.device
        )
        query_out = torch.full(
            (1, 8, 576), 3.0, dtype=torch.float8_e4m3fn, device=self.device
        )
        outputs = self._allocate_outputs(8)
        tq_mla_frontend_rope_out(
            query,
            query_rope,
            x[:1],
            rope[:1],
            torch.ones((4, 64), dtype=torch.float32, device=self.device),
            torch.tensor([4], dtype=torch.int64, device=self.device),
            torch.tensor([3], dtype=torch.int64, device=self.device),
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            query_out,
            *outputs,
            grid=cfg.grid,
            rotation_fused=True,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(outputs[3].item(), 2)
        self.assertTrue(torch.all(query_out == 3.0).item())
        self.assertTrue(torch.all(outputs[0] == 0xA5).item())

        # The combined entry must reject a bad cache slot before emitting even
        # the otherwise independent query output.
        query_out.fill_(3.0)
        outputs = self._allocate_outputs(8)
        tq_mla_frontend_rope_out(
            query,
            query_rope,
            x[:1],
            rope[:1],
            torch.ones((4, 64), dtype=torch.float32, device=self.device),
            torch.tensor([3], dtype=torch.int64, device=self.device),
            torch.tensor([8], dtype=torch.int64, device=self.device),
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            query_out,
            *outputs,
            grid=cfg.grid,
            rotation_fused=True,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(outputs[3].item(), 1)
        self.assertTrue(torch.all(query_out == 3.0).item())
        self.assertTrue(torch.all(outputs[0] == 0xA5).item())

    def test_pool_dispatch_and_graph_replay_are_allocation_stable(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolNativeE2M1

        with (
            envs.SGLANG_TQ_MLA_NATIVE_SM100_WRITER.override(True),
            envs.SGLANG_TQ_MLA_KV_WRITE_WORKSPACE_TOKENS.override(8),
        ):
            pool = MLATokenToKVPoolNativeE2M1(
                size=64,
                page_size=32,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=1,
                device=str(self.device),
                enable_memory_saver=False,
            )
            torch.manual_seed(91)
            x = torch.randn((5, 1, 512), dtype=torch.bfloat16, device=self.device)
            rope = torch.randn((5, 1, 64), dtype=torch.bfloat16, device=self.device)
            locations = torch.tensor(
                [31, 1, 32, 63, 2], dtype=torch.int64, device=self.device
            )
            expected = self._reference_write(
                x, rope, locations, pool.size + pool.page_size
            )
            pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), locations, x, rope)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pool.set_mla_kv_buffer(
                    SimpleNamespace(layer_id=0), locations, x, rope
                )
            torch.cuda.synchronize(self.device)
            before = torch.cuda.memory_allocated(self.device)
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize(self.device)
            after = torch.cuda.memory_allocated(self.device)
            self.assertEqual(after, before)
            actual = pool.get_native_e2m1_buffers(0)
            torch.testing.assert_close(
                actual.packed_nope[locations], expected[0][locations], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.nope_scale[locations], expected[1][locations], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.rope[locations].view(torch.uint8),
                expected[2][locations].view(torch.uint8),
                rtol=0,
                atol=0,
            )
            self.assertEqual(pool.get_native_e2m1_fault_status().item(), 0)


if __name__ == "__main__":
    unittest.main()
