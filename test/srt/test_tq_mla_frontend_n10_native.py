from __future__ import annotations

import unittest
from pathlib import Path

import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 0)


@unittest.skipUnless(_is_sm100(), "native N10 front end requires SM100")
class TestTQMLAFrontendN10Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        cls.cfg = NativeE2M1MLAConfig(device=cls.device)

    def _outputs(self, tokens: int, pool_size: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.full(
                (tokens, 8, 512),
                3.0,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
            torch.full(
                (tokens, 8, 64),
                3.0,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.full(
                (pool_size, 1, 256),
                0xA5,
                dtype=torch.uint8,
                device=self.device,
            ),
            torch.full(
                (pool_size, 1),
                -2.0,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.full(
                (pool_size, 1, 64),
                3.0,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.zeros((1,), dtype=torch.int32, device=self.device),
            torch.zeros((1,), dtype=torch.int64, device=self.device),
        )

    def _assert_outputs_equal(
        self, expected: tuple[torch.Tensor, ...], actual: tuple[torch.Tensor, ...]
    ) -> None:
        self.assertEqual(len(expected), len(actual))
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            self.assertTrue(
                torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8)),
                f"output {index} differs",
            )

    def test_native_codegen_contract(self) -> None:
        import sglang.kernels.jit.tq_mla_frontend_n10_native as native_module
        import sglang.kernels.jit.tq_mla_frontend_n10_sm100f as s0f_module
        from sglang.kernels.jit.tq_mla_frontend_n10_native_sass import (
            audit_modules,
        )

        source_dir = Path(native_module.__file__).resolve().parent / "csrc"
        source_dir /= "tq_mla_frontend"
        summary = audit_modules(
            native_module._get_module().__file__,
            s0f_module._get_module().__file__,
            source_dir / "tq_mla_frontend_n10_native_sm100f.cu",
            source_dir / "tq_mla_frontend_n10_sm100.cu",
        )
        self.assertEqual(summary["native_specializations"], 20)
        self.assertEqual(summary["s0f_specializations"], 20)
        self.assertEqual(summary["native_total_encode"], 160)
        self.assertEqual(summary["native_total_decode"], 160)
        self.assertLessEqual(summary["native_max_registers"], 56)

    def test_s0_sm100f_and_native_full_matrix_are_byte_exact(self) -> None:
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_frontend_out,
            tq_mla_n10_frontend_rope_out,
        )
        from sglang.kernels.jit.tq_mla_frontend_n10_native import (
            tq_mla_n10_native_frontend_out,
            tq_mla_n10_native_frontend_rope_out,
        )
        from sglang.kernels.jit.tq_mla_frontend_n10_sm100f import (
            tq_mla_n10_sm100f_frontend_out,
            tq_mla_n10_sm100f_frontend_rope_out,
        )

        torch.manual_seed(0xA170)
        tokens = 7
        pool_size = 64
        query_latent = torch.randn(
            (tokens, 8, 512), dtype=torch.bfloat16, device=self.device
        )
        query_rope = torch.randn(
            (tokens, 8, 64), dtype=torch.bfloat16, device=self.device
        )
        cache_latent = torch.randn(
            (tokens * 2, 1, 512), dtype=torch.bfloat16, device=self.device
        )[::2]
        cache_latent[0].zero_()
        cache_rope = torch.randn(
            (tokens * 2, 1, 64), dtype=torch.bfloat16, device=self.device
        )[::2]
        cos_sin_cache = torch.randn(
            (256, 64), dtype=torch.float32, device=self.device
        )
        positions = torch.tensor(
            [3, 17, 31, 63, 127, 191, 255],
            dtype=torch.int64,
            device=self.device,
        )
        base_locations = [31, 1, 32, 0, 33, 17, 48]
        cfg = self.cfg

        for location_dtype in (torch.int32, torch.int64):
            locations = torch.tensor(
                base_locations, dtype=location_dtype, device=self.device
            )
            for apply_rope in (False, True):
                for rotation_fused in (False, True):
                    for strict in (False, True):
                        with self.subTest(
                            location_dtype=location_dtype,
                            apply_rope=apply_rope,
                            rotation_fused=rotation_fused,
                            strict=strict,
                        ):
                            s0 = self._outputs(tokens, pool_size)
                            s0f = self._outputs(tokens, pool_size)
                            native = self._outputs(tokens, pool_size)
                            common = (
                                query_latent,
                                query_rope,
                                cache_latent,
                                cache_rope,
                            )
                            table_args = (
                                locations,
                                cfg.signs1,
                                cfg.signs2,
                                cfg.boundaries,
                                cfg.levels,
                                cfg.codes,
                            )
                            native_args = (locations, cfg.signs1, cfg.signs2)
                            if apply_rope:
                                rope_args = (cos_sin_cache, positions)
                                tq_mla_n10_frontend_rope_out(
                                    *common,
                                    *rope_args,
                                    *table_args,
                                    *s0,
                                    grid=cfg.grid,
                                    rotation_fused=rotation_fused,
                                    strict=strict,
                                )
                                tq_mla_n10_sm100f_frontend_rope_out(
                                    *common,
                                    *rope_args,
                                    *table_args,
                                    *s0f,
                                    grid=cfg.grid,
                                    rotation_fused=rotation_fused,
                                    strict=strict,
                                )
                                tq_mla_n10_native_frontend_rope_out(
                                    *common,
                                    *rope_args,
                                    *native_args,
                                    *native,
                                    grid=cfg.grid,
                                    rotation_fused=rotation_fused,
                                    strict=strict,
                                )
                            else:
                                tq_mla_n10_frontend_out(
                                    *common,
                                    *table_args,
                                    *s0,
                                    grid=cfg.grid,
                                    rotation_fused=rotation_fused,
                                    strict=strict,
                                )
                                tq_mla_n10_sm100f_frontend_out(
                                    *common,
                                    *table_args,
                                    *s0f,
                                    grid=cfg.grid,
                                    rotation_fused=rotation_fused,
                                    strict=strict,
                                )
                                tq_mla_n10_native_frontend_out(
                                    *common,
                                    *native_args,
                                    *native,
                                    grid=cfg.grid,
                                    rotation_fused=rotation_fused,
                                    strict=strict,
                                )
                            torch.cuda.synchronize(self.device)
                            self._assert_outputs_equal(s0, s0f)
                            self._assert_outputs_equal(s0f, native)

    def test_native_faults_fail_before_write(self) -> None:
        from sglang.kernels.jit.tq_mla_frontend_n10_native import (
            tq_mla_n10_native_frontend_rope_out,
        )

        torch.manual_seed(41)
        query_latent = torch.randn(
            (1, 8, 512), dtype=torch.bfloat16, device=self.device
        )
        query_rope = torch.randn(
            (1, 8, 64), dtype=torch.bfloat16, device=self.device
        )
        cache_latent = torch.randn(
            (1, 1, 512), dtype=torch.bfloat16, device=self.device
        )
        cache_rope = torch.randn(
            (1, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        cos_sin_cache = torch.randn(
            (8, 64), dtype=torch.float32, device=self.device
        )
        cases = (
            ("location", 1, torch.tensor([8], device=self.device), None, None),
            (
                "position",
                2,
                torch.tensor([3], device=self.device),
                torch.tensor([8], device=self.device),
                None,
            ),
            (
                "query_latent",
                4,
                torch.tensor([3], device=self.device),
                None,
                "query_latent",
            ),
            (
                "query_rope",
                8,
                torch.tensor([3], device=self.device),
                None,
                "query_rope",
            ),
            (
                "cache_underflow",
                16,
                torch.tensor([3], device=self.device),
                None,
                "cache_underflow",
            ),
            (
                "cache_rope",
                32,
                torch.tensor([3], device=self.device),
                None,
                "cache_rope",
            ),
            (
                "cos_sin",
                40,
                torch.tensor([3], device=self.device),
                None,
                "cos_sin",
            ),
        )
        for name, expected_fault, locations, position_override, mutation in cases:
            with self.subTest(name=name):
                q_latent = query_latent.clone()
                q_rope = query_rope.clone()
                k_latent = cache_latent.clone()
                k_rope = cache_rope.clone()
                rope_table = cos_sin_cache.clone()
                if mutation == "query_latent":
                    q_latent[0, 0, 0] = float("nan")
                elif mutation == "query_rope":
                    q_rope[0, 0, 0] = float("nan")
                elif mutation == "cache_underflow":
                    k_latent.zero_()
                    k_latent[0, 0, 0] = torch.finfo(torch.bfloat16).tiny
                elif mutation == "cache_rope":
                    k_rope[0, 0, 0] = float("inf")
                elif mutation == "cos_sin":
                    rope_table[3, 0] = float("nan")
                positions = (
                    position_override
                    if position_override is not None
                    else torch.tensor([3], dtype=torch.int64, device=self.device)
                )
                outputs = self._outputs(1, 8)
                tq_mla_n10_native_frontend_rope_out(
                    q_latent,
                    q_rope,
                    k_latent,
                    k_rope,
                    rope_table,
                    positions,
                    locations,
                    self.cfg.signs1,
                    self.cfg.signs2,
                    *outputs,
                    grid=self.cfg.grid,
                    rotation_fused=True,
                )
                torch.cuda.synchronize(self.device)
                self.assertEqual(outputs[5].item(), expected_fault)
                self.assertEqual(outputs[6].item(), 0)
                self.assertTrue(torch.all(outputs[0] == 3.0).item())
                self.assertTrue(torch.all(outputs[1] == 3.0).item())
                self.assertTrue(torch.all(outputs[2] == 0xA5).item())
                self.assertTrue(torch.all(outputs[3] == -2.0).item())
                self.assertTrue(torch.all(outputs[4] == 3.0).item())

    def test_cache_only_graph_empty_and_differential_contract(self) -> None:
        from sglang.kernels.jit.tq_mla_frontend_n10_native import (
            tq_mla_n10_native_cache_writer_out,
        )
        from sglang.kernels.jit.tq_mla_frontend_n10_sm100f import (
            tq_mla_n10_sm100f_cache_writer_out,
        )

        cfg = self.cfg
        empty = self._outputs(0, 8)[2:]
        tq_mla_n10_native_cache_writer_out(
            torch.empty((0, 1, 512), dtype=torch.bfloat16, device=self.device),
            torch.empty((0, 1, 64), dtype=torch.bfloat16, device=self.device),
            torch.empty((0,), dtype=torch.int32, device=self.device),
            cfg.signs1,
            cfg.signs2,
            *empty,
            grid=cfg.grid,
        )
        torch.cuda.synchronize(self.device)
        self.assertTrue(torch.all(empty[0] == 0xA5).item())
        self.assertTrue(torch.all(empty[1] == -2.0).item())
        self.assertTrue(torch.all(empty[2] == 3.0).item())
        self.assertEqual(empty[3].item(), 0)
        self.assertEqual(empty[4].item(), 0)

        torch.manual_seed(0xD1FF)
        tokens = 2048
        pool_size = 4096
        cache_latent = torch.randn(
            (tokens * 2, 1, 512), dtype=torch.bfloat16, device=self.device
        )[::2]
        cache_rope = torch.randn(
            (tokens * 2, 1, 64), dtype=torch.bfloat16, device=self.device
        )[::2]
        cache_latent[0].zero_()
        cache_latent[1:33] = cache_latent[1]
        locations = torch.randperm(
            pool_size, dtype=torch.int64, device=self.device
        )[:tokens]
        s0f = self._outputs(tokens, pool_size)[2:]
        native = self._outputs(tokens, pool_size)[2:]
        tq_mla_n10_sm100f_cache_writer_out(
            cache_latent,
            cache_rope,
            locations,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *s0f,
            grid=cfg.grid,
        )

        def run_native() -> None:
            tq_mla_n10_native_cache_writer_out(
                cache_latent,
                cache_rope,
                locations,
                cfg.signs1,
                cfg.signs2,
                *native,
                grid=cfg.grid,
            )

        run_native()
        torch.cuda.synchronize(self.device)
        self._assert_outputs_equal(s0f, native)
        self.assertEqual(native[3].item(), 0)
        self.assertEqual(native[4].item(), 1)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_native()
        torch.cuda.synchronize(self.device)
        count_after_capture = native[4].item()
        memory_before = torch.cuda.memory_allocated(self.device)
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize(self.device)
        self.assertEqual(torch.cuda.memory_allocated(self.device), memory_before)
        self.assertEqual(native[4].item() - count_after_capture, 100)


if __name__ == "__main__":
    unittest.main()
