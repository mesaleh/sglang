import unittest

import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 0)


@unittest.skipUnless(_is_sm100(), "native N10 front end requires SM100")
class TestTQMLAFrontendN10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        cls.cfg = NativeE2M1MLAConfig(device=cls.device)

    def _outputs(self, tokens: int, pool_size: int):
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
                (pool_size, 1), -2.0, dtype=torch.bfloat16, device=self.device
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

    def _w5_outputs(self, tokens: int, pool_size: int):
        return (
            torch.empty(
                (tokens, 8, 576),
                dtype=torch.float8_e4m3fn,
                device=self.device,
            ),
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

    def test_fused_rope_matches_w5_and_bf16_reciprocal_oracle(self):
        import flashinfer.rope

        from sglang.kernels.jit.tq_mla_frontend import tq_mla_frontend_rope_out
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_frontend_rope_out,
        )

        torch.manual_seed(20260815)
        tokens = 5
        pool_size = 40
        query_latent = torch.randn(
            (tokens, 8, 512), dtype=torch.bfloat16, device=self.device
        )
        query_rope = torch.randn(
            (tokens, 8, 64), dtype=torch.bfloat16, device=self.device
        )
        cache_latent = torch.randn(
            (tokens, 1, 512), dtype=torch.bfloat16, device=self.device
        )
        cache_latent[0].zero_()
        cache_rope = torch.randn(
            (tokens, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        cos_sin_cache = torch.randn(
            (128, 64), dtype=torch.float32, device=self.device
        )
        positions = torch.tensor(
            [3, 17, 31, 63, 127], dtype=torch.int64, device=self.device
        )
        base_locations = [11, 3, 19, 7, 23]
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

        for location_dtype in (torch.int32, torch.int64):
            locations = torch.tensor(
                base_locations, dtype=location_dtype, device=self.device
            )
            for rotation_fused in (True, False):
                with self.subTest(
                    location_dtype=location_dtype,
                    rotation_fused=rotation_fused,
                ):
                    w5 = self._w5_outputs(tokens, pool_size)
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
                        *w5,
                        grid=cfg.grid,
                        rotation_fused=rotation_fused,
                        num_warps=9,
                    )
                    actual = self._outputs(tokens, pool_size)
                    tq_mla_n10_frontend_rope_out(
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
                        *actual,
                        grid=cfg.grid,
                        rotation_fused=rotation_fused,
                    )
                    torch.cuda.synchronize(self.device)
                    self.assertEqual(actual[5].item(), 0)
                    self.assertEqual(actual[6].item(), 1)
                    self.assertTrue(
                        torch.equal(
                            actual[0].view(torch.uint8),
                            w5[0][..., :512].view(torch.uint8),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            actual[1].view(torch.int16),
                            query_rope_ref.view(torch.int16),
                        )
                    )
                    for token, slot in enumerate(base_locations):
                        self.assertTrue(torch.equal(actual[2][slot], w5[1][slot]))
                        if token == 0:
                            self.assertEqual(actual[3][slot].item(), 1.0)
                            expected_rope = cache_rope_ref[token]
                        else:
                            self.assertTrue(
                                torch.equal(
                                    actual[3][slot].view(torch.int16),
                                    w5[2][slot].view(torch.int16),
                                )
                            )
                            expected_rope = (
                                cache_rope_ref[token].float()
                                / w5[2][slot].float()
                            ).to(torch.bfloat16)
                        self.assertTrue(
                            torch.equal(
                                actual[4][slot].view(torch.int16),
                                expected_rope.view(torch.int16),
                            )
                        )

    def test_fail_before_write_for_invalid_and_nonfinite_inputs(self):
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_frontend_rope_out,
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
        cfg = self.cfg

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
                "cache_rope",
                32,
                torch.tensor([3], device=self.device),
                None,
                "cache_rope",
            ),
            (
                "cache_underflow",
                16,
                torch.tensor([3], device=self.device),
                None,
                "cache_underflow",
            ),
            (
                "cos_sin",
                40,
                torch.tensor([3], device=self.device),
                None,
                "cos_sin",
            ),
        )
        for name, expected_fault, locations, positions_override, mutation in cases:
            with self.subTest(name=name):
                q_latent = query_latent.clone()
                q_rope = query_rope.clone()
                k_rope = cache_rope.clone()
                k_latent = cache_latent.clone()
                rope_table = cos_sin_cache.clone()
                if mutation == "query_latent":
                    q_latent[0, 0, 0] = float("nan")
                elif mutation == "query_rope":
                    q_rope[0, 0, 0] = float("nan")
                elif mutation == "cache_rope":
                    k_rope[0, 0, 0] = float("inf")
                elif mutation == "cache_underflow":
                    k_latent.zero_()
                    k_latent[0, 0, 0] = torch.finfo(torch.bfloat16).tiny
                elif mutation == "cos_sin":
                    rope_table[3, 0] = float("nan")
                positions = (
                    positions_override
                    if positions_override is not None
                    else torch.tensor([3], dtype=torch.int64, device=self.device)
                )
                outputs = self._outputs(1, 8)
                tq_mla_n10_frontend_rope_out(
                    q_latent,
                    q_rope,
                    k_latent,
                    k_rope,
                    rope_table,
                    positions,
                    locations,
                    cfg.signs1,
                    cfg.signs2,
                    cfg.boundaries,
                    cfg.levels,
                    cfg.codes,
                    *outputs,
                    grid=cfg.grid,
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

    def test_empty_and_post_rope_combined_entry(self):
        from sglang.kernels.jit.tq_mla_frontend import tq_mla_frontend_out
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_frontend_out,
        )

        cfg = self.cfg
        pool_size = 8
        empty_outputs = self._outputs(0, pool_size)
        tq_mla_n10_frontend_out(
            torch.empty((0, 8, 512), dtype=torch.bfloat16, device=self.device),
            torch.empty((0, 8, 64), dtype=torch.bfloat16, device=self.device),
            torch.empty((0, 1, 512), dtype=torch.bfloat16, device=self.device),
            torch.empty((0, 1, 64), dtype=torch.bfloat16, device=self.device),
            torch.empty((0,), dtype=torch.int32, device=self.device),
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *empty_outputs,
            grid=cfg.grid,
            rotation_fused=True,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(empty_outputs[5].item(), 0)
        self.assertEqual(empty_outputs[6].item(), 0)
        self.assertTrue(torch.all(empty_outputs[2] == 0xA5).item())
        self.assertTrue(torch.all(empty_outputs[3] == -2.0).item())
        self.assertTrue(torch.all(empty_outputs[4] == 3.0).item())

        torch.manual_seed(97)
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
        locations = torch.tensor(
            [7, 1, 6, 2, 5], dtype=torch.int32, device=self.device
        )
        w5 = self._w5_outputs(tokens, pool_size)
        tq_mla_frontend_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            locations,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *w5,
            grid=cfg.grid,
            rotation_fused=True,
            num_warps=9,
        )
        actual = self._outputs(tokens, pool_size)
        tq_mla_n10_frontend_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            locations,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *actual,
            grid=cfg.grid,
            rotation_fused=True,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(actual[5].item(), 0)
        self.assertEqual(actual[6].item(), 0)
        self.assertTrue(
            torch.equal(
                actual[0].view(torch.uint8),
                w5[0][..., :512].view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(actual[1].view(torch.int16), query_rope.view(torch.int16))
        )
        self.assertTrue(torch.equal(actual[2][locations], w5[1][locations]))
        self.assertTrue(
            torch.equal(
                actual[3][locations].view(torch.int16),
                w5[2][locations].view(torch.int16),
            )
        )
        expected_rope = (
            cache_rope.float() / w5[2][locations].unsqueeze(-1).float()
        ).to(torch.bfloat16)
        self.assertTrue(
            torch.equal(
                actual[4][locations].view(torch.int16),
                expected_rope.view(torch.int16),
            )
        )

    def test_int32_location_with_packed_destination_offset_above_2gib(self):
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_cache_writer_out,
        )

        torch.manual_seed(131)
        cfg = self.cfg
        cache_latent = torch.randn(
            (1, 1, 512), dtype=torch.bfloat16, device=self.device
        )
        cache_rope = torch.randn(
            (1, 1, 64), dtype=torch.bfloat16, device=self.device
        )

        def run_at_slot(slot: int, pool_size: int):
            packed = torch.empty(
                (pool_size, 1, 256), dtype=torch.uint8, device=self.device
            )
            scale = torch.empty(
                (pool_size, 1), dtype=torch.bfloat16, device=self.device
            )
            reciprocal_rope = torch.empty(
                (pool_size, 1, 64), dtype=torch.bfloat16, device=self.device
            )
            fault = torch.zeros((1,), dtype=torch.int32, device=self.device)
            zero_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
            locations = torch.tensor([slot], dtype=torch.int32, device=self.device)
            tq_mla_n10_cache_writer_out(
                cache_latent,
                cache_rope,
                locations,
                cfg.signs1,
                cfg.signs2,
                cfg.boundaries,
                cfg.levels,
                cfg.codes,
                packed,
                scale,
                reciprocal_rope,
                fault,
                zero_count,
                grid=cfg.grid,
            )
            return packed, scale, reciprocal_rope, fault, zero_count

        reference = run_at_slot(0, 1)
        large_slot = (2**31 // 256) + 1
        self.assertGreater(large_slot * 256, 2**31)
        actual = run_at_slot(large_slot, large_slot + 1)
        torch.cuda.synchronize(self.device)
        self.assertEqual(reference[3].item(), 0)
        self.assertEqual(actual[3].item(), 0)
        self.assertEqual(actual[4].item(), 0)
        self.assertTrue(torch.equal(actual[0][large_slot], reference[0][0]))
        self.assertTrue(
            torch.equal(
                actual[1][large_slot].view(torch.int16),
                reference[1][0].view(torch.int16),
            )
        )
        self.assertTrue(
            torch.equal(
                actual[2][large_slot].view(torch.int16),
                reference[2][0].view(torch.int16),
            )
        )

    def test_all_target_layers_and_draft_output_isolation(self):
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_cache_writer_out,
        )

        torch.manual_seed(167)
        cfg = self.cfg
        pool_size = 4
        location = torch.tensor([2], dtype=torch.int64, device=self.device)
        cache_rope = torch.randn(
            (1, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        target_layers = [self._outputs(1, pool_size)[2:] for _ in range(61)]
        draft = self._outputs(1, pool_size)[2:]

        for layer_id, outputs in enumerate(target_layers):
            cache_latent = torch.full(
                (1, 1, 512),
                (layer_id + 1) / 128.0,
                dtype=torch.bfloat16,
                device=self.device,
            )
            tq_mla_n10_cache_writer_out(
                cache_latent,
                cache_rope,
                location,
                cfg.signs1,
                cfg.signs2,
                cfg.boundaries,
                cfg.levels,
                cfg.codes,
                *outputs,
                grid=cfg.grid,
            )

        torch.cuda.synchronize(self.device)
        for outputs in target_layers:
            self.assertEqual(outputs[3].item(), 0)
            self.assertEqual(outputs[4].item(), 0)
            self.assertTrue(torch.all(outputs[0][0] == 0xA5).item())
            self.assertTrue(torch.all(outputs[0][1] == 0xA5).item())
            self.assertTrue(torch.all(outputs[0][3] == 0xA5).item())
            self.assertTrue(torch.all(outputs[1][0] == -2.0).item())
            self.assertTrue(torch.all(outputs[2][0] == 3.0).item())
        self.assertTrue(
            any(
                not torch.equal(
                    target_layers[index][1][location].view(torch.int16),
                    target_layers[index + 1][1][location].view(torch.int16),
                )
                for index in range(60)
            )
        )
        self.assertTrue(torch.all(draft[0] == 0xA5).item())
        self.assertTrue(torch.all(draft[1] == -2.0).item())
        self.assertTrue(torch.all(draft[2] == 3.0).item())
        self.assertEqual(draft[3].item(), 0)
        self.assertEqual(draft[4].item(), 0)

    def test_cache_only_zero_contract_and_graph_counter(self):
        from sglang.kernels.jit.tq_mla_frontend import tq_mla_cache_writer_out
        from sglang.kernels.jit.tq_mla_frontend_n10 import (
            tq_mla_n10_cache_writer_out,
        )

        torch.manual_seed(73)
        tokens = 5
        pool_size = 40
        cache_latent = torch.randn(
            (tokens * 2, 1, 512), dtype=torch.bfloat16, device=self.device
        )[::2]
        cache_rope = torch.randn(
            (tokens * 2, 1, 64), dtype=torch.bfloat16, device=self.device
        )[::2]
        cache_latent[0].zero_()
        locations = torch.tensor(
            [31, 1, 32, 0, 33], dtype=torch.int64, device=self.device
        )
        cfg = self.cfg
        w5 = self._w5_outputs(tokens, pool_size)[1:]
        tq_mla_cache_writer_out(
            cache_latent,
            cache_rope,
            locations,
            cfg.signs1,
            cfg.signs2,
            cfg.boundaries,
            cfg.levels,
            cfg.codes,
            *w5,
            grid=cfg.grid,
        )
        outputs = self._outputs(tokens, pool_size)[2:]

        def run() -> None:
            tq_mla_n10_cache_writer_out(
                cache_latent,
                cache_rope,
                locations,
                cfg.signs1,
                cfg.signs2,
                cfg.boundaries,
                cfg.levels,
                cfg.codes,
                *outputs,
                grid=cfg.grid,
            )

        run()
        torch.cuda.synchronize(self.device)
        self.assertEqual(outputs[3].item(), 0)
        self.assertEqual(outputs[4].item(), 1)
        self.assertTrue(torch.equal(outputs[0][locations], w5[0][locations]))
        for token, slot in enumerate(locations.tolist()):
            if token == 0:
                self.assertEqual(outputs[1][slot].item(), 1.0)
                expected_rope = cache_rope[token]
            else:
                self.assertTrue(
                    torch.equal(
                        outputs[1][slot].view(torch.int16),
                        w5[1][slot].view(torch.int16),
                    )
                )
                expected_rope = (
                    cache_rope[token].float() / w5[1][slot].float()
                ).to(torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    outputs[2][slot].view(torch.int16),
                    expected_rope.view(torch.int16),
                )
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize(self.device)
        count_after_capture = outputs[4].item()
        memory_before = torch.cuda.memory_allocated(self.device)
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize(self.device)
        memory_after = torch.cuda.memory_allocated(self.device)
        self.assertEqual(memory_after, memory_before)
        self.assertEqual(outputs[4].item() - count_after_capture, 100)


if __name__ == "__main__":
    unittest.main()
