from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 0)


@unittest.skipUnless(_is_sm100(), "native N10 prefix gather requires SM100")
class TestTQMLAPrefixN10Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from sglang.srt.layers.quantization.kv_turboquant import (
            NativeE2M1MLAConfig,
        )

        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        cls.cfg = NativeE2M1MLAConfig(device=cls.device)
        # Numeric value indexed by the physical hardware E2M1 nibble.
        cls.level_by_code = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=cls.device,
        )

    def _reference(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        reciprocal_rope: torch.Tensor,
        locations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.kernels.ops.quantization.hadamard import (
            hadamard_transform_with_signs,
        )

        selected = packed[locations.long()]
        codes = torch.stack((selected & 0x0F, (selected >> 4) & 0x0F), dim=-1).reshape(
            locations.numel(), 1, 512
        )
        selected_scale = scale[locations.long()].float()
        rotated = self.level_by_code[codes.long()] * selected_scale.unsqueeze(-1)
        latent = hadamard_transform_with_signs(
            rotated.contiguous(),
            self.cfg.signs2,
            self.cfg.signs1,
            scale=1.0 / math.sqrt(512),
        ).to(torch.bfloat16)
        rope = (
            reciprocal_rope[locations.long()].float() * selected_scale.unsqueeze(-1)
        ).to(torch.bfloat16)
        return latent, rope

    def _planar_output(
        self, capacity: int, fill_value: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = torch.full(
            (capacity * 576,),
            fill_value,
            dtype=torch.bfloat16,
            device=self.device,
        )
        latent_elements = capacity * 512
        latent = output[:latent_elements].view(capacity, 1, 512)
        rope = output[latent_elements:].view(capacity, 1, 64)
        return output, latent, rope

    def _run(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        reciprocal_rope: torch.Tensor,
        locations: torch.Tensor,
        output: torch.Tensor,
        validation_status: torch.Tensor,
        sticky_fault_status: torch.Tensor,
    ) -> None:
        from sglang.kernels.jit.tq_mla_prefix_n10_native import (
            tq_mla_n10_prefix_gather_out,
        )

        tq_mla_n10_prefix_gather_out(
            packed,
            scale,
            reciprocal_rope,
            locations,
            self.cfg.signs1,
            self.cfg.signs2,
            output,
            validation_status,
            sticky_fault_status,
        )

    def test_every_code_permuted_duplicate_and_page_boundary_rows(self) -> None:
        torch.manual_seed(0xE0A17)
        pool_size = 96
        capacity = 32
        packed = torch.randint(
            0,
            256,
            (pool_size, 1, 256),
            dtype=torch.uint8,
            device=self.device,
        )
        # Force a row that contains every physical nibble repeatedly, including
        # negative zero (code 8), and a writer-compatible zero row.
        nibbles = torch.arange(16, dtype=torch.uint8, device=self.device)
        code_row = nibbles.repeat(32)
        packed[31, 0] = code_row[0::2] | (code_row[1::2] << 4)
        packed[0].zero_()
        scale = (
            torch.rand((pool_size, 1), dtype=torch.bfloat16, device=self.device) + 0.5
        )
        scale[0] = 1.0
        reciprocal_rope = torch.randn(
            (pool_size, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        locations_list = [31, 0, 32, 33, 95, 1, 31, 64, 63]

        for location_dtype in (torch.int32, torch.int64):
            with self.subTest(location_dtype=location_dtype):
                locations = torch.tensor(
                    locations_list, dtype=location_dtype, device=self.device
                )
                output, output_latent, output_rope = self._planar_output(
                    capacity, 7.0
                )
                validation = torch.full(
                    (1,), 123, dtype=torch.int32, device=self.device
                )
                sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)
                self._run(
                    packed,
                    scale,
                    reciprocal_rope,
                    locations,
                    output,
                    validation,
                    sticky,
                )
                torch.cuda.synchronize(self.device)
                expected_latent, expected_rope = self._reference(
                    packed, scale, reciprocal_rope, locations
                )
                torch.testing.assert_close(
                    output_latent[: locations.numel()],
                    expected_latent,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    output_rope[: locations.numel()],
                    expected_rope,
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(
                    torch.all(output_latent[locations.numel() :] == 7.0).item()
                )
                self.assertTrue(
                    torch.all(output_rope[locations.numel() :] == 7.0).item()
                )
                self.assertEqual(validation.item(), 0)
                self.assertEqual(sticky.item(), 0)

    def test_invalid_location_fails_before_any_output_write(self) -> None:
        pool_size = 8
        packed = torch.zeros((pool_size, 1, 256), dtype=torch.uint8, device=self.device)
        scale = torch.ones((pool_size, 1), dtype=torch.bfloat16, device=self.device)
        reciprocal_rope = torch.randn(
            (pool_size, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        output, _, _ = self._planar_output(4, 3.0)
        before = output.clone()
        validation = torch.zeros((1,), dtype=torch.int32, device=self.device)
        sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)
        locations = torch.tensor([1, pool_size, 2], device=self.device)

        self._run(
            packed,
            scale,
            reciprocal_rope,
            locations,
            output,
            validation,
            sticky,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(validation.item(), 1)
        self.assertEqual(sticky.item(), 1)
        self.assertTrue(torch.equal(output.view(torch.uint8), before.view(torch.uint8)))

    def test_sanitizer_candidate_only(self) -> None:
        pool_size = 64
        packed = torch.randint(
            0,
            256,
            (pool_size, 1, 256),
            dtype=torch.uint8,
            device=self.device,
        )
        scale = torch.ones((pool_size, 1), dtype=torch.bfloat16, device=self.device)
        reciprocal_rope = torch.randn(
            (pool_size, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        output, _, _ = self._planar_output(8, 9.0)
        validation = torch.zeros((1,), dtype=torch.int32, device=self.device)
        sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)
        valid_locations = torch.tensor(
            [0, 31, 32, 63, 31], dtype=torch.int32, device=self.device
        )
        self._run(
            packed,
            scale,
            reciprocal_rope,
            valid_locations,
            output,
            validation,
            sticky,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(validation.item(), 0)
        self.assertEqual(sticky.item(), 0)

        before = output.clone()
        invalid_locations = torch.tensor(
            [1, pool_size, 2], dtype=torch.int64, device=self.device
        )
        self._run(
            packed,
            scale,
            reciprocal_rope,
            invalid_locations,
            output,
            validation,
            sticky,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(validation.item(), 1)
        self.assertEqual(sticky.item(), 1)
        self.assertTrue(torch.equal(output.view(torch.uint8), before.view(torch.uint8)))

    def test_empty_and_graph100_are_allocation_and_pointer_stable(self) -> None:
        torch.manual_seed(0x100)
        pool_size = 64
        capacity = 16
        packed = torch.randint(
            0,
            256,
            (pool_size, 1, 256),
            dtype=torch.uint8,
            device=self.device,
        )
        scale = (
            torch.rand((pool_size, 1), dtype=torch.bfloat16, device=self.device) + 0.25
        )
        reciprocal_rope = torch.randn(
            (pool_size, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        output, output_latent, output_rope = self._planar_output(capacity, -5.0)
        validation = torch.full((1,), 7, dtype=torch.int32, device=self.device)
        sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)

        self._run(
            packed,
            scale,
            reciprocal_rope,
            torch.empty((0,), dtype=torch.int32, device=self.device),
            output,
            validation,
            sticky,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(validation.item(), 0)
        self.assertTrue(torch.all(output == -5.0).item())

        locations = torch.tensor(
            [0, 1, 31, 32, 33, 63, 31], dtype=torch.int32, device=self.device
        )

        def run() -> None:
            self._run(
                packed,
                scale,
                reciprocal_rope,
                locations,
                output,
                validation,
                sticky,
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize(self.device)
        expected_latent = output_latent[: locations.numel()].clone()
        expected_rope = output_rope[: locations.numel()].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        torch.cuda.synchronize(self.device)
        pointer = output.data_ptr()
        allocated = torch.cuda.memory_allocated(self.device)
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize(self.device)
        self.assertEqual(output.data_ptr(), pointer)
        self.assertEqual(torch.cuda.memory_allocated(self.device), allocated)
        torch.testing.assert_close(
            output_latent[: locations.numel()], expected_latent, rtol=0, atol=0
        )
        torch.testing.assert_close(
            output_rope[: locations.numel()], expected_rope, rtol=0, atol=0
        )
        self.assertEqual(validation.item(), 0)
        self.assertEqual(sticky.item(), 0)

    def test_output_alias_is_rejected_before_launch(self) -> None:
        backing = torch.empty((8 * 576,), dtype=torch.bfloat16, device=self.device)
        packed = backing.view(torch.uint8).view(36, 1, 256)
        scale = torch.ones((36, 1), dtype=torch.bfloat16, device=self.device)
        reciprocal_rope = torch.zeros(
            (36, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        validation = torch.zeros((1,), dtype=torch.int32, device=self.device)
        sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "must not overlap packed_cache"):
            self._run(
                packed,
                scale,
                reciprocal_rope,
                torch.empty((0,), dtype=torch.int32, device=self.device),
                backing,
                validation,
                sticky,
            )

    def test_backend_returns_only_bounded_scratch_views(self) -> None:
        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            TokenspeedTQE2M1MLABackend,
        )

        pool_size = 64
        capacity = 8
        packed = torch.zeros((pool_size, 1, 256), dtype=torch.uint8, device=self.device)
        scale = torch.ones((pool_size, 1), dtype=torch.bfloat16, device=self.device)
        reciprocal_rope = torch.randn(
            (pool_size, 1, 64), dtype=torch.bfloat16, device=self.device
        )
        sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)

        buffers = type("Buffers", (), {})()
        buffers.packed_nope = packed
        buffers.nope_scale = scale
        buffers.reciprocal_rope = reciprocal_rope

        class Pool:
            tq_config = self.cfg

            def get_native_e2m1_recip_bf16_buffers(self, layer_id):
                self.layer_id = layer_id
                return buffers

            def get_native_e2m1_fault_status(self):
                return sticky

        backend = object.__new__(TokenspeedTQE2M1MLABackend)
        backend.token_to_kv_pool = Pool()
        backend._n10_prefix_capacity = capacity
        backend._n10_prefix_scratch = torch.empty(
            (capacity * 576,), dtype=torch.bfloat16, device=self.device
        )
        latent_elements = capacity * 512
        backend._n10_prefix_latent_scratch = backend._n10_prefix_scratch[
            :latent_elements
        ].view(capacity, 1, 512)
        backend._n10_prefix_rope_scratch = backend._n10_prefix_scratch[
            latent_elements:
        ].view(capacity, 1, 64)
        backend._n10_prefix_validation_status = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        locations = torch.tensor([31, 32, 31], dtype=torch.int32, device=self.device)
        latent, rope = backend.gather_prefix_chunk_latent(
            layer=type("Layer", (), {"layer_id": 5})(),
            kv_indices=locations,
            dst_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize(self.device)
        self.assertEqual(tuple(latent.shape), (3, 1, 512))
        self.assertEqual(tuple(rope.shape), (3, 1, 64))
        self.assertTrue(latent.is_contiguous())
        self.assertTrue(rope.is_contiguous())
        self.assertEqual(latent.stride(), (512, 512, 1))
        self.assertEqual(rope.stride(), (64, 64, 1))
        self.assertEqual(
            latent.untyped_storage().data_ptr(), rope.untyped_storage().data_ptr()
        )
        self.assertEqual(
            backend.get_n10_prefix_workspace_size_bytes(), capacity * 576 * 2 + 4
        )
        self.assertEqual(backend.token_to_kv_pool.layer_id, 5)
        with self.assertRaisesRegex(RuntimeError, "exceeds.*capacity"):
            backend.gather_prefix_chunk_latent(
                layer=type("Layer", (), {"layer_id": 5})(),
                kv_indices=torch.arange(
                    capacity + 1, dtype=torch.int32, device=self.device
                ),
                dst_dtype=torch.bfloat16,
            )

    def test_real_projection_pack_attention_and_chunk_merge_chain(self) -> None:
        import tokenspeed_mla
        import torch.nn.functional as F
        from sgl_kernel import merge_state_v2

        from sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8 import (
            mla_kv_pack_quantize_fp8,
        )

        torch.manual_seed(0xA17E0)
        pool_size = 96
        kv_lens = [33, 17]
        q_lens = [5, 3]
        total_kv = sum(kv_lens)
        total_q = sum(q_lens)
        num_heads = 8
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        packed = torch.randint(
            0,
            256,
            (pool_size, 1, 256),
            dtype=torch.uint8,
            device=self.device,
        )
        scale = (
            torch.rand((pool_size, 1), dtype=torch.bfloat16, device=self.device) + 0.25
        )
        reciprocal_rope = torch.randn(
            (pool_size, 1, qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        locations = torch.randperm(pool_size, device=self.device)[:total_kv].to(
            torch.int32
        )
        scratch, scratch_latent, scratch_rope = self._planar_output(total_kv, 0.0)
        validation = torch.zeros((1,), dtype=torch.int32, device=self.device)
        sticky = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self._run(
            packed,
            scale,
            reciprocal_rope,
            locations,
            scratch,
            validation,
            sticky,
        )
        expected_latent, expected_rope = self._reference(
            packed, scale, reciprocal_rope, locations
        )

        projection_weight = (
            torch.randn(
                (
                    num_heads * (qk_nope_head_dim + v_head_dim),
                    512,
                ),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.04
        )

        def project_and_pack(
            latent: torch.Tensor, rope: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            projected = F.linear(latent.squeeze(1), projection_weight).view(
                total_kv, num_heads, qk_nope_head_dim + v_head_dim
            )
            k_nope = projected[..., :qk_nope_head_dim]
            value = projected[..., qk_nope_head_dim:]
            return mla_kv_pack_quantize_fp8(k_nope, rope, value)

        candidate_k, candidate_v = project_and_pack(scratch_latent, scratch_rope)
        reference_k, reference_v = project_and_pack(expected_latent, expected_rope)
        torch.cuda.synchronize(self.device)
        self.assertTrue(
            torch.equal(candidate_k.view(torch.uint8), reference_k.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(candidate_v.view(torch.uint8), reference_v.view(torch.uint8))
        )

        query = torch.randn(
            (total_q, num_heads, qk_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        cum_q = torch.tensor(
            [0, q_lens[0], total_q], dtype=torch.int32, device=self.device
        )
        cum_kv = torch.tensor(
            [0, kv_lens[0], total_kv], dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=self.device)

        def run_attention(
            key: torch.Tensor,
            value: torch.Tensor,
            chunk_lens: list[int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cum_chunk = torch.tensor(
                [0, chunk_lens[0], sum(chunk_lens)],
                dtype=torch.int32,
                device=self.device,
            )
            return tokenspeed_mla.tokenspeed_mla_prefill(
                query=query,
                key=key,
                value=value,
                seq_lens=torch.tensor(
                    chunk_lens, dtype=torch.int32, device=self.device
                ),
                cum_seq_lens=cum_chunk,
                max_seq_len=max(chunk_lens),
                batch_size=len(chunk_lens),
                softmax_scale=1.0 / math.sqrt(qk_head_dim),
                is_causal=False,
                return_lse=True,
                cum_seq_lens_q=cum_q,
                max_seq_len_q=max(q_lens),
                enable_pdl=True,
            )

        candidate_output, candidate_lse = tokenspeed_mla.tokenspeed_mla_prefill(
            query=query,
            key=candidate_k,
            value=candidate_v,
            seq_lens=seq_lens,
            cum_seq_lens=cum_kv,
            max_seq_len=max(kv_lens),
            batch_size=len(kv_lens),
            softmax_scale=1.0 / math.sqrt(qk_head_dim),
            is_causal=False,
            return_lse=True,
            cum_seq_lens_q=cum_q,
            max_seq_len_q=max(q_lens),
            enable_pdl=True,
        )
        reference_output, reference_lse = tokenspeed_mla.tokenspeed_mla_prefill(
            query=query,
            key=reference_k,
            value=reference_v,
            seq_lens=seq_lens,
            cum_seq_lens=cum_kv,
            max_seq_len=max(kv_lens),
            batch_size=len(kv_lens),
            softmax_scale=1.0 / math.sqrt(qk_head_dim),
            is_causal=False,
            return_lse=True,
            cum_seq_lens_q=cum_q,
            max_seq_len_q=max(q_lens),
            enable_pdl=True,
        )
        torch.testing.assert_close(candidate_output, reference_output, rtol=0, atol=0)
        torch.testing.assert_close(candidate_lse, reference_lse, rtol=0, atol=0)

        first_indices = torch.cat(
            (
                torch.arange(0, 17, device=self.device),
                torch.arange(kv_lens[0], kv_lens[0] + 9, device=self.device),
            )
        )
        second_indices = torch.cat(
            (
                torch.arange(17, kv_lens[0], device=self.device),
                torch.arange(kv_lens[0] + 9, total_kv, device=self.device),
            )
        )
        first_output, first_lse = run_attention(
            candidate_k[first_indices], candidate_v[first_indices], [17, 9]
        )
        second_output, second_lse = run_attention(
            candidate_k[second_indices], candidate_v[second_indices], [16, 8]
        )
        merged_output = torch.empty_like(candidate_output)
        merged_lse = torch.empty_like(candidate_lse)
        merge_state_v2(
            first_output,
            first_lse,
            second_output,
            second_lse,
            merged_output,
            merged_lse,
        )
        torch.cuda.synchronize(self.device)
        output_delta = (merged_output.float() - candidate_output.float()).abs()
        self.assertLessEqual(output_delta.max().item(), 0.125)
        self.assertLessEqual(output_delta.mean().item(), 0.01)
        torch.testing.assert_close(merged_lse, candidate_lse, rtol=1e-5, atol=1e-5)
        self.assertEqual(validation.item(), 0)
        self.assertEqual(sticky.item(), 0)

    def test_chunked_prefix_prefers_explicit_backend_gather(self) -> None:
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
            DeepseekMHAForwardMixin,
        )

        calls: list[tuple[object, torch.Tensor, torch.dtype]] = []

        class DummyAttention:
            layer_id = 17

            def __call__(self, q, k, v, forward_batch, **kwargs):
                self.key_value_num_tokens = kwargs["key_value_num_tokens"]
                return torch.full((1, 1, 1), 4.0), torch.full((1, 1), 5.0)

        class DummyProjection:
            def __call__(self, kv_a):
                if not kv_a.is_contiguous():
                    raise AssertionError("FP8 kv_b_proj requires contiguous activation")
                self.input = kv_a
                return torch.zeros((kv_a.shape[0], 1, 2)), None

        attention = DummyAttention()
        projection = DummyProjection()
        model = object.__new__(DeepseekMHAForwardMixin)
        model.attn_mha = attention
        model.kv_b_proj = projection
        model.num_local_heads = 1
        model.qk_nope_head_dim = 1
        model.v_head_dim = 1
        model._get_mla_kv_buffer = lambda *args, **kwargs: self.fail(
            "generic dense gather must remain unreachable"
        )
        dispatch_scratch = torch.zeros((576,), dtype=torch.bfloat16)

        def gather(*, layer, kv_indices, dst_dtype):
            calls.append((layer, kv_indices, dst_dtype))
            return (
                dispatch_scratch[:512].view(1, 1, 512),
                dispatch_scratch[512:].view(1, 1, 64),
            )

        backend = SimpleNamespace(
            gather_prefix_chunk_latent=gather,
            pack_prefix_chunk_kv=lambda k_nope, k_pe, v: (
                torch.zeros((1, 1, 65)),
                v,
            ),
        )
        forward_batch = SimpleNamespace(
            num_prefix_chunks=1,
            prefix_chunk_kv_indices=[torch.tensor([31], dtype=torch.int32)],
            prefix_chunk_seq_lens_cpu=[torch.tensor([1], dtype=torch.int32)],
            prefix_chunk_starts_cpu=[torch.tensor([0], dtype=torch.int32)],
            set_prefix_chunk_idx=lambda index: setattr(
                forward_batch, "prefix_chunk_idx", index
            ),
        )

        def merge(output, lse, accum_output, accum_lse, tmp_output, tmp_lse):
            tmp_output.copy_(output)
            tmp_lse.copy_(lse)

        module = (
            "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha"
        )
        with (
            patch(f"{module}._resolve_attn_backend", return_value=backend),
            patch(
                f"{module}.all_gather_kv_cache_for_mha_chunk_extend",
                side_effect=AssertionError(
                    "typed N10 gather must bypass the allocating DCP helper"
                ),
            ),
            patch(f"{module}.merge_state_v2", side_effect=merge),
        ):
            output = model._chunked_prefix_attn_mha(
                q=torch.zeros((1, 1, 65)),
                accum_output=torch.zeros((1, 1, 1)),
                accum_lse=torch.zeros((1, 1)),
                forward_batch=forward_batch,
            )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], attention)
        self.assertEqual(calls[0][2], torch.bfloat16)
        self.assertEqual(tuple(projection.input.shape), (1, 512))
        self.assertEqual(attention.key_value_num_tokens, 1)
        torch.testing.assert_close(output, torch.full((1, 1, 1), 4.0))


if __name__ == "__main__":
    unittest.main()
