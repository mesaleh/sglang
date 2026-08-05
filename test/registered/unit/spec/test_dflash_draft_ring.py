import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_draft_ring import (
    actual_dflash_ring_live_page_span,
    build_dflash_draft_ring_config,
    compact_draft_seq_len,
    configured_dflash_draft_ring_reprefill_tail_tokens,
    dflash_draft_ring_reprefill_tail_tokens,
    draft_ring_cache_locs,
    select_dflash_ring_prefill_slices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestDFlashDraftRing(unittest.TestCase):
    def setUp(self):
        self.config = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )

    def test_production_capacity_has_a_full_guard_page(self):
        cfg = self.config
        self.assertEqual(cfg.max_compact_len, 2079)
        self.assertEqual(cfg.max_live_pages, 66)
        self.assertEqual(cfg.ring_pages, 67)
        self.assertEqual(cfg.row_stride, 2144)
        self.assertEqual(cfg.physical_tokens, 19296)
        self.assertEqual(cfg.padded_tokens, 19328)

        observed = max(
            actual_dflash_ring_live_page_span(
                prefix_len=prefix_len,
                window_size=cfg.window_size,
                page_size=cfg.page_size,
                block_size=cfg.block_size,
            )
            for prefix_len in range(0, 256_001)
        )
        self.assertLessEqual(observed, cfg.max_live_pages)
        self.assertEqual(cfg.ring_pages - observed, 1)

    def test_ring_mapping_preserves_pages_and_separates_requests(self):
        cfg = self.config
        positions = torch.arange(0, cfg.row_stride, dtype=torch.int64).view(1, -1)
        reqs = torch.tensor([0, 1, 8], dtype=torch.int64)
        locs = draft_ring_cache_locs(reqs, positions, cfg)

        self.assertEqual(tuple(locs.shape), (3, cfg.row_stride))
        self.assertEqual(int(locs.min()), cfg.page_size)
        self.assertEqual(int(locs.max()), cfg.padded_tokens - 1)
        for row in range(3):
            self.assertEqual(int(torch.unique(locs[row]).numel()), cfg.row_stride)
        self.assertFalse(bool(torch.isin(locs[0], locs[1]).any()))
        self.assertFalse(bool(torch.isin(locs[1], locs[2]).any()))

        wrapped = draft_ring_cache_locs(
            torch.tensor([1]),
            torch.tensor([7, cfg.ring_pages * cfg.page_size + 7]),
            cfg,
        )
        self.assertEqual(int(wrapped[0]), int(wrapped[1]))
        self.assertEqual(int(wrapped[0] % cfg.page_size), 7)

    def test_prefill_selects_only_the_visible_new_suffix(self):
        slices = select_dflash_ring_prefill_slices(
            prefix_lens=[0, 16384, 8139],
            extend_lens=[10218, 100, 2079],
            window_size=2048,
            page_size=32,
        )
        self.assertEqual(slices[0].absolute_start, 8160)
        self.assertEqual(slices[0].absolute_end, 10218)
        self.assertEqual(slices[0].length, 2058)
        self.assertEqual(slices[1].length, 100)
        self.assertEqual(slices[2].absolute_start, 8160)
        self.assertEqual(slices[2].absolute_end, 10218)
        self.assertEqual(slices[2].length, 2058)

        # The flattened hidden-state offsets account for preceding sequences.
        self.assertEqual(slices[1].flat_start, 10218)
        self.assertEqual(slices[1].flat_end, 10318)
        self.assertEqual(slices[2].flat_start, 10339)
        self.assertEqual(slices[2].flat_end, 12397)

    def test_long_chunk_has_unique_ring_destinations(self):
        cfg = self.config
        (selected,) = select_dflash_ring_prefill_slices(
            prefix_lens=[0],
            extend_lens=[65536],
            window_size=cfg.window_size,
            page_size=cfg.page_size,
        )
        positions = torch.arange(
            selected.absolute_start, selected.absolute_end, dtype=torch.int64
        )
        locs = draft_ring_cache_locs(torch.tensor([3]), positions, cfg)
        self.assertLessEqual(selected.length, cfg.max_compact_len)
        self.assertEqual(int(torch.unique(locs).numel()), selected.length)

    def test_prefill_suffix_is_unique_across_page_residues(self):
        cfg = self.config
        extend_lengths = (0, 1, 31, 32, 33, 2048, 2079, 4096, 65536)
        for prefix_residue in range(cfg.page_size):
            prefix_len = 8192 + prefix_residue
            for extend_len in extend_lengths:
                with self.subTest(prefix_residue=prefix_residue, extend_len=extend_len):
                    (selected,) = select_dflash_ring_prefill_slices(
                        prefix_lens=[prefix_len],
                        extend_lens=[extend_len],
                        window_size=cfg.window_size,
                        page_size=cfg.page_size,
                    )
                    positions = torch.arange(
                        selected.absolute_start,
                        selected.absolute_end,
                        dtype=torch.int64,
                    )
                    locs = draft_ring_cache_locs(torch.tensor([4]), positions, cfg)
                    self.assertLessEqual(selected.length, cfg.max_compact_len)
                    self.assertEqual(int(torch.unique(locs).numel()), selected.length)
                    self.assertEqual(selected.absolute_end, prefix_len + extend_len)
                    self.assertGreaterEqual(selected.absolute_start, prefix_len)

    def test_prefix_replay_and_invalid_inputs(self):
        self.assertEqual(dflash_draft_ring_reprefill_tail_tokens(2048, 32), 2079)
        self.assertEqual(compact_draft_seq_len(10218, 2048, 32), 2058)
        with self.assertRaisesRegex(ValueError, "alloc_reserve"):
            build_dflash_draft_ring_config(
                window_size=2048,
                page_size=32,
                block_size=5,
                alloc_reserve=4,
                request_rows=9,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            draft_ring_cache_locs(torch.tensor([9]), torch.tensor([0]), self.config)

    def test_reprefill_holdback_is_strictly_gate_and_algorithm_scoped(self):
        from sglang.srt.environ import envs

        dflash = SimpleNamespace(
            speculative_algorithm="DFLASH",
            speculative_draft_window_size=2048,
            page_size=32,
        )
        eagle = SimpleNamespace(
            speculative_algorithm="EAGLE",
            speculative_draft_window_size=2048,
            page_size=32,
        )
        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_RING.override(False):
            self.assertEqual(
                configured_dflash_draft_ring_reprefill_tail_tokens(dflash), 0
            )
        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_RING.override(True):
            self.assertEqual(
                configured_dflash_draft_ring_reprefill_tail_tokens(eagle), 0
            )
            self.assertEqual(
                configured_dflash_draft_ring_reprefill_tail_tokens(dflash), 2079
            )

    def test_session_request_is_rejected_at_scheduler_admission(self):
        from sglang.srt.environ import envs
        from sglang.srt.speculative.dflash_utils import validate_dflash_request

        req = SimpleNamespace(
            return_logprob=False,
            return_hidden_states=False,
            session=object(),
            sampling_params=SimpleNamespace(
                json_schema=None,
                regex=None,
                ebnf=None,
                structural_tag=None,
            ),
        )
        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_RING.override(False):
            self.assertIsNone(validate_dflash_request(req, enable_overlap=True))
        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_RING.override(True):
            self.assertIn(
                "does not yet support session requests",
                validate_dflash_request(req, enable_overlap=True),
            )

        req.session = None
        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_RING.override(True):
            self.assertIsNone(validate_dflash_request(req, enable_overlap=True))

    def test_worker_allocates_only_the_proven_physical_draft_span(self):
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        draft_runner = SimpleNamespace()
        captured = {}

        class FakeDraftWorker:
            def alloc_memory_pool(
                self,
                *,
                memory_pool_config,
                req_to_token_pool,
                token_to_kv_pool_allocator,
            ):
                captured["config"] = memory_pool_config
                captured["req_to_token_pool"] = req_to_token_pool
                captured["allocator"] = token_to_kv_pool_allocator
                draft_runner.req_to_token_pool = SimpleNamespace(_alloc_size=9)
                draft_runner.token_to_kv_pool = SimpleNamespace(
                    size=19296,
                    # MLA-family pools expose one packed byte count rather than
                    # the MHA pool's (K, V) tuple.
                    get_kv_size_bytes=lambda: 2048,
                )

        fake_worker = SimpleNamespace(
            use_physical_draft_ring=True,
            use_compact_draft_cache=True,
            draft_window_size=2048,
            page_size=32,
            block_size=5,
            server_args=SimpleNamespace(
                speculative_algorithm="DFLASH",
                speculative_num_steps=1,
                speculative_eagle_topk=1,
                max_speculative_num_draft_tokens=5,
                page_size=32,
            ),
            _draft_worker=FakeDraftWorker(),
            draft_model_runner=draft_runner,
            _draft_ring_config=None,
            ps=SimpleNamespace(tp_rank=1),
        )
        target_pool = SimpleNamespace(_alloc_size=9)
        DFlashWorkerV2.alloc_memory_pool(
            fake_worker,
            memory_pool_config=MemoryPoolConfig(
                max_total_num_tokens=262144,
                max_running_requests=8,
                full_max_total_num_tokens=200000,
                swa_max_total_num_tokens=100000,
                c4_max_total_num_tokens=1,
                c128_max_total_num_tokens=2,
                c4_state_pool_size=3,
                c128_state_pool_size=4,
            ),
            req_to_token_pool=target_pool,
            token_to_kv_pool_allocator=object(),
        )
        config = captured["config"]
        self.assertEqual(config.max_total_num_tokens, 19296)
        self.assertEqual(config.max_running_requests, 8)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)
        self.assertEqual(config.c4_max_total_num_tokens, 0)
        self.assertIsNone(captured["req_to_token_pool"])
        self.assertIsNone(captured["allocator"])
        self.assertEqual(fake_worker._draft_ring_config.padded_tokens, 19328)

    def test_worker_default_off_preserves_allocator_delegation(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        memory_pool_config = object()
        target_req_pool = object()
        target_allocator = object()
        for compact, expected_req_pool in (
            (False, target_req_pool),
            (True, None),
        ):
            with self.subTest(compact=compact):
                captured = {}

                class FakeDraftWorker:
                    def alloc_memory_pool(self, **kwargs):
                        captured.update(kwargs)

                fake_worker = SimpleNamespace(
                    use_physical_draft_ring=False,
                    use_compact_draft_cache=compact,
                    _draft_worker=FakeDraftWorker(),
                )
                DFlashWorkerV2.alloc_memory_pool(
                    fake_worker,
                    memory_pool_config=memory_pool_config,
                    req_to_token_pool=target_req_pool,
                    token_to_kv_pool_allocator=target_allocator,
                )
                self.assertIs(captured["memory_pool_config"], memory_pool_config)
                self.assertIs(captured["req_to_token_pool"], expected_req_pool)
                self.assertIs(captured["token_to_kv_pool_allocator"], target_allocator)

    def test_prefill_rejects_hidden_row_contract_drift(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        batch = SimpleNamespace(
            extend_lens=[2],
            prefix_lens=[0],
            out_cache_loc=object(),
        )
        with self.assertRaisesRegex(RuntimeError, "hidden rows"):
            DFlashWorkerV2._append_prefill_target_hidden_to_draft_cache(
                SimpleNamespace(),
                batch=batch,
                target_hidden=torch.empty((3, 8)),
            )

    def test_worker_gate_rejects_unqualified_topology(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        fake_worker = SimpleNamespace(
            use_physical_draft_ring=True,
            use_compact_draft_cache=True,
            draft_window_size=2048,
            block_size=5,
            page_size=32,
            ps=SimpleNamespace(tp_size=8),
            server_args=SimpleNamespace(
                pp_size=1,
                dcp_size=1,
                disaggregation_mode="null",
                enable_hierarchical_cache=False,
                enable_lmcache=False,
                enable_streaming_session=False,
                enable_session_radix_cache=False,
                enable_unified_memory=False,
            ),
            draft_model_runner=SimpleNamespace(is_hybrid_swa=False),
        )
        DFlashWorkerV2._validate_draft_ring_configuration(fake_worker)

        fake_worker.ps.tp_size = 4
        with self.assertRaisesRegex(RuntimeError, "tp_size=4"):
            DFlashWorkerV2._validate_draft_ring_configuration(fake_worker)

        fake_worker.ps.tp_size = 8
        fake_worker.draft_model_runner.is_hybrid_swa = True
        with self.assertRaisesRegex(RuntimeError, "hybrid-SWA draft pool"):
            DFlashWorkerV2._validate_draft_ring_configuration(fake_worker)

        fake_worker.draft_model_runner.is_hybrid_swa = False
        fake_worker.server_args.enable_unified_memory = True
        with self.assertRaisesRegex(RuntimeError, "unified memory"):
            DFlashWorkerV2._validate_draft_ring_configuration(fake_worker)


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernel requires CUDA")
class TestDFlashDraftRingPrepareKernel(unittest.TestCase):
    def test_kernel_matches_independent_ring_oracle(self):
        from sglang.kernels.ops.speculative.dflash_prepare_block import (
            _prepare_dflash_ring_draft_block_unchecked,
        )

        config = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        device = torch.device("cuda")
        prefix_lens = torch.tensor(
            [0, 31, 32, 2048, 2079, 10000], dtype=torch.int64, device=device
        )
        req_indices = torch.arange(1, 7, dtype=torch.int64, device=device)
        verified_ids = torch.arange(100, 106, dtype=torch.int64, device=device)
        bs = int(prefix_lens.numel())
        width = 10100
        target = torch.arange(
            config.request_rows * width, dtype=torch.int32, device=device
        ).view(config.request_rows, width)
        draft = torch.full(
            (config.request_rows, config.max_compact_len + config.block_size + 8),
            -1,
            dtype=torch.int32,
            device=device,
        )
        block_ids = torch.empty(
            (bs, config.block_size), dtype=torch.int64, device=device
        )
        positions = torch.empty_like(block_ids)
        target_locs = torch.empty_like(block_ids)
        draft_locs = torch.empty_like(block_ids)
        draft_lens = torch.empty((bs,), dtype=torch.int32, device=device)
        block_end = torch.empty_like(draft_lens)
        live_pages = torch.empty_like(draft_lens)

        _prepare_dflash_ring_draft_block_unchecked(
            verified_id=verified_ids,
            prefix_lens=prefix_lens,
            req_pool_indices=req_indices,
            target_req_to_token=target,
            draft_req_to_token=draft,
            block_ids_out=block_ids,
            positions_out=positions,
            target_cache_loc_out=target_locs,
            draft_cache_loc_out=draft_locs,
            draft_seq_lens_out=draft_lens,
            block_end_out=block_end,
            live_pages_out=live_pages,
            request_rows=config.request_rows,
            window_size=config.window_size,
            page_size=config.page_size,
            max_compact_len=config.max_compact_len,
            max_live_pages=config.max_live_pages,
            ring_pages=config.ring_pages,
            row_stride=config.row_stride,
            mask_token_id=999,
        )
        torch.cuda.synchronize()

        for row, (prefix_len, req_idx) in enumerate(
            zip(prefix_lens.tolist(), req_indices.tolist())
        ):
            draft_len = compact_draft_seq_len(
                prefix_len, config.window_size, config.page_size
            )
            absolute_prefix = torch.arange(
                prefix_len - draft_len,
                prefix_len,
                dtype=torch.int64,
                device=device,
            )
            expected_prefix_locs = draft_ring_cache_locs(
                torch.tensor([req_idx], dtype=torch.int64, device=device),
                absolute_prefix,
                config,
            ).to(torch.int32)
            torch.testing.assert_close(
                draft[req_idx, :draft_len], expected_prefix_locs, rtol=0, atol=0
            )

            expected_positions = torch.arange(
                prefix_len,
                prefix_len + config.block_size,
                dtype=torch.int64,
                device=device,
            )
            expected_draft_locs = draft_ring_cache_locs(
                torch.tensor([req_idx], dtype=torch.int64, device=device),
                expected_positions,
                config,
            )
            torch.testing.assert_close(
                positions[row], expected_positions, rtol=0, atol=0
            )
            torch.testing.assert_close(
                draft_locs[row], expected_draft_locs, rtol=0, atol=0
            )
            torch.testing.assert_close(
                target_locs[row],
                target[req_idx, expected_positions].to(torch.int64),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                draft[req_idx, draft_len : draft_len + config.block_size],
                expected_draft_locs.to(torch.int32),
                rtol=0,
                atol=0,
            )
            self.assertEqual(int(draft_lens[row]), draft_len)
            self.assertEqual(int(block_end[row]), draft_len + config.block_size)
            self.assertEqual(
                int(live_pages[row]),
                actual_dflash_ring_live_page_span(
                    prefix_len=prefix_len,
                    window_size=config.window_size,
                    page_size=config.page_size,
                    block_size=config.block_size,
                ),
            )
            self.assertTrue(
                bool(
                    (
                        draft[
                            req_idx,
                            draft_len + config.block_size :,
                        ]
                        == -1
                    ).all()
                )
            )


if __name__ == "__main__":
    unittest.main()
