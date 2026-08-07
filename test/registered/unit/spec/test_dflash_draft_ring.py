import unittest
from array import array
import json
from types import SimpleNamespace
from unittest.mock import patch

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
from sglang.srt.speculative.dflash_draft_snapshot import (
    DFlashDraftSnapshotDirectory,
    DFlashDraftSnapshotStore,
    build_dflash_draft_snapshot_config,
    build_dflash_snapshot_keys,
    match_prefix_with_dflash_snapshot,
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

    def test_worker_snapshot_allocation_uses_bounded_pool(self):
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        draft_runner = SimpleNamespace()
        captured = {}
        fake_store = object()

        class FakeDraftWorker:
            def alloc_memory_pool(self, *, memory_pool_config, **_):
                captured["config"] = memory_pool_config
                draft_runner.req_to_token_pool = SimpleNamespace(_alloc_size=9)
                draft_runner.token_to_kv_pool = SimpleNamespace(
                    size=62304,
                    get_kv_size_bytes=lambda: 6_291_456,
                )

        fake_worker = SimpleNamespace(
            use_physical_draft_ring=True,
            use_compact_draft_cache=True,
            use_draft_snapshot=True,
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
            _draft_snapshot_config=None,
            _draft_snapshot_store=None,
            ps=SimpleNamespace(tp_rank=1),
        )
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.DFlashDraftSnapshotStore",
            return_value=fake_store,
        ) as store_cls:
            DFlashWorkerV2.alloc_memory_pool(
                fake_worker,
                memory_pool_config=MemoryPoolConfig(
                    max_total_num_tokens=262144,
                    max_running_requests=8,
                ),
                req_to_token_pool=SimpleNamespace(_alloc_size=9),
                token_to_kv_pool_allocator=object(),
            )

        self.assertEqual(captured["config"].max_total_num_tokens, 62304)
        self.assertEqual(fake_worker._draft_snapshot_config.padded_tokens, 62336)
        self.assertIs(fake_worker._draft_snapshot_store, fake_store)
        store_cls.assert_called_once_with(
            draft_runner.token_to_kv_pool, fake_worker._draft_snapshot_config
        )

    def test_worker_snapshot_publication_requires_cross_rank_agreement(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        ring = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        config = build_dflash_draft_snapshot_config(ring, min_prefix_length=32)
        directory = DFlashDraftSnapshotDirectory(config, namespace="test")
        key = directory.make_key(list(range(32)), 32, extra_key=None)
        req = SimpleNamespace(rid="rank-agreement", req_pool_idx=3)
        candidates = [(req, key, 32, None)]
        fake_worker = SimpleNamespace(ps=SimpleNamespace(tp_size=2))

        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=SimpleNamespace(
                all_gather_object=lambda value: [value, value]
            ),
        ):
            DFlashWorkerV2._require_snapshot_publication_batch_rank_agreement(
                fake_worker, directory=directory, candidates=candidates
            )
        self.assertEqual(directory.stats()["pending"], 0)

        def divergent_gather(value):
            directory_signature, candidate_signature = value
            divergent = (
                directory_signature,
                (*candidate_signature, ("other-rid", 4, 32, b"x" * 32, 32, None)),
            )
            return [value, divergent]

        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=SimpleNamespace(all_gather_object=divergent_gather),
        ):
            with self.assertRaisesRegex(RuntimeError, "diverged across TP ranks"):
                DFlashWorkerV2._require_snapshot_publication_batch_rank_agreement(
                    fake_worker, directory=directory, candidates=candidates
                )
        self.assertEqual(directory.stats()["pending"], 0)
        self.assertEqual(directory.stats()["free"], config.physical_slots)

    def test_worker_snapshot_publication_batches_one_collective_and_reuses_reserve(
        self,
    ):
        from sglang.srt.environ import envs
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        ring = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        config = build_dflash_draft_snapshot_config(ring, min_prefix_length=32)
        directory = DFlashDraftSnapshotDirectory(config, namespace="test")
        for value in range(20):
            tokens = list(range(value, value + 32))
            key = directory.make_key(tokens, 32, extra_key=None)
            publication = directory.begin_publish(key, valid_rows=32)
            directory.commit_publish(publication)

        copied = []
        gather_calls = []
        worker = SimpleNamespace(
            use_draft_snapshot=True,
            _draft_snapshot_directory=directory,
            _draft_snapshot_store=SimpleNamespace(
                publish=lambda publication, request_pool_index: copied.append(
                    (publication, request_pool_index)
                ),
                row_elements=128,
                kv_pool=SimpleNamespace(k_buffer=[object()] * 6),
            ),
            _draft_snapshot_config=config,
            page_size=32,
            ps=SimpleNamespace(tp_size=8, tp_rank=0),
            _logged_first_draft_snapshot_publication=True,
        )
        worker._require_snapshot_publication_batch_rank_agreement = lambda **kwargs: (
            DFlashWorkerV2._require_snapshot_publication_batch_rank_agreement(
                worker, **kwargs
            )
        )
        worker._log_dflash_snapshot_telemetry = lambda event, **fields: (
            DFlashWorkerV2._log_dflash_snapshot_telemetry(worker, event, **fields)
        )
        worker._snapshot_handle_telemetry = DFlashWorkerV2._snapshot_handle_telemetry
        reqs = [
            SimpleNamespace(
                rid=f"batch-{index}",
                req_pool_idx=index + 1,
                kv_committed_len=config.snapshot_rows,
                cache_protected_len=config.snapshot_rows,
                extra_key=None,
                get_fill_ids=lambda value=value: list(
                    range(value, value + config.snapshot_rows)
                ),
            )
            for index, value in enumerate(range(100, 108))
        ]
        tree_cache = SimpleNamespace(dflash_snapshot_directory=lambda: directory)

        def gather(value):
            gather_calls.append(value)
            return [value] * 8

        with (
            envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_TELEMETRY.override(True),
            patch(
                "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
                return_value=SimpleNamespace(all_gather_object=gather),
            ),
            patch("sglang.srt.speculative.dflash_worker_v2.logger.info") as log_info,
        ):
            publications = DFlashWorkerV2.maybe_publish_dflash_snapshots(
                worker, reqs=reqs, tree_cache=tree_cache
            )

        self.assertEqual(len(gather_calls), 1)
        self.assertEqual(len(publications), 8)
        self.assertEqual(len(copied), 8)
        self.assertEqual(directory.stats()["resident"], 20)
        self.assertEqual(directory.stats()["pending"], 0)
        self.assertEqual(directory.stats()["free"], 1)
        telemetry = [
            json.loads(call.args[2])
            for call in log_info.call_args_list
            if call.args[:2] == ("%s%s", "DFLASH_SNAPSHOT_TELEMETRY ")
        ]
        self.assertEqual(len(telemetry), 8)
        self.assertTrue(all(item["event"] == "publication" for item in telemetry))
        self.assertTrue(all(item["copied_bytes"] == 6_291_456 for item in telemetry))

    def test_worker_snapshot_invalid_candidate_joins_collective_before_error(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        ring = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        config = build_dflash_draft_snapshot_config(ring, min_prefix_length=32)
        directory = DFlashDraftSnapshotDirectory(config, namespace="test")
        worker = SimpleNamespace(
            use_draft_snapshot=True,
            _draft_snapshot_directory=directory,
            _draft_snapshot_store=SimpleNamespace(),
            _draft_snapshot_config=config,
            page_size=32,
            ps=SimpleNamespace(tp_size=8, tp_rank=1),
        )
        worker._require_snapshot_publication_batch_rank_agreement = lambda **kwargs: (
            DFlashWorkerV2._require_snapshot_publication_batch_rank_agreement(
                worker, **kwargs
            )
        )
        req = SimpleNamespace(
            rid="invalid",
            req_pool_idx=None,
            kv_committed_len=32,
            cache_protected_len=32,
            extra_key=None,
            get_fill_ids=lambda: list(range(32)),
        )
        gathered = []

        def gather(value):
            gathered.append(value)
            return [value] * 8

        with (
            patch(
                "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
                return_value=SimpleNamespace(all_gather_object=gather),
            ),
            self.assertRaisesRegex(RuntimeError, "requires a request row"),
        ):
            DFlashWorkerV2.maybe_publish_dflash_snapshots(
                worker,
                reqs=[req],
                tree_cache=SimpleNamespace(dflash_snapshot_directory=lambda: directory),
            )
        self.assertEqual(len(gathered), 1)
        self.assertEqual(directory.stats()["pending"], 0)

    def test_worker_snapshot_publication_floor_keeps_collective_sentinels(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        ring = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        config = build_dflash_draft_snapshot_config(ring)
        directory = DFlashDraftSnapshotDirectory(config, namespace="test")
        copied = []
        gathered = []
        worker = SimpleNamespace(
            use_draft_snapshot=True,
            _draft_snapshot_directory=directory,
            _draft_snapshot_store=SimpleNamespace(
                publish=lambda publication, request_pool_index: copied.append(
                    (publication, request_pool_index)
                )
            ),
            _draft_snapshot_config=config,
            page_size=32,
            ps=SimpleNamespace(tp_size=8, tp_rank=1),
            _logged_first_draft_snapshot_publication=True,
        )
        worker._require_snapshot_publication_batch_rank_agreement = lambda **kwargs: (
            DFlashWorkerV2._require_snapshot_publication_batch_rank_agreement(
                worker, **kwargs
            )
        )
        reqs = [
            SimpleNamespace(
                rid=f"boundary-{boundary}",
                req_pool_idx=index + 1,
                kv_committed_len=boundary,
                cache_protected_len=boundary,
                extra_key=None,
                get_fill_ids=lambda boundary=boundary: list(range(boundary)),
            )
            for index, boundary in enumerate((10_176, 10_208))
        ]

        def gather(value):
            gathered.append(value)
            return [value] * 8

        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=SimpleNamespace(all_gather_object=gather),
        ):
            publications = DFlashWorkerV2.maybe_publish_dflash_snapshots(
                worker,
                reqs=reqs,
                tree_cache=SimpleNamespace(dflash_snapshot_directory=lambda: directory),
            )

        self.assertEqual(len(gathered), 1)
        candidate_signature = gathered[0][1]
        self.assertEqual(candidate_signature[0][2:], (None, None, 0, None))
        self.assertEqual(candidate_signature[1][2], 10_208)
        self.assertEqual(len(publications), 1)
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0][0].key.prefix_length, 10_208)
        self.assertEqual(directory.stats()["resident"], 1)

    def test_worker_snapshot_restore_consumes_handle_but_retains_pin_lifecycle(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        restored = []
        store = SimpleNamespace(restore=lambda items: restored.append(list(items)))
        handle = SimpleNamespace(key=SimpleNamespace(prefix_length=64))
        req = SimpleNamespace(
            rid="chunked",
            req_pool_idx=3,
            dflash_snapshot_handle=handle,
        )
        worker = SimpleNamespace(
            use_draft_snapshot=True,
            _draft_snapshot_store=store,
            _logged_first_draft_snapshot_restore=True,
            ps=SimpleNamespace(tp_rank=1),
        )
        batch = SimpleNamespace(reqs=[req], prefix_lens=[64])

        DFlashWorkerV2._restore_dflash_snapshots_for_prefill(worker, batch)
        self.assertEqual(restored, [[(handle, 3, 64)]])
        self.assertIsNone(req.dflash_snapshot_handle)

        # A later continuation chunk keeps the directory's request-id pin for
        # final release, but cannot consume the physical snapshot twice.
        batch.prefix_lens = [96]
        DFlashWorkerV2._restore_dflash_snapshots_for_prefill(worker, batch)
        self.assertEqual(restored[-1], [])

    def test_worker_snapshot_restore_emits_one_structured_decision(self):
        from sglang.srt.environ import envs
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        restored = []
        store = SimpleNamespace(
            restore=lambda items: restored.append(list(items)),
            row_elements=128,
            kv_pool=SimpleNamespace(k_buffer=[object()] * 6),
        )
        handle = SimpleNamespace(
            key=SimpleNamespace(prefix_length=64, digest=b"d" * 32),
            slot=2,
            slot_generation=3,
            cache_generation=4,
            valid_rows=64,
        )
        req = SimpleNamespace(
            rid="telemetry",
            req_pool_idx=3,
            dflash_snapshot_handle=handle,
            dflash_snapshot_telemetry_emitted=False,
            dflash_snapshot_match_duration_ns=1234,
            dflash_snapshot_match_plan=SimpleNamespace(
                status="snapshot_selected",
                normal_boundary=32,
                full_boundary=64,
                selected_boundary=64,
                delta=0,
            ),
        )
        directory = SimpleNamespace(
            stats=lambda: {
                "cache_generation": 4,
                "resident": 1,
                "pending": 0,
                "request_refs": 1,
                "free": 20,
                "pinned": 1,
            }
        )
        worker = SimpleNamespace(
            use_draft_snapshot=True,
            _draft_snapshot_store=store,
            _draft_snapshot_directory=directory,
            _logged_first_draft_snapshot_restore=True,
            ps=SimpleNamespace(tp_rank=0),
        )
        worker._log_dflash_snapshot_telemetry = lambda event, **fields: (
            DFlashWorkerV2._log_dflash_snapshot_telemetry(worker, event, **fields)
        )
        worker._snapshot_handle_telemetry = DFlashWorkerV2._snapshot_handle_telemetry
        batch = SimpleNamespace(reqs=[req], prefix_lens=[64])

        with (
            envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_TELEMETRY.override(True),
            patch("sglang.srt.speculative.dflash_worker_v2.logger.info") as log_info,
        ):
            DFlashWorkerV2._restore_dflash_snapshots_for_prefill(worker, batch)

        telemetry = [
            json.loads(call.args[2])
            for call in log_info.call_args_list
            if call.args[:2] == ("%s%s", "DFLASH_SNAPSHOT_TELEMETRY ")
        ]
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(telemetry[0]["event"], "restore_decision")
        self.assertEqual(telemetry[0]["status"], "snapshot_selected")
        self.assertEqual(telemetry[0]["copied_bytes"], 196_608)
        self.assertEqual(telemetry[0]["match_duration_ns"], 1234)
        self.assertTrue(req.dflash_snapshot_telemetry_emitted)
        self.assertIsNone(req.dflash_snapshot_handle)

    def test_worker_default_off_preserves_allocator_delegation(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        memory_pool_config = object()
        target_req_pool = object()
        target_kv_pool = object()
        target_allocator = SimpleNamespace(get_kvcache=lambda: target_kv_pool)
        for compact in (False, True):
            with self.subTest(compact=compact):
                captured = {}
                draft_runner = SimpleNamespace()
                current_target_req_pool = (
                    SimpleNamespace(_alloc_size=9) if compact else target_req_pool
                )
                expected_req_pool = None if compact else current_target_req_pool

                class FakeDraftWorker:
                    def alloc_memory_pool(self, **kwargs):
                        captured.update(kwargs)
                        if compact:
                            draft_runner.req_to_token_pool = SimpleNamespace(
                                _alloc_size=9
                            )
                            draft_runner.token_to_kv_pool_allocator = target_allocator
                            draft_runner.token_to_kv_pool = SimpleNamespace(size=262144)

                fake_worker = SimpleNamespace(
                    use_physical_draft_ring=False,
                    use_compact_draft_cache=compact,
                    _draft_worker=FakeDraftWorker(),
                    draft_model_runner=draft_runner,
                    ps=SimpleNamespace(tp_rank=1),
                )
                fake_worker._attest_dense_draft_pool_allocation = lambda **kwargs: (
                    DFlashWorkerV2._attest_dense_draft_pool_allocation(
                        fake_worker, **kwargs
                    )
                )
                DFlashWorkerV2.alloc_memory_pool(
                    fake_worker,
                    memory_pool_config=memory_pool_config,
                    req_to_token_pool=current_target_req_pool,
                    token_to_kv_pool_allocator=target_allocator,
                )
                self.assertIs(captured["memory_pool_config"], memory_pool_config)
                self.assertIs(captured["req_to_token_pool"], expected_req_pool)
                self.assertIs(captured["token_to_kv_pool_allocator"], target_allocator)

    def test_dense_compact_allocation_attestation_fails_closed(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        target_kv_pool = object()
        target_allocator = SimpleNamespace(get_kvcache=lambda: target_kv_pool)
        target_req_pool = SimpleNamespace(_alloc_size=9)
        draft_req_pool = SimpleNamespace(_alloc_size=9)
        draft_kv_pool = SimpleNamespace(size=262144)
        worker = SimpleNamespace(
            use_compact_draft_cache=True,
            draft_model_runner=SimpleNamespace(
                req_to_token_pool=draft_req_pool,
                token_to_kv_pool_allocator=target_allocator,
                token_to_kv_pool=draft_kv_pool,
            ),
            ps=SimpleNamespace(tp_rank=0),
        )
        with patch("sglang.srt.speculative.dflash_worker_v2.logger.info") as info:
            DFlashWorkerV2._attest_dense_draft_pool_allocation(
                worker,
                target_req_to_token_pool=target_req_pool,
                target_token_allocator=target_allocator,
            )
        info.assert_called_once()
        self.assertIn("allocator_shared=True", info.call_args.args[0])

        worker.draft_model_runner.token_to_kv_pool_allocator = object()
        with self.assertRaisesRegex(RuntimeError, "not co-located"):
            DFlashWorkerV2._attest_dense_draft_pool_allocation(
                worker,
                target_req_to_token_pool=target_req_pool,
                target_token_allocator=target_allocator,
            )

    def test_prepare_path_latch_logs_once_on_tp0(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        worker = SimpleNamespace(
            _logged_first_prepare_path=False,
            ps=SimpleNamespace(tp_rank=0),
        )
        with patch("sglang.srt.speculative.dflash_worker_v2.logger.info") as info:
            DFlashWorkerV2._latch_first_prepare_path(
                worker, "triton_dense_compact", 7
            )
            DFlashWorkerV2._latch_first_prepare_path(worker, "triton_ring", 1)
        info.assert_called_once_with(
            "DFLASH prepare path latched: path=%s, batch_size=%d.",
            "triton_dense_compact",
            7,
        )
        self.assertTrue(worker._logged_first_prepare_path)

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

        fake_worker.server_args.enable_unified_memory = False
        fake_worker.server_args.enable_dp_attention = True
        with self.assertRaisesRegex(RuntimeError, "DP attention"):
            DFlashWorkerV2._validate_draft_ring_configuration(fake_worker)


class TestDFlashDraftSnapshot(unittest.TestCase):
    def setUp(self):
        ring = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        self.config = build_dflash_draft_snapshot_config(ring, min_prefix_length=32)

    def _key(self, directory, value, boundary=32):
        tokens = list(range(value, value + boundary))
        return directory.make_key(tokens, boundary, extra_key="tenant")

    def _publish(self, directory, value, boundary=32):
        key = self._key(directory, value, boundary)
        publication = directory.begin_publish(
            key, valid_rows=min(boundary, self.config.snapshot_rows)
        )
        self.assertTrue(publication.copy_required)
        return directory.commit_publish(publication)

    def test_snapshot_layout_uses_aligned_window_not_replay_span(self):
        cfg = build_dflash_draft_snapshot_config(self.config.ring)
        self.assertEqual(cfg.min_prefix_length, 10_208)
        self.assertEqual(cfg.snapshot_rows, 2048)
        self.assertEqual(cfg.snapshot_base, 19328)
        self.assertEqual(cfg.physical_slots, 21)
        self.assertEqual(cfg.service_slots, 20)
        self.assertEqual(cfg.pool_size, 62304)
        self.assertEqual(cfg.padded_tokens, 62336)
        self.assertEqual(cfg.slot_base(0), 19328)
        self.assertEqual(cfg.slot_base(20) + cfg.snapshot_rows, 62336)
        with self.assertRaisesRegex(ValueError, "outside"):
            cfg.slot_base(21)

    def test_snapshot_floor_is_inclusive_and_rejects_one_page_below(self):
        cfg = build_dflash_draft_snapshot_config(self.config.ring)
        directory = DFlashDraftSnapshotDirectory(cfg, namespace="test")
        tokens = array("q", range(cfg.min_prefix_length))

        below = directory.make_key(
            tokens,
            cfg.min_prefix_length - cfg.ring.page_size,
            extra_key="tenant",
        )
        with self.assertRaisesRegex(ValueError, "below min_prefix_length"):
            directory.begin_publish(below, valid_rows=cfg.snapshot_rows)
        self.assertEqual(directory.stats()["free"], cfg.physical_slots)

        exact = directory.make_key(tokens, cfg.min_prefix_length, extra_key="tenant")
        publication = directory.begin_publish(exact, valid_rows=cfg.snapshot_rows)
        handle = directory.commit_publish(publication)
        self.assertEqual(handle.key.prefix_length, 10_208)
        self.assertEqual(
            directory.find_tentative(
                tokens,
                extra_key="tenant",
                normal_boundary=10_176,
                full_boundary=10_208,
                max_delta=0,
            ),
            handle,
        )
        self.assertIsNone(
            directory.find_tentative(
                tokens,
                extra_key="tenant",
                normal_boundary=10_144,
                full_boundary=10_176,
                max_delta=0,
            )
        )

    def test_aligned_snapshot_plus_delta_covers_every_compact_suffix(self):
        cfg = self.config
        for boundary in range(0, 256_001, cfg.ring.page_size):
            snapshot_start = max(0, boundary - cfg.snapshot_rows)
            for delta in range(0, 2113):
                end = min(256_000, boundary + delta)
                compact_len = compact_draft_seq_len(
                    end, cfg.ring.window_size, cfg.ring.page_size
                )
                self.assertGreaterEqual(end - compact_len, snapshot_start)

    def test_content_keys_are_incremental_and_namespace_exact(self):
        tokens = array("q", range(96))
        keys = build_dflash_snapshot_keys(
            tokens,
            [32, 64, 96],
            namespace="model|draft|source|rope",
            extra_key="tenant-a",
            cache_generation=7,
        )
        self.assertEqual(
            keys[96].digest.hex(),
            "5ea7d2c06df46ec027baea0d23cc96f6f44b1c0597f7d39f10f3a2f020349dc4",
        )
        direct = build_dflash_snapshot_keys(
            tokens,
            [96],
            namespace="model|draft|source|rope",
            extra_key="tenant-a",
            cache_generation=7,
        )
        self.assertEqual(keys[96], direct[96])
        none_extra = build_dflash_snapshot_keys(
            tokens,
            [96],
            namespace="model|draft|source|rope",
            extra_key=None,
            cache_generation=7,
        )
        empty_extra = build_dflash_snapshot_keys(
            tokens,
            [96],
            namespace="model|draft|source|rope",
            extra_key="",
            cache_generation=7,
        )
        self.assertNotEqual(none_extra[96], empty_extra[96])

    def test_directory_two_phase_reserve_pin_and_generation(self):
        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        handles = [self._publish(directory, value) for value in range(20)]
        self.assertEqual(directory.stats()["resident"], 20)
        self.assertEqual(directory.stats()["free"], 1)

        self.assertEqual(
            directory.acquire_for_request(handles[0], request_id="rid-0"),
            handles[0],
        )
        replacement_key = self._key(directory, 100)
        replacement = directory.begin_publish(replacement_key, valid_rows=32)
        self.assertTrue(replacement.copy_required)
        self.assertIsNotNone(replacement.victim)
        self.assertNotEqual(replacement.victim, handles[0])
        self.assertEqual(directory.stats()["pending"], 1)
        replacement_handle = directory.commit_publish(replacement)
        self.assertEqual(directory.stats()["resident"], 20)
        self.assertEqual(directory.stats()["free"], 1)
        self.assertIsNone(
            directory.acquire_for_request(replacement.victim, request_id="stale-victim")
        )
        self.assertEqual(
            directory.acquire_for_request(replacement_handle, request_id="rid-new"),
            replacement_handle,
        )
        self.assertTrue(directory.release_request("rid-new"))
        self.assertTrue(directory.release_request("rid-0"))
        old_generation = directory.cache_generation
        directory.reset()
        self.assertEqual(directory.cache_generation, old_generation + 1)
        self.assertEqual(directory.stats()["resident"], 0)
        self.assertIsNone(
            directory.acquire_for_request(replacement_handle, request_id="stale-gen")
        )

    def test_directory_rank_agreement_signature_covers_lru_and_refs(self):
        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        initial = directory.rank_agreement_signature()
        handle = self._publish(directory, 1)
        published = directory.rank_agreement_signature()
        self.assertNotEqual(initial, published)
        directory.acquire_for_request(handle, request_id="rid")
        pinned = directory.rank_agreement_signature()
        self.assertNotEqual(published, pinned)
        directory.release_request("rid")
        released = directory.rank_agreement_signature()
        self.assertNotEqual(pinned, released)

    def test_directory_release_and_reset_telemetry_reconciles_occupancy(self):
        from sglang.srt.environ import envs

        directory = DFlashDraftSnapshotDirectory(
            self.config, namespace="test", emit_telemetry=True
        )
        handle = self._publish(directory, 1)
        directory.acquire_for_request(handle, request_id="rid")
        with (
            envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_TELEMETRY.override(True),
            patch(
                "sglang.srt.speculative.dflash_draft_snapshot.logger.info"
            ) as log_info,
        ):
            directory.release_request("rid")
            directory.reset()

        telemetry = [json.loads(call.args[2]) for call in log_info.call_args_list]
        self.assertEqual([item["event"] for item in telemetry], ["release", "reset"])
        self.assertEqual(telemetry[0]["request_refs"], 0)
        self.assertEqual(telemetry[0]["pinned"], 0)
        self.assertEqual(telemetry[1]["resident"], 0)
        self.assertEqual(telemetry[1]["free"], 21)

        suppressed = DFlashDraftSnapshotDirectory(
            self.config, namespace="nonzero-rank", emit_telemetry=False
        )
        handle = self._publish(suppressed, 2)
        suppressed.acquire_for_request(handle, request_id="rank-1")
        with (
            envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_TELEMETRY.override(True),
            patch(
                "sglang.srt.speculative.dflash_draft_snapshot.logger.info"
            ) as suppressed_log,
        ):
            suppressed.release_request("rank-1")
            suppressed.reset()
        suppressed_log.assert_not_called()

    def test_directory_declines_duplicate_pending_and_all_pinned(self):
        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        key = self._key(directory, 1)
        first = directory.begin_publish(key, valid_rows=32)
        duplicate = directory.begin_publish(key, valid_rows=32)
        self.assertEqual(duplicate.reason, "duplicate_pending")
        directory.abort_publish(first)

        handles = [self._publish(directory, value + 10) for value in range(20)]
        for index, handle in enumerate(handles):
            directory.acquire_for_request(handle, request_id=f"rid-{index}")
        rejected = directory.begin_publish(self._key(directory, 1000), valid_rows=32)
        self.assertEqual(rejected.reason, "all_residents_pinned")
        with self.assertRaisesRegex(RuntimeError, "zero references"):
            directory.reset()

    def test_tentative_lookup_obeys_content_delta_and_generation(self):
        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        tokens = array("q", range(256))
        key = directory.make_key(tokens, 128, extra_key="tenant")
        publication = directory.begin_publish(key, valid_rows=128)
        handle = directory.commit_publish(publication)
        self.assertEqual(
            directory.find_tentative(
                tokens,
                extra_key="tenant",
                normal_boundary=64,
                full_boundary=160,
                max_delta=32,
            ),
            handle,
        )
        self.assertIsNone(
            directory.find_tentative(
                tokens,
                extra_key="other",
                normal_boundary=64,
                full_boundary=160,
                max_delta=32,
            )
        )
        self.assertIsNone(
            directory.find_tentative(
                tokens,
                extra_key="tenant",
                normal_boundary=64,
                full_boundary=160,
                max_delta=0,
            )
        )

    def test_snapshot_match_validates_target_and_falls_back_atomically(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

        tokens = array("q", range(256))
        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        key = directory.make_key(tokens, 224, extra_key="tenant")
        publication = directory.begin_publish(key, valid_rows=224)
        handle = directory.commit_publish(publication)

        def make_cache(prefix_length):
            cache = RadixCache.create_simulated(page_size=32)
            cache.register_dflash_snapshot_directory(directory)
            cache.reprefill_tail_tokens = lambda: 64
            cache.insert(
                InsertParams(
                    key=RadixKey(tokens[:prefix_length], extra_key="tenant"),
                    value=torch.arange(prefix_length, dtype=torch.int64),
                )
            )
            return cache

        req = SimpleNamespace(extra_key="tenant", rid="hit")
        cache = make_cache(224)
        with (
            envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_MAX_DELTA.override(0),
            patch(
                "sglang.srt.speculative.dflash_draft_snapshot."
                "build_dflash_snapshot_keys",
                wraps=build_dflash_snapshot_keys,
            ) as build_keys,
        ):
            match_prefix_with_dflash_snapshot(
                tree_cache=cache,
                req=req,
                token_ids=array("q", tokens),
                base_key_limit=224,
                cow_mamba=False,
                include_req=True,
                acquire=False,
            )
            result = match_prefix_with_dflash_snapshot(
                tree_cache=cache,
                req=req,
                token_ids=tokens,
                base_key_limit=224,
                cow_mamba=False,
                include_req=True,
                acquire=True,
            )
        self.assertEqual(build_keys.call_count, 1)
        self.assertEqual(len(result.device_indices), 224)
        self.assertEqual(req.dflash_snapshot_match_plan.status, "snapshot_selected")
        self.assertEqual(req.dflash_snapshot_handle, handle)
        self.assertIsNone(req._dflash_snapshot_key_cache)
        self.assertTrue(directory.release_request(req.rid))

        req = SimpleNamespace(extra_key="tenant", rid="target-miss")
        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_MAX_DELTA.override(0):
            result = match_prefix_with_dflash_snapshot(
                tree_cache=make_cache(192),
                req=req,
                token_ids=tokens,
                base_key_limit=224,
                cow_mamba=False,
                include_req=True,
                acquire=True,
            )
        self.assertEqual(len(result.device_indices), 192)
        self.assertEqual(
            req.dflash_snapshot_match_plan.status, "target_validation_failed"
        )
        self.assertIsNone(req.dflash_snapshot_handle)
        self.assertEqual(directory.stats()["request_refs"], 0)

    def test_snapshot_match_accepts_page_aligned_delta_and_reports_it(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

        tokens = array("q", range(256))
        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        key = directory.make_key(tokens, 224, extra_key="tenant")
        publication = directory.begin_publish(key, valid_rows=224)
        handle = directory.commit_publish(publication)
        cache = RadixCache.create_simulated(page_size=32)
        cache.register_dflash_snapshot_directory(directory)
        cache.reprefill_tail_tokens = lambda: 64
        cache.insert(
            InsertParams(
                key=RadixKey(tokens, extra_key="tenant"),
                value=torch.arange(len(tokens), dtype=torch.int64),
            )
        )
        req = SimpleNamespace(extra_key="tenant", rid="delta-hit")

        with envs.SGLANG_OMNIVA_DFLASH_DRAFT_SNAPSHOT_MAX_DELTA.override(32):
            result = match_prefix_with_dflash_snapshot(
                tree_cache=cache,
                req=req,
                token_ids=tokens,
                base_key_limit=256,
                cow_mamba=False,
                include_req=True,
                acquire=True,
            )

        self.assertEqual(len(result.device_indices), 224)
        self.assertEqual(req.dflash_snapshot_match_plan.status, "snapshot_selected")
        self.assertEqual(req.dflash_snapshot_match_plan.selected_boundary, 224)
        self.assertEqual(req.dflash_snapshot_match_plan.delta, 32)
        self.assertEqual(req.dflash_snapshot_handle, handle)
        self.assertTrue(directory.release_request(req.rid))

    def test_tree_cache_release_clears_request_pin(self):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        directory = DFlashDraftSnapshotDirectory(self.config, namespace="test")
        handle = self._publish(directory, 1)
        cache = RadixCache.create_simulated(page_size=32)
        cache.register_dflash_snapshot_directory(directory)
        req = SimpleNamespace(rid="release", dflash_snapshot_handle=handle)
        directory.acquire_for_request(handle, request_id=req.rid)

        self.assertTrue(cache.release_dflash_snapshot_for_req(req))
        self.assertIsNone(req.dflash_snapshot_handle)
        self.assertEqual(directory.stats()["request_refs"], 0)

    def test_cache_insert_boolean_preserves_disabled_cache_bookkeeping(self):
        from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

        calls = []
        cache = SimpleNamespace(
            disable=True,
            cache_unfinished_req=lambda req, **kwargs: calls.append((req, kwargs)),
        )
        req = SimpleNamespace(skip_radix_cache_insert=False)
        self.assertFalse(maybe_cache_unfinished_req(req, cache, chunked=True))
        self.assertEqual(calls, [(req, {"chunked": True})])

        req.skip_radix_cache_insert = True
        self.assertFalse(maybe_cache_unfinished_req(req, cache))
        self.assertEqual(len(calls), 1)

    def test_cache_insert_boolean_requires_qualified_committed_boundary(self):
        from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

        directory = SimpleNamespace(
            config=SimpleNamespace(ring=SimpleNamespace(page_size=32))
        )

        def cache_unfinished_req(req, **_):
            req.cache_protected_len = req.inserted_len

        cache = SimpleNamespace(
            disable=False,
            cache_unfinished_req=cache_unfinished_req,
            dflash_snapshot_directory=lambda: directory,
        )
        req = SimpleNamespace(
            skip_radix_cache_insert=False,
            kv_committed_len=95,
            inserted_len=64,
            cache_protected_len=0,
        )
        self.assertTrue(maybe_cache_unfinished_req(req, cache))

        req.inserted_len = 32
        self.assertFalse(maybe_cache_unfinished_req(req, cache))

        ordinary_cache = SimpleNamespace(
            disable=False,
            cache_unfinished_req=cache_unfinished_req,
        )
        req.inserted_len = 64
        self.assertFalse(maybe_cache_unfinished_req(req, ordinary_cache))


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


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernel requires CUDA")
class TestDFlashDraftSnapshotCopyKernel(unittest.TestCase):
    def test_batched_snapshot_restore_matches_wrapped_ring_oracle(self):
        ring = build_dflash_draft_ring_config(
            window_size=2048,
            page_size=32,
            block_size=5,
            alloc_reserve=10,
            request_rows=9,
        )
        config = build_dflash_draft_snapshot_config(ring, min_prefix_length=32)

        class FakePool:
            size = config.pool_size
            kv_cache_layout = "nhd"

            def __init__(self):
                self.k_buffer = [
                    torch.zeros(
                        (config.padded_tokens, 1, 128),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    for _ in range(6)
                ]
                self.v_buffer = [torch.zeros_like(self.k_buffer[0]) for _ in range(6)]

        pool = FakePool()
        store = DFlashDraftSnapshotStore(pool, config)
        directory = DFlashDraftSnapshotDirectory(config, namespace="cuda-test")
        items = []
        expected = {}
        ring_locations = []
        for request_index in range(8):
            boundary = 4096 + request_index * ring.page_size
            tokens = array("q", range(boundary))
            key = directory.make_key(tokens, boundary, extra_key="tenant")
            publication = directory.begin_publish(key, valid_rows=config.snapshot_rows)
            positions = torch.arange(
                boundary - config.snapshot_rows,
                boundary,
                dtype=torch.int64,
                device="cuda",
            )
            locations = draft_ring_cache_locs(
                torch.tensor(request_index, dtype=torch.int64, device="cuda"),
                positions,
                ring,
            )
            ring_locations.append(locations)
            for buffer_index, buffer in enumerate([*pool.k_buffer, *pool.v_buffer]):
                value = float(buffer_index * 8 + request_index + 1)
                buffer[locations] = value
                expected[(buffer_index, request_index)] = value
            store.publish(publication, request_pool_index=request_index)
            handle = directory.commit_publish(publication)
            items.append((handle, request_index, boundary))

        torch.cuda.synchronize()
        first_snapshot = pool.k_buffer[0][
            config.slot_base(items[0][0].slot) : config.slot_base(items[0][0].slot)
            + config.snapshot_rows
        ]
        self.assertTrue(
            bool((first_snapshot == expected[(0, 0)]).all()),
            msg=(
                "snapshot publication mismatch: "
                f"range={float(first_snapshot.min())}.."
                f"{float(first_snapshot.max())}"
            ),
        )
        for locations in ring_locations:
            for buffer in [*pool.k_buffer, *pool.v_buffer]:
                buffer[locations] = 0

        store.restore(items[:1])
        torch.cuda.synchronize()
        first_locations = ring_locations[0]
        for buffer_index, buffer in enumerate([*pool.k_buffer, *pool.v_buffer]):
            self.assertTrue(
                bool((buffer[first_locations] == expected[(buffer_index, 0)]).all())
            )
            buffer[first_locations] = 0

        store.restore(items)
        torch.cuda.synchronize()
        for handle, request_index, boundary in items:
            positions = torch.arange(
                boundary - handle.valid_rows,
                boundary,
                dtype=torch.int64,
                device="cuda",
            )
            locations = draft_ring_cache_locs(
                torch.tensor(request_index, dtype=torch.int64, device="cuda"),
                positions,
                ring,
            )
            for buffer_index, buffer in enumerate([*pool.k_buffer, *pool.v_buffer]):
                actual = buffer[locations]
                wanted = torch.full_like(
                    actual, expected[(buffer_index, request_index)]
                )
                self.assertTrue(
                    torch.equal(actual, wanted),
                    msg=(
                        f"buffer={buffer_index}, request={request_index}, "
                        f"boundary={boundary}, nonmatching="
                        f"{int((actual != wanted).sum())}/{actual.numel()}, "
                        f"range={float(actual.min())}..{float(actual.max())}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
