"""Unit tests for HostKVCache alloc/free bookkeeping (double-alloc / double-free detection)."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHostTurboQuant,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHostKVCache(CustomTestCase):
    def setUp(self):
        self.page_size = 2
        # Small device pool is enough to construct the host pool.
        self.device_pool = MHATokenToKVPool(
            size=self.page_size * 2,
            page_size=self.page_size,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def test_double_alloc(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        # Mimic bookkeeping corruption: push an already-used slot back to the
        # head of free_slots so the next alloc would hand out an in-use slot.
        leak = torch.tensor([int(indices[0])])
        self.host_pool.free_slots = torch.cat([leak, self.host_pool.free_slots])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.alloc(4)
        msg = str(ctx.exception)
        self.assertIn("Double-alloc", msg)
        self.assertIn(f"[{int(leak[0])}]", msg)

    def test_double_free(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        self.host_pool.free(indices[:2])
        # indices[1] is double freed.
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices[1:])
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[1])}]", msg)

    def test_free_unallocated(self):
        indices = torch.tensor([1])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[0])}]", msg)

    def test_free_after_clear(self):
        indices = self.host_pool.alloc(4)
        self.host_pool.clear()
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(str(indices.tolist()), msg)


class TestMLATurboQuantHostKVCache(CustomTestCase):
    def setUp(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolTurboQuant

        self.page_size = 2
        with envs.SGLANG_TQ_MLA_FUSED_KV_WRITE.override(False):
            self.device_pool = MLATokenToKVPoolTurboQuant(
                size=self.page_size * 2,
                page_size=self.page_size,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                enable_fp8_codebook=True,
            )
        self.host_pool = MLATokenToKVPoolHostTurboQuant(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def test_codebook_allocation_and_transfer_contract(self):
        self.assertEqual(self.host_pool._cell_bytes, 402)
        self.assertEqual(self.host_pool.kv_nope_codebook_host.shape[-2:], (1, 16))

        host_indices = torch.tensor([0, 1])
        device_indices = torch.tensor([2, 3])
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_kv_direct",
            create=True,
        ) as transfer:
            self.host_pool.load_to_device_per_layer(
                self.device_pool,
                host_indices,
                device_indices,
                layer_id=0,
                io_backend="direct",
            )
            self.assertEqual(transfer.call_count, 4)
            transfer.reset_mock()
            self.host_pool.backup_from_device_all_layer(
                self.device_pool,
                host_indices,
                device_indices,
                io_backend="direct",
            )
            self.assertEqual(transfer.call_count, 4)


if __name__ == "__main__":
    unittest.main()
