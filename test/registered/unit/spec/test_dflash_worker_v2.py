import types
import unittest

from sglang.srt.speculative.dflash_worker_v2 import (
    _copy_dflash_draft_server_args,
)


class TestDFlashWorkerV2(unittest.TestCase):
    def test_draft_uses_unquantized_kv_without_mutating_target_args(self):
        for target_dtype in ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1"):
            with self.subTest(target_dtype=target_dtype):
                target_args = types.SimpleNamespace(
                    kv_cache_dtype=target_dtype,
                    nested=types.SimpleNamespace(value=1),
                )

                draft_args = _copy_dflash_draft_server_args(target_args)

                self.assertEqual(target_args.kv_cache_dtype, target_dtype)
                self.assertEqual(draft_args.kv_cache_dtype, "auto")
                self.assertIsNot(draft_args, target_args)
                self.assertIsNot(draft_args.nested, target_args.nested)

    def test_draft_preserves_supported_kv_dtype(self):
        for target_dtype in ("auto", "bfloat16"):
            with self.subTest(target_dtype=target_dtype):
                target_args = types.SimpleNamespace(kv_cache_dtype=target_dtype)

                draft_args = _copy_dflash_draft_server_args(target_args)

                self.assertEqual(draft_args.kv_cache_dtype, target_dtype)
                self.assertIsNot(draft_args, target_args)


if __name__ == "__main__":
    unittest.main()
