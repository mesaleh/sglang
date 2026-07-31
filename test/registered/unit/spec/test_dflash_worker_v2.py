import unittest

from sglang.srt.speculative.dflash_worker_v2 import (
    _max_compact_draft_seq_len,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashWorkerV2(unittest.TestCase):
    def test_compact_window_includes_page_alignment_tail(self):
        self.assertEqual(_max_compact_draft_seq_len(2048, 32), 2079)
        self.assertEqual(_max_compact_draft_seq_len(2048, 1), 2048)


if __name__ == "__main__":
    unittest.main()
