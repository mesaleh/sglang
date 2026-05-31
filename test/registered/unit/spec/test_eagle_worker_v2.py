import unittest

import torch

from sglang.srt.speculative.eagle_worker_v2 import _compact_tree_accept_outputs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestCompactTreeAcceptOutputs(unittest.TestCase):
    def test_global_accept_indices_are_compacted_per_request(self):
        draft_token_num = 8
        predict = torch.arange(16, dtype=torch.int32)
        hidden_states = torch.arange(32, dtype=torch.float32).reshape(16, 2)
        accept_index = torch.tensor(
            [
                [0, 2, 4, -1, -1],
                [8, 9, 15, -1, -1],
            ],
            dtype=torch.int32,
        )
        accept_lens = torch.tensor([3, 3], dtype=torch.int32)

        compact_predict, compact_hidden, bonus_tokens = _compact_tree_accept_outputs(
            predict,
            hidden_states,
            accept_index,
            accept_lens,
            draft_token_num,
        )

        self.assertEqual(
            compact_predict.reshape(2, draft_token_num).tolist(),
            [
                [0, 2, 4, 0, 0, 5, 6, 7],
                [8, 9, 15, 8, 8, 13, 14, 15],
            ],
        )
        self.assertEqual(
            compact_hidden.reshape(2, draft_token_num, 2).tolist(),
            [
                [
                    [0.0, 1.0],
                    [4.0, 5.0],
                    [8.0, 9.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [10.0, 11.0],
                    [12.0, 13.0],
                    [14.0, 15.0],
                ],
                [
                    [16.0, 17.0],
                    [18.0, 19.0],
                    [30.0, 31.0],
                    [16.0, 17.0],
                    [16.0, 17.0],
                    [26.0, 27.0],
                    [28.0, 29.0],
                    [30.0, 31.0],
                ],
            ],
        )
        self.assertEqual(bonus_tokens.tolist(), [4, 15])


if __name__ == "__main__":
    unittest.main()
