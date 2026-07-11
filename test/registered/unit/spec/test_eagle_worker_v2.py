import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.tokenspeed_workspace import (
    tokenspeed_workspace_bytes,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
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

        worker = SimpleNamespace(speculative_num_draft_tokens=draft_token_num)
        compact_predict = EAGLEWorkerV2._compact_accept_to_front(
            worker, predict, accept_index, bs=2
        )
        compact_hidden = EAGLEWorkerV2._compact_accept_to_front(
            worker, hidden_states, accept_index, bs=2
        )
        compact_predict_2d = compact_predict.reshape(2, draft_token_num)
        bonus_tokens = compact_predict_2d[
            torch.arange(2), accept_lens.to(torch.long) - 1
        ].to(torch.int32)

        self.assertEqual(compact_predict_2d[0, :3].tolist(), [0, 2, 4])
        self.assertEqual(compact_predict_2d[1, :3].tolist(), [8, 9, 15])
        self.assertEqual(compact_predict_2d[:, 5:].tolist(), [[5, 6, 7], [13, 14, 15]])
        self.assertEqual(
            compact_hidden.reshape(2, draft_token_num, 2)[:, :3].tolist(),
            [
                [
                    [0.0, 1.0],
                    [4.0, 5.0],
                    [8.0, 9.0],
                ],
                [
                    [16.0, 17.0],
                    [18.0, 19.0],
                    [30.0, 31.0],
                ],
            ],
        )
        self.assertEqual(bonus_tokens.tolist(), [4, 15])


class TestTokenspeedWorkspaceSizing(unittest.TestCase):
    def test_small_head_q_chunk_tree_rows_scale_with_raw_q_len(self):
        num_sms = 120
        num_heads = 16
        kv_lora_rank = 512
        q8 = tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 8)

        self.assertEqual(q8, num_sms * 128 * (kv_lora_rank + 1) * 4)
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 16),
            2 * q8,
        )
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 24),
            3 * q8,
        )
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 32),
            4 * q8,
        )


if __name__ == "__main__":
    unittest.main()
