import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
    _bind_eagle_tree_mask_buffer,
    _refresh_eagle_tree_mask,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _RecordingBackend:
    def __init__(self):
        self.shape_key = None

    def can_run(self, forward_batch, shape_key):
        self.shape_key = shape_key
        return True


class TestDecodeCudaGraphRunner(unittest.TestCase):
    def test_eagle_tree_mask_binds_backend_owned_storage(self):
        original = torch.ones(4, dtype=torch.bool)
        backend_mask = torch.zeros(16, dtype=torch.bool)
        buffers = SimpleNamespace(custom_mask=original)
        backend = SimpleNamespace(
            get_verify_buffers_to_fill_after_draft=lambda: [backend_mask, None]
        )

        _bind_eagle_tree_mask_buffer(
            buffers,
            backend,
            max_total_num_tokens=64,
            max_num_token=8,
            captured_req_width=4,
            device=torch.device("cpu"),
        )

        self.assertIs(buffers.custom_mask, backend_mask)
        self.assertIs(
            _refresh_eagle_tree_mask(buffers.custom_mask, backend_mask), backend_mask
        )

    def test_eagle_tree_mask_fallback_uses_capture_width(self):
        buffers = SimpleNamespace(custom_mask=torch.ones(4, dtype=torch.bool))
        backend = SimpleNamespace(
            get_verify_buffers_to_fill_after_draft=lambda: [None, None]
        )

        _bind_eagle_tree_mask_buffer(
            buffers,
            backend,
            max_total_num_tokens=10,
            max_num_token=6,
            captured_req_width=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(buffers.custom_mask.numel(), 48)
        replay_mask = torch.tensor([False, True, False], dtype=torch.bool)
        refreshed = _refresh_eagle_tree_mask(buffers.custom_mask, replay_mask)
        self.assertEqual(refreshed[:3].tolist(), replay_mask.tolist())

    def test_eagle_tree_mask_rejects_oversized_replay(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            _refresh_eagle_tree_mask(
                torch.ones(2, dtype=torch.bool),
                torch.zeros(3, dtype=torch.bool),
            )

    def test_disable_padding_uses_typed_capture_key(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.require_mlp_tp_gather = False
        runner.enable_pdmux = False
        runner.disable_padding = True
        runner.max_bs = 8
        runner.require_mlp_sync = False
        runner.is_encoder_decoder = False
        runner.capture_hidden_mode = CaptureHiddenMode.FULL
        runner.enable_two_batch_overlap = False
        runner.record_nolora_graph = False
        runner.ragged_verify_mode = False
        runner.captured_req_width = 1
        runner.backend = _RecordingBackend()
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
        )

        forward_batch = SimpleNamespace(
            replace_embeds=None,
            batch_size=3,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=SimpleNamespace(
                capture_hidden_mode=CaptureHiddenMode.FULL,
                num_tokens_per_req=0,
            ),
        )

        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(runner.backend.shape_key, ShapeKey(size=3))

        runner.attn_backend = SimpleNamespace(
            can_run_cuda_graph=lambda forward_batch: False
        )
        runner.backend.shape_key = None
        self.assertFalse(runner.can_run_graph(forward_batch))
        self.assertIsNone(runner.backend.shape_key)

        runner.attn_backend = SimpleNamespace(
            can_run_cuda_graph=lambda forward_batch: True
        )
        runner.backend.shape_key = None
        forward_batch.disable_decode_cuda_graph = True
        self.assertFalse(runner.can_run_graph(forward_batch))
        self.assertIsNone(runner.backend.shape_key)

    def test_spec_graph_runners_honor_attention_replay_policy(self):
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )
        from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
            FrozenKVMTPCudaGraphRunner,
        )
        from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
            MultiLayerEagleDraftExtendCudaGraphRunner,
        )

        for runner_type in (
            EAGLEDraftCudaGraphRunner,
            EAGLEDraftExtendCudaGraphRunner,
            MultiLayerEagleDraftExtendCudaGraphRunner,
            FrozenKVMTPCudaGraphRunner,
        ):
            runner = runner_type.__new__(runner_type)
            runner.attn_backend = SimpleNamespace(
                can_run_cuda_graph=lambda forward_batch: False
            )
            self.assertFalse(runner.can_run_graph(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
