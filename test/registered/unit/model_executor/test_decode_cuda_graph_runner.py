import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
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
        runner.backend = _RecordingBackend()
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
        )

        forward_batch = SimpleNamespace(
            replace_embeds=None,
            batch_size=3,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL),
        )

        self.assertTrue(runner.can_run_graph(forward_batch))
        self.assertEqual(runner.backend.shape_key, ShapeKey(size=3))


if __name__ == "__main__":
    unittest.main()
