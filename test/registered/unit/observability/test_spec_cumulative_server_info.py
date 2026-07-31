from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import GetInternalStateReq, SetInternalStateReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_scheduler(spec_algorithm: SpeculativeAlgorithm):
    reporter = types.SimpleNamespace(
        last_gen_throughput=0.0,
        spec_total_num_accept_tokens=0,
        spec_total_num_forward_ct=0,
        spec_cumulative_num_accept_tokens=0,
        spec_cumulative_num_forward_ct=0,
        step_time_dict={},
    )
    model_runner = types.SimpleNamespace(
        weight_load_mem_usage=1.0,
        graph_mem_usage=2.0,
    )
    return types.SimpleNamespace(
        metrics_reporter=reporter,
        spec_algorithm=spec_algorithm,
        tp_worker=types.SimpleNamespace(model_runner=model_runner),
        token_to_kv_pool_allocator=types.SimpleNamespace(
            get_kvcache=lambda: types.SimpleNamespace(mem_usage=3.0)
        ),
        max_total_num_tokens=1024,
        max_running_requests=16,
        ps=types.SimpleNamespace(pp_size=1, dp_rank=None, attn_dp_rank=0),
    )


class TestSpecCumulativeServerInfo(unittest.TestCase):
    def _get_internal_state(self, scheduler):
        server_args = types.SimpleNamespace(model_config=None)
        with patch(
            "sglang.srt.managers.scheduler.get_global_server_args",
            return_value=server_args,
        ):
            return Scheduler.get_internal_state(
                scheduler, GetInternalStateReq()
            ).internal_state

    def test_spec_server_exposes_zero_cumulative_counters(self):
        scheduler = _make_scheduler(SpeculativeAlgorithm.DFLASH)

        state = self._get_internal_state(scheduler)

        self.assertEqual(state["spec_cumulative_num_accept_tokens"], 0)
        self.assertEqual(state["spec_cumulative_num_forward_ct"], 0)
        self.assertEqual(state["spec_counter_dp_rank"], 0)

    def test_spec_server_exposes_raw_cumulative_counters(self):
        scheduler = _make_scheduler(SpeculativeAlgorithm.DFLASH)
        scheduler.ps.dp_rank = 3
        scheduler.metrics_reporter.spec_total_num_accept_tokens = 12
        scheduler.metrics_reporter.spec_total_num_forward_ct = 4
        scheduler.metrics_reporter.spec_cumulative_num_accept_tokens = 123
        scheduler.metrics_reporter.spec_cumulative_num_forward_ct = 45

        state = self._get_internal_state(scheduler)

        self.assertEqual(state["avg_spec_accept_length"], 3.0)
        self.assertEqual(state["spec_counter_dp_rank"], 3)
        self.assertEqual(state["spec_cumulative_num_accept_tokens"], 123)
        self.assertEqual(state["spec_cumulative_num_forward_ct"], 45)

    def test_non_spec_server_omits_cumulative_counters(self):
        scheduler = _make_scheduler(SpeculativeAlgorithm.NONE)

        state = self._get_internal_state(scheduler)

        self.assertNotIn("spec_cumulative_num_accept_tokens", state)
        self.assertNotIn("spec_cumulative_num_forward_ct", state)
        self.assertNotIn("spec_counter_dp_rank", state)

    def test_state_update_does_not_reset_cumulative_counters(self):
        scheduler = _make_scheduler(SpeculativeAlgorithm.DFLASH)
        scheduler.metrics_reporter.spec_total_num_accept_tokens = 12
        scheduler.metrics_reporter.spec_total_num_forward_ct = 4
        scheduler.metrics_reporter.spec_cumulative_num_accept_tokens = 123
        scheduler.metrics_reporter.spec_cumulative_num_forward_ct = 45
        server_args = types.SimpleNamespace(
            model_config=None,
            speculative_accept_threshold_single=0.0,
        )

        with patch(
            "sglang.srt.managers.scheduler.get_global_server_args",
            return_value=server_args,
        ):
            result = Scheduler.set_internal_state(
                scheduler,
                SetInternalStateReq(
                    server_args={"speculative_accept_threshold_single": 0.1}
                ),
            )

        self.assertTrue(result.updated)
        self.assertEqual(scheduler.metrics_reporter.spec_total_num_accept_tokens, 0)
        self.assertEqual(scheduler.metrics_reporter.spec_total_num_forward_ct, 0)
        self.assertEqual(
            scheduler.metrics_reporter.spec_cumulative_num_accept_tokens, 123
        )
        self.assertEqual(scheduler.metrics_reporter.spec_cumulative_num_forward_ct, 45)


if __name__ == "__main__":
    unittest.main()
