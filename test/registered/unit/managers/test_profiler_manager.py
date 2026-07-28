import unittest

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)


class TestSchedulerProfilerManager(unittest.TestCase):
    def _manager(self):
        return SchedulerProfilerManager(
            ps=None,
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )

    def _init_profile(self, manager, *, profile_by_stage, num_steps):
        return manager._init_profile(
            output_dir=None,
            start_step=None,
            num_steps=num_steps,
            activities=None,
            with_stack=None,
            record_shapes=None,
            profile_by_stage=profile_by_stage,
            profile_id="test",
        )

    def test_stage_profile_requires_positive_num_steps(self):
        with envs.SGLANG_PROFILE_V2.override(False):
            for num_steps in (None, 0, -1):
                with self.subTest(num_steps=num_steps):
                    manager = self._manager()
                    result = self._init_profile(
                        manager,
                        profile_by_stage=True,
                        num_steps=num_steps,
                    )

                    self.assertFalse(result.success)
                    self.assertIn("positive num_steps", result.message)
                    self.assertFalse(manager.profile_by_stage)
                    self.assertIsNone(manager.profiler_decode_ct)
                    self.assertIsNone(manager.profiler_target_decode_ct)

    def test_stage_profile_initializes_step_counters(self):
        with envs.SGLANG_PROFILE_V2.override(False):
            manager = self._manager()
            result = self._init_profile(
                manager,
                profile_by_stage=True,
                num_steps=5,
            )

            self.assertTrue(result.success)
            self.assertEqual(manager.profiler_prefill_ct, 0)
            self.assertEqual(manager.profiler_decode_ct, 0)
            self.assertEqual(manager.profiler_target_prefill_ct, 5)
            self.assertEqual(manager.profiler_target_decode_ct, 5)

    def test_non_stage_profile_still_allows_manual_stop(self):
        with envs.SGLANG_PROFILE_V2.override(False):
            manager = self._manager()
            result = self._init_profile(
                manager,
                profile_by_stage=False,
                num_steps=None,
            )

            self.assertTrue(result.success)
            self.assertFalse(manager.profile_by_stage)


if __name__ == "__main__":
    unittest.main()
