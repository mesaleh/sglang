import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("h44_exact_telemetry.py")
SPEC = importlib.util.spec_from_file_location("h44_exact_telemetry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def state(accept: int, forward: int, dp_rank: int = 0) -> dict[str, int]:
    return {
        MODULE.COUNTER_DP_RANK_KEY: dp_rank,
        MODULE.COUNTER_KEYS[0]: accept,
        MODULE.COUNTER_KEYS[1]: forward,
    }


def event(
    sequence: int,
    timestamp: int,
    *,
    text: bool = False,
    cumulative: int | None = None,
    delta: int | None = None,
    finish_reason: str | None = None,
) -> dict:
    return {
        "sequence": sequence,
        "monotonic_ns": timestamp,
        "has_text": text,
        "completion_tokens_cumulative": cumulative,
        "completion_tokens_delta": delta,
        "finish_reason": finish_reason,
    }


class TestCounterAttribution(unittest.TestCase):
    def test_clean_single_dp_delta(self):
        result = MODULE.compute_counter_delta([state(10, 4)], [state(15, 6)])

        self.assertEqual(result["active_dp_state_index"], 0)
        self.assertEqual(result["internal_accept_tokens"], 5)
        self.assertEqual(result["forward_ct"], 2)
        self.assertEqual(result["internal_accept_length"], 2.5)

    def test_legacy_rollover_is_not_part_of_snapshot(self):
        info = {
            "internal_states": [
                {
                    **state(15, 6),
                    "spec_num_accept_tokens": 0,
                    "spec_total_num_accept_tokens": 999,
                    "spec_total_num_forward_ct": 333,
                }
            ]
        }

        self.assertEqual(MODULE.extract_counter_states(info), [state(15, 6)])

    def test_multi_dp_accepts_exactly_one_active_state(self):
        result = MODULE.compute_counter_delta(
            [state(1, 1, 0), state(8, 4, 1)],
            [state(1, 1, 0), state(13, 6, 1)],
        )

        self.assertEqual(result["active_dp_state_index"], 1)
        self.assertEqual(result["active_dp_rank"], 1)

    def test_multiple_active_dp_states_are_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "exactly one active"):
            MODULE.compute_counter_delta(
                [state(1, 1, 0), state(8, 4, 1)],
                [state(2, 2, 0), state(13, 6, 1)],
            )

    def test_counter_regression_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "regression"):
            MODULE.compute_counter_delta([state(10, 4)], [state(9, 5)])

    def test_zero_forward_delta_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "exactly one active"):
            MODULE.compute_counter_delta([state(10, 4)], [state(10, 4)])

    def test_accept_without_forward_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "without"):
            MODULE.compute_counter_delta([state(10, 4)], [state(11, 4)])

    def test_missing_counter_field_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "nonnegative integer"):
            MODULE.extract_counter_states(
                {"internal_states": [{MODULE.COUNTER_KEYS[0]: 1}]}
            )

    def test_between_request_contamination_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "contamination"):
            MODULE.require_counter_continuity([state(10, 4)], [state(11, 5)])

    def test_equal_between_request_snapshots_are_clean(self):
        MODULE.require_counter_continuity(
            [state(10, 4, 0), state(1, 1, 1)],
            [state(10, 4, 0), state(1, 1, 1)],
        )

    def test_arrival_order_is_canonicalized_by_dp_rank(self):
        extracted = MODULE.extract_counter_states(
            {
                "internal_states": [
                    state(20, 8, 1),
                    state(10, 4, 0),
                ]
            }
        )

        self.assertEqual(extracted, [state(10, 4, 0), state(20, 8, 1)])

    def test_duplicate_dp_rank_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "duplicate"):
            MODULE.extract_counter_states(
                {
                    "internal_states": [
                        state(10, 4, 0),
                        state(20, 8, 0),
                    ]
                }
            )

    def test_counter_result_is_integer_json(self):
        result = MODULE.compute_counter_delta([state(10, 4)], [state(15, 6)])
        encoded = json.dumps(result)

        self.assertIn('"internal_accept_tokens": 5', encoded)
        self.assertIsInstance(result["internal_accept_tokens"], int)
        self.assertIsInstance(result["forward_ct"], int)


class TestSpecRequestIdentity(unittest.TestCase):
    def _validate(self, *, completion: int, correct: int, verify: int):
        internal_accept = correct + verify
        return MODULE.validate_spec_request_identity(
            counter_delta={
                "internal_accept_tokens": internal_accept,
                "forward_ct": verify,
            },
            completion_tokens=completion,
            verify_ct=verify,
            correct_drafts=correct,
            proposed_drafts=verify * 4,
            reported_accept_length=completion / verify,
            configured_draft_tokens=5,
        )

    def test_initial_token_can_make_internal_count_one_lower(self):
        result = self._validate(completion=8, correct=1, verify=6)

        self.assertEqual(result["internal_accept_tokens"], 7)
        self.assertEqual(result["boundary_adjustment_tokens"], -1)
        self.assertEqual(result["response_accept_length"], 8 / 6)
        self.assertEqual(result["emitted_interval_yield"], 7 / 6)

    def test_final_speculative_run_can_overshoot_output_limit(self):
        result = self._validate(completion=32, correct=19, verify=16)

        self.assertEqual(result["internal_accept_tokens"], 35)
        self.assertEqual(result["boundary_adjustment_tokens"], 3)
        self.assertEqual(result["internal_accept_length"], 35 / 16)
        self.assertEqual(result["response_accept_length"], 2.0)

    def test_counter_must_match_drafts_plus_bonus(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "drafts-plus-bonus"):
            MODULE.validate_spec_request_identity(
                counter_delta={"internal_accept_tokens": 34, "forward_ct": 16},
                completion_tokens=32,
                verify_ct=16,
                correct_drafts=19,
                proposed_drafts=64,
                reported_accept_length=2.0,
                configured_draft_tokens=5,
            )

    def test_response_accept_length_uses_emitted_tokens(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "emitted-completion"):
            MODULE.validate_spec_request_identity(
                counter_delta={"internal_accept_tokens": 35, "forward_ct": 16},
                completion_tokens=32,
                verify_ct=16,
                correct_drafts=19,
                proposed_drafts=64,
                reported_accept_length=35 / 16,
                configured_draft_tokens=5,
            )

    def test_boundary_adjustment_outside_draft_bound_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "boundary adjustment"):
            MODULE.validate_spec_request_identity(
                counter_delta={"internal_accept_tokens": 36, "forward_ct": 16},
                completion_tokens=32,
                verify_ct=16,
                correct_drafts=20,
                proposed_drafts=64,
                reported_accept_length=2.0,
                configured_draft_tokens=5,
            )

    def test_stream_boundary_records_all_three_conventions(self):
        result = MODULE.validate_stream_counter_boundary(
            counter_delta={"internal_accept_tokens": 64, "forward_ct": 40},
            completion_tokens=64,
            configured_draft_tokens=5,
        )

        self.assertEqual(result["internal_accept_length"], 1.6)
        self.assertEqual(result["response_accept_length"], 1.6)
        self.assertEqual(result["emitted_interval_yield"], 63 / 40)


class TestSseEventSeries(unittest.TestCase):
    def _parse(self, body: dict, sequence: int, timestamp: int, previous: int):
        return MODULE.make_sse_event(
            raw_line=b"data: " + json.dumps(body).encode(),
            sequence=sequence,
            monotonic_ns=timestamp,
            realtime_ns=timestamp + 1000,
            previous_known_tokens=previous,
        )

    def test_multi_token_and_duplicate_events_preserve_exact_deltas(self):
        first, first_text, known = self._parse(
            {
                "text": "a",
                "meta_info": {"completion_tokens": 1},
                "choices": [{"delta": {"content": "a"}}],
            },
            0,
            10,
            0,
        )
        second, second_text, known = self._parse(
            {
                "text": "abc",
                "meta_info": {"completion_tokens": 3},
                "choices": [{"delta": {"content": "bc"}}],
            },
            1,
            20,
            known,
        )
        duplicate, duplicate_text, known = self._parse(
            {
                "meta_info": {"completion_tokens": 3},
                "choices": [{"delta": {"content": ""}}],
            },
            2,
            21,
            known,
        )

        self.assertEqual((first_text, second_text, duplicate_text), ("a", "bc", ""))
        self.assertEqual(first["completion_tokens_delta"], 1)
        self.assertEqual(second["completion_tokens_delta"], 2)
        self.assertEqual(duplicate["completion_tokens_delta"], 0)
        self.assertEqual(known, 3)

    def test_done_event_is_a_finish_marker_without_token_inference(self):
        done, text, known = MODULE.make_sse_event(
            raw_line=b"data: [DONE]",
            sequence=0,
            monotonic_ns=10,
            realtime_ns=20,
            previous_known_tokens=7,
        )

        self.assertTrue(done["finish_marker"])
        self.assertTrue(done["done_marker"])
        self.assertIsNone(done["completion_tokens_delta"])
        self.assertEqual(text, "")
        self.assertEqual(known, 7)

    def test_first_token_boundary_and_metric_reconstruction(self):
        events = [
            event(0, 5, text=False),
            event(1, 10, text=True, cumulative=1, delta=1),
            event(2, 20, text=True, cumulative=3, delta=2),
            event(
                3,
                25,
                text=False,
                cumulative=3,
                delta=0,
                finish_reason="length",
            ),
        ]

        metrics = MODULE.reconstruct_stream_metrics(
            events=events,
            started_monotonic_ns=0,
            ended_monotonic_ns=30,
            final_completion_tokens=3,
        )

        self.assertEqual(metrics["first_token_monotonic_ns"], 10)
        self.assertEqual(metrics["last_token_monotonic_ns"], 20)
        self.assertEqual(metrics["ttft_ms"], 0.00001)
        self.assertEqual(metrics["decode_interval_ms"], 0.00001)
        self.assertEqual(metrics["tpot_ms"], 0.000005)
        self.assertEqual(metrics["e2e_ms"], 0.00003)
        self.assertEqual(metrics["finish_reason"], "length")

    def test_monotonic_clock_regression_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "clock regressed"):
            MODULE.reconstruct_stream_metrics(
                events=[event(0, 10, text=True), event(1, 9, text=True)],
                started_monotonic_ns=0,
                ended_monotonic_ns=20,
                final_completion_tokens=2,
            )

    def test_final_usage_mismatch_is_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "does not reconcile"):
            MODULE.reconstruct_stream_metrics(
                events=[event(0, 10, text=True, cumulative=2, delta=2)],
                started_monotonic_ns=0,
                ended_monotonic_ns=20,
                final_completion_tokens=3,
            )

    def test_zero_completion_tokens_are_rejected(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "zero"):
            MODULE.reconstruct_stream_metrics(
                events=[event(0, 10, text=True)],
                started_monotonic_ns=0,
                ended_monotonic_ns=20,
                final_completion_tokens=0,
            )

    def test_openai_usage_only_event_is_retained_without_per_chunk_inference(self):
        text_event, _, known = self._parse(
            {"choices": [{"delta": {"reasoning_content": "multi token text"}}]},
            0,
            10,
            0,
        )
        usage_event, _, known = self._parse(
            {"choices": [], "usage": {"completion_tokens": 4}},
            1,
            20,
            known,
        )

        self.assertIsNone(text_event["completion_tokens_cumulative"])
        self.assertIsNone(text_event["completion_tokens_delta"])
        self.assertEqual(usage_event["completion_tokens_cumulative"], 4)
        self.assertEqual(usage_event["completion_tokens_delta"], 4)
        self.assertEqual(known, 4)


class TestHarnessStaticContract(unittest.TestCase):
    def test_timed_request_uses_injected_canonical_module(self):
        canonical = SimpleNamespace(
            make_payload=lambda **kwargs: {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
                "stream": True,
            }
        )
        before = {
            "states": [state(10, 4)],
            "speculative_num_draft_tokens": 5,
        }
        after = {
            "states": [state(521, 304)],
            "speculative_num_draft_tokens": 5,
        }
        stream = {
            "ok": True,
            "error": None,
            "prompt_tokens": MODULE.EXPECTED_PROMPT_TOKENS,
            "completion_tokens": MODULE.OUTPUT_TOKENS,
            "finish_reason": "length",
            "decode_interval_ms": 1000.0,
        }
        prompt_entry = {
            "salt": "salt",
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
            ],
        }

        with (
            patch.object(MODULE, "fetch_counter_snapshot", side_effect=[before, after]),
            patch.object(MODULE, "stream_openai_chat", return_value=stream),
            patch.object(MODULE.time, "sleep"),
        ):
            row = MODULE.run_timed_request(
                canonical=canonical,
                base_url="http://127.0.0.1:30000",
                headers={},
                role="fp8_pre",
                request_index=0,
                is_warmup=False,
                prompt_entry=prompt_entry,
                previous_after=[state(10, 4)],
                timeout_s=10,
                identity={"source_commit": "test"},
            )

        self.assertTrue(row["telemetry_valid"])
        self.assertTrue(row["performance_eligible"])
        self.assertEqual(row["counter_delta"]["forward_ct"], 300)
        self.assertEqual(
            row["counter_reconciliation"]["emitted_interval_yield"], 511 / 300
        )

    def test_failed_stream_preserves_partial_evidence_without_counter_masking(self):
        canonical = SimpleNamespace(
            make_payload=lambda **kwargs: {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
                "stream": True,
            }
        )
        snapshot = {
            "states": [state(10, 4)],
            "speculative_num_draft_tokens": 5,
        }
        stream = {
            "ok": False,
            "error": "TimeoutError: client timed out",
            "events": [{"sequence": 0, "body_sha256": "partial"}],
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
        }
        prompt_entry = {
            "salt": "salt",
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
            ],
        }

        with (
            patch.object(
                MODULE,
                "fetch_counter_snapshot",
                side_effect=[snapshot, snapshot],
            ),
            patch.object(MODULE, "stream_openai_chat", return_value=stream),
            patch.object(MODULE.time, "sleep"),
        ):
            row = MODULE.run_timed_request(
                canonical=canonical,
                base_url="http://127.0.0.1:30000",
                headers={},
                role="fp8_pre",
                request_index=0,
                is_warmup=False,
                prompt_entry=prompt_entry,
                previous_after=[state(10, 4)],
                timeout_s=10,
                identity={"source_commit": "test"},
            )

        self.assertFalse(row["telemetry_valid"])
        self.assertIsNone(row["counter_delta"])
        self.assertEqual(row["stream"]["events"][0]["body_sha256"], "partial")
        self.assertEqual(row["validation_errors"][0], "TimeoutError: client timed out")
        self.assertIn("counter delta unavailable", row["validation_errors"][1])

    def test_harness_never_calls_set_internal_state(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("/set_internal_state", source)

    def test_frozen_contract_constants(self):
        self.assertEqual(MODULE.PROMPT_TARGET, 16_000)
        self.assertEqual(MODULE.EXPECTED_PROMPT_TOKENS, 10_218)
        self.assertEqual(MODULE.OUTPUT_TOKENS, 512)
        self.assertEqual(MODULE.WARMUP_REQUESTS, 2)
        self.assertEqual(MODULE.QUIET_INTERVAL_S, 0.250)

    def test_frozen_kimi_renderer_is_narrow_and_exact(self):
        rendered = MODULE.render_frozen_kimi_chat(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "USER"},
            ]
        )

        self.assertEqual(
            rendered,
            "<|im_system|>system<|im_middle|>SYS<|im_end|>"
            "<|im_user|>user<|im_middle|>USER<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|><think>",
        )

    def test_frozen_kimi_renderer_rejects_other_message_shapes(self):
        with self.assertRaisesRegex(MODULE.TelemetryError, "system,user"):
            MODULE.render_frozen_kimi_chat([{"role": "user", "content": "USER"}])

    def test_prompt_manifest_freezes_shared_role_independent_prompts(self):
        warmups = []
        measured = []
        for index in range(32):
            messages = [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": f"USER-{index}"},
            ]
            entry = {
                "salt": f"salt-{index}",
                "messages": messages,
                "prompt_sha256": MODULE.sha256_json(messages),
                "prompt_tokens": MODULE.EXPECTED_PROMPT_TOKENS,
            }
            if index < 2:
                warmups.append(entry)
            else:
                fp8_output = {
                    "reasoning_content": f"output-{index}",
                    "content": None,
                    "finish_reason": "length",
                    "matched_stop": None,
                }
                entry.update(
                    {
                        "selected_index": index - 2,
                        "fp8_completion_tokens": MODULE.OUTPUT_TOKENS,
                        "fp8_output": fp8_output,
                        "fp8_output_sha256": MODULE.sha256_json(fp8_output),
                    }
                )
                measured.append(entry)
        manifest = {
            "schema_version": 1,
            "corpus_id": "test-corpus",
            "qualification_prompt": {
                "salt": "qualification-salt",
                "messages": [
                    {"role": "system", "content": "QUAL-SYS"},
                    {"role": "user", "content": "QUAL-USER"},
                ],
                "prompt_sha256": MODULE.sha256_json(
                    [
                        {"role": "system", "content": "QUAL-SYS"},
                        {"role": "user", "content": "QUAL-USER"},
                    ]
                ),
                "prompt_tokens": MODULE.EXPECTED_PROMPT_TOKENS,
                "rendered_prompt_sha256": MODULE.sha256_bytes(
                    MODULE.render_frozen_kimi_chat(
                        [
                            {"role": "system", "content": "QUAL-SYS"},
                            {"role": "user", "content": "QUAL-USER"},
                        ]
                    ).encode()
                ),
                "native_completion_tokens": MODULE.OUTPUT_TOKENS,
                "native_output_ids_sha256": "a" * 64,
            },
            "warmup_prompts": warmups,
            "measured_prompts": measured,
        }
        raw = json.dumps(manifest, sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_bytes(raw)

            loaded, observed_hash = MODULE.load_prompt_manifest(
                path, hashlib.sha256(raw).hexdigest(), 30
            )

        self.assertEqual(loaded["corpus_id"], "test-corpus")
        self.assertEqual(observed_hash, hashlib.sha256(raw).hexdigest())

    def test_prompt_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(MODULE.TelemetryError, "hash mismatch"):
                MODULE.load_prompt_manifest(path, "0" * 64, 10)


if __name__ == "__main__":
    unittest.main()
