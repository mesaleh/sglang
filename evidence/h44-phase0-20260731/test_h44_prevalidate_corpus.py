import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("h44_prevalidate_corpus.py")
SPEC = importlib.util.spec_from_file_location("h44_prevalidate_corpus", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
TELEMETRY = MODULE.TELEMETRY


def record(index: int, *, prompt_tokens: int = 10_218, completion_tokens: int = 512):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": f"USER-{index}"},
    ]
    output = {
        "reasoning_content": f"reasoning-{index}",
        "content": None,
        "finish_reason": "length",
        "matched_stop": None,
    }
    return {
        "candidate_index": index,
        "salt": f"salt-{index}",
        "messages": messages,
        "prompt_sha256": TELEMETRY.sha256_json(messages),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output": output,
        "output_sha256": TELEMETRY.sha256_json(output),
        "rendered_prompt_sha256": TELEMETRY.sha256_bytes(
            TELEMETRY.render_frozen_kimi_chat(messages).encode()
        ),
        "native_output_ids_sha256": TELEMETRY.sha256_json(
            list(range(completion_tokens))
        ),
    }


class TestInitialSelection(unittest.TestCase):
    def test_qualification_prevalidation_uses_native_rendered_prompt(self):
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ]
        canonical = SimpleNamespace(build_messages=lambda *args: messages)
        response = {
            "meta_info": {
                "prompt_tokens": TELEMETRY.EXPECTED_PROMPT_TOKENS,
                "completion_tokens": TELEMETRY.OUTPUT_TOKENS,
            },
            "output_ids": list(range(TELEMETRY.OUTPUT_TOKENS)),
        }

        with patch.object(TELEMETRY, "_post_json", return_value=response) as post:
            observed = MODULE.make_native_qualification_record(
                canonical=canonical,
                native_url="http://127.0.0.1:30000/generate",
                headers={},
                corpus_id="corpus",
                candidate_index=0,
                timeout_s=10,
            )

        payload = post.call_args.args[1]
        self.assertEqual(payload["text"], TELEMETRY.render_frozen_kimi_chat(messages))
        self.assertEqual(
            observed["rendered_prompt_sha256"],
            TELEMETRY.sha256_bytes(payload["text"].encode()),
        )
        self.assertEqual(observed["completion_tokens"], TELEMETRY.OUTPUT_TOKENS)

    def test_first_thirty_eligible_candidates_are_selected_in_order(self):
        warmups = [record(index, completion_tokens=1) for index in range(10)]
        candidates = [record(index) for index in range(40)]
        candidates[0] = record(0, prompt_tokens=10_219)
        candidates[1] = record(1, completion_tokens=511)

        manifest = MODULE.select_initial_manifest(
            corpus_id="corpus",
            qualification_records=[record(100)],
            warmup_records=warmups,
            candidate_records=candidates,
            identity={"image_digest": "sha256:image"},
        )

        self.assertEqual(len(manifest["warmup_prompts"]), 2)
        self.assertEqual(len(manifest["measured_prompts"]), 30)
        self.assertEqual(
            [entry["candidate_index"] for entry in manifest["measured_prompts"]],
            list(range(2, 32)),
        )
        self.assertEqual(manifest["endpoint_subset_indices"], list(range(10)))

    def test_too_few_eligible_candidates_are_rejected(self):
        warmups = [record(index, completion_tokens=1) for index in range(10)]
        candidates = [record(index, completion_tokens=511) for index in range(40)]

        with self.assertRaisesRegex(TELEMETRY.TelemetryError, "only 0"):
            MODULE.select_initial_manifest(
                corpus_id="corpus",
                qualification_records=[record(100)],
                warmup_records=warmups,
                candidate_records=candidates,
                identity={},
            )

    def test_too_few_exact_warmups_are_rejected(self):
        warmups = [record(index, prompt_tokens=10_219) for index in range(10)]
        candidates = [record(index) for index in range(40)]

        with self.assertRaisesRegex(TELEMETRY.TelemetryError, "warmup"):
            MODULE.select_initial_manifest(
                corpus_id="corpus",
                qualification_records=[record(100)],
                warmup_records=warmups,
                candidate_records=candidates,
                identity={},
            )

    def test_qualification_prompt_must_be_exact_and_full(self):
        warmups = [record(index, completion_tokens=1) for index in range(10)]
        candidates = [record(index) for index in range(40)]

        with self.assertRaisesRegex(TELEMETRY.TelemetryError, "qualification"):
            MODULE.select_initial_manifest(
                corpus_id="corpus",
                qualification_records=[record(100, completion_tokens=511)],
                warmup_records=warmups,
                candidate_records=candidates,
                identity={},
            )


class TestEndpointReplay(unittest.TestCase):
    def setUp(self):
        warmups = [record(index, completion_tokens=1) for index in range(10)]
        candidates = [record(index) for index in range(40)]
        self.manifest = MODULE.select_initial_manifest(
            corpus_id="corpus",
            qualification_records=[record(100)],
            warmup_records=warmups,
            candidate_records=candidates,
            identity={},
        )
        self.replays = []
        for entry in self.manifest["measured_prompts"][:10]:
            self.replays.append(
                {
                    "prompt_sha256": entry["prompt_sha256"],
                    "prompt_tokens": TELEMETRY.EXPECTED_PROMPT_TOKENS,
                    "completion_tokens": TELEMETRY.OUTPUT_TOKENS,
                    "output": entry["fp8_output"],
                    "output_sha256": entry["fp8_output_sha256"],
                }
            )

    def test_exact_ten_prompt_replay_passes(self):
        result = MODULE.verify_endpoint_replay(self.manifest, self.replays)

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["comparisons"]), 10)

    def test_output_byte_change_is_rejected(self):
        self.replays[3] = dict(self.replays[3])
        self.replays[3]["output"] = dict(self.replays[3]["output"])
        self.replays[3]["output"]["reasoning_content"] += "changed"

        with self.assertRaisesRegex(TELEMETRY.TelemetryError, "byte replay"):
            MODULE.verify_endpoint_replay(self.manifest, self.replays)

    def test_dropped_endpoint_prompt_is_rejected(self):
        with self.assertRaisesRegex(TELEMETRY.TelemetryError, "ten prompts"):
            MODULE.verify_endpoint_replay(self.manifest, self.replays[:-1])


class TestStaticContract(unittest.TestCase):
    def test_frozen_prevalidation_counts(self):
        self.assertEqual(MODULE.INITIAL_CANDIDATES, 40)
        self.assertEqual(MODULE.QUALIFICATION_CANDIDATES, 10)
        self.assertEqual(MODULE.SELECTED_PROMPTS, 30)
        self.assertEqual(MODULE.ENDPOINT_PROMPTS, 10)


if __name__ == "__main__":
    unittest.main()
