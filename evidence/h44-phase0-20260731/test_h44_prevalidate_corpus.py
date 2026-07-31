import hashlib
import importlib.util
import json
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


def record(
    index: int,
    *,
    salt_index: int | None = None,
    prompt_tokens: int = 10_218,
    completion_tokens: int = 512,
):
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
        "salt_index": index if salt_index is None else salt_index,
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
        self.assertEqual(observed["salt_index"], 200)
        self.assertEqual(observed["salt"], MODULE.build_prompt_salt(200))

    def test_campaign_identity_does_not_change_candidate_messages(self):
        response = {
            "choices": [
                {
                    "message": {"reasoning_content": "R", "content": ""},
                    "finish_reason": "length",
                    "matched_stop": None,
                }
            ],
            "usage": {
                "prompt_tokens": TELEMETRY.EXPECTED_PROMPT_TOKENS,
                "completion_tokens": 1,
            },
        }
        canonical = SimpleNamespace(
            build_messages=lambda *args: [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": args[4]},
            ]
        )
        with patch.object(TELEMETRY, "_post_json", return_value=response):
            first = MODULE.make_candidate_record(
                canonical=canonical,
                chat_url="http://127.0.0.1:30000/v1/chat/completions",
                headers={},
                corpus_id="campaign-a",
                candidate_index=0,
                timeout_s=10,
                warmup_probe=True,
            )
            second = MODULE.make_candidate_record(
                canonical=canonical,
                chat_url="http://127.0.0.1:30000/v1/chat/completions",
                headers={},
                corpus_id="campaign-b",
                candidate_index=0,
                timeout_s=10,
                warmup_probe=True,
            )

        self.assertEqual(first["messages"], second["messages"])
        self.assertEqual(first["salt"], second["salt"])
        self.assertEqual(first["salt_index"], 100)
        self.assertNotIn("campaign-a", first["messages"][1]["content"])
        self.assertNotIn("campaign-b", second["messages"][1]["content"])

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
        self.assertEqual(
            manifest["generator_contract"],
            {
                "h43_canonical_sha256": TELEMETRY.H43_CANONICAL_SHA256,
                "prompt_target": TELEMETRY.PROMPT_TARGET,
                "prompt_tokens": TELEMETRY.EXPECTED_PROMPT_TOKENS,
                "output_tokens": TELEMETRY.OUTPUT_TOKENS,
                "temperature": 0,
                "top_p": 1,
                "thinking": "unset",
                "ignore_eos": False,
                "cache_mode": "unique-prefix",
                "frozen_prompt_salt_prefix": MODULE.FROZEN_PROMPT_SALT_PREFIX,
                "candidate_salt_indices": [0, 39],
                "warmup_salt_indices": [100, 109],
                "qualification_salt_indices": [200, 209],
                "campaign_identity_in_prompt": False,
            },
        )

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

    def test_frozen_salt_ranges_are_disjoint_and_complete(self):
        candidate = set(MODULE.CANDIDATE_SALT_INDICES)
        warmup = set(MODULE.WARMUP_SALT_INDICES)
        qualification = set(MODULE.QUALIFICATION_SALT_INDICES)
        self.assertFalse(candidate & warmup)
        self.assertFalse(candidate & qualification)
        self.assertFalse(warmup & qualification)
        self.assertEqual(
            MODULE.ALLOWED_SALT_INDICES,
            candidate | warmup | qualification,
        )
        with self.assertRaises(ValueError):
            MODULE.build_prompt_salt(99)

    def test_live_probed_boundary_messages_remain_byte_exact(self):
        expected = {
            0: (
                "95812e7fd2a9f179b2d4e66b2e63547a03f177c6eafb4d9824f65f31dec615d3",
                "a60d30dd203f9055ed3b47de286b48b21928a75721d6cb1c9ae7bf5146c14d05",
            ),
            9: (
                "a1105ad65775f6c288157709b04acebfcd4b6ef718e2ddd8b2f160a25255a4c6",
                "8c98900053e4da2092bb2717528c129bb35cf3abf182673c4205ba9e555175c1",
            ),
            10: (
                "84a825357d434843339bdb7735d3f0d103d1af855b387a6a6d35dc17124b3d58",
                "517352550bb7f6b0107c93fa642e1e3c0ed977bda945c52761a09b4300f1c752",
            ),
            39: (
                "f32efbe0bfa02bb1b154cb1f101dfddfb9341e3137e77b87ed2eb59e815446d0",
                "5709a646ed4668f57c46eec4e99c6e02933c3db0fa7523ca50bc87eae929978c",
            ),
            100: (
                "3ef2a13f7e169cb326e26b83f2d7412c1bdbd376f6152bda4ac56a5b9c3aba96",
                "ef39a8f0c7aaadb60b86520490b1922b44a217dbaa93e7598b9ea302ceeaa660",
            ),
            109: (
                "a818b701613698e81727de7222010d6236ac2367fcc8231a37612db7e69d59e0",
                "9b276608ea96b3aa63097169a2c59e7739a0972fc1f7dba8f82f8c2c8cec0fd5",
            ),
            200: (
                "27aaeb1fa9e182f515f4c3f3ad8f36cb23545047192f9946d085ad06cd232c9b",
                "29011efd2b7046983d81e8b0b2c968d81cbc6cbb5835b2ca87399533b4322763",
            ),
            209: (
                "adae5b277c1e6500202ffb919520b2efb7cb18a1eba9e4e1b8d7ff1194c569f7",
                "9b8c1813c21ab6f31978c4136a32276b3f0c040c099732e6ce26b63cfa2a51a3",
            ),
        }
        canonical = TELEMETRY.load_h43_canonical()
        for salt_index, (prompt_sha256, rendered_sha256) in expected.items():
            with self.subTest(salt_index=salt_index):
                messages = MODULE.build_messages(
                    canonical, MODULE.build_prompt_salt(salt_index)
                )
                observed_prompt = hashlib.sha256(
                    json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                observed_rendered = hashlib.sha256(
                    TELEMETRY.render_frozen_kimi_chat(messages).encode()
                ).hexdigest()
                self.assertEqual(observed_prompt, prompt_sha256)
                self.assertEqual(observed_rendered, rendered_sha256)

    def test_all_frozen_prompt_messages_are_unique(self):
        canonical = TELEMETRY.load_h43_canonical()
        prompt_hashes = {
            TELEMETRY.sha256_json(
                MODULE.build_messages(canonical, MODULE.build_prompt_salt(salt_index))
            )
            for salt_index in MODULE.ALLOWED_SALT_INDICES
        }
        self.assertEqual(len(prompt_hashes), len(MODULE.ALLOWED_SALT_INDICES))


if __name__ == "__main__":
    unittest.main()
