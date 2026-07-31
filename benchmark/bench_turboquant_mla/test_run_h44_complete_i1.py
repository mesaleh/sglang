#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).with_name("run_h44_complete_i1.py")
SPEC = importlib.util.spec_from_file_location("run_h44_complete_i1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def args_for(
    root: Path,
    *,
    phase: str = "phase0_complete_i1",
    result: Path | None = None,
    verify_only: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        source_commit="a" * 40,
        input_manifest=root / "input-manifest.json",
        input_manifest_sha256="b" * 64,
        verify_inputs_only=verify_only,
        result=result,
    )


class CompleteI1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_sentinel_contract(self) -> None:
        value = {
            "sentinel_before": {"graph_us": 100.0, "telemetry_valid": True},
            "sentinel_after": {"graph_us": 100.4, "telemetry_valid": True},
        }
        reasons, drift = runner.sentinel_reasons(value)
        self.assertEqual(reasons, [])
        self.assertAlmostEqual(drift, 0.4 / 100.2)
        value["sentinel_after"]["graph_us"] = 102.0
        reasons, _ = runner.sentinel_reasons(value)
        self.assertEqual(len(reasons), 1)
        self.assertIn("exceeds", reasons[0])

    def test_last_json_value_accepts_preceding_logs(self) -> None:
        value = runner.last_json_value('log\n{"status":"PASS"}\n', "fixture")
        self.assertEqual(value, {"status": "PASS"})
        with self.assertRaisesRegex(runner.I1Error, "no stdout"):
            runner.last_json_value("", "fixture")

    def test_sealed_analyzer_exposes_process_environment_gate(self) -> None:
        source_root = self.root / "source"
        analyzer_dir = source_root / "benchmark" / "bench_turboquant_mla"
        analyzer_dir.mkdir(parents=True)
        shutil.copyfile(
            MODULE_PATH.with_name("analyze_h41_i1_integrated.py"),
            analyzer_dir / "analyze_h41_i1_integrated.py",
        )
        analyzer = runner.load_analyzer(source_root)
        reasons = analyzer.process_environment_reasons({})
        self.assertEqual(len(reasons), 2)
        self.assertTrue(all("analyzer validation" in reason for reason in reasons))
        self.assertFalse((analyzer_dir / "__pycache__").exists())

    def test_git_tree_oid_matches_git_write_tree(self) -> None:
        repository = self.root / "repository"
        exported = self.root / "exported"
        repository.mkdir()
        (repository / "dir").mkdir()
        (repository / "a.txt").write_text("alpha\n", encoding="utf-8")
        executable = repository / "run"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o555)
        (repository / "dir" / "b.txt").write_text("beta\n", encoding="utf-8")
        (repository / "link").symlink_to("a.txt")
        subprocess.run(["git", "init", "-q", repository], check=True)
        subprocess.run(["git", "-C", repository, "add", "."], check=True)
        expected = subprocess.run(
            ["git", "-C", repository, "write-tree"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        shutil.copytree(
            repository,
            exported,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        self.assertEqual(runner.git_tree_oid(exported), expected)

    def test_inventory_preserves_top_level_mount_for_symlinks(self) -> None:
        root = self.root / "sealed"
        (root / "source" / "shared").mkdir(parents=True)
        (root / "source" / "nested").mkdir()
        (root / "aot").mkdir()
        target = root / "source" / "shared" / "target"
        target.write_text("inside\n", encoding="utf-8")
        target.chmod(0o444)
        aot_target = root / "aot" / "target"
        aot_target.write_text("cross-mount\n", encoding="utf-8")
        aot_target.chmod(0o444)
        (root / "source" / "nested" / "link").symlink_to("../shared/target")
        with mock.patch.object(runner, "SEALED_OWNER_UID", os.getuid()):
            inventory = runner.inventory_tree(root, manifest_name="input-manifest.json")
        self.assertIn(
            {
                "kind": "symlink",
                "path": "source/nested/link",
                "target": "../shared/target",
            },
            inventory,
        )

        (root / "source" / "nested" / "cross").symlink_to("../../aot/target")
        with (
            mock.patch.object(runner, "SEALED_OWNER_UID", os.getuid()),
            self.assertRaisesRegex(runner.I1Error, "top-level container mount"),
        ):
            runner.inventory_tree(root, manifest_name="input-manifest.json")

        (root / "source" / "nested" / "cross").unlink()
        (root / "source" / "nested" / "reentry").symlink_to(
            "../../../sealed/source/shared/target"
        )
        with (
            mock.patch.object(runner, "SEALED_OWNER_UID", os.getuid()),
            self.assertRaisesRegex(runner.I1Error, "top-level container mount"),
        ):
            runner.inventory_tree(root, manifest_name="input-manifest.json")

        (root / "source" / "nested" / "reentry").unlink()
        (root / "source" / "masked-climb").symlink_to("../source/shared/target")
        with (
            mock.patch.object(runner, "SEALED_OWNER_UID", os.getuid()),
            self.assertRaisesRegex(runner.I1Error, "top-level container mount"),
        ):
            runner.inventory_tree(root, manifest_name="input-manifest.json")

    def test_one_total_replacement_is_enforced(self) -> None:
        campaign = runner.CompleteI1(args_for(self.root, verify_only=True))
        campaign.host = "beacon-cr-ct13"
        campaign.rank = 0
        calls: list[tuple[int, int, int]] = []

        def attempt(context: int, sequence: int, attempt_number: int, timeout: int):
            calls.append((context, sequence, attempt_number))
            if (context, sequence, attempt_number) in {
                (10219, 1, 1),
                (10219, 2, 1),
            }:
                return {}, ["invalid"], 0.006
            return {}, [], 0.001

        campaign.run_attempt = attempt  # type: ignore[method-assign]
        with self.assertRaisesRegex(runner.I1Error, "one total validity replacement"):
            campaign.run_sequences(runner.time.monotonic() + 600)
        self.assertEqual(campaign.replacements, 1)
        self.assertEqual(calls[:3], [(10219, 1, 1), (10219, 1, 2), (10219, 2, 1)])

    def test_sequence_uses_remaining_soft_window_budget(self) -> None:
        campaign = runner.CompleteI1(args_for(self.root, verify_only=True))
        campaign.run_attempt = mock.Mock(  # type: ignore[method-assign]
            return_value=({}, [], 0.001)
        )
        with mock.patch.object(runner.time, "monotonic", return_value=881.0):
            campaign.run_sequences(1000.0)
        self.assertEqual(campaign.run_attempt.call_count, 20)
        self.assertTrue(
            all(call.args[3] == 119 for call in campaign.run_attempt.call_args_list)
        )

    def test_truncated_attempt_timeout_is_attributed_to_phase_budget(self) -> None:
        campaign = runner.CompleteI1(args_for(self.root, verify_only=True))
        campaign.container_id = "c" * 64
        campaign.evidence_root = self.root / "evidence"
        campaign.evidence_root.mkdir()
        timed_out = subprocess.CompletedProcess([], 124, "", "timeout\n")
        with (
            mock.patch.object(runner, "docker", return_value=timed_out),
            self.assertRaisesRegex(runner.I1Error, "truncated final attempt"),
        ):
            campaign.run_attempt(10219, 1, 1, 119)

    def test_result_transforms_analysis_to_gate_schema(self) -> None:
        campaign = runner.CompleteI1(args_for(self.root, verify_only=True))
        campaign.container_id = "c" * 64
        campaign.manifest = {
            "files": [{"kind": "file", "path": "source/a", "sha256": "d" * 64}],
            "image_id": "sha256:" + "e" * 64,
            "image_ref": "registry/image@sha256:" + "e" * 64,
            "source_tree": "f" * 40,
        }
        campaign.attempts = [
            {
                "attempt": 1,
                "context": context,
                "sentinel_drift_fraction": 0.001 + index / 10000,
                "valid": True,
            }
            for context in runner.CONTEXTS
            for index in range(10)
        ]
        analysis = {
            "status": "PASS",
            "analyses": [
                {
                    "context": 10219,
                    "delta_bootstrap_one_sided_95_upper_us": 11.3,
                },
                {
                    "context": 37932,
                    "delta_bootstrap_one_sided_95_upper_us": 25.9,
                },
            ],
        }
        value = campaign.result(status="PASS", analysis=analysis)
        self.assertEqual(value["maximum_replacements_total"], 1)
        self.assertEqual(value["contexts"]["10219"]["valid_processes"], 10)
        self.assertEqual(value["contexts"]["37932"]["split_kv"], 40)
        self.assertEqual(value["contexts"]["10219"]["replacement_count"], 0)

    def test_create_container_is_exact_restore_known_and_read_only(self) -> None:
        input_root = self.root / "inputs"
        source = input_root / "source"
        aot = input_root / "aot"
        loader = input_root / "work" / "h43_aot_loader.py"
        for directory in (source, aot, loader.parent):
            directory.mkdir(parents=True, exist_ok=True)
        loader.write_text("# loader\n", encoding="utf-8")
        evidence_parent = self.root / "evidence"
        evidence_parent.mkdir()
        campaign = runner.CompleteI1(
            args_for(input_root, result=evidence_parent / "complete-i1-result.json")
        )
        campaign.host = "beacon-cr-ct13"
        campaign.rank = 0
        campaign.input_manifest_path = input_root / "input-manifest.json"
        campaign.result_path = evidence_parent / "complete-i1-result.json"
        campaign.evidence_root = evidence_parent / "complete-i1-evidence"
        campaign.evidence_root.mkdir()
        image_id = "sha256:" + "1" * 64
        image_ref = "registry/image@sha256:" + "2" * 64
        campaign.manifest = {
            "aot_expected_entries": 14,
            "aot_installed_mla_sha256": "3" * 64,
            "aot_root": "aot",
            "aot_source_manifest_digest": "4" * 64,
            "image_id": image_id,
            "image_ref": image_ref,
            "loader_path": "work/h43_aot_loader.py",
            "native_extension_path": "/opt/omniva/h43/sglang_tq_mla_frontend_sm100_h43_i3_v2.so",
            "native_extension_sha256": "5" * 64,
            "source_root": "source",
        }
        container_id = "6" * 64
        responses = [
            subprocess.CompletedProcess([], 0, container_id + "\n", ""),
            subprocess.CompletedProcess(
                [], 0, f"/{campaign.container_name} {image_id} {image_ref}\n", ""
            ),
            subprocess.CompletedProcess([], 0, container_id + "\n", ""),
        ]
        with mock.patch.object(runner, "docker", side_effect=responses) as docker:
            campaign.create_container()
        create = docker.call_args_list[0]
        command = list(create.args)
        self.assertEqual(command[0], "create")
        self.assertIn(campaign.container_name, command)
        self.assertIn("device=0", command)
        self.assertIn("none", command)
        self.assertNotIn("--cap-add", command)
        self.assertNotIn("--ipc", command)
        self.assertIn(f"{source.resolve()}:/i2:ro", command)
        self.assertIn(f"{aot.resolve()}:/h44-aot:ro", command)
        self.assertEqual(campaign.container_id, container_id)

    def test_input_manifest_binds_full_inventory_and_image(self) -> None:
        gate_root = self.root / "gates"
        input_root = gate_root / "phase0"
        for directory in (
            input_root / "source",
            input_root / "aot",
            input_root / "work",
            input_root / "roundtrip",
            input_root / "provenance",
        ):
            directory.mkdir(parents=True, mode=0o755)
        for path, payload in (
            (input_root / "source" / "module.py", b"source\n"),
            (input_root / "aot" / "h43-aot-manifest.json", b"{}\n"),
            (input_root / "work" / "h43_aot_loader.py", b"loader\n"),
            (input_root / "roundtrip" / "result.json", b"{}\n"),
        ):
            path.write_bytes(payload)
            path.chmod(0o444)
        source_tree = runner.git_tree_oid(input_root / "source")
        commit_payload = f"tree {source_tree}\n\nH44 test commit\n".encode()
        source_commit = runner._git_object_oid("commit", commit_payload).hex()
        commit_object = input_root / runner.SOURCE_COMMIT_OBJECT_PATH
        commit_object.write_bytes(commit_payload)
        commit_object.chmod(0o444)
        with mock.patch.object(runner, "SEALED_OWNER_UID", os.getuid()):
            files = runner.inventory_tree(
                input_root, manifest_name="input-manifest.json"
            )
        image_id = "sha256:" + "7" * 64
        local_image_id = "sha256:" + "6" * 64
        image_ref = "registry/image@sha256:" + "8" * 64
        image_inspect = {
            "Architecture": "arm64",
            "Config": {
                "Labels": {
                    "com.omniva.inference.h43-native-sha256": "2" * 64,
                }
            },
            "Created": "2026-07-31T00:00:00Z",
            "Id": local_image_id,
            "Os": "linux",
            "RepoDigests": [image_ref],
            "RootFS": {"Layers": ["sha256:" + "3" * 64], "Type": "layers"},
        }
        _, _, image_runtime_sha256 = runner.image_identity_from_inspect(
            json.dumps([image_inspect]),
            image_ref=image_ref,
        )
        value = {
            "allowed_nodes": ["ct13", "ct14"],
            "aot_expected_entries": 14,
            "aot_installed_mla_sha256": "9" * 64,
            "aot_root": "aot",
            "aot_source_manifest_digest": "a" * 64,
            "files": files,
            "gate": "complete_i1_inputs",
            "image_id": image_id,
            "image_ref": image_ref,
            "image_runtime_sha256": image_runtime_sha256,
            "loader_path": "work/h43_aot_loader.py",
            "native_extension_path": "/opt/omniva/h43/sglang_tq_mla_frontend_sm100_h43_i3_v2.so",
            "native_extension_sha256": "2" * 64,
            "phase": "phase0_complete_i1",
            "roundtrip_root": "roundtrip",
            "schema_version": 2,
            "source_commit": source_commit,
            "source_commit_object_path": runner.SOURCE_COMMIT_OBJECT_PATH,
            "source_root": "source",
            "source_tree": source_tree,
        }
        path = input_root / "input-manifest.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        path.chmod(0o444)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        image = subprocess.CompletedProcess([], 0, json.dumps([image_inspect]), "")
        with (
            mock.patch.object(runner, "SEALED_OWNER_UID", os.getuid()),
            mock.patch.object(runner, "SEALED_GATE_ROOT", gate_root),
            mock.patch.object(runner, "docker", return_value=image),
        ):
            observed = runner.load_input_manifest(
                path,
                expected_sha256=digest,
                phase="phase0_complete_i1",
                source_commit=source_commit,
            )
        self.assertEqual(observed, value)

    def test_image_identity_requires_exact_repository_digest(self) -> None:
        image_ref = "registry/image@sha256:" + "4" * 64
        image = {
            "Architecture": "arm64",
            "Config": {
                "Labels": {
                    "com.omniva.inference.h43-native-sha256": "5" * 64,
                }
            },
            "Created": "2026-07-31T00:00:00Z",
            "Id": "sha256:" + "6" * 64,
            "Os": "linux",
            "RepoDigests": ["registry/image@sha256:" + "7" * 64],
            "RootFS": {"Layers": [], "Type": "layers"},
        }
        with self.assertRaisesRegex(runner.I1Error, "exact repository digest"):
            runner.image_identity_from_inspect(
                json.dumps([image]),
                image_ref=image_ref,
            )

    def test_write_new_refuses_overwrite(self) -> None:
        path = self.root / "artifact"
        runner.write_new(path, b"first")
        with self.assertRaises(FileExistsError):
            runner.write_new(path, b"second")
        self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main(verbosity=2)
