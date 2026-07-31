#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("seal_h44_complete_i1_inputs.py")
SPEC = importlib.util.spec_from_file_location(
    "seal_h44_complete_i1_inputs", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealer)


class SealCompleteI1InputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gate_root = Path(self.temp.name) / "gates"
        self.root = self.gate_root / ".staging-phase0_complete_i1-test"
        for relative in ("source", "aot", "work", "roundtrip", "provenance"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        tools = self.root / "source" / "benchmark" / "bench_turboquant_mla"
        tools.mkdir(parents=True)
        for name in (
            "run_h44_complete_i1.py",
            "bench_h41_i1_integrated.py",
            "analyze_h41_i1_integrated.py",
        ):
            (tools / name).write_text(f"# {name}\n", encoding="utf-8")
        (self.root / "work" / "h43_aot_loader.py").write_text(
            "# loader\n", encoding="utf-8"
        )
        aot_manifest = {
            "entries": {str(index): {} for index in range(14)},
            "expected_entries": 14,
            "installed_mla_sha256": "a" * 64,
            "source_manifest_digest": "b" * 64,
        }
        (self.root / "aot" / "h43-aot-manifest.json").write_text(
            json.dumps(aot_manifest), encoding="utf-8"
        )
        for context in sealer.runner.CONTEXTS:
            for q_len in (1, 5):
                path = (
                    self.root
                    / "roundtrip"
                    / f"context{context}"
                    / f"q{q_len}"
                    / "result.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")

    def test_sealer_import_does_not_write_runner_bytecode(self) -> None:
        tools = self.root / "import-check"
        tools.mkdir()
        for name in ("seal_h44_complete_i1_inputs.py", "run_h44_complete_i1.py"):
            (tools / name).write_bytes(MODULE_PATH.with_name(name).read_bytes())
        completed = subprocess.run(
            [sys.executable, tools / "seal_h44_complete_i1_inputs.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse((tools / "__pycache__").exists())

    def test_seal_emits_self_verifying_manifest_and_final_paths(self) -> None:
        source_tree = sealer.runner.git_tree_oid(self.root / "source")
        commit_payload = f"tree {source_tree}\n\nH44 test commit\n".encode()
        source_commit = sealer.runner._git_object_oid("commit", commit_payload).hex()
        (self.root / "provenance" / "source-commit-object").write_bytes(commit_payload)
        image_id = "sha256:" + "c" * 64
        native_sha256 = "d" * 64
        image = subprocess.CompletedProcess([], 0, f"{image_id} {native_sha256}\n", "")
        args = argparse.Namespace(
            phase="phase0_complete_i1",
            source_commit=source_commit,
            source_tree=source_tree,
            image_ref="registry/image@sha256:" + "f" * 64,
            root=self.root,
        )
        with (
            mock.patch.object(sealer.os, "geteuid", return_value=0),
            mock.patch.object(sealer.os, "chown"),
            mock.patch.object(sealer.runner, "SEALED_OWNER_UID", os.getuid()),
            mock.patch.object(sealer.runner, "SEALED_GATE_ROOT", self.gate_root),
            mock.patch.object(
                sealer,
                "docker_image_identity",
                return_value=(image_id, native_sha256),
            ),
            mock.patch.object(sealer.runner, "docker", return_value=image),
        ):
            result = sealer.seal(args)
        manifest_path = self.root / "input-manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_tree"], source_tree)
        self.assertEqual(manifest["image_id"], image_id)
        self.assertEqual(result["status"], "PASS")
        final_root = self.gate_root.resolve() / "phase0_complete_i1"
        self.assertEqual(
            result["input_manifest_path"], str(final_root / "input-manifest.json")
        )
        self.assertEqual(
            result["executable_path"],
            str(final_root / "executable" / "run_h44_complete_i1.py"),
        )
        executable = self.root / "executable" / "run_h44_complete_i1.py"
        self.assertTrue(stat.S_IMODE(executable.stat().st_mode) & 0o111)
        self.assertFalse(stat.S_IMODE(manifest_path.stat().st_mode) & 0o222)

    def test_seal_refuses_a_root_not_bound_to_phase(self) -> None:
        args = argparse.Namespace(
            phase="phase0_complete_i1",
            source_commit="e" * 40,
            source_tree="f" * 40,
            image_ref="registry/image@sha256:" + "a" * 64,
            root=self.root.with_name(".staging-wrong-phase-test"),
        )
        args.root.mkdir()
        with (
            mock.patch.object(sealer.os, "geteuid", return_value=0),
            mock.patch.object(sealer.runner, "SEALED_GATE_ROOT", self.gate_root),
            self.assertRaisesRegex(sealer.SealError, "not bound"),
        ):
            sealer.seal(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
