"""Fail-closed SASS audit for the research-only SM100f E2M1 probe."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_KERNEL_TOKEN = "e2m1_native_probe_kernel"
_TEXT_SECTION_HEADER = re.compile(r"^//-+\s+\.text\.[^\n]+$", re.MULTILINE)
_INSTRUCTION = re.compile(
    r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?"
    r"([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\b"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def _kernel_section(sass: str) -> str:
    # The mangled kernel name also appears in debug and metadata sections.
    # Select the unique executable .text section header itself rather than the
    # first textual occurrence of the token.
    matches = [
        match
        for match in _TEXT_SECTION_HEADER.finditer(sass)
        if _KERNEL_TOKEN in match.group(0)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one probe-kernel .text section, observed {len(matches)}"
        )
    start = matches[0].start()
    end = sass.find("\n//--------------------- ", matches[0].end())
    return sass[start:] if end < 0 else sass[start:end]


def _resource_record(resources: str) -> dict[str, int]:
    pattern = re.compile(
        rf"Function [^\n]*{_KERNEL_TOKEN}[^\n]*:\s*\n\s*"
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"
    )
    matches = pattern.findall(resources)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one probe resource record, observed {len(matches)}"
        )
    registers, stack, shared, local = (int(value) for value in matches[0])
    if stack != 0 or local != 0:
        raise AssertionError(
            f"probe spills: registers={registers} stack={stack} local={local}"
        )
    return {
        "registers": registers,
        "stack_bytes": stack,
        "shared_bytes": shared,
        "local_bytes": local,
    }


def audit_extension(extension: str | Path) -> dict[str, Any]:
    """Prove native FP4 conversion and reject emulation/spill/branch paths."""

    shared_object = Path(extension).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="n10-n0-gate0-") as directory:
        extraction = Path(directory)
        _run(
            ["cuobjdump", "--extract-elf", "all", str(shared_object)],
            cwd=extraction,
        )
        cubins = sorted(extraction.glob("*.cubin"))
        if len(cubins) != 1:
            raise AssertionError(f"expected one SM100 cubin, found {cubins}")
        cubin = cubins[0]
        # CUDA 12.9 preserves the family-specific target in the fatbin resource
        # record but names an extracted SM100-family ELF with the base sm_100
        # architecture.  Pair this base-name check with the authoritative
        # `arch = sm_100f` resource check below instead of expecting an
        # sm_100f suffix that cuobjdump does not emit.
        if not cubin.name.endswith(".sm_100.cubin"):
            raise AssertionError(f"extracted cubin is not SM100-family: {cubin.name}")
        sass = _run(["nvdisasm", "-g", str(cubin)])
        resources = _run(["cuobjdump", "-res-usage", str(shared_object)])
        cubin_sha256 = _sha256(cubin)

    if re.search(r"^\s*arch\s*=\s*sm_100f\s*$", resources, re.MULTILINE) is None:
        raise AssertionError("cuobjdump did not identify the cubin as sm_100f")
    section = _kernel_section(sass)
    opcodes = _INSTRUCTION.findall(section)
    native_lines = [
        line.strip()
        for line in section.splitlines()
        if re.search(r"\bF2F[A-Z0-9_.]*\b", line)
        and re.search(r"(?:E2M1|FP4)", line)
    ]
    if len(native_lines) != 2:
        raise AssertionError(
            "expected exactly one native E2M1 encode and one native decode; "
            f"observed {native_lines}"
        )
    calls = [opcode for opcode in opcodes if opcode.startswith("CALL")]
    local_ops = [
        opcode for opcode in opcodes if opcode.startswith(("LDL", "STL"))
    ]
    branches = [opcode for opcode in opcodes if opcode.startswith("BRA")]
    float_compares = [
        opcode for opcode in opcodes if opcode.startswith(("FSET", "FSETP"))
    ]
    if calls or local_ops or len(branches) > 1 or float_compares:
        raise AssertionError(
            "probe SASS gate failed: "
            f"calls={calls} local={local_ops} branches={branches} "
            f"float_compares={float_compares}"
        )
    resource_record = _resource_record(resources)
    return {
        "cubin_sha256": cubin_sha256,
        "cubin_name": cubin.name,
        "architecture": "sm_100f",
        "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
        "native_instructions": native_lines,
        "branch_count": len(branches),
        "call_count": len(calls),
        "float_compare_count": len(float_compares),
        **resource_record,
    }


__all__ = ["audit_extension"]
