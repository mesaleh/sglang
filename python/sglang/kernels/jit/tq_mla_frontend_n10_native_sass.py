"""Fail-closed static audit for the N0-F native writer and S0-F control."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_EXPECTED_S0_SOURCE_SHA256 = (
    "fdce7462acd97516993eab2263c91af6c7f2ee1ace3d69fa7827834977f3012b"
)
_TEXT_SECTION_HEADER = re.compile(
    r"^//-+\s+(\.text\.\S+)\s+-+\s*$", re.MULTILINE
)
_RESOURCE_RECORD = re.compile(
    r"^ Function (?P<name>[^\n]+):\n"
    r"  REG:(?P<registers>\d+) STACK:(?P<stack>\d+) "
    r"SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)
_INSTRUCTION_ADDRESS = re.compile(r"/\*[0-9A-Fa-f]+\*/")
_INSTRUCTION = re.compile(
    r"/\*[0-9A-Fa-f]+\*/\s+(?:@!?U?P(?:T|[0-9]+)\s+)?"
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
        timeout=180,
    )
    return result.stdout


def _extract(shared_object: Path) -> tuple[str, str, str]:
    with tempfile.TemporaryDirectory(prefix="n10-n0f-static-") as directory:
        extraction = Path(directory)
        _run(
            ["cuobjdump", "--extract-elf", "all", str(shared_object)],
            cwd=extraction,
        )
        cubins = sorted(extraction.glob("*.cubin"))
        if len(cubins) != 1 or not cubins[0].name.endswith(".sm_100.cubin"):
            raise AssertionError(f"unexpected SM100-family cubins: {cubins}")
        cubin = cubins[0]
        sass = _run(["nvdisasm", "-g", str(cubin)])
        cubin_sha256 = _sha256(cubin)
    resources = _run(["cuobjdump", "-res-usage", str(shared_object)])
    if re.search(r"^arch = sm_100f$", resources, re.MULTILINE) is None:
        raise AssertionError(f"{shared_object.name} is not an sm_100f fatbin")
    return sass, resources, cubin_sha256


def _kernel_sections(sass: str) -> dict[str, str]:
    headers = list(_TEXT_SECTION_HEADER.finditer(sass))
    sections: dict[str, str] = {}
    for index, header in enumerate(headers):
        name = header.group(1)
        if "tq_mla_frontend_" not in name and "tq_mla_cache_writer_" not in name:
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(sass)
        if name in sections:
            raise AssertionError(f"duplicate writer section: {name}")
        sections[name] = sass[header.start() : end]
    if len(sections) != 20:
        raise AssertionError(f"expected 20 writer specializations, got {len(sections)}")
    return sections


def _resource_records(resources: str) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    for match in _RESOURCE_RECORD.finditer(resources):
        name = match.group("name")
        if "tq_mla_frontend_" not in name and "tq_mla_cache_writer_" not in name:
            continue
        records[name] = {
            "registers": int(match.group("registers")),
            "stack_bytes": int(match.group("stack")),
            "shared_bytes": int(match.group("shared")),
            "local_bytes": int(match.group("local")),
        }
    if len(records) != 20:
        raise AssertionError(f"expected 20 resource records, got {len(records)}")
    for name, record in records.items():
        if (
            record["registers"] > 56
            or record["stack_bytes"] != 0
            or record["local_bytes"] != 0
        ):
            raise AssertionError(f"resource gate failed for {name}: {record}")
    return records


def _instruction_opcodes(text: str, context: str) -> list[str]:
    opcodes: list[str] = []
    for line in text.splitlines():
        if _INSTRUCTION_ADDRESS.search(line) is None:
            continue
        match = _INSTRUCTION.search(line)
        if match is None:
            raise AssertionError(f"unparsed SASS instruction in {context}: {line}")
        opcodes.append(match.group(1))
    return opcodes


def _audit_native_section(name: str, section: str) -> dict[str, int]:
    lines = section.splitlines()
    encode = [
        index
        for index, line in enumerate(lines)
        if "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C" in line
    ]
    decode = [
        index
        for index, line in enumerate(lines)
        if "F2FP.F16.E2M1.UNPACK_B" in line
    ]
    if len(encode) != 8 or len(decode) != 8:
        raise AssertionError(
            f"{name} native conversion count is encode={len(encode)} "
            f"decode={len(decode)}"
        )
    for pair, (encode_line, decode_line) in enumerate(zip(encode, decode)):
        if encode_line >= decode_line:
            raise AssertionError(f"{name} pair {pair} decode precedes encode")
        if pair + 1 < len(encode) and decode_line >= encode[pair + 1]:
            raise AssertionError(f"{name} native encode/decode pairs interleave")
        correction_region = "\n".join(lines[encode_line + 1 : decode_line])
        opcodes = _instruction_opcodes(
            correction_region, f"{name} pair {pair} correction"
        )
        forbidden = [
            opcode
            for opcode in opcodes
            if opcode.startswith(("BRA", "CALL", "LDG", "LDL", "STL"))
        ]
        if forbidden:
            raise AssertionError(
                f"{name} pair {pair} correction is not register-only: {forbidden}"
            )
    all_opcodes = _instruction_opcodes(section, name)
    local_ops = [
        opcode for opcode in all_opcodes if opcode.startswith(("LDL", "STL"))
    ]
    if local_ops:
        raise AssertionError(f"{name} contains local-memory ops: {local_ops}")
    return {
        "native_encode": len(encode),
        "native_decode": len(decode),
        "branch_count": sum(opcode.startswith("BRA") for opcode in all_opcodes),
        "call_count": sum(opcode.startswith("CALL") for opcode in all_opcodes),
    }


def audit_modules(
    native_extension: str | Path,
    s0f_extension: str | Path,
    native_source: str | Path,
    s0_source: str | Path,
) -> dict[str, Any]:
    """Audit all generated specializations and immutable-control identity."""

    native_so = Path(native_extension).resolve(strict=True)
    s0f_so = Path(s0f_extension).resolve(strict=True)
    native_source_path = Path(native_source).resolve(strict=True)
    s0_source_path = Path(s0_source).resolve(strict=True)
    if _sha256(s0_source_path) != _EXPECTED_S0_SOURCE_SHA256:
        raise AssertionError("S0-F CUDA body differs from immutable S0")
    native_text = native_source_path.read_text()
    forbidden_source = [
        token
        for token in (
            "boundaries",
            "levels",
            "storage_codes",
            "select_native_e2m1_bin",
        )
        if token in native_text
    ]
    if forbidden_source:
        raise AssertionError(f"N0-F retains dynamic codebook paths: {forbidden_source}")

    native_sass, native_resources, native_cubin_sha256 = _extract(native_so)
    s0f_sass, s0f_resources, s0f_cubin_sha256 = _extract(s0f_so)
    native_sections = _kernel_sections(native_sass)
    s0f_sections = _kernel_sections(s0f_sass)
    native_records = _resource_records(native_resources)
    s0f_records = _resource_records(s0f_resources)
    native_metrics = {
        name: _audit_native_section(name, section)
        for name, section in native_sections.items()
    }
    if any(re.search(r"\bF2FP\b[^\n]*E2M1", section) for section in s0f_sections.values()):
        raise AssertionError("table-driven S0-F unexpectedly contains native E2M1")
    return {
        "native_cubin_sha256": native_cubin_sha256,
        "s0f_cubin_sha256": s0f_cubin_sha256,
        "native_sass_sha256": hashlib.sha256(native_sass.encode()).hexdigest(),
        "s0f_sass_sha256": hashlib.sha256(s0f_sass.encode()).hexdigest(),
        "native_source_sha256": _sha256(native_source_path),
        "s0_source_sha256": _sha256(s0_source_path),
        "native_specializations": len(native_sections),
        "s0f_specializations": len(s0f_sections),
        "native_max_registers": max(
            record["registers"] for record in native_records.values()
        ),
        "s0f_max_registers": max(
            record["registers"] for record in s0f_records.values()
        ),
        "native_total_encode": sum(
            metric["native_encode"] for metric in native_metrics.values()
        ),
        "native_total_decode": sum(
            metric["native_decode"] for metric in native_metrics.values()
        ),
        "native_resources": native_records,
        "s0f_resources": s0f_records,
        "native_codegen": native_metrics,
    }


__all__ = ["audit_modules"]
