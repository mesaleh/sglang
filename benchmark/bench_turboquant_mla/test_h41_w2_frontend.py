"""Correctness and safety gate for the H41 SM100 combined MLA front end."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import dataclass
from typing import Any

import torch

from sglang.jit_kernel.tq_mla_frontend import tq_mla_frontend_out
from sglang.srt.layers.attention.tokenspeed_mla_backend import _quantize_tq4_query
from sglang.srt.layers.attention.triton_ops.turboquant_quantize import (
    fused_turboquant_quantize_and_store,
)
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

LATENT = 512
ROPE = 64
HEADS = 8
FP8 = torch.float8_e4m3fn


@dataclass
class Buffers:
    query: torch.Tensor
    packed: torch.Tensor
    scale: torch.Tensor
    rope: torch.Tensor
    status: torch.Tensor
    guards: tuple[torch.Tensor, ...]


def raw_fp8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def allocate_guarded(tokens: int, pool_size: int, device: torch.device) -> Buffers:
    query_base = torch.empty(tokens + 2, HEADS, LATENT + ROPE, dtype=FP8, device=device)
    packed_base = torch.full(
        (pool_size + 2, 1, LATENT // 2), 0xA5, dtype=torch.uint8, device=device
    )
    scale_base = torch.full(
        (pool_size + 2, 1), -3.25, dtype=torch.bfloat16, device=device
    )
    rope_base = torch.empty(pool_size + 2, 1, ROPE, dtype=FP8, device=device)
    raw_fp8(query_base).fill_(0x5A)
    raw_fp8(rope_base).fill_(0xA5)
    return Buffers(
        query=query_base[1:-1],
        packed=packed_base[1:-1],
        scale=scale_base[1:-1],
        rope=rope_base[1:-1],
        status=torch.zeros(1, dtype=torch.int32, device=device),
        guards=(
            raw_fp8(query_base[0]).clone(),
            raw_fp8(query_base[-1]).clone(),
            packed_base[0].clone(),
            packed_base[-1].clone(),
            scale_base[0].clone(),
            scale_base[-1].clone(),
            raw_fp8(rope_base[0]).clone(),
            raw_fp8(rope_base[-1]).clone(),
            query_base,
            packed_base,
            scale_base,
            rope_base,
        ),
    )


def assert_guards(buffers: Buffers) -> None:
    (
        query_low,
        query_high,
        packed_low,
        packed_high,
        scale_low,
        scale_high,
        rope_low,
        rope_high,
        query_base,
        packed_base,
        scale_base,
        rope_base,
    ) = buffers.guards
    assert torch.equal(raw_fp8(query_base[0]), query_low)
    assert torch.equal(raw_fp8(query_base[-1]), query_high)
    assert torch.equal(packed_base[0], packed_low)
    assert torch.equal(packed_base[-1], packed_high)
    assert torch.equal(scale_base[0], scale_low)
    assert torch.equal(scale_base[-1], scale_high)
    assert torch.equal(raw_fp8(rope_base[0]), rope_low)
    assert torch.equal(raw_fp8(rope_base[-1]), rope_high)


def make_inputs(
    tokens: int,
    device: torch.device,
    generator: torch.Generator,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shapes = (
        (tokens, HEADS, LATENT),
        (tokens, HEADS, ROPE),
        (tokens, 1, LATENT),
        (tokens, 1, ROPE),
    )
    if kind == "zero":
        return tuple(
            torch.zeros(shape, dtype=torch.bfloat16, device=device) for shape in shapes
        )  # type: ignore[return-value]
    if kind == "repeated":
        base = [
            torch.empty((1, *shape[1:]), dtype=torch.bfloat16, device=device).uniform_(
                -0.25, 0.25, generator=generator
            )
            for shape in shapes
        ]
        return tuple(value.expand(shape).contiguous() for value, shape in zip(base, shapes))  # type: ignore[return-value]
    values = [
        torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            mean=0.0, std=0.125, generator=generator
        )
        for shape in shapes
    ]
    if kind == "impulse":
        for value in values:
            value.zero_()
        values[0][..., 0] = 1.0
        values[1][..., 1] = -1.0
        values[2][..., 127] = 1.0
        values[3][..., 63] = -1.0
    return tuple(values)  # type: ignore[return-value]


def query_reference(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    config: TurboQuantConfig,
    rotation_fused: bool,
) -> torch.Tensor:
    latent = query_latent if rotation_fused else config.rotate_query(query_latent)
    return _quantize_tq4_query(torch.cat((latent, query_rope), dim=-1), LATENT, False)


def rope_reference(cache_rope: torch.Tensor) -> torch.Tensor:
    tokens = cache_rope.shape[0]
    carrier = torch.zeros(
        tokens, HEADS, LATENT + ROPE, dtype=torch.bfloat16, device=cache_rope.device
    )
    carrier[:, 0, LATENT:] = cache_rope[:, 0]
    return _quantize_tq4_query(carrier, LATENT, False)[:, 0:1, LATENT:].clone()


def writer_reference(
    cache_latent: torch.Tensor,
    cache_rope: torch.Tensor,
    locations: torch.Tensor,
    config: TurboQuantConfig,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = cache_latent.shape[0]
    packed = torch.full(
        (pool_size, 1, LATENT // 2), 0xA5, dtype=torch.uint8, device=cache_latent.device
    )
    scale = torch.full(
        (pool_size, 1), -3.25, dtype=torch.bfloat16, device=cache_latent.device
    )
    rope_bf16 = torch.zeros(
        (pool_size, 1, ROPE), dtype=torch.bfloat16, device=cache_latent.device
    )
    unit = torch.empty(tokens, 1, LATENT, dtype=torch.float32, device=cache_latent.device)
    norms = torch.empty(tokens, 1, dtype=torch.float32, device=cache_latent.device)
    rotated = torch.empty_like(unit)
    fused_turboquant_quantize_and_store(
        cache_latent,
        config.signs1,
        config.signs2,
        config.k_quant_centroids,
        config.k_boundaries,
        4,
        packed,
        scale,
        locations,
        pre_unit=unit,
        pre_norms=norms,
        pre_y=rotated,
        storage_code_lut=config.k_storage_code_lut,
        decode_centroids=config.k_centroids,
        dequant_scale_multiplier=config.k_dequant_scale_multiplier,
        rope_src=cache_rope,
        rope_buffer=rope_bf16,
    )
    return packed, scale, rope_reference(cache_rope)


def launch(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    locations: torch.Tensor,
    config: TurboQuantConfig,
    buffers: Buffers,
    rotation_fused: bool,
    warps: int,
    strict: bool = False,
) -> None:
    query_latent, query_rope, cache_latent, cache_rope = inputs
    assert config.k_storage_code_lut is not None
    tq_mla_frontend_out(
        query_latent,
        query_rope,
        cache_latent,
        cache_rope,
        locations,
        config.signs1,
        config.signs2,
        config.k_boundaries,
        config.k_quant_centroids,
        config.k_storage_code_lut,
        buffers.query,
        buffers.packed,
        buffers.scale,
        buffers.rope,
        buffers.status,
        scale_multiplier=config.k_dequant_scale_multiplier,
        rotation_fused=rotation_fused,
        num_warps=warps,
        strict=strict,
    )


def assert_case(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    locations: torch.Tensor,
    config: TurboQuantConfig,
    rotation_fused: bool,
    warps: int,
    pool_size: int,
) -> dict[str, Any]:
    buffers = allocate_guarded(inputs[0].shape[0], pool_size, inputs[0].device)
    query_expected = query_reference(inputs[0], inputs[1], config, rotation_fused)
    packed_expected, scale_expected, rope_expected = writer_reference(
        inputs[2], inputs[3], locations, config, pool_size
    )
    launch(inputs, locations, config, buffers, rotation_fused, warps)
    torch.cuda.synchronize()
    query_mismatches = int(
        (raw_fp8(buffers.query) != raw_fp8(query_expected)).sum().item()
    )
    selected_packed = buffers.packed[locations]
    selected_scale = buffers.scale[locations]
    selected_rope = buffers.rope[locations]
    packed_mismatches = int(
        (selected_packed != packed_expected[locations]).sum().item()
    )
    scale_byte_mismatches = int(
        (
            selected_scale.view(torch.uint16)
            != scale_expected[locations].view(torch.uint16)
        )
        .sum()
        .item()
    )
    rope_mismatches = int(
        (raw_fp8(selected_rope) != raw_fp8(rope_expected)).sum().item()
    )
    scale_max_abs = float(
        (selected_scale.float() - scale_expected[locations].float()).abs().max().item()
    )
    assert query_mismatches == 0, query_mismatches
    assert packed_mismatches == 0, packed_mismatches
    assert scale_byte_mismatches == 0, (scale_byte_mismatches, scale_max_abs)
    assert rope_mismatches == 0, rope_mismatches
    assert int(buffers.status.item()) == 0
    assert_guards(buffers)
    return {
        "query_mismatches": query_mismatches,
        "packed_mismatches": packed_mismatches,
        "scale_byte_mismatches": scale_byte_mismatches,
        "scale_max_abs": scale_max_abs,
        "rope_mismatches": rope_mismatches,
    }


def test_graph_replay(
    config: TurboQuantConfig,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    for tokens in (1, 5):
        inputs = make_inputs(tokens, device, generator, "random")
        pool_size = tokens + 7
        locations = torch.arange(tokens, dtype=torch.int64, device=device) + 3
        buffers = allocate_guarded(tokens, pool_size, device)
        launch(inputs, locations, config, buffers, True, 8)
        torch.cuda.synchronize()
        eager_before = torch.cuda.memory_allocated(device)
        for _ in range(100):
            launch(inputs, locations, config, buffers, True, 8)
        torch.cuda.synchronize()
        eager_after = torch.cuda.memory_allocated(device)
        assert eager_after == eager_before
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(inputs, locations, config, buffers, True, 8)
        graph.replay()
        torch.cuda.synchronize()
        replay_before = torch.cuda.memory_allocated(device)
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        replay_after = torch.cuda.memory_allocated(device)
        assert replay_after == replay_before
        expected = query_reference(inputs[0], inputs[1], config, True)
        assert torch.equal(raw_fp8(buffers.query), raw_fp8(expected))
        assert int(buffers.status.item()) == 0
        assert_guards(buffers)


def test_invalid_sticky(
    config: TurboQuantConfig,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    inputs = make_inputs(3, device, generator, "random")
    pool_size = 8
    buffers = allocate_guarded(3, pool_size, device)
    initial_packed = buffers.packed.clone()
    initial_scale = buffers.scale.clone()
    initial_rope = raw_fp8(buffers.rope).clone()
    invalid = torch.tensor([-1, pool_size, pool_size + 17], dtype=torch.int64, device=device)
    launch(inputs, invalid, config, buffers, True, 8)
    torch.cuda.synchronize()
    assert int(buffers.status.item()) == 1
    assert torch.equal(buffers.packed, initial_packed)
    assert torch.equal(buffers.scale, initial_scale)
    assert torch.equal(raw_fp8(buffers.rope), initial_rope)
    expected_query = query_reference(inputs[0], inputs[1], config, True)
    assert torch.equal(raw_fp8(buffers.query), raw_fp8(expected_query))
    valid = torch.tensor([0, 3, pool_size - 1], dtype=torch.int64, device=device)
    launch(inputs, valid, config, buffers, True, 8)
    torch.cuda.synchronize()
    assert int(buffers.status.item()) == 1
    assert_guards(buffers)


def expect_error(fn, text: str) -> None:
    try:
        fn()
    except (RuntimeError, TypeError, ValueError) as error:
        if text not in str(error):
            raise AssertionError(f"expected {text!r} in {error!r}") from error
    else:
        raise AssertionError(f"expected an error containing {text!r}")


def test_wrapper_rejections(
    config: TurboQuantConfig,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    inputs = make_inputs(1, device, generator, "random")
    locations = torch.tensor([1], dtype=torch.int64, device=device)
    buffers = allocate_guarded(1, 4, device)

    def invoke(
        changed_inputs=inputs,
        changed_locations=locations,
        changed_query_out=buffers.query,
        warps=8,
    ) -> None:
        query_latent, query_rope, cache_latent, cache_rope = changed_inputs
        tq_mla_frontend_out(
            query_latent,
            query_rope,
            cache_latent,
            cache_rope,
            changed_locations,
            config.signs1,
            config.signs2,
            config.k_boundaries,
            config.k_quant_centroids,
            config.k_storage_code_lut,
            changed_query_out,
            buffers.packed,
            buffers.scale,
            buffers.rope,
            buffers.status,
            scale_multiplier=config.k_dequant_scale_multiplier,
            rotation_fused=True,
            num_warps=warps,
        )

    noncontiguous = torch.empty(
        1, HEADS, LATENT * 2, dtype=torch.bfloat16, device=device
    )[..., ::2]
    expect_error(lambda: invoke((noncontiguous, *inputs[1:])), "must be contiguous")
    expect_error(
        lambda: invoke((inputs[0].float(), *inputs[1:])), "must be bfloat16"
    )
    expect_error(lambda: invoke(changed_locations=locations.int()), "must be int64")
    expect_error(
        lambda: invoke(changed_query_out=torch.empty_like(inputs[0])),
        "must be float8_e4m3fn",
    )
    alias = (
        inputs[0]
        .view(torch.uint8)
        .reshape(-1)[: HEADS * (LATENT + ROPE)]
        .view(FP8)
        .reshape(1, HEADS, LATENT + ROPE)
    )
    expect_error(lambda: invoke(changed_query_out=alias), "must not overlap")
    expect_error(lambda: invoke(warps=3), "num_warps must be")


def test_special_fp8(
    config: TurboQuantConfig,
    device: torch.device,
) -> None:
    special = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-9,
            -(2.0**-9),
            448.0,
            -448.0,
            500.0,
            -500.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    inputs = (
        special.repeat((HEADS * LATENT + special.numel() - 1) // special.numel())[
            : HEADS * LATENT
        ].view(1, HEADS, LATENT),
        special.repeat((HEADS * ROPE + special.numel() - 1) // special.numel())[
            : HEADS * ROPE
        ].view(1, HEADS, ROPE),
        torch.zeros(1, 1, LATENT, dtype=torch.bfloat16, device=device),
        special.repeat((ROPE + special.numel() - 1) // special.numel())[:ROPE].view(
            1, 1, ROPE
        ),
    )
    locations = torch.tensor([2], dtype=torch.int64, device=device)
    buffers = allocate_guarded(1, 5, device)
    launch(inputs, locations, config, buffers, True, 8)
    torch.cuda.synchronize()
    expected_query = query_reference(inputs[0], inputs[1], config, True)
    expected_rope = rope_reference(inputs[3])
    assert torch.equal(raw_fp8(buffers.query), raw_fp8(expected_query))
    assert torch.equal(raw_fp8(buffers.rope[locations]), raw_fp8(expected_rope))
    assert int(buffers.status.item()) == 0
    assert_guards(buffers)


def run_strict_invalid(
    config: TurboQuantConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    inputs = make_inputs(1, device, generator, "random")
    buffers = allocate_guarded(1, 4, device)
    invalid = torch.tensor([4], dtype=torch.int64, device=device)
    launch(inputs, invalid, config, buffers, True, 8, strict=True)
    try:
        torch.cuda.synchronize()
    except RuntimeError as error:
        return {"strict_trap_observed": True, "error": str(error)}
    raise AssertionError("strict invalid-location launch did not trap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("correctness", "strict-invalid", "compile"), default="correctness"
    )
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H41 W2 requires SM100")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    config = TurboQuantConfig(
        bit_width=4,
        head_dim=LATENT,
        device=device,
        k_bit_width=4,
        v_bit_width=4,
        uniform=False,
        e2m1=True,
    )

    if args.mode == "strict-invalid":
        result = run_strict_invalid(config, device, generator)
        print(json.dumps(result, sort_keys=True))
        return

    compile_inputs = make_inputs(1, device, generator, "random")
    compile_buffers = allocate_guarded(1, 4, device)
    compile_locations = torch.tensor([1], dtype=torch.int64, device=device)
    launch(compile_inputs, compile_locations, config, compile_buffers, True, 8)
    torch.cuda.synchronize()
    if args.mode == "compile":
        print(json.dumps({"compiled": True}, sort_keys=True))
        return

    cases: list[dict[str, Any]] = []
    for tokens in (1, 2, 5, 7, 10, 15, 20, 25, 30, 35, 40):
        pool_size = tokens + 11
        locations = torch.randperm(pool_size, device=device)[:tokens].to(torch.int64)
        for kind in ("random", "zero", "impulse", "repeated"):
            inputs = make_inputs(tokens, device, generator, kind)
            for rotation_fused in (True, False):
                for warps in (1, 2, 4, 8):
                    metrics = assert_case(
                        inputs,
                        locations,
                        config,
                        rotation_fused,
                        warps,
                        pool_size,
                    )
                    cases.append(
                        {
                            "tokens": tokens,
                            "kind": kind,
                            "rotation_fused": rotation_fused,
                            "warps": warps,
                            **metrics,
                        }
                    )
    test_graph_replay(config, device, generator)
    test_invalid_sticky(config, device, generator)
    test_wrapper_rejections(config, device, generator)
    test_special_fp8(config, device)
    result = {
        "status": "PASS",
        "experiment": "H41_W2_FRONTEND_CORRECTNESS",
        "pid": os.getpid(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "case_count": len(cases),
        "cases": cases,
        "graph_rows": [1, 5],
        "invalid_sticky": "PASS",
        "wrapper_rejections": "PASS",
        "special_fp8": "PASS",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
