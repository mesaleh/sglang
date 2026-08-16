"""Measure folding N10 latent rotations into absorbed MLA weights."""

import json
import math
import os
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

# Importing the TQ config first avoids the inherited partial-overlay cycle
# between fp8_kernel and quantization.__init__.
# isort: off
from sglang.srt.layers.quantization.kv_turboquant import NativeE2M1MLAConfig

# isort: on

from sglang.kernels.ops.gemm import bmm_fp8
from sglang.kernels.ops.quantization.fp8_kernel import per_tensor_quant_mla_fp8
from sglang.kernels.ops.quantization.hadamard import hadamard_transform_with_signs

HEADS = 8
LATENT = 512
VALUE = 128
FP8_MAX = 448.0


def _quantize_weight(weight: torch.Tensor):
    scale = weight.float().abs().amax().clamp_min(1e-12) / FP8_MAX
    # The production FlashInfer BMM consumes the loader's transposed view:
    # logical [H, K, N], physical contiguous [H, N, K].
    quantized = (
        torch.clamp(
            weight.float().transpose(1, 2).contiguous() / scale,
            -FP8_MAX,
            FP8_MAX,
        )
        .to(torch.float8_e4m3fn)
        .transpose(1, 2)
    )
    return quantized, scale.reshape(1).to(torch.float32)


def _capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def _measure(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / replays


def _paired(candidate, control, *, windows: int, replays: int):
    candidate_samples = []
    control_samples = []
    for window in range(windows):
        order = (
            ((candidate, candidate_samples), (control, control_samples))
            if window % 2 == 0
            else ((control, control_samples), (candidate, candidate_samples))
        )
        for graph, samples in order:
            samples.append(_measure(graph, replays))
    ratios = [
        math.log(candidate_us / control_us)
        for candidate_us, control_us in zip(candidate_samples, control_samples)
    ]
    critical = 2.045229642 if windows == 30 else 2.262157163
    mean_log = statistics.fmean(ratios)
    standard_error = statistics.stdev(ratios) / math.sqrt(windows)
    return {
        "candidate_us": statistics.fmean(candidate_samples),
        "control_us": statistics.fmean(control_samples),
        "ratio": math.exp(mean_log),
        "lower": math.exp(mean_log - critical * standard_error),
        "upper": math.exp(mean_log + critical * standard_error),
    }


def _error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = candidate.float() - reference.float()
    flat_candidate = candidate.float().flatten()
    flat_reference = reference.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        flat_candidate, flat_reference, dim=0
    )
    return {
        "mse": delta.square().mean().item(),
        "rel_l2": (delta.norm() / flat_reference.norm().clamp_min(1e-12)).item(),
        "max_abs": delta.abs().max().item(),
        "mean_error": delta.mean().item(),
        "cosine": cosine.item(),
    }


def _numerics(config: NativeE2M1MLAConfig, rows: int):
    torch.manual_seed(20260816)
    device = torch.device("cuda")
    rotated = torch.randn(rows, HEADS, LATENT, dtype=torch.bfloat16, device=device)
    true_weight = torch.randn(
        HEADS, LATENT, VALUE, dtype=torch.bfloat16, device=device
    ) / math.sqrt(LATENT)
    weight_fp8, weight_scale = _quantize_weight(true_weight)
    effective_weight = weight_fp8.float() * weight_scale
    folded_fp32 = config.fuse_inverse_rotation_into_mla_v_weight(
        effective_weight
    ).float()

    original_fp32 = hadamard_transform_with_signs(
        rotated.float(),
        config.signs2,
        config.signs1,
        scale=1.0 / math.sqrt(LATENT),
    )
    algebra_reference = torch.bmm(original_fp32.transpose(0, 1), effective_weight)
    algebra_folded = torch.bmm(rotated.float().transpose(0, 1), folded_fp32)
    torch.testing.assert_close(algebra_folded, algebra_reference, rtol=2e-5, atol=2e-5)

    original_bf16 = original_fp32.to(torch.bfloat16)
    scale_buffer = torch.zeros(1, dtype=torch.float32, device=device)
    original_fp8, original_scale = per_tensor_quant_mla_fp8(
        original_bf16.transpose(0, 1), scale_buffer
    )
    production_reference = bmm_fp8(
        original_fp8,
        weight_fp8,
        original_scale,
        weight_scale,
        torch.bfloat16,
    )

    folded_bf16_weight = folded_fp32.to(torch.bfloat16)
    folded_bf16 = torch.bmm(rotated.transpose(0, 1), folded_bf16_weight)
    folded_fp8_weight, folded_fp8_weight_scale = _quantize_weight(folded_fp32)
    rotated_scale_buffer = torch.zeros(1, dtype=torch.float32, device=device)
    rotated_fp8, rotated_scale = per_tensor_quant_mla_fp8(
        rotated.transpose(0, 1), rotated_scale_buffer
    )
    folded_fp8 = bmm_fp8(
        rotated_fp8,
        folded_fp8_weight,
        rotated_scale,
        folded_fp8_weight_scale,
        torch.bfloat16,
    )
    torch.cuda.synchronize()

    for name, output in (
        ("production_reference", production_reference),
        ("folded_bf16", folded_bf16),
        ("folded_fp8", folded_fp8),
    ):
        versus_ideal = _error(output, algebra_reference)
        versus_production = _error(output, production_reference)
        print(
            "TQ_ABSORBED_FOLD_NUMERICS "
            f"variant={name} rows={rows} "
            + " ".join(
                f"ideal_{key}={value:.9g}" for key, value in versus_ideal.items()
            )
            + " "
            + " ".join(
                f"production_{key}={value:.9g}"
                for key, value in versus_production.items()
            ),
            flush=True,
        )


def _load_real_w_vc(model_dir: Path, layer_id: int, head_start: int) -> torch.Tensor:
    key = f"language_model.model.layers.{layer_id}.self_attn.kv_b_proj.weight"
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_path = model_dir / index["weight_map"][key]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        kv_b = handle.get_tensor(key)
    w_vc = kv_b.unflatten(0, (-1, 256)).split([128, 128], dim=1)[1]
    local = w_vc[head_start : head_start + HEADS]
    if tuple(local.shape) != (HEADS, VALUE, LATENT):
        raise ValueError(f"unexpected local w_vc shape {tuple(local.shape)} for {key}")
    # Match the production loader's logical [H, K, N] layout.
    return local.contiguous().transpose(1, 2).to("cuda")


def _real_weight_numerics(
    config: NativeE2M1MLAConfig,
    model_dir: Path,
    layer_ids: list[int],
    rows: int,
    head_start: int,
):
    for layer_id in layer_ids:
        weight = _load_real_w_vc(model_dir, layer_id, head_start)
        folded_fp32 = config.fuse_inverse_rotation_into_mla_v_weight(
            weight.float()
        ).float()
        folded_bf16_weight = folded_fp32.to(torch.bfloat16)
        for seed in (20260816, 20260817, 20260818):
            torch.manual_seed(seed + layer_id)
            rotated = torch.randn(
                rows, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
            )
            original_fp32 = hadamard_transform_with_signs(
                rotated.float(),
                config.signs2,
                config.signs1,
                scale=1.0 / math.sqrt(LATENT),
            )
            ideal = torch.bmm(original_fp32.transpose(0, 1), weight.float())
            algebra_folded = torch.bmm(rotated.float().transpose(0, 1), folded_fp32)
            torch.testing.assert_close(algebra_folded, ideal, rtol=2e-5, atol=2e-5)

            original_bf16 = hadamard_transform_with_signs(
                rotated,
                config.signs2,
                config.signs1,
                scale=1.0 / math.sqrt(LATENT),
            )
            production = torch.bmm(original_bf16.transpose(0, 1), weight)
            folded = torch.bmm(rotated.transpose(0, 1), folded_bf16_weight)
            torch.cuda.synchronize()
            production_error = _error(production, ideal)
            folded_error = _error(folded, ideal)
            versus_production = _error(folded, production)
            print(
                "TQ_ABSORBED_FOLD_REAL_NUMERICS "
                f"layer={layer_id} head_start={head_start} seed={seed} "
                f"rows={rows} weight_dtype={weight.dtype} "
                + " ".join(
                    f"production_{key}={value:.9g}"
                    for key, value in production_error.items()
                )
                + " "
                + " ".join(
                    f"folded_{key}={value:.9g}" for key, value in folded_error.items()
                )
                + " "
                + " ".join(
                    f"delta_{key}={value:.9g}"
                    for key, value in versus_production.items()
                ),
                flush=True,
            )


def _timing(config: NativeE2M1MLAConfig, query_len: int, windows: int, replays: int):
    torch.manual_seed(20260816 + query_len)
    device = torch.device("cuda")
    rotated = torch.randn(query_len, HEADS, LATENT, dtype=torch.bfloat16, device=device)
    true_weight = torch.randn(
        HEADS, LATENT, VALUE, dtype=torch.bfloat16, device=device
    ) / math.sqrt(LATENT)
    weight_fp8, weight_scale = _quantize_weight(true_weight)
    effective_weight = weight_fp8.float() * weight_scale
    folded_fp32 = config.fuse_inverse_rotation_into_mla_v_weight(
        effective_weight
    ).float()
    folded_bf16_weight = folded_fp32.to(torch.bfloat16)
    folded_fp8_weight, folded_fp8_weight_scale = _quantize_weight(folded_fp32)

    inverse_buffer = torch.empty_like(rotated)
    control_scale = torch.zeros(1, dtype=torch.float32, device=device)
    folded_scale = torch.zeros(1, dtype=torch.float32, device=device)
    control_out = torch.empty(
        HEADS, query_len, VALUE, dtype=torch.bfloat16, device=device
    )
    folded_bf16_out = torch.empty_like(control_out)
    folded_fp8_out = torch.empty_like(control_out)

    def control():
        hadamard_transform_with_signs(
            rotated,
            config.signs2,
            config.signs1,
            scale=1.0 / math.sqrt(LATENT),
            out=inverse_buffer,
        )
        value, scale = per_tensor_quant_mla_fp8(
            inverse_buffer.transpose(0, 1), control_scale
        )
        bmm_fp8(
            value,
            weight_fp8,
            scale,
            weight_scale,
            torch.bfloat16,
            out=control_out,
        )

    def candidate_bf16():
        torch.bmm(
            rotated.transpose(0, 1),
            folded_bf16_weight,
            out=folded_bf16_out,
        )

    def candidate_fp8():
        value, scale = per_tensor_quant_mla_fp8(rotated.transpose(0, 1), folded_scale)
        bmm_fp8(
            value,
            folded_fp8_weight,
            scale,
            folded_fp8_weight_scale,
            torch.bfloat16,
            out=folded_fp8_out,
        )

    control_graph = _capture(control)
    bf16_graph = _capture(candidate_bf16)
    fp8_graph = _capture(candidate_fp8)
    for graph in (control_graph, bf16_graph, fp8_graph):
        _measure(graph, replays)
    for name, graph in (("folded_bf16", bf16_graph), ("folded_fp8", fp8_graph)):
        result = _paired(graph, control_graph, windows=windows, replays=replays)
        print(
            "TQ_ABSORBED_FOLD_TIMING "
            f"variant={name} q_len={query_len} "
            f"control_us={result['control_us']:.6f} "
            f"candidate_us={result['candidate_us']:.6f} "
            f"candidate_over_control={result['ratio']:.6f} "
            f"ci95=[{result['lower']:.6f},{result['upper']:.6f}] "
            f"windows={windows} replays={replays}",
            flush=True,
        )


def main():
    windows = int(os.environ.get("TQ_FOLD_WINDOWS", "30"))
    replays = int(os.environ.get("TQ_FOLD_REPLAYS", "500"))
    rows = int(os.environ.get("TQ_FOLD_NUMERIC_ROWS", "2048"))
    if windows not in (10, 30):
        raise ValueError("TQ_FOLD_WINDOWS must be 10 or 30")
    config = NativeE2M1MLAConfig(device="cuda")
    _numerics(config, rows)
    model_dir = os.environ.get("TQ_FOLD_MODEL_DIR")
    if model_dir:
        layer_ids = [
            int(value)
            for value in os.environ.get("TQ_FOLD_LAYER_IDS", "0,15,30,45,60").split(",")
        ]
        head_start = int(os.environ.get("TQ_FOLD_HEAD_START", "0"))
        _real_weight_numerics(config, Path(model_dir), layer_ids, rows, head_start)
    for query_len in (1, 5):
        _timing(config, query_len, windows, replays)


if __name__ == "__main__":
    main()
