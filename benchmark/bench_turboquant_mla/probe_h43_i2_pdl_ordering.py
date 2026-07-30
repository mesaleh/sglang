"""Positive writer-to-reader PDL ordering sensitivity gate for H43 I2."""

from __future__ import annotations

import argparse
import json
import math

import torch

from h43_aot_loader import install_h43_aot_from_environment
from sglang.jit_kernel.tq_mla_frontend import tq_mla_frontend_out
from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
from tokenspeed_mla import tokenspeed_mla_decode_tq4

_H43_AOT_LOADER = install_h43_aot_from_environment()

LATENT = 512
ROPE = 64
HEADS = 8
PAGE = 32
CACHE_ROWS = 256_000
CACHE_PAGES = CACHE_ROWS // PAGE
FP8 = torch.float8_e4m3fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(10219, 37932), required=True)
    parser.add_argument("--q-len", type=int, choices=(1, 5), required=True)
    parser.add_argument("--split-kv", type=int, choices=(40, 64), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--reader", choices=("pre-move", "post-wait"), required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.split_kv != {10219: 64, 37932: 40}[args.context]:
        raise ValueError("context and split-kv do not match the frozen contract")
    if args.steps != 1000:
        raise ValueError("the H43 I2 sensitivity gate requires exactly 1,000 steps")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("H43 I2 PDL ordering requires SM100")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    page_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    config = TurboQuantConfig(
        bit_width=4,
        head_dim=LATENT,
        device=device,
        k_bit_width=4,
        v_bit_width=4,
        uniform=False,
        e2m1=True,
    )
    assert config.k_storage_code_lut is not None

    packed = torch.empty(
        CACHE_PAGES, PAGE, LATENT // 2, dtype=torch.uint8, device=device
    ).random_(0, 256, generator=generator)
    scale = torch.empty(
        CACHE_PAGES, PAGE, dtype=torch.bfloat16, device=device
    ).uniform_(0.05, 0.20, generator=generator)
    rope_scratch = torch.empty(
        CACHE_PAGES, PAGE, ROPE, dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.1, generator=generator)
    rope = rope_scratch.to(FP8)
    del rope_scratch
    codebook = (
        (scale.float()[..., None] * config.k_centroids)
        .to(FP8)
        .view(torch.uint8)
        .contiguous()
    )
    if codebook.data_ptr() % 16:
        raise AssertionError("codebook must be 16-byte aligned")

    page_table = torch.randperm(
        CACHE_PAGES, dtype=torch.int32, device=device, generator=page_generator
    ).view(1, CACHE_PAGES)
    seq_lens = torch.tensor([args.context], dtype=torch.int32, device=device)
    positions = torch.arange(
        args.context - args.q_len,
        args.context,
        dtype=torch.int64,
        device=device,
    )
    logical_pages = torch.div(positions, PAGE, rounding_mode="floor")
    locations = (
        page_table[0, logical_pages].to(torch.int64) * PAGE
        + torch.remainder(positions, PAGE)
    ).contiguous()

    query_latent = torch.empty(
        args.q_len, HEADS, LATENT, dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.125, generator=generator)
    query_rope = torch.empty(
        args.q_len, HEADS, ROPE, dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.125, generator=generator)
    cache_rope = torch.empty(
        args.q_len, 1, ROPE, dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.125, generator=generator)

    def make_writer_inputs(scale_factor: float):
        cache_latent = torch.empty(
            args.q_len, 1, LATENT, dtype=torch.bfloat16, device=device
        ).normal_(0.0, 0.125 * scale_factor, generator=generator)
        return query_latent, query_rope, cache_latent, cache_rope

    writer_inputs = (make_writer_inputs(0.125), make_writer_inputs(8.0))
    query = torch.empty(args.q_len, HEADS, LATENT + ROPE, dtype=FP8, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    pdl_workspace = torch.empty(64 << 20, dtype=torch.int8, device=device)
    ordered_workspace = torch.empty_like(pdl_workspace)
    pdl_output = torch.empty(
        1, args.q_len, HEADS, LATENT, dtype=torch.bfloat16, device=device
    )
    ordered_output = torch.empty_like(pdl_output)
    softmax_scale = 1.0 / math.sqrt(LATENT + ROPE)

    def writer(variant: int) -> None:
        query_latent, query_rope, cache_latent, cache_rope = writer_inputs[variant]
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
            query,
            packed.view(CACHE_ROWS, 1, LATENT // 2),
            scale.view(CACHE_ROWS, 1),
            rope.view(CACHE_ROWS, 1, ROPE),
            status,
            decode_centroids=config.k_centroids,
            codebook_cache=codebook.view(CACHE_ROWS, 1, 16),
            scale_multiplier=config.k_dequant_scale_multiplier,
            rotation_fused=True,
            num_warps=8,
            strict=False,
        )

    def reader(output: torch.Tensor, workspace: torch.Tensor, enable_pdl: bool) -> None:
        tokenspeed_mla_decode_tq4(
            query=query.view(1, args.q_len, HEADS, LATENT + ROPE),
            kv_nope_packed=packed,
            kv_nope_scale=scale,
            kv_rope=rope,
            centroids=config.k_centroids,
            workspace_buffer=workspace,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=CACHE_ROWS,
            softmax_scale=softmax_scale,
            out=output,
            causal_mask=True,
            enable_pdl=enable_pdl,
            split_kv_override=args.split_kv,
            kv_nope_codebook=codebook,
            fp8_rope=True,
        )

    # Compile both dispatch keys before the unsynchronized sensitivity loop.
    writer(0)
    reader(pdl_output, pdl_workspace, True)
    reader(ordered_output, ordered_workspace, False)
    torch.cuda.synchronize()

    pdl_history = torch.empty(
        args.steps, *pdl_output.shape, dtype=pdl_output.dtype, device=device
    )
    ordered_history = torch.empty_like(pdl_history)
    for step in range(args.steps):
        writer(step & 1)
        reader(pdl_output, pdl_workspace, True)
        pdl_history[step].copy_(pdl_output)
        reader(ordered_output, ordered_workspace, False)
        ordered_history[step].copy_(ordered_output)
    torch.cuda.synchronize()
    mismatch_mask = pdl_history.view(args.steps, -1) != ordered_history.view(
        args.steps, -1
    )
    mismatched_steps = int(mismatch_mask.any(dim=1).sum().item())
    mismatched_values = int(mismatch_mask.sum().item())

    # A second same-stream, PDL-disabled loop is the ordered control.
    control_a = torch.empty_like(ordered_history)
    control_b = torch.empty_like(ordered_history)
    for step in range(args.steps):
        writer(step & 1)
        reader(ordered_output, ordered_workspace, False)
        control_a[step].copy_(ordered_output)
        reader(ordered_output, ordered_workspace, False)
        control_b[step].copy_(ordered_output)
    torch.cuda.synchronize()
    ordered_control_mismatches = int((control_a != control_b).sum().item())

    if ordered_control_mismatches != 0:
        raise AssertionError("PDL-disabled same-stream control was not deterministic")
    if args.reader == "pre-move" and mismatched_steps == 0:
        raise AssertionError("pre-move reader did not establish positive sensitivity")
    if args.reader == "post-wait" and mismatched_steps != 0:
        raise AssertionError("post-wait reader observed stale writer state")
    if int(status.item()) != 0:
        raise AssertionError("writer sticky status changed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment": "H43_I2_PDL_ORDERING_SENSITIVITY",
                "reader": args.reader,
                "context": args.context,
                "q_len": args.q_len,
                "split_kv": args.split_kv,
                "steps": args.steps,
                "mismatched_steps": mismatched_steps,
                "mismatched_values": mismatched_values,
                "ordered_control_mismatches": ordered_control_mismatches,
                "sticky_status": int(status.item()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
