import argparse

import torch
import triton

from sglang.jit_kernel.fp8_quantize import fp8_quantize
from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    _quantize_tq4_query,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-length", type=int, choices=(1, 5), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    query = torch.randn(
        1,
        args.query_length,
        8,
        576,
        device=args.device,
        dtype=torch.bfloat16,
    )

    def incumbent():
        output = torch.empty_like(query, dtype=torch.float8_e4m3fn)
        fp8_quantize(query[..., :512], out=output[..., :512], enable_pdl=True)
        fp8_quantize(query[..., 512:], out=output[..., 512:], enable_pdl=True)
        return output

    def fused():
        return _quantize_tq4_query(query, 512, enable_pdl=True)

    incumbent_output = incumbent()
    fused_output = fused()
    torch.testing.assert_close(
        fused_output.float(), incumbent_output.float(), rtol=0, atol=0
    )

    incumbent_ms = triton.testing.do_bench(incumbent, warmup=100, rep=1000)
    fused_ms = triton.testing.do_bench(fused, warmup=100, rep=1000)
    print(
        f"q_len={args.query_length} incumbent_us={incumbent_ms * 1000:.3f} "
        f"fused_us={fused_ms * 1000:.3f} speedup={incumbent_ms / fused_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
