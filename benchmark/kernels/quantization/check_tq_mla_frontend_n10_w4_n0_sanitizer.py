"""Native-only N10 W4-N0 driver for Compute Sanitizer."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch


EXPECTED_MODULE = "sglang_tq_mla_frontend_sm100f_a17_n10_w4_n0f_v1"
EXPECTED_WRAPPER_SHA256 = (
    "de571139c008788036c7f41f93328af2eed9987a45aa942fbee06b69e209a00d"
)
EXPECTED_SOURCE_SHA256 = (
    "4f3477017beb88e33f42a4bae3c7def939f8d98dbfcaf824f93ca75548186fd2"
)
EXPECTED_SHARED_OBJECT_SHA256 = (
    "cf1af54a49830877149e63f848bfc06f7dfd8fd1a97d11776ddc4efdc78a1ce9"
)


def load(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


kv_tq = load(
    "sglang.srt.layers.quantization.kv_turboquant",
    "/tmp/s5w3/kv_turboquant.py",
)
native = load(
    "n10_w4_n0_native_sanitizer",
    "/tmp/n10w4n0full/sglang/kernels/jit/tq_mla_frontend_n10_native.py",
)


def allocate_outputs(
    tokens: int, pool_size: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    def initialized(
        shape: tuple[int, ...], dtype: torch.dtype, value: float | int
    ) -> torch.Tensor:
        return torch.full(shape, value, dtype=dtype).to(device)

    return (
        initialized((tokens, 8, 512), torch.float8_e4m3fn, 0),
        initialized((tokens, 8, 64), torch.bfloat16, 0),
        initialized((pool_size, 1, 256), torch.uint8, 0),
        initialized((pool_size, 1), torch.bfloat16, 1),
        initialized((pool_size, 1, 64), torch.bfloat16, 0),
        initialized((1,), torch.int32, 0),
        initialized((1,), torch.int64, 0),
    )


def require_clean(outputs: tuple[torch.Tensor, ...], label: str) -> None:
    torch.cuda.synchronize()
    if int(outputs[5].item()) != 0:
        raise AssertionError(f"{label} set fault status {outputs[5].item()}")
    if int(outputs[6].item()) != 0:
        raise AssertionError(f"{label} set zero count {outputs[6].item()}")


def main() -> None:
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
        raise RuntimeError("sanitizer driver requires uncached CUDA allocations")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("N10 W4-N0 sanitizer driver requires SM100")
    if native._MODULE_NAME != EXPECTED_MODULE:
        raise AssertionError(f"unexpected module {native._MODULE_NAME}")

    wrapper = Path(native.__file__).resolve()
    source = (
        wrapper.parent
        / "csrc"
        / "tq_mla_frontend"
        / "tq_mla_frontend_n10_native_sm100f.cu"
    )
    extension = native._get_module()
    identities = {
        "wrapper": (sha256_file(wrapper), EXPECTED_WRAPPER_SHA256),
        "source": (sha256_file(source), EXPECTED_SOURCE_SHA256),
        "shared_object": (
            sha256_file(extension.__file__),
            EXPECTED_SHARED_OBJECT_SHA256,
        ),
    }
    for label, (observed, expected) in identities.items():
        if observed != expected:
            raise AssertionError(
                f"{label} hash {observed} does not match pinned {expected}"
            )

    tokens = 5
    pool_size = 48
    generator = torch.Generator(device="cpu").manual_seed(20260816)

    def random_input(
        shape: tuple[int, ...], dtype: torch.dtype, low: float, high: float
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype).uniform_(
            low, high, generator=generator
        ).to(device)

    query_latent = random_input((tokens, 8, 512), torch.bfloat16, -0.25, 0.25)
    query_rope = random_input((tokens, 8, 64), torch.bfloat16, -0.25, 0.25)
    cache_latent = random_input(
        (tokens * 2, 1, 512), torch.bfloat16, -0.25, 0.25
    )[::2]
    cache_rope = random_input(
        (tokens * 2, 1, 64), torch.bfloat16, 0.125, 0.5
    )[::2]
    cos_sin_cache = random_input((128, 64), torch.float32, -1.0, 1.0)
    positions = torch.tensor([3, 17, 31, 63, 127], dtype=torch.int64).to(device)
    config_cpu = kv_tq.NativeE2M1MLAConfig(device="cpu")
    signs1 = config_cpu.signs1.to(device)
    signs2 = config_cpu.signs2.to(device)
    locations_host = torch.tensor([31, 1, 32, 0, 47], dtype=torch.int64)

    for location_dtype in (torch.int32, torch.int64):
        locations = locations_host.to(dtype=location_dtype).to(device)
        cache_outputs = allocate_outputs(tokens, pool_size, device)
        native.tq_mla_n10_native_cache_writer_out(
            cache_latent,
            cache_rope,
            locations,
            signs1,
            signs2,
            *cache_outputs[2:],
            grid=config_cpu.grid,
            strict=False,
        )
        require_clean(cache_outputs, f"cache-{location_dtype}")

        for rotation_fused in (False, True):
            post_outputs = allocate_outputs(tokens, pool_size, device)
            native.tq_mla_n10_native_frontend_out(
                query_latent,
                query_rope,
                cache_latent,
                cache_rope,
                locations,
                signs1,
                signs2,
                *post_outputs,
                grid=config_cpu.grid,
                rotation_fused=rotation_fused,
                strict=False,
            )
            require_clean(post_outputs, f"post-{location_dtype}-{rotation_fused}")

            rope_outputs = allocate_outputs(tokens, pool_size, device)

            def run_rope() -> None:
                native.tq_mla_n10_native_frontend_rope_out(
                    query_latent,
                    query_rope,
                    cache_latent,
                    cache_rope,
                    cos_sin_cache,
                    positions,
                    locations,
                    signs1,
                    signs2,
                    *rope_outputs,
                    grid=config_cpu.grid,
                    rotation_fused=rotation_fused,
                    strict=False,
                )

            run_rope()
            require_clean(rope_outputs, f"rope-{location_dtype}-{rotation_fused}")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_rope()
            for _ in range(10):
                graph.replay()
            require_clean(rope_outputs, f"graph-{location_dtype}-{rotation_fused}")

    print("PASS native-only N10 W4-N0 sanitizer driver")


if __name__ == "__main__":
    main()
