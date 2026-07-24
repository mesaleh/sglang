import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="native TQ4 decode requires SM100",
    ),
]


@pytest.mark.parametrize(
    ("context", "query_length"),
    ((128, 1), (129, 5)),
)
@pytest.mark.parametrize("e2m1", (False, True), ids=("lloyd", "e2m1"))
def test_tokenspeed_tq4_backend_reads_token_major_packed_pool(
    context, query_length, e2m1
):
    import tokenspeed_mla
    from tokenspeed_mla.tq4_contract import dequantize_tq4_reference

    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        TokenspeedMLABackend,
        _quantize_tq4_query,
    )
    from sglang.srt.layers.quantization.kv_turboquant import (
        TurboQuantConfig,
        batched_quantize,
    )

    if not hasattr(tokenspeed_mla, "tokenspeed_mla_decode_tq4"):
        pytest.skip("installed tokenspeed_mla has no native TQ4 entry point")

    torch.manual_seed(20260722)
    device = torch.device("cuda")
    page_size = 32
    pages = math.ceil(context / 128) * 4
    tokens = pages * page_size
    # Official Kimi K2.6 has 64 query heads, so dual-node TP8 serves 8
    # TP-local heads. TokenSpeed's 16-head shape is the single-node TP4 case.
    heads = 8
    latent = 512
    rope_dim = 64

    config = TurboQuantConfig(4, latent, str(device), e2m1=e2m1)
    config.mla_absorb_rotation_fused = True
    latent_original = torch.randn(
        tokens, 1, latent, device=device, dtype=torch.bfloat16
    )
    packed, norms, quant_norms = batched_quantize(
        latent_original,
        config.signs1,
        config.signs2,
        config.k_quant_centroids,
        config.k_boundaries,
        4,
        storage_code_lut=config.k_storage_code_lut,
    )
    safe_quant_norms = torch.where(
        quant_norms > 1e-10, quant_norms, torch.ones_like(quant_norms)
    )
    dequant_scale = ((norms / safe_quant_norms) * config.k_dequant_scale_multiplier).to(
        torch.bfloat16
    )
    codebook = (
        config.k_centroids.float().view(1, 1, 16) * dequant_scale.float().unsqueeze(-1)
    ).to(torch.float8_e4m3fn)
    rope = torch.randn(tokens, 1, rope_dim, device=device, dtype=torch.bfloat16)

    pool = SimpleNamespace(
        start_layer=0,
        tq_config=config,
        kv_nope_packed_buffer=[packed],
        kv_nope_scale_buffer=[dequant_scale],
        kv_nope_codebook_buffer=(
            None if e2m1 else [codebook.view(torch.uint8).contiguous()]
        ),
        kv_rope_buffer=[rope],
    )
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend._tq_pool = pool
    backend._tq_config = config
    backend.page_size = page_size
    backend.max_context_len = 32768
    backend.kv_lora_rank = latent
    backend.qk_rope_head_dim = rope_dim
    backend.num_q_heads = heads
    backend._tokenspeed_workspace = torch.empty(
        64 << 20, dtype=torch.int8, device=device
    )

    layer = SimpleNamespace(layer_id=0, scaling=1.0 / math.sqrt(latent + rope_dim))
    packed_pages = backend._get_decode_kv_cache(layer)
    assert packed_pages.shape == (pages, 1, page_size, latent // 2)

    query_original = torch.randn(
        1, query_length, heads, latent, device=device, dtype=torch.bfloat16
    )
    query_rope = torch.randn(
        1, query_length, heads, rope_dim, device=device, dtype=torch.bfloat16
    )
    query_rotated = torch.cat((config.rotate_query(query_original), query_rope), dim=-1)
    page_table = torch.randperm(pages, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([context], device=device, dtype=torch.int32)

    actual = backend._run_decode_kernel(
        query=query_rotated,
        kv_cache=packed_pages,
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=context,
        layer=layer,
    )

    latent_rotated = dequantize_tq4_reference(
        packed,
        dequant_scale,
        config.k_centroids,
        dtype=torch.float32,
    )
    dense_cache = torch.cat((latent_rotated, rope), dim=-1).to(torch.float8_e4m3fn)
    expected = tokenspeed_mla.tokenspeed_mla_decode(
        query=_quantize_tq4_query(query_rotated, latent, enable_pdl=True),
        kv_cache=dense_cache.view(pages, page_size, latent + rope_dim),
        workspace_buffer=backend._tokenspeed_workspace,
        kv_lora_rank=latent,
        qk_rope_head_dim=rope_dim,
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=context,
        softmax_scale=layer.scaling,
        enable_pdl=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.002)


@pytest.mark.parametrize("query_length", (1, 5))
def test_tq4_query_quantization_fuses_latent_and_rope_casts(query_length):
    from sglang.jit_kernel.fp8_quantize import fp8_quantize
    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        _quantize_tq4_query,
    )

    torch.manual_seed(20260723)
    query = torch.randn(1, query_length, 8, 576, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    fp8_quantize(query[..., :512], out=expected[..., :512], enable_pdl=True)
    fp8_quantize(query[..., 512:], out=expected[..., 512:], enable_pdl=True)

    actual = _quantize_tq4_query(query, 512, enable_pdl=True)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_tq4_query_quantization_matches_every_bf16_encoding():
    from sglang.jit_kernel.fp8_quantize import fp8_quantize
    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        _quantize_tq4_query,
    )

    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
    padding = (-bits.numel()) % 576
    bits = torch.nn.functional.pad(bits, (0, padding)).view(-1, 576)
    query = bits.view(torch.bfloat16)
    expected = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    fp8_quantize(query[..., :512], out=expected[..., :512], enable_pdl=True)
    fp8_quantize(query[..., 512:], out=expected[..., 512:], enable_pdl=True)

    actual = _quantize_tq4_query(query, 512, enable_pdl=True)

    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
