import math
from types import SimpleNamespace

import pytest
import torch


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 10,
        reason="native TQ4 decode requires SM100",
    ),
]


@pytest.mark.parametrize(
    ("context", "query_length"),
    ((128, 1), (129, 5), (10_000, 5)),
)
def test_tokenspeed_tq4_backend_matches_dense_rotated_oracle(
    context, query_length
):
    import tokenspeed_mla

    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        TokenspeedMLABackend,
        _quantize_tq4_query,
    )
    from sglang.srt.layers.quantization.kv_turboquant import (
        TurboQuantConfig,
        batched_quantize,
    )
    from tokenspeed_mla.tq4_contract import dequantize_tq4_reference

    if not hasattr(tokenspeed_mla, "tokenspeed_mla_decode_tq4"):
        pytest.skip("installed tokenspeed_mla has no native TQ4 entry point")

    torch.manual_seed(20260803 + context + query_length)
    device = torch.device("cuda")
    page_size = 32
    # The packed kernel stages 128-token tiles (four page-32 pages).
    pages = math.ceil(context / 128) * 4
    tokens = pages * page_size
    # Official Kimi K2.6 has 64 query heads, so dual-node TP8 serves eight
    # TP-local heads.
    heads = 8
    latent = 512
    rope_dim = 64

    config = TurboQuantConfig(4, latent, str(device))
    latent_original = torch.randn(
        tokens, 1, latent, device=device, dtype=torch.bfloat16
    )
    packed, norms, quant_norms = batched_quantize(
        latent_original,
        config.signs1,
        config.signs2,
        config.k_centroids,
        config.k_boundaries,
        4,
    )
    safe_quant_norms = torch.where(
        quant_norms > 1e-10, quant_norms, torch.ones_like(quant_norms)
    )
    dequant_scale = (norms / safe_quant_norms).to(torch.bfloat16)
    rope = torch.randn(
        tokens, 1, rope_dim, device=device, dtype=torch.bfloat16
    )

    pool = SimpleNamespace(
        start_layer=0,
        tq_config=config,
        kv_nope_packed_buffer=[packed],
        kv_nope_scale_buffer=[dequant_scale],
        kv_rope_buffer=[rope],
    )
    backend = object.__new__(TokenspeedMLABackend)
    backend._tq4_cache = True
    backend._tq_pool = pool
    backend._tq_config = config
    backend.page_size = page_size
    backend.kv_lora_rank = latent
    backend.qk_rope_head_dim = rope_dim
    backend.num_q_heads = heads
    backend.max_context_len = 256_000
    backend._tokenspeed_workspace = torch.empty(
        64 << 20, dtype=torch.int8, device=device
    )

    layer = SimpleNamespace(layer_id=0, scaling=1.0 / math.sqrt(latent + rope_dim))
    packed_pages = backend._get_decode_kv_cache(layer)
    assert packed_pages.shape == (pages, 1, page_size, latent // 2)
    assert packed_pages.data_ptr() == packed.data_ptr()

    query_original = torch.randn(
        1, query_length, heads, latent, device=device, dtype=torch.bfloat16
    )
    query_rope = torch.randn(
        1, query_length, heads, rope_dim, device=device, dtype=torch.bfloat16
    )
    query = torch.cat((query_original, query_rope), dim=-1)
    query_rotated = torch.cat(
        (config.rotate_query(query_original), query_rope), dim=-1
    )
    page_table = torch.randperm(pages, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([context], device=device, dtype=torch.int32)

    actual = backend._run_decode_kernel(
        query=query,
        kv_cache=packed_pages,
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=context,
        layer=layer,
    )

    latent_rotated = dequantize_tq4_reference(
        packed.view(pages, page_size, latent // 2),
        dequant_scale.view(pages, page_size),
        config.k_centroids,
        dtype=torch.float32,
    )
    dense_cache = torch.cat(
        (latent_rotated, rope.view(pages, page_size, rope_dim)), dim=-1
    ).to(torch.float8_e4m3fn)
    expected_rotated = tokenspeed_mla.tokenspeed_mla_decode(
        query=_quantize_tq4_query(query_rotated, latent, enable_pdl=True),
        kv_cache=dense_cache,
        workspace_buffer=backend._tokenspeed_workspace,
        kv_lora_rank=latent,
        qk_rope_head_dim=rope_dim,
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=context,
        softmax_scale=layer.scaling,
        enable_pdl=True,
    )
    expected = config.inverse_rotate_output(expected_rotated)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0.004)
