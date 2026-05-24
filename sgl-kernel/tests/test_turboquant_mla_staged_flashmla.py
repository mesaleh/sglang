import pytest
import torch


PAGE_SIZE = 64
PACKED_DIM = 256
ROPE_DIM = 64
FULL_DIM = 576


def _reference_stage(
    req_to_token,
    req_pool_indices,
    seq_lens,
    k_nope_packed,
    k_scale,
    k_rope,
    k_centroids,
    req_stride,
    pages_per_req,
):
    req_to_token_cpu = req_to_token.cpu()
    req_pool_indices_cpu = req_pool_indices.cpu()
    seq_lens_cpu = seq_lens.cpu()
    k_nope_packed_cpu = k_nope_packed.reshape(k_nope_packed.shape[0], PACKED_DIM).cpu()
    k_scale_cpu = k_scale.reshape(k_scale.shape[0]).float().cpu()
    k_rope_cpu = k_rope.reshape(k_rope.shape[0], ROPE_DIM).cpu()
    k_centroids_cpu = k_centroids.cpu()
    out = torch.zeros((k_nope_packed.shape[0], FULL_DIM), dtype=torch.bfloat16)

    for batch_id in range(req_pool_indices.numel()):
        req_pool_index = int(req_pool_indices_cpu[batch_id])
        seq_len = int(seq_lens_cpu[batch_id])
        for page_in_req in range(pages_per_req):
            base_seq = page_in_req * PAGE_SIZE
            if base_seq >= seq_len:
                continue
            for token in range(min(PAGE_SIZE, seq_len - base_seq)):
                kv_loc = int(req_to_token_cpu[req_pool_index, base_seq + token])
                scale = float(k_scale_cpu[kv_loc])
                for packed_dim in range(PACKED_DIM):
                    packed = int(k_nope_packed_cpu[kv_loc, packed_dim])
                    lo = packed & 0x0F
                    hi = (packed >> 4) & 0x0F
                    out[kv_loc, packed_dim * 2] = k_centroids_cpu[lo] * scale
                    out[kv_loc, packed_dim * 2 + 1] = k_centroids_cpu[hi] * scale
                out[kv_loc, 512:FULL_DIM] = k_rope_cpu[kv_loc]

    return out.to(req_to_token.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("pool_index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("kv_layout", ["flat", "live"])
@pytest.mark.parametrize("out_padding", [0, PAGE_SIZE])
def test_stage_tq_mla_pages_to_physical_matches_reference(
    pool_index_dtype, kv_layout, out_padding
):
    import sgl_kernel

    assert hasattr(sgl_kernel, "stage_tq_mla_pages_to_physical")

    device = "cuda"
    req_stride = 128
    pages_per_req = 2
    num_tokens = 128

    req_to_token = torch.full((2, req_stride), -1, dtype=torch.int32, device=device)
    req_to_token[1, :70] = torch.arange(0, 70, dtype=torch.int32, device=device)
    req_to_token[0, :17] = torch.arange(70, 87, dtype=torch.int32, device=device)
    req_pool_indices = torch.tensor([1, 0], dtype=pool_index_dtype, device=device)
    seq_lens = torch.tensor([70, 17], dtype=torch.int32, device=device)

    packed_base = torch.arange(num_tokens * PACKED_DIM, dtype=torch.int32, device=device)
    k_nope_packed = (packed_base % 256).to(torch.uint8).reshape(num_tokens, PACKED_DIM)
    k_scale = torch.ones(num_tokens, dtype=torch.bfloat16, device=device)
    k_rope = (
        torch.arange(num_tokens * ROPE_DIM, dtype=torch.float32, device=device)
        .remainder(97)
        .reshape(num_tokens, ROPE_DIM)
        .to(torch.bfloat16)
    )
    if kv_layout == "live":
        k_nope_packed = k_nope_packed.view(num_tokens, 1, PACKED_DIM)
        k_scale = k_scale.view(num_tokens, 1)
        k_rope = k_rope.view(num_tokens, 1, ROPE_DIM)
    k_centroids = torch.arange(-8, 8, dtype=torch.float32, device=device)
    out_tokens = num_tokens + out_padding
    if kv_layout == "live":
        out = torch.zeros((out_tokens, 1, FULL_DIM), dtype=torch.bfloat16, device=device)
    else:
        out = torch.zeros((out_tokens, FULL_DIM), dtype=torch.bfloat16, device=device)

    sgl_kernel.stage_tq_mla_pages_to_physical(
        req_to_token,
        req_pool_indices,
        seq_lens,
        k_nope_packed,
        k_scale,
        k_rope,
        k_centroids,
        out,
        req_stride,
        pages_per_req,
        512,
    )
    torch.cuda.synchronize()

    ref = _reference_stage(
        req_to_token,
        req_pool_indices,
        seq_lens,
        k_nope_packed,
        k_scale,
        k_rope,
        k_centroids,
        req_stride,
        pages_per_req,
    )
    torch.testing.assert_close(
        out.reshape(out_tokens, FULL_DIM)[:num_tokens].float(),
        ref.float(),
        atol=0,
        rtol=0,
    )
    if out_padding:
        assert torch.count_nonzero(out.reshape(out_tokens, FULL_DIM)[num_tokens:]).item() == 0
