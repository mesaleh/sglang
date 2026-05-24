/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int PAGE_SIZE = 64;
constexpr int LORA_RANK = 512;
constexpr int PACKED_DIM = 256;
constexpr int ROPE_DIM = 64;
constexpr int FULL_DIM = LORA_RANK + ROPE_DIM;

template <typename pool_index_t>
__global__ void stage_tq_mla_pages_to_physical_kernel(
    const int32_t* __restrict__ req_to_token,
    const pool_index_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ seq_lens,
    const uint8_t* __restrict__ k_nope_packed,
    const __nv_bfloat16* __restrict__ k_scale,
    const __nv_bfloat16* __restrict__ k_rope,
    const float* __restrict__ k_centroids,
    __nv_bfloat16* __restrict__ out_k_cache,
    int req_stride,
    int pages_per_req) {
  __shared__ int kv_loc_s[PAGE_SIZE];
  __shared__ float scale_s[PAGE_SIZE];
  __shared__ float codebook_s[16];

  const int tid = threadIdx.x;
  const int active_page = blockIdx.x;
  const int batch_id = active_page / pages_per_req;
  const int page_in_req = active_page - batch_id * pages_per_req;
  const int req_pool_index = static_cast<int>(req_pool_indices[batch_id]);
  const int seq_len = seq_lens[batch_id];
  const int base_seq = page_in_req * PAGE_SIZE;

  if (tid < 16) {
    codebook_s[tid] = k_centroids[tid];
  }

  if (tid < PAGE_SIZE) {
    const int seq_offset = base_seq + tid;
    if (seq_offset < seq_len) {
      const int kv_loc = req_to_token[req_pool_index * req_stride + seq_offset];
      kv_loc_s[tid] = kv_loc;
      scale_s[tid] = __bfloat162float(k_scale[kv_loc]);
    } else {
      kv_loc_s[tid] = 0;
      scale_s[tid] = 0.0f;
    }
  }

  __syncthreads();

  if (base_seq >= seq_len) {
    return;
  }

  for (int idx = tid; idx < PAGE_SIZE * PACKED_DIM; idx += blockDim.x) {
    const int token = idx >> 8;
    const int packed_dim = idx & (PACKED_DIM - 1);
    if (base_seq + token >= seq_len) {
      continue;
    }

    const int kv_loc = kv_loc_s[token];
    const uint8_t packed = k_nope_packed[kv_loc * PACKED_DIM + packed_dim];
    const int lo = packed & 0x0F;
    const int hi = (packed >> 4) & 0x0F;
    const float scale = scale_s[token];
    const int out_idx = kv_loc * FULL_DIM + (packed_dim << 1);

    out_k_cache[out_idx] = __float2bfloat16(codebook_s[lo] * scale);
    out_k_cache[out_idx + 1] = __float2bfloat16(codebook_s[hi] * scale);
  }

  for (int idx = tid; idx < PAGE_SIZE * ROPE_DIM; idx += blockDim.x) {
    const int token = idx >> 6;
    const int rope_dim = idx & (ROPE_DIM - 1);
    if (base_seq + token >= seq_len) {
      continue;
    }

    const int kv_loc = kv_loc_s[token];
    out_k_cache[kv_loc * FULL_DIM + LORA_RANK + rope_dim] = k_rope[kv_loc * ROPE_DIM + rope_dim];
  }
}

void check_stage_tq_mla_inputs(
    const torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& seq_lens,
    const torch::Tensor& k_nope_packed,
    const torch::Tensor& k_scale,
    const torch::Tensor& k_rope,
    const torch::Tensor& k_centroids,
    const torch::Tensor& out_k_cache) {
  TORCH_CHECK(req_to_token.is_cuda(), "req_to_token must be CUDA");
  TORCH_CHECK(req_pool_indices.is_cuda(), "req_pool_indices must be CUDA");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA");
  TORCH_CHECK(k_nope_packed.is_cuda(), "k_nope_packed must be CUDA");
  TORCH_CHECK(k_scale.is_cuda(), "k_scale must be CUDA");
  TORCH_CHECK(k_rope.is_cuda(), "k_rope must be CUDA");
  TORCH_CHECK(k_centroids.is_cuda(), "k_centroids must be CUDA");
  TORCH_CHECK(out_k_cache.is_cuda(), "out_k_cache must be CUDA");
  TORCH_CHECK(req_to_token.scalar_type() == torch::kInt32, "req_to_token must be int32");
  TORCH_CHECK(
      req_pool_indices.scalar_type() == torch::kInt32 || req_pool_indices.scalar_type() == torch::kInt64,
      "req_pool_indices must be int32 or int64");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(k_nope_packed.scalar_type() == torch::kUInt8, "k_nope_packed must be uint8");
  TORCH_CHECK(k_scale.scalar_type() == torch::kBFloat16, "k_scale must be bfloat16");
  TORCH_CHECK(k_rope.scalar_type() == torch::kBFloat16, "k_rope must be bfloat16");
  TORCH_CHECK(k_centroids.scalar_type() == torch::kFloat32, "k_centroids must be float32");
  TORCH_CHECK(out_k_cache.scalar_type() == torch::kBFloat16, "out_k_cache must be bfloat16");
  TORCH_CHECK(req_to_token.is_contiguous(), "req_to_token must be contiguous");
  TORCH_CHECK(req_pool_indices.is_contiguous(), "req_pool_indices must be contiguous");
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
  TORCH_CHECK(k_nope_packed.is_contiguous(), "k_nope_packed must be contiguous");
  TORCH_CHECK(k_scale.is_contiguous(), "k_scale must be contiguous");
  TORCH_CHECK(k_rope.is_contiguous(), "k_rope must be contiguous");
  TORCH_CHECK(k_centroids.is_contiguous(), "k_centroids must be contiguous");
  TORCH_CHECK(out_k_cache.is_contiguous(), "out_k_cache must be contiguous");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be rank 2");
  TORCH_CHECK(req_pool_indices.dim() == 1, "req_pool_indices must be rank 1");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be rank 1");
  TORCH_CHECK(
      k_nope_packed.dim() == 2 || (k_nope_packed.dim() == 3 && k_nope_packed.size(1) == 1),
      "k_nope_packed must be rank 2 or rank 3 with singleton head dimension");
  TORCH_CHECK(
      k_scale.dim() == 1 || (k_scale.dim() == 2 && k_scale.size(1) == 1),
      "k_scale must be rank 1 or rank 2 with singleton head dimension");
  TORCH_CHECK(
      k_rope.dim() == 2 || (k_rope.dim() == 3 && k_rope.size(1) == 1),
      "k_rope must be rank 2 or rank 3 with singleton head dimension");
  TORCH_CHECK(k_centroids.dim() == 1, "k_centroids must be rank 1");
  TORCH_CHECK(
      out_k_cache.dim() == 2 || (out_k_cache.dim() == 3 && out_k_cache.size(1) == 1),
      "out_k_cache must be rank 2 or rank 3 with singleton head dimension");
  TORCH_CHECK(seq_lens.numel() == req_pool_indices.numel(), "seq_lens and req_pool_indices length mismatch");
  TORCH_CHECK(k_nope_packed.size(-1) == PACKED_DIM, "expected packed dim 256");
  const int64_t num_pool_tokens = k_nope_packed.size(0);
  TORCH_CHECK(k_scale.size(0) == num_pool_tokens, "k_scale token dimension mismatch");
  TORCH_CHECK(k_rope.size(-1) == ROPE_DIM, "expected rope dim 64");
  TORCH_CHECK(k_rope.size(0) == num_pool_tokens, "k_rope token dimension mismatch");
  TORCH_CHECK(k_centroids.numel() == 16, "expected 16 TurboQuant centroids");
  TORCH_CHECK(out_k_cache.size(0) >= num_pool_tokens, "output token dimension must cover pool tokens");
  TORCH_CHECK(out_k_cache.size(-1) == FULL_DIM, "expected output dim 576");
}

template <typename pool_index_t>
void launch_stage_tq_mla_pages_to_physical(
    const torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& seq_lens,
    const torch::Tensor& k_nope_packed,
    const torch::Tensor& k_scale,
    const torch::Tensor& k_rope,
    const torch::Tensor& k_centroids,
    torch::Tensor& out_k_cache,
    int64_t req_stride,
    int64_t pages_per_req,
    int64_t threads) {
  check_stage_tq_mla_inputs(
      req_to_token,
      req_pool_indices,
      seq_lens,
      k_nope_packed,
      k_scale,
      k_rope,
      k_centroids,
      out_k_cache);
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "threads must be 128, 256, or 512");
  TORCH_CHECK(req_stride > 0 && req_stride <= std::numeric_limits<int>::max(), "req_stride out of range");
  TORCH_CHECK(pages_per_req > 0 && pages_per_req <= std::numeric_limits<int>::max(), "pages_per_req out of range");

  const int bs = static_cast<int>(req_pool_indices.numel());
  const int active_pages = bs * static_cast<int>(pages_per_req);
  if (active_pages == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(req_to_token));
  stage_tq_mla_pages_to_physical_kernel<pool_index_t>
      <<<active_pages, static_cast<int>(threads), 0, at::cuda::getCurrentCUDAStream()>>>(
          req_to_token.data_ptr<int32_t>(),
          req_pool_indices.data_ptr<pool_index_t>(),
          seq_lens.data_ptr<int32_t>(),
          k_nope_packed.data_ptr<uint8_t>(),
          reinterpret_cast<const __nv_bfloat16*>(k_scale.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr<at::BFloat16>()),
          k_centroids.data_ptr<float>(),
          reinterpret_cast<__nv_bfloat16*>(out_k_cache.data_ptr<at::BFloat16>()),
          static_cast<int>(req_stride),
          static_cast<int>(pages_per_req));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void stage_tq_mla_pages_to_physical(
    const torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& seq_lens,
    const torch::Tensor& k_nope_packed,
    const torch::Tensor& k_scale,
    const torch::Tensor& k_rope,
    const torch::Tensor& k_centroids,
    torch::Tensor& out_k_cache,
    int64_t req_stride,
    int64_t pages_per_req,
    int64_t threads) {
  if (req_pool_indices.scalar_type() == torch::kInt32) {
    launch_stage_tq_mla_pages_to_physical<int32_t>(
        req_to_token,
        req_pool_indices,
        seq_lens,
        k_nope_packed,
        k_scale,
        k_rope,
        k_centroids,
        out_k_cache,
        req_stride,
        pages_per_req,
        threads);
  } else if (req_pool_indices.scalar_type() == torch::kInt64) {
    launch_stage_tq_mla_pages_to_physical<int64_t>(
        req_to_token,
        req_pool_indices,
        seq_lens,
        k_nope_packed,
        k_scale,
        k_rope,
        k_centroids,
        out_k_cache,
        req_stride,
        pages_per_req,
        threads);
  } else {
    TORCH_CHECK(false, "req_pool_indices must be int32 or int64");
  }
}
