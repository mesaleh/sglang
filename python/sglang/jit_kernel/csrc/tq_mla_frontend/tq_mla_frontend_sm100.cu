#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

constexpr int kQueryHeads = 8;
constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kQueryDim = kLatentDim + kRopeDim;
constexpr int kPackedDim = kLatentDim / 2;
constexpr float kInvSqrt512 = 0.044194173824159220275f;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_fp8x2_storage_t float2_to_e4m3(
    const float first, const float second) {
  return __nv_cvt_float2_to_fp8x2(make_float2(first, second),
                                  __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ void hadamard_512_bf16_layout(
    float (&values)[16], const int lane) {
#pragma unroll
  for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int low = (item / step) * 2 * step + (item % step);
        const int base = chunk * 8;
        const float lhs = values[base + low];
        const float rhs = values[base + low + step];
        values[base + low] = lhs + rhs;
        values[base + low + step] = lhs - rhs;
      }
    }
  }

#pragma unroll
  for (int lane_mask = 1; lane_mask < 32; lane_mask <<= 1) {
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      const float own = values[item];
      const float other =
          __shfl_xor_sync(0xffffffffu, own, lane_mask, 32);
      values[item] = (lane & lane_mask) == 0 ? own + other : other - own;
    }
  }

#pragma unroll
  for (int item = 0; item < 8; ++item) {
    const float lhs = values[item];
    const float rhs = values[8 + item];
    values[item] = lhs + rhs;
    values[8 + item] = lhs - rhs;
  }
}

__device__ __forceinline__ void hadamard_512_fp32_layout(
    float (&values)[16], const int lane) {
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int base = chunk * 4;
    const float x0 = values[base];
    const float x1 = values[base + 1];
    const float x2 = values[base + 2];
    const float x3 = values[base + 3];
    const float y0 = x0 + x1;
    const float y1 = x0 - x1;
    const float y2 = x2 + x3;
    const float y3 = x2 - x3;
    values[base] = y0 + y2;
    values[base + 1] = y1 + y3;
    values[base + 2] = y0 - y2;
    values[base + 3] = y1 - y3;
  }

#pragma unroll
  for (int lane_mask = 1; lane_mask < 32; lane_mask <<= 1) {
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      const float own = values[item];
      const float other =
          __shfl_xor_sync(0xffffffffu, own, lane_mask, 32);
      values[item] = (lane & lane_mask) == 0 ? own + other : other - own;
    }
  }

#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const float x0 = values[item];
    const float x1 = values[4 + item];
    const float x2 = values[8 + item];
    const float x3 = values[12 + item];
    const float y0 = x0 + x1;
    const float y1 = x0 - x1;
    const float y2 = x2 + x3;
    const float y3 = x2 - x3;
    values[item] = y0 + y2;
    values[4 + item] = y1 + y3;
    values[8 + item] = y0 - y2;
    values[12 + item] = y1 - y3;
  }
}

template <bool kRotationFused>
__device__ __forceinline__ void write_query_head(
    const __nv_bfloat16* query_latent, const __nv_bfloat16* query_rope,
    const float* signs1, const float* signs2, uint8_t* query_out,
    const int token, const int head, const int lane) {
  const int64_t latent_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kLatentDim;
  const int64_t rope_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kRopeDim;
  const int64_t out_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kQueryDim;

  float values[16];
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int index =
        kRotationFused ? lane * 16 + item
                       : (item / 8) * 256 + lane * 8 + (item % 8);
    float value = bf16_to_float(query_latent[latent_base + index]);
    if constexpr (!kRotationFused) {
      value *= signs1[index];
    }
    values[item] = value;
  }

  if constexpr (!kRotationFused) {
    hadamard_512_bf16_layout(values, lane);
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      const int index = (item / 8) * 256 + lane * 8 + (item % 8);
      const float rotated = values[item] * kInvSqrt512 * signs2[index];
      // The accepted explicit-rotation oracle materializes a BF16 tensor
      // before its separate FP8 query conversion. Preserve that rounding
      // point even though this kernel does not materialize the tensor.
      values[item] = bf16_to_float(__float2bfloat16_rn(rotated));
    }
  }

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const int item = pair * 2;
    const int index =
        kRotationFused ? lane * 16 + item
                       : (item / 8) * 256 + lane * 8 + (item % 8);
    *reinterpret_cast<__nv_fp8x2_storage_t*>(query_out + out_base + index) =
        float2_to_e4m3(values[item], values[item + 1]);
  }

  const int rope_index = lane * 2;
  *reinterpret_cast<__nv_fp8x2_storage_t*>(
      query_out + out_base + kLatentDim + rope_index) =
      float2_to_e4m3(bf16_to_float(query_rope[rope_base + rope_index]),
                     bf16_to_float(query_rope[rope_base + rope_index + 1]));
}

template <bool kStrict, bool kWriteCodebook>
__device__ __forceinline__ void write_cache_row(
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const int64_t* locations, const float* signs1, const float* signs2,
    const float* boundaries, const float* quant_centroids,
    const uint8_t* storage_code_lut, const float* decode_centroids,
    uint8_t* packed_cache, __nv_bfloat16* scale_cache, uint8_t* rope_cache,
    uint8_t* codebook_cache, int32_t* fault_status, const int token,
    const int64_t pool_size, const float scale_multiplier, const int lane) {
  const int64_t slot = locations[token];
  if (slot < 0 || slot >= pool_size) {
    if (lane == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(fault_status), 1u);
      if constexpr (kStrict) {
        asm volatile("trap;");
      }
    }
    return;
  }

  const int64_t latent_base = static_cast<int64_t>(token) * kLatentDim;
  float values[16];
  float norm_squared = 0.0f;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int index = (item / 4) * 128 + lane * 4 + (item % 4);
    const float value = bf16_to_float(cache_latent[latent_base + index]);
    values[item] = value;
    norm_squared = fmaf(value, value, norm_squared);
  }
  norm_squared = warp_sum(norm_squared);
  const float norm = sqrtf(norm_squared);
  const float safe_norm = norm > 0.0f ? norm : 1.0f;

#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int index = (item / 4) * 128 + lane * 4 + (item % 4);
    values[item] = (values[item] / safe_norm) * signs1[index];
  }
  hadamard_512_fp32_layout(values, lane);

  uint8_t bins[16];
  float quant_norm_squared = 0.0f;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int index = (item / 4) * 128 + lane * 4 + (item % 4);
    const float rotated = values[item] * kInvSqrt512 * signs2[index];
    int bin = 0;
#pragma unroll
    for (int boundary = 0; boundary < 15; ++boundary) {
      bin += rotated > boundaries[boundary];
    }
    bins[item] = static_cast<uint8_t>(bin);
    const float quantized = quant_centroids[bin];
    quant_norm_squared = fmaf(quantized, quantized, quant_norm_squared);
  }
  quant_norm_squared = warp_sum(quant_norm_squared);
  const float quant_norm = sqrtf(quant_norm_squared);
  const float safe_quant_norm = quant_norm > 1.0e-10f ? quant_norm : 1.0f;

  const int64_t packed_base = slot * kPackedDim;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      const uint8_t even = storage_code_lut[bins[chunk * 4 + 2 * item]];
      const uint8_t odd = storage_code_lut[bins[chunk * 4 + 2 * item + 1]];
      packed_cache[packed_base + chunk * 64 + lane * 2 + item] =
          static_cast<uint8_t>((odd << 4) | even);
    }
  }

  const float scale = (norm / safe_quant_norm) * scale_multiplier;
  const __nv_bfloat16 rounded_scale = __float2bfloat16_rn(scale);
  if (lane == 0) {
    scale_cache[slot] = rounded_scale;
  }

  const int64_t rope_src_base = static_cast<int64_t>(token) * kRopeDim;
  const int64_t rope_dst_base = slot * kRopeDim;
  const int rope_index = lane * 2;
  *reinterpret_cast<__nv_fp8x2_storage_t*>(rope_cache + rope_dst_base +
                                           rope_index) =
      float2_to_e4m3(bf16_to_float(cache_rope[rope_src_base + rope_index]),
                     bf16_to_float(cache_rope[rope_src_base + rope_index + 1]));

  if constexpr (kWriteCodebook) {
    if (lane < 8) {
      const float rounded_scale_fp32 = bf16_to_float(rounded_scale);
      const int centroid = lane * 2;
      const int64_t codebook_base = slot * 16;
      *reinterpret_cast<__nv_fp8x2_storage_t*>(
          codebook_cache + codebook_base + centroid) =
          float2_to_e4m3(rounded_scale_fp32 * decode_centroids[centroid],
                         rounded_scale_fp32 * decode_centroids[centroid + 1]);
    }
  }
}

template <int kWarps, bool kRotationFused, bool kStrict, bool kWriteCodebook>
__global__ void tq_mla_frontend_kernel(
    const __nv_bfloat16* query_latent, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const int64_t* locations, const float* signs1, const float* signs2,
    const float* boundaries, const float* quant_centroids,
    const uint8_t* storage_code_lut, const float* decode_centroids,
    uint8_t* query_out, uint8_t* packed_cache, __nv_bfloat16* scale_cache,
    uint8_t* rope_cache, uint8_t* codebook_cache, int32_t* fault_status,
    const int64_t pool_size,
    const float scale_multiplier) {
  const int token = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  for (int head = warp; head < kQueryHeads; head += kWarps) {
    write_query_head<kRotationFused>(query_latent, query_rope, signs1, signs2,
                                     query_out, token, head, lane);
  }

  if (warp == 0) {
    write_cache_row<kStrict, kWriteCodebook>(
        cache_latent, cache_rope, locations, signs1, signs2, boundaries,
        quant_centroids, storage_code_lut, decode_centroids, packed_cache,
        scale_cache, rope_cache, codebook_cache, fault_status, token,
        pool_size, scale_multiplier, lane);
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& tensor, const torch::Device device,
                       const char* name) {
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device,
              "; got ", tensor.device());
}

bool tensors_overlap(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (lhs.nbytes() == 0 || rhs.nbytes() == 0) {
    return false;
  }
  const auto lhs_begin = reinterpret_cast<uintptr_t>(lhs.data_ptr());
  const auto rhs_begin = reinterpret_cast<uintptr_t>(rhs.data_ptr());
  const auto lhs_end = lhs_begin + lhs.nbytes();
  const auto rhs_end = rhs_begin + rhs.nbytes();
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

template <int kWarps, bool kRotationFused, bool kStrict, bool kWriteCodebook>
void launch_frontend(
    const torch::Tensor& query_latent, const torch::Tensor& query_rope,
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2, const torch::Tensor& boundaries,
    const torch::Tensor& quant_centroids,
    const torch::Tensor& storage_code_lut,
    const std::optional<torch::Tensor>& decode_centroids,
    torch::Tensor& query_out, torch::Tensor& packed_cache,
    torch::Tensor& scale_cache, torch::Tensor& rope_cache,
    const std::optional<torch::Tensor>& codebook_cache,
    torch::Tensor& fault_status,
    const float scale_multiplier, const cudaStream_t stream) {
  const int64_t tokens = query_latent.size(0);
  if (tokens == 0) {
    return;
  }
  const dim3 block(kWarps * 32);
  const dim3 grid(tokens);
  tq_mla_frontend_kernel<kWarps, kRotationFused, kStrict, kWriteCodebook>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query_latent.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(query_rope.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(cache_latent.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(cache_rope.data_ptr()),
          locations.data_ptr<int64_t>(), signs1.data_ptr<float>(),
          signs2.data_ptr<float>(), boundaries.data_ptr<float>(),
          quant_centroids.data_ptr<float>(),
          storage_code_lut.data_ptr<uint8_t>(),
          kWriteCodebook ? decode_centroids->data_ptr<float>() : nullptr,
          reinterpret_cast<uint8_t*>(query_out.data_ptr()),
          packed_cache.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(scale_cache.data_ptr()),
          reinterpret_cast<uint8_t*>(rope_cache.data_ptr()),
          kWriteCodebook ? codebook_cache->data_ptr<uint8_t>() : nullptr,
          fault_status.data_ptr<int32_t>(), packed_cache.size(0),
          scale_multiplier);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void tq_mla_frontend_out(
    const torch::Tensor& query_latent, const torch::Tensor& query_rope,
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2, const torch::Tensor& boundaries,
    const torch::Tensor& quant_centroids,
    const torch::Tensor& storage_code_lut, torch::Tensor query_out,
    torch::Tensor packed_cache, torch::Tensor scale_cache,
    torch::Tensor rope_cache, torch::Tensor fault_status,
    const std::optional<torch::Tensor>& decode_centroids,
    const std::optional<torch::Tensor>& codebook_cache,
    const double scale_multiplier, const bool rotation_fused,
    const int64_t num_warps, const bool strict) {
  check_cuda_contiguous(query_latent, "query_latent");
  check_cuda_contiguous(query_rope, "query_rope");
  check_cuda_contiguous(cache_latent, "cache_latent");
  check_cuda_contiguous(cache_rope, "cache_rope");
  check_cuda_contiguous(locations, "locations");
  check_cuda_contiguous(signs1, "signs1");
  check_cuda_contiguous(signs2, "signs2");
  check_cuda_contiguous(boundaries, "boundaries");
  check_cuda_contiguous(quant_centroids, "quant_centroids");
  check_cuda_contiguous(storage_code_lut, "storage_code_lut");
  check_cuda_contiguous(query_out, "query_out");
  check_cuda_contiguous(packed_cache, "packed_cache");
  check_cuda_contiguous(scale_cache, "scale_cache");
  check_cuda_contiguous(rope_cache, "rope_cache");
  check_cuda_contiguous(fault_status, "fault_status");

  const bool write_codebook = codebook_cache.has_value();
  TORCH_CHECK(write_codebook == decode_centroids.has_value(),
              "decode_centroids and codebook_cache must be provided together");
  if (write_codebook) {
    check_cuda_contiguous(*decode_centroids, "decode_centroids");
    check_cuda_contiguous(*codebook_cache, "codebook_cache");
  }

  const torch::Device device = query_latent.device();
  check_same_device(query_rope, device, "query_rope");
  check_same_device(cache_latent, device, "cache_latent");
  check_same_device(cache_rope, device, "cache_rope");
  check_same_device(locations, device, "locations");
  check_same_device(signs1, device, "signs1");
  check_same_device(signs2, device, "signs2");
  check_same_device(boundaries, device, "boundaries");
  check_same_device(quant_centroids, device, "quant_centroids");
  check_same_device(storage_code_lut, device, "storage_code_lut");
  check_same_device(query_out, device, "query_out");
  check_same_device(packed_cache, device, "packed_cache");
  check_same_device(scale_cache, device, "scale_cache");
  check_same_device(rope_cache, device, "rope_cache");
  check_same_device(fault_status, device, "fault_status");
  if (write_codebook) {
    check_same_device(*decode_centroids, device, "decode_centroids");
    check_same_device(*codebook_cache, device, "codebook_cache");
  }

  TORCH_CHECK(query_latent.scalar_type() == at::kBFloat16,
              "query_latent must be bfloat16");
  TORCH_CHECK(query_rope.scalar_type() == at::kBFloat16,
              "query_rope must be bfloat16");
  TORCH_CHECK(cache_latent.scalar_type() == at::kBFloat16,
              "cache_latent must be bfloat16");
  TORCH_CHECK(cache_rope.scalar_type() == at::kBFloat16,
              "cache_rope must be bfloat16");
  TORCH_CHECK(locations.scalar_type() == at::kLong,
              "locations must be int64");
  TORCH_CHECK(signs1.scalar_type() == at::kFloat,
              "signs1 must be float32");
  TORCH_CHECK(signs2.scalar_type() == at::kFloat,
              "signs2 must be float32");
  TORCH_CHECK(boundaries.scalar_type() == at::kFloat,
              "boundaries must be float32");
  TORCH_CHECK(quant_centroids.scalar_type() == at::kFloat,
              "quant_centroids must be float32");
  TORCH_CHECK(storage_code_lut.scalar_type() == at::kByte,
              "storage_code_lut must be uint8");
  TORCH_CHECK(query_out.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "query_out must be float8_e4m3fn");
  TORCH_CHECK(packed_cache.scalar_type() == at::kByte,
              "packed_cache must be uint8");
  TORCH_CHECK(scale_cache.scalar_type() == at::kBFloat16,
              "scale_cache must be bfloat16");
  TORCH_CHECK(rope_cache.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "rope_cache must be float8_e4m3fn");
  TORCH_CHECK(fault_status.scalar_type() == at::kInt,
              "fault_status must be int32");
  if (write_codebook) {
    TORCH_CHECK(decode_centroids->scalar_type() == at::kFloat,
                "decode_centroids must be float32");
    TORCH_CHECK(codebook_cache->scalar_type() == at::kByte,
                "codebook_cache must be uint8");
  }

  TORCH_CHECK(query_latent.dim() == 3 && query_latent.size(1) == kQueryHeads &&
                  query_latent.size(2) == kLatentDim,
              "query_latent must have shape (T, 8, 512)");
  const int64_t tokens = query_latent.size(0);
  TORCH_CHECK(tokens <= std::numeric_limits<int>::max(),
              "token count exceeds the CUDA grid limit");
  TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == tokens &&
                  query_rope.size(1) == kQueryHeads &&
                  query_rope.size(2) == kRopeDim,
              "query_rope must have shape (T, 8, 64)");
  TORCH_CHECK(cache_latent.dim() == 3 && cache_latent.size(0) == tokens &&
                  cache_latent.size(1) == 1 &&
                  cache_latent.size(2) == kLatentDim,
              "cache_latent must have shape (T, 1, 512)");
  TORCH_CHECK(cache_rope.dim() == 3 && cache_rope.size(0) == tokens &&
                  cache_rope.size(1) == 1 && cache_rope.size(2) == kRopeDim,
              "cache_rope must have shape (T, 1, 64)");
  TORCH_CHECK(locations.dim() == 1 && locations.size(0) == tokens,
              "locations must have shape (T,)");
  TORCH_CHECK(signs1.dim() == 1 && signs1.size(0) == kLatentDim,
              "signs1 must have shape (512,)");
  TORCH_CHECK(signs2.dim() == 1 && signs2.size(0) == kLatentDim,
              "signs2 must have shape (512,)");
  TORCH_CHECK(boundaries.dim() == 1 && boundaries.size(0) == 15,
              "boundaries must have shape (15,)");
  TORCH_CHECK(quant_centroids.dim() == 1 &&
                  quant_centroids.size(0) == 16,
              "quant_centroids must have shape (16,)");
  TORCH_CHECK(storage_code_lut.dim() == 1 &&
                  storage_code_lut.size(0) == 16,
              "storage_code_lut must have shape (16,)");
  TORCH_CHECK(query_out.dim() == 3 && query_out.size(0) == tokens &&
                  query_out.size(1) == kQueryHeads &&
                  query_out.size(2) == kQueryDim,
              "query_out must have shape (T, 8, 576)");
  TORCH_CHECK(packed_cache.dim() == 3 && packed_cache.size(1) == 1 &&
                  packed_cache.size(2) == kPackedDim,
              "packed_cache must have shape (P, 1, 256)");
  TORCH_CHECK(scale_cache.dim() == 2 &&
                  scale_cache.size(0) == packed_cache.size(0) &&
                  scale_cache.size(1) == 1,
              "scale_cache must have shape (P, 1)");
  TORCH_CHECK(rope_cache.dim() == 3 &&
                  rope_cache.size(0) == packed_cache.size(0) &&
                  rope_cache.size(1) == 1 && rope_cache.size(2) == kRopeDim,
              "rope_cache must have shape (P, 1, 64)");
  TORCH_CHECK(fault_status.numel() == 1,
              "fault_status must contain exactly one int32 value");
  if (write_codebook) {
    TORCH_CHECK(decode_centroids->dim() == 1 &&
                    decode_centroids->size(0) == 16,
                "decode_centroids must have shape (16,)");
    TORCH_CHECK(codebook_cache->dim() == 3 &&
                    codebook_cache->size(0) == packed_cache.size(0) &&
                    codebook_cache->size(1) == 1 &&
                    codebook_cache->size(2) == 16,
                "codebook_cache must have shape (P, 1, 16)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(codebook_cache->data_ptr()) % 16 == 0,
                "codebook_cache must be 16-byte aligned");
  }
  TORCH_CHECK(packed_cache.size(0) > 0, "pool size must be positive");
  TORCH_CHECK(scale_multiplier > 0.0,
              "scale_multiplier must be positive");
  TORCH_CHECK(num_warps == 1 || num_warps == 2 || num_warps == 4 ||
                  num_warps == 8,
              "num_warps must be one of 1, 2, 4, or 8");

  std::vector<std::pair<const torch::Tensor*, const char*>> outputs = {
      {&query_out, "query_out"},       {&packed_cache, "packed_cache"},
      {&scale_cache, "scale_cache"},   {&rope_cache, "rope_cache"},
      {&fault_status, "fault_status"},
  };
  std::vector<std::pair<const torch::Tensor*, const char*>> inputs = {
      {&query_latent, "query_latent"},
      {&query_rope, "query_rope"},
      {&cache_latent, "cache_latent"},
      {&cache_rope, "cache_rope"},
      {&locations, "locations"},
      {&signs1, "signs1"},
      {&signs2, "signs2"},
      {&boundaries, "boundaries"},
      {&quant_centroids, "quant_centroids"},
      {&storage_code_lut, "storage_code_lut"},
  };
  if (write_codebook) {
    outputs.push_back({&*codebook_cache, "codebook_cache"});
    inputs.push_back({&*decode_centroids, "decode_centroids"});
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = i + 1; j < outputs.size(); ++j) {
      TORCH_CHECK(!tensors_overlap(*outputs[i].first, *outputs[j].first),
                  outputs[i].second, " must not overlap ", outputs[j].second);
    }
    for (const auto& input : inputs) {
      TORCH_CHECK(!tensors_overlap(*outputs[i].first, *input.first),
                  outputs[i].second, " must not overlap ", input.second);
    }
  }

  c10::cuda::CUDAGuard device_guard(device);
  const int device_index = device.index();
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(properties->major == 10 && properties->minor == 0,
              "H43 native frontend requires SM100; got compute capability ",
              properties->major, ".", properties->minor);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device_index).stream();
  const float multiplier = static_cast<float>(scale_multiplier);

#define DISPATCH_FRONTEND_CODEBOOK(WARPS, ROTATION, STRICT, CODEBOOK)     \
  launch_frontend<WARPS, ROTATION, STRICT, CODEBOOK>(                     \
      query_latent, query_rope, cache_latent, cache_rope, locations,      \
      signs1, signs2, boundaries, quant_centroids, storage_code_lut,      \
      decode_centroids, query_out, packed_cache, scale_cache, rope_cache,  \
      codebook_cache, fault_status, multiplier, stream)

#define DISPATCH_FRONTEND(WARPS, ROTATION, STRICT)                    \
  if (write_codebook) {                                               \
    DISPATCH_FRONTEND_CODEBOOK(WARPS, ROTATION, STRICT, true);        \
  } else {                                                            \
    DISPATCH_FRONTEND_CODEBOOK(WARPS, ROTATION, STRICT, false);       \
  }

#define DISPATCH_STRICT(WARPS, ROTATION) \
  if (strict) {                         \
    DISPATCH_FRONTEND(WARPS, ROTATION, true); \
  } else {                              \
    DISPATCH_FRONTEND(WARPS, ROTATION, false); \
  }

#define DISPATCH_ROTATION(WARPS)       \
  if (rotation_fused) {                \
    DISPATCH_STRICT(WARPS, true);      \
  } else {                             \
    DISPATCH_STRICT(WARPS, false);     \
  }

  switch (num_warps) {
    case 1:
      DISPATCH_ROTATION(1);
      break;
    case 2:
      DISPATCH_ROTATION(2);
      break;
    case 4:
      DISPATCH_ROTATION(4);
      break;
    case 8:
      DISPATCH_ROTATION(8);
      break;
  }

#undef DISPATCH_ROTATION
#undef DISPATCH_STRICT
#undef DISPATCH_FRONTEND
#undef DISPATCH_FRONTEND_CODEBOOK
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("tq_mla_frontend_out", &tq_mla_frontend_out,
             "H43 SM100 TurboQuant MLA query and codebook-writer front end");
}
