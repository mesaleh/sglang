#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

constexpr int kQueryHeads = 8;
constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kPackedDim = kLatentDim / 2;
constexpr float kInvSqrt512 = 0.044194173824159220275f;

constexpr unsigned int kFaultInvalidLocation = 1u << 0;
constexpr unsigned int kFaultInvalidPosition = 1u << 1;
constexpr unsigned int kFaultNonfiniteLatent = 1u << 2;
constexpr unsigned int kFaultNonfiniteQueryRope = 1u << 3;
constexpr unsigned int kFaultInvalidScale = 1u << 4;
constexpr unsigned int kFaultNonfiniteCacheRope = 1u << 5;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_fp8x2_storage_t float2_to_e4m3(
    const float first, const float second) {
  return __nv_cvt_float2_to_fp8x2(make_float2(first, second),
                                  __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float apply_neox_rope_bf16(
    const __nv_bfloat16* rope, const int64_t rope_base, const int index,
    const float* cos_sin_cache, const int64_t cos_sin_base) {
  const int half_index = index & (kRopeDim / 2 - 1);
  const float first = bf16_to_float(rope[rope_base + half_index]);
  const float second =
      bf16_to_float(rope[rope_base + half_index + kRopeDim / 2]);
  const float cosine = cos_sin_cache[cos_sin_base + half_index];
  const float sine =
      cos_sin_cache[cos_sin_base + half_index + kRopeDim / 2];
  const float rotated = index < kRopeDim / 2
                            ? first * cosine - second * sine
                            : second * cosine + first * sine;
  // Preserve the selected-path oracle's explicit BF16 RoPE materialization
  // before the subsequent SATFINITE E4M3 conversion.
  return bf16_to_float(__float2bfloat16_rn(rotated));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ uint8_t native_e2m1x2_rne(
    const float even, const float odd) {
  uint32_t packed;
  // PTX places the first source in the upper nibble. Passing odd first and
  // even second preserves N10's (odd << 4) | even byte convention.
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(packed)
      : "f"(even), "f"(odd));
  return static_cast<uint8_t>(packed);
}

__device__ __forceinline__ uint8_t correct_n10_code(
    const uint8_t raw_code, const float value) {
  constexpr uint32_t kMagnitudeMask = 0x7fffffffu;
  constexpr uint32_t kMidpointLowMask = 0x001fffffu;
  constexpr uint32_t kPositiveCorrectionMask = 0x00008840u;
  constexpr uint32_t kNegativeCorrectionMask = 0x00022201u;

  const uint32_t bits = __float_as_uint(value);
  const uint32_t sign = bits >> 31;
  const uint32_t magnitude_bits = bits & kMagnitudeMask;
  const uint32_t offset = (magnitude_bits >> 21) - 500u;
  const uint32_t sign_mask = 0u - sign;
  const uint32_t correction_mask =
      kPositiveCorrectionMask ^
      ((kPositiveCorrectionMask ^ kNegativeCorrectionMask) & sign_mask);
  const uint32_t correction =
      ((correction_mask >> (offset & 31u)) & 1u) *
      static_cast<uint32_t>((magnitude_bits & kMidpointLowMask) == 0u) *
      static_cast<uint32_t>(offset <= 17u);

  const uint32_t adjusted = static_cast<uint32_t>(raw_code) +
                            correction * ((sign << 1) - 1u);
  const uint32_t magnitude_code = adjusted & 7u;
  const uint32_t keep =
      (magnitude_code | (0u - magnitude_code)) >> 31;
  return static_cast<uint8_t>(adjusted * keep);
}

__device__ __forceinline__ float2 native_e2m1x2_to_float2(
    const uint8_t packed) {
  const uint32_t source = packed;
  uint32_t fp16x2;
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "mov.b32 {byte0, byte1, byte2, byte3}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
      "}\n"
      : "=r"(fp16x2)
      : "r"(source));
  const __half low =
      __ushort_as_half(static_cast<unsigned short>(fp16x2));
  const __half high =
      __ushort_as_half(static_cast<unsigned short>(fp16x2 >> 16));
  return __half22float2(__halves2half2(low, high));
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

struct LaneCandidate {
  uint64_t data0;
  uint64_t data1;
  uint32_t aux0;
  uint32_t aux1;
  unsigned int faults;
  bool zero_row;
};

__device__ __forceinline__ uint16_t fp8x2_bits(
    const __nv_fp8x2_storage_t value) {
  union {
    __nv_fp8x2_storage_t fp8;
    uint16_t bits;
  } cast;
  cast.fp8 = value;
  return cast.bits;
}

__device__ __forceinline__ __nv_fp8x2_storage_t bits_to_fp8x2(
    const uint16_t value) {
  union {
    __nv_fp8x2_storage_t fp8;
    uint16_t bits;
  } cast;
  cast.bits = value;
  return cast.fp8;
}

__device__ __forceinline__ uint16_t bf16_bits(
    const __nv_bfloat16 value) {
  union {
    __nv_bfloat16 bf16;
    uint16_t bits;
  } cast;
  cast.bf16 = value;
  return cast.bits;
}

__device__ __forceinline__ __nv_bfloat16 bits_to_bf16(
    const uint16_t value) {
  union {
    __nv_bfloat16 bf16;
    uint16_t bits;
  } cast;
  cast.bits = value;
  return cast.bf16;
}

template <bool kRotationFused, bool kApplyRope>
__device__ __forceinline__ LaneCandidate compute_query_head(
    const __nv_bfloat16* query_latent, const __nv_bfloat16* query_rope,
    const float* signs1, const float* signs2,
    const float* cos_sin_cache, const int64_t* positions,
    const int64_t cos_sin_stride_t, const int token, const int head,
    const int lane) {
  LaneCandidate candidate;
  candidate.data0 = 0;
  candidate.data1 = 0;
  candidate.aux0 = 0;
  candidate.aux1 = 0;
  candidate.faults = 0;
  candidate.zero_row = false;
  const int64_t latent_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kLatentDim;
  const int64_t rope_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kRopeDim;

  float values[16];
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int index =
        kRotationFused ? lane * 16 + item
                       : (item / 8) * 256 + lane * 8 + (item % 8);
    float value = bf16_to_float(query_latent[latent_base + index]);
    if (!isfinite(value)) {
      candidate.faults |= kFaultNonfiniteLatent;
    }
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
      if (!isfinite(values[item])) {
        candidate.faults |= kFaultNonfiniteLatent;
      }
    }
  }

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const int item = pair * 2;
    const int index =
        kRotationFused ? lane * 16 + item
                       : (item / 8) * 256 + lane * 8 + (item % 8);
    (void)index;
    const uint64_t bits = fp8x2_bits(
        float2_to_e4m3(values[item], values[item + 1]));
    if (pair < 4) {
      candidate.data0 |= bits << (pair * 16);
    } else {
      candidate.data1 |= bits << ((pair - 4) * 16);
    }
  }

  const int rope_index = lane * 2;
  float rope_first;
  float rope_second;
  if constexpr (kApplyRope) {
    const int64_t cos_sin_base =
        positions[token] * cos_sin_stride_t;
    rope_first = apply_neox_rope_bf16(
        query_rope, rope_base, rope_index, cos_sin_cache, cos_sin_base);
    rope_second = apply_neox_rope_bf16(
        query_rope, rope_base, rope_index + 1, cos_sin_cache, cos_sin_base);
  } else {
    rope_first = bf16_to_float(query_rope[rope_base + rope_index]);
    rope_second =
        bf16_to_float(query_rope[rope_base + rope_index + 1]);
  }
  const __nv_bfloat16 rope_first_bf16 = __float2bfloat16_rn(rope_first);
  const __nv_bfloat16 rope_second_bf16 = __float2bfloat16_rn(rope_second);
  candidate.aux0 = static_cast<uint32_t>(bf16_bits(rope_first_bf16)) |
                   (static_cast<uint32_t>(bf16_bits(rope_second_bf16)) << 16);
  if (!isfinite(rope_first) || !isfinite(rope_second) ||
      !isfinite(bf16_to_float(rope_first_bf16)) ||
      !isfinite(bf16_to_float(rope_second_bf16))) {
    candidate.faults |= kFaultNonfiniteQueryRope;
  }
  return candidate;
}

template <bool kRotationFused>
__device__ __forceinline__ void store_query_head(
    const LaneCandidate& candidate, uint8_t* query_latent_out,
    __nv_bfloat16* query_rope_out, const int token, const int head,
    const int lane) {
  const int64_t latent_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kLatentDim;
  const int64_t rope_base =
      (static_cast<int64_t>(token) * kQueryHeads + head) * kRopeDim;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const int item = pair * 2;
    const int index =
        kRotationFused ? lane * 16 + item
                       : (item / 8) * 256 + lane * 8 + (item % 8);
    const uint64_t data = pair < 4 ? candidate.data0 : candidate.data1;
    const int shift = (pair < 4 ? pair : pair - 4) * 16;
    *reinterpret_cast<__nv_fp8x2_storage_t*>(
        query_latent_out + latent_base + index) =
        bits_to_fp8x2(static_cast<uint16_t>(data >> shift));
  }
  query_rope_out[rope_base + lane * 2] =
      bits_to_bf16(static_cast<uint16_t>(candidate.aux0));
  query_rope_out[rope_base + lane * 2 + 1] =
      bits_to_bf16(static_cast<uint16_t>(candidate.aux0 >> 16));
}

template <bool kApplyRope>
__device__ __forceinline__ LaneCandidate compute_cache_row(
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const float* signs1, const float* signs2,
    const int token, const float grid,
    const int64_t latent_stride_t, const int64_t latent_stride_h,
    const int64_t rope_src_stride_t, const int64_t rope_src_stride_h,
    const float* cos_sin_cache,
    const int64_t* positions, const int64_t cos_sin_stride_t,
    const int lane) {
  LaneCandidate candidate;
  candidate.data0 = 0;
  candidate.data1 = 0;
  candidate.aux0 = 0;
  candidate.aux1 = 0;
  candidate.faults = 0;
  const int64_t latent_base =
      static_cast<int64_t>(token) * latent_stride_t;
  float values[16];
  float norm_squared = 0.0f;
  bool lane_has_nonzero = false;
  bool lane_has_nonfinite = false;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int index = (item / 4) * 128 + lane * 4 + (item % 4);
    const float value = bf16_to_float(
        cache_latent[latent_base + index * latent_stride_h]);
    values[item] = value;
    lane_has_nonzero = lane_has_nonzero || value != 0.0f;
    lane_has_nonfinite = lane_has_nonfinite || !isfinite(value);
    norm_squared = fmaf(value, value, norm_squared);
  }
  const bool row_has_nonzero =
      __any_sync(0xffffffffu, lane_has_nonzero);
  const bool row_has_nonfinite =
      __any_sync(0xffffffffu, lane_has_nonfinite);
  candidate.zero_row = !row_has_nonzero && !row_has_nonfinite;
  if (row_has_nonfinite) {
    candidate.faults |= kFaultNonfiniteLatent;
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

  float quant_norm_squared = 0.0f;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const int even_item = pair * 2;
    const int odd_item = even_item + 1;
    const int even_index =
        (even_item / 4) * 128 + lane * 4 + (even_item % 4);
    const int odd_index =
        (odd_item / 4) * 128 + lane * 4 + (odd_item % 4);
    const float even_rotated =
        values[even_item] * kInvSqrt512 * signs2[even_index];
    const float odd_rotated =
        values[odd_item] * kInvSqrt512 * signs2[odd_index];
    const float even_scaled = even_rotated / grid;
    const float odd_scaled = odd_rotated / grid;
    const uint8_t raw = native_e2m1x2_rne(even_scaled, odd_scaled);
    const uint8_t even_code = correct_n10_code(raw & 0x0fu, even_scaled);
    const uint8_t odd_code = correct_n10_code(raw >> 4, odd_scaled);
    const uint8_t packed =
        static_cast<uint8_t>((odd_code << 4) | even_code);
    if (pair < 4) {
      candidate.data0 |= static_cast<uint64_t>(packed) << (pair * 8);
    } else {
      candidate.data1 |=
          static_cast<uint64_t>(packed) << ((pair - 4) * 8);
    }
    const float2 reconstructed = native_e2m1x2_to_float2(packed);
    const float even_quantized = reconstructed.x * grid;
    const float odd_quantized = reconstructed.y * grid;
    // Preserve S0's even-then-odd sequential accumulation order.
    quant_norm_squared =
        fmaf(even_quantized, even_quantized, quant_norm_squared);
    quant_norm_squared =
        fmaf(odd_quantized, odd_quantized, quant_norm_squared);
  }
  quant_norm_squared = warp_sum(quant_norm_squared);
  const float quant_norm = sqrtf(quant_norm_squared);
  const float safe_quant_norm = quant_norm > 0.0f ? quant_norm : 1.0f;

  const float scale = norm > 0.0f ? (norm / safe_quant_norm) * grid : 0.0f;
  const __nv_bfloat16 scale_bf16 =
      candidate.zero_row ? __float2bfloat16_rn(1.0f)
                         : __float2bfloat16_rn(scale);
  const float rounded_scale = bf16_to_float(scale_bf16);
  const bool scale_is_valid =
      candidate.zero_row || (rounded_scale != 0.0f && isfinite(rounded_scale));
  if (!scale_is_valid) {
    candidate.faults |= kFaultInvalidScale;
  }

  const int64_t rope_src_base =
      static_cast<int64_t>(token) * rope_src_stride_t;
  const int rope_index = lane * 2;
  float rope_first;
  float rope_second;
  if constexpr (kApplyRope) {
    const int64_t cos_sin_base =
        positions[token] * cos_sin_stride_t;
    rope_first = apply_neox_rope_bf16(
        cache_rope, rope_src_base, rope_index, cos_sin_cache, cos_sin_base);
    rope_second = apply_neox_rope_bf16(
        cache_rope, rope_src_base, rope_index + 1, cos_sin_cache,
        cos_sin_base);
  } else {
    rope_first = bf16_to_float(
        cache_rope[rope_src_base + rope_index * rope_src_stride_h]);
    rope_second = bf16_to_float(
        cache_rope[rope_src_base + (rope_index + 1) * rope_src_stride_h]);
  }
  if (!isfinite(rope_first) || !isfinite(rope_second)) {
    candidate.faults |= kFaultNonfiniteCacheRope;
  }
  const float division_scale = scale_is_valid ? rounded_scale : 1.0f;
  const float stored_first =
      candidate.zero_row ? rope_first : rope_first / division_scale;
  const float stored_second =
      candidate.zero_row ? rope_second : rope_second / division_scale;
  const __nv_bfloat16 stored_first_bf16 = __float2bfloat16_rn(stored_first);
  const __nv_bfloat16 stored_second_bf16 = __float2bfloat16_rn(stored_second);
  candidate.aux0 = static_cast<uint32_t>(bf16_bits(scale_bf16)) |
                   (static_cast<uint32_t>(bf16_bits(stored_first_bf16)) << 16);
  candidate.aux1 = static_cast<uint32_t>(bf16_bits(stored_second_bf16));
  if (!isfinite(stored_first) || !isfinite(stored_second) ||
      !isfinite(bf16_to_float(stored_first_bf16)) ||
      !isfinite(bf16_to_float(stored_second_bf16))) {
    candidate.faults |= kFaultNonfiniteCacheRope;
  }
  return candidate;
}

__device__ __forceinline__ void store_cache_row(
    const LaneCandidate& candidate, uint8_t* packed_cache,
    __nv_bfloat16* scale_cache,
    __nv_bfloat16* rope_cache, const int64_t slot,
    const int64_t packed_stride_s, const int64_t packed_stride_h,
    const int64_t scale_stride_s, const int64_t rope_dst_stride_s,
    const int64_t rope_dst_stride_h, const int lane) {
  const int64_t packed_base = slot * packed_stride_s;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      const int pair = chunk * 2 + item;
      const uint64_t data =
          pair < 4 ? candidate.data0 : candidate.data1;
      const int shift = (pair < 4 ? pair : pair - 4) * 8;
      packed_cache[packed_base + chunk * 64 +
                   lane * 2 * packed_stride_h + item * packed_stride_h] =
          static_cast<uint8_t>(data >> shift);
    }
  }
  if (lane == 0) {
    scale_cache[slot * scale_stride_s] =
        bits_to_bf16(static_cast<uint16_t>(candidate.aux0));
  }
  const int64_t rope_dst_base = slot * rope_dst_stride_s;
  rope_cache[rope_dst_base + lane * 2 * rope_dst_stride_h] =
      bits_to_bf16(static_cast<uint16_t>(candidate.aux0 >> 16));
  rope_cache[rope_dst_base + (lane * 2 + 1) * rope_dst_stride_h] =
      bits_to_bf16(static_cast<uint16_t>(candidate.aux1));
}

template <bool kRotationFused, bool kStrict, typename index_t, bool kApplyRope>
__device__ __forceinline__ void tq_mla_frontend_body(
    const __nv_bfloat16* query_latent, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const index_t* locations, const float* signs1, const float* signs2,
    uint8_t* query_latent_out, __nv_bfloat16* query_rope_out,
    uint8_t* packed_cache,
    __nv_bfloat16* scale_cache, __nv_bfloat16* rope_cache,
    int32_t* fault_status, int64_t* zero_count, const int64_t pool_size,
    const float grid, const int64_t cache_latent_stride_t,
    const int64_t cache_latent_stride_h, const int64_t cache_rope_stride_t,
    const int64_t cache_rope_stride_h, const int64_t packed_stride_s,
    const int64_t packed_stride_h, const int64_t scale_stride_s,
    const int64_t rope_cache_stride_s, const int64_t rope_cache_stride_h,
    const float* cos_sin_cache, const int64_t* positions,
    const int64_t cos_sin_stride_t, const int64_t max_positions,
    unsigned int* warp_faults) {
  const int token = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  unsigned int faults = 0;
  const int64_t slot = static_cast<int64_t>(locations[token]);
  if (slot < 0 || slot >= pool_size) {
    faults |= kFaultInvalidLocation;
  }
  if constexpr (kApplyRope) {
    const int64_t position = positions[token];
    if (position < 0 || position >= max_positions) {
      faults |= kFaultInvalidPosition;
    }
  }
  if (faults != 0) {
    if (threadIdx.x == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(fault_status), faults);
      if constexpr (kStrict) {
        asm volatile("trap;");
      }
    }
    return;
  }

  LaneCandidate candidate;
  if (warp < kQueryHeads) {
    candidate = compute_query_head<kRotationFused, kApplyRope>(
        query_latent, query_rope, signs1, signs2, cos_sin_cache, positions,
        cos_sin_stride_t, token, warp, lane);
  } else {
    candidate = compute_cache_row<kApplyRope>(
        cache_latent, cache_rope, signs1, signs2, token, grid,
        cache_latent_stride_t, cache_latent_stride_h,
        cache_rope_stride_t, cache_rope_stride_h, cos_sin_cache, positions,
        cos_sin_stride_t, lane);
  }
  faults = candidate.faults;

  const unsigned int warp_fault = __reduce_or_sync(0xffffffffu, faults);
  if (lane == 0) {
    warp_faults[warp] = warp_fault;
  }
  __syncthreads();

  unsigned int block_fault = 0;
#pragma unroll
  for (int index = 0; index < kQueryHeads + 1; ++index) {
    block_fault |= warp_faults[index];
  }
  if (block_fault != 0) {
    if (threadIdx.x == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(fault_status), block_fault);
      if constexpr (kStrict) {
        asm volatile("trap;");
      }
    }
    return;
  }

  if (warp < kQueryHeads) {
    store_query_head<kRotationFused>(candidate, query_latent_out,
                                     query_rope_out, token, warp, lane);
  } else {
    store_cache_row(candidate, packed_cache, scale_cache, rope_cache, slot,
                    packed_stride_s, packed_stride_h,
                    scale_stride_s, rope_cache_stride_s,
                    rope_cache_stride_h, lane);
    if (lane == 0 && candidate.zero_row) {
      atomicAdd(reinterpret_cast<unsigned long long*>(zero_count), 1ULL);
    }
  }
}

template <bool kStrict, typename index_t, bool kApplyRope>
__global__ void tq_mla_frontend_unfused_kernel(
    const __nv_bfloat16* query_latent, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const index_t* locations, const float* signs1, const float* signs2,
    uint8_t* query_latent_out, __nv_bfloat16* query_rope_out,
    uint8_t* packed_cache,
    __nv_bfloat16* scale_cache, __nv_bfloat16* rope_cache,
    int32_t* fault_status, int64_t* zero_count, const int64_t pool_size,
    const float grid, const int64_t cache_latent_stride_t,
    const int64_t cache_latent_stride_h, const int64_t cache_rope_stride_t,
    const int64_t cache_rope_stride_h, const int64_t packed_stride_s,
    const int64_t packed_stride_h, const int64_t scale_stride_s,
    const int64_t rope_cache_stride_s, const int64_t rope_cache_stride_h,
    const float* cos_sin_cache, const int64_t* positions,
    const int64_t cos_sin_stride_t, const int64_t max_positions) {
  __shared__ unsigned int warp_faults[kQueryHeads + 1];
  tq_mla_frontend_body<false, kStrict, index_t, kApplyRope>(
      query_latent, query_rope, cache_latent, cache_rope, locations, signs1,
      signs2, query_latent_out, query_rope_out, packed_cache, scale_cache,
      rope_cache, fault_status,
      zero_count, pool_size, grid, cache_latent_stride_t,
      cache_latent_stride_h, cache_rope_stride_t, cache_rope_stride_h,
      packed_stride_s, packed_stride_h, scale_stride_s,
      rope_cache_stride_s, rope_cache_stride_h, cos_sin_cache, positions,
      cos_sin_stride_t, max_positions, warp_faults);
}

template <bool kStrict, typename index_t, bool kApplyRope>
__global__ void __launch_bounds__(288, 4) tq_mla_frontend_fused_kernel(
    const __nv_bfloat16* query_latent, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const index_t* locations, const float* signs1, const float* signs2,
    uint8_t* query_latent_out, __nv_bfloat16* query_rope_out,
    uint8_t* packed_cache,
    __nv_bfloat16* scale_cache, __nv_bfloat16* rope_cache,
    int32_t* fault_status, int64_t* zero_count, const int64_t pool_size,
    const float grid, const int64_t cache_latent_stride_t,
    const int64_t cache_latent_stride_h, const int64_t cache_rope_stride_t,
    const int64_t cache_rope_stride_h, const int64_t packed_stride_s,
    const int64_t packed_stride_h, const int64_t scale_stride_s,
    const int64_t rope_cache_stride_s, const int64_t rope_cache_stride_h,
    const float* cos_sin_cache, const int64_t* positions,
    const int64_t cos_sin_stride_t, const int64_t max_positions) {
  __shared__ unsigned int warp_faults[kQueryHeads + 1];
  tq_mla_frontend_body<true, kStrict, index_t, kApplyRope>(
      query_latent, query_rope, cache_latent, cache_rope, locations, signs1,
      signs2, query_latent_out, query_rope_out, packed_cache, scale_cache,
      rope_cache, fault_status,
      zero_count, pool_size, grid, cache_latent_stride_t,
      cache_latent_stride_h, cache_rope_stride_t, cache_rope_stride_h,
      packed_stride_s, packed_stride_h, scale_stride_s,
      rope_cache_stride_s, rope_cache_stride_h, cos_sin_cache, positions,
      cos_sin_stride_t, max_positions, warp_faults);
}

template <bool kStrict, typename index_t>
__global__ void tq_mla_cache_writer_kernel(
    const __nv_bfloat16* cache_latent, const __nv_bfloat16* cache_rope,
    const index_t* locations, const float* signs1, const float* signs2,
    uint8_t* packed_cache, __nv_bfloat16* scale_cache,
    __nv_bfloat16* rope_cache,
    int32_t* fault_status, int64_t* zero_count, const int64_t pool_size,
    const float grid,
    const int64_t cache_latent_stride_t,
    const int64_t cache_latent_stride_h, const int64_t cache_rope_stride_t,
    const int64_t cache_rope_stride_h, const int64_t packed_stride_s,
    const int64_t packed_stride_h, const int64_t scale_stride_s,
    const int64_t rope_cache_stride_s, const int64_t rope_cache_stride_h) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x;
  const int64_t slot = static_cast<int64_t>(locations[token]);
  if (slot < 0 || slot >= pool_size) {
    if (lane == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(fault_status),
               kFaultInvalidLocation);
      if constexpr (kStrict) {
        asm volatile("trap;");
      }
    }
    return;
  }

  const LaneCandidate candidate = compute_cache_row<false>(
      cache_latent, cache_rope, signs1, signs2, token, grid,
      cache_latent_stride_t, cache_latent_stride_h,
      cache_rope_stride_t, cache_rope_stride_h, nullptr, nullptr, 0, lane);
  const unsigned int faults =
      __reduce_or_sync(0xffffffffu, candidate.faults);
  if (faults != 0) {
    if (lane == 0) {
      atomicOr(reinterpret_cast<unsigned int*>(fault_status), faults);
      if constexpr (kStrict) {
        asm volatile("trap;");
      }
    }
    return;
  }

  store_cache_row(candidate, packed_cache, scale_cache, rope_cache, slot,
                  packed_stride_s, packed_stride_h,
                  scale_stride_s, rope_cache_stride_s,
                  rope_cache_stride_h, lane);
  if (lane == 0 && candidate.zero_row) {
    atomicAdd(reinterpret_cast<unsigned long long*>(zero_count), 1ULL);
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_inner_contiguous(const torch::Tensor& tensor,
                                 const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() > 0 && tensor.stride(tensor.dim() - 1) == 1,
              name, " must have a contiguous feature dimension");
}

void check_same_device(const torch::Tensor& tensor, const torch::Device device,
                       const char* name) {
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device,
              "; got ", tensor.device());
}

uintptr_t tensor_span_end(const torch::Tensor& tensor) {
  if (tensor.numel() == 0) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr());
  }
  int64_t max_element_offset = 0;
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    TORCH_CHECK(tensor.stride(dim) >= 0,
                "negative-stride tensors are not supported");
    if (tensor.size(dim) > 0) {
      max_element_offset +=
          (tensor.size(dim) - 1) * tensor.stride(dim);
    }
  }
  const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  return begin + static_cast<uintptr_t>(max_element_offset + 1) *
                     tensor.element_size();
}

bool tensors_overlap(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (lhs.numel() == 0 || rhs.numel() == 0) {
    return false;
  }
  const auto lhs_begin = reinterpret_cast<uintptr_t>(lhs.data_ptr());
  const auto rhs_begin = reinterpret_cast<uintptr_t>(rhs.data_ptr());
  return lhs_begin < tensor_span_end(rhs) &&
         rhs_begin < tensor_span_end(lhs);
}

using NamedTensor = std::pair<const torch::Tensor*, const char*>;

void check_output_aliases(const std::vector<NamedTensor>& outputs,
                          const std::vector<NamedTensor>& inputs) {
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
}

void validate_fixed_n10_cache_contract(
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    const torch::Tensor& packed_cache, const torch::Tensor& scale_cache,
    const torch::Tensor& rope_cache, const torch::Tensor& fault_status,
    const torch::Tensor& zero_count, const double grid) {
  check_cuda_inner_contiguous(cache_latent, "cache_latent");
  check_cuda_inner_contiguous(cache_rope, "cache_rope");
  check_cuda_contiguous(locations, "locations");
  check_cuda_contiguous(signs1, "signs1");
  check_cuda_contiguous(signs2, "signs2");
  check_cuda_contiguous(packed_cache, "packed_cache");
  check_cuda_contiguous(scale_cache, "scale_cache");
  check_cuda_contiguous(rope_cache, "rope_cache");
  check_cuda_contiguous(fault_status, "fault_status");
  check_cuda_contiguous(zero_count, "zero_count");

  const torch::Device device = cache_latent.device();
  for (const auto& named : std::vector<NamedTensor>{
           {&cache_rope, "cache_rope"},
           {&locations, "locations"},
           {&signs1, "signs1"},
           {&signs2, "signs2"},
           {&packed_cache, "packed_cache"},
           {&scale_cache, "scale_cache"},
           {&rope_cache, "rope_cache"},
           {&fault_status, "fault_status"},
           {&zero_count, "zero_count"},
       }) {
    check_same_device(*named.first, device, named.second);
  }

  TORCH_CHECK(cache_latent.scalar_type() == at::kBFloat16,
              "cache_latent must be bfloat16");
  TORCH_CHECK(cache_rope.scalar_type() == at::kBFloat16,
              "cache_rope must be bfloat16");
  TORCH_CHECK(locations.scalar_type() == at::kInt ||
                  locations.scalar_type() == at::kLong,
              "locations must be int32 or int64");
  TORCH_CHECK(signs1.scalar_type() == at::kFloat,
              "signs1 must be float32");
  TORCH_CHECK(signs2.scalar_type() == at::kFloat,
              "signs2 must be float32");
  TORCH_CHECK(packed_cache.scalar_type() == at::kByte,
              "packed_cache must be uint8");
  TORCH_CHECK(scale_cache.scalar_type() == at::kBFloat16,
              "scale_cache must be bfloat16");
  TORCH_CHECK(rope_cache.scalar_type() == at::kBFloat16,
              "rope_cache must be bfloat16");
  TORCH_CHECK(fault_status.scalar_type() == at::kInt,
              "fault_status must be int32");
  TORCH_CHECK(zero_count.scalar_type() == at::kLong,
              "zero_count must be int64");

  TORCH_CHECK(cache_latent.dim() == 3 && cache_latent.size(1) == 1 &&
                  cache_latent.size(2) == kLatentDim,
              "cache_latent must have shape (T, 1, 512)");
  const int64_t tokens = cache_latent.size(0);
  TORCH_CHECK(tokens <= std::numeric_limits<int>::max(),
              "token count exceeds the CUDA grid limit");
  TORCH_CHECK(cache_rope.dim() == 3 && cache_rope.size(0) == tokens &&
                  cache_rope.size(1) == 1 &&
                  cache_rope.size(2) == kRopeDim,
              "cache_rope must have shape (T, 1, 64)");
  TORCH_CHECK(locations.dim() == 1 && locations.size(0) == tokens,
              "locations must have shape (T,)");
  TORCH_CHECK(signs1.dim() == 1 && signs1.size(0) == kLatentDim,
              "signs1 must have shape (512,)");
  TORCH_CHECK(signs2.dim() == 1 && signs2.size(0) == kLatentDim,
              "signs2 must have shape (512,)");
  TORCH_CHECK(packed_cache.dim() == 3 && packed_cache.size(1) == 1 &&
                  packed_cache.size(2) == kPackedDim,
              "packed_cache must have shape (P, 1, 256)");
  TORCH_CHECK(scale_cache.dim() == 2 &&
                  scale_cache.size(0) == packed_cache.size(0) &&
                  scale_cache.size(1) == 1,
              "scale_cache must have shape (P, 1)");
  TORCH_CHECK(rope_cache.dim() == 3 &&
                  rope_cache.size(0) == packed_cache.size(0) &&
                  rope_cache.size(1) == 1 &&
                  rope_cache.size(2) == kRopeDim,
              "rope_cache must have shape (P, 1, 64)");
  TORCH_CHECK(fault_status.numel() == 1,
              "fault_status must contain exactly one int32 value");
  TORCH_CHECK(zero_count.numel() == 1,
              "zero_count must contain exactly one int64 value");
  TORCH_CHECK(packed_cache.size(0) > 0, "pool size must be positive");
  TORCH_CHECK(grid > 0.0 && grid < std::numeric_limits<float>::infinity(),
              "grid must be finite and positive");

  check_output_aliases(
      {{&packed_cache, "packed_cache"},
       {&scale_cache, "scale_cache"},
       {&rope_cache, "rope_cache"},
       {&fault_status, "fault_status"},
       {&zero_count, "zero_count"}},
      {{&cache_latent, "cache_latent"},
       {&cache_rope, "cache_rope"},
       {&locations, "locations"},
       {&signs1, "signs1"},
       {&signs2, "signs2"}});
}

void validate_sm100(const torch::Device device) {
  c10::cuda::CUDAGuard device_guard(device);
  const cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(device.index());
  TORCH_CHECK(properties->major == 10 && properties->minor == 0,
              "native N10 frontend requires SM100; got compute capability ",
              properties->major, ".", properties->minor);
}

template <bool kStrict, typename index_t>
void launch_cache_writer(
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    const torch::Tensor& packed_cache, const torch::Tensor& scale_cache,
    const torch::Tensor& rope_cache, const torch::Tensor& fault_status,
    const torch::Tensor& zero_count, const float grid,
    const cudaStream_t stream) {
  const int64_t tokens = cache_latent.size(0);
  if (tokens == 0) {
    return;
  }
  tq_mla_cache_writer_kernel<kStrict, index_t>
      <<<dim3(tokens), dim3(32), 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(cache_latent.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(cache_rope.data_ptr()),
          locations.data_ptr<index_t>(), signs1.data_ptr<float>(),
          signs2.data_ptr<float>(), packed_cache.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(scale_cache.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(rope_cache.data_ptr()),
          fault_status.data_ptr<int32_t>(), zero_count.data_ptr<int64_t>(),
          packed_cache.size(0), grid,
          cache_latent.stride(0), cache_latent.stride(2),
          cache_rope.stride(0), cache_rope.stride(2),
          packed_cache.stride(0), packed_cache.stride(2),
          scale_cache.stride(0), rope_cache.stride(0),
          rope_cache.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kRotationFused, bool kStrict, typename index_t, bool kApplyRope>
void launch_frontend(
    const torch::Tensor& query_latent, const torch::Tensor& query_rope,
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    const torch::Tensor& query_latent_out,
    const torch::Tensor& query_rope_out, const torch::Tensor& packed_cache,
    const torch::Tensor& scale_cache, const torch::Tensor& rope_cache,
    const torch::Tensor& fault_status, const torch::Tensor& zero_count,
    const float grid,
    const torch::Tensor* cos_sin_cache, const torch::Tensor* positions,
    const cudaStream_t stream) {
  const int64_t tokens = query_latent.size(0);
  if (tokens == 0) {
    return;
  }
  const float* cos_sin_ptr = nullptr;
  const int64_t* positions_ptr = nullptr;
  int64_t cos_sin_stride_t = 0;
  int64_t max_positions = 0;
  if constexpr (kApplyRope) {
    cos_sin_ptr = cos_sin_cache->data_ptr<float>();
    positions_ptr = positions->data_ptr<int64_t>();
    cos_sin_stride_t = cos_sin_cache->stride(0);
    max_positions = cos_sin_cache->size(0);
  }
  if constexpr (kRotationFused) {
    tq_mla_frontend_fused_kernel<kStrict, index_t, kApplyRope>
        <<<dim3(tokens), dim3((kQueryHeads + 1) * 32), 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query_latent.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(query_rope.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(cache_latent.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(cache_rope.data_ptr()),
            locations.data_ptr<index_t>(), signs1.data_ptr<float>(),
            signs2.data_ptr<float>(),
            reinterpret_cast<uint8_t*>(query_latent_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(query_rope_out.data_ptr()),
            packed_cache.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(scale_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(rope_cache.data_ptr()),
            fault_status.data_ptr<int32_t>(), zero_count.data_ptr<int64_t>(),
            packed_cache.size(0), grid,
            cache_latent.stride(0), cache_latent.stride(2),
            cache_rope.stride(0), cache_rope.stride(2),
            packed_cache.stride(0), packed_cache.stride(2),
            scale_cache.stride(0), rope_cache.stride(0),
            rope_cache.stride(2), cos_sin_ptr, positions_ptr,
            cos_sin_stride_t, max_positions);
  } else {
    tq_mla_frontend_unfused_kernel<kStrict, index_t, kApplyRope>
        <<<dim3(tokens), dim3((kQueryHeads + 1) * 32), 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query_latent.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(query_rope.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(cache_latent.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(cache_rope.data_ptr()),
            locations.data_ptr<index_t>(), signs1.data_ptr<float>(),
            signs2.data_ptr<float>(),
            reinterpret_cast<uint8_t*>(query_latent_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(query_rope_out.data_ptr()),
            packed_cache.data_ptr<uint8_t>(),
            reinterpret_cast<__nv_bfloat16*>(scale_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(rope_cache.data_ptr()),
            fault_status.data_ptr<int32_t>(), zero_count.data_ptr<int64_t>(),
            packed_cache.size(0), grid,
            cache_latent.stride(0), cache_latent.stride(2),
            cache_rope.stride(0), cache_rope.stride(2),
            packed_cache.stride(0), packed_cache.stride(2),
            scale_cache.stride(0), rope_cache.stride(0),
            rope_cache.stride(2), cos_sin_ptr, positions_ptr,
            cos_sin_stride_t, max_positions);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename index_t, bool kApplyRope>
void dispatch_frontend(
    const torch::Tensor& query_latent, const torch::Tensor& query_rope,
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    const torch::Tensor& query_latent_out,
    const torch::Tensor& query_rope_out, const torch::Tensor& packed_cache,
    const torch::Tensor& scale_cache, const torch::Tensor& rope_cache,
    const torch::Tensor& fault_status, const torch::Tensor& zero_count,
    const float grid, const bool rotation_fused, const bool strict,
    const torch::Tensor* cos_sin_cache, const torch::Tensor* positions,
    const cudaStream_t stream) {
#define LAUNCH_FRONTEND(ROTATION, STRICT)                                \
  launch_frontend<ROTATION, STRICT, index_t, kApplyRope>(                 \
      query_latent, query_rope, cache_latent, cache_rope, locations,     \
      signs1, signs2, query_latent_out, query_rope_out, packed_cache,    \
      scale_cache,                                                       \
      rope_cache, fault_status, zero_count, grid, cos_sin_cache,         \
      positions, stream)

#define DISPATCH_STRICT(ROTATION)       \
  if (strict) {                         \
    LAUNCH_FRONTEND(ROTATION, true);    \
  } else {                              \
    LAUNCH_FRONTEND(ROTATION, false);   \
  }

  if (rotation_fused) {
    DISPATCH_STRICT(true);
  } else {
    DISPATCH_STRICT(false);
  }

#undef DISPATCH_STRICT
#undef LAUNCH_FRONTEND
}

}  // namespace

void tq_mla_n10_native_cache_writer_out(
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    torch::Tensor packed_cache, torch::Tensor scale_cache,
    torch::Tensor rope_cache, torch::Tensor fault_status,
    torch::Tensor zero_count, const double grid, const bool strict) {
  validate_fixed_n10_cache_contract(
      cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
      scale_cache, rope_cache, fault_status, zero_count, grid);
  const torch::Device device = cache_latent.device();
  validate_sm100(device);
  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device.index()).stream();
  const float grid_f = static_cast<float>(grid);

  if (locations.scalar_type() == at::kInt) {
    if (strict) {
      launch_cache_writer<true, int32_t>(
          cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
          scale_cache, rope_cache,
          fault_status, zero_count, grid_f, stream);
    } else {
      launch_cache_writer<false, int32_t>(
          cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
          scale_cache, rope_cache,
          fault_status, zero_count, grid_f, stream);
    }
  } else {
    if (strict) {
      launch_cache_writer<true, int64_t>(
          cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
          scale_cache, rope_cache,
          fault_status, zero_count, grid_f, stream);
    } else {
      launch_cache_writer<false, int64_t>(
          cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
          scale_cache, rope_cache,
          fault_status, zero_count, grid_f, stream);
    }
  }
}

void tq_mla_n10_native_frontend_out(
    const torch::Tensor& query_latent, const torch::Tensor& query_rope,
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    torch::Tensor query_latent_out, torch::Tensor query_rope_out,
    torch::Tensor packed_cache, torch::Tensor scale_cache,
    torch::Tensor rope_cache, torch::Tensor fault_status,
    torch::Tensor zero_count, const double grid,
    const bool rotation_fused, const bool strict) {
  validate_fixed_n10_cache_contract(
      cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
      scale_cache, rope_cache, fault_status, zero_count, grid);
  check_cuda_contiguous(query_latent, "query_latent");
  check_cuda_contiguous(query_rope, "query_rope");
  check_cuda_contiguous(query_latent_out, "query_latent_out");
  check_cuda_contiguous(query_rope_out, "query_rope_out");

  const torch::Device device = cache_latent.device();
  check_same_device(query_latent, device, "query_latent");
  check_same_device(query_rope, device, "query_rope");
  check_same_device(query_latent_out, device, "query_latent_out");
  check_same_device(query_rope_out, device, "query_rope_out");
  TORCH_CHECK(query_latent.scalar_type() == at::kBFloat16,
              "query_latent must be bfloat16");
  TORCH_CHECK(query_rope.scalar_type() == at::kBFloat16,
              "query_rope must be bfloat16");
  TORCH_CHECK(
      query_latent_out.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "query_latent_out must be float8_e4m3fn");
  TORCH_CHECK(query_rope_out.scalar_type() == at::kBFloat16,
              "query_rope_out must be bfloat16");

  const int64_t tokens = cache_latent.size(0);
  TORCH_CHECK(query_latent.dim() == 3 &&
                  query_latent.size(0) == tokens &&
                  query_latent.size(1) == kQueryHeads &&
                  query_latent.size(2) == kLatentDim,
              "query_latent must have shape (T, 8, 512)");
  TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == tokens &&
                  query_rope.size(1) == kQueryHeads &&
                  query_rope.size(2) == kRopeDim,
              "query_rope must have shape (T, 8, 64)");
  TORCH_CHECK(query_latent_out.dim() == 3 &&
                  query_latent_out.size(0) == tokens &&
                  query_latent_out.size(1) == kQueryHeads &&
                  query_latent_out.size(2) == kLatentDim,
              "query_latent_out must have shape (T, 8, 512)");
  TORCH_CHECK(query_rope_out.dim() == 3 &&
                  query_rope_out.size(0) == tokens &&
                  query_rope_out.size(1) == kQueryHeads &&
                  query_rope_out.size(2) == kRopeDim,
              "query_rope_out must have shape (T, 8, 64)");

  check_output_aliases(
      {{&query_latent_out, "query_latent_out"},
       {&query_rope_out, "query_rope_out"},
       {&packed_cache, "packed_cache"},
       {&scale_cache, "scale_cache"},
       {&rope_cache, "rope_cache"},
       {&fault_status, "fault_status"},
       {&zero_count, "zero_count"}},
      {{&query_latent, "query_latent"},
       {&query_rope, "query_rope"},
       {&cache_latent, "cache_latent"},
       {&cache_rope, "cache_rope"},
       {&locations, "locations"},
       {&signs1, "signs1"},
       {&signs2, "signs2"}});

  validate_sm100(device);
  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device.index()).stream();
  const float grid_f = static_cast<float>(grid);
  if (locations.scalar_type() == at::kInt) {
    dispatch_frontend<int32_t, false>(
        query_latent, query_rope, cache_latent, cache_rope, locations, signs1,
        signs2, query_latent_out, query_rope_out, packed_cache, scale_cache,
        rope_cache, fault_status,
        zero_count, grid_f, rotation_fused, strict, nullptr, nullptr, stream);
  } else {
    dispatch_frontend<int64_t, false>(
        query_latent, query_rope, cache_latent, cache_rope, locations, signs1,
        signs2, query_latent_out, query_rope_out, packed_cache, scale_cache,
        rope_cache, fault_status,
        zero_count, grid_f, rotation_fused, strict, nullptr, nullptr, stream);
  }
}

void tq_mla_n10_native_frontend_rope_out(
    const torch::Tensor& query_latent, const torch::Tensor& query_rope,
    const torch::Tensor& cache_latent, const torch::Tensor& cache_rope,
    const torch::Tensor& cos_sin_cache, const torch::Tensor& positions,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2,
    torch::Tensor query_latent_out, torch::Tensor query_rope_out,
    torch::Tensor packed_cache, torch::Tensor scale_cache,
    torch::Tensor rope_cache, torch::Tensor fault_status,
    torch::Tensor zero_count, const double grid,
    const bool rotation_fused, const bool strict) {
  validate_fixed_n10_cache_contract(
      cache_latent, cache_rope, locations, signs1, signs2, packed_cache,
      scale_cache, rope_cache, fault_status, zero_count, grid);
  check_cuda_contiguous(query_latent, "query_latent");
  check_cuda_contiguous(query_rope, "query_rope");
  check_cuda_contiguous(cos_sin_cache, "cos_sin_cache");
  check_cuda_contiguous(positions, "positions");
  check_cuda_contiguous(query_latent_out, "query_latent_out");
  check_cuda_contiguous(query_rope_out, "query_rope_out");

  const torch::Device device = cache_latent.device();
  check_same_device(query_latent, device, "query_latent");
  check_same_device(query_rope, device, "query_rope");
  check_same_device(cos_sin_cache, device, "cos_sin_cache");
  check_same_device(positions, device, "positions");
  check_same_device(query_latent_out, device, "query_latent_out");
  check_same_device(query_rope_out, device, "query_rope_out");
  TORCH_CHECK(query_latent.scalar_type() == at::kBFloat16,
              "query_latent must be bfloat16");
  TORCH_CHECK(query_rope.scalar_type() == at::kBFloat16,
              "query_rope must be bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::kFloat,
              "cos_sin_cache must be float32");
  TORCH_CHECK(positions.scalar_type() == at::kLong,
              "positions must be int64");
  TORCH_CHECK(
      query_latent_out.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "query_latent_out must be float8_e4m3fn");
  TORCH_CHECK(query_rope_out.scalar_type() == at::kBFloat16,
              "query_rope_out must be bfloat16");

  const int64_t tokens = cache_latent.size(0);
  TORCH_CHECK(query_latent.dim() == 3 &&
                  query_latent.size(0) == tokens &&
                  query_latent.size(1) == kQueryHeads &&
                  query_latent.size(2) == kLatentDim,
              "query_latent must have shape (T, 8, 512)");
  TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == tokens &&
                  query_rope.size(1) == kQueryHeads &&
                  query_rope.size(2) == kRopeDim,
              "query_rope must have shape (T, 8, 64)");
  TORCH_CHECK(cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(0) > 0 &&
                  cos_sin_cache.size(1) == kRopeDim,
              "cos_sin_cache must have shape (max_positions, 64)");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == tokens,
              "positions must have shape (T,)");
  TORCH_CHECK(query_latent_out.dim() == 3 &&
                  query_latent_out.size(0) == tokens &&
                  query_latent_out.size(1) == kQueryHeads &&
                  query_latent_out.size(2) == kLatentDim,
              "query_latent_out must have shape (T, 8, 512)");
  TORCH_CHECK(query_rope_out.dim() == 3 &&
                  query_rope_out.size(0) == tokens &&
                  query_rope_out.size(1) == kQueryHeads &&
                  query_rope_out.size(2) == kRopeDim,
              "query_rope_out must have shape (T, 8, 64)");

  check_output_aliases(
      {{&query_latent_out, "query_latent_out"},
       {&query_rope_out, "query_rope_out"},
       {&packed_cache, "packed_cache"},
       {&scale_cache, "scale_cache"},
       {&rope_cache, "rope_cache"},
       {&fault_status, "fault_status"},
       {&zero_count, "zero_count"}},
      {{&query_latent, "query_latent"},
       {&query_rope, "query_rope"},
       {&cache_latent, "cache_latent"},
       {&cache_rope, "cache_rope"},
       {&cos_sin_cache, "cos_sin_cache"},
       {&positions, "positions"},
       {&locations, "locations"},
       {&signs1, "signs1"},
       {&signs2, "signs2"}});

  validate_sm100(device);
  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(device.index()).stream();
  const float grid_f = static_cast<float>(grid);
  if (locations.scalar_type() == at::kInt) {
    dispatch_frontend<int32_t, true>(
        query_latent, query_rope, cache_latent, cache_rope, locations, signs1,
        signs2, query_latent_out, query_rope_out, packed_cache, scale_cache,
        rope_cache, fault_status,
        zero_count, grid_f, rotation_fused, strict, &cos_sin_cache, &positions,
        stream);
  } else {
    dispatch_frontend<int64_t, true>(
        query_latent, query_rope, cache_latent, cache_rope, locations, signs1,
        signs2, query_latent_out, query_rope_out, packed_cache, scale_cache,
        rope_cache, fault_status,
        zero_count, grid_f, rotation_fused, strict, &cos_sin_cache, &positions,
        stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("tq_mla_n10_native_cache_writer_out",
             &tq_mla_n10_native_cache_writer_out,
             "SM100 N10 E2M1 MLA cache-only writer");
  module.def("tq_mla_n10_native_frontend_out",
             &tq_mla_n10_native_frontend_out,
             "SM100 N10 E2M1 MLA query and writer front end");
  module.def("tq_mla_n10_native_frontend_rope_out",
             &tq_mla_n10_native_frontend_rope_out,
             "SM100 N10 E2M1 MLA fused-RoPE query and writer front end");
}
