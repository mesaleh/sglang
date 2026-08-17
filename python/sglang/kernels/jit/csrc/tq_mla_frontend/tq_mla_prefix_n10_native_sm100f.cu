#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kPackedDim = kLatentDim / 2;
constexpr int kCombinedDim = kLatentDim + kRopeDim;
constexpr float kInvSqrt512 = 0.044194173824159220275f;
constexpr unsigned int kFaultInvalidLocation = 1u << 0;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float e2m1_to_float(const uint8_t code) {
  const uint8_t magnitude = code & 7u;
  float value;
  switch (magnitude) {
    case 0:
      value = 0.0f;
      break;
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    default:
      value = 6.0f;
      break;
  }
  return (code & 8u) != 0 ? -value : value;
}

__device__ __forceinline__ float2 e2m1x2_to_float2(const uint8_t packed) {
  return make_float2(e2m1_to_float(packed & 0x0fu),
                     e2m1_to_float(packed >> 4));
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

template <typename index_t>
__global__ void validate_locations_kernel(
    const index_t* locations, int32_t* validation_status,
    int32_t* sticky_fault_status, const int64_t count,
    const int64_t pool_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const int64_t location = static_cast<int64_t>(locations[index]);
  if (location < 0 || location >= pool_size) {
    atomicOr(reinterpret_cast<unsigned int*>(validation_status),
             kFaultInvalidLocation);
    atomicOr(reinterpret_cast<unsigned int*>(sticky_fault_status),
             kFaultInvalidLocation);
  }
}

template <typename index_t>
__global__ void reconstruct_rows_kernel(
    const uint8_t* packed_cache, const __nv_bfloat16* scale_cache,
    const __nv_bfloat16* reciprocal_rope_cache, const index_t* locations,
    const float* signs1, const float* signs2, __nv_bfloat16* output,
    const int32_t* validation_status, const int64_t count,
    const int64_t output_capacity,
    const int64_t packed_stride_s, const int64_t packed_stride_h,
    const int64_t scale_stride_s, const int64_t rope_stride_s,
    const int64_t rope_stride_h) {
  const int64_t token = blockIdx.x;
  if (token >= count || *validation_status != 0) {
    return;
  }

  const int lane = threadIdx.x;
  const int64_t slot = static_cast<int64_t>(locations[token]);
  const int64_t packed_base = slot * packed_stride_s;
  const float scale = bf16_to_float(scale_cache[slot * scale_stride_s]);
  __shared__ float rotated[kLatentDim];
  float values[16];

#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const int packed_index = chunk * 64 + lane * 2 + pair;
      const uint8_t packed =
          packed_cache[packed_base + packed_index * packed_stride_h];
      const float2 decoded = e2m1x2_to_float2(packed);
      const int even_index = chunk * 128 + lane * 4 + pair * 2;
      rotated[even_index] = decoded.x * scale;
      rotated[even_index + 1] = decoded.y * scale;
    }
  }
  __syncwarp();

#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int chunk = item / 4;
    const int index = chunk * 128 + lane * 4 + (item % 4);
    values[item] = rotated[index] * signs2[index];
  }

  hadamard_512_fp32_layout(values, lane);
  const int64_t latent_output_base = token * kLatentDim;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int chunk = item / 4;
    const int index = chunk * 128 + lane * 4 + (item % 4);
    const float reconstructed =
        values[item] * kInvSqrt512 * signs1[index];
    output[latent_output_base + index] =
        __float2bfloat16_rn(reconstructed);
  }

  const int rope_index = lane * 2;
  const int64_t rope_base = slot * rope_stride_s;
  const int64_t rope_output_base =
      output_capacity * kLatentDim + token * kRopeDim;
  const float rope0 = bf16_to_float(
      reciprocal_rope_cache[rope_base + rope_index * rope_stride_h]);
  const float rope1 = bf16_to_float(
      reciprocal_rope_cache[rope_base + (rope_index + 1) * rope_stride_h]);
  output[rope_output_base + rope_index] =
      __float2bfloat16_rn(rope0 * scale);
  output[rope_output_base + rope_index + 1] =
      __float2bfloat16_rn(rope1 * scale);
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& tensor,
                       const torch::Device device, const char* name) {
  TORCH_CHECK(tensor.device() == device, name,
              " must be on the same device as packed_cache");
}

uintptr_t tensor_span_end(const torch::Tensor& tensor) {
  if (tensor.numel() == 0) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr());
  }
  const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  return begin + static_cast<uintptr_t>(tensor.numel()) *
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

template <typename index_t>
void launch_prefix_gather(
    const torch::Tensor& packed_cache, const torch::Tensor& scale_cache,
    const torch::Tensor& reciprocal_rope_cache,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2, const torch::Tensor& output,
    const torch::Tensor& validation_status,
    const torch::Tensor& sticky_fault_status) {
  const int64_t count = locations.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(validation_status.data_ptr<int32_t>(), 0,
                                 sizeof(int32_t), stream));
  if (count == 0) {
    return;
  }

  constexpr int kValidationThreads = 256;
  const int64_t validation_blocks =
      (count + kValidationThreads - 1) / kValidationThreads;
  validate_locations_kernel<<<validation_blocks, kValidationThreads, 0,
                              stream>>>(
      locations.data_ptr<index_t>(), validation_status.data_ptr<int32_t>(),
      sticky_fault_status.data_ptr<int32_t>(), count, packed_cache.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  reconstruct_rows_kernel<<<count, 32, 0, stream>>>(
      packed_cache.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(scale_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(
          reciprocal_rope_cache.data_ptr()),
      locations.data_ptr<index_t>(), signs1.data_ptr<float>(),
      signs2.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      validation_status.data_ptr<int32_t>(), count,
      output.numel() / kCombinedDim, packed_cache.stride(0),
      packed_cache.stride(2), scale_cache.stride(0),
      reciprocal_rope_cache.stride(0), reciprocal_rope_cache.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tq_mla_n10_prefix_gather_out(
    const torch::Tensor& packed_cache, const torch::Tensor& scale_cache,
    const torch::Tensor& reciprocal_rope_cache,
    const torch::Tensor& locations, const torch::Tensor& signs1,
    const torch::Tensor& signs2, const torch::Tensor& output,
    const torch::Tensor& validation_status,
    const torch::Tensor& sticky_fault_status) {
  check_cuda_contiguous(packed_cache, "packed_cache");
  check_cuda_contiguous(scale_cache, "scale_cache");
  check_cuda_contiguous(reciprocal_rope_cache, "reciprocal_rope_cache");
  check_cuda_contiguous(locations, "locations");
  check_cuda_contiguous(signs1, "signs1");
  check_cuda_contiguous(signs2, "signs2");
  check_cuda_contiguous(output, "output");
  check_cuda_contiguous(validation_status, "validation_status");
  check_cuda_contiguous(sticky_fault_status, "sticky_fault_status");
  const torch::Device device = packed_cache.device();
  check_same_device(scale_cache, device, "scale_cache");
  check_same_device(reciprocal_rope_cache, device,
                    "reciprocal_rope_cache");
  check_same_device(locations, device, "locations");
  check_same_device(signs1, device, "signs1");
  check_same_device(signs2, device, "signs2");
  check_same_device(output, device, "output");
  check_same_device(validation_status, device, "validation_status");
  check_same_device(sticky_fault_status, device, "sticky_fault_status");

  TORCH_CHECK(packed_cache.scalar_type() == at::kByte,
              "packed_cache must be uint8");
  TORCH_CHECK(scale_cache.scalar_type() == at::kBFloat16,
              "scale_cache must be bfloat16");
  TORCH_CHECK(reciprocal_rope_cache.scalar_type() == at::kBFloat16,
              "reciprocal_rope_cache must be bfloat16");
  TORCH_CHECK(locations.scalar_type() == at::kInt ||
                  locations.scalar_type() == at::kLong,
              "locations must be int32 or int64");
  TORCH_CHECK(signs1.scalar_type() == at::kFloat &&
                  signs2.scalar_type() == at::kFloat,
              "sign vectors must be float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16,
              "output must be bfloat16");
  TORCH_CHECK(validation_status.scalar_type() == at::kInt &&
                  sticky_fault_status.scalar_type() == at::kInt,
              "status words must be int32");

  TORCH_CHECK(packed_cache.dim() == 3 && packed_cache.size(1) == 1 &&
                  packed_cache.size(2) == kPackedDim,
              "packed_cache must have shape (P, 1, 256)");
  TORCH_CHECK(scale_cache.dim() == 2 && scale_cache.size(0) == packed_cache.size(0) &&
                  scale_cache.size(1) == 1,
              "scale_cache must have shape (P, 1)");
  TORCH_CHECK(reciprocal_rope_cache.dim() == 3 &&
                  reciprocal_rope_cache.size(0) == packed_cache.size(0) &&
                  reciprocal_rope_cache.size(1) == 1 &&
                  reciprocal_rope_cache.size(2) == kRopeDim,
              "reciprocal_rope_cache must have shape (P, 1, 64)");
  TORCH_CHECK(locations.dim() == 1, "locations must be one-dimensional");
  TORCH_CHECK(signs1.dim() == 1 && signs1.size(0) == kLatentDim &&
                  signs2.dim() == 1 && signs2.size(0) == kLatentDim,
              "sign vectors must each have shape (512,)");
  TORCH_CHECK(output.dim() == 1 && output.numel() % kCombinedDim == 0,
              "output must be a flat planar workspace with a multiple of "
              "576 BF16 elements");
  TORCH_CHECK(output.numel() / kCombinedDim >= locations.numel(),
              "output capacity is smaller than the requested row count");
  TORCH_CHECK(validation_status.numel() == 1 &&
                  sticky_fault_status.numel() == 1,
              "status tensors must each contain one element");
  TORCH_CHECK(packed_cache.size(0) > 0, "pool size must be positive");
  TORCH_CHECK(!tensors_overlap(output, packed_cache),
              "output must not overlap packed_cache");
  TORCH_CHECK(!tensors_overlap(output, scale_cache),
              "output must not overlap scale_cache");
  TORCH_CHECK(!tensors_overlap(output, reciprocal_rope_cache),
              "output must not overlap reciprocal_rope_cache");
  TORCH_CHECK(!tensors_overlap(output, locations),
              "output must not overlap locations");
  TORCH_CHECK(!tensors_overlap(output, signs1) &&
                  !tensors_overlap(output, signs2),
              "output must not overlap sign vectors");
  TORCH_CHECK(!tensors_overlap(output, validation_status) &&
                  !tensors_overlap(output, sticky_fault_status),
              "output must not overlap status tensors");
  TORCH_CHECK(!tensors_overlap(validation_status, sticky_fault_status),
              "validation_status must not overlap sticky_fault_status");

  const c10::cuda::CUDAGuard device_guard(packed_cache.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 10 && properties->minor == 0,
              "N10 prefix gather requires GB200 compute capability 10.0");
  TORCH_CHECK(locations.numel() <= std::numeric_limits<int>::max(),
              "too many prefix locations");

  if (locations.scalar_type() == at::kInt) {
    launch_prefix_gather<int32_t>(
        packed_cache, scale_cache, reciprocal_rope_cache, locations, signs1,
        signs2, output, validation_status, sticky_fault_status);
  } else {
    launch_prefix_gather<int64_t>(
        packed_cache, scale_cache, reciprocal_rope_cache, locations, signs1,
        signs2, output, validation_status, sticky_fault_status);
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("tq_mla_n10_prefix_gather_out",
             &tq_mla_n10_prefix_gather_out,
             "Bounded native N10 MLA prefix gather into caller-owned output");
}
