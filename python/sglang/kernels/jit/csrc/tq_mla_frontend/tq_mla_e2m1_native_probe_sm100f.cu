#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

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

  // sign==0 maps correction to -1, sign==1 maps it to +1.
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
  return __half22float2(reinterpret_cast<const __half2&>(fp16x2));
}

__global__ void e2m1_native_probe_kernel(
    const float* values, uint8_t* raw_packed, uint8_t* corrected_packed,
    float* decoded, const int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) {
    return;
  }
  const float even = values[pair * 2];
  const float odd = values[pair * 2 + 1];
  const uint8_t raw = native_e2m1x2_rne(even, odd);
  const uint8_t even_code = correct_n10_code(raw & 0x0fu, even);
  const uint8_t odd_code = correct_n10_code(raw >> 4, odd);
  const uint8_t corrected = static_cast<uint8_t>((odd_code << 4) | even_code);
  const float2 reconstructed = native_e2m1x2_to_float2(corrected);
  raw_packed[pair] = raw;
  corrected_packed[pair] = corrected;
  decoded[pair * 2] = reconstructed.x;
  decoded[pair * 2 + 1] = reconstructed.y;
}

std::vector<torch::Tensor> e2m1_native_probe(torch::Tensor values) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(values.scalar_type() == at::kFloat, "values must be float32");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(values.dim() == 2 && values.size(1) == 2,
              "values must have shape (pairs, 2)");
  TORCH_CHECK(values.size(0) <= std::numeric_limits<int64_t>::max(),
              "too many pairs");
  const c10::cuda::CUDAGuard device_guard(values.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 10 && properties->minor == 0,
              "native probe requires GB200 compute capability 10.0");

  const int64_t pairs = values.size(0);
  auto raw = torch::empty({pairs}, values.options().dtype(torch::kUInt8));
  auto corrected = torch::empty({pairs}, values.options().dtype(torch::kUInt8));
  auto decoded = torch::empty_like(values);
  if (pairs == 0) {
    return {raw, corrected, decoded};
  }
  constexpr int kThreads = 256;
  const int64_t blocks = (pairs + kThreads - 1) / kThreads;
  e2m1_native_probe_kernel<<<blocks, kThreads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<float>(), raw.data_ptr<uint8_t>(),
      corrected.data_ptr<uint8_t>(), decoded.data_ptr<float>(), pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {raw, corrected, decoded};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("e2m1_native_probe", &e2m1_native_probe,
             "Native SM100f E2M1 encode/correct/decode probe");
}
