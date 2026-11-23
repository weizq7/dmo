// Full RMSNorm for the pinned PyTorch-2.8 CUDA single-row reduction path.
// Preserve Reduce.cuh's four accumulators, block-x reduction and ascending
// warp-shuffle order. Square/reduce/tail execute in one CTA, without storing
// the squared vector or mean to global memory. Supported widths: 32..4096,
// multiples of 32; these use one CTA in the original FP32 mean implementation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

template<class T>
__global__ void full_rms_norm(const T* x, const float* weight, float* out,
                              int width, float eps) {
  extern __shared__ float partials[];
  int tid = threadIdx.x;
  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  if (width > 128) {
    for (int vector = tid; vector * 4 + 3 < width; vector += blockDim.x) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float value = float(x[4 * vector + j]);
        accum[j] = __fadd_rn(accum[j], __fmul_rn(value, value));
      }
    }
  } else {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int i = tid + j * blockDim.x;
      if (i < width) {
        float value = float(x[i]);
        accum[j] = __fmul_rn(value, value);
      }
    }
  }
  float sum = accum[0];
  #pragma unroll
  for (int j = 1; j < 4; ++j) sum = __fadd_rn(sum, accum[j]);
  partials[tid] = sum;
  for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    __syncthreads();
    if (tid < stride) {
      sum = __fadd_rn(sum, partials[tid + stride]);
      partials[tid] = sum;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
  if (tid == 0) {
    float mean = __fmul_rn(sum, __fdiv_rn(1.0f, float(width)));
    partials[0] = ::rsqrt(__fadd_rn(mean, eps));
  }
  __syncthreads();
  float scale = partials[0];
  for (int i = tid; i < width; i += blockDim.x) {
    float normalized = __fmul_rn(float(x[i]), scale);
    out[i] = __fmul_rn(normalized, weight[i]);
  }
}

extern "C" int launch_full_rmsnorm(uint64_t input, uint64_t weight,
    uint64_t output, int width, int bf16, float eps, uint64_t stream) {
  if (width < 32 || width > 4096 || width % 32) return int(cudaErrorInvalidValue);
  int dim = width > 128 ? width / 4 : width;
  int threads = 1;
  while (threads * 2 <= dim && threads < 512) threads *= 2;
  auto s = (cudaStream_t)stream;
  if (bf16) full_rms_norm<<<1, threads, threads * sizeof(float), s>>>(
      (const __nv_bfloat16*)input, (const float*)weight, (float*)output, width, eps);
  else full_rms_norm<<<1, threads, threads * sizeof(float), s>>>(
      (const float*)input, (const float*)weight, (float*)output, width, eps);
  return int(cudaGetLastError());
}
