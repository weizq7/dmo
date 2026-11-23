// Exact RoPE arithmetic in one kernel per Q/K.
// Preserve the two FP32 products and their rounded FP32 addition; no FMA.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

template<class T>
__global__ void exact_rope(const T *x, const float *cos, const float *sin,
                           float *out, int count, int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int coordinate = i % head_dim;
  int half = head_dim / 2;
  int mate = coordinate < half ? i + half : i - half;
  float rotated = coordinate < half ? -float(x[mate]) : float(x[mate]);
  float left = __fmul_rn(float(x[i]), cos[coordinate % half]);
  float right = __fmul_rn(rotated, sin[coordinate % half]);
  out[i] = __fadd_rn(left, right);
}

extern "C" int launch_exact_rope(uint64_t x, uint64_t cos, uint64_t sin,
                                 uint64_t out, int count, int head_dim,
                                 int bf16, uint64_t stream) {
  if (bf16) exact_rope<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)x, (const float *)cos, (const float *)sin,
      (float *)out, count, head_dim);
  else exact_rope<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const float *)x, (const float *)cos, (const float *)sin,
      (float *)out, count, head_dim);
  return int(cudaGetLastError());
}
