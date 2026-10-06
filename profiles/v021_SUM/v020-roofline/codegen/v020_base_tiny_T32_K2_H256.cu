#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(256, 1) reduce_fused_kernel_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens) {
  float reduced_fragment[1];
  float topk_weights_local[2];
  int topk_to_pos_local[2];
  float s = 0x0p+0f/*0.000000e+00*/;
  reduced_fragment[0] = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(topk_weights_local + 0) = *(float2*)(topk_weights + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  *(int2*)(topk_to_pos_local + 0) = *(int2*)(token_topk_to_pos + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    int pos = topk_to_pos_local[k];
    if (0 <= pos) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      reduced_fragment[0] = (reduced_fragment[0] + (((float)x[((((int64_t)pos) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]) * s));
    }
  }
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))] = ((half_t)reduced_fragment[0]);
}

