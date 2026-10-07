#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x) {
  float reduced[1];
  int positions[2];
  float weights[2];
  float scale = 0x0p+0f/*0.000000e+00*/;
  reduced[0] = 0x0p+0f/*0.000000e+00*/;
  *(int2*)(positions + 0) = *(int2*)(token_topk_to_pos + (((int)blockIdx.x) * 2));
  *(float2*)(weights + 0) = *(float2*)(topk_weights + (((int)blockIdx.x) * 2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    int pos = positions[k];
    if (0 <= pos) {
      scale = 0x1p+0f/*1.000000e+00*/;
      scale = weights[k];
      reduced[0] = (reduced[0] + (((float)x[((((int64_t)pos) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]) * scale));
    }
  }
  out[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = ((half_t)reduced[0]);
}

