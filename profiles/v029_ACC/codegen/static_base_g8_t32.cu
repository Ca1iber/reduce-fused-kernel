#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(32, 1) tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x) {
  float reduced[64];
  int positions[2];
  float weights[2];
  float scale = 0x0p+0f/*0.000000e+00*/;
  half_t out_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(reduced + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(int2*)(positions + 0) = *(int2*)(token_topk_to_pos + ((((int)blockIdx.x) * 16) + ((((int)threadIdx.x) >> 2) * 2)));
  *(float2*)(weights + 0) = *(float2*)(topk_weights + ((((int)blockIdx.x) * 16) + ((((int)threadIdx.x) >> 2) * 2)));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      int pos = positions[k];
      if (0 <= pos) {
        scale = 0x1p+0f/*1.000000e+00*/;
        scale = weights[k];
        reduced[i_1] = (reduced[i_1] + (((float)x[(((((int64_t)pos) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)64)) + ((int64_t)i_1))]) * scale));
      }
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    for (int vec = 0; vec < 2; ++vec) {
      uint2 __1;
      float4 v_ = *(float4*)(reduced + ((i_2 * 8) + (vec * 4)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(out_local_cast + (vec * 4)) = __1;
    }
    *(uint4*)(out + (((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 64)) + (i_2 * 8))) = *(uint4*)(out_local_cast + 0);
  }
}

