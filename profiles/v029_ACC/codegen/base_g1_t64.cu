#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(64, 1) tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens) {
  float reduced[4];
  int positions[2];
  float weights[2];
  float scale = 0x0p+0f/*0.000000e+00*/;
  half_t out_local_cast[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(reduced + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  *(int2*)(positions + 0) = *(int2*)(token_topk_to_pos + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  *(float2*)(weights + 0) = *(float2*)(topk_weights + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int pos = positions[k];
      if (0 <= pos) {
        scale = 0x1p+0f/*1.000000e+00*/;
        scale = weights[k];
        reduced[i] = (reduced[i] + (((float)x[(((((int64_t)pos) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + ((int64_t)i))]) * scale));
      }
    }
  }
  uint2 __1;
  float4 v_ = *(float4*)(reduced + 0);
  ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
  ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
  *(uint2*)(out_local_cast + 0) = __1;
  *(uint2*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(out_local_cast + 0);
}

