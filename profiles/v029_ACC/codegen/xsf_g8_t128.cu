#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) tiny_group_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens) {
  float reduced[16];
  int positions[2];
  float weights[2];
  float scale = 0x0p+0f/*0.000000e+00*/;
  half_t out_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(reduced + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  int broadcast_var_1 = 0;
  int2 condval;
  if ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)) {
    condval = *(int2*)(token_topk_to_pos + ((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2)));
  } else {
    condval = make_int2(broadcast_var_1, broadcast_var_1);
  }
  *(int2*)(positions + 0) = condval;
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  float2 condval_1;
  if ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)) {
    condval_1 = *(float2*)(topk_weights + ((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2)));
  } else {
    condval_1 = make_float2(broadcast_var_2, broadcast_var_2);
  }
  *(float2*)(weights + 0) = condval_1;
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens) {
        int pos = positions[k];
        if (0 <= pos) {
          scale = 0x1p+0f/*1.000000e+00*/;
          scale = weights[k];
          scale = (scale * x_sf[((int64_t)pos)]);
          reduced[i_1] = (reduced[i_1] + (((float)x[(((((int64_t)pos) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)16)) + ((int64_t)i_1))]) * scale));
        }
      }
    }
  }
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __1;
        float4 v_ = *(float4*)(reduced + ((i_2 * 8) + (vec * 4)));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
        *(uint2*)(out_local_cast + (vec * 4)) = __1;
      }
      *(uint4*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + (((int64_t)i_2) * (int64_t)8))) = *(uint4*)(out_local_cast + 0);
    }
  }
}

