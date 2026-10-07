#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_local_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(64, 1) tiny_local_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x) {
  float reduced[4];
  int positions[2];
  float weights[2];
  int pos = 0;
  float scale = 0x0p+0f/*0.000000e+00*/;
  half_t loaded[4];
  half_t encoded_values[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    reduced[j] = 0x0p+0f/*0.000000e+00*/;
  }
  *(int2*)(positions + 0) = *(int2*)(token_topk_to_pos + (((int)blockIdx.x) * 2));
  *(float2*)(weights + 0) = *(float2*)(topk_weights + (((int)blockIdx.x) * 2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    pos = positions[k];
    if (0 <= pos) {
      scale = 0x1p+0f/*1.000000e+00*/;
      scale = weights[k];
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval;
      if ((0 <= pos)) {
        condval = *(uint2*)(x + ((((int64_t)pos) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      } else {
        condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
      }
      *(uint2*)(loaded + 0) = condval;
      #pragma unroll
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        reduced[j_1] = (reduced[j_1] + (((float)loaded[j_1]) * scale));
      }
    }
  }
  #pragma unroll
  for (int j_2 = 0; j_2 < 4; ++j_2) {
    encoded_values[j_2] = ((half_t)reduced[j_2]);
  }
  *(uint2*)(out + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 4))) = *(uint2*)(encoded_values + 0);
}

