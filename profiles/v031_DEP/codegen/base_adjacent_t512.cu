#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_quad_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(512, 1) tiny_quad_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x) {
  float scale = 0x0p+0f/*0.000000e+00*/;
  float loaded[2];
  int pos = 0;
  half_t x_local_cast[2];
  float scale0 = 0x0p+0f/*0.000000e+00*/;
  float scale1 = 0x0p+0f/*0.000000e+00*/;
  int pos1 = 0;
  half_t encoded_values[2];
  float value1 = 0x0p+0f/*0.000000e+00*/;
  float reduced[2];
  scale = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    loaded[j] = 0x0p+0f/*0.000000e+00*/;
  }
  pos = token_topk_to_pos[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
  if (0 <= pos) {
    if (2 <= (((int)threadIdx.x) & 3)) {
      scale = 0x1p+0f/*1.000000e+00*/;
      scale = topk_weights[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
    }
    if ((((int)threadIdx.x) & 3) < 2) {
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      uint1 condval;
      if ((0 <= pos)) {
        condval = *(uint1*)(x + ((((int64_t)pos) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)2)));
      } else {
        condval = make_uint1(__pack_half2(broadcast_var, broadcast_var));
      }
      *(uint1*)(x_local_cast + 0) = condval;
      float2 __1;
      uint1 v_ = *(uint1*)(x_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(loaded + 0) = __1;
    }
  }
  scale0 = __shfl_sync((uint64_t)18446744073709551615, scale, ((((((int)threadIdx.x) & 63) >> 2) * 4) + 2), 64);
  scale1 = __shfl_sync((uint64_t)18446744073709551615, scale, ((((((int)threadIdx.x) & 63) >> 2) * 4) + 3), 64);
  pos1 = __shfl_sync((uint64_t)18446744073709551615, pos, ((((((int)threadIdx.x) & 63) >> 2) * 4) + 1), 64);
  #pragma unroll
  for (int j_1 = 0; j_1 < 2; ++j_1) {
    value1 = __shfl_sync((uint64_t)18446744073709551615, loaded[j_1], ((((((int)threadIdx.x) & 63) >> 2) * 4) + 1), 64);
    if ((((int)threadIdx.x) % 4) == 0) {
      reduced[j_1] = 0x0p+0f/*0.000000e+00*/;
      if (0 <= pos) {
        reduced[j_1] = (reduced[j_1] + (loaded[j_1] * scale0));
      }
      if (0 <= pos1) {
        reduced[j_1] = (reduced[j_1] + (value1 * scale1));
      }
      encoded_values[j_1] = ((half_t)reduced[j_1]);
    }
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    *(uint1*)(out + ((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 2) * 2))) = *(uint1*)(encoded_values + 0);
  }
}

