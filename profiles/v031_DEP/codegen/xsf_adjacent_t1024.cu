#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_quad_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf);
extern "C" __global__ void __launch_bounds__(1024, 1) tiny_quad_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf) {
  float scale = 0x0p+0f/*0.000000e+00*/;
  float loaded[1];
  int pos = 0;
  float scale0 = 0x0p+0f/*0.000000e+00*/;
  float scale1 = 0x0p+0f/*0.000000e+00*/;
  int pos1 = 0;
  float value1 = 0x0p+0f/*0.000000e+00*/;
  float reduced[1];
  half_t encoded_values[1];
  scale = 0x0p+0f/*0.000000e+00*/;
  loaded[0] = 0x0p+0f/*0.000000e+00*/;
  pos = token_topk_to_pos[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
  if (0 <= pos) {
    if (2 <= (((int)threadIdx.x) & 3)) {
      scale = 0x1p+0f/*1.000000e+00*/;
      scale = topk_weights[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
      float condval;
      if ((0 <= pos)) {
        condval = x_sf[pos];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      scale = (scale * condval);
    }
    if ((((int)threadIdx.x) & 3) < 2) {
      half_t condval_1;
      if ((0 <= pos)) {
        condval_1 = x[((((int64_t)pos) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2))];
      } else {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      loaded[0] = ((float)condval_1);
    }
  }
  scale0 = __shfl_sync((uint64_t)18446744073709551615, scale, ((((((int)threadIdx.x) & 63) >> 2) * 4) + 2), 64);
  scale1 = __shfl_sync((uint64_t)18446744073709551615, scale, ((((((int)threadIdx.x) & 63) >> 2) * 4) + 3), 64);
  pos1 = __shfl_sync((uint64_t)18446744073709551615, pos, ((((((int)threadIdx.x) & 63) >> 2) * 4) + 1), 64);
  value1 = __shfl_sync((uint64_t)18446744073709551615, loaded[0], ((((((int)threadIdx.x) & 63) >> 2) * 4) + 1), 64);
  if ((((int)threadIdx.x) % 4) == 0) {
    reduced[0] = 0x0p+0f/*0.000000e+00*/;
    if (0 <= pos) {
      reduced[0] = (reduced[0] + (loaded[0] * scale0));
    }
    if (0 <= pos1) {
      reduced[0] = (reduced[0] + (value1 * scale1));
    }
    encoded_values[0] = ((half_t)reduced[0]);
    out[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2))] = encoded_values[0];
  }
}

