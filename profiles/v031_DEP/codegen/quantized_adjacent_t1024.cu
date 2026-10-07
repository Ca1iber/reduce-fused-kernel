#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_quad_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf);
extern "C" __global__ void __launch_bounds__(1024, 1) tiny_quad_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf) {
  float scale = 0x0p+0f/*0.000000e+00*/;
  float loaded[1];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  int pos = 0;
  float scale0 = 0x0p+0f/*0.000000e+00*/;
  float scale1 = 0x0p+0f/*0.000000e+00*/;
  int pos1 = 0;
  float value1 = 0x0p+0f/*0.000000e+00*/;
  float reduced[1];
  fp8_e4_t encoded_values[1];
  scale = 0x0p+0f/*0.000000e+00*/;
  loaded[0] = 0x0p+0f/*0.000000e+00*/;
  sf_var = sf[0];
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
    float value = (reduced[0] * sf_var);
    uint raw = (*(uint *)(&(value)));
    uint magnitude = ((*(uint *)(&(value))) & (uint)2147483647);
    uint sign = (((*(uint *)(&(value))) >> (uint)24) & (uint)128);
    uint normal = ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960);
    uint v_ = (*(uint *)(&(value))) & (uint)2147483647;
    float v__1 = (*(float *)(&(v_))) + 0x1p+14f/*1.638400e+04*/;
    uint denormal = ((*(uint *)(&(v__1))) - (uint)1182793728);
    uint v__2 = (*(uint *)(&(value))) & (uint)2147483647;
    float v__3 = (*(float *)(&(v__2))) + 0x1p+14f/*1.638400e+04*/;
    uint encoded = (((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__3))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960)));
    uint v__4 = (*(uint *)(&(value))) & (uint)2147483647;
    float v__5 = (*(float *)(&(v__4))) + 0x1p+14f/*1.638400e+04*/;
    uint final_encoded = (((uint)2139095040 < ((*(uint *)(&(value))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__5))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(value))) >> (uint)24) & (uint)128)));
    uint v__6 = (*(uint *)(&(value))) & (uint)2147483647;
    float v__7 = (*(float *)(&(v__6))) + 0x1p+14f/*1.638400e+04*/;
    uchar v__8 = (uchar)(((uint)2139095040 < ((*(uint *)(&(value))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__7))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(value))) >> (uint)24) & (uint)128)));
    encoded_values[0] = (*(fp8_e4_t *)(&(v__8)));
    out[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2))] = encoded_values[0];
  }
}

