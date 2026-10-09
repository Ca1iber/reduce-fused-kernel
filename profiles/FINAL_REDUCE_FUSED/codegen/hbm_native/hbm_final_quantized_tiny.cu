#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void hbm_final_quantized_tiny_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf);
extern "C" __global__ void __launch_bounds__(256, 1) hbm_final_quantized_tiny_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf) {
  float reduced[1];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  int positions[2];
  float weights[2];
  int pos = 0;
  float scale = 0x0p+0f/*0.000000e+00*/;
  half_t loaded[1];
  fp8_e4_t encoded_values[1];
  reduced[0] = 0x0p+0f/*0.000000e+00*/;
  sf_var = sf[0];
  *(int2*)(positions + 0) = *(int2*)(token_topk_to_pos + (((int)blockIdx.x) * 2));
  *(float2*)(weights + 0) = *(float2*)(topk_weights + (((int)blockIdx.x) * 2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    pos = positions[k];
    if (0 <= pos) {
      scale = 0x1p+0f/*1.000000e+00*/;
      scale = weights[k];
      float condval;
      if ((0 <= pos)) {
        condval = x_sf[pos];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      scale = (scale * condval);
      half_t condval_1;
      if ((0 <= pos)) {
        condval_1 = x[((((int64_t)pos) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      loaded[0] = condval_1;
      reduced[0] = (reduced[0] + (((float)loaded[0]) * scale));
    }
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
  out[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = encoded_values[0];
}

