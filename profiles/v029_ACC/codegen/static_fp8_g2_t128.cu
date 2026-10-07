#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_group_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) tiny_group_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x) {
  float reduced[4];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  int positions[2];
  float weights[2];
  float scale = 0x0p+0f/*0.000000e+00*/;
  fp8_e4_t encoded_values[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(reduced + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  sf_var = sf[0];
  *(int2*)(positions + 0) = *(int2*)(token_topk_to_pos + ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) >> 6) * 2)));
  *(float2*)(weights + 0) = *(float2*)(topk_weights + ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) >> 6) * 2)));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int pos = positions[k];
      if (0 <= pos) {
        scale = 0x1p+0f/*1.000000e+00*/;
        scale = weights[k];
        reduced[i] = (reduced[i] + (((float)x[(((((int64_t)pos) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)4)) + ((int64_t)i))]) * scale));
      }
    }
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float value = (reduced[i_1] * sf_var);
    uint raw = (*(uint *)(&(value)));
    uint magnitude = ((*(uint *)(&(value))) & (uint)2147483647);
    uint sign = (((*(uint *)(&(value))) >> (uint)24) & (uint)128);
    uint normal = ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960);
    uint v_ = (*(uint *)(&(value))) & (uint)2147483647;
    float v__1 = (*(float *)(&(v_))) + 0x1p+14f/*1.638400e+04*/;
    uint denormal = ((*(uint *)(&(v__1))) - (uint)1182793728);
    uint encoded = (((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__1))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960)));
    uint final_encoded = (((uint)2139095040 < ((*(uint *)(&(value))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__1))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(value))) >> (uint)24) & (uint)128)));
    uchar v__2 = (uchar)(((uint)2139095040 < ((*(uint *)(&(value))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__1))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(value))) >> (uint)24) & (uint)128)));
    encoded_values[i_1] = (*(fp8_e4_t *)(&(v__2)));
  }
  *(fp8_e4_4_t*)(out + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4))) = *(fp8_e4_4_t*)(encoded_values + 0);
}

