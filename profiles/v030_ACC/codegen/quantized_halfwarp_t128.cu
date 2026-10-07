#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_pair_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf);
extern "C" __global__ void __launch_bounds__(128, 1) tiny_pair_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf) {
  float scale = 0x0p+0f/*0.000000e+00*/;
  float loaded[4];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  int pos = 0;
  half_t x_local_cast[4];
  int peer_pos = 0;
  float peer_scale = 0x0p+0f/*0.000000e+00*/;
  fp8_e4_t encoded_values[4];
  float peer_value = 0x0p+0f/*0.000000e+00*/;
  float reduced[4];
  scale = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    loaded[j] = 0x0p+0f/*0.000000e+00*/;
  }
  sf_var = sf[0];
  pos = token_topk_to_pos[((((int)blockIdx.x) * 2) + ((((int)threadIdx.x) & 63) >> 5))];
  if (0 <= pos) {
    scale = 0x1p+0f/*1.000000e+00*/;
    scale = topk_weights[((((int)blockIdx.x) * 2) + ((((int)threadIdx.x) & 63) >> 5))];
    float condval;
    if ((0 <= pos)) {
      condval = x_sf[pos];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    scale = (scale * condval);
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint2 condval_1;
    if ((0 <= pos)) {
      condval_1 = *(uint2*)(x + (((((int64_t)pos) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
    } else {
      condval_1 = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint2*)(x_local_cast + 0) = condval_1;
    float4 __1;
    uint2 v_ = *(uint2*)(x_local_cast + 0);
    ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
    ((float2*)(&__1))[1] = __half22float2(((half2*)(&v_))[1]);
    *(float4*)(loaded + 0) = __1;
  }
  peer_pos = __shfl_sync((uint64_t)18446744073709551615, pos, ((((int)threadIdx.x) & 63) ^ 32), 64);
  peer_scale = __shfl_sync((uint64_t)18446744073709551615, scale, ((((int)threadIdx.x) & 63) ^ 32), 64);
  #pragma unroll
  for (int j_1 = 0; j_1 < 4; ++j_1) {
    peer_value = __shfl_sync((uint64_t)18446744073709551615, loaded[j_1], ((((int)threadIdx.x) & 63) ^ 32), 64);
    if (((((int)threadIdx.x) & 63) >> 5) == 0) {
      reduced[j_1] = 0x0p+0f/*0.000000e+00*/;
      if (0 <= pos) {
        reduced[j_1] = (reduced[j_1] + (loaded[j_1] * scale));
      }
      if (0 <= peer_pos) {
        reduced[j_1] = (reduced[j_1] + (peer_value * peer_scale));
      }
      float value = (reduced[j_1] * sf_var);
      uint raw = (*(uint *)(&(value)));
      uint magnitude = ((*(uint *)(&(value))) & (uint)2147483647);
      uint sign = (((*(uint *)(&(value))) >> (uint)24) & (uint)128);
      uint normal = ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960);
      uint v__1 = (*(uint *)(&(value))) & (uint)2147483647;
      float v__2 = (*(float *)(&(v__1))) + 0x1p+14f/*1.638400e+04*/;
      uint denormal = ((*(uint *)(&(v__2))) - (uint)1182793728);
      uint v__3 = (*(uint *)(&(value))) & (uint)2147483647;
      float v__4 = (*(float *)(&(v__3))) + 0x1p+14f/*1.638400e+04*/;
      uint encoded = (((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__4))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960)));
      uint v__5 = (*(uint *)(&(value))) & (uint)2147483647;
      float v__6 = (*(float *)(&(v__5))) + 0x1p+14f/*1.638400e+04*/;
      uint final_encoded = (((uint)2139095040 < ((*(uint *)(&(value))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__6))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(value))) >> (uint)24) & (uint)128)));
      uint v__7 = (*(uint *)(&(value))) & (uint)2147483647;
      float v__8 = (*(float *)(&(v__7))) + 0x1p+14f/*1.638400e+04*/;
      uchar v__9 = (uchar)(((uint)2139095040 < ((*(uint *)(&(value))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(value))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__8))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(value))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(value))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(value))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(value))) >> (uint)24) & (uint)128)));
      encoded_values[j_1] = (*(fp8_e4_t *)(&(v__9)));
    }
  }
  if (((((int)threadIdx.x) & 63) >> 5) == 0) {
    *(fp8_e4_4_t*)(out + (((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 6) * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(fp8_e4_4_t*)(encoded_values + 0);
  }
}

