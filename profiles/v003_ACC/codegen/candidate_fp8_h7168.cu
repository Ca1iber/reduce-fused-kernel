#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_candidate_kernel(fp8_e4_t* __restrict__ out, const int* __restrict__ positions, const float* __restrict__ sf, const float* __restrict__ weights, const bfloat16_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) reduce_fused_candidate_kernel(fp8_e4_t* __restrict__ out, const int* __restrict__ positions, const float* __restrict__ sf, const float* __restrict__ weights, const bfloat16_t* __restrict__ x, int num_tokens) {
  float acc[56];
  float sf_value = 0x0p+0f/*0.000000e+00*/;
  float w[8];
  int p[8];
  float scale = 0x0p+0f/*0.000000e+00*/;
  bfloat16_t x_local_cast[8];
  fp8_e4_t out_local_cast_1[4];
  #pragma unroll
  for (int i = 0; i < 14; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  sf_value = sf[0];
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(w + (i_1 * 4)) = *(float4*)(weights + ((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((int64_t)i_1) * (int64_t)4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(int4*)(p + (i_2 * 4)) = *(int4*)(positions + ((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((int64_t)i_2) * (int64_t)4)));
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int pos = p[k];
    if (0 <= pos) {
      scale = 0x1p+0f/*1.000000e+00*/;
      scale = w[k];
      #pragma unroll
      for (int i_3 = 0; i_3 < 7; ++i_3) {
        *(uint4*)(x_local_cast + 0) = *(uint4*)(x + (((((int64_t)pos) * (int64_t)7168) + (((int64_t)i_3) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        for (int vec = 0; vec < 2; ++vec) {
          float4 __1;
            float4 v_ = *(float4*)(acc + ((i_3 * 8) + (vec * 4)));
            float4 __2;
              float4 __3;
              uint2 v__1 = *(uint2*)(x_local_cast + (vec * 4));
              ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[0]);
              ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[1]);
              float4 v__2 = make_float4(scale, scale, scale, scale);
              __2.x = (__3.x*v__2.x);
              __2.y = (__3.y*v__2.y);
              __2.z = (__3.z*v__2.z);
              __2.w = (__3.w*v__2.w);
            __1.x = (v_.x+__2.x);
            __1.y = (v_.y+__2.y);
            __1.z = (v_.z+__2.z);
            __1.w = (v_.w+__2.w);
          *(float4*)(acc + ((i_3 * 8) + (vec * 4))) = __1;
        }
      }
    }
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 14; ++i_4) {
    for (int vec_1 = 0; vec_1 < 4; ++vec_1) {
      float v__3 = acc[((i_4 * 4) + vec_1)] * sf_value;
      uint v__4 = (*(uint *)(&(v__3))) & (uint)2147483647;
      float v__5 = (*(float *)(&(v__4))) + 0x1p+14f/*1.638400e+04*/;
      uchar v__6 = (uchar)(((uint)2139095040 < ((*(uint *)(&(v__3))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(v__3))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__5))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(v__3))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(v__3))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(v__3))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(v__3))) >> (uint)24) & (uint)128)));
      out_local_cast_1[vec_1] = (*(fp8_e4_t *)(&(v__6)));
    }
    *(fp8_e4_4_t*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)7168) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)4))) = *(fp8_e4_4_t*)(out_local_cast_1 + 0);
  }
}

