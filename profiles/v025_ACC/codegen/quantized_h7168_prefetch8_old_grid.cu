#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const bfloat16_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const bfloat16_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens) {
  float reduced_fragment[4];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  float topk_weights_local[8];
  int topk_to_pos_local[8];
  bfloat16_t prefetched[32];
  float s = 0x0p+0f/*0.000000e+00*/;
  fp8_e4_t out_local_cast[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(reduced_fragment + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  sf_var = sf[0];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(topk_weights_local + (i * 4)) = *(float4*)(topk_weights + ((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((int64_t)i) * (int64_t)4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(int4*)(topk_to_pos_local + (i_1 * 4)) = *(int4*)(token_topk_to_pos + ((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((int64_t)i_1) * (int64_t)4)));
  }
  #pragma unroll
  for (int first = 0; first < 8; ++first) {
    int pos = topk_to_pos_local[first];
    if (0 <= pos) {
      *(uint2*)(prefetched + (first * 4)) = *(uint2*)(x + (((((int64_t)pos) * (int64_t)7168) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
    }
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int pos_1 = topk_to_pos_local[k];
    if (0 <= pos_1) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      s = (s * x_sf[((int64_t)pos_1)]);
      float4 __1;
        float4 v_ = *(float4*)(reduced_fragment + 0);
        float4 __2;
          float4 __3;
          uint2 v__1 = *(uint2*)(prefetched + (k * 4));
          ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[0]);
          ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v__1))[1]);
          float4 v__2 = make_float4(s, s, s, s);
          __2.x = (__3.x*v__2.x);
          __2.y = (__3.y*v__2.y);
          __2.z = (__3.z*v__2.z);
          __2.w = (__3.w*v__2.w);
        __1.x = (v_.x+__2.x);
        __1.y = (v_.y+__2.y);
        __1.z = (v_.z+__2.z);
        __1.w = (v_.w+__2.w);
      *(float4*)(reduced_fragment + 0) = __1;
    }
  }
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    float v__3 = reduced_fragment[i_2] * sf_var;
    uint v__4 = (*(uint *)(&(v__3))) & (uint)2147483647;
    float v__5 = (*(float *)(&(v__4))) + 0x1p+14f/*1.638400e+04*/;
    uchar v__6 = (uchar)(((uint)2139095040 < ((*(uint *)(&(v__3))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(v__3))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__5))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(v__3))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(v__3))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(v__3))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(v__3))) >> (uint)24) & (uint)128)));
    out_local_cast[i_2] = (*(fp8_e4_t *)(&(v__6)));
  }
  *(fp8_e4_4_t*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)7168) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(fp8_e4_4_t*)(out_local_cast + 0);
}

