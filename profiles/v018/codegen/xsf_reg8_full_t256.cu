#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(bfloat16_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const bfloat16_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens);
extern "C" __global__ void __launch_bounds__(256, 1) reduce_fused_kernel_kernel(bfloat16_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const bfloat16_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens) {
  float reduced_fragment[28];
  float topk_weights_local[8];
  int topk_to_pos_local[8];
  bfloat16_t prefetched[224];
  float s = 0x0p+0f/*0.000000e+00*/;
  bfloat16_t out_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(reduced_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(topk_weights_local + (i_1 * 4)) = *(float4*)(topk_weights + ((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((int64_t)i_1) * (int64_t)4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(int4*)(topk_to_pos_local + (i_2 * 4)) = *(int4*)(token_topk_to_pos + ((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((int64_t)i_2) * (int64_t)4)));
  }
  #pragma unroll
  for (int first = 0; first < 8; ++first) {
    int pos = topk_to_pos_local[first];
    if (0 <= pos) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 7; ++i_3) {
        *(uint2*)(prefetched + ((first * 28) + (i_3 * 4))) = *(uint2*)(x + (((((int64_t)pos) * (int64_t)7168) + (((int64_t)i_3) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      }
    }
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int pos_1 = topk_to_pos_local[k];
    if (0 <= pos_1) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      s = (s * x_sf[((int64_t)pos_1)]);
      #pragma unroll
      for (int i_4 = 0; i_4 < 7; ++i_4) {
        float4 __1;
          float4 v_ = *(float4*)(reduced_fragment + (i_4 * 4));
          float4 __2;
            float4 __3;
            uint2 v__1 = *(uint2*)(prefetched + ((k * 28) + (i_4 * 4)));
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
        *(float4*)(reduced_fragment + (i_4 * 4)) = __1;
      }
    }
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 7; ++i_5) {
    uint2 __4;
    float4 v__3 = *(float4*)(reduced_fragment + (i_5 * 4));
    (reinterpret_cast<__maca_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
    (reinterpret_cast<__maca_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
    *(uint2*)(out_local_cast + 0) = __4;
    *(uint2*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)7168) + (((int64_t)i_5) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(out_local_cast + 0);
  }
}

