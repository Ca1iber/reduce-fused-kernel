#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens);
extern "C" __global__ void __launch_bounds__(32, 1) reduce_fused_kernel_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf, int num_tokens) {
  float reduced_fragment[8];
  float topk_weights_local[2];
  int topk_to_pos_local[2];
  float s = 0x0p+0f/*0.000000e+00*/;
  half_t x_local_cast[8];
  half_t out_local_cast_1[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(reduced_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(float2*)(topk_weights_local + 0) = *(float2*)(topk_weights + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  *(int2*)(topk_to_pos_local + 0) = *(int2*)(token_topk_to_pos + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    int pos = topk_to_pos_local[k];
    if (0 <= pos) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      s = (s * x_sf[((int64_t)pos)]);
      *(uint4*)(x_local_cast + 0) = *(uint4*)(x + ((((int64_t)pos) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        float4 __1;
          float4 v_ = *(float4*)(reduced_fragment + (i_1 * 4));
          float4 __2;
            float4 __3;
            uint2 v__1 = *(uint2*)(x_local_cast + (i_1 * 4));
            ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
            ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
            float4 v__2 = make_float4(s, s, s, s);
            __2.x = (__3.x*v__2.x);
            __2.y = (__3.y*v__2.y);
            __2.z = (__3.z*v__2.z);
            __2.w = (__3.w*v__2.w);
          __1.x = (v_.x+__2.x);
          __1.y = (v_.y+__2.y);
          __1.z = (v_.z+__2.z);
          __1.w = (v_.w+__2.w);
        *(float4*)(reduced_fragment + (i_1 * 4)) = __1;
      }
    }
  }
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    uint2 __4;
    float4 v__3 = *(float4*)(reduced_fragment + (i_2 * 4));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
    ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
    *(uint2*)(out_local_cast_1 + (i_2 * 4)) = __4;
  }
  *(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(out_local_cast_1 + 0);
}

