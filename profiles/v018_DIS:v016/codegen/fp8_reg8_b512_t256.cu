#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const bfloat16_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(256, 1) reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const bfloat16_t* __restrict__ x, int num_tokens) {
  float reduced_fragment[2];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  float topk_weights_local[8];
  int topk_to_pos_local[8];
  bfloat16_t prefetched[16];
  float s = 0x0p+0f/*0.000000e+00*/;
  fp8_e4_t out_local_cast[2];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(reduced_fragment + 0) = make_float2(broadcast_var, broadcast_var);
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
      *(uint1*)(prefetched + (first * 2)) = *(uint1*)(x + (((((int64_t)pos) * (int64_t)7168) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)));
    }
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    int pos_1 = topk_to_pos_local[k];
    if (0 <= pos_1) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      float2 __1;
      uint1 v_ = *(uint1*)(prefetched + (k * 2));
      ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__maca_bfloat162*>(&v_))[0]);
      *(float2*)(reduced_fragment + 0) = tl::add2(*(float2*)(reduced_fragment + 0), tl::mul2(__1, make_float2(s, s)));
    }
  }
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    float v__1 = reduced_fragment[i_2] * sf_var;
    uint v__2 = (*(uint *)(&(v__1))) & (uint)2147483647;
    float v__3 = (*(float *)(&(v__2))) + 0x1p+14f/*1.638400e+04*/;
    uchar v__4 = (uchar)(((uint)2139095040 < ((*(uint *)(&(v__1))) & (uint)2147483647)) ? (uint)127 : ((((((*(uint *)(&(v__1))) & (uint)2147483647) / (uint)1015021568) < (uint)1) ? ((*(uint *)(&(v__3))) - (uint)1182793728) : (((uint)1 <= (((*(uint *)(&(v__1))) & (uint)2147483647) / (uint)1138753536)) ? (uint)126 : ((((((*(uint *)(&(v__1))) & (uint)2147483647) + (uint)524287) + ((((*(uint *)(&(v__1))) & (uint)2147483647) >> (uint)20) & (uint)1)) >> (uint)20) - (uint)960))) | (((*(uint *)(&(v__1))) >> (uint)24) & (uint)128)));
    out_local_cast[i_2] = (*(fp8_e4_t *)(&(v__4)));
  }
  *(fp8_e4_2_t*)(out + (((((int64_t)((int)blockIdx.x)) * (int64_t)7168) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2))) = *(fp8_e4_2_t*)(out_local_cast + 0);
}

