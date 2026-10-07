#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(256, 1) reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens) {
  float reduced_fragment[1];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  float topk_weights_local[2];
  int topk_to_pos_local[2];
  float s = 0x0p+0f/*0.000000e+00*/;
  fp8_e4_t encoded_fragment[1];
  reduced_fragment[0] = 0x0p+0f/*0.000000e+00*/;
  sf_var = sf[0];
  *(float2*)(topk_weights_local + 0) = *(float2*)(topk_weights + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  *(int2*)(topk_to_pos_local + 0) = *(int2*)(token_topk_to_pos + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    int pos = topk_to_pos_local[k];
    if (0 <= pos) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      reduced_fragment[0] = (reduced_fragment[0] + (((float)x[((((int64_t)pos) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]) * s));
    }
  }
  float value = (reduced_fragment[0] * sf_var);
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
  encoded_fragment[0] = (*(fp8_e4_t *)(&(v__8)));
  out[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))] = encoded_fragment[0];
}

