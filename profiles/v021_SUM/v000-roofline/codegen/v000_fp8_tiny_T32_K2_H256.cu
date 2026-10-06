#include <tl_templates/maca/maca_fp8.h>
#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) reduce_fused_kernel_kernel(fp8_e4_t* __restrict__ out, const float* __restrict__ sf, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, int num_tokens) {
  float reduced_fragment[2];
  float sf_var = 0x0p+0f/*0.000000e+00*/;
  float topk_weights_local[2];
  int topk_to_pos_local[2];
  float s = 0x0p+0f/*0.000000e+00*/;
  half_t x_local_cast[2];
  fp8_e4_t out_local_cast_1[2];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(reduced_fragment + 0) = make_float2(broadcast_var, broadcast_var);
  sf_var = sf[0];
  *(float2*)(topk_weights_local + 0) = *(float2*)(topk_weights + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  *(int2*)(topk_to_pos_local + 0) = *(int2*)(token_topk_to_pos + (((int64_t)((int)blockIdx.x)) * (int64_t)2));
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    int pos = topk_to_pos_local[k];
    if (0 <= pos) {
      s = 0x1p+0f/*1.000000e+00*/;
      s = topk_weights_local[k];
      *(uint1*)(x_local_cast + 0) = *(uint1*)(x + ((((int64_t)pos) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)));
      float2 __1;
      uint1 v_ = *(uint1*)(x_local_cast + 0);
      ((float2*)(&__1))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(reduced_fragment + 0) = tl::add2(*(float2*)(reduced_fragment + 0), tl::mul2(__1, make_float2(s, s)));
    }
  }
  fp8_e4_2_t __2;
  float2 v__1 = tl::mul2(*(float2*)(reduced_fragment + 0), make_float2(sf_var, sf_var));
  (reinterpret_cast<__maca_fp8x2_storage_t*>(&__2))[0] = __maca_cvt_float2_to_fp8x2(((float2*)(&v__1))[0], __MACA_SATFINITE, __MACA_E4M3);
  *(fp8_e4_2_t*)(out_local_cast_1 + 0) = __2;
  *(fp8_e4_2_t*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)2))) = *(fp8_e4_2_t*)(out_local_cast_1 + 0);
}

