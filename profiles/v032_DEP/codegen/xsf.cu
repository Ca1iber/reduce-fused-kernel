#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_packed_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf);
extern "C" __global__ void __launch_bounds__(256, 1) tiny_packed_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x, const float* __restrict__ x_sf) {
  float scale = 0x0p+0f/*0.000000e+00*/;
  half_t loaded[2];
  int pos = 0;
  float peer_scale = 0x0p+0f/*0.000000e+00*/;
  half_t encoded_values[2];
  uint packed = (uint)0;
  uint peer_packed = (uint)0;
  float peer_value = 0x0p+0f/*0.000000e+00*/;
  float reduced[2];
  scale = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    loaded[j] = half_t(0x0p+0f/*0.000000e+00*/);
  }
  pos = token_topk_to_pos[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
  if (0 <= pos) {
    scale = 0x1p+0f/*1.000000e+00*/;
    scale = topk_weights[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
    float condval;
    if ((0 <= pos)) {
      condval = x_sf[pos];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    scale = (scale * condval);
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint1 condval_1;
    if ((0 <= pos)) {
      condval_1 = *(uint1*)(x + ((((int64_t)pos) * (int64_t)256) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)1) * (int64_t)2)));
    } else {
      condval_1 = make_uint1(__pack_half2(broadcast_var, broadcast_var));
    }
    *(uint1*)(loaded + 0) = condval_1;
  }
  peer_scale = __shfl_sync((uint64_t)18446744073709551615, scale, ((((int)threadIdx.x) & 63) ^ 1), 64);
  #pragma unroll
  for (int j_1 = 0; j_1 < 2; ++j_1) {
    half_t v_ = loaded[j_1];
    packed = (((uint)(*(ushort *)(&(v_)))) | ((0 <= pos) ? (uint)65536 : (uint)0));
    peer_packed = __shfl_sync((uint64_t)18446744073709551615, packed, ((((int)threadIdx.x) & 63) ^ 1), 64);
    ushort v__1 = (ushort)(peer_packed & (uint)65535);
    peer_value = ((float)(*(half_t *)(&(v__1))));
    if ((((int)threadIdx.x) % 2) == 0) {
      reduced[j_1] = 0x0p+0f/*0.000000e+00*/;
      if (0 <= pos) {
        reduced[j_1] = (reduced[j_1] + (((float)loaded[j_1]) * scale));
      }
      if ((peer_packed & (uint)65536) != (uint)0) {
        reduced[j_1] = (reduced[j_1] + (peer_value * peer_scale));
      }
      encoded_values[j_1] = ((half_t)reduced[j_1]);
    }
  }
  if ((((int)threadIdx.x) % 2) == 0) {
    *(uint1*)(out + ((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 2))) = *(uint1*)(encoded_values + 0);
  }
}

