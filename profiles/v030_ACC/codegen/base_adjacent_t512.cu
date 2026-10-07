#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void tiny_pair_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(512, 1) tiny_pair_kernel(half_t* __restrict__ out, const int* __restrict__ token_topk_to_pos, const float* __restrict__ topk_weights, const half_t* __restrict__ x) {
  float scale = 0x0p+0f/*0.000000e+00*/;
  float loaded[1];
  int pos = 0;
  int peer_pos = 0;
  float peer_scale = 0x0p+0f/*0.000000e+00*/;
  float peer_value = 0x0p+0f/*0.000000e+00*/;
  float reduced[1];
  half_t encoded_values[1];
  scale = 0x0p+0f/*0.000000e+00*/;
  loaded[0] = 0x0p+0f/*0.000000e+00*/;
  pos = token_topk_to_pos[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
  if (0 <= pos) {
    scale = 0x1p+0f/*1.000000e+00*/;
    scale = topk_weights[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) & 1))];
    half_t condval;
    if ((0 <= pos)) {
      condval = x[((((int64_t)pos) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) >> (int64_t)1))];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    loaded[0] = ((float)condval);
  }
  peer_pos = __shfl_sync((uint64_t)18446744073709551615, pos, ((((int)threadIdx.x) & 63) ^ 1), 64);
  peer_scale = __shfl_sync((uint64_t)18446744073709551615, scale, ((((int)threadIdx.x) & 63) ^ 1), 64);
  peer_value = __shfl_sync((uint64_t)18446744073709551615, loaded[0], ((((int)threadIdx.x) & 63) ^ 1), 64);
  if ((((int)threadIdx.x) % 2) == 0) {
    reduced[0] = 0x0p+0f/*0.000000e+00*/;
    if (0 <= pos) {
      reduced[0] = (reduced[0] + (loaded[0] * scale));
    }
    if (0 <= peer_pos) {
      reduced[0] = (reduced[0] + (peer_value * peer_scale));
    }
    encoded_values[0] = ((half_t)reduced[0]);
    out[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 1))] = encoded_values[0];
  }
}

