#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void probe_kernel(half_t* __restrict__ out, const half_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) probe_kernel(half_t* __restrict__ out, const half_t* __restrict__ x) {
  extern __shared__ __align__(1024) half_t buffer[];
  b128vectype ticket[1];
  ticket[0] = memcpy_async<16>((void* __restrict__)(&(buffer[(((int)threadIdx.x) * 8)])), (void* __restrict__)(&(x[((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8))])));
  barrier_arrive_and_wait(ticket[0]);
  __syncthreads();
  *(uint4*)(out + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(buffer + (((int)threadIdx.x) * 8));
}

