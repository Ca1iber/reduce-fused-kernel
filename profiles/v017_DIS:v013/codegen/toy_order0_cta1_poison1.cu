#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void minimal_kernel(int* __restrict__ out, const int* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) minimal_kernel(int* __restrict__ out, const int* __restrict__ x) {
  extern __shared__ __align__(1024) int shared[];
  b64vectype tickets[2];
  int broadcast_var = -999;
  *(int4*)(shared + (((int)threadIdx.x) * 4)) = make_int4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  __syncthreads();
  tickets[0] = memcpy_async<8>((void* __restrict__)(&(shared[(((int)threadIdx.x) * 2)])), (void* __restrict__)(&(x[(((int)threadIdx.x) * 2)])));
  tickets[1] = memcpy_async<8>((void* __restrict__)(&(shared[((((int)threadIdx.x) * 2) + 256)])), (void* __restrict__)(&(x[((((int)threadIdx.x) * 2) + 256)])));
  barrier_arrive_and_wait(tickets[0]);
  __syncthreads();
  *(int2*)(out + (((int)threadIdx.x) * 2)) = *(int2*)(shared + (((int)threadIdx.x) * 2));
  barrier_arrive_and_wait(tickets[1]);
  __syncthreads();
  *(int2*)(out + ((((int)threadIdx.x) * 2) + 256)) = *(int2*)(shared + ((((int)threadIdx.x) * 2) + 256));
}

