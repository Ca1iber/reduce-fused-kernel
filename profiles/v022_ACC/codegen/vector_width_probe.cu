#include <mc_runtime.h>
struct __align__(64) Vec512 { uint4 a, b, c, d; };
extern "C" __global__ void width64(const uint2* __restrict__ x, uint2* __restrict__ y) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x; y[i]=x[i];
}
extern "C" __global__ void width128(const uint4* __restrict__ x, uint4* __restrict__ y) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x; y[i]=x[i];
}
extern "C" __global__ void width256(const ulonglong4* __restrict__ x, ulonglong4* __restrict__ y) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x; y[i]=x[i];
}
extern "C" __global__ void width512(const Vec512* __restrict__ x, Vec512* __restrict__ y) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x; y[i]=x[i];
}
