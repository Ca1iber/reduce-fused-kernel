#include <mc_runtime.h>
#include <mc_common.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CHECK(call) do { mcError_t s = (call); if (s != mcSuccess) { \
  std::fprintf(stderr, "MACA error line %d: %s\n", __LINE__, mcGetErrorString(s)); \
  std::exit(1); } } while (0)

__host__ __device__ uint32_t mix(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15;
  v *= 0x846ca68bU; v ^= v >> 16; return v;
}
__host__ __device__ uint4 pattern(size_t i, uint32_t salt) {
  uint32_t v = static_cast<uint32_t>(i) + salt;
  return make_uint4(mix(v), mix(v + 0x9e3779b9U),
                    mix(v + 0x3c6ef372U), mix(v + 0xdaa66d2bU));
}
__global__ void initialize(uint4* a, size_t n, uint32_t salt) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) a[i] = pattern(i, salt);
}
__global__ void copy_probe(const uint4* __restrict__ x, uint4* __restrict__ y, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) y[i] = x[i];
}
template<bool NARROW>
__global__ void read8_probe(const uint4* __restrict__ x, void* __restrict__ y, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    uint4 s = make_uint4(0, 0, 0, 0);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint4 v = x[static_cast<size_t>(k) * n + i];
      s.x ^= v.x; s.y ^= v.y; s.z ^= v.z; s.w ^= v.w;
    }
    if constexpr (NARROW) {
      reinterpret_cast<uint2*>(y)[i] = make_uint2(s.x ^ s.z, s.y ^ s.w);
    } else {
      reinterpret_cast<uint4*>(y)[i] = s;
    }
  }
}

void launch(int mode, const uint4* x, void* y, size_t input_bytes, int threads, int blocks) {
  size_t n = input_bytes / sizeof(uint4);
  if (mode == 0) copy_probe<<<blocks, threads>>>(x, static_cast<uint4*>(y), n);
  else if (mode == 1) read8_probe<false><<<blocks, threads>>>(x, y, n / 8);
  else read8_probe<true><<<blocks, threads>>>(x, y, n / 8);
  CHECK(mcGetLastError());
}

void verify(int mode, const uint4* x, void* y, size_t input_bytes, int threads, int blocks) {
  launch(mode, x, y, input_bytes, threads, blocks);
  CHECK(mcDeviceSynchronize());
  size_t n = input_bytes / sizeof(uint4) / (mode == 0 ? 1 : 8);
  size_t samples[] = {0, 1, 17, n / 2, n - 1};
  for (size_t i : samples) {
    uint4 expected = make_uint4(0, 0, 0, 0);
    for (int k = 0; k < (mode == 0 ? 1 : 8); ++k) {
      uint4 v = pattern(static_cast<size_t>(k) * n + i, 0x12345678U);
      expected.x ^= v.x; expected.y ^= v.y; expected.z ^= v.z; expected.w ^= v.w;
    }
    if (mode == 2) {
      uint2 actual;
      CHECK(mcMemcpy(&actual, static_cast<uint2*>(y) + i, sizeof(actual), mcMemcpyDeviceToHost));
      if (actual.x != (expected.x ^ expected.z) || actual.y != (expected.y ^ expected.w)) std::exit(2);
    } else {
      uint4 actual;
      CHECK(mcMemcpy(&actual, static_cast<uint4*>(y) + i, sizeof(actual), mcMemcpyDeviceToHost));
      if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) std::exit(2);
    }
  }
}

double measure(int mode, const uint4* x, void* y, size_t input_bytes, int threads,
               int blocks, int repeats, mcEvent_t start, mcEvent_t stop) {
  CHECK(mcEventRecord(start));
  for (int i = 0; i < repeats; ++i) launch(mode, x, y, input_bytes, threads, blocks);
  CHECK(mcEventRecord(stop)); CHECK(mcEventSynchronize(stop));
  float elapsed;
  CHECK(mcEventElapsedTime(&elapsed, start, stop));
  return static_cast<double>(elapsed) / repeats;
}

int main(int argc, char** argv) {
  int input_mib = argc > 1 ? std::atoi(argv[1]) : 1024;
  bool confirm = argc > 2;
  int confirm_mode = confirm ? std::atoi(argv[2]) : -1;
  int confirm_threads = confirm ? std::atoi(argv[3]) : 0;
  int confirm_blocks = confirm ? std::atoi(argv[4]) : 0;
  size_t bytes = static_cast<size_t>(input_mib) * 1024 * 1024;
  mcDeviceProp_t prop{}; CHECK(mcGetDeviceProperties(&prop, 0));
  std::fprintf(stderr, "device=%s AP=%d visible_bytes=%zu input_MiB=%d\n",
               prop.name, prop.multiProcessorCount, static_cast<size_t>(prop.totalGlobalMem), input_mib);
  uint4* x; void* y;
  CHECK(mcMalloc(reinterpret_cast<void**>(&x), bytes)); CHECK(mcMalloc(&y, bytes));
  initialize<<<prop.multiProcessorCount * 8, 256>>>(x, bytes / sizeof(uint4), 0x12345678U);
  initialize<<<prop.multiProcessorCount * 8, 256>>>(static_cast<uint4*>(y), bytes / sizeof(uint4), 0x87654321U);
  CHECK(mcGetLastError()); CHECK(mcDeviceSynchronize());
  mcEvent_t start, stop; CHECK(mcEventCreate(&start)); CHECK(mcEventCreate(&stop));
  const char* names[] = {"copy_1to1", "read8_write_8to1", "read8_write_16to1"};
  std::printf("mode,input_MiB,read_bytes,write_bytes,threads,blocks,round,latency_ms,total_TBps\n");
  for (int t : {128, 256, 512}) {
    if (confirm && t != confirm_threads) continue;
    for (int factor : {4, 8, 16}) {
      int blocks = confirm ? confirm_blocks : prop.multiProcessorCount * factor;
      if (confirm && factor != 4) continue;
      for (int mode = 0; mode < 3; ++mode) {
        if (confirm && mode != confirm_mode) continue;
        verify(mode, x, y, bytes, t, blocks);
        for (int i = 0; i < 8; ++i) launch(mode, x, y, bytes, t, blocks);
        CHECK(mcDeviceSynchronize());
        size_t write_bytes = mode == 0 ? bytes : mode == 1 ? bytes / 8 : bytes / 16;
        for (int round = 0; round < 5; ++round) {
          double ms = measure(mode, x, y, bytes, t, blocks, 32, start, stop);
          double tbps = static_cast<double>(bytes + write_bytes) / (ms * 1.0e9);
          std::printf("%s,%d,%zu,%zu,%d,%d,%d,%.9f,%.9f\n", names[mode], input_mib,
                       bytes, write_bytes, t, blocks, round, ms, tbps);
          std::fflush(stdout);
        }
      }
    }
  }
  CHECK(mcEventDestroy(start)); CHECK(mcEventDestroy(stop));
  CHECK(mcFree(x)); CHECK(mcFree(y));
  std::fprintf(stderr, "sample_verification=PASS\n");
}
