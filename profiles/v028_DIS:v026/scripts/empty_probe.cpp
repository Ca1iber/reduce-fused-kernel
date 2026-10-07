#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#define CHECK(call) do { mcError_t e=(call); if(e!=mcSuccess){std::fprintf(stderr,"error=%s line=%d\n",mcGetErrorString(e),__LINE__);std::exit(1);} } while(0)
// Six pointer slots and two dynamic dimensions, as in the formal kernel.
// This is a real GPU entry point with an empty body, not an empty host function.
__global__ void tiny_empty(const void* x,const void* weights,const void* positions,
                          void* out,const void* sf,const void* xsf,int tokens,int expanded) {}
extern "C" int launch_empty(int threads) {
 tiny_empty<<<32,threads>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,32,67);
 return static_cast<int>(mcGetLastError());
}
int main() {
 mcDeviceProp_t p{};CHECK(mcGetDeviceProperties(&p,0));
 std::fprintf(stderr,"device=%s AP=%d visible_bytes=%zu grid=32\n",p.name,p.multiProcessorCount,static_cast<size_t>(p.totalGlobalMem));
 mcEvent_t start,stop;CHECK(mcEventCreate(&start));CHECK(mcEventCreate(&stop));
 std::printf("threads,blocks,batch,round,event_average_us,cpu_submit_loop_average_us,wall_completed_batch_average_us\n");
 for(int threads:{128,256}) {
  for(int i=0;i<100;i++)CHECK(static_cast<mcError_t>(launch_empty(threads)));
  CHECK(mcDeviceSynchronize());
  for(int n:{128,1024,8192}) {
   for(int round=0;round<5;round++) {
    CHECK(mcDeviceSynchronize());
    auto wall_begin=std::chrono::steady_clock::now();CHECK(mcEventRecord(start));
    auto submit_begin=std::chrono::steady_clock::now();
    for(int i=0;i<n;i++)CHECK(static_cast<mcError_t>(launch_empty(threads)));
    auto submit_end=std::chrono::steady_clock::now();CHECK(mcEventRecord(stop));CHECK(mcEventSynchronize(stop));
    auto wall_end=std::chrono::steady_clock::now();float ms;CHECK(mcEventElapsedTime(&ms,start,stop));
    double submit_us=std::chrono::duration<double,std::micro>(submit_end-submit_begin).count()/n;
    double wall_us=std::chrono::duration<double,std::micro>(wall_end-wall_begin).count()/n;
    std::printf("%d,32,%d,%d,%.9f,%.9f,%.9f\n",threads,n,round,ms*1000.0/n,submit_us,wall_us);std::fflush(stdout);
   }
  }
 }
 CHECK(mcEventDestroy(start));CHECK(mcEventDestroy(stop));std::fprintf(stderr,"all_launches_and_synchronizations=PASS\n");
}
