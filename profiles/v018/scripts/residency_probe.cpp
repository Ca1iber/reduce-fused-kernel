#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <cstdint>
static void ck(mcError_t e,const char*w){if(e!=mcSuccess){fprintf(stderr,"%s: %d %s\n",w,(int)e,mcGetErrorString(e));exit(2);}}
int main(int argc,char**argv){
 if(argc<9)return 1;
 int threads=atoi(argv[2]),smem=atoi(argv[3]),outbytes=atoi(argv[4]),gy=atoi(argv[5]),tokens=atoi(argv[6]),hidden=7168,topk=8;
 unsigned long long cycles=strtoull(argv[8],nullptr,10);
 mcDeviceProp_t prop{};ck(mcGetDeviceProperties(&prop,0),"device");ck(mcSetDevice(0),"setdevice");
 mcModule_t m;mcFunction_t f;ck(mcModuleLoad(&m,argv[1]),"load");ck(mcModuleGetFunction(&f,m,"reduce_fused_kernel_kernel"),"function");
 int regs=0,privatebytes=0,capacity=0;ck(mcFuncGetAttribute(&regs,MC_FUNC_ATTRIBUTE_NUM_REGS,f),"regs");ck(mcFuncGetAttribute(&privatebytes,MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f),"private");ck(mcModuleOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,f,threads,smem),"capacity");
 size_t n=(size_t)tokens*topk*hidden;std::vector<uint16_t> hx(n,0x3f80);std::vector<float>hw((size_t)tokens*topk,0.125f),hxs((size_t)tokens*topk,1.0f);std::vector<int>hp((size_t)tokens*topk);for(size_t i=0;i<hp.size();++i)hp[i]=i;
 void *x,*w,*pos,*out,*sf,*xs,*counters;ck(mcMalloc(&x,n*2),"x alloc");ck(mcMalloc(&w,hw.size()*4),"w alloc");ck(mcMalloc(&pos,hp.size()*4),"pos alloc");ck(mcMalloc(&out,(size_t)tokens*hidden*outbytes),"out alloc");ck(mcMalloc(&sf,4),"sf alloc");ck(mcMalloc(&xs,hxs.size()*4),"xs alloc");ck(mcMalloc(&counters,8),"counter alloc");float scale=1.0f;
 ck(mcMemcpy(x,hx.data(),n*2,mcMemcpyHostToDevice),"x copy");ck(mcMemcpy(w,hw.data(),hw.size()*4,mcMemcpyHostToDevice),"w copy");ck(mcMemcpy(pos,hp.data(),hp.size()*4,mcMemcpyHostToDevice),"pos copy");ck(mcMemcpy(sf,&scale,4,mcMemcpyHostToDevice),"sf copy");ck(mcMemcpy(xs,hxs.data(),hxs.size()*4,mcMemcpyHostToDevice),"xs copy");
 std::map<std::string,void*>args={{"out",&out},{"sf",&sf},{"token_topk_to_pos",&pos},{"topk_weights",&w},{"x",&x},{"x_sf",&xs},{"num_tokens",&tokens},{"hold_cycles",&cycles},{"counters",&counters}};
 std::stringstream order(argv[7]);std::string key;std::vector<void*>params;while(std::getline(order,key,',')){if(!args.count(key)){fprintf(stderr,"unknown arg %s\n",key.c_str());return 3;}params.push_back(args[key]);}
 printf("{\"regs_per_thread\":%d,\"private_bytes\":%d,\"threads\":%d,\"shared_bytes\":%d,\"capacity_per_ap\":%d,\"ap_count\":%d,\"grid_ctas\":%d,\"hold_cycles\":%llu,\"runs\":[",regs,privatebytes,threads,smem,capacity,prop.multiProcessorCount,tokens*gy,cycles);
 for(int run=0;run<3;++run){
  ck(mcMemset(counters,0,8),"zero counters");ck(mcMemset(out,0,(size_t)tokens*hidden*outbytes),"zero out");
  ck(mcModuleLaunchKernel(f,tokens,gy,1,threads,1,1,smem,0,params.data(),nullptr),"launch");ck(mcDeviceSynchronize(),"sync");
  unsigned int hc[2]={0,0};ck(mcMemcpy(hc,counters,8,mcMemcpyDeviceToHost),"counts");uint16_t sample[16]={};ck(mcMemcpy(sample,out,16*outbytes,mcMemcpyDeviceToHost),"sample");int wrong=0;
  if(outbytes==1){auto*b=(unsigned char*)sample;for(int i=0;i<16;++i)if(b[i]!=0x38)++wrong;}else{for(int i=0;i<16;++i)if(sample[i]!=0x3f80)++wrong;}
  printf("%s{\"live_after\":%u,\"peak_live_ctas\":%u,\"peak_div_ap\":%.6f,\"wrong_sample\":%d}",run?",":"",hc[0],hc[1],(double)hc[1]/prop.multiProcessorCount,wrong);fflush(stdout);
 }
 printf("]}\n");mcModuleUnload(m);mcFree(counters);mcFree(xs);mcFree(sf);mcFree(out);mcFree(pos);mcFree(w);mcFree(x);return 0;
}
