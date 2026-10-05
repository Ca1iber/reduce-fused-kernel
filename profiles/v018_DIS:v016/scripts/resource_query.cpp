#include <mc_runtime.h>
#include <mc_occupancy.h>
#include <cstdio>
#include <cstdlib>
static void ck(mcError_t e,const char* w){if(e!=mcSuccess){fprintf(stderr,"%s: %d %s\n",w,(int)e,mcGetErrorString(e));exit(2);}}
int main(int argc,char**argv){
 mcDeviceProp_t p{};ck(mcGetDeviceProperties(&p,0),"properties");
 printf("{\"device\":\"%s\",\"major\":%d,\"minor\":%d,\"ap_count\":%d,\"warp_size\":%d,\"regs_per_ap_32bit\":%d,\"regs_per_block\":%d,\"max_threads_per_ap\":%d,\"max_threads_per_block\":%d,\"max_blocks_per_ap\":%d,\"shared_per_ap_bytes\":%zu,\"shared_per_block_bytes\":%zu,\"visible_global_bytes\":%zu",p.name,p.major,p.minor,p.multiProcessorCount,p.warpSize,p.regsPerMultiprocessor,p.regsPerBlock,p.maxThreadsPerMultiProcessor,p.maxThreadsPerBlock,p.maxBlocksPerMultiProcessor,p.sharedMemPerMultiprocessor,p.sharedMemPerBlock,p.totalGlobalMem);
 if(argc>=4){
  mcModule_t m;mcFunction_t f;ck(mcModuleLoad(&m,argv[1]),"load");ck(mcModuleGetFunction(&f,m,argv[2]),"function");
  int regs=0,smem=0,priv=0,limit=0,maxblock=0;int threads=atoi(argv[3]);size_t dynamic=argc>4?strtoull(argv[4],nullptr,10):0;
  ck(mcFuncGetAttribute(&regs,MC_FUNC_ATTRIBUTE_NUM_REGS,f),"registers");ck(mcFuncGetAttribute(&smem,MC_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f),"shared");ck(mcFuncGetAttribute(&priv,MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f),"private");ck(mcFuncGetAttribute(&maxblock,MC_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f),"maxblock");
  mcError_t e=mcModuleOccupancyMaxActiveBlocksPerMultiprocessor(&limit,f,threads,dynamic);
  printf(",\"registers_per_thread\":%d,\"static_shared_bytes\":%d,\"private_bytes\":%d,\"kernel_max_threads\":%d,\"block_threads\":%d,\"dynamic_shared_bytes\":%zu,\"module_occupancy_status\":%d,\"module_active_blocks_per_ap\":%d",regs,smem,priv,maxblock,threads,dynamic,(int)e,limit);
  mcOccDeviceProp dp(p);mcOccFuncAttributes a;a.numRegs=regs;a.sharedSizeBytes=smem;a.maxThreadsPerBlock=maxblock;mcOccDeviceState state;mcOccResult result{};
  mcOccError oe=mcOccMaxActiveBlocksPerMultiprocessor(&result,&dp,&a,&state,threads,dynamic);
  printf(",\"header_occupancy_status\":%d,\"header_active_blocks_per_ap\":%d,\"allocated_regs_per_block_32bit\":%d,\"reg_limit_blocks\":%d,\"thread_limit_blocks\":%d,\"block_limit_blocks\":%d,\"shared_limit_blocks\":%d,\"limiting_factors\":%u",(int)oe,result.activeBlocksPerMultiprocessor,result.allocatedRegistersPerBlock,result.blockLimitRegs,result.blockLimitWarps,result.blockLimitBlocks,result.blockLimitSharedMem,result.limitingFactors);
  mcModuleUnload(m);
 }
 printf("}\n");return 0;
}
