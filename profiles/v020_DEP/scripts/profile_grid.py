from pathlib import Path
import argparse,torch,json
from grid_kernel import get_reduce_fused_kernel as candidate
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel as baseline
from workloads.moe import MoeReduceFusedWorkload
r=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--case',choices=['v010','grid_hidden_first','all'],default='all');a.add_argument('--workload',choices=['h7168','prefill'],default='h7168');args=a.parse_args()
t=512 if args.workload=='h7168'else 4096;tile=512 if t==512 else 1024
torch.set_num_threads(1);torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(t,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((t,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device='cuda')
(r/f'raw/profile_positions_{args.workload}_{args.case}.json').write_text(json.dumps(pos.cpu().tolist())+'\n')
# Native profiling follows the previously validated minimal driver, separately from the L2-flushed benchmark.
# No extra flush kernel in native capture; dataset is larger than L2.
for name in ['v010','grid_hidden_first'] if args.case=='all'else [args.case]:
 k=(baseline if name=='v010'else candidate)(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=tile,num_threads=128)
 (r/f'codegen/profile_{args.workload}_{name}.cu').write_text(k.get_kernel_source())
 for _ in range(10):k(x,w,pos,out,sf,unused)
 torch.cuda.synchronize();torch.cuda.profiler.start()
 k(x,w,pos,out,sf,unused);torch.cuda.synchronize();torch.cuda.profiler.stop()
 print('PROFILE_COMPLETE',args.workload,name,flush=True)
