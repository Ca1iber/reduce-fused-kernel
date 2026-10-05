from pathlib import Path
import argparse,json,torch
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
from pipeline_kernel import get_pipeline_kernel
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--case',default='all');args=parser.parse_args()
cases={'baseline':None,'shared_single':(1,False,True),'sync_ring2':(2,False,False),'async_ring2':(2,True,False),'async_ring3':(3,True,False)}
names=list(cases) if args.case=='all' else [args.case]
torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device='cuda')
for name in names:
    cfg=cases[name]
    if cfg is None:kernel=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128)
    else:kernel=get_pipeline_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128,stages=cfg[0],asynchronous=cfg[1],old_two_barriers=cfg[2])
    (root/f'codegen/profile_{name}.cu').write_text(kernel.get_kernel_source())
    for _ in range(10):kernel(x,w,pos,out,sf,unused)
    torch.cuda.synchronize();torch.cuda.profiler.start();kernel(x,w,pos,out,sf,unused);torch.cuda.synchronize();torch.cuda.profiler.stop()
    print('PROFILE_CASE_DONE',name,flush=True)
