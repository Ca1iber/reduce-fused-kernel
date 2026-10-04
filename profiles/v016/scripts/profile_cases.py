from pathlib import Path
import argparse,torch
from prefetch_kernel import get_prefetch_kernel
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',default='all');a=p.parse_args()
names=('baseline','register2','register4','register8') if a.case=='all' else [a.case]
torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device='cuda')
for name in names:
 if name=='baseline':k=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128)
 else:k=get_prefetch_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128,prefetch_rows=int(name[-1]))
 (r/f'codegen/profile_{name}.cu').write_text(k.get_kernel_source())
 for _ in range(10):k(x,w,pos,out,sf,unused)
 torch.cuda.synchronize();torch.cuda.profiler.start();k(x,w,pos,out,sf,unused);torch.cuda.synchronize();torch.cuda.profiler.stop()
 print('PROFILE',name,flush=True)
