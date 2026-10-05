from pathlib import Path
import argparse,torch
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from cooperative_kernel import get_cooperative_kernel as local_positions
from cooperative_global_positions import get_cooperative_kernel as global_positions
r=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--case',default='all');a=parser.parse_args()
cases=('baseline','direct1024t512','warp512_local','warp512_global','warp1024_global')
chosen=cases if a.case=='all' else [a.case]
torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device='cuda')
for name in chosen:
 if name=='baseline':k=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128)
 elif name=='direct1024t512':k=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=1024,num_threads=512)
 else:
    tile=1024 if name=='warp1024_global' else 512
    threads=512 if tile==1024 else 256
    factory=local_positions if name.endswith('_local') else global_positions
    k=factory(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=tile,num_threads=threads,cooperative=True)
 (r/f'codegen/profile_{name}.cu').write_text(k.get_kernel_source())
 for _ in range(10):k(x,w,pos,out,sf,unused)
 torch.cuda.synchronize();torch.cuda.profiler.start();k(x,w,pos,out,sf,unused);torch.cuda.synchronize();torch.cuda.profiler.stop()
 print('PROFILE_CASE',name,flush=True)
