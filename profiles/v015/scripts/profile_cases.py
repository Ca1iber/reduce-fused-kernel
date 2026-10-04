from pathlib import Path
import argparse,torch
from hidden_kernel import get_hidden_kernel
from grouped_hidden_kernel import get_hidden_kernel as get_grouped
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--case',default='all');a=p.parse_args()
names=('baseline','stream_direct','stream_sync','stream_async','group2_direct','group2_async','group4_async') if a.case=='all' else [a.case]
torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device='cuda')
args=(7168,8,'bfloat16','float8_e4m3fn',True,True,False)
base=get_reduce_fused_kernel(*args,tile_hidden=512,num_threads=128)
base(x,w,pos,out,sf,unused);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name in names:
 if name=='baseline':k=base
 elif name.startswith('group'):
  k=get_grouped(*args,tile_hidden=512,num_threads=128,mode=0 if name.endswith('direct') else 2,group_size=int(name[5]))
 else:
  k=get_hidden_kernel(*args,tile_hidden=512,num_threads=128,mode={'stream_direct':0,'stream_sync':1,'stream_async':2}[name])
 (r/f'codegen/profile_{name}.cu').write_text(k.get_kernel_source())
 for _ in range(10):k(x,w,pos,out,sf,unused)
 torch.cuda.synchronize();assert torch.equal(expected,out.view(torch.uint8)),name
 torch.cuda.profiler.start();k(x,w,pos,out,sf,unused);torch.cuda.synchronize();torch.cuda.profiler.stop()
 print('PROFILE',name,flush=True)
