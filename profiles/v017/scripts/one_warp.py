from pathlib import Path
import torch,json
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from pipeline_probe import get_pipeline_kernel
r=Path(__file__).resolve().parents[1]
torch.manual_seed(1235);x,pos,w=MoeReduceFusedWorkload(512,2,7168,torch.bfloat16).gen_inputs();sf=torch.empty(1,device='cuda');unused=torch.empty(x.shape[0],device='cuda');out=torch.empty((512,7168),device='cuda',dtype=torch.float32)
base=get_reduce_fused_kernel(7168,2,'bfloat16','float32',False,True,False,tile_hidden=256,num_threads=64);base(x,w,pos,out,sf,unused);torch.cuda.synchronize();expected=out.clone();rows=[]
for mode in (0,1,2):
 k=get_pipeline_kernel(7168,2,'bfloat16','float32',False,True,False,tile_hidden=256,num_threads=64,stages=2,asynchronous=True,sync_mode=mode)
 (r/f'codegen/one_warp_mode{mode}.cu').write_text(k.get_kernel_source());counts=[]
 for n in range(5):
  k(x,w,pos,out,sf,unused);torch.cuda.synchronize();counts.append(int((out!=expected).sum()))
 rows.append({'mode':mode,'wrong_elements':counts});print('ONE_WARP',mode,counts,flush=True)
(r/'raw/one_warp.json').write_text(json.dumps(rows,indent=2)+'\n')
