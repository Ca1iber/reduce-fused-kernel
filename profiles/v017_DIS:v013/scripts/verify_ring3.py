from pathlib import Path
import torch,json
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from probe_forcearrive import get_pipeline_kernel
r=Path(__file__).resolve().parents[1];rows=[]
for variant,xsf,sf in [('base',False,False),('xsf',True,False),('fp8',False,True),('quantized',True,True)]:
 torch.manual_seed(1235);values=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_x_sf=xsf,with_sf=sf).gen_inputs()
 x,pos,w=values[:3];scale=values[-1] if sf else torch.empty(1,device='cuda');xs=values[3]if xsf else torch.empty(x.shape[0],device='cuda');dtype='float8_e4m3fn'if sf else 'bfloat16';out=torch.empty((512,7168),device='cuda',dtype=getattr(torch,dtype))
 base=get_reduce_fused_kernel(7168,8,'bfloat16',dtype,sf,True,xsf,tile_hidden=512,num_threads=128)
 k=get_pipeline_kernel(7168,8,'bfloat16',dtype,sf,True,xsf,tile_hidden=512,num_threads=128,stages=3,asynchronous=True,sync_mode=0)
 for invalid in (False,True):
  if invalid:pos[::3,::2]=-1
  base(x,w,pos,out,scale,xs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone();counts=[]
  for rep in range(5):k(x,w,pos,out,scale,xs);torch.cuda.synchronize();counts.append(int((out.view(torch.uint8)!=expected).sum()))
  rows.append(dict(variant=variant,invalid_routes=invalid,wrong_bytes=counts));print(rows[-1],flush=True)
 (r/'raw/ring3_workaround.json').write_text(json.dumps(rows,indent=2)+'\n')
