from pathlib import Path
import torch,json,time,traceback,shutil
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from pipeline_probe import get_pipeline_kernel
r=Path(__file__).resolve().parents[1];rows=[]
for k in (1,2,8):
 for sf in (False,True):
  t,h=512,7168;torch.manual_seed(1235)
  values=MoeReduceFusedWorkload(t,k,h,torch.bfloat16,with_sf=sf).gen_inputs()
  x,pos,w=values[:3];scale=values[-1] if sf else torch.empty(1,device='cuda');unused=torch.empty(x.shape[0],device='cuda')
  dtype='float8_e4m3fn' if sf else 'float32';out=torch.empty((t,h),device='cuda',dtype=getattr(torch,dtype))
  base=get_reduce_fused_kernel(h,k,'bfloat16',dtype,sf,True,False,tile_hidden=512,num_threads=128)
  base(x,w,pos,out,scale,unused);torch.cuda.synchronize();expected=out.clone();expected_bytes=expected.view(torch.uint8).clone()
  for stages in (2,3):
   for mode in (2,0,1,3):
    label=f'k{k}_sf{int(sf)}_s{stages}_mode{mode}'
    try:
     kernel=get_pipeline_kernel(h,k,'bfloat16',dtype,sf,True,False,tile_hidden=512,num_threads=128,stages=stages,asynchronous=True,sync_mode=mode)
     (r/f'codegen/{label}.cu').write_text(kernel.get_kernel_source())
     result=[]
     for n in range(3):
      out.zero_();kernel(x,w,pos,out,scale,unused);torch.cuda.synchronize()
      neq=out.view(torch.uint8)!=expected_bytes;index=neq.nonzero()
      result.append({'wrong_bytes':int(neq.sum()),'max_abs':float((out.float()-expected.float()).abs().max()),'first_indices':index[:6].tolist()})
     row={'label':label,'results':result,'adapter_type':str(type(kernel.adapter)),'adapter_fields':list(vars(kernel.adapter))}
    except Exception as e:traceback.print_exc();row={'label':label,'error':repr(e)}
    rows.append(row);(r/'raw/reproduce.json').write_text(json.dumps(rows,indent=2)+'\n');print('RESULT',row,flush=True)
(r/'meta/reproduce_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
