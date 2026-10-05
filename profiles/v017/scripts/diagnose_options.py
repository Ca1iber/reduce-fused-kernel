from pathlib import Path
import torch,json,importlib,traceback
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
r=Path(__file__).resolve().parents[1];rows=[]
for topk in (2,8):
 torch.manual_seed(1235);values=MoeReduceFusedWorkload(512,topk,7168,torch.bfloat16).gen_inputs()
 x,pos,w=values[:3];out=torch.empty((512,7168),device='cuda',dtype=torch.float32);sf=torch.empty(1,device='cuda');unused=torch.empty(x.shape[0],device='cuda')
 base=get_reduce_fused_kernel(7168,topk,'bfloat16','float32',False,True,False,tile_hidden=512,num_threads=128);base(x,w,pos,out,sf,unused);torch.cuda.synchronize();expected=out.clone()
 for name in ('sharedbarrier','forcezero','forcearrive','restoremd','storemd','O0','forcebsm','forcegvm'):
  try:
   mod=importlib.import_module('pipeline_probe' if name=='sharedbarrier' else 'probe_'+name)
   kernel=mod.get_pipeline_kernel(7168,topk,'bfloat16','float32',False,True,False,tile_hidden=512,num_threads=128,stages=2,asynchronous=True,sync_mode=4 if name=='sharedbarrier' else 0)
   (r/f'codegen/k{topk}_{name}.cu').write_text(kernel.get_kernel_source())
   results=[]
   for n in range(5):
    out.zero_();kernel(x,w,pos,out,sf,unused);torch.cuda.synchronize();neq=out!=expected
    results.append({'wrong_elements':int(neq.sum()),'max_abs':float((out-expected).abs().max()),'first_actual':out.flatten()[:8].tolist(),'first_expected':expected.flatten()[:8].tolist()})
   row={'k':topk,'option':name,'results':results}
  except Exception as e:traceback.print_exc();row={'k':topk,'option':name,'error':repr(e)}
  rows.append(row);(r/'raw/options_results.json').write_text(json.dumps(rows,indent=2)+'\n');print('RESULT',topk,name,[x['wrong_elements']for x in row.get('results',[])],row.get('error'),flush=True)
(r/'meta/options_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
