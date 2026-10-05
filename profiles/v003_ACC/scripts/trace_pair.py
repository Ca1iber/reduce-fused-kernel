"""Short baseline/candidate trace, no correctness verdict."""
from pathlib import Path
import torch
from tileops.ops.moe import MoeReduceFusedFp8FwdOp
from workloads.moe import MoeReduceFusedWorkload
from candidate_fp8_bits import get_candidate
root=Path('/data/TileOPs-Metax/profiles/v003')
torch.manual_seed(1235)
x,pos,weights,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),dtype=torch.float8_e4m3fn,device='cuda')
unused=torch.empty((x.shape[0],),dtype=torch.float32,device='cuda')
base=MoeReduceFusedFp8FwdOp(512,8,7168,torch.bfloat16).kernel.kernel
candidate=get_candidate(7168,8,'bfloat16','float8_e4m3fn',True,True,False)
args=(x,weights,pos,out,sf,unused)
for label,kernel in (('baseline',base),('candidate',candidate)):
    for _ in range(10):kernel(*args)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    kernel(*args)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print('TRACE_COMPLETED',label,flush=True)
