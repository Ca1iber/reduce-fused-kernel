"""One kernel in a native mcProfiler ROI."""
import argparse
from pathlib import Path
import torch
from tileops.ops.moe import MoeReduceFusedFp8FwdOp
from workloads.moe import MoeReduceFusedWorkload
from candidate_fp8_bits import get_candidate
parser=argparse.ArgumentParser()
parser.add_argument('--implementation',choices=('baseline','candidate'),required=True)
args=parser.parse_args()
torch.manual_seed(1235)
x,pos,weights,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),dtype=torch.float8_e4m3fn,device='cuda')
unused=torch.empty((x.shape[0],),dtype=torch.float32,device='cuda')
kernel=(MoeReduceFusedFp8FwdOp(512,8,7168,torch.bfloat16).kernel.kernel if args.implementation=='baseline'
        else get_candidate(7168,8,'bfloat16','float8_e4m3fn',True,True,False))
values=(x,weights,pos,out,sf,unused)
for _ in range(10):kernel(*values)
torch.cuda.synchronize()
torch.cuda.profiler.start()
kernel(*values)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print('PROFILE_COMPLETED',args.implementation,flush=True)
