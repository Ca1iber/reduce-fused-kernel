import argparse,json
from pathlib import Path
import torch
from workloads.moe import MoeReduceFusedWorkload
from versions import naive_factory,v003_factory,v004_factory
from candidate_parallel import get_candidate
parser=argparse.ArgumentParser();parser.add_argument("--version",choices=("naive","v003","v004","v006","all"),required=True)
args=parser.parse_args()
root=Path("/data/TileOPs-Metax/profiles")
torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),device="cuda",dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device="cuda")
factories={"naive":naive_factory,"v003":v003_factory,"v004":v004_factory,"v006":get_candidate}
chosen=["v004","v006"] if args.version=="all" else [args.version]
for label in chosen:
    kernel=factories[label](7168,8,"bfloat16","float8_e4m3fn",True,True,False)
    destination=root/{'naive': 'v003_ACC', 'v003': 'v003_ACC', 'v004': 'v004_ACC', 'v005': 'v005_DEP', 'v006': 'v006_DEP'}[label]/"codegen"/("formal_"+label+"_h7168.cu")
    destination.write_text(kernel.get_kernel_source())
    for _ in range(10):kernel(x,w,pos,out,sf,unused)
    torch.cuda.synchronize();torch.cuda.profiler.start()
    kernel(x,w,pos,out,sf,unused)
    torch.cuda.synchronize();torch.cuda.profiler.stop()
    print("PROFILE_VERSION_COMPLETED",label,flush=True)
