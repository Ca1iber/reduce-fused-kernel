from pathlib import Path
import json,torch
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
root=Path(__file__).resolve().parents[1]
def config_for(tokens,k,h,dtype,sf,weights=True):
    obj=object.__new__(MoeReduceFusedKernel)
    obj.num_tokens=tokens;obj.num_topk=k;obj.hidden=h;obj.dtype=dtype;obj.with_sf=sf;obj.with_weights=weights
    return obj.default_config
assert config_for(513,8,7168,torch.bfloat16,False)==dict(tile_hidden=7168,num_threads=128)
assert config_for(513,8,7168,torch.bfloat16,True)==dict(tile_hidden=1024,num_threads=128)
assert config_for(512,8,7168,torch.float32,False)==dict(tile_hidden=7168,num_threads=128)
assert config_for(512,8,7168,torch.bfloat16,False,False)==dict(tile_hidden=7168,num_threads=128)
for cfg in ({'tile_hidden':0},{'tile_hidden':768},{'tile_hidden':True},{'num_threads':0},{'num_threads':128.0}):
    try:MoeReduceFusedKernel(32,2,256,torch.float16,config=cfg)
    except ValueError:pass
    else:raise AssertionError(cfg)
torch.manual_seed(1235)
x,pos,w=MoeReduceFusedWorkload(32,2,256,torch.float16).gen_inputs()
a=MoeReduceFusedKernel(32,2,256,torch.float16)
b=MoeReduceFusedKernel(32,2,256,torch.float16,config={'num_threads':64})
assert b.config==dict(tile_hidden=256,num_threads=64)
assert torch.equal(a(x,pos,w).view(torch.uint8),b(x,pos,w).view(torch.uint8))
(root/'meta/dispatch_check.json').write_text(json.dumps(dict(status='passed',fallback=True,manual_override=True,invalid_configs=True),indent=2)+'\n')
print('DISPATCH_CHECK_PASSED',flush=True)
