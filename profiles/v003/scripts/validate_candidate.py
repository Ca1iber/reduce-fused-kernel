"""Run existing correctness tests with the candidate selected only for FP8."""
from pathlib import Path
import importlib,json,sys,time
import pytest
import torch
from candidate_fp8_bits import get_candidate
root=Path('/root/TileOPs-Metax/profiles/v003')
module=importlib.import_module('tileops.kernels.moe.reduce_fused')
original=module.get_reduce_fused_kernel

def selected(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf):
    fn=get_candidate if with_sf else original
    return fn(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf)

# Compare encoded bytes against the original SDK conversion at all E4M3
# midpoints, adjacent FP32 values, special values, and random FP32 bit patterns.
positive=torch.arange(127,dtype=torch.uint8).view(torch.float8_e4m3fn).float()
mid=(positive[:-1]+positive[1:])*0.5
edges=torch.cat((positive,mid,torch.nextafter(mid,torch.full_like(mid,float('inf'))),
                 torch.nextafter(mid,torch.full_like(mid,-float('inf'))),
                 torch.tensor([464.,448.,480.,0.,-0.,float('inf'),float('nan')]))).to('cuda')
edges=torch.cat((edges,-edges))
torch.manual_seed(1235)
random=torch.randint(0,1<<32,(32768,),device='cuda',dtype=torch.int64).to(torch.int32).view(torch.float32)
values=torch.cat((edges,random))
padding=(-values.numel())%256
values=torch.cat((values,torch.zeros(padding,device='cuda')))
x=values.reshape(-1,256).contiguous()
tokens=x.shape[0]
pos=torch.arange(tokens,device='cuda',dtype=torch.int32).reshape(tokens,1)
weights=torch.ones((tokens,1),device='cuda')
unused=torch.ones(tokens,device='cuda')
old_out=torch.empty_like(x,dtype=torch.float8_e4m3fn)
new_out=torch.empty_like(old_out)
old=original(256,1,'float32','float8_e4m3fn',True,True,False)
new=get_candidate(256,1,'float32','float8_e4m3fn',True,True,False)
for scalar in (1.,-1.,0.75,0.,float('inf'),float('nan')):
    sf=torch.tensor([scalar],device='cuda')
    old(x,weights,pos,old_out,sf,unused)
    new(x,weights,pos,new_out,sf,unused)
    torch.cuda.synchronize()
    mismatch=old_out.view(torch.uint8)!=new_out.view(torch.uint8)
    count=int(mismatch.sum().item())
    print('CONVERSION_BYTE_CHECK',scalar,'elements',values.numel(),'mismatches',count,flush=True)
    if count:
        index=mismatch.flatten().nonzero()[0].item()
        print('FIRST_MISMATCH',index,float(values[index].item()),
              int(old_out.view(torch.uint8).flatten()[index].item()),
              int(new_out.view(torch.uint8).flatten()[index].item()),flush=True)
        (root/'meta/validation.json').write_text(json.dumps(dict(status='failed',gate='conversion byte comparison',mismatches=count),indent=2)+'\n')
        raise SystemExit(1)
module.get_reduce_fused_kernel=selected
started=time.time()
code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py',
                  '--junitxml='+str(root/'raw/correctness.xml')])
(root/'meta/validation.json').write_text(json.dumps(dict(status='passed' if code==0 else 'failed',
    pytest_exitcode=int(code),conversion_elements_per_scale=values.numel(),conversion_scale_cases=6,
    seconds=time.time()-started,scope='131 existing correctness cases; candidate only for with_sf paths'),indent=2)+'\n')
raise SystemExit(code)
