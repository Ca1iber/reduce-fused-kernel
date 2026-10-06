from pathlib import Path
import importlib,time,json,pytest
from prefetch_combined import get_reduce_fused_kernel
r=Path(__file__).resolve().parents[1];m=importlib.import_module('tileops.kernels.moe.reduce_fused');original=m.get_reduce_fused_kernel;candidate_calls=[]
def factory(h,k,ind,outd,sf,weights,xsf,**kwargs):
 if sf:
  kwargs['tile_hidden']=512 if h%512==0 else 256
  kwargs['num_threads']=128
  candidate_calls.append({'hidden':h,'topk':k,'dtype':str(ind),'weights':weights})
  kwargs['prefetch_rows']=8
  return get_reduce_fused_kernel(h,k,ind,outd,sf,weights,xsf,**kwargs)
 return original(h,k,ind,outd,sf,weights,xsf,**kwargs)
m.get_reduce_fused_kernel=factory
started=time.time();code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness.xml')])
(r/'meta/validation.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-started,'candidate_sf_calls':len(candidate_calls),'candidate_cases':candidate_calls},indent=2)+'\n');raise SystemExit(code)
