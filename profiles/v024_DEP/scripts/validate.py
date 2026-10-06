from pathlib import Path
import pytest,importlib,json,time
from store128 import get_reduce_fused_kernel
r=Path(__file__).resolve().parents[1];m=importlib.import_module('tileops.kernels.moe.reduce_fused');original=m.get_reduce_fused_kernel
def factory(h,k,ind,outd,sf,weights,xsf,**kwargs):
 if not sf:return original(h,k,ind,outd,sf,weights,xsf,**kwargs)
 kwargs['tile_hidden']=1024 if h%1024==0 else 512 if h%512==0 else 256
 kwargs['num_threads']=64;kwargs['vector_store']=True
 return get_reduce_fused_kernel(h,k,ind,outd,sf,weights,xsf,**kwargs)
m.get_reduce_fused_kernel=factory
started=time.time();code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness.xml')])
(r/'meta/validation.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-started},indent=2)+'\n');raise SystemExit(code)
