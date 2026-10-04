from pathlib import Path
import importlib,pytest,json,time
from cooperative_global_positions import get_cooperative_kernel
r=Path(__file__).resolve().parents[1]
m=importlib.import_module('tileops.kernels.moe.reduce_fused');original=m.get_reduce_fused_kernel

def factory(hidden,k,in_dtype,out_dtype,sf,weights,xsf,**kwargs):
    if hidden>=1024 and hidden%1024==0 and str(in_dtype)=='bfloat16':
        return get_cooperative_kernel(hidden,k,in_dtype,out_dtype,sf,weights,xsf,tile_hidden=1024,num_threads=512,cooperative=True)
    return original(hidden,k,in_dtype,out_dtype,sf,weights,xsf,**kwargs)
m.get_reduce_fused_kernel=factory
start=time.time();code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness_best.xml')])
(r/'meta/validation_best.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-start},indent=2)+'\n');raise SystemExit(code)
