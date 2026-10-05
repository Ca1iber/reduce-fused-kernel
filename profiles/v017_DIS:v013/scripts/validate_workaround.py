from pathlib import Path
import importlib,pytest,json,time
from probe_forcearrive import get_pipeline_kernel
r=Path(__file__).resolve().parents[1]
m=importlib.import_module('tileops.kernels.moe.reduce_fused')
def factory(hidden,k,ind,outd,sf,weights,xsf,**kwargs):
 tile=512 if hidden%512==0 else 256
 return get_pipeline_kernel(hidden,k,ind,outd,sf,weights,xsf,tile_hidden=tile,num_threads=128,stages=2,asynchronous=True,sync_mode=0)
m.get_reduce_fused_kernel=factory
start=time.time();code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness_workaround.xml')])
(r/'meta/validation.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-start},indent=2)+'\n');raise SystemExit(code)
