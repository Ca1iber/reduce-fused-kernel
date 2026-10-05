from pathlib import Path
import importlib,pytest,json,time
from hidden_fragment import get_hidden_kernel
r=Path(__file__).resolve().parents[1]
m=importlib.import_module('tileops.kernels.moe.reduce_fused')
def factory(h,k,ind,outd,sf,weights,xsf,**kwargs):
 tile=512 if h%512==0 else 256
 return get_hidden_kernel(h,k,ind,outd,sf,weights,xsf,tile_hidden=tile,num_threads=128,group_size=2,mode=3)
m.get_reduce_fused_kernel=factory
start=time.time();code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness_hidden_fragment.xml')])
(r/'meta/hidden_validation.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-start},indent=2)+'\n');raise SystemExit(code)
