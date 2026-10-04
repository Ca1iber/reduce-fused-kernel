from pathlib import Path
import importlib,sys,json,time
import pytest
from pipeline_kernel import get_pipeline_kernel
root=Path(__file__).resolve().parents[1]
stages=int(sys.argv[1])
module=importlib.import_module('tileops.kernels.moe.reduce_fused')
def factory(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf,**kwargs):
    tile=512 if hidden%512==0 else 256
    return get_pipeline_kernel(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf,tile_hidden=tile,num_threads=128,stages=stages,asynchronous=True,old_two_barriers=False)
module.get_reduce_fused_kernel=factory
start=time.time()
code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(root/f'raw/correctness_async_ring{stages}.xml')])
(root/f'meta/validation_ring{stages}.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-start},indent=2)+'\n')
raise SystemExit(code)
