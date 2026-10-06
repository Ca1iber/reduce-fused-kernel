from pathlib import Path
import importlib,json,time,pytest
from grid_kernel import get_reduce_fused_kernel
r=Path(__file__).resolve().parents[1]
m=importlib.import_module('tileops.kernels.moe.reduce_fused')
m.get_reduce_fused_kernel=get_reduce_fused_kernel
start=time.time();code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness.xml')])
(r/'meta/validation.json').write_text(json.dumps({'exitcode':int(code),'seconds':time.time()-start},indent=2)+'\n')
raise SystemExit(code)
