from pathlib import Path
import importlib, sys, json, time
import pytest
from factory import factory
root=Path(__file__).resolve().parents[1]
tile=int(sys.argv[1])
module=importlib.import_module('tileops.kernels.moe.reduce_fused')
module.get_reduce_fused_kernel=factory(tile)
start=time.time()
code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(root/f'raw/correctness_tile_{tile}.xml')])
(root/f'meta/validation_{tile}.json').write_text(json.dumps(dict(tile=tile,exitcode=int(code),seconds=time.time()-start),indent=2)+'\n')
raise SystemExit(code)
