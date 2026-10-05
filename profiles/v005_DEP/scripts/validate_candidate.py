from pathlib import Path
import importlib,json,time
import pytest
from candidate_parallel import get_candidate
root=Path("/data/TileOPs-Metax/profiles/v005")
module=importlib.import_module("tileops.kernels.moe.reduce_fused")
module.get_reduce_fused_kernel=get_candidate
started=time.time()
code=pytest.main(["-q","tests/ops/test_moe_reduce_fused.py","--junitxml="+str(root/"raw/correctness.xml")])
(root/"meta/validation.json").write_text(json.dumps(dict(status="passed" if code==0 else "failed",exitcode=int(code),seconds=time.time()-started),indent=2)+"\n")
raise SystemExit(code)
