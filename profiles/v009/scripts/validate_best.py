from pathlib import Path
import importlib,csv,json,time,sys
import pytest
from factory import factory
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'raw/base_xsf_parallel_sweep_16g.csv').open()))
selected=set()
for variant in ('base','xsf'):
    for workload in ('tiny','h3072','h7168','prefill'):
        group=[r for r in rows if r['variant']==variant and r['workload']==workload and r['mode']!='baseline']
        row=min(group,key=lambda r:float(r['latency_us']))
        selected.add((row['mode'],int(row['threads']),int(row['tile'])))
# Validate the best split even when a full-row candidate is faster.
split_rows=[r for r in rows if r['mode']=='split']
split_best=max(split_rows,key=lambda r:float(r['speedup_vs_baseline']))
selected.add((split_best['mode'],int(split_best['threads']),int(split_best['tile'])))
configs=sorted(selected)
(root/'meta/validation_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
module=importlib.import_module('tileops.kernels.moe.reduce_fused')
if len(sys.argv)>1:configs=[(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))]
for mode,threads,tile in configs:
    print('VALIDATE',mode,threads,tile,flush=True)
    module.get_reduce_fused_kernel=factory(mode,threads,tile)
    label=f'{mode}_threads{threads}_tile{tile}'
    started=time.time()
    code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(root/f'raw/correctness_{label}.xml')])
    (root/f'meta/validation_{label}.json').write_text(json.dumps(dict(mode=mode,threads=threads,tile=tile,exitcode=int(code),seconds=time.time()-started),indent=2)+'\n')
    if code:raise SystemExit(code)
print('VALIDATION_COMPLETE',configs,flush=True)
