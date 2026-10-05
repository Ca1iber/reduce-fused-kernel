from pathlib import Path
import subprocess,time,json,csv
root=Path(__file__).resolve().parents[1]
status={'stage':'sweep','started':time.time(),'one_process_per_workload':True}
def save():(root/'meta/status.json').write_text(json.dumps(status,indent=2)+'\n')
def run(stage,args):
    status['stage']=stage;save()
    with (root/f'logs/{stage}.log').open('w') as log:
        p=subprocess.run(['/opt/conda/bin/python','-u',str(root/'scripts'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
    status[stage+'_exitcode']=p.returncode;save()
    if p.returncode:
        status['stage']='failed';save();raise SystemExit(p.returncode)
save()
for variant in ('base','xsf'):
    for workload in ('tiny','h3072','h7168','prefill'):
        path=root/'raw/base_xsf_parallel_sweep_16g.csv'
        previous=list(csv.DictReader(path.open())) if path.exists() else []
        if any(r['variant']==variant and r['workload']==workload for r in previous):continue
        run(f'sweep_{variant}_{workload}',['sweep.py','--variant',variant,'--workload',workload,'--resume'])
rows=list(csv.DictReader((root/'raw/base_xsf_parallel_sweep_16g.csv').open()))
selected=set()
for variant in ('base','xsf'):
    for workload in ('tiny','h3072','h7168','prefill'):
        group=[r for r in rows if r['variant']==variant and r['workload']==workload and r['mode']!='baseline']
        r=min(group,key=lambda r:float(r['latency_us']))
        selected.add((r['mode'],int(r['threads']),int(r['tile'])))
r=max((r for r in rows if r['mode']=='split'),key=lambda r:float(r['speedup_vs_baseline']))
selected.add((r['mode'],int(r['threads']),int(r['tile'])))
for mode,threads,tile in sorted(selected):
    run(f'validate_{mode}_threads{threads}_tile{tile}',['validate_best.py',mode,str(threads),str(tile)])
status['stage']='complete';status['finished']=time.time();save()
