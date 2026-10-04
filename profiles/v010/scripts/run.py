from pathlib import Path
import subprocess,time,json
root=Path(__file__).resolve().parents[1]
status={'stage':'starting','started':time.time()}
def save():(root/'meta/status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
commands=[('dispatch',[str(root/'scripts/check_dispatch.py')]),('correctness',['-m','pytest','-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(root/'raw/correctness.xml')])]
commands += [('paired_'+label,[str(root/'scripts/paired_selected.py'),'--workload',label]) for label in ('tiny','h3072','h7168','prefill')]
commands += [('benchmark',['-m','pytest','-q','benchmarks/ops/bench_moe_reduce_fused.py','--junitxml='+str(root/'raw/benchmark.xml')])]
for stage,args in commands:
    status['stage']=stage;save()
    with (root/f'logs/{stage}.log').open('w') as log:
        p=subprocess.run(['/opt/conda/bin/python','-u',*args],stdout=log,stderr=subprocess.STDOUT)
    status[stage+'_exitcode']=p.returncode;save()
    if p.returncode:
        status['stage']='failed';save();raise SystemExit(p.returncode)
status['stage']='complete';status['finished']=time.time();save()
