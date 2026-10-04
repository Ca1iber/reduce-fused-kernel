from pathlib import Path
import subprocess,json,time
root=Path(__file__).resolve().parents[1];status={'stage':'hardware','started':time.time()}
def save():(root/'meta/status.json').write_text(json.dumps(status,indent=2)+'\n')
def run(stage,command,cwd=None):
    status['stage']=stage;save()
    with (root/f'logs/{stage}.log').open('w') as log:
        p=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    status[stage+'_exitcode']=p.returncode;save();return p.returncode
save();run('hardware',['mx-smi','--summary'])
for variant in ('base','xsf','fp8','quantized'):
    code=run('bench_'+variant,['/opt/conda/bin/python','-u',str(root/'scripts/baseline_driver.py'),'--variant',variant,'--mode','bench'])
    if code:status['stage']='failed';save();raise SystemExit(code)
trace=root/'raw/tracer_baseline_fp8';trace.mkdir(exist_ok=True)
run('tracer',['/opt/maca/bin/mcTracer','--mctx','--odname','tracer_baseline_fp8','--name','baseline_fp8','/opt/conda/bin/python',str(root/'scripts/baseline_driver.py'),'--variant','fp8','--mode','profile'],cwd=root/'raw')
run('native',['/opt/conda/bin/python','-u',str(root/'scripts/collect_native.py')])
run('async_probe',['/opt/conda/bin/python','-u',str(root/'scripts/async_probe.py')])
status['stage']='audit_done';status['finished']=time.time();save()
