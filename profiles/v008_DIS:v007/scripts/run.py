from pathlib import Path
import subprocess, json, time
root=Path(__file__).resolve().parents[1]
status={'started':time.time(),'stage':'sweep'}
def save(): (root/'meta/status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
commands=[('sweep',['sweep.py'])]+[(f'validate_threads_{threads}_tile_{tile}',['validate.py',str(tile),str(threads)]) for threads in (64,128,256) for tile in (512,1024)]
for stage,args in commands:
    status['stage']=stage;save()
    with (root/f'logs/{stage}.log').open('w') as log:
        p=subprocess.run(['/opt/conda/bin/python','-u',str(root/'scripts'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
    status[stage+'_exitcode']=p.returncode;save()
    if p.returncode:
        status['stage']='failed';save();raise SystemExit(p.returncode)
status['stage']='complete';status['finished']=time.time();save()
