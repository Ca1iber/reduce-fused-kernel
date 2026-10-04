from pathlib import Path
import subprocess,time,json
root=Path(__file__).resolve().parents[1]
status={'stage':'waiting_main','started':time.time()}
def save(): (root/'meta/extra_status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
while True:
    main=json.loads((root/'meta/status.json').read_text())
    if main['stage']=='complete':break
    if main['stage']=='failed' or time.time()-status['started']>900:
        status['stage']='failed';status['reason']='main failed or wait timed out';save();raise SystemExit(1)
    time.sleep(2)
for stage,args in [('sweep_extra',['sweep_extra.py']),('validate_threads_64_tile_256',['validate.py','256','64'])]:
    status['stage']=stage;save()
    with (root/f'logs/{stage}.log').open('w') as log:
        p=subprocess.run(['/opt/conda/bin/python','-u',str(root/'scripts'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
    status[stage+'_exitcode']=p.returncode;save()
    if p.returncode:
        status['stage']='failed';save();raise SystemExit(p.returncode)
status['stage']='complete';status['finished']=time.time();save()
