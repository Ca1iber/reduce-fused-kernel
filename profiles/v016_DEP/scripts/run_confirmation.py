from pathlib import Path
import subprocess,time,json
r=Path(__file__).resolve().parents[1]
status={'stage':'start'}
for label in ('h3072','h7168'):
 status['stage']=label;(r/'meta/confirmation_status.json').write_text(json.dumps(status,indent=2)+'\n')
 with (r/f'logs/confirmation_{label}.log').open('w') as log:
    p=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/confirm_workload.py'),'--workload',label],stdout=log,stderr=subprocess.STDOUT)
 status[label+'_exitcode']=p.returncode
 if p.returncode:status['stage']='failed';break
else:status['stage']='done'
(r/'meta/confirmation_status.json').write_text(json.dumps(status,indent=2)+'\n')
