from pathlib import Path
import subprocess,time,json
r=Path(__file__).resolve().parents[1]
while True:
 try:
    if Path('/proc/85673/stat').read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(2)
status={'stage':'start'}
for label in ('h3072','h7168','prefill'):
 status['stage']=label;(r/'meta/expanded_status.json').write_text(json.dumps(status,indent=2)+'\n')
 with (r/f'logs/expanded_{label}.log').open('w') as log:
    p=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/compare_workload.py'),'--workload',label],stdout=log,stderr=subprocess.STDOUT)
 status[label+'_exitcode']=p.returncode
 if p.returncode:status['stage']='failed';break
else:status['stage']='done'
(r/'meta/expanded_status.json').write_text(json.dumps(status,indent=2)+'\n')
