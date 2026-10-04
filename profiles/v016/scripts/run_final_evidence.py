from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[2]
s={'stage':'confirmation','started':time.time()}
p=r/'v016/meta/final_evidence.json'
for label,script in [('confirmation',r/'v016/scripts/run_confirmation.py'),('hidden_profile',r/'v015/scripts/collect_profiles.py')]:
 s['stage']=label;p.write_text(json.dumps(s,indent=2)+'\n')
 with (r/f'v016/logs/final_{label}.log').open('w') as log:
  c=subprocess.Popen(['/opt/conda/bin/python',str(script)],stdout=log,stderr=subprocess.STDOUT)
  s['child_pid']=c.pid;p.write_text(json.dumps(s,indent=2)+'\n');code=c.wait()
 s[label+'_exitcode']=code
 if code:s['stage']='failed';break
else:s['stage']='done'
s['finished']=time.time();p.write_text(json.dumps(s,indent=2)+'\n')
