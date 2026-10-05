from pathlib import Path
import subprocess,json,time,csv,os
r=Path(__file__).resolve().parents[1];state={'stage':'start','started':time.time(),'completed':[]}
for workload in ('tiny','h3072','h7168','prefill'):
 for variant in ('base','xsf','fp8','quantized'):
  label=variant+'_'+workload;state['stage']=label
  with (r/f'logs/benchmark_{label}.log').open('w')as log:
   p=subprocess.Popen(['/opt/conda/bin/python','-u',str(r/'scripts/benchmark_hidden_fragments.py'),'--workload',workload,'--variant',variant],cwd='/data/TileOPs-Metax',stdout=log,stderr=subprocess.STDOUT)
   state['pid']=p.pid;(r/'meta/benchmark_all_status.json').write_text(json.dumps(state,indent=2)+'\n');code=p.wait()
  if code:state['stage']='failed';state['failed_case']=label;state['exitcode']=code;break
  state['completed'].append(label);(r/'meta/benchmark_all_status.json').write_text(json.dumps(state,indent=2)+'\n');print('DONE',label,flush=True)
 else:continue
 break
else:state['stage']='done'
state['finished']=time.time();(r/'meta/benchmark_all_status.json').write_text(json.dumps(state,indent=2)+'\n')
rows=[]
for label in state['completed']:
 with (r/f'raw/benchmark_{label}_16g.csv').open()as f:rows.extend(csv.DictReader(f))
if rows:
 with (r/'raw/benchmark_all_16g.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
