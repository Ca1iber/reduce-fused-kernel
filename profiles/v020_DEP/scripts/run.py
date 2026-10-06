from pathlib import Path
import subprocess,json,time,os,csv
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
status=[]
tasks=[(v+'_'+w,['benchmark.py','--variant',v,'--workload',w])for w in ['tiny','h3072','h7168','prefill']for v in ['fp8','quantized']]+[('correctness',['validate.py'])]
for name,args in tasks:
 if any(z['task']==name for z in status):continue
 (r/'meta/status.json').write_text(json.dumps({'running':name,'completed':status},indent=2)+'\n')
 started=time.time()
 with (r/f'logs/{name}.log').open('w')as log:
  code=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts'/args[0]),*args[1:]],cwd=repo,stdout=log,stderr=subprocess.STDOUT).returncode
 status.append({'task':name,'exitcode':code,'seconds':time.time()-started})
 print('FINISHED',name,status[-1],flush=True)
 if code:
  (r/'meta/status.json').write_text(json.dumps({'failed':name,'completed':status},indent=2)+'\n');raise SystemExit(code)
rows=[]
for v in ['fp8','quantized']:
 for w in ['tiny','h3072','h7168','prefill']:rows.extend(csv.DictReader((r/f'raw/{v}_{w}.csv').open()))
with (r/'raw/comparison.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(r/'meta/status.json').write_text(json.dumps({'done':True,'completed':status},indent=2)+'\n')
