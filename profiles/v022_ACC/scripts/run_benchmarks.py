from pathlib import Path
import subprocess,json,time,csv,os
r=Path(__file__).resolve().parents[1];repo=r.parents[1];completed=[]
compile_pid=json.loads((r/'meta/compile_job.json').read_text())['pid']
while not(r/'meta/static_status.json').exists():
 try:os.kill(compile_pid,0)
 except ProcessLookupError:raise SystemExit('Static compilation failed; inspect compile_candidates.log')
 time.sleep(2)
assert json.loads((r/'meta/static_status.json').read_text())['done']
tasks=[(v+'_'+w,['benchmark.py','--variant',v,'--workload',w])for w in ['prefill','h7168','h3072','tiny']for v in ['fp8','quantized']]+[('correctness',['validate.py'])]
for name,args in tasks:
 if any(z['task']==name for z in completed):continue
 (r/'meta/benchmark_status.json').write_text(json.dumps({'running':name,'completed':completed},indent=2)+'\n')
 started=time.time()
 with(r/f'logs/{name}.log').open('w')as log:
  code=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts'/args[0]),*args[1:]],cwd=repo,stdout=log,stderr=subprocess.STDOUT).returncode
 completed.append({'task':name,'exitcode':code,'seconds':time.time()-started})
 if code:
  (r/'meta/benchmark_status.json').write_text(json.dumps({'failed':name,'completed':completed},indent=2)+'\n');raise SystemExit(code)
 print('FINISHED',name,flush=True)
rows=[]
for v in ['fp8','quantized']:
 for w in ['tiny','h3072','h7168','prefill']:rows.extend(csv.DictReader((r/f'raw/{v}_{w}.csv').open()))
with(r/'raw/comparison_16g.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(r/'meta/benchmark_status.json').write_text(json.dumps({'done':True,'completed':completed},indent=2)+'\n')
