from pathlib import Path
import subprocess,json,time,csv,os
r=Path(__file__).resolve().parents[1];repo=r.parents[1];completed=[]
tasks=[('dispatch',['check_integration.py'])]+[(v+'_'+w,['benchmark_integrated.py','--variant',v,'--workload',w])for w in ['prefill','tiny','h3072','h7168']for v in ['fp8','quantized']]+[('correctness',None)]
for name,args in tasks:
 (r/'meta/integration_status.json').write_text(json.dumps({'running':name,'completed':completed},indent=2)+'\n')
 command=['/opt/conda/bin/python','-u',str(r/'scripts'/args[0]),*args[1:]]if args else['/opt/conda/bin/python','-m','pytest','-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/integrated_correctness.xml')]
 started=time.time()
 with(r/f'logs/integration_{name}.log').open('w')as log:
  code=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT).returncode
 completed.append({'task':name,'exitcode':code,'seconds':time.time()-started});print('FINISHED',name,code,flush=True)
 if code:
  (r/'meta/integration_status.json').write_text(json.dumps({'failed':name,'completed':completed},indent=2)+'\n');raise SystemExit(code)
rows=[]
for v in ['fp8','quantized']:
 for w in ['tiny','h3072','h7168','prefill']:rows.extend(csv.DictReader((r/f'raw/integrated_{v}_{w}.csv').open()))
with(r/'raw/integrated_comparison_16g.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(r/'meta/integration_status.json').write_text(json.dumps({'done':True,'completed':completed},indent=2)+'\n')
