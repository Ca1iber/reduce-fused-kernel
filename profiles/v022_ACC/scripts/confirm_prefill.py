from pathlib import Path
import json,time,subprocess,os,shutil
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
pid=json.loads((r/'meta/benchmark_job.json').read_text())['pid'];deadline=time.time()+360
while not json.loads((r/'meta/benchmark_status.json').read_text()).get('done'):
 status=json.loads((r/'meta/benchmark_status.json').read_text())
 if status.get('failed'):raise SystemExit('Initial benchmark/test failed')
 if time.time()>deadline:raise SystemExit('Benchmark did not finish in time')
 time.sleep(2)
completed=[]
for v in ['fp8','quantized']:
 for suffix in ['.csv','_progress.json']:
  path=r/f'raw/{v}_prefill{suffix}';shutil.copy2(path,r/f'raw/first_{v}_prefill{suffix}')
 (r/'meta/confirmation_status.json').write_text(json.dumps({'running':v,'completed':completed},indent=2)+'\n')
 with(r/f'logs/confirm_{v}_prefill.log').open('w')as log:
  result=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/benchmark.py'),'--variant',v,'--workload','prefill'],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(result.returncode)
 for suffix in ['.csv','_progress.json']:shutil.copy2(r/f'raw/{v}_prefill{suffix}',r/f'raw/confirmation_{v}_prefill{suffix}')
 completed.append({'variant':v,'exitcode':result.returncode})
 (r/'meta/confirmation_status.json').write_text(json.dumps({'completed':completed},indent=2)+'\n')
# Keep the initial 8-case comparison intact; confirmation has its own files.
for v in ['fp8','quantized']:
 for suffix in ['.csv','_progress.json']:shutil.copy2(r/f'raw/first_{v}_prefill{suffix}',r/f'raw/{v}_prefill{suffix}')
with(r/'logs/resources.log').open('w')as log:
 q=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/query_resources.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
(r/'meta/confirmation_status.json').write_text(json.dumps({'done':True,'completed':completed,'resource_exitcode':q.returncode},indent=2)+'\n')
