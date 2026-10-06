from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[1];repo=r.parents[1];results=[]
for variant in ['fp8','quantized']:
 (r/'meta/tiny_status.json').write_text(json.dumps({'running':variant,'completed':results},indent=2)+'\n')
 started=time.time()
 with(r/f'logs/{variant}_tiny.log').open('w')as log:
  code=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/benchmark.py'),'--variant',variant,'--workload','tiny'],cwd=repo,stdout=log,stderr=subprocess.STDOUT).returncode
 results.append({'variant':variant,'exitcode':code,'seconds':time.time()-started})
 if code:
  (r/'meta/tiny_status.json').write_text(json.dumps({'failed':variant,'completed':results},indent=2)+'\n');raise SystemExit(code)
(r/'meta/tiny_status.json').write_text(json.dumps({'done':True,'completed':results},indent=2)+'\n')
