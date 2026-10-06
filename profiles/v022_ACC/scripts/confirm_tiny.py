from pathlib import Path
import json,time,runpy,sys,shutil,gc,subprocess
r=Path(__file__).resolve().parents[1];deadline=time.time()+240
while not(r/'meta/confirmation_status.json').exists() or not json.loads((r/'meta/confirmation_status.json').read_text()).get('done'):
 if time.time()>deadline:raise SystemExit('Prefill confirmation did not finish')
 time.sleep(2)
(r/'meta/gpu_before_tiny_confirmation.txt').write_text(subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,timeout=15).stdout)
completed=[]
sys.path.insert(0,str(r/'scripts'))
for v in ['fp8','quantized']:
 for suffix in ['.csv','_progress.json']:shutil.copy2(r/f'raw/{v}_tiny{suffix}',r/f'raw/first_{v}_tiny{suffix}')
 (r/'meta/tiny_confirmation_status.json').write_text(json.dumps({'running':v,'completed':completed},indent=2)+'\n')
 sys.argv=[str(r/'scripts/benchmark.py'),'--variant',v,'--workload','tiny']
 result=runpy.run_path(str(r/'scripts/benchmark.py'),run_name='__main__')
 for suffix in ['.csv','_progress.json']:
  shutil.copy2(r/f'raw/{v}_tiny{suffix}',r/f'raw/confirmation_{v}_tiny{suffix}')
  shutil.copy2(r/f'raw/first_{v}_tiny{suffix}',r/f'raw/{v}_tiny{suffix}')
 del result;gc.collect();completed.append({'variant':v,'exitcode':0})
(r/'meta/tiny_confirmation_status.json').write_text(json.dumps({'done':True,'completed':completed},indent=2)+'\n')
