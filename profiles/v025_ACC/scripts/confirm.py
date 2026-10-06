from pathlib import Path
import sys,time,json,runpy,gc,torch,ctypes,contextlib,shutil,csv
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'scripts'));deadline=time.time()+240
while True:
 status=json.loads((r/'meta/validation.json').read_text())
 if 'candidate_sf_calls' in status:
  assert status['exitcode']==0 and status['candidate_sf_calls']>0;break
 if time.time()>deadline:raise SystemExit('Broad SF validation did not complete')
 time.sleep(2)
torch.set_num_threads(1);libc=ctypes.CDLL(None);completed=[];rows=[]
for w in ['h3072','h7168']:
 for v in ['fp8','quantized']:
  name=v+'_'+w
  for suffix in ['.csv','_progress.json']:shutil.copy2(r/f'raw/{name}{suffix}',r/f'raw/first_{name}{suffix}')
  (r/'meta/confirmation_status.json').write_text(json.dumps({'running':name,'completed':completed},indent=2)+'\n')
  sys.argv=[str(r/'scripts/benchmark.py'),'--variant',v,'--workload',w]
  with(r/f'logs/confirmation_{name}.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=runpy.run_path(str(r/'scripts/benchmark.py'),run_name='__main__')
  for suffix in ['.csv','_progress.json']:
   shutil.copy2(r/f'raw/{name}{suffix}',r/f'raw/confirmation_{name}{suffix}')
   shutil.copy2(r/f'raw/first_{name}{suffix}',r/f'raw/{name}{suffix}')
  rows.extend(csv.DictReader((r/f'raw/confirmation_{name}.csv').open()))
  del result;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
  completed.append({'task':name,'exitcode':0});print('CONFIRMED',name,flush=True)
with(r/'raw/confirmation_comparison_16g.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(r/'meta/confirmation_status.json').write_text(json.dumps({'done':True,'completed':completed},indent=2)+'\n');print('CONFIRMATION_DONE',flush=True)
