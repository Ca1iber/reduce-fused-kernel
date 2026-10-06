from pathlib import Path
import json,time,csv,sys,runpy,gc,ctypes,contextlib,torch,pytest
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
completed=[z for z in json.loads((r/'meta/integration_oom_status.json').read_text())['completed']if z['exitcode']==0]
sys.path.insert(0,str(r/'scripts'));torch.set_num_threads(1);libc=ctypes.CDLL(None)
for w in ['prefill','tiny','h3072','h7168']:
 for v in ['fp8','quantized']:
  name=v+'_'+w
  if any(z['task']==name for z in completed):continue
  (r/'meta/integration_status.json').write_text(json.dumps({'running':name,'completed':completed},indent=2)+'\n')
  started=time.time();sys.argv=[str(r/'scripts/benchmark_integrated.py'),'--variant',v,'--workload',w]
  with(r/f'logs/integration_{name}.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
   result=runpy.run_path(str(r/'scripts/benchmark_integrated.py'),run_name='__main__')
  del result;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
  completed.append({'task':name,'exitcode':0,'seconds':time.time()-started});print('FINISHED',name,flush=True)
(r/'meta/integration_status.json').write_text(json.dumps({'running':'correctness','completed':completed},indent=2)+'\n')
started=time.time()
with(r/'logs/integration_correctness.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/integrated_correctness.xml')])
completed.append({'task':'correctness','exitcode':int(code),'seconds':time.time()-started})
if code:
 (r/'meta/integration_status.json').write_text(json.dumps({'failed':'correctness','completed':completed},indent=2)+'\n');raise SystemExit(code)
rows=[]
for v in ['fp8','quantized']:
 for w in ['tiny','h3072','h7168','prefill']:rows.extend(csv.DictReader((r/f'raw/integrated_{v}_{w}.csv').open()))
with(r/'raw/integrated_comparison_16g.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(r/'meta/integration_status.json').write_text(json.dumps({'done':True,'completed':completed},indent=2)+'\n');print('INTEGRATION_VALIDATED',flush=True)
