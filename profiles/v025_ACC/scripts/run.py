from pathlib import Path
import sys,json,runpy,torch,gc,ctypes,contextlib,time,csv
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'scripts'));torch.set_num_threads(1);libc=ctypes.CDLL(None);completed=[]
for w in ['h3072','h7168']:
 for v in ['fp8','quantized']:
  name=v+'_'+w;(r/'meta/status.json').write_text(json.dumps({'running':name,'completed':completed},indent=2)+'\n');started=time.time();sys.argv=[str(r/'scripts/benchmark.py'),'--variant',v,'--workload',w]
  with(r/f'logs/{name}.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=runpy.run_path(str(r/'scripts/benchmark.py'),run_name='__main__')
  del result;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0);completed.append({'task':name,'exitcode':0,'seconds':time.time()-started});print('DONE',name,flush=True)
(r/'meta/status.json').write_text(json.dumps({'running':'correctness','completed':completed},indent=2)+'\n')
with(r/'logs/correctness.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 try:runpy.run_path(str(r/'scripts/validate.py'),run_name='__main__')
 except SystemExit as e:code=e.code
if code:raise SystemExit(code)
rows=[]
for v in ['fp8','quantized']:
 for w in ['h3072','h7168']:rows.extend(csv.DictReader((r/f'raw/{v}_{w}.csv').open()))
with(r/'raw/comparison_16g.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
(r/'meta/status.json').write_text(json.dumps({'done':True,'completed':completed,'correctness_exitcode':0},indent=2)+'\n');print('DONE_ALL',flush=True)
