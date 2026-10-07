from pathlib import Path
import sys,json,csv,statistics,importlib.util,traceback,ctypes,gc,time,contextlib
import torch
from tilelang import language as T
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from tiny_pair import get_pair_kernel
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v030_base',r/'codegen/baseline_v025.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
rows=[];failures=[];started=time.time()
for variant,sf,xsf in [('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]:
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();outdtype=torch.float8_e4m3fn if sf else torch.float16;out=torch.empty((32,256),device='cuda',dtype=outdtype);wrapper=base.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf)
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 call(wrapper.kernel,inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone();checks=[(inputs,expected)]
 for route in ['identity','padded','duplicates','all-invalid']:
  torch.manual_seed(1235);v=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True,route=route).gen_inputs();call(wrapper.kernel,v);torch.cuda.synchronize();checks.append((v,out.view(torch.uint8).clone()))
 for layout in ['adjacent','halfwarp']:
  for threads in [128,256,512]:
   name=f'{variant.lower()}_{layout}_t{threads}';(r/'meta/status.json').write_text(json.dumps({'running':name,'completed':len(rows),'failures':failures},indent=2)+'\n')
   try:
    with(r/f'logs/{name}_compile.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):k=get_pair_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,num_threads=threads,pair_layout=layout)
    (r/f'codegen/{name}.cu').write_text(k.get_kernel_source())
    for v,ref in checks:call(k,v);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),ref),(name,'byte mismatch')
    samples={'baseline':[],'candidate':[]}
    for turn in range(5):
     for label in list(samples)[::1 if turn%2==0 else -1]:
      def fn(*v):call(wrapper.kernel if label=='baseline'else k,v)
      us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[label].append(us);gc.collect();libc.malloc_trim(0)
    a,b=[statistics.median(samples[label])for label in samples]
    row=dict(variant=variant,pair_layout=layout,threads=threads,ctas=32,elements_per_pair=512//threads,baseline_us=a,candidate_us=b,speedup=a/b,latency_reduction_pct=(1-b/a)*100,byte_equal_routes=5,timing='cupti',samples=json.dumps(samples));rows.append(row);print('RESULT',json.dumps({key:value for key,value in row.items()if key!='samples'}),flush=True)
    with(r/'raw/sweep_16g.csv').open('w',newline='')as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    del k,fn;gc.collect();torch.cuda.empty_cache();libc.malloc_trim(0)
   except Exception as e:
    (r/f'logs/{name}_failure.log').write_text(traceback.format_exc());failures.append({'case':name,'error':str(e)[:300]});print('FAILED',name,str(e)[:200],flush=True)
    if not rows:raise
 del inputs,out,wrapper,expected,checks;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/status.json').write_text(json.dumps({'done':True,'rows':len(rows),'failures':failures,'seconds':time.time()-started},indent=2)+'\n');print('PAIR_SWEEP_DONE',len(rows),flush=True)
