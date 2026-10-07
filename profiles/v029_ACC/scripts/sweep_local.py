from pathlib import Path
import sys,json,csv,statistics,importlib.util,traceback,ctypes,gc,time,contextlib
import torch
from tilelang import language as T
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from tiny_local import get_tiny_local_kernel
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v029_baseline',r/'codegen/baseline_v025.py');baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
variants=[('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]
rows=[];failures=[];started=time.time()
def save(stage):
 with(r/'raw/local_sweep_16g.csv').open('w',newline='')as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (r/'meta/local_status.json').write_text(json.dumps({'stage':stage,'completed':len(rows),'failures':failures,'seconds':time.time()-started},indent=2)+'\n')
for variant,sf,xsf in variants:
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs()
 wrapper=baseline.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);outdtype=torch.float8_e4m3fn if sf else torch.float16
 out=torch.empty((32,256),dtype=outdtype,device='cuda')
 def call(kernel,v):kernel(v[0],v[2],v[1],out,v[4],v[3])
 call(wrapper.kernel,inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
 correctness_inputs=[]
 for route in ['identity','padded','duplicates','all-invalid']:
  torch.manual_seed(1235);v=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True,route=route).gen_inputs();call(wrapper.kernel,v);torch.cuda.synchronize();correctness_inputs.append((v,out.view(torch.uint8).clone()))
 for group in [1,2,4,8]:
  for threads in [32,64,128,256]:
   if (group,threads) not in [(1,32),(1,64),(1,128),(1,256),(2,128),(2,256),(4,256),(8,256)]:continue
   name=f'{variant.lower()}_g{group}_t{threads}';save(name)
   try:
    with(r/f'logs/local_{name}_compile.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
     if group==0:
      candidate=baseline.get_reduce_fused_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,tile_hidden=256,num_threads=threads,vector_store=sf)
     else:
      candidate=get_tiny_local_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,tokens_per_cta=group,num_threads=threads)
    (r/f'codegen/local_{name}.cu').write_text(candidate.get_kernel_source())
    for v,ref in [(inputs,expected),*correctness_inputs]:
     call(candidate,v);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),ref),name
    samples={'baseline':[],'candidate':[]};kernels={'baseline':wrapper.kernel,'candidate':candidate}
    for round_id in range(5):
     for label in list(kernels)[::1 if round_id%2==0 else -1]:
      def fn(*v):call(kernels[label],v)
      us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
      timing=getattr(_bench_meta,'timing',None);assert timing=='cupti',(name,timing)
      samples[label].append(us);gc.collect();libc.malloc_trim(0)
    a,b=[statistics.median(samples[label])for label in samples]
    row=dict(variant=variant,family='formal_threads'if group==0 else'local_vector',tokens_per_cta=max(1,group),threads=threads,ctas=32//max(1,group),elements_per_thread=256*max(1,group)//threads,baseline_us=a,candidate_us=b,speedup=a/b,latency_reduction_pct=(1-b/a)*100,byte_equal_routes=5,timing='cupti',samples=json.dumps(samples))
    rows.append(row);print('RESULT',json.dumps({k:v for k,v in row.items()if k!='samples'}),flush=True)
    del candidate,kernels,fn;gc.collect();torch.cuda.empty_cache();libc.malloc_trim(0)
   except Exception as e:
    (r/f'logs/local_{name}_failure.log').write_text(traceback.format_exc());failures.append({'case':name,'error':str(e)[:300]});print('FAILED',name,str(e)[:200],flush=True)
   save(name)
 del correctness_inputs,inputs,wrapper,out,expected;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/local_status.json').write_text(json.dumps({'done':True,'rows':len(rows),'failures':failures,'seconds':time.time()-started},indent=2)+'\n');print('SWEEP_DONE',len(rows),flush=True)
