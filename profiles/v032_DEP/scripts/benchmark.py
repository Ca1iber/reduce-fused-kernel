from pathlib import Path
import json,csv,statistics,ctypes,gc,contextlib
import torch
from tilelang import language as T
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from tiny_packed import get_packed_pair_kernel
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None);rows=[]
for variant,sf,xsf in [('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]:
 (r/'meta/status.json').write_text(json.dumps({'running':variant,'completed':len(rows)},indent=2)+'\n')
 current=MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);dtype=current.out_dtype;threads=current.config['num_threads'];out=torch.empty((32,256),device='cuda',dtype=dtype)
 if variant=='Quantized':candidate=current.kernel
 else:
  with(r/f'logs/compile_{variant.lower()}.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):candidate=get_packed_pair_kernel(256,2,T.float16,T.dtype(dtype),sf,True,xsf,num_threads=threads)
  (r/f'codegen/{variant.lower()}.cu').write_text(candidate.get_kernel_source())
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 for route in ['random','padded']:
  torch.manual_seed(1235);v=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True,route=route).gen_inputs();call(current.kernel,v);torch.cuda.synchronize();expected=out.view(torch.uint8).clone();call(candidate,v);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),(variant,route)
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();kernels={'current_v030':current.kernel,'packed':candidate};samples={key:[]for key in kernels}
 for turn in range(5):
  for key in list(kernels)[::1 if turn%2==0 else -1]:
   def fn(*v):call(kernels[key],v)
   us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[key].append(us);gc.collect();libc.malloc_trim(0)
 old,new=[statistics.median(values)for values in samples.values()];row=dict(variant=variant,threads=threads,ctas=32,packing_enabled=variant!='Quantized',current_us=old,candidate_us=new,reduction_pct=(1-new/old)*100,byte_equal=True,timing='cupti',rounds=5);rows.append(row);(r/f'raw/{variant.lower()}_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
 with(r/'raw/comparison_16g.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('RESULT',json.dumps(row),flush=True);del current,candidate,out,v,inputs,kernels,expected,fn;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/status.json').write_text(json.dumps({'done':True,'tiny_cases':4,'byte_equal_routes':['random','padded'],'extra_test_suites_run':False},indent=2)+'\n');print('PACKED_SHUFFLE_COMPLETE',flush=True)
