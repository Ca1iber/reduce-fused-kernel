from pathlib import Path
import sys,json,csv,statistics,importlib.util,ctypes,gc
import torch
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_pre_v030_formal',r/'codegen/pre_integration_formal_v029.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
rows=[]
for variant,sf,xsf in [('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]:
 (r/'meta/integration_status.json').write_text(json.dumps({'running':variant,'completed':len(rows)},indent=2)+'\n')
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();a=old.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);b=MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);out=torch.empty((32,256),device='cuda',dtype=b.out_dtype)
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 call(a.kernel,inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone();call(b.kernel,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),variant
 kernels={'v029':a.kernel,'integrated_v030':b.kernel};samples={key:[]for key in kernels}
 for turn in range(5):
  for key in list(kernels)[::1 if turn%2==0 else -1]:
   def fn(*v):call(kernels[key],v)
   us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[key].append(us);gc.collect();libc.malloc_trim(0)
 before,after=[statistics.median(v)for v in samples.values()];row=dict(variant=variant,old_threads=a.config['num_threads'],new_threads=b.config['num_threads'],implementation=b.tiny_implementation,v029_us=before,integrated_v030_us=after,reduction_pct=(1-after/before)*100,byte_equal=True,timing='cupti',rounds=5);rows.append(row)
 (r/f'raw/integrated_{variant.lower()}_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
 with(r/'raw/integrated_comparison_16g.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('RESULT',json.dumps(row),flush=True);del a,b,out,inputs,kernels,expected,fn;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/integration_status.json').write_text(json.dumps({'done':True,'tiny_cases':4,'all_byte_equal':True,'extra_test_suites_run':False},indent=2)+'\n');print('FOUR_TINY_COMPLETE',flush=True)
