from pathlib import Path
import sys,json,csv,statistics,importlib.util,ctypes,gc,contextlib
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
(r/'meta/integration_status.json').write_text(json.dumps({'stage':'imports'},indent=2)+'\n')
import torch,pytest
from tilelang import language as T
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
sys.path.insert(0,str(r/'scripts'))
from tiny_group_static import get_tiny_group_static
from tiny_local import get_tiny_local_kernel
torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_pre_v029',r/'codegen/pre_integration_formal.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
variants=[('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)];dispatch=[]
for w,t,k,h,dtype in [('tiny',32,2,256,torch.float16),('h3072',512,8,3072,torch.bfloat16),('h7168',512,8,7168,torch.bfloat16),('prefill',4096,8,7168,torch.bfloat16)]:
 for variant,sf,xsf in variants:
  a=old.MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf,with_x_sf=xsf);b=MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf,with_x_sf=xsf);expected=w=='tiny'and (sf or xsf);assert (b.tiny_implementation is not None)==expected
  assert b.grid_hidden_first==a.grid_hidden_first and b.prefetch_rows==a.prefetch_rows and b.vector_store==a.vector_store
  equal=a.kernel.get_kernel_source()==b.kernel.get_kernel_source()
  if not expected:assert a.config==b.config and equal,(w,variant,'unselected changed')
  else:
   candidate=(get_tiny_local_kernel if sf and xsf else get_tiny_group_static)(h,k,T.dtype(dtype),T.dtype(b.out_dtype),sf,True,xsf,num_threads=256);assert candidate.get_kernel_source()==b.kernel.get_kernel_source(),(variant,'production differs from candidate')
  dispatch.append({'variant':variant,'workload':w,'implementation':b.tiny_implementation,'old_config':a.config,'new_config':b.config,'cpp_equal_to_old':equal,'candidate_cpp_identical':expected})
for shape,weights,config in [((32,2,256,torch.bfloat16),True,None),((16,2,256,torch.float16),True,None),((32,2,256,torch.float16),False,None),((32,2,256,torch.float16),True,{'num_threads':128}),((32,2,256,torch.float16),True,{'num_threads':64})]:
 a=old.MoeReduceFusedKernel(*shape,with_sf=True,with_x_sf=True,with_weights=weights,config=config);b=MoeReduceFusedKernel(*shape,with_sf=True,with_x_sf=True,with_weights=weights,config=config);assert b.tiny_implementation is None and a.kernel.get_kernel_source()==b.kernel.get_kernel_source()
(r/'meta/integration_dispatch.json').write_text(json.dumps({'selected_count':3,'unselected_cpp_identical_count':13,'fallback_checks':5,'cases':dispatch},indent=2)+'\n');print('DISPATCH_PASSED',flush=True)
rows=[];checks=[]
for variant,sf,xsf in variants:
 (r/'meta/integration_status.json').write_text(json.dumps({'stage':'benchmark','variant':variant,'completed':len(rows)},indent=2)+'\n')
 a=old.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);b=MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);out=torch.empty((32,256),dtype=b.out_dtype,device='cuda');kernels={'before':a.kernel,'integrated':b.kernel}
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 for route in ['random','identity','padded','duplicates','all-invalid','invalid-nan']:
  torch.manual_seed(1235);v=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True,route='padded'if route=='invalid-nan'else route).gen_inputs()
  if route=='invalid-nan':v[1][v[1]==0]=1;v[0][0]=float('nan');v[3][0]=float('nan');v[2][v[1]<0]=float('nan')
  call(a.kernel,v);torch.cuda.synchronize();expected=out.view(torch.uint8).clone();call(b.kernel,v);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),(variant,route);checks.append({'variant':variant,'route':route,'byte_equal':True})
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();samples={name:[]for name in kernels}
 for turn in range(20):
  for name in list(kernels)[::1 if turn%2==0 else -1]:
   def fn(*v):call(kernels[name],v)
   us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[name].append(us);gc.collect();libc.malloc_trim(0)
 before,after=[statistics.median(values)for values in samples.values()];row=dict(variant=variant,implementation=b.tiny_implementation or'original',old_threads=a.config['num_threads'],new_threads=b.config['num_threads'],before_us=before,integrated_us=after,latency_reduction_pct=(1-after/before)*100,byte_equal=True,timing='cupti',paired_rounds=20);rows.append(row)
 (r/f'raw/integrated_{variant.lower()}_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
 with(r/'raw/integrated_comparison_16g.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 print('INTEGRATED_RESULT',json.dumps(row),flush=True)
 del inputs,v,a,b,kernels,out,expected,fn;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/integration_route_checks.json').write_text(json.dumps(checks,indent=2)+'\n');(r/'meta/integration_status.json').write_text(json.dumps({'stage':'correctness'},indent=2)+'\n')
with(r/'logs/integrated_correctness.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/integrated_correctness.xml')])
assert code==0,'formal correctness failed'
(r/'meta/integration_status.json').write_text(json.dumps({'done':True,'correctness_exitcode':int(code),'benchmark_cases':4,'byte_equal_route_cases':len(checks),'selected_count':3,'unselected_cpp_identical_count':13},indent=2)+'\n');print('V029_INTEGRATION_VALIDATED',flush=True)
