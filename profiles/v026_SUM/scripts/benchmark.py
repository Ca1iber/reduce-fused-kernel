from pathlib import Path
import sys,json,csv,time,statistics,importlib.util,gc,ctypes,hashlib
r=Path(__file__).resolve().parents[1]
(r/'meta/status.json').write_text(json.dumps({'running':'imports','completed':[]},indent=2)+'\n')
import torch
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
torch.set_num_threads(1);libc=ctypes.CDLL(None)
def load(version):
 spec=importlib.util.spec_from_file_location('_summary_'+version,r/f'codegen/{version}.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
modules={v:load(v)for v in ['v000','v025']}
shapes=[('tiny',32,2,256,torch.float16),('h3072',512,8,3072,torch.bfloat16),('h7168',512,8,7168,torch.bfloat16),('prefill',4096,8,7168,torch.bfloat16)]
variants=[('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]
rows=[];completed=[];started=time.time()
for variant,sf_flag,xsf_flag in variants:
 for w,t,k,h,dtype in shapes:
  name=variant.lower()+'_'+w;case_start=time.time()
  (r/'meta/status.json').write_text(json.dumps({'running':name,'completed':completed},indent=2)+'\n')
  torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(t,k,h,dtype,with_sf=sf_flag,with_x_sf=xsf_flag).gen_inputs()
  wrappers={v:m.MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf_flag,with_x_sf=xsf_flag)for v,m in modules.items()}
  out=torch.empty((t,h),device='cuda',dtype=torch.float8_e4m3fn if sf_flag else dtype)
  unused_sf=torch.empty(1,device='cuda');unused_xsf=torch.empty(inputs[0].shape[0],device='cuda')
  def call(wrapper,values):wrapper.kernel(values[0],values[2],values[1],out,values[-1]if sf_flag else unused_sf,values[3]if xsf_flag else unused_xsf)
  call(wrappers['v000'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
  call(wrappers['v025'],inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),(variant,w,'output mismatch')
  samples={v:[]for v in wrappers}
  for round_id in range(5):
   for v in list(wrappers)[::1 if round_id%2==0 else -1]:
    def fn(*values):call(wrappers[v],values)
    us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
    timing=getattr(_bench_meta,'timing',None);assert timing=='cupti',(name,v,timing)
    samples[v].append({'us':us,'timing':timing,'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
    gc.collect();libc.malloc_trim(0)
  baseline,current=[statistics.median(z['us']for z in samples[v])for v in wrappers]
  conf=wrappers['v025'].config
  row=dict(variant=variant,workload=w,num_tokens=t,num_topk=k,hidden=h,dtype=str(dtype),baseline_us=baseline,current_us=current,speedup=baseline/current,latency_reduction_pct=(1-current/baseline)*100,tile_hidden=conf['tile_hidden'],num_threads=conf['num_threads'],grid_hidden_first=wrappers['v025'].grid_hidden_first,vector_store=wrappers['v025'].vector_store,prefetch_rows=wrappers['v025'].prefetch_rows,byte_equal=True,timing='cupti',inputs_cloned=samples['v025'][0]['inputs_cloned'])
  rows.append(row)
  (r/f'raw/{name}_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
  with(r/'raw/comparison_16g.csv').open('w',newline='')as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
  print('FINISHED',json.dumps(row),flush=True)
  del fn,inputs,wrappers,out,unused_sf,unused_xsf,expected;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
  completed.append({'case':name,'seconds':time.time()-case_start})
source=json.loads((r/'meta/source.json').read_text());assert hashlib.sha256((r.parents[1]/'tileops/kernels/moe/reduce_fused.py').read_bytes()).hexdigest()==source['current_source_sha256'],'formal source changed during collection'
(r/'meta/environment.json').write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'device_properties':str(torch.cuda.get_device_properties(0))},indent=2)+'\n')
(r/'meta/status.json').write_text(json.dumps({'done':True,'cases':len(rows),'seconds':time.time()-started,'completed':completed},indent=2)+'\n')
print('SUMMARY_BENCHMARK_COMPLETE',len(rows),flush=True)
