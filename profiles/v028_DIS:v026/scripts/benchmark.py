from pathlib import Path
import ctypes,json,csv,statistics,gc,time
import torch
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
lib=ctypes.CDLL(str(r/'codegen/empty_probe.so'));lib.launch_empty.argtypes=[ctypes.c_int];lib.launch_empty.restype=ctypes.c_int
libc=ctypes.CDLL(None);rows=[]
for threads,names in [(256,[('Base',False,False),('XSF',False,True)]),(128,[('FP8',True,False),('Quantized',True,True)])]:
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs()
 out_half=torch.empty((32,256),device='cuda',dtype=torch.float16);out_fp8=torch.empty((32,256),device='cuda',dtype=torch.float8_e4m3fn)
 wrappers={name:MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf)for name,sf,xsf in names};assert all(v.config['num_threads']==threads for v in wrappers.values())
 def empty(*values):
  error=lib.launch_empty(threads)
  if error:raise RuntimeError('MACA launch error '+str(error))
 functions={'empty_'+str(threads):empty}
 def make_call(wrapper,sf,xsf):
  def call(*v):wrapper.kernel(v[0],v[2],v[1],out_fp8 if sf else out_half,v[4],v[3])
  return call
 for name,sf,xsf in names:functions[name]=make_call(wrappers[name],sf,xsf)
 samples={name:[]for name in functions}
 for round_id in range(5):
  for name in list(functions)[::1 if round_id%2==0 else -1]:
   us=bench_kernel(functions[name],args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
   timing=getattr(_bench_meta,'timing',None);assert timing=='cupti',(name,timing)
   samples[name].append({'us':us,'timing':timing,'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
   print('TIMING',threads,round_id,name,us,flush=True);gc.collect();libc.malloc_trim(0)
  (r/f'raw/profiler_samples_threads{threads}.json').write_text(json.dumps(samples,indent=2)+'\n')
 for name,values in samples.items():
  us=[z['us']for z in values];rows.append({'case':name,'threads':threads,'blocks':32,'median_trial_mean_us':statistics.median(us),'min_round_us':min(us),'max_round_us':max(us),'timing':'cupti','l2_flush':True,'rounds':5})
 del functions,wrappers,inputs,out_half,out_fp8;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
with(r/'raw/profiler_comparison_16g.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(r/'meta/python_environment.json').write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'properties':str(torch.cuda.get_device_properties(0))},indent=2)+'\n')
print('PROFILER_BENCHMARK_DONE',json.dumps(rows),flush=True)
