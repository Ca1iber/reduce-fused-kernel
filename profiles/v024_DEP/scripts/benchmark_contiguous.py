from pathlib import Path
import argparse,csv,json,statistics,torch,gc,ctypes,importlib.util,sys
from store128_contiguous import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v024_baseline',r/'codegen/baseline_v022.py');base_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base_module;spec.loader.exec_module(base_module)
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['fp8','quantized'],required=True);p.add_argument('--workload',choices=['tiny','h3072','h7168','prefill'],required=True);a=p.parse_args()
t,h={'tiny':(32,256),'h3072':(512,3072),'h7168':(512,7168),'prefill':(4096,7168)}[a.workload];topk=2 if a.workload=='tiny'else 8;dtype=torch.float16 if a.workload=='tiny'else torch.bfloat16;ind='float16'if a.workload=='tiny'else'bfloat16';xsf=a.variant=='quantized';tile=256 if a.workload=='tiny'else 1024
base=base_module.MoeReduceFusedKernel(t,topk,h,dtype,with_sf=True,with_x_sf=xsf)
args=(h,topk,ind,'float8_e4m3fn',True,True,xsf)
opts={'tile_hidden':tile,'num_threads':64,'grid_hidden_first':base.grid_hidden_first,'contiguous_layout':True}
kernels={'v022':base.kernel,'same_geometry_direct':get_reduce_fused_kernel(*args,**opts,vector_store=False),'store128':get_reduce_fused_kernel(*args,**opts,vector_store=True)}
torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(t,topk,h,dtype,with_sf=True,with_x_sf=xsf).gen_inputs()
out=torch.empty((t,h),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(inputs[0].shape[0],device='cuda')
def call(k,v):k(v[0],v[2],v[1],out,v[-1],v[3]if xsf else unused)
call(kernels['v022'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name,k in kernels.items():
 call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
 (r/f'codegen/{a.variant}_{a.workload}_{name}.cu').write_text(k.get_kernel_source())
print('BYTE_EQUAL',a.variant,a.workload,flush=True)
measurements={name:[]for name in kernels};names=list(kernels)
for round_id in range(5):
 order=names[round_id%3:]+names[:round_id%3]
 if round_id%2:order=list(reversed(order))
 for name in order:
  def fn(*v):call(kernels[name],v)
  us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
  measurements[name].append({'us':us,'timing':getattr(_bench_meta,'timing','unknown'),'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
  print('TIMING',round_id,name,us,flush=True);gc.collect();libc.malloc_trim(0)
 (r/f'raw/{a.variant}_{a.workload}_progress.json').write_text(json.dumps(measurements,indent=2)+'\n')
med={name:statistics.median(x['us']for x in values)for name,values in measurements.items()}
rows=[dict(variant=a.variant,workload=a.workload,name=name,tile_hidden=base.config['tile_hidden']if name=='v022'else tile,num_threads=base.config['num_threads']if name=='v022'else 64,latency_us=med[name],speedup_vs_v022=med['v022']/med[name],byte_equal=True,measurements=json.dumps(measurements[name]))for name in names]
with(r/f'raw/{a.variant}_{a.workload}.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print('RESULT',a.variant,a.workload,med,flush=True)
