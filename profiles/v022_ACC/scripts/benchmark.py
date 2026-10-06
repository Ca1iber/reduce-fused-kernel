from pathlib import Path
import argparse,csv,json,statistics,torch,gc,ctypes
from vector_store import get_reduce_fused_kernel
import importlib.util,sys
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v022_fixed_baseline',r/'codegen/baseline_formal.py');baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
MoeReduceFusedKernel=baseline.MoeReduceFusedKernel
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['fp8','quantized'],required=True);p.add_argument('--workload',choices=['tiny','h3072','h7168','prefill'],required=True);a=p.parse_args()
t,h={'tiny':(32,256),'h3072':(512,3072),'h7168':(512,7168),'prefill':(4096,7168)}[a.workload]
topk=2 if a.workload=='tiny'else 8;dtype=torch.float16 if a.workload=='tiny'else torch.bfloat16;ind='float16'if a.workload=='tiny'else'bfloat16';xsf=a.variant=='quantized'
torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(t,topk,h,dtype,with_sf=True,with_x_sf=xsf).gen_inputs()
base=MoeReduceFusedKernel(t,topk,h,dtype,with_sf=True,with_x_sf=xsf)
kernels={'formal_v020':base.kernel,'vector_store':get_reduce_fused_kernel(h,topk,ind,'float8_e4m3fn',True,True,xsf,grid_hidden_first=base.grid_hidden_first,**base.config)}
out=torch.empty((t,h),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(inputs[0].shape[0],device='cuda')
def call(k,v):k(v[0],v[2],v[1],out,v[-1],v[3]if xsf else unused)
call(kernels['formal_v020'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name,k in kernels.items():
 call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
print('BYTE_EQUAL',a.variant,a.workload,flush=True)
measurements={name:[]for name in kernels}
for round_id in range(5):
 for name in list(kernels)[::1 if round_id%2==0 else -1]:
  def fn(*v):call(kernels[name],v)
  us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
  measurements[name].append({'us':us,'timing':getattr(_bench_meta,'timing','unknown'),'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
  print('TIMING',round_id,name,us,flush=True);gc.collect();libc.malloc_trim(0)
 (r/f'raw/{a.variant}_{a.workload}_progress.json').write_text(json.dumps(measurements,indent=2)+'\n')
b,c=[statistics.median(z['us']for z in measurements[name])for name in kernels]
row=dict(variant=a.variant,workload=a.workload,**base.config,grid_hidden_first=base.grid_hidden_first,baseline_us=b,candidate_us=c,speedup=b/c,latency_reduction_pct=(1-c/b)*100,byte_equal=True,measurements=json.dumps(measurements))
with(r/f'raw/{a.variant}_{a.workload}.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');writer.writeheader();writer.writerow(row)
print('RESULT',json.dumps({k:v for k,v in row.items()if k!='measurements'}),flush=True)
