from pathlib import Path
import argparse,csv,json,statistics,torch,gc,ctypes
torch.set_num_threads(1)
libc=ctypes.CDLL(None)
from grid_kernel import get_reduce_fused_kernel
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['fp8','quantized'],required=True);p.add_argument('--workload',choices=['tiny','h3072','h7168','prefill'],required=True);a=p.parse_args()
t,h={'tiny':(32,256),'h3072':(512,3072),'h7168':(512,7168),'prefill':(4096,7168)}[a.workload];xsf=a.variant=='quantized'
topk=2 if a.workload=='tiny' else 8
dtype=torch.float16 if a.workload=='tiny' else torch.bfloat16
ind='float16' if a.workload=='tiny' else 'bfloat16'
torch.manual_seed(1235)
inputs=MoeReduceFusedWorkload(t,topk,h,dtype,with_sf=True,with_x_sf=xsf).gen_inputs()
base=MoeReduceFusedKernel(t,topk,h,dtype,with_sf=True,with_x_sf=xsf)
kernels={'v010':base.kernel,'grid_hidden_first':get_reduce_fused_kernel(h,topk,ind,'float8_e4m3fn',True,True,xsf,**base.config)}
out=torch.empty((t,h),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(inputs[0].shape[0],device='cuda')
def call(k,v):k(v[0],v[2],v[1],out,v[-1],v[3] if xsf else unused)
call(kernels['v010'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name,k in kernels.items():
 call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
 (r/f'codegen/{a.variant}_{a.workload}_{name}.cu').write_text(k.get_kernel_source())
print('BYTE_EQUAL',a.variant,a.workload,base.config,flush=True)
measurements={name:[] for name in kernels}
for round_id in range(5):
 for name in list(kernels)[::1 if round_id%2==0 else -1]:
  def fn(*v):call(kernels[name],v)
  us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
  measurements[name].append({'us':us,'timing':getattr(_bench_meta,'timing','unknown'),'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
  print('TIMING',round_id,name,us,flush=True)
  gc.collect();libc.malloc_trim(0)
 (r/f'raw/{a.variant}_{a.workload}_progress.json').write_text(json.dumps(measurements,indent=2)+'\n')
b,c=[statistics.median(z['us'] for z in measurements[name])for name in kernels]
row=dict(variant=a.variant,workload=a.workload,**base.config,baseline_us=b,candidate_us=c,speedup=b/c,latency_reduction_pct=(1-c/b)*100,byte_equal=True,measurements=json.dumps(measurements))
with (r/f'raw/{a.variant}_{a.workload}.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();w.writerow(row)
print('RESULT',json.dumps({k:v for k,v in row.items()if k!='measurements'}),flush=True)
