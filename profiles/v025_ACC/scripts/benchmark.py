from pathlib import Path
import argparse,csv,json,statistics,torch,gc,ctypes,importlib.util,sys
from prefetch_combined import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1];torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v025_baseline',r/'codegen/baseline_v022.py');baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['fp8','quantized'],required=True);p.add_argument('--workload',choices=['h3072','h7168'],required=True);a=p.parse_args()
h=3072 if a.workload=='h3072'else 7168;xsf=a.variant=='quantized'
base=baseline.MoeReduceFusedKernel(512,8,h,torch.bfloat16,with_sf=True,with_x_sf=xsf);assert base.config=={'tile_hidden':512,'num_threads':128}and not base.vector_store
args=(h,8,'bfloat16','float8_e4m3fn',True,True,xsf)
kwargs={'tile_hidden':512,'num_threads':128,'vector_store':base.vector_store,'prefetch_rows':8}
kernels={'v022':base.kernel,'prefetch8_current_grid':get_reduce_fused_kernel(*args,**kwargs,grid_hidden_first=base.grid_hidden_first)}
if base.grid_hidden_first:kernels['prefetch8_old_grid']=get_reduce_fused_kernel(*args,**kwargs,grid_hidden_first=False)
torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(512,8,h,torch.bfloat16,with_sf=True,with_x_sf=xsf).gen_inputs();out=torch.empty((512,h),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(inputs[0].shape[0],device='cuda')
def call(k,v):k(v[0],v[2],v[1],out,v[-1],v[3]if xsf else unused)
call(kernels['v022'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name,k in kernels.items():
 call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
 (r/f'codegen/{a.variant}_{a.workload}_{name}.cu').write_text(k.get_kernel_source())
print('BYTE_EQUAL',a.variant,a.workload,flush=True)
measurements={name:[]for name in kernels};names=list(kernels)
for round_id in range(5):
 order=names[round_id%len(names):]+names[:round_id%len(names)]
 if round_id%2:order=list(reversed(order))
 for name in order:
  def fn(*v):call(kernels[name],v)
  us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
  measurements[name].append({'us':us,'timing':getattr(_bench_meta,'timing','unknown'),'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
  print('TIMING',round_id,name,us,flush=True);gc.collect();libc.malloc_trim(0)
 (r/f'raw/{a.variant}_{a.workload}_progress.json').write_text(json.dumps(measurements,indent=2)+'\n')
med={name:statistics.median(z['us']for z in values)for name,values in measurements.items()}
rows=[dict(variant=a.variant,workload=a.workload,name=name,grid_hidden_first=base.grid_hidden_first if name!='prefetch8_old_grid'else False,latency_us=med[name],speedup_vs_v022=med['v022']/med[name],latency_reduction_pct=(1-med[name]/med['v022'])*100,byte_equal=True,measurements=json.dumps(measurements[name]))for name in names]
with(r/f'raw/{a.variant}_{a.workload}.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print('RESULT',a.variant,a.workload,med,flush=True)
