from pathlib import Path
import argparse,json,csv,statistics,torch
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel,get_reduce_fused_kernel
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from cooperative_global_positions import get_cooperative_kernel
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--workload',required=True);a=p.parse_args()
t,h={'h3072':(512,3072),'h7168':(512,7168),'prefill':(4096,7168)}[a.workload]
path=r/'raw/confirmation_16g.csv';rows=list(csv.DictReader(path.open())) if path.exists() else []
for variant,xsf,sf in [('base',False,False),('xsf',True,False),('fp8',False,True),('quantized',True,True)]:
 torch.manual_seed(1235)
 inputs=MoeReduceFusedWorkload(t,8,h,torch.bfloat16,with_x_sf=xsf,with_sf=sf).gen_inputs()
 obj=MoeReduceFusedKernel(t,8,h,torch.bfloat16,with_sf=sf,with_x_sf=xsf)
 out=torch.empty((t,h),device='cuda',dtype=obj.out_dtype);unused_sf=torch.empty(1,device='cuda');unused_xsf=torch.empty(inputs[0].shape[0],device='cuda')
 def call(k,v):k(v[0],v[2],v[1],out,v[-1] if sf else unused_sf,v[3] if xsf else unused_xsf)
 dtype='float8_e4m3fn' if sf else 'bfloat16'
 kernels={'current':obj.kernel,'direct_geometry':get_reduce_fused_kernel(h,8,'bfloat16',dtype,sf,True,xsf,tile_hidden=1024,num_threads=512),'cooperative':get_cooperative_kernel(h,8,'bfloat16',dtype,sf,True,xsf,tile_hidden=1024,num_threads=512,cooperative=True)}
 call(kernels['current'],inputs);expected=out.view(torch.uint8).clone()
 for n,k in kernels.items():
    call(k,inputs);torch.cuda.synchronize();assert torch.equal(expected,out.view(torch.uint8)),(variant,a.workload,n)
 measured={n:[] for n in kernels}
 names=list(kernels)
 for order in (names,names[1:]+names[:1],list(reversed(names)),names[2:]+names[:2],names):
    for n in order:
        k=kernels[n]
        def fn(*v):call(k,v)
        us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
        measured[n].append({'us':us,'timing':getattr(_bench_meta,'timing','unknown'),'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)})
 med={n:statistics.median(v['us'] for v in measured[n]) for n in kernels}
 for n in kernels:rows.append({'variant':variant,'workload':a.workload,'name':n,'latency_us':med[n],'speedup':med['current']/med[n],'byte_equal':True,'measurements':json.dumps(measured[n])})
 with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('RESULT',variant,a.workload,json.dumps(med),flush=True)
 del inputs,out,expected,kernels,obj,k,fn
 torch.cuda.empty_cache()
