"""Paired baseline/candidate measurements using the project benchmark timer."""
from pathlib import Path
import csv,json,statistics
import torch
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from candidate_fp8_bits import get_candidate
root=Path('/root/TileOPs-Metax/profiles/v003')
if json.loads((root/'meta/validation.json').read_text()).get('status')!='passed':
    raise SystemExit('Correctness gate not passed')
rows=[]
for variant,with_xsf in (('fp8',False),('quantized',True)):
 for label,tokens,k,hidden,dtype in (('tiny',32,2,256,torch.float16),
                                    ('h3072',512,8,3072,torch.bfloat16),
                                    ('h7168',512,8,7168,torch.bfloat16),
                                    ('prefill',4096,8,7168,torch.bfloat16)):
    torch.manual_seed(1235)
    inputs=MoeReduceFusedWorkload(tokens,k,hidden,dtype,with_x_sf=with_xsf,with_sf=True).gen_inputs()
    out=torch.empty((tokens,hidden),device='cuda',dtype=torch.float8_e4m3fn)
    dummy=torch.empty(inputs[0].shape[0],device='cuda')
    old=get_reduce_fused_kernel(hidden,k,str(dtype).removeprefix('torch.'),'float8_e4m3fn',True,True,with_xsf)
    new=get_candidate(hidden,k,str(dtype).removeprefix('torch.'),'float8_e4m3fn',True,True,with_xsf)
    def call(kernel,public_inputs):
        x,pos,weights=public_inputs[:3]
        xsf=public_inputs[3] if with_xsf else dummy
        sf=public_inputs[-1]
        kernel(x,weights,pos,out,sf,xsf)
    measured={'baseline':[],'candidate':[]}
    records=[]
    for implementation in ('baseline','candidate','candidate','baseline'):
        kernel=old if implementation=='baseline' else new
        def fn(*args):call(kernel,args)
        latency=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)
        measured[implementation].append(latency)
        records.append(dict(implementation=implementation,latency_ms=latency,
             timing=getattr(_bench_meta,'timing','unknown'),
             inputs_cloned=getattr(_bench_meta,'inputs_cloned',None)))
    a,b=(statistics.median(measured[name]) for name in ('baseline','candidate'))
    row=dict(variant=variant,workload=label,T=tokens,K=k,H=hidden,dtype=str(dtype),
             baseline_us=a*1000,candidate_us=b*1000,speedup=a/b,
             latency_reduction_percent=100*(1-b/a),paired_order='A-B-B-A',
             measurements=json.dumps(records))
    rows.append(row)
    print('PAIRED_RESULT',json.dumps(row),flush=True)
    with (root/'raw/paired_benchmark_16g.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
print('PAIRED_BENCHMARK_COMPLETED',len(rows),flush=True)
