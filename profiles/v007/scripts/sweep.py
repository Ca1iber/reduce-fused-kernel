from pathlib import Path
import csv, json, statistics, time, gc
import torch
from benchmarks.benchmark_base import bench_kernel, _bench_meta
from workloads.moe import MoeReduceFusedWorkload
from factory import factory, baseline
root = Path(__file__).resolve().parents[1]
rows=[]
order=(1024,512,256,256,1024,512,512,256,1024)
for variant, xsf in (('fp8',False),('quantized',True)):
    for label,tokens,hidden in (('h3072',512,3072),('h7168',512,7168),('prefill',4096,7168)):
        print('START',variant,label,flush=True)
        torch.manual_seed(1235)
        inputs=MoeReduceFusedWorkload(tokens,8,hidden,torch.bfloat16,with_x_sf=xsf,with_sf=True).gen_inputs()
        out=torch.empty((tokens,hidden),device='cuda',dtype=torch.float8_e4m3fn)
        dummy=torch.empty(inputs[0].shape[0],device='cuda')
        def call(kernel,values):
            kernel(values[0],values[2],values[1],out,values[-1],values[3] if xsf else dummy)
        reference=baseline(hidden,8,'bfloat16','float8_e4m3fn',True,True,xsf)
        call(reference,inputs)
        expected=out.view(torch.uint8).clone()
        kernels={tile:factory(tile)(hidden,8,'bfloat16','float8_e4m3fn',True,True,xsf) for tile in (256,512,1024)}
        for tile,kernel in kernels.items():
            call(kernel,inputs)
            torch.cuda.synchronize()
            if not torch.equal(expected,out.view(torch.uint8)):
                raise RuntimeError(f'byte mismatch {variant}/{label}/{tile}')
        records={tile:[] for tile in kernels}
        for tile in order:
            kernel=kernels[tile]
            def fn(*values):call(kernel,values)
            latency=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)
            records[tile].append(dict(us=latency*1000,timing=getattr(_bench_meta,'timing','unknown'),inputs_cloned=getattr(_bench_meta,'inputs_cloned',None)))
        med={tile:statistics.median(r['us'] for r in records[tile]) for tile in kernels}
        for tile in kernels:
            rows.append(dict(variant=variant,workload=label,T=tokens,K=8,H=hidden,threads=128,tile=tile,cta_count=tokens*(hidden//tile),latency_us=med[tile],speedup_vs_1024=med[1024]/med[tile],byte_equal=True,measurements=json.dumps(records[tile])))
        with (root/'raw/tile_sweep.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('RESULT',variant,label,json.dumps(med),flush=True)
        del kernels,reference,expected,out,dummy,inputs,kernel,fn
        gc.collect();torch.cuda.empty_cache()
print('SWEEP_COMPLETE',flush=True)
