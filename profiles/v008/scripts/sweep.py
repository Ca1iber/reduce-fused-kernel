from pathlib import Path
import csv, json, statistics, gc, platform, time
import torch
from benchmarks.benchmark_base import bench_kernel, _bench_meta
from workloads.moe import MoeReduceFusedWorkload
from factory import factory, baseline
root=Path(__file__).resolve().parents[1]
configs=[(threads,tile) for threads in (64,128,256) for tile in (512,1024)]
orders=[configs,configs[2:]+configs[:2],configs[4:]+configs[:4]]
properties=torch.cuda.get_device_properties(0)
(root/'meta/runtime.json').write_text(json.dumps(dict(hostname=platform.node(),torch_version=torch.__version__,device_name=properties.name,total_memory=properties.total_memory,multiprocessor_count=properties.multi_processor_count,orders=orders,started=time.time()),indent=2)+'\n')
rows=[]
for variant,xsf in (('fp8',False),('quantized',True)):
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
        kernels={config:factory(config[1],config[0])(hidden,8,'bfloat16','float8_e4m3fn',True,True,xsf) for config in configs}
        for config,kernel in kernels.items():
            call(kernel,inputs)
            torch.cuda.synchronize()
            if not torch.equal(expected,out.view(torch.uint8)):
                raise RuntimeError(f'byte mismatch {variant}/{label}/{config}')
        records={config:[] for config in configs}
        for order in orders:
            for config in order:
                kernel=kernels[config]
                def fn(*values):call(kernel,values)
                latency=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)
                records[config].append(dict(us=latency*1000,timing=getattr(_bench_meta,'timing','unknown'),inputs_cloned=getattr(_bench_meta,'inputs_cloned',None)))
        med={config:statistics.median(record['us'] for record in records[config]) for config in configs}
        for threads,tile in configs:
            config=(threads,tile)
            rows.append(dict(variant=variant,workload=label,T=tokens,K=8,H=hidden,threads=threads,tile=tile,cta_count=tokens*(hidden//tile),latency_us=med[config],speedup_vs_v004=med[(128,1024)]/med[config],speedup_vs_v007_best=min(med[(128,512)],med[(128,1024)])/med[config],byte_equal=True,measurements=json.dumps(records[config])))
        with (root/'raw/threads_tile_sweep_16g.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('RESULT',variant,label,json.dumps({f'threads{threads}_tile{tile}':med[(threads,tile)] for threads,tile in configs}),flush=True)
        del kernels,reference,expected,out,dummy,inputs,kernel,fn
        gc.collect();torch.cuda.empty_cache()
print('SWEEP_COMPLETE',flush=True)
