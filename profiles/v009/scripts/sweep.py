from pathlib import Path
import csv,json,statistics,gc,platform,time,argparse
import torch
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from workloads.moe import MoeReduceFusedWorkload
from factory import factory,baseline
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--variant');parser.add_argument('--workload');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
all_configs=[('baseline',128,0),('full',64,0),('full',256,0),('split',64,256)]+[('split',threads,tile) for threads in (64,128,256) for tile in (512,1024)]
properties=torch.cuda.get_device_properties(0)
(root/('meta/runtime_'+str(args.variant)+'_'+str(args.workload)+'.json')).write_text(json.dumps(dict(hostname=platform.node(),torch_version=torch.__version__,device_name=properties.name,total_memory=properties.total_memory,multiprocessor_count=properties.multi_processor_count,configs=all_configs,started=time.time()),indent=2)+'\n')
rows=list(csv.DictReader((root/'raw/base_xsf_parallel_sweep_16g.csv').open())) if args.resume and (root/'raw/base_xsf_parallel_sweep_16g.csv').exists() else []
for variant,xsf in (('base',False),('xsf',True)):
    for label,tokens,k,hidden,dtype in (('tiny',32,2,256,torch.float16),('h3072',512,8,3072,torch.bfloat16),('h7168',512,8,7168,torch.bfloat16),('prefill',4096,8,7168,torch.bfloat16)):
        if args.variant and variant!=args.variant:continue
        if args.workload and label!=args.workload:continue
        if args.resume and any(r['variant']==variant and r['workload']==label for r in rows):continue
        print('START',variant,label,flush=True)
        configs=all_configs[:3] if label=='tiny' else all_configs
        orders=[configs,configs[1:]+configs[:1],list(reversed(configs))]
        torch.manual_seed(1235)
        inputs=MoeReduceFusedWorkload(tokens,k,hidden,dtype,with_x_sf=xsf,with_sf=False).gen_inputs()
        out=torch.empty((tokens,hidden),device='cuda',dtype=dtype)
        unused_sf=torch.empty(1,device='cuda');unused_xsf=torch.empty(inputs[0].shape[0],device='cuda')
        def call(kernel,values):
            kernel(values[0],values[2],values[1],out,unused_sf,values[3] if xsf else unused_xsf)
        dtype_name=str(dtype).removeprefix('torch.')
        reference=baseline(hidden,k,dtype_name,dtype_name,False,True,xsf)
        call(reference,inputs);expected=out.view(torch.uint8).clone()
        kernels={config:(reference if config[0]=='baseline' else factory(*config)(hidden,k,dtype_name,dtype_name,False,True,xsf)) for config in configs}
        for config,kernel in kernels.items():
            call(kernel,inputs);torch.cuda.synchronize()
            if not torch.equal(expected,out.view(torch.uint8)):
                raise RuntimeError(f'byte mismatch {variant}/{label}/{config}')
        records={config:[] for config in configs}
        for order in orders:
            for config in order:
                kernel=kernels[config]
                def fn(*values):call(kernel,values)
                latency=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)
                records[config].append(dict(us=latency*1000,timing=getattr(_bench_meta,'timing','unknown'),inputs_cloned=getattr(_bench_meta,'inputs_cloned',None)))
        med={config:statistics.median(r['us'] for r in records[config]) for config in configs}
        for mode,threads,tile in configs:
            config=(mode,threads,tile)
            block=hidden if mode in ('baseline','full') else tile
            rows.append(dict(variant=variant,workload=label,T=tokens,K=k,H=hidden,dtype=str(dtype),mode=mode,threads=threads,tile=tile,effective_tile=block,cta_count=tokens*(hidden//block),latency_us=med[config],speedup_vs_baseline=med[('baseline',128,0)]/med[config],byte_equal=True,measurements=json.dumps(records[config])))
        with (root/'raw/base_xsf_parallel_sweep_16g.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('RESULT',variant,label,json.dumps({f'{mode}_threads{threads}_tile{tile}':med[(mode,threads,tile)] for mode,threads,tile in configs}),flush=True)
        del kernels,reference,expected,out,unused_sf,unused_xsf,inputs,kernel,fn
        gc.collect();torch.cuda.empty_cache()
print('SWEEP_COMPLETE',flush=True)
