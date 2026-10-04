from pathlib import Path
import csv,json,statistics,importlib.util,sys,torch,argparse
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_v010_baseline',root/'codegen/baseline_v004.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
parser=argparse.ArgumentParser();parser.add_argument('--workload',required=True);args=parser.parse_args()
cases={'tiny':(32,2,256,torch.float16),'h3072':(512,8,3072,torch.bfloat16),'h7168':(512,8,7168,torch.bfloat16),'prefill':(4096,8,7168,torch.bfloat16)}
tokens,k,hidden,dtype=cases[args.workload]
path=root/'raw/paired_selection_16g.csv'
rows=list(csv.DictReader(path.open())) if path.exists() else []
for variant,xsf,sf in (('base',False,False),('xsf',True,False),('fp8',False,True),('quantized',True,True)):
    torch.manual_seed(1235)
    inputs=MoeReduceFusedWorkload(tokens,k,hidden,dtype,with_x_sf=xsf,with_sf=sf).gen_inputs()
    old=module.MoeReduceFusedKernel(tokens,k,hidden,dtype,with_x_sf=xsf,with_sf=sf)
    new=MoeReduceFusedKernel(tokens,k,hidden,dtype,with_x_sf=xsf,with_sf=sf)
    expected_tile=hidden if not sf or hidden==256 else 1024 if tokens==4096 else 512
    expected_threads=128 if sf else 256
    assert new.config==dict(tile_hidden=expected_tile,num_threads=expected_threads),new.config
    out=torch.empty((tokens,hidden),device='cuda',dtype=new.out_dtype)
    def call(implementation,values):
        kwargs={'out':out}
        if xsf:kwargs['x_sf']=values[3]
        if sf:kwargs['sf']=values[-1]
        return implementation(values[0],values[1],values[2],**kwargs)
    call(old,inputs);expected=out.view(torch.uint8).clone()
    call(new,inputs);torch.cuda.synchronize()
    assert torch.equal(expected,out.view(torch.uint8)),(variant,args.workload)
    measured={'v004':[],'selected':[]};records=[]
    for label in ('v004','selected','selected','v004','v004','selected'):
        implementation=old if label=='v004' else new
        def fn(*values):return call(implementation,values)
        latency=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
        measured[label].append(latency)
        records.append(dict(implementation=label,us=latency,timing=getattr(_bench_meta,'timing','unknown'),inputs_cloned=getattr(_bench_meta,'inputs_cloned',None)))
    a,b=[statistics.median(measured[label]) for label in ('v004','selected')]
    row=dict(variant=variant,workload=args.workload,T=tokens,K=k,H=hidden,dtype=str(dtype),tile_hidden=expected_tile,num_threads=expected_threads,baseline_us=a,selected_us=b,speedup=a/b,byte_equal=True,measurements=json.dumps(records))
    rows.append(row)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    print('RESULT',json.dumps(row),flush=True)
    del old,new,inputs,out,expected,implementation,fn
    torch.cuda.empty_cache()
print('PAIRED_COMPLETE',args.workload,flush=True)
