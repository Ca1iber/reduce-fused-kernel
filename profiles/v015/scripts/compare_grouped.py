from pathlib import Path
import json,csv,statistics,torch,traceback
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from grouped_hidden_kernel import get_hidden_kernel
root=Path(__file__).resolve().parents[1]
torch.manual_seed(1235)
values=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
out=torch.empty((512,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(values[0].shape[0],device='cuda')
def call(kernel,values):kernel(values[0],values[2],values[1],out,values[-1],unused)
baseline=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128)
call(baseline,values);expected=out.view(torch.uint8).clone()
kernels={'baseline':baseline};failures=[]
for name,mode,group in [('group2_direct',0,2),('group2_async',2,2),('group4_direct',0,4),('group4_async',2,4)]:
    try:
        kernel=get_hidden_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128,mode=mode,group_size=group)
        source=kernel.get_kernel_source();(root/f'codegen/{name}.cu').write_text(source)
        call(kernel,values);torch.cuda.synchronize()
        if not torch.equal(expected,out.view(torch.uint8)):raise RuntimeError('output bytes differ')
        kernels[name]=kernel
        print('CORRECT',name,'async calls',source.count('memcpy_async<'),flush=True)
    except Exception as e:
        traceback.print_exc();failures.append(dict(name=name,error=repr(e)))
(root/'meta/grouped_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
names=list(kernels);records={name:[] for name in names}
for order in (names,names[1:]+names[:1],list(reversed(names))):
    for name in order:
        kernel=kernels[name]
        def fn(*args):call(kernel,args)
        us=bench_kernel(fn,args=values,n_warmup=10,n_repeat=50,n_trials=3)*1000
        records[name].append(dict(us=us,timing=getattr(_bench_meta,'timing','unknown')))
        print('TIMING',name,us,flush=True)
median={name:statistics.median(r['us'] for r in records[name]) for name in names}
rows=[dict(name=name,latency_us=median[name],speedup=median['baseline']/median[name],byte_equal=True,measurements=json.dumps(records[name])) for name in names]
with (root/'raw/grouped_fp8_h7168_16g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('INITIAL_COMPLETE',json.dumps(median),flush=True)
