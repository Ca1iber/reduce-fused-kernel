from pathlib import Path
import csv,json,statistics,torch,traceback
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from cooperative_kernel import get_cooperative_kernel
root=Path(__file__).resolve().parents[1]
torch.manual_seed(1235)
x,pos,w,sf=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_sf=True).gen_inputs()
inputs=(x,pos,w,sf);out=torch.empty((512,7168),device='cuda',dtype=torch.float8_e4m3fn);unused=torch.empty(x.shape[0],device='cuda')
def call(k,values):k(values[0],values[2],values[1],out,values[3],unused)
base=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=128)
call(base,inputs);expected=out.view(torch.uint8).clone();kernels={'baseline':base};fail=[]
for threads in (128,256):
 for cooperative in (False,True):
    name=('warp' if cooperative else 'allk')+str(threads)
    try:
        k=get_cooperative_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=512,num_threads=threads,cooperative=cooperative)
        (root/f'codegen/{name}.cu').write_text(k.get_kernel_source());call(k,inputs);torch.cuda.synchronize()
        assert torch.equal(expected,out.view(torch.uint8))
        kernels[name]=k;print('CORRECT',name,flush=True)
    except Exception as e:traceback.print_exc();fail.append({'name':name,'error':repr(e)})
(root/'meta/failures.json').write_text(json.dumps(fail,indent=2)+'\n')
names=list(kernels);records={n:[] for n in names}
for order in (names,names[1:]+names[:1],list(reversed(names))):
 for name in order:
    k=kernels[name]
    def fn(*a):call(k,a)
    us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
    records[name].append(us);print('TIME',name,us,flush=True)
med={n:statistics.median(records[n]) for n in names}
rows=[{'name':n,'latency_us':med[n],'speedup':med['baseline']/med[n],'measurements':json.dumps(records[n])} for n in names]
with (root/'raw/initial_16g.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('COMPLETE',json.dumps(med),flush=True)
