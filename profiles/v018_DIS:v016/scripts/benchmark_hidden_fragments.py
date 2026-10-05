from pathlib import Path
import sys,torch,json,csv,statistics,argparse,time
repo=Path('/data/TileOPs-Metax');r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'profiles/v016_DEP/scripts'));sys.path.insert(0,str(repo/'profiles/v015_DEP/scripts'))
from prefetch_kernel import get_prefetch_kernel
from grouped_hidden_kernel import get_hidden_kernel as get_grouped
from hidden_fragment import get_hidden_kernel as get_fragment
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
p=argparse.ArgumentParser();p.add_argument('--workload',default='h7168',choices=['tiny','h3072','h7168','prefill']);p.add_argument('--variant',default='fp8',choices=['base','xsf','fp8','quantized']);a=p.parse_args()
t,h={'tiny':(32,256),'h3072':(512,3072),'h7168':(512,7168),'prefill':(4096,7168)}[a.workload];topk=2 if a.workload=='tiny'else 8;ind=torch.float16 if a.workload=='tiny'else torch.bfloat16;ind_name='float16'if a.workload=='tiny'else 'bfloat16';sf=a.variant in ('fp8','quantized');xsf=a.variant in ('xsf','quantized');tile=256 if a.workload=='tiny'else 1024 if a.workload=='prefill'else 512
torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(t,topk,h,ind,with_sf=sf,with_x_sf=xsf).gen_inputs();outd='float8_e4m3fn'if sf else ind_name;out=torch.empty((t,h),device='cuda',dtype=getattr(torch,outd));unused=torch.empty(inputs[0].shape[0],device='cuda');unused_sf=torch.empty(1,device='cuda')
args=(h,topk,ind_name,outd,sf,True,xsf)
base=MoeReduceFusedKernel(t,topk,h,ind,with_sf=sf,with_x_sf=xsf)
kernels={'v010':base.kernel,'v016_register8':get_prefetch_kernel(*args,tile_hidden=tile,num_threads=128,prefetch_rows=8),'group2_direct':get_grouped(*args,tile_hidden=tile,num_threads=128,group_size=2,mode=0),'v015_shared_double':get_grouped(*args,tile_hidden=tile,num_threads=128,group_size=2,mode=2),'v018_fragment_double':get_fragment(*args,tile_hidden=tile,num_threads=128,group_size=2,mode=3)}
def call(k,v):k(v[0],v[2],v[1],out,v[-1]if sf else unused_sf,v[3]if xsf else unused)
call(kernels['v010'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name,k in kernels.items():
 call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
 print('CORRECT',name,flush=True)
names=list(kernels);measurements={n:[]for n in names}
for round in range(5):
 order=names[round:]+names[:round]
 if round%2:order=list(reversed(order))
 for name in order:
  def fn(*v):call(kernels[name],v)
  us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000
  z={'us':us,'timing':getattr(_bench_meta,'timing','unknown'),'inputs_cloned':getattr(_bench_meta,'inputs_cloned',None)};measurements[name].append(z)
  print('TIMING',round,name,us,flush=True)
 (r/f'raw/bench_{a.variant}_{a.workload}_progress.json').write_text(json.dumps(measurements,indent=2)+'\n')
med={name:statistics.median(z['us']for z in zs)for name,zs in measurements.items()}
rows=[dict(variant=a.variant,workload=a.workload,name=name,latency_us=med[name],speedup=med['v010']/med[name],latency_reduction_pct=(1-med[name]/med['v010'])*100,byte_equal=True,measurements=json.dumps(measurements[name]))for name in names]
with (r/f'raw/benchmark_{a.variant}_{a.workload}_16g.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(r/f'meta/bench_{a.variant}_{a.workload}_status.json').write_text(json.dumps({'stage':'done','results':med},indent=2)+'\n');print('COMPLETE',med,flush=True)
