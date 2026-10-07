from pathlib import Path
import sys,json,csv,statistics,ctypes,gc,contextlib,traceback
import torch
from tilelang import language as T
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from tiny_quad import get_quad_kernel
r=Path(__file__).resolve().parents[1];repo=r.parents[1];sys.path.insert(0,str(repo/'profiles/v030_ACC/scripts'))
from tiny_pair import get_pair_kernel
torch.set_num_threads(1);libc=ctypes.CDLL(None);rows=[];failures=[]
for variant,sf,xsf,pair_threads in [('Base',False,False,256),('XSF',False,True,256),('FP8',True,False,512),('Quantized',True,True,512)]:
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();outdtype=torch.float8_e4m3fn if sf else torch.float16;out=torch.empty((32,256),dtype=outdtype,device='cuda');current=MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf);paired=get_pair_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,num_threads=pair_threads)
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 call(current.kernel,inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone();call(paired,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected)
 for layout in ['adjacent','striped16']:
  for threads in [256,512,1024]:
   name=f'{variant.lower()}_{layout}_t{threads}';(r/'meta/status.json').write_text(json.dumps({'running':name,'completed':len(rows),'failures':failures},indent=2)+'\n')
   try:
    with(r/f'logs/{name}_compile.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):new=get_quad_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,num_threads=threads,role_layout=layout)
    (r/f'codegen/{name}.cu').write_text(new.get_kernel_source());call(new,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),(name,'output mismatch')
    kernels={'v029':current.kernel,'v030':paired,'v031':new};samples={label:[]for label in kernels}
    for turn in range(5):
     labels=list(kernels);labels=labels[turn%3:]+labels[:turn%3]
     if turn%2:labels.reverse()
     for label in labels:
      def fn(*v):call(kernels[label],v)
      us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[label].append(us);gc.collect();libc.malloc_trim(0)
    med={label:statistics.median(values)for label,values in samples.items()};row=dict(variant=variant,role_layout=layout,threads=threads,ctas=32,elements_per_quad=1024//threads,v029_us=med['v029'],v030_us=med['v030'],v031_us=med['v031'],reduction_vs_v029_pct=(1-med['v031']/med['v029'])*100,reduction_vs_v030_pct=(1-med['v031']/med['v030'])*100,byte_equal=True,timing='cupti',rounds=5)
    rows.append(row);(r/f'raw/{name}_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    with(r/'raw/comparison_16g.csv').open('w',newline='')as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('RESULT',json.dumps(row),flush=True);del new,kernels,fn;gc.collect();torch.cuda.empty_cache();libc.malloc_trim(0)
   except Exception as e:
    (r/f'logs/{name}_failure.log').write_text(traceback.format_exc());failures.append({'case':name,'error':str(e)[:200]});print('FAILED',name,str(e)[:200],flush=True)
    if not rows:raise
 del inputs,out,current,paired,expected;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/status.json').write_text(json.dumps({'done':True,'rows':len(rows),'failures':failures},indent=2)+'\n');print('QUAD_STUDY_DONE',len(rows),flush=True)
