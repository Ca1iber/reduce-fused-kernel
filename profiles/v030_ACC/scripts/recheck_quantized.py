from pathlib import Path
import sys,json,statistics,random,ctypes,gc,importlib.util
import torch
from tilelang import language as T
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
r=Path(__file__).resolve().parents[1];repo=r.parents[1];sys.path.insert(0,str(r/'scripts'));sys.path.insert(0,str(repo/'profiles/v029_ACC/scripts'))
from tiny_pair import get_pair_kernel
from tiny_local import get_tiny_local_kernel
torch.set_num_threads(1);libc=ctypes.CDLL(None);spec=importlib.util.spec_from_file_location('_q_check_base',r/'codegen/baseline_v025.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();out=torch.empty((32,256),device='cuda',dtype=torch.float8_e4m3fn)
wrapper=base.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=True,with_x_sf=True)
kernels={'v025':wrapper.kernel,'v029':get_tiny_local_kernel(256,2,T.float16,T.float8_e4m3fn,True,True,True,num_threads=256),'pair256':get_pair_kernel(256,2,T.float16,T.float8_e4m3fn,True,True,True,num_threads=256),'pair512':get_pair_kernel(256,2,T.float16,T.float8_e4m3fn,True,True,True,num_threads=512)}
def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
call(kernels['v025'],inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
for name,k in kernels.items():call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
samples={name:[]for name in kernels}
for turn in range(20):
 names=list(kernels);names=names[turn%4:]+names[:turn%4]
 if turn%2:names.reverse()
 for name in names:
  def fn(*v):call(kernels[name],v)
  us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[name].append(us);gc.collect();libc.malloc_trim(0)
med={name:statistics.median(values)for name,values in samples.items()};result={'medians_us':med,'rounds':20,'byte_equal':True,'comparisons':[]}
for name in ['pair256','pair512']:
 for reference in ['v025','v029']:
  diff=[a-b for a,b in zip(samples[reference],samples[name])];rng=random.Random(1235);boot=sorted(statistics.mean(rng.choices(diff,k=20))for _ in range(5000));result['comparisons'].append({'candidate':name,'reference':reference,'reduction_pct':(1-med[name]/med[reference])*100,'paired_saved_mean_us':statistics.mean(diff),'ci95_low_us':boot[125],'ci95_high_us':boot[4874]})
(r/'raw/quantized_close_configs_recheck_samples.json').write_text(json.dumps(samples,indent=2)+'\n');(r/'raw/quantized_close_configs_recheck.json').write_text(json.dumps(result,indent=2)+'\n');print('QUANTIZED_RECHECK',json.dumps(result),flush=True)
