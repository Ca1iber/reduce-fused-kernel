from pathlib import Path
import sys,os,time,json,csv,statistics,random,ctypes,gc,importlib.util,contextlib,subprocess,re
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
assert json.loads((r/'meta/status.json').read_text()).get('done'),'sweep incomplete'
sys.path.insert(0,str(r/'scripts'));sys.path.insert(0,str(repo/'profiles/v029_ACC/scripts'))
import torch,pytest
from tilelang import language as T
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from tiny_pair import get_pair_kernel
from tiny_group_static import get_tiny_group_static
from tiny_local import get_tiny_local_kernel
torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v030_base',r/'codegen/baseline_v025.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
sweep=list(csv.DictReader((r/'raw/sweep_16g.csv').open()));assert len(sweep)==24
oldselect=json.loads((repo/'profiles/v029_ACC/meta/selected.json').read_text());selected={};results=[];queries=[]
variants=[('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]
for variant,sf,xsf in variants:
 z=max([z for z in sweep if z['variant']==variant],key=lambda z:float(z['speedup']));selected[variant]=z
 (r/'meta/final_status.json').write_text(json.dumps({'stage':'confirm','variant':variant,'completed':len(results)},indent=2)+'\n')
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();outdtype=torch.float8_e4m3fn if sf else torch.float16;out=torch.empty((32,256),device='cuda',dtype=outdtype);wrapper=base.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf)
 prior=oldselect[variant]
 if prior['family']=='formal_threads':old=base.get_reduce_fused_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,tile_hidden=256,num_threads=int(prior['threads']),vector_store=sf)
 elif prior['family']=='local_vector':old=get_tiny_local_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,tokens_per_cta=int(prior['tokens_per_cta']),num_threads=int(prior['threads']))
 else:old=get_tiny_group_static(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,tokens_per_cta=int(prior['tokens_per_cta']),num_threads=int(prior['threads']))
 new=get_pair_kernel(256,2,T.float16,T.dtype(outdtype),sf,True,xsf,num_threads=int(z['threads']),pair_layout=z['pair_layout'])
 kernels={'v025':wrapper.kernel,'v029':old,'v030':new}
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 call(wrapper.kernel,inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
 for label,k in kernels.items():call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),(variant,label)
 samples={label:[]for label in kernels}
 for turn in range(20):
  labels=list(kernels);labels=labels[turn%3:]+labels[:turn%3]
  if turn%2:labels.reverse()
  for label in labels:
   def fn(*v):call(kernels[label],v)
   us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[label].append(us);gc.collect();libc.malloc_trim(0)
 med={label:statistics.median(values)for label,values in samples.items()};row=dict(variant=variant,pair_layout=z['pair_layout'],threads=int(z['threads']),ctas=32,elements_per_pair=int(z['elements_per_pair']),v025_us=med['v025'],v029_us=med['v029'],v030_us=med['v030'],reduction_vs_v025_pct=(1-med['v030']/med['v025'])*100,reduction_vs_v029_pct=(1-med['v030']/med['v029'])*100,byte_equal=True,rounds=20)
 for reference in ['v025','v029']:
  diff=[a-b for a,b in zip(samples[reference],samples['v030'])];rng=random.Random(1235);boot=sorted(statistics.mean(rng.choices(diff,k=20))for _ in range(5000));row[reference+'_paired_saved_mean_us']=statistics.mean(diff);row[reference+'_ci95_low_us']=boot[125];row[reference+'_ci95_high_us']=boot[4874]
 results.append(row);(r/f'raw/confirmation_{variant.lower()}_samples.json').write_text(json.dumps(samples,indent=2)+'\n');(r/'raw/confirmation_16g.json').write_text(json.dumps(results,indent=2)+'\n');print('CONFIRMED',json.dumps(row),flush=True)
 for label,k in kernels.items():
  name=variant.lower()+'_'+label;p=r/f'codegen/confirm_{name}.cu';p.write_text(k.get_kernel_source());queries.append({'name':name,'source':str(p.relative_to(r)),'threads':wrapper.config['num_threads']if label=='v025'else int(prior['threads'])if label=='v029'else int(z['threads'])})
 del kernels,wrapper,old,new,k,inputs,out,expected,fn;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/selected.json').write_text(json.dumps(selected,indent=2)+'\n')
import importlib
m=importlib.import_module('tileops.kernels.moe.reduce_fused');original=m.get_reduce_fused_kernel;calls=[]
def factory(h,k,ind,outd,sf,weights,xsf,**kwargs):
 if h==256 and k==2:
  variant='Quantized'if sf and xsf else'FP8'if sf else'XSF'if xsf else'Base';z=selected[variant];calls.append({'variant':variant,'dtype':str(ind),'threads':int(z['threads']),'layout':z['pair_layout']})
  return get_pair_kernel(h,k,ind,outd,sf,weights,xsf,num_threads=int(z['threads']),pair_layout=z['pair_layout'],num_tokens_hint=None)
 return original(h,k,ind,outd,sf,weights,xsf,**kwargs)
m.get_reduce_fused_kernel=factory;(r/'meta/final_status.json').write_text(json.dumps({'stage':'correctness','completed':len(results)},indent=2)+'\n')
with(r/'logs/correctness.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/correctness.xml')])
m.get_reduce_fused_kernel=original;(r/'meta/validation.json').write_text(json.dumps({'exitcode':int(code),'candidate_kernel_constructions':len(calls),'candidate_calls':calls},indent=2)+'\n');assert code==0,'correctness failed';print('CORRECTNESS_PASSED',len(calls),flush=True)
resources=[]
for z in queries:
 p=r/z['source'];target=p.with_suffix('.mcbin');cmd=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','-O3','-lineinfo','--offload-arch=xcore1000','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__',str(p),'-o',str(target)];q=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(r/f"logs/resource_{z['name']}.log").write_text(q.stdout+q.stderr)
 if q.returncode:resources.append({**z,'compile_error':q.stderr});continue
 symbol=re.search(r'extern "C" __global__ void ([^(]+)',p.read_text()).group(1);q=subprocess.run([str(repo/'profiles/v018_DIS:v016/scripts/resource_query'),str(target),symbol,str(z['threads']),'0'],capture_output=True,text=True,timeout=30);resources.append({**z,'resource':json.loads(q.stdout)if q.returncode==0 else{'error':q.stderr}});(r/'raw/resources.json').write_text(json.dumps(resources,indent=2)+'\n')
(r/'meta/final_status.json').write_text(json.dumps({'done':True,'confirmations':4,'correctness_exitcode':int(code),'resources':len(resources)},indent=2)+'\n');print('PAIR_VALIDATION_DONE',flush=True)
