from pathlib import Path
import sys,os,json,csv,time,statistics,random,importlib.util,ctypes,gc,subprocess,re,contextlib
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
assert json.loads((r/'meta/static_status.json').read_text()).get('done'),'static scan incomplete'
assert json.loads((r/'meta/local_status.json').read_text()).get('done'),'local scan incomplete'
sys.path.insert(0,str(r/'scripts'))
import torch,pytest
from tilelang import language as T
from workloads.moe import MoeReduceFusedWorkload
from benchmarks.benchmark_base import bench_kernel,_bench_meta
from tiny_group_static import get_tiny_group_static
from tiny_group import get_tiny_group_kernel
from tiny_local import get_tiny_local_kernel
torch.set_num_threads(1);libc=ctypes.CDLL(None)
spec=importlib.util.spec_from_file_location('_v029_confirm_base',r/'codegen/baseline_v025.py');base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
rows=[]
for f in ['static_sweep_16g.csv','local_sweep_16g.csv']:rows.extend(csv.DictReader((r/'raw'/f).open()))
variants=[('Base',False,False),('XSF',False,True),('FP8',True,False),('Quantized',True,True)]
selected={};confirmation=[];kernels_for_query=[]
def make_kernel(z,sf,xsf,ind=T.float16,outd=None,weights=True,generic=False):
 if outd is None:outd=T.float8_e4m3fn if sf else T.float16
 group=int(z['tokens_per_cta']);threads=int(z['threads'])
 if z['family']=='formal_threads':return base.get_reduce_fused_kernel(256,2,ind,outd,sf,weights,xsf,tile_hidden=256,num_threads=threads,vector_store=sf)
 if z['family']=='local_vector':return get_tiny_local_kernel(256,2,ind,outd,sf,weights,xsf,tokens_per_cta=group,num_threads=threads,num_tokens_hint=None if generic else 32)
 factory=get_tiny_group_kernel if generic else get_tiny_group_static
 return factory(256,2,ind,outd,sf,weights,xsf,tokens_per_cta=group,num_threads=threads)
for variant,sf,xsf in variants:
 ranked=sorted([z for z in rows if z['variant']==variant],key=lambda z:float(z['speedup']),reverse=True)
 choices=[ranked[0]]
 for subset in [[z for z in ranked if int(z['ctas'])<32],[z for z in ranked if int(z['threads'])<(128 if sf else 256)]]:
  if subset and all((subset[0]['family'],subset[0]['tokens_per_cta'],subset[0]['threads'])!=(z['family'],z['tokens_per_cta'],z['threads'])for z in choices):choices.append(subset[0])
 torch.manual_seed(1235);inputs=MoeReduceFusedWorkload(32,2,256,torch.float16,with_sf=True,with_x_sf=True).gen_inputs();out=torch.empty((32,256),dtype=torch.float8_e4m3fn if sf else torch.float16,device='cuda');wrapper=base.MoeReduceFusedKernel(32,2,256,torch.float16,with_sf=sf,with_x_sf=xsf)
 def call(k,v):k(v[0],v[2],v[1],out,v[4],v[3])
 call(wrapper.kernel,inputs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
 for index,z in enumerate(choices):
  name=f"{variant.lower()}_{z['family']}_g{z['tokens_per_cta']}_t{z['threads']}";(r/'meta/final_status.json').write_text(json.dumps({'stage':'confirm','running':name,'completed':len(confirmation)},indent=2)+'\n')
  k=make_kernel(z,sf,xsf);call(k,inputs);torch.cuda.synchronize();assert torch.equal(out.view(torch.uint8),expected),name
  samples={'baseline':[],'candidate':[]};pairdiff=[]
  for turn in range(20):
   values={}
   for label in list(samples)[::1 if turn%2==0 else -1]:
    def fn(*v):call(wrapper.kernel if label=='baseline'else k,v)
    us=bench_kernel(fn,args=inputs,n_warmup=10,n_repeat=50,n_trials=3)*1000;assert getattr(_bench_meta,'timing',None)=='cupti';samples[label].append(us);values[label]=us;gc.collect();libc.malloc_trim(0)
   pairdiff.append(values['baseline']-values['candidate'])
  a,b=[statistics.median(samples[label])for label in samples];rng=random.Random(1235);boot=sorted(statistics.mean(rng.choices(pairdiff,k=len(pairdiff)))for _ in range(5000))
  result=dict(variant=variant,family=z['family'],tokens_per_cta=int(z['tokens_per_cta']),threads=int(z['threads']),ctas=int(z['ctas']),baseline_us=a,candidate_us=b,latency_reduction_pct=(1-b/a)*100,paired_mean_saved_us=statistics.mean(pairdiff),paired_saved_ci95_low_us=boot[125],paired_saved_ci95_high_us=boot[4874],rounds=20,byte_equal=True)
  confirmation.append(result);(r/f'raw/confirm_{name}_samples.json').write_text(json.dumps({'samples':samples,'paired_saved_us':pairdiff,'confidence_method':'paired round bootstrap of mean saved microseconds;5000 resamples'},indent=2)+'\n')
  (r/'raw/confirmation_16g.json').write_text(json.dumps(confirmation,indent=2)+'\n');print('CONFIRMED',json.dumps(result),flush=True)
  source=k.get_kernel_source();(r/f'codegen/confirm_{name}.cu').write_text(source);kernels_for_query.append({'name':name,'source':f'codegen/confirm_{name}.cu','threads':int(z['threads'])})
  if index==0:selected[variant]=z
  del k,fn;gc.collect();torch.cuda.empty_cache();libc.malloc_trim(0)
 # Preserve formal baseline resource/codegen with this variant's actual configuration.
 p=r/f'codegen/formal_{variant.lower()}.cu';p.write_text(wrapper.kernel.get_kernel_source());kernels_for_query.append({'name':'formal_'+variant.lower(),'source':str(p.relative_to(r)),'threads':wrapper.config['num_threads']})
 del wrapper,inputs,out,expected;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();libc.malloc_trim(0)
(r/'meta/selected.json').write_text(json.dumps(selected,indent=2)+'\n')
# Existing suite exercises the same candidate templates with dynamic token tails and dtypes.
import importlib
m=importlib.import_module('tileops.kernels.moe.reduce_fused');original=m.get_reduce_fused_kernel;calls=[]
def factory(h,k,ind,outd,sf,weights,xsf,**kwargs):
 if h==256 and k==2:
  variant='Quantized'if sf and xsf else'FP8'if sf else'XSF'if xsf else'Base';z=selected[variant];calls.append({'variant':variant,'dtype':str(ind),'family':z['family'],'group':z['tokens_per_cta'],'threads':z['threads']})
  return make_kernel(z,sf,xsf,ind,outd,weights,generic=True)
 return original(h,k,ind,outd,sf,weights,xsf,**kwargs)
m.get_reduce_fused_kernel=factory;(r/'meta/final_status.json').write_text(json.dumps({'stage':'correctness','completed':len(confirmation)},indent=2)+'\n')
with(r/'logs/selected_correctness.log').open('w')as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):code=pytest.main(['-q','tests/ops/test_moe_reduce_fused.py','--junitxml='+str(r/'raw/selected_correctness.xml')])
m.get_reduce_fused_kernel=original;(r/'meta/selected_validation.json').write_text(json.dumps({'exitcode':int(code),'candidate_kernel_constructions':len(calls),'candidate_calls':calls},indent=2)+'\n');assert code==0,'correctness failed';print('CORRECTNESS_PASSED',len(calls),flush=True)
# Resource queries occur only after performance measurement.
resources=[]
for z in kernels_for_query:
 p=r/z['source'];target=p.with_suffix('.mcbin');cmd=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','-O3','-lineinfo','--offload-arch=xcore1000','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__',str(p),'-o',str(target)];q=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(r/f"logs/resource_compile_{z['name']}.log").write_text(q.stdout+q.stderr)
 if q.returncode:resources.append({**z,'compile_error':q.stderr});continue
 symbol=re.search(r'extern "C" __global__ void ([^(]+)',p.read_text()).group(1)
 q=subprocess.run([str(repo/'profiles/v018_DIS:v016/scripts/resource_query'),str(target),symbol,str(z['threads']),'0'],capture_output=True,text=True,timeout=30)
 resources.append({**z,'symbol':symbol,'resource':json.loads(q.stdout)if q.returncode==0 else{'error':q.stderr}})
 (r/'raw/kernel_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
(r/'meta/final_status.json').write_text(json.dumps({'done':True,'confirmations':len(confirmation),'correctness_exitcode':int(code),'resource_cases':len(resources)},indent=2)+'\n');print('FINAL_VALIDATION_DONE',flush=True)
