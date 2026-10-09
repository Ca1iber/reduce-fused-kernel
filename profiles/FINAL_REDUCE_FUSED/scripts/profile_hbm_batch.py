from pathlib import Path
import argparse,importlib.util,sys,json,hashlib,re,time
import torch,tilelang
from workloads.moe import MoeReduceFusedWorkload
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--isolated-roi',action='store_true');parser.add_argument('--original-symbol',action='store_true');parser.add_argument('--warmup',type=int,default=10);args=parser.parse_args()
plan=json.loads(Path(args.plan).read_text())
torch.set_num_threads(1)
modules={}
for version in {x['version']for x in plan}:
    path=root/'codegen/hbm_native/v000.py' if version=='v000' else root/'codegen/final_reduce_fused.py'
    spec=importlib.util.spec_from_file_location('_native_hbm_'+version,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);modules[version]=mod
ready=[]
for case in plan:
    version,variant,workload=case['version'],case['variant'],case['workload']
    name=f'hbm_{version}_{variant}_{workload}'
    t,k,h,dtype={'tiny':(32,2,256,torch.float16),'h3072':(512,8,3072,torch.bfloat16),'h7168':(512,8,7168,torch.bfloat16),'prefill':(4096,8,7168,torch.bfloat16)}[workload]
    sf_flag=variant in ('fp8','quantized');xsf_flag=variant in ('xsf','quantized')
    torch.manual_seed(1235)
    values=MoeReduceFusedWorkload(t,k,h,dtype,with_sf=True,with_x_sf=True).gen_inputs()
    x,pos,weights,xsf,sf=values
    out=torch.empty((t,h),device='cuda',dtype=torch.float8_e4m3fn if sf_flag else dtype)
    wrapper=modules[version].MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf_flag,with_x_sf=xsf_flag)
    original=wrapper.kernel
    if args.original_symbol:
        if len(plan)!=1:raise RuntimeError('Original symbol mode requires one case')
        renamed=original
    else:
        renamed=tilelang.compile(original.prim_func.with_attr('global_symbol',name),
            execution_backend=original.execution_backend,target=original.target,
            target_host=original.target_host,pass_configs=original.pass_configs,
            compile_flags=original.compile_flags)
    original_code=original.get_kernel_source();code=renamed.get_kernel_source()
    symbols=lambda text:re.findall(r'extern "C" __global__ void (\w+)\(',text)
    old_symbol=symbols(original_code)[0];new_symbol=symbols(code)[0]
    old_normal=original_code.replace(old_symbol,'SAME_KERNEL_SYMBOL')
    new_normal=code.replace(new_symbol,'SAME_KERNEL_SYMBOL')
    if old_normal!=new_normal:raise RuntimeError('Renaming changed generated code: '+name)
    (root/'codegen/hbm_native'/f'{name}.cu').write_text(code)
    metadata=dict(case,case=name,kernel_symbol=new_symbol,T=t,K=k,H=h,input_dtype=str(dtype),
        config=wrapper.config,grid_hidden_first=getattr(wrapper,'grid_hidden_first',False),
        prefetch_rows=getattr(wrapper,'prefetch_rows',0),tiny_implementation=getattr(wrapper,'tiny_implementation',None),
        expected_x_bytes=t*k*h*x.element_size(),expected_out_bytes=out.numel()*out.element_size(),
        renamed_generated_code_identical=True,source_sha256=hashlib.sha256(Path(modules[version].__file__).read_bytes()).hexdigest())
    (root/'raw/hbm_native'/name).mkdir(exist_ok=True)
    (root/'raw/hbm_native'/name/'case.json').write_text(json.dumps(metadata,indent=2)+'\n')
    argv=(x,weights,pos,out,sf,xsf)
    torch.cuda.synchronize()
    for _ in range(args.warmup):renamed(*argv)
    torch.cuda.synchronize()
    ready.append((name,renamed,argv))
    print('READY',name,new_symbol,flush=True)
print('ROI_BEGIN',len(ready),flush=True)
if args.isolated_roi:
    for name,kernel,argv in ready:
        # Drain previous device work; keep a distinct counter window per case.
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        kernel(*argv)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print('ISOLATED_ROI_DONE',name,flush=True)
        time.sleep(0.1)
else:
    torch.cuda.profiler.start()
    for name,kernel,argv in ready:
        kernel(*argv)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
print('PROFILE_BATCH_DONE',len(ready),flush=True)
