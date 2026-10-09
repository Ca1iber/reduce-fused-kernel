from pathlib import Path
import argparse,importlib.util,sys,json,torch
from workloads.moe import MoeReduceFusedWorkload
root=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--version',choices=['v000','v020'],required=True);a.add_argument('--variant',choices=['base','fp8'],required=True);a.add_argument('--workload',choices=['tiny','h3072','h7168','prefill'],required=True);args=a.parse_args()
t,k,h,dtype={'tiny':(32,2,256,torch.float16),'h3072':(512,8,3072,torch.bfloat16),'h7168':(512,8,7168,torch.bfloat16),'prefill':(4096,8,7168,torch.bfloat16)}[args.workload]
folder=root/(args.version+'-roofline');name=f'{args.version}_{args.variant}_{args.workload}_T{t}_K{k}_H{h}'
spec=importlib.util.spec_from_file_location('_roofline_'+args.version,folder/'codegen/source.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
torch.set_num_threads(1);torch.manual_seed(1235);sf_flag=args.variant=='fp8'
inputs=MoeReduceFusedWorkload(t,k,h,dtype,with_sf=sf_flag).gen_inputs();x,pos,weights=inputs[:3]
out=torch.empty((t,h),device='cuda',dtype=torch.float8_e4m3fn if sf_flag else dtype)
sf=inputs[-1] if sf_flag else torch.empty(1,device='cuda');unused=torch.empty(x.shape[0],device='cuda')
wrapper=module.MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf_flag)
code=wrapper.kernel.get_kernel_source();(folder/f'codegen/{name}.cu').write_text(code)
meta={'version':args.version,'variant':args.variant,'workload':args.workload,'T':t,'K':k,'H':h,'input_dtype':str(dtype),'config':wrapper.config,'grid_hidden_first':getattr(wrapper,'grid_hidden_first',False),'expected_x_bytes':t*k*h*x.element_size(),'expected_out_bytes':out.numel()*out.element_size()}
(folder/f'raw/{name}').mkdir(exist_ok=True);(folder/f'raw/{name}/case.json').write_text(json.dumps(meta,indent=2)+'\n')
# Drain input generation; native collection starts only inside the marked ROI.
torch.cuda.synchronize()
for _ in range(10):wrapper.kernel(x,weights,pos,out,sf,unused)
torch.cuda.synchronize();torch.cuda.profiler.start()
wrapper.kernel(x,weights,pos,out,sf,unused)
torch.cuda.synchronize();torch.cuda.profiler.stop()
print('PROFILE_DONE',name,flush=True)
