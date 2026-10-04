from pathlib import Path
import argparse,json,re,torch
from workloads.moe import MoeReduceFusedWorkload
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
from benchmarks.benchmark_base import bench_kernel,_bench_meta
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--variant',default='fp8');parser.add_argument('--mode',choices=('bench','profile'),default='profile');args=parser.parse_args()
sf=args.variant in ('fp8','quantized');xsf=args.variant in ('xsf','quantized')
torch.manual_seed(1235)
values=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_x_sf=xsf,with_sf=sf).gen_inputs()
obj=MoeReduceFusedKernel(512,8,7168,torch.bfloat16,with_x_sf=xsf,with_sf=sf)
out=torch.empty((512,7168),device='cuda',dtype=obj.out_dtype)
unused_sf=torch.empty(1,device='cuda');unused_xsf=torch.empty(values[0].shape[0],device='cuda')
def call(*values):obj.kernel(values[0],values[2],values[1],out,values[-1] if sf else unused_sf,values[3] if xsf else unused_xsf)
code=obj.kernel.get_kernel_source()
(root/f'codegen/baseline_{args.variant}_h7168.cu').write_text(code)
line=next(line for line in code.splitlines() if '__global__' in line and 'void' in line)
symbols=[s for s in re.findall(r'(\w+)\s*\(',line) if not s.startswith('__')]
metadata=dict(variant=args.variant,shape=[512,8,7168],config=obj.config,kernel_symbols=symbols,expected_input_bytes=512*8*7168*2,expected_output_bytes=out.numel()*out.element_size())
(root/f'meta/baseline_{args.variant}.json').write_text(json.dumps(metadata,indent=2)+'\n')
for _ in range(10):call(*values)
torch.cuda.synchronize()
if args.mode=='bench':
    us=bench_kernel(call,args=values,n_warmup=10,n_repeat=50,n_trials=3)*1000
    metadata.update(latency_us=us,timing=getattr(_bench_meta,'timing','unknown'))
    (root/f'raw/baseline_bench_{args.variant}.json').write_text(json.dumps(metadata,indent=2)+'\n')
else:
    torch.cuda.profiler.start();call(*values);torch.cuda.synchronize();torch.cuda.profiler.stop()
print('BASELINE_DONE',json.dumps(metadata),flush=True)
