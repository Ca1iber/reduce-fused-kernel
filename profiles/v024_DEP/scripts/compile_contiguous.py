from pathlib import Path
import json,subprocess,re,collections
from store128_contiguous import get_reduce_fused_kernel
r=Path(__file__).resolve().parents[1]
k=get_reduce_fused_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,False,tile_hidden=1024,num_threads=64,grid_hidden_first=True,vector_store=True,contiguous_layout=True)
f=r/'codegen/fp8_prefill_contiguous128.cu';f.write_text(k.get_kernel_source())
cmd=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','--offload-arch=xcore1000','-O3','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__','-mllvm','-print-after=stack-frame-layout','-mllvm','-filter-print-funcs=reduce_fused_kernel_kernel',str(f),'-o',str(f.with_suffix('.mcbin'))]
result=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(r/'logs/compile_contiguous_native.log').write_text(result.stdout+result.stderr)
assert result.returncode==0
text=result.stderr[result.stderr.rfind('*** IR Dump After'):];(r/'codegen/fp8_prefill_contiguous128_final.mir').write_text(text)
ops=dict(collections.Counter(re.findall(r'\b(?:LDG|STG)_[A-Z]?\d+\b',text)));assert ops.get('STG_B128',0)==1,ops
(r/'analysis/contiguous128_static.json').write_text(json.dumps({'command':cmd,'opcodes':ops,'single_128bit_store_confirmed':True},indent=2)+'\n')
(r/'meta/contiguous_compile_status.json').write_text(json.dumps({'done':True,'opcodes':ops},indent=2)+'\n');print('CONFIRMED',ops,flush=True)
