from pathlib import Path
import json,subprocess
from hidden_fragment import get_hidden_kernel
r=Path(__file__).resolve().parents[1];rows=[]
for variant,xsf in [('fp8',False),('quantized',True)]:
 k=get_hidden_kernel(7168,8,'bfloat16','float8_e4m3fn',True,True,xsf,tile_hidden=1024,num_threads=128,mode=3,group_size=2)
 p=r/f'codegen/double_prefill_{variant}.cu';p.write_text(k.get_kernel_source());o=p.with_suffix('.mcbin')
 cmd=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','-O3','-lineinfo','--offload-arch=xcore1000','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__',str(p),'-o',str(o)]
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(r/f'logs/query_double_prefill_{variant}.log').write_text(q.stdout+q.stderr);q.check_returncode()
 q=subprocess.run([str(r/'scripts/resource_query'),str(o),'reduce_fused_kernel_kernel','128','0'],capture_output=True,text=True,timeout=30);q.check_returncode();z=json.loads(q.stdout);rows.append({'variant':variant,'resource':z,'compiler_command':cmd});print('RESOURCE',variant,z,flush=True)
(r/'raw/double_prefill_resources.json').write_text(json.dumps(rows,indent=2)+'\n')
