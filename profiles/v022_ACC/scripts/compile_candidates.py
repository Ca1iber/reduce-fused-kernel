from pathlib import Path
import json,subprocess,torch,re
import importlib.util,sys
from vector_store import get_reduce_fused_kernel
r=Path(__file__).resolve().parents[1];records=[]
spec=importlib.util.spec_from_file_location('_v022_fixed_baseline',r/'codegen/baseline_formal.py');baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
MoeReduceFusedKernel=baseline.MoeReduceFusedKernel
for label,t,h in [('tiny',32,256),('h3072',512,3072),('h7168',512,7168),('prefill',4096,7168)]:
 topk=2 if label=='tiny'else 8;dtype=torch.float16 if label=='tiny'else torch.bfloat16;ind='float16'if label=='tiny'else'bfloat16'
 for variant in ['base','xsf','fp8','quantized']:
  sf=variant in ['fp8','quantized'];xsf=variant in ['xsf','quantized']
  base=MoeReduceFusedKernel(t,topk,h,dtype,with_sf=sf,with_x_sf=xsf)
  kernels={'baseline':base.kernel}
  if sf:kernels['vector_store']=get_reduce_fused_kernel(h,topk,ind,'float8_e4m3fn',sf,True,xsf,grid_hidden_first=base.grid_hidden_first,**base.config)
  for name,k in kernels.items():
   case=f'{variant}_{label}_{name}';f=r/f'codegen/{case}.cu';f.write_text(k.get_kernel_source())
   records.append({'case':case,'variant':variant,'workload':label,'implementation':name,'config':base.config,'grid_hidden_first':base.grid_hidden_first})
   print('GENERATED',case,flush=True)
(r/'meta/generated_cases.json').write_text(json.dumps(records,indent=2)+'\n')
# Static compilation: final machine IR dump at the last stack-frame pass, no kernel launch.
for record in records:
 if record['case']not in ['base_h7168_baseline','xsf_h7168_baseline','fp8_h7168_baseline','quantized_h7168_baseline','fp8_prefill_baseline','quantized_prefill_baseline','fp8_prefill_vector_store','quantized_prefill_vector_store']:continue
 case=record['case'];source=r/f'codegen/{case}.cu'
 command=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','--offload-arch=xcore1000','-O3','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__','-mllvm','-print-after=stack-frame-layout','-mllvm','-filter-print-funcs=reduce_fused_kernel_kernel',str(source),'-o',str(source.with_suffix('.mcbin'))]
 q=subprocess.run(command,capture_output=True,text=True,timeout=60)
 (r/f'logs/compile_{case}.log').write_text(q.stdout+q.stderr)
 if q.returncode:raise RuntimeError(case+' native compile failed')
 text=q.stderr;start=text.rfind('*** IR Dump After')
 (r/f'codegen/{case}_final.mir').write_text(text[start:]if start>=0 else text)
 record['native_command']=command;print('NATIVE_COMPILED',case,flush=True)
(r/'meta/generated_cases.json').write_text(json.dumps(records,indent=2)+'\n')
(r/'meta/static_status.json').write_text(json.dumps({'done':True,'generated':len(records)},indent=2)+'\n')
