from pathlib import Path
import importlib.util,sys,json,torch
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
r=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_pre_v025',r/'codegen/pre_integration_formal.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
rows=[]
for w,t,k,h,dtype in [('tiny',32,2,256,torch.float16),('h3072',512,8,3072,torch.bfloat16),('h7168',512,8,7168,torch.bfloat16),('prefill',4096,8,7168,torch.bfloat16)]:
 for name,sf,xsf in [('base',False,False),('xsf',False,True),('fp8',True,False),('quantized',True,True)]:
  a=old.MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf,with_x_sf=xsf);b=MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf,with_x_sf=xsf)
  expected=sf and w in ['h3072','h7168'];assert b.prefetch_rows==(8 if expected else 0)
  assert a.vector_store==b.vector_store
  assert a.config==b.config and a.grid_hidden_first==b.grid_hidden_first
  sa=a.kernel.get_kernel_source();sb=b.kernel.get_kernel_source();equal=sa==sb
  if not expected:assert equal,(name,w,'unselected code changed')
  rows.append({'variant':name,'workload':w,'prefetch_rows':b.prefetch_rows,'vector_store':b.vector_store,'grid_hidden_first':b.grid_hidden_first,'config':b.config,'cpp_equal_to_v022':equal})
for shape,config,weights in [((512,8,3072,torch.float16),None,True),((256,8,3072,torch.bfloat16),None,True),((512,8,7168,torch.bfloat16),{'tile_hidden':1024},True),((512,8,3072,torch.bfloat16),{'num_threads':256},True),((512,8,7168,torch.bfloat16),None,False)]:
 b=MoeReduceFusedKernel(*shape,with_sf=True,with_weights=weights,config=config);assert b.prefetch_rows==0
(r/'meta/integration_dispatch.json').write_text(json.dumps({'measured_cases':rows,'fallback_checks':5,'selected_count':sum(z['prefetch_rows']>0 for z in rows),'unselected_cpp_identical':True},indent=2)+'\n');print('DISPATCH_CHECK_PASSED',flush=True)
