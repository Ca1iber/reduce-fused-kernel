from pathlib import Path
import importlib.util,sys,json,torch
from tileops.kernels.moe.reduce_fused import MoeReduceFusedKernel
r=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_pre_v022',r/'codegen/pre_integration_formal.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
rows=[]
for w,t,k,h,dtype in [('tiny',32,2,256,torch.float16),('h3072',512,8,3072,torch.bfloat16),('h7168',512,8,7168,torch.bfloat16),('prefill',4096,8,7168,torch.bfloat16)]:
 for name,sf,xsf in [('base',False,False),('xsf',False,True),('fp8',True,False),('quantized',True,True)]:
  a=old.MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf,with_x_sf=xsf);b=MoeReduceFusedKernel(t,k,h,dtype,with_sf=sf,with_x_sf=xsf)
  expected=sf and w in ['tiny','prefill'];assert b.vector_store==expected
  assert a.config==b.config and a.grid_hidden_first==b.grid_hidden_first
  sa=a.kernel.get_kernel_source();sb=b.kernel.get_kernel_source();equal=sa==sb
  if not expected:assert equal,(name,w,'unselected code changed')
  rows.append({'variant':name,'workload':w,'vector_store':b.vector_store,'grid_hidden_first':b.grid_hidden_first,'config':b.config,'cpp_equal_to_v020':equal})
for shape,config,weights in [((512,8,3072,torch.bfloat16),None,True),((32,2,256,torch.bfloat16),None,True),((4096,8,7168,torch.bfloat16),{'tile_hidden':512},True),((32,2,256,torch.float16),{'num_threads':64},True),((4096,8,7168,torch.bfloat16),None,False)]:
 b=MoeReduceFusedKernel(*shape,with_sf=True,with_weights=weights,config=config);assert not b.vector_store
(r/'meta/integration_dispatch.json').write_text(json.dumps({'measured_cases':rows,'fallback_checks':5,'selected_count':sum(z['vector_store']for z in rows),'unselected_cpp_identical':True},indent=2)+'\n');print('DISPATCH_CHECK_PASSED',flush=True)
