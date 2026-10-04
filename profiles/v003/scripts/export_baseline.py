from pathlib import Path
import hashlib,json,subprocess
import torch
from tileops.ops.moe import MoeReduceFusedFwdOp, MoeReduceFusedFp8FwdOp
root=Path('/root/TileOPs-Metax/profiles/v003')
for name in ('scripts','raw','analysis','codegen','logs','meta'):
 (root/name).mkdir(parents=True,exist_ok=True)
source=Path('/root/TileOPs-Metax/tileops/kernels/moe/reduce_fused.py')
(root/'codegen/baseline_reduce_fused.py').write_bytes(source.read_bytes())
meta=dict(head=subprocess.check_output(['git','-C','/root/TileOPs-Metax','rev-parse','HEAD'],text=True).strip(),kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,target=dict(T=512,K=8,H=7168,dtype='bfloat16'))
for label,cls in (('base',MoeReduceFusedFwdOp),('fp8',MoeReduceFusedFp8FwdOp)):
 op=cls(512,8,7168,torch.bfloat16)
 kernel=op.kernel.kernel
 text=kernel.get_kernel_source()
 (root/f'codegen/{label}_h7168.cu').write_text(text)
 print(label,'SOURCE_BYTES',len(text),flush=True)
 print('KERNEL_ATTRS',list(vars(kernel)),flush=True)
 for name,value in vars(kernel).items():
  if isinstance(value,str) and len(value)<500 and any(s in name.lower() for s in ('path','lib','cache')):print(name,value,flush=True)
 for name in ('adapter','lib'):
  value=getattr(kernel,name,None)
  if value is not None:print(name,type(value).__name__,list(vars(value)) if hasattr(value,'__dict__') else '',flush=True)
(root/'meta/baseline.json').write_text(json.dumps(meta,indent=2)+'\n')
