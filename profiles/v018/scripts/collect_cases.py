from pathlib import Path
import sys,json,torch,importlib,hashlib,time
repo=Path('/data/TileOPs-Metax');r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'profiles/v016/scripts'));sys.path.insert(0,str(repo/'profiles/v015/scripts'))
from prefetch_kernel import get_prefetch_kernel
from hidden_fragment import get_hidden_kernel
from tileops.kernels.moe.reduce_fused import get_reduce_fused_kernel
from workloads.moe import MoeReduceFusedWorkload
cases=[('fp8_b512_t128',False,True,512,128,0),('fp8_reg2_b512_t128',False,True,512,128,2),('fp8_reg4_b512_t128',False,True,512,128,4),('fp8_reg8_b512_t128',False,True,512,128,8),('fp8_b512_t256',False,True,512,256,0),('fp8_reg8_b512_t256',False,True,512,256,8),('fp8_b1024_t128',False,True,1024,128,0),('fp8_reg8_b1024_t128',False,True,1024,128,8),('base_full_t256',False,False,7168,256,0),('base_reg8_full_t256',False,False,7168,256,8),('xsf_full_t256',True,False,7168,256,0),('xsf_reg8_full_t256',True,False,7168,256,8),('quant_b512_t128',True,True,512,128,0),('quant_reg8_b512_t128',True,True,512,128,8),('hidden_double_fragment',False,True,512,128,-3),('hidden_double_shared',False,True,512,128,-2)]
rows=[]
for name,xsf,sf,tile,threads,depth in cases:
 print('START',name,flush=True);torch.manual_seed(1235)
 values=MoeReduceFusedWorkload(512,8,7168,torch.bfloat16,with_x_sf=xsf,with_sf=sf).gen_inputs();x,pos,w=values[:3];scale=values[-1]if sf else torch.empty(1,device='cuda');xs=values[3]if xsf else torch.empty(x.shape[0],device='cuda');out_dtype='float8_e4m3fn'if sf else 'bfloat16';out=torch.empty((512,7168),device='cuda',dtype=getattr(torch,out_dtype))
 args=(7168,8,'bfloat16',out_dtype,sf,True,xsf)
 base=get_reduce_fused_kernel(*args,tile_hidden=tile,num_threads=threads);base(x,w,pos,out,scale,xs);torch.cuda.synchronize();expected=out.view(torch.uint8).clone()
 if depth==0:k=base
 elif depth>0:k=get_prefetch_kernel(*args,tile_hidden=tile,num_threads=threads,prefetch_rows=depth)
 else:k=get_hidden_kernel(*args,tile_hidden=tile,num_threads=threads,mode=-depth,group_size=2)
 source=k.get_kernel_source();(r/f'codegen/{name}.cu').write_text(source)
 mismatches=[]
 for _ in range(10):k(x,w,pos,out,scale,xs)
 torch.cuda.synchronize();mismatches.append(int((out.view(torch.uint8)!=expected).sum()))
 torch.cuda.profiler.start();k(x,w,pos,out,scale,xs);torch.cuda.synchronize();torch.cuda.profiler.stop()
 mismatches.append(int((out.view(torch.uint8)!=expected).sum()))
 rows.append(dict(name=name,with_x_sf=xsf,with_sf=sf,tile=tile,threads=threads,prefetch_rows=depth,wrong_bytes=mismatches,shared_bytes=16384 if depth==-2 else 0,source_sha256=hashlib.sha256(source.encode()).hexdigest()))
 (r/'meta/cases.json').write_text(json.dumps(rows,indent=2)+'\n');print('CASE',rows[-1],flush=True)
 del values,x,pos,w,scale,xs,out,expected,k,base;torch.cuda.empty_cache()
(r/'meta/trace_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
