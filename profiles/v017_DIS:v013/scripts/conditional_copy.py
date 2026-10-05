from pathlib import Path
import torch,tilelang,json
from tilelang import language as T
r=Path(__file__).resolve().parents[1]
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True})
def build(order:int,cta:bool,poison:bool,blocks:int,conditional:bool):
 @T.prim_func
 def minimal(x:T.Tensor[(blocks,2,256),T.int32],p:T.Tensor[(2,),T.int32],out:T.Tensor[(blocks,2,256),T.int32]):
  with T.Kernel(blocks,threads=128) as bx:
   shared=T.alloc_shared((2,256),T.int32)
   tickets=T.alloc_buffer((2,),"uint64",scope="local.barrier",annotations={"barrier_type":"b64vectype"})
   if poison:
    T.fill(shared,-999)
    T.sync_threads()
   if not conditional or p[0]>=0:
    T.maca_async_copy(x[bx,0,:],shared[0,:],barrier=tickets[0],annotations={"coalesced_width":2})
   if order==0:
    if not conditional or p[1]>=0:
     T.maca_async_copy(x[bx,1,:],shared[1,:],barrier=tickets[1],annotations={"coalesced_width":2})
   if not conditional or p[0]>=0:
    T.maca_barrier_arrive_and_wait(tickets[0])
   if cta:T.sync_threads()
   if order==1:
    if not conditional or p[1]>=0:
     T.maca_async_copy(x[bx,1,:],shared[1,:],barrier=tickets[1],annotations={"coalesced_width":2})
   T.copy(shared[0,:],out[bx,0,:],coalesced_width=2)
   if order==2:
    if not conditional or p[1]>=0:
     T.maca_async_copy(x[bx,1,:],shared[1,:],barrier=tickets[1],annotations={"coalesced_width":2})
   if not conditional or p[1]>=0:
    T.maca_barrier_arrive_and_wait(tickets[1])
   if cta:T.sync_threads()
   T.copy(shared[1,:],out[bx,1,:],coalesced_width=2)
 return minimal
rows=[]
for blocks in (1,7168):
 x=torch.arange(blocks*512,dtype=torch.int32,device='cuda').reshape(blocks,2,256)+10000;out=torch.empty_like(x);pos=torch.tensor([0,1],dtype=torch.int32,device='cuda')
 for conditional in (False,True):
  for order in (0,1):
   for cta in (False,True):
    k=build(order,cta,True,blocks,conditional);name=f'conditional_b{blocks}_if{int(conditional)}_order{order}_cta{int(cta)}'
    (r/f'codegen/{name}.cu').write_text(k.get_kernel_source());results=[]
    for rep in range(5):
     out.fill_(-777);k(x,pos,out);torch.cuda.synchronize()
     results.append(dict(wrong_per_row=[int((out[:,i]!=x[:,i]).sum())for i in range(2)],poison_count=int((out==-999).sum()),first_per_row=[out[0,i,:12].tolist()for i in range(2)]))
    rows.append(dict(name=name,results=results));(r/'raw/conditional_copy.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,[z['wrong_per_row']for z in results],flush=True)
(r/'meta/conditional_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
