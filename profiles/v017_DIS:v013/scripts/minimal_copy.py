from pathlib import Path
import torch,tilelang,json
from tilelang import language as T
r=Path(__file__).resolve().parents[1]
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True})
def build(order:int,cta:bool,poison:bool):
 @T.prim_func
 def minimal(x:T.Tensor[(2,256),T.int32],out:T.Tensor[(2,256),T.int32]):
  with T.Kernel(1,threads=128):
   shared=T.alloc_shared((2,256),T.int32)
   tickets=T.alloc_buffer((2,),"uint64",scope="local.barrier",annotations={"barrier_type":"b64vectype"})
   if poison:
    T.fill(shared,-999)
    T.sync_threads()
   T.maca_async_copy(x[0,:],shared[0,:],barrier=tickets[0],annotations={"coalesced_width":2})
   if order==0:
    T.maca_async_copy(x[1,:],shared[1,:],barrier=tickets[1],annotations={"coalesced_width":2})
   T.maca_barrier_arrive_and_wait(tickets[0])
   if cta:T.sync_threads()
   if order==1:
    T.maca_async_copy(x[1,:],shared[1,:],barrier=tickets[1],annotations={"coalesced_width":2})
   T.copy(shared[0,:],out[0,:],coalesced_width=2)
   if order==2:
    T.maca_async_copy(x[1,:],shared[1,:],barrier=tickets[1],annotations={"coalesced_width":2})
   T.maca_barrier_arrive_and_wait(tickets[1])
   if cta:T.sync_threads()
   T.copy(shared[1,:],out[1,:],coalesced_width=2)
 return minimal
x=torch.arange(512,dtype=torch.int32,device='cuda').reshape(2,256)+10000;out=torch.empty_like(x);rows=[]
for poison in (False,True):
 for order in (0,1,2):
  for cta in (False,True):
   k=build(order,cta,poison);name=f'toy_order{order}_cta{int(cta)}_poison{int(poison)}'
   (r/f'codegen/{name}.cu').write_text(k.get_kernel_source());results=[]
   for rep in range(10):
    out.fill_(-777);k(x,out);torch.cuda.synchronize()
    results.append(dict(wrong_per_row=[int((out[i]!=x[i]).sum())for i in range(2)],poison_count=int((out==-999).sum()),first_per_row=[out[i,:12].tolist()for i in range(2)]))
   rows.append(dict(name=name,results=results));(r/'raw/minimal_copy.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,[z['wrong_per_row']for z in results],flush=True)
(r/'meta/minimal_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
