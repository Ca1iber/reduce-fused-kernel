from pathlib import Path
import json,traceback,torch,tilelang
from tilelang import language as T
root=Path(__file__).resolve().parents[1]
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True})
def get_probe():
    @T.prim_func
    def probe(x:T.Tensor[(8192,),T.float16],out:T.Tensor[(8192,),T.float16]):
        with T.Kernel(8,threads=128) as bx:
            buffer=T.alloc_shared((1024,),T.float16)
            ticket=T.alloc_buffer((1,), "void", scope="local.barrier", annotations={"barrier_type":"b128vectype"})
            T.maca_async_copy(x[bx*1024:(bx+1)*1024],buffer,barrier=ticket)
            T.maca_barrier_arrive_and_wait(ticket)
            T.sync_threads()
            T.copy(buffer,out[bx*1024:(bx+1)*1024])
    return probe
try:
    kernel=get_probe()
    code=kernel.get_kernel_source();(root/'codegen/async_probe_annotated.cu').write_text(code)
    x=torch.randn(8192,device='cuda',dtype=torch.float16);out=torch.empty_like(x)
    kernel(x,out);torch.cuda.synchronize();assert torch.equal(x,out)
    status=dict(status='passed',generated_memcpy_async='memcpy_async<' in code,generated_wait='barrier_arrive_and_wait' in code)
except Exception as e:
    status=dict(status='failed',error=repr(e));traceback.print_exc()
(root/'meta/async_probe_annotated.json').write_text(json.dumps(status,indent=2)+'\n');print('ASYNC_PROBE',json.dumps(status),flush=True)
