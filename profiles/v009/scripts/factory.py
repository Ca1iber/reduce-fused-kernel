from pathlib import Path
import importlib.util,sys
from candidate_kernel import get_candidate_kernel
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_v009_baseline',root/'codegen/baseline_v004.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
baseline=module.get_reduce_fused_kernel
def factory(mode,threads,tile):
    def get_kernel(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf):
        if with_sf:
            return baseline(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf)
        if mode=='full':
            block_hidden=hidden
        elif hidden>1024 and hidden%tile==0:
            block_hidden=tile
        else:
            return baseline(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf)
        return get_candidate_kernel(hidden,num_topk,in_dtype,out_dtype,with_sf,with_weights,with_x_sf,block_hidden,threads)
    return get_kernel
