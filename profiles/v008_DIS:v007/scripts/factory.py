from pathlib import Path
import importlib.util, sys
from tile_kernel import get_split_kernel
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_v008_baseline', root/'codegen/baseline_v004.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
baseline = module.get_reduce_fused_kernel
def factory(tile, threads):
    def get_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf):
        if not (with_sf and str(out_dtype) == 'float8_e4m3fn') or hidden <= 1024:
            return baseline(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf)
        if hidden % tile:
            return baseline(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf)
        return get_split_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf, tile, threads)
    return get_kernel
