from pathlib import Path
import importlib.util,sys
ROOT=Path('/root/TileOPs-Metax/profiles')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.get_reduce_fused_kernel
naive_factory=load("_profile_naive_snapshot",ROOT/"v003/codegen/baseline_reduce_fused.py")
v003_factory=load("_profile_v003_snapshot",ROOT/"v004/codegen/baseline_v003.py")
