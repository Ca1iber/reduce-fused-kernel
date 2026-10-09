from pathlib import Path
import ctypes
root=Path('/data/TileOPs-Metax')
probe=ctypes.CDLL(str(root/'profiles/v028_DIS:v026/codegen/empty_probe.so'))
probe.launch_empty.argtypes=[ctypes.c_int];probe.launch_empty.restype=ctypes.c_int
runtime=ctypes.CDLL('/opt/maca/lib/libmcruntime.so')
for name in ['mcProfilerStart','mcProfilerStop','mcDeviceSynchronize']:
    getattr(runtime,name).restype=ctypes.c_int
for _ in range(10):
    if probe.launch_empty(512):raise RuntimeError('Empty launch failed')
if runtime.mcDeviceSynchronize():raise RuntimeError('Sync failed')
if runtime.mcProfilerStart():raise RuntimeError('Profiler start failed')
if probe.launch_empty(512):raise RuntimeError('Empty launch failed')
if runtime.mcDeviceSynchronize():raise RuntimeError('Sync failed')
if runtime.mcProfilerStop():raise RuntimeError('Profiler stop failed')
print('EMPTY_COUNTER_CONTROL_DONE',flush=True)
