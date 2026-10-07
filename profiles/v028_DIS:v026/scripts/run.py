from pathlib import Path
import json,subprocess,os,time
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
env=dict(os.environ,MACA_PATH='/opt/maca',LD_LIBRARY_PATH='/opt/maca/lib:/opt/maca/lib64:'+os.environ.get('LD_LIBRARY_PATH',''),PATH='/opt/conda/bin:/opt/maca/bin:'+os.environ['PATH'],PYTHONPATH='/opt/tilelang-metax-v0.1.10:'+str(repo),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MALLOC_ARENA_MAX='2')
base=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','--offload-arch=xcore1000','-O3','-std=c++17',str(r/'scripts/empty_probe.cpp'),'--maca-path=/opt/maca']
for name,extra,target in [('library',['-shared','-fPIC'],r/'codegen/empty_probe.so'),('executable',[],r/'codegen/empty_probe')]:
 cmd=base+extra+['-o',str(target)];(r/'meta/status.json').write_text(json.dumps({'stage':'compile_'+name,'command':cmd},indent=2)+'\n');p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60);(r/f'logs/compile_{name}.log').write_text(p.stdout+p.stderr)
 if p.returncode:raise SystemExit(p.returncode)
(r/'meta/status.json').write_text(json.dumps({'stage':'native_events'},indent=2)+'\n')
with(r/'raw/native_events_16g.csv').open('w')as out,(r/'logs/native_events.log').open('w')as err:
 p=subprocess.run([str(r/'codegen/empty_probe')],env=env,stdout=out,stderr=err,timeout=60)
if p.returncode:raise SystemExit(p.returncode)
print('NATIVE_EVENTS_DONE',flush=True)
(r/'meta/status.json').write_text(json.dumps({'stage':'profiler_benchmark'},indent=2)+'\n')
with(r/'logs/profiler_benchmark.log').open('w')as out:
 p=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/benchmark.py')],cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=180)
if p.returncode:raise SystemExit(p.returncode)
(r/'meta/gpu_after.txt').write_text(subprocess.check_output(['/usr/bin/mx-smi'],text=True))
(r/'meta/status.json').write_text(json.dumps({'done':True,'native_event_batches':[128,1024,8192],'native_rounds':5,'benchmark_rounds':5,'profile_cases':6},indent=2)+'\n');print('EMPTY_KERNEL_MEASUREMENT_COMPLETE',flush=True)
