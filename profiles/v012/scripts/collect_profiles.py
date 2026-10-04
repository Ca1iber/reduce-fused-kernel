from pathlib import Path
import subprocess,json,time,os,signal,re,shutil
root=Path(__file__).resolve().parents[1];status={'stage':'tracer','started':time.time()}
def save():(root/'meta/profile_status.json').write_text(json.dumps(status,indent=2)+'\n')
save();trace=root/'raw/trace_comparison';trace.mkdir(exist_ok=True)
with (root/'logs/trace_comparison.log').open('w') as log:
    p=subprocess.run(['/opt/maca/bin/mcTracer','--mctx','--odname','trace_comparison','--name','k_pipeline','/opt/conda/bin/python',str(root/'scripts/profile_cases.py')],cwd=root/'raw',stdout=log,stderr=subprocess.STDOUT)
status['tracer_exitcode']=p.returncode;save()
for name in ('baseline','shared_single','sync_ring2','async_ring2','async_ring3'):
    status['stage']='native_'+name;save()
    folder=root/'raw'/('native_'+name);folder.mkdir(exist_ok=True)
    cmd=f'env MCTX_TARGET_PROFILE_PATH={folder} /opt/conda/bin/python {root}/scripts/profile_cases.py --case {name}'
    metrics=['Total Instructions','Compute Instructions','Memory Instructions','Total Cycles','Global Memory Read bytes','Global Memory Write bytes','ISU stall cycles layout','load instructions','store instructions','average conflict cycles per instruction','shared memory access efficiency','Achieved waves','Dispatched waves']
    command=['/opt/mcProfiler-ubuntu18.04/mcProfiler','perf_exec','--cmdline',cmd,'--kernelname','v012_'+name,'--casename','v012_'+name,'--cwd',str(folder),'--custom','--kernelnames','reduce_fused_kernel_kernel','--counts','1','--metrics',*metrics]
    with (root/f'logs/native_{name}.log').open('w') as log:
        p=subprocess.Popen(command,cwd='/opt/mcProfiler-ubuntu18.04',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status['native_pid']=p.pid;save()
        try:code=p.wait(timeout=240)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);code=-1
    status[name+'_exitcode']=code;save()
    content=(root/f'logs/native_{name}.log').read_text(errors='replace');found=re.findall(r'output path is: (\S+)',content)
    if found and Path(found[-1]).exists():shutil.copytree(found[-1],folder/'tool_output',dirs_exist_ok=True)
status['stage']='profile_done';status['finished']=time.time();save()
