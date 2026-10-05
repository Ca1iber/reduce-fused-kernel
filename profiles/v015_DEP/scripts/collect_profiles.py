from pathlib import Path
import subprocess,json,time,re,shutil
r=Path(__file__).resolve().parents[1]
s={'stage':'tracer','started':time.time()}
def save():(r/'meta/profile_status.json').write_text(json.dumps(s,indent=2)+'\n')
save()
with (r/'logs/trace_comparison.log').open('w') as log:
 p=subprocess.run(['/opt/maca/bin/mcTracer','--mctx','--odname','trace_comparison','--name','hidden_pipeline','/opt/conda/bin/python',str(r/'scripts/profile_cases.py')],cwd=r/'raw',stdout=log,stderr=subprocess.STDOUT)
s['tracer_exitcode']=p.returncode;save()
for name in ('stream_async','group2_async'):
 s['stage']='native_'+name;save();d=r/'raw'/('native_'+name);d.mkdir(exist_ok=True)
 cmd=f'env MCTX_TARGET_PROFILE_PATH={d} /opt/conda/bin/python {r}/scripts/profile_cases.py --case {name}'
 metrics=['Total Instructions','Compute Instructions','Memory Instructions','Global Memory Read bytes','Global Memory Write bytes','ISU stall cycles layout','load instructions','store instructions','average conflict cycles per instruction','shared memory access efficiency']
 a=['/opt/mcProfiler-ubuntu18.04/mcProfiler','perf_exec','--cmdline',cmd,'--kernelname','v015_'+name,'--casename','v015_'+name,'--cwd',str(d),'--custom','--per-kernel','--kernelnames','reduce_fused_kernel_kernel','--counts','1','--metrics',*metrics]
 (r/f'meta/command_{name}.json').write_text(json.dumps(a,indent=2)+'\n')
 with (r/f'logs/native_{name}.log').open('w') as log:
  p=subprocess.Popen(a,cwd='/opt/mcProfiler-ubuntu18.04',stdout=log,stderr=subprocess.STDOUT)
  s['native_pid']=p.pid;save();code=p.wait()
 s[name+'_exitcode']=code;save()
 content=(r/f'logs/native_{name}.log').read_text(errors='replace');found=re.findall(r'output path is: (\S+)',content)
 if found and Path(found[-1]).exists():shutil.copytree(found[-1],d/'tool_output',dirs_exist_ok=True)
s['stage']='done';s['finished']=time.time();save()
