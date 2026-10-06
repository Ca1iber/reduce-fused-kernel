from pathlib import Path
import subprocess,json,time,os,signal,re,shutil
r=Path(__file__).resolve().parents[1];status=[]
metrics=['Total Instructions','Compute Instructions','Memory Instructions','Total Cycles','Global Memory Read bytes','Global Memory Write bytes','ISU stall cycles layout']
for name in ['v010','grid_hidden_first']:
 folder=r/f'raw/profile_h7168_{name}';folder.mkdir(exist_ok=True)
 cmd=f'env MCTX_TARGET_PROFILE_PATH={folder} /opt/conda/bin/python {r}/scripts/profile_grid.py --workload h7168 --case {name}'
 command=['/opt/mcProfiler-ubuntu18.04/mcProfiler','perf_exec','--cmdline',cmd,'--kernelname','v020_'+name,'--casename','v020_'+name,'--cwd',str(folder),'--custom','--per-kernel','--kernelnames','reduce_fused_kernel_kernel','--counts','1','--metrics',*metrics]
 record={'case':name,'command':command,'started':time.time()}
 (r/'meta/profile_status.json').write_text(json.dumps({'running':name,'completed':status},indent=2)+'\n')
 with(r/f'logs/profile_h7168_{name}.log').open('w')as log:
  child=subprocess.Popen(command,cwd='/opt/mcProfiler-ubuntu18.04',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record['pid']=child.pid
  try:record['exitcode']=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM);record['exitcode']=-1;record['timeout']=True
 record['seconds']=time.time()-record['started']
 content=(r/f'logs/profile_h7168_{name}.log').read_text(errors='replace');paths=re.findall(r'output path is: (\S+)',content)
 if paths and Path(paths[-1]).exists():
  shutil.copytree(paths[-1],folder/'tool_output',dirs_exist_ok=True);record['output']=paths[-1]
 status.append(record);print('PROFILE_FINISHED',name,record,flush=True)
 (r/'meta/profile_status.json').write_text(json.dumps({'completed':status},indent=2)+'\n')
 if record['exitcode']:raise SystemExit(record['exitcode'])
(r/'meta/profile_status.json').write_text(json.dumps({'done':True,'completed':status},indent=2)+'\n')
