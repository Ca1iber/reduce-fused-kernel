from pathlib import Path
import subprocess,os,signal,json,time,re,shutil
root=Path(__file__).resolve().parents[1];repo=root.parents[1];tool=Path('/opt/mcProfiler-ubuntu18.04')
status=[];failures=[]
# Put the most useful before/after comparisons on disk first.
tasks=[(version,variant,workload)for workload in ['h7168','prefill','tiny','h3072']for variant in ['fp8','base']for version in ['v020','v000']]
tasks.remove(('v000','base','prefill'));tasks.append(('v000','base','prefill'))
for version,variant,workload in tasks:
 t,k,h={'tiny':(32,2,256),'h3072':(512,8,3072),'h7168':(512,8,7168),'prefill':(4096,8,7168)}[workload]
 name=f'{version}_{variant}_{workload}_T{t}_K{k}_H{h}';folder=root/(version+'-roofline');raw=folder/'raw'/name;raw.mkdir(exist_ok=True);image=folder/(name+'.png')
 if image.exists():
  record_file=raw/'collection.json'
  previous=json.loads(record_file.read_text())if record_file.exists()else{'case':name,'image':str(image.relative_to(root))}
  status.append(previous);continue
 command=f'env MACA_PATH=/opt/maca PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/tilelang-metax-v0.1.10:{repo} MCTX_TARGET_PROFILE_PATH={raw} /opt/conda/bin/python {root}/scripts/profile_case.py --version {version} --variant {variant} --workload {workload}'
 args=[str(tool/'mcProfiler'),'perf_exec','--cmdline',command,'--kernelname','reduce_fused_kernel_kernel','--casename',name,'--cwd',str(raw),'--per-kernel','--profile-from-start','0','--counts','1','--metrics','RoofLine','Total Cycles','WORKGROUPS']
 record={'case':name,'command':args,'started':time.time()}
 (root/'meta/status.json').write_text(json.dumps({'running':name,'completed':status},indent=2)+'\n')
 print('START',name,flush=True)
 with(folder/f'logs/{name}.log').open('w')as log:
  child=subprocess.Popen(args,cwd=tool,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record['pid']=child.pid
  try:record['exitcode']=child.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM);record['exitcode']=-1;record['timeout']=True
 record['seconds']=time.time()-record['started'];content=(folder/f'logs/{name}.log').read_text(errors='replace');outputs=re.findall(r'output path is: (\S+)',content)
 if not outputs:outputs=re.findall(r'please check report file (\S+)',content)
 if outputs and Path(outputs[-1]).is_dir():
  output=Path(outputs[-1]);shutil.copytree(output,raw/'mcprofiler_output',dirs_exist_ok=True)
  target_json=list(output.glob('*reduce_fused_kernel_kernel.txt.json'));picture=None
  if len(target_json)==1:
   data=json.loads(target_json[0].read_text())
   for values in data.values():
    if isinstance(values,list):
     for entry in values:
      if entry.get('name')=='RoofLine'and isinstance(entry.get('value'),dict):
       candidate=entry['value'].get('filename')
       if candidate and Path(candidate).is_file():picture=Path(candidate)
  if picture is None:
   pictures=sorted(output.glob('RoofLine*.png'))
   if len(pictures)==3:picture=pictures[1]
  if picture and record['exitcode']==0:
   shutil.copy2(picture,image);record['image']=str(image.relative_to(root));print('SAVED',image,flush=True)
  record['tool_output']=str((raw/'mcprofiler_output').relative_to(root))
 status.append(record);(raw/'collection.json').write_text(json.dumps(record,indent=2)+'\n')
 (root/'meta/status.json').write_text(json.dumps({'completed':status},indent=2)+'\n')
 if not record.get('image'):
  failures.append(name);print('FAILED_CONTINUE',name,flush=True)
(root/'meta/status.json').write_text(json.dumps({'done':not failures,'failed_cases':failures,'completed':status},indent=2)+'\n')
