from pathlib import Path
import argparse,subprocess,os,json,time,signal,shlex,re,shutil
v=Path(__file__).resolve().parents[1];repo=v.parents[1];tool=Path('/opt/mcProfiler-ubuntu18.04')
p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--plan',required=True);p.add_argument('--isolated-roi',action='store_true');p.add_argument('--original-symbol',action='store_true');p.add_argument('--warmup',type=int,default=10);a=p.parse_args()
plan_path=Path(a.plan);plan=json.loads(plan_path.read_text());raw=v/'raw/hbm_native'/a.batch;raw.mkdir(exist_ok=True)
command=['env','MACA_PATH=/opt/maca','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/opt/tilelang-metax-v0.1.10:'+str(repo),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','MCTX_TARGET_PROFILE_PATH='+str(raw),'/opt/conda/bin/python',str(v/'scripts/profile_hbm_batch.py'),'--plan',str(plan_path)]
if a.isolated_roi:command.append('--isolated-roi')
if a.original_symbol:command.append('--original-symbol')
command.extend(['--warmup',str(a.warmup)])
symbols=[f"hbm_{x['version']}_{x['variant']}_{x['workload']}_kernel"for x in plan]
args=[str(tool/'mcProfiler'),'perf_exec','--cmdline',shlex.join(command),'--kernelname','hbm_profile_'+a.batch,'--kernelnames',*symbols,'--casename','reduce_fused_'+a.batch,'--cwd',str(raw),'--per-kernel','--profile-from-start','0','--counts',str(len(plan)),'--metrics','RoofLine','Total Cycles','WORKGROUPS']
if a.original_symbol:
    if len(plan)!=1:raise RuntimeError('Single case required')
    case=plan[0]
    kernel_symbol='reduce_fused_kernel_kernel'
    if case['version']=='final'and case['workload']=='tiny':
        kernel_symbol='tiny_local_kernel'if case['variant']=='quantized'else'tiny_pair_kernel'
    args=[str(tool/'mcProfiler'),'perf_exec','--cmdline',shlex.join(command),'--kernelname',kernel_symbol,'--kernelnames',kernel_symbol,'--casename','reduce_fused_'+a.batch,'--cwd',str(raw),'--per-kernel','--profile-from-start','0','--counts','1','--metrics','RoofLine','Total Cycles','WORKGROUPS']
log=v/'logs/hbm_native'/(a.batch+'.log');record={'command':args,'started':time.time(),'cases':len(plan),'isolated_roi':a.isolated_roi};status=v/'meta/hbm_native'/(a.batch+'_status.json')
print('START',a.batch,len(plan),flush=True)
with log.open('w')as f:
    child=subprocess.Popen(args,cwd=tool,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PATH='/opt/conda/bin:/opt/maca/bin:'+os.environ['PATH'],MACA_PATH='/opt/maca'))
    record['pid']=child.pid;status.write_text(json.dumps(record,indent=2)+'\n')
    while True:
        try:record['exitcode']=child.wait(timeout=40);break
        except subprocess.TimeoutExpired:
            lines=log.read_text(errors='replace').splitlines()
            progress=[x for x in lines if any(t in x for t in ['READY','ISOLATED_ROI_DONE','PROFILE_BATCH_DONE','Traceback'])]
            print('PROGRESS',a.batch,'seconds',round(time.time()-record['started']),progress[-1:]if progress else'initializing',flush=True)
            if time.time()-record['started']>600:
                os.killpg(child.pid,signal.SIGTERM);record['exitcode']=-1;record['timeout']=True;break
record['seconds']=time.time()-record['started'];content=log.read_text(errors='replace');outputs=re.findall(r'output path is: (\S+)',content)
if not outputs:outputs=re.findall(r'please check report file (\S+)',content)
if outputs and Path(outputs[-1]).is_dir():
    output=Path(outputs[-1]);shutil.copytree(output,raw/'mcprofiler_output',dirs_exist_ok=True);record['tool_output']=str(raw/'mcprofiler_output')
    record['kernel_reports']=len(list(output.glob('*_kernel.txt.json')))
status.write_text(json.dumps(record,indent=2)+'\n')
print('FINISHED',a.batch,json.dumps({k:record.get(k)for k in ['exitcode','seconds','cases','kernel_reports']}),flush=True)
for line in content.splitlines():
    if any(t in line for t in ['Traceback','RuntimeError','MemoryError','Error:']):print(line[:400],flush=True)
