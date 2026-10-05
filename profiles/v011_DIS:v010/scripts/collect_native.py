from pathlib import Path
import subprocess,json,re,shutil,os,signal,time
root=Path(__file__).resolve().parents[1];tool=Path('/opt/mcProfiler-ubuntu18.04')
meta=json.loads((root/'meta/baseline_fp8.json').read_text());symbol=meta['kernel_symbols'][0]
destination=root/'raw/native_baseline_fp8';destination.mkdir(exist_ok=True)
command=f'env MCTX_TARGET_PROFILE_PATH={destination} /opt/conda/bin/python {root}/scripts/baseline_driver.py --variant fp8 --mode profile'
args=[str(tool/'mcProfiler'),'perf_exec','--cmdline',command,'--kernelname','v011_baseline_fp8','--casename','v011_baseline_fp8','--cwd',str(destination),'--custom','--kernelnames',symbol,'--counts','1','--metrics','Total Instructions','Compute Instructions','Memory Instructions','Total Cycles','Global Memory Read bytes','Global Memory Write bytes','ISU stall cycles layout']
record=dict(command=args,started=time.time())
with (root/'logs/native_baseline_fp8.log').open('w') as log:
    process=subprocess.Popen(args,cwd=tool,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record['pid']=process.pid;(root/'meta/native_run.json').write_text(json.dumps(record,indent=2)+'\n')
    try:record['exitcode']=process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM);record['exitcode']=-1;record['timeout']=True
record['finished']=time.time()
content=(root/'logs/native_baseline_fp8.log').read_text(errors='replace')
paths=re.findall(r'output path is: (\S+)',content)
if paths and Path(paths[-1]).exists():
    shutil.copytree(paths[-1],destination/'tool_output',dirs_exist_ok=True);record['output']=paths[-1]
(root/'meta/native_run.json').write_text(json.dumps(record,indent=2)+'\n');print('NATIVE_FINISHED',json.dumps(record),flush=True)
