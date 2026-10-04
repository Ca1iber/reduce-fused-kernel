"""Collect only instruction counters for the baseline and candidate."""
from pathlib import Path
import subprocess,os,signal,re,shutil,json
root=Path('/root/TileOPs-Metax/profiles/v004')
tool=Path('/opt/mcProfiler-ubuntu18.04')
records=[]
for implementation in ('v003','v004'):
    raw=root/'raw'/('mcprofiler_'+implementation)
    raw.mkdir(exist_ok=True)
    command=('env MACA_PATH=/opt/maca PYTHONDONTWRITEBYTECODE=1 '
             'PYTHONPATH=/opt/tilelang-metax-v0.1.10:/root/TileOPs-Metax '
             f'MCTX_TARGET_PROFILE_PATH={raw} /opt/conda/bin/python '
             f'{root}/scripts/profile_versions.py --version {implementation}')
    kernel='reduce_fused_kernel_kernel'
    args=[str(tool/'mcProfiler'),'perf_exec','--cmdline',command,'--kernelname',kernel,
          '--casename','formal_'+implementation+'_fp8_h7168','--cwd',str(raw),
          '--per-kernel','--profile-from-start','0','--counts','1',
          '--metrics','RoofLine','Total Instructions','Compute Instructions','Memory Instructions','Total Cycles']
    log=root/'logs'/('mcprofiler_'+implementation+'.log')
    print('START',implementation,flush=True)
    with log.open('w') as output:
        process=subprocess.Popen(args,cwd=tool,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            code=-1
    content=log.read_text(errors='replace')
    found=re.findall(r'output path is: (\S+)',content)
    record=dict(implementation=implementation,returncode=code,command=args,log=str(log))
    if found and Path(found[-1]).exists():
        shutil.copytree(found[-1],raw/'mcprofiler_output',dirs_exist_ok=True)
        record['original_output']=found[-1]
    records.append(record)
    (root/'meta/mcprofiler_runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print('FINISHED',implementation,code,flush=True)
    if code: break
