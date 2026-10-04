from pathlib import Path
import os,time,subprocess,json
repo=Path('/root/TileOPs-Metax');p4=repo/'profiles/v004'
pid=json.loads((p4/'meta/formal_profiles_process.json').read_text())['pid']
deadline=time.time()+430
while time.time()<deadline:
    try:cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes()
    except OSError:break
    if b'collect_formal_profiles.py' not in cmd:break
    time.sleep(1)
else:raise SystemExit('Previous native profile still running')
env=dict(os.environ,MACA_PATH='/opt/maca',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/opt/tilelang-metax-v0.1.10:/root/TileOPs-Metax')
for version in ('v005','v006'):
    root=repo/'profiles'/version
    stages=[]
    tasks=[
      ('trace',['/opt/maca/bin/mcTracer','--mctx','/opt/conda/bin/python',str(root/'scripts/profile_versions.py'),'--version','all']),
      ('correctness',['/opt/conda/bin/python',str(root/'scripts/validate_candidate.py')]),
      ('paired_benchmark',['/opt/conda/bin/python',str(root/'scripts/bench_candidate.py')]),
    ]
    for label,command in tasks:
        print('START',version,label,flush=True)
        cwd=root/'raw/trace' if label=='trace' else repo
        cwd.mkdir(parents=True,exist_ok=True)
        with (root/'logs'/(label+'.log')).open('w') as output:
            result=subprocess.run(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        stages.append(dict(stage=label,returncode=result.returncode))
        (root/'meta/run_status.json').write_text(json.dumps(stages,indent=2)+'\n')
        print('FINISHED',version,label,result.returncode,flush=True)
        if result.returncode:break
print('EXPLORATIONS_DONE',flush=True)
