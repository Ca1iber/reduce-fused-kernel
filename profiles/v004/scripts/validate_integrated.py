from pathlib import Path
import subprocess,os,time,json
repo=Path('/root/TileOPs-Metax');root=repo/'profiles/v004'
pid=96201
while True:
    try:cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes()
    except OSError:break
    if b'run_explorations.py' not in cmd:break
    time.sleep(1)
env=dict(os.environ,MACA_PATH='/opt/maca',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/opt/tilelang-metax-v0.1.10:/root/TileOPs-Metax')
records=[]
for label,path,total in (('integrated_correctness','tests/ops/test_moe_reduce_fused.py',131),('integrated_benchmark','benchmarks/ops/bench_moe_reduce_fused.py',16)):
    print('START',label,flush=True)
    command=['/opt/conda/bin/python','-m','pytest','-q',path,'--junitxml='+str(root/'raw'/(label+'.xml'))]
    with (root/'logs'/(label+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(stage=label,returncode=result.returncode,expected_tests=total))
    (root/'meta/integrated_validation.json').write_text(json.dumps(records,indent=2)+'\n')
    print('FINISHED',label,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
print('INTEGRATED_VALIDATION_DONE',flush=True)
