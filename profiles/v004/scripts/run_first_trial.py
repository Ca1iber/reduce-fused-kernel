from pathlib import Path
import subprocess,os,json
root=Path("/root/TileOPs-Metax/profiles/v004")
env=dict(os.environ,MACA_PATH="/opt/maca",PYTHONDONTWRITEBYTECODE="1",PYTHONPATH="/opt/tilelang-metax-v0.1.10:/root/TileOPs-Metax")
stages=[]
commands=[
 ("trace",["/opt/maca/bin/mcTracer","--mctx","/opt/conda/bin/python",str(root/"scripts/profile_versions.py"),"--version","all"]),
 ("correctness",["/opt/conda/bin/python",str(root/"scripts/validate_candidate.py")]),
 ("paired_benchmark",["/opt/conda/bin/python",str(root/"scripts/bench_candidate.py")]),
]
for label,command in commands:
    print("START",label,flush=True)
    cwd=root/"raw/trace" if label=="trace" else Path("/root/TileOPs-Metax")
    cwd.mkdir(parents=True,exist_ok=True)
    with (root/"logs"/(label+".log")).open("w") as output:
        result=subprocess.run(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
    stages.append(dict(stage=label,returncode=result.returncode,command=command))
    (root/"meta/run_status.json").write_text(json.dumps(stages,indent=2)+"\n")
    print("FINISHED",label,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
print("DONE",flush=True)
