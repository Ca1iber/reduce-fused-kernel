from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[1]
command=['/opt/maca/bin/mcTracer','--mctx','--odname','trace_grid','--name','grid','/opt/conda/bin/python',str(r/'scripts/profile_grid.py'),'--case','all','--workload','h7168']
started=time.time()
with(r/'logs/trace_grid.log').open('w')as log:
 code=subprocess.run(command,cwd=r/'raw',stdout=log,stderr=subprocess.STDOUT).returncode
(r/'meta/trace_grid_status.json').write_text(json.dumps({'exitcode':code,'seconds':time.time()-started,'command':command},indent=2)+'\n')
