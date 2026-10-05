from pathlib import Path
import subprocess,json,os
r=Path(__file__).resolve().parents[1]
with (r/'logs/trace.log').open('w')as log:
 p=subprocess.Popen(['/opt/maca/bin/mcTracer','--mctx','--odname','trace_resources','--name','occupancy','/opt/conda/bin/python',str(r/'scripts/collect_cases.py')],cwd=r/'raw',stdout=log,stderr=subprocess.STDOUT)
 (r/'meta/trace_job.json').write_text(json.dumps({'pid':p.pid},indent=2)+'\n');code=p.wait()
(r/'meta/trace_process.json').write_text(json.dumps({'exitcode':code},indent=2)+'\n')
