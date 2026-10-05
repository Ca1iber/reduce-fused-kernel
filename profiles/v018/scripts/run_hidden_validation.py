from pathlib import Path
import json,subprocess,time
r=Path(__file__).resolve().parents[1]
while True:
 p=Path('/proc/113145')
 try:
  if p.joinpath('stat').read_text().split()[2]=='Z' or b'benchmark_all.py'not in p.joinpath('cmdline').read_bytes():break
 except FileNotFoundError:break
 time.sleep(1)
s=json.loads((r/'meta/benchmark_all_status.json').read_text())
if s['stage']!='done':raise RuntimeError('benchmark did not complete')
with (r/'logs/hidden_validation.log').open('w')as log:
 p=subprocess.Popen(['/opt/conda/bin/python','-u',str(r/'scripts/validate_hidden_fragment.py')],cwd='/data/TileOPs-Metax',stdout=log,stderr=subprocess.STDOUT)
 (r/'meta/hidden_validation_job.json').write_text(json.dumps({'pid':p.pid},indent=2)+'\n');code=p.wait()
(r/'meta/hidden_validation_process.json').write_text(json.dumps({'exitcode':code},indent=2)+'\n')
