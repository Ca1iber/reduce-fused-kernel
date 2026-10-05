from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[1]
while True:
 p=Path('/proc/222225')
 try:
  if p.joinpath('stat').read_text().split()[2]=='Z' or b'mask_copy.py'not in p.joinpath('cmdline').read_bytes():break
 except FileNotFoundError:break
 time.sleep(1)
with (r/'logs/validation.log').open('w')as log:
 p=subprocess.Popen(['/opt/conda/bin/python','-u',str(r/'scripts/validate_workaround.py')],cwd='/data/TileOPs-Metax',stdout=log,stderr=subprocess.STDOUT)
 (r/'meta/validation_job.json').write_text(json.dumps({'pid':p.pid},indent=2)+'\n');code=p.wait()
(r/'meta/validation_process.json').write_text(json.dumps({'exitcode':code},indent=2)+'\n')
