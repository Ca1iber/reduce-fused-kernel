from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parents[1]
while True:
 p=Path('/proc/223142')
 try:
  if p.joinpath('stat').read_text().split()[2]=='Z' or b'validate_workaround.py'not in p.joinpath('cmdline').read_bytes():break
 except FileNotFoundError:break
 time.sleep(1)
s={}
for name in ('mask_individual','one_warp'):
 with (r/f'logs/{name}.log').open('w')as log:
  p=subprocess.Popen(['/opt/conda/bin/python','-u',str(r/f'scripts/{name}.py')],cwd='/data/TileOPs-Metax',stdout=log,stderr=subprocess.STDOUT)
  s['stage']=name;s['pid']=p.pid;(r/'meta/final_controls.json').write_text(json.dumps(s,indent=2)+'\n');s[name+'_exitcode']=p.wait()
s['stage']='done';(r/'meta/final_controls.json').write_text(json.dumps(s,indent=2)+'\n')
