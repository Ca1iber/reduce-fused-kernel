from pathlib import Path
import os,time,json,subprocess
r=Path(__file__).resolve().parents[1]
while True:
    try:
        state=Path('/proc/69497/stat').read_text().split()[2]
        if state=='Z':break
    except FileNotFoundError:break
    time.sleep(2)
with (r/'logs/initial_compare.log').open('w') as log:
    p=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/initial_compare.py')],stdout=log,stderr=subprocess.STDOUT)
(r/'meta/status.json').write_text(json.dumps({'exitcode':p.returncode},indent=2)+'\n')
