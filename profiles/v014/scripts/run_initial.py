from pathlib import Path
import os,time,subprocess,json
root=Path(__file__).resolve().parents[1]
while True:
    try:os.kill(56257,0)
    except ProcessLookupError:break
    time.sleep(2)
with (root/'logs/initial_compare.log').open('w') as log:
 p=subprocess.run(['/opt/conda/bin/python','-u',str(root/'scripts/initial_compare.py')],stdout=log,stderr=subprocess.STDOUT)
(root/'meta/status.json').write_text(json.dumps({'exitcode':p.returncode},indent=2)+'\n')
