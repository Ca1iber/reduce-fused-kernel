from pathlib import Path
import time,subprocess
r=Path(__file__).resolve().parents[1]
pid=101971
while True:
 try:
    if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(2)
with (r/'logs/validation.log').open('w') as log:
 subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/validate.py')],stdout=log,stderr=subprocess.STDOUT)
