from pathlib import Path
import time,subprocess,json
r=Path(__file__).resolve().parents[1]
wait_pid=81442
while True:
    try:
        if Path(f'/proc/{wait_pid}/stat').read_text().split()[2]=='Z':break
    except FileNotFoundError:break
    time.sleep(2)
with (r/'logs/validation_best.log').open('w') as log:
    p=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/validate_best.py')],stdout=log,stderr=subprocess.STDOUT)
