from pathlib import Path
import subprocess,json
root=Path(__file__).resolve().parents[1]
for stage in (2,3):
    with (root/f'logs/validation_ring{stage}.log').open('w') as log:
        p=subprocess.run(['/opt/conda/bin/python','-u',str(root/'scripts/validate.py'),str(stage)],stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise SystemExit(p.returncode)
