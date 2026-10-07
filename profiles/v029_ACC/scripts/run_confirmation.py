from pathlib import Path
import subprocess,os,json
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
env=dict(os.environ,MACA_PATH='/opt/maca',LD_LIBRARY_PATH='/opt/maca/lib:'+os.environ.get('LD_LIBRARY_PATH',''),PYTHONPATH='/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MALLOC_ARENA_MAX='2',PATH='/opt/conda/bin:/opt/maca/bin:'+os.environ['PATH'])
with(r/'logs/confirm_validate.log').open('w')as out:
 p=subprocess.run(['/opt/conda/bin/python','-u',str(r/'scripts/confirm_validate.py')],cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=600)
(r/'meta/final_process_exit.json').write_text(json.dumps({'exitcode':p.returncode},indent=2)+'\n');print('FINAL_PROCESS_EXIT',p.returncode,flush=True)
