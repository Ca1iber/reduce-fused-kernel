from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('ingest_hbm_resume.py')),run_name='__main__')
