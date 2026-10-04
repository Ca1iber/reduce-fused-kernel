from pathlib import Path
import time,subprocess,json,re,shutil
r=Path(__file__).resolve().parents[1]
dst=r/'raw/native_register8_minimal';dst.mkdir(exist_ok=True)
cmd=f'env MCTX_TARGET_PROFILE_PATH={dst} /opt/conda/bin/python {r}/scripts/profile_cases.py --case register8'
a=['/opt/mcProfiler-ubuntu18.04/mcProfiler','perf_exec','--cmdline',cmd,'--kernelname','v016_register8_minimal','--casename','v016_register8_minimal','--cwd',str(dst),'--custom','--per-kernel','--kernelnames','reduce_fused_kernel_kernel','--counts','1','--metrics','Total Instructions','Compute Instructions','Memory Instructions','Global Memory Read bytes','Global Memory Write bytes']
with (r/'logs/native_register8_minimal.log').open('w') as log:
 p=subprocess.Popen(a,cwd='/opt/mcProfiler-ubuntu18.04',stdout=log,stderr=subprocess.STDOUT)
 (r/'meta/recollection_job.json').write_text(json.dumps({'pid':p.pid,'command':a},indent=2)+'\n')
 code=p.wait()
text=(r/'logs/native_register8_minimal.log').read_text(errors='replace');paths=re.findall(r'output path is: (\S+)',text)
if paths and Path(paths[-1]).exists():shutil.copytree(paths[-1],dst/'tool_output',dirs_exist_ok=True)
(r/'meta/recollection_result.json').write_text(json.dumps({'exitcode':code},indent=2)+'\n')
