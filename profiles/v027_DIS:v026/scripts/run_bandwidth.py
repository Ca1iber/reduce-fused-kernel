from pathlib import Path
import subprocess,json,csv,statistics,os,time
r=Path(__file__).resolve().parents[1]
source=r/'scripts/hbm_ratio_probe.cpp';binary=r/'scripts/hbm_ratio_probe'
command=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','--offload-arch=xcore1000','-O3','-std=c++17',str(source),'-o',str(binary),'--maca-path=/opt/maca']
(r/'meta/status.json').write_text(json.dumps({'stage':'compile','command':command},indent=2)+'\n')
result=subprocess.run(command,capture_output=True,text=True,timeout=90)
(r/'logs/compile.log').write_text(result.stdout+result.stderr)
if result.returncode:raise SystemExit(result.returncode)
def run(label,args):
 (r/'meta/status.json').write_text(json.dumps({'stage':label,'args':args},indent=2)+'\n')
 started=time.time()
 with(r/f'raw/{label}.csv').open('w')as out,(r/f'logs/{label}.log').open('w')as err:
  result=subprocess.run([str(binary),*map(str,args)],stdout=out,stderr=err,timeout=90)
 if result.returncode:raise SystemExit(result.returncode)
 if 'sample_verification=PASS'not in(r/f'logs/{label}.log').read_text():raise SystemExit('probe verification missing')
 print('DONE',label,round(time.time()-started,2),flush=True)
 return list(csv.DictReader((r/f'raw/{label}.csv').open()))
rows=run('sweep_1g',[1024]);groups={}
for z in rows:groups.setdefault((z['mode'],int(z['threads']),int(z['blocks'])),[]).append(z)
summary=[]
for (mode,threads,blocks),samples in groups.items():
 assert len(samples)==5
 speeds=[float(z['total_TBps'])for z in samples]
 summary.append({'mode':mode,'threads':threads,'blocks':blocks,'median_TBps':statistics.median(speeds),'min_TBps':min(speeds),'max_TBps':max(speeds)})
(r/'analysis/sweep_summary_16g.json').write_text(json.dumps(summary,indent=2)+'\n')
best={mode:max([z for z in summary if z['mode']==mode],key=lambda z:z['median_TBps'])for mode in ['copy_1to1','read8_write_8to1','read8_write_16to1']}
(r/'analysis/selected_configs_16g.json').write_text(json.dumps(best,indent=2)+'\n');print('SELECTED',best,flush=True)
confirm=[]
for idx,mode in enumerate(best):
 config=best[mode]
 for mib in [1024,2048]:
  samples=run(f'confirm_{mode}_{mib}m',[mib,idx,config['threads'],config['blocks']]);speeds=[float(z['total_TBps'])for z in samples]
  confirm.append({'mode':mode,'input_MiB':mib,'threads':config['threads'],'blocks':config['blocks'],'median_TBps':statistics.median(speeds),'min_TBps':min(speeds),'max_TBps':max(speeds),'samples_TBps':speeds})
  (r/'analysis/confirmation_16g.json').write_text(json.dumps(confirm,indent=2)+'\n')
(r/'meta/gpu_after.txt').write_text(subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,timeout=15).stdout)
(r/'meta/status.json').write_text(json.dumps({'done':True,'sweep_configurations':len(summary),'confirmations':len(confirm),'all_sample_checks_passed':True},indent=2)+'\n')
print('COMPLETE',confirm,flush=True)
