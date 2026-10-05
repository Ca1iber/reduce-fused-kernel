from pathlib import Path
import subprocess,json,os
r=Path(__file__).resolve().parents[1];rows=[]
for s in json.loads((r/'meta/hold_specs.json').read_text()):
 name=s['name'];p=r/f'codegen/hold_{name}.cu';o=p.with_suffix('.mcbin')
 cmd=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','-O3','-lineinfo','--offload-arch=xcore1000','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__',str(p),'-o',str(o)]
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(r/f'logs/hold_compile_{name}.log').write_text(q.stdout+q.stderr)
 if q.returncode:raise RuntimeError(name+' failed compile '+q.stderr[-500:])
 for cycles in (1000000,5000000):
  args=[str(r/'scripts/residency_probe'),str(o),str(s['threads']),str(s['shared_bytes']),str(s['out_bytes']),str(s['grid_y']),str(s['tokens']),s['arg_order'],str(cycles)]
  q=subprocess.run(args,capture_output=True,text=True,timeout=90)
  (r/f'logs/hold_run_{name}_{cycles}.log').write_text(q.stdout+q.stderr)
  if q.returncode:raise RuntimeError(name+' failed run '+q.stderr[-500:])
  rows.append(dict(name=name,result=json.loads(q.stdout),command=args));(r/'raw/residency_probe.json').write_text(json.dumps(rows,indent=2)+'\n');print('PROBE',name,cycles,json.loads(q.stdout),flush=True)
(r/'meta/residency_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
