from pathlib import Path
import subprocess,json,os
r=Path(__file__).resolve().parents[1];rows=[]
for c in json.loads((r/'analysis/trace_summary.json').read_text()):
 p=r/f"codegen/{c['name']}.cu";o=p.with_suffix('.mcbin')
 cmd=['/opt/maca/mxgpu_llvm/bin/mxcc','-x','maca','-device-obj','-O3','-lineinfo','--offload-arch=xcore1000','-std=c++17','-I/opt/tilelang-metax-v0.1.10/src','-D__FAST_HALF_CVT__',str(p),'-o',str(o)]
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 (r/f"logs/compile_{c['name']}.log").write_text(q.stdout+q.stderr)
 if q.returncode:raise RuntimeError(c['name']+' compile failed')
 q=subprocess.run([str(r/'scripts/resource_query'),str(o),'reduce_fused_kernel_kernel',str(c['threads']),str(c['shared_bytes'])],capture_output=True,text=True,timeout=30)
 if q.returncode:raise RuntimeError(c['name']+' query failed '+q.stderr)
 rows.append({'name':c['name'],'compiler_command':cmd,'resource':json.loads(q.stdout)})
 (r/'raw/kernel_resources.json').write_text(json.dumps(rows,indent=2)+'\n');print('RESOURCE',c['name'],json.loads(q.stdout),flush=True)
(r/'meta/query_status.json').write_text(json.dumps({'stage':'done','cases':len(rows)},indent=2)+'\n')
