from pathlib import Path
import json,subprocess
r=Path(__file__).resolve().parents[1]
tool=r.parents[1]/'profiles/v018_DIS:v016/scripts/resource_query';rows=[]
for name in ['base_h7168_baseline','fp8_h7168_baseline','fp8_prefill_baseline','fp8_prefill_vector_store','quantized_prefill_baseline','quantized_prefill_vector_store']:
 threads=256 if name.startswith('base_')else 128
 result=subprocess.run([str(tool),str(r/f'codegen/{name}.mcbin'),'reduce_fused_kernel_kernel',str(threads),'0'],capture_output=True,text=True,timeout=30)
 if result.returncode:raise RuntimeError(result.stderr)
 rows.append({'case':name,'threads':threads,'resource':json.loads(result.stdout)})
 (r/'raw/resources_16g.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('RESOURCE',name,rows[-1]['resource'],flush=True)
