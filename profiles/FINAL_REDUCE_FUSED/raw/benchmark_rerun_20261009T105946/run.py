from pathlib import Path
import sys,json,csv,time,traceback
root=Path(__file__).resolve().parent
started=time.time()
status=root/'status.json'
def update(value):status.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
update({'running':'imports','started':started})
print('BENCHMARK_RUN_START',flush=True)
try:
 import torch,pytest
 from benchmarks.benchmark_base import BenchmarkReport
 torch.set_num_threads(1)
 records={}
 class Capture:
  def pytest_runtest_setup(self,item):
   torch.manual_seed(1235)
   update({'running':item.nodeid,'completed_records':len(records)})
  def pytest_runtest_logreport(self,report):
   if report.when!='call':return
   for op,items in BenchmarkReport._records.items():
    for item in items:
     key=(op,item['params'].get('label'),item['tag'])
     records[key]={'op':op,**item}
   (root/'records.json').write_text(json.dumps(list(records.values()),default=str,ensure_ascii=False,indent=2)+'\n')
   print('CASE_RESULT',report.nodeid,report.outcome,flush=True)
   for record in records.values():
    if record['tag']=='tileops'and record['params'].get('label')in report.nodeid:
     print('KERNEL_RESULT',record['op'],record['params'].get('label'),record['result'],flush=True)
 exitcode=pytest.main(['benchmarks/ops/bench_moe_reduce_fused.py','-v','-s','--junitxml='+str(root/'results.xml')],plugins=[Capture()])
 mapping={'MoeReduceFusedFwdOp':'Base','MoeReduceFusedWithXsfFwdOp':'XSF','MoeReduceFusedFp8FwdOp':'FP8','MoeReduceFusedQuantizedFwdOp':'Quantized'}
 with (Path('/data/TileOPs-Metax/profiles/FINAL_REDUCE_FUSED/raw/summary_16g.csv')).open()as f:
  previous={(x['variant'],x['workload']):x for x in csv.DictReader(f)}
 rows=[]
 for record in records.values():
  if record['tag']!='tileops':continue
  params=record['params'];variant=mapping[record['op']];workload=params['label'].split('-')[0]
  latency=float(record['result']['latency_ms'])*1000;old=float(previous[variant,workload]['current_us'])
  rows.append(dict(variant=variant,workload=workload,previous_us=old,current_us=latency,change_pct=(latency/old-1)*100,result=record['result'],config=record.get('config',{})))
 if rows:
  with (root/'comparison_16g.csv').open('w',newline='')as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 update({'done':exitcode==0,'exitcode':exitcode,'kernel_cases':len(rows),'seconds':time.time()-started,'torch':torch.__version__})
 for row in rows:print('COMPARISON',json.dumps(row,default=str),flush=True)
 print('BENCHMARK_COMPLETE',exitcode,len(rows),flush=True)
 raise SystemExit(exitcode)
except BaseException as e:
 if isinstance(e,SystemExit):raise
 traceback.print_exc();update({'done':False,'error':str(e),'seconds':time.time()-started});raise
