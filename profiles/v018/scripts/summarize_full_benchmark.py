from pathlib import Path
import json,csv,statistics
r=Path(__file__).resolve().parents[1];s=json.loads((r/'meta/benchmark_all_status.json').read_text())
assert s['stage']=='done'and len(s['completed'])==16,s
rows=list(csv.DictReader((r/'raw/benchmark_all_16g.csv').open()));assert len(rows)==80
by={}
for z in rows:
 assert z['byte_equal']=='True',z
 m=json.loads(z['measurements']);assert len(m)==5 and all(t['timing']=='cupti'for t in m),z
 by.setdefault((z['variant'],z['workload']),{})[z['name']]=z
headers=['变体','workload','v010 µs','v016 K预取 µs','group2直接读 µs','shared双缓冲 µs','fragment双缓冲 µs','fragment/v010加速比','fragment/v016加速比']
lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'];summary=[]
for workload in ('tiny','h3072','h7168','prefill'):
 for variant in ('base','xsf','fp8','quantized'):
  d=by[(variant,workload)];t={n:float(z['latency_us'])for n,z in d.items()};candidate=t['v018_fragment_double'];old=t['v010'];prev=t['v016_register8']
  lines.append('| '+' | '.join([variant,workload]+[f'{t[n]:.3f}'for n in ('v010','v016_register8','group2_direct','v015_shared_double','v018_fragment_double')]+[f'{old/candidate:.4f}×',f'{prev/candidate:.4f}×'])+' |')
  summary.append(dict(variant=variant,workload=workload,baseline_us=old,register8_us=prev,fragment_double_us=candidate,speedup_vs_v010=old/candidate,speedup_vs_register8=prev/candidate,latency_change_pct=(candidate/old-1)*100,measurements=json.loads(d['v018_fragment_double']['measurements'])))
(r/'analysis/full_benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
p=r/'analysis/v018_full_benchmark_16g.md';text='# v018：hidden 双fragment的完整16组benchmark（16g）\n\n'
text+='## 覆盖与方法\n\n四变体Base/XSF/FP8/Quantized，每个包含tiny、h3072、h7168、prefill，共16组、每组5个kernel对照，共80行。全部逐字节与v010同输入输出一致。\n\n'
text+='| workload | T | K | H | 输入dtype | 候选tile/threads |\n|---|---:|---:|---:|---|---|\n| tiny | 32 | 2 | 256 | FP16 | 256/128 |\n| h3072 | 512 | 8 | 3072 | BF16 | 512/128 |\n| h7168 | 512 | 8 | 7168 | BF16 | 512/128 |\n| prefill | 4096 | 8 | 7168 | BF16 | 1024/128 |\n\n'
text+='v010使用已有各workload/变体的生产配置；v016寄存器预取和三个group2方案统一使用上表tile/threads。group2方案每CTA处理最多2个hidden块；包含同几何直接读、shared双缓冲和寄存器fragment双缓冲。Base/XSF的候选与原基线几何不同，因此所有差异不能归为寄存器预取单因素。\n\n'
text+='计时采用项目bench_kernel：10warmup、50repeat×3trials、CUPTI、L2flush，每组五方案交替5外层轮次，取中位数。tiny/h3072/h7168输入clone；prefill按项目阈值不clone。逐轮样本保留，未删异常值。这里只比较同一组配对数据，不与前一轮单例pilot拼接。\n\n'
text+='## 全部结果\n\n'+'\n'.join(lines)+'\n\n'
text+='## 解释边界\n\n- tiny只有1个hidden块，没有下一块供hidden流水线预取；即使候选变快，也只能说明生成代码/配置/预取K输入等整体差异，不能证明hidden双缓冲重叠有效。\n- h3072采用group2后总CTA1536，小于104AP×16=1664，grid供给也可能限制达到资源容量；这是数量模型，不是逐AP时间轨迹。\n- h7168/group2有3584CTA，供给足够；此前39寄存器/线程与16CTA/AP的资源结论限定tile512/128，不能直接套到prefill的tile1024/128。\n- 寄存器占用更多而驻留不下降，并不保证更快；还涉及单CTA工作量、load调度、元数据复用和尾部。\n- 结果为当前sc-16g与所测输入；单组收益不能代替完整覆盖，也不能仅凭微小时间差宣布普遍有效。\n\n'
text+='## 数据与复现\n\n[80行完整CSV](../raw/benchmark_all_16g.csv)、[汇总JSON](full_benchmark_summary.json)、[逐组运行脚本](../scripts/benchmark_all.py)、[五方案worker](../scripts/benchmark_hidden_fragments.py)。每组原始CSV和进度JSON在raw。\n\n`python profiles/v018/scripts/benchmark_all.py` 顺序运行16组；每组完成后即写独立CSV。已有131用例单独顺序运行，结果保存在raw/correctness_hidden_fragment.xml，避免与benchmark并发。\n'
valid=r/'meta/hidden_validation.json'
if valid.exists():text+='\n完整现有测试结果：'+valid.read_text()+'\n'
else:text+='\n完整131用例仍在运行；本报告的16组benchmark逐字节对照已全部通过，不能将它代替完整测试。\n'
p.write_text(text)
print('SUMMARY',[(z['variant'],z['workload'],round(z['speedup_vs_v010'],4),round(z['speedup_vs_register8'],4))for z in summary])
