from pathlib import Path
import csv,json,math,hashlib,xml.etree.ElementTree as ET,subprocess,sys
from html import escape
r=Path(__file__).resolve().parents[1];repo=r.parents[1]
status=json.loads((r/'meta/status.json').read_text());assert status.get('done')and status['cases']==16,status
rows=list(csv.DictReader((r/'raw/comparison_16g.csv').open()));assert len(rows)==16
variants=['Base','XSF','FP8','Quantized'];workloads=['tiny','h3072','h7168','prefill'];lookup={(z['variant'],z['workload']):z for z in rows};assert set(lookup)=={(v,w)for v in variants for w in workloads}
assert all(z['byte_equal']=='True' and z['timing']=='cupti' for z in rows)
source=json.loads((r/'meta/source.json').read_text());assert hashlib.sha256((repo/'tileops/kernels/moe/reduce_fused.py').read_bytes()).hexdigest()==source['current_source_sha256']
# SVG chart generated directly from the measured CSV; no extra plotting dependency.
colors={'Base':'#10b981','XSF':'#3b82f6','FP8':'#ec4899','Quantized':'#f59e0b'}
width,height=1120,570;left,top,plot_w,plot_h=80,100,1000,360;ymax=math.ceil(max(float(z['speedup'])for z in rows)*1.15*2)/2
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#263238} .tick{font-size:13px} .value{font-size:12px}</style>','<text x="80" y="30" font-size="22" font-weight="bold">v000 vs current v025: 16 workloads</text>','<text x="80" y="54" font-size="13">Speedup = v000 latency / v025 latency; sc-16g, paired CUPTI measurements</text>']
for i,v in enumerate(variants):
 x=80+i*160;parts.extend([f'<rect x="{x}" y="70" width="14" height="14" fill="{colors[v]}"/>',f'<text x="{x+21}" y="82" font-size="13">{v}</text>'])
for tick in range(int(ymax*2)+1):
 val=tick/2;y=top+plot_h*(1-val/ymax)
 parts.extend([f'<line x1="{left}" x2="{left+plot_w}" y1="{y}" y2="{y}" stroke="#e5e7eb"/>',f'<text x="{left-12}" y="{y+4}" class="tick" text-anchor="end">{val:.1f}</text>'])
y=top+plot_h*(1-1/ymax);parts.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y}" y2="{y}" stroke="#475569" stroke-dasharray="6 4"/>')
for wi,w in enumerate(workloads):
 center=left+(wi+.5)*plot_w/4
 for vi,v in enumerate(variants):
  z=lookup[v,w];val=float(z['speedup']);x=center-100+vi*52;h=plot_h*val/ymax;y=top+plot_h-h
  parts.append(f'<rect x="{x}" y="{y}" width="42" height="{h}" fill="{colors[v]}"><title>{escape(v)} {w}: {float(z["baseline_us"]):.3f} us / {float(z["current_us"]):.3f} us = {val:.4f}x</title></rect>')
  parts.append(f'<text x="{x+21}" y="{y-7}" class="value" text-anchor="middle">{val:.3f}x</text>')
 parts.append(f'<text x="{center}" y="{top+plot_h+27}" font-size="16" text-anchor="middle">{w}</text>')
 z=lookup['Base',w];parts.append(f'<text x="{center}" y="{top+plot_h+47}" font-size="12" text-anchor="middle">T{z["num_tokens"]} / K{z["num_topk"]} / H{z["hidden"]}</text>')
parts+=['<text transform="translate(23 280) rotate(-90)" font-size="14" text-anchor="middle">Speedup (x), higher is better</text>','<text x="80" y="548" font-size="12">Dashed line: 1.0x (same latency as v000). Median of 5 alternating paired rounds.</text>','</svg>']
chart=r/'analysis/speedup_v000_vs_v025_16g.svg';chart.write_text('\n'.join(parts)+'\n');ET.parse(chart)
text=f'''# v026：v000 与当前 v025 的 16 组性能总结

## 比较范围

本次在同一 sc-16g 实例重新配对测量全部16组，不拼接历史轮次。

- **v000**：最初迁移的 naive kernel，Git `{source['baseline_commit'][:8]}`；每个token一个CTA，128线程，FP8使用SDK转换。
- **当前 v025**：正式kernel，Git `{source['current_commit'][:8]}`，包含截至v025所有已接入优化。
- tiny为T32/K2/H256、FP16；其余为BF16：h3072=T512/K8/H3072，h7168=T512/K8/H7168，prefill=T4096/K8/H7168。
- 四变体均使用top-k权重；XSF增加输入行缩放，FP8增加最终sf并输出E4M3，Quantized同时使用两个缩放。

## 16 组 workload 对比

**加速比 = v000耗时 ÷ v025耗时**；耗时下降 = (1 − v025耗时 ÷ v000耗时) × 100%。加速比大于1表示变快。

![16组加速比](speedup_v000_vs_v025_16g.svg)

| 变体 | workload | v000 μs | 当前v025 μs | 加速比 | 耗时下降 |
|---|---|---:|---:|---:|---:|
'''
for v in variants:
 for w in workloads:
  z=lookup[v,w];text+=f"| {v} | {w} | {float(z['baseline_us']):.3f} | {float(z['current_us']):.3f} | {float(z['speedup']):.3f}× | {float(z['latency_reduction_pct']):.2f}% |\n"
text+='\n## 结论与当前采用的优化\n\n'
for v in variants:
 values=[float(lookup[v,w]['speedup'])for w in workloads];text+=f'- **{v}**：四组加速比范围 {min(values):.3f}～{max(values):.3f}×。\n'
best=max(rows,key=lambda z:float(z['speedup']));text+=f"\n最大加速来自 **{best['variant']} / {best['workload']}：{float(best['speedup']):.3f}×**。Base/XSF只有线程配置等调整，小幅差异仍可能包含运行波动；FP8/Quantized还包含编码、分块、grid、写回和预取优化。本次是累计效果对照，不能从这张表单独分摊每项优化的收益。\n\n"
text+='''| 已采用方法 | 当前使用范围 |
|---|---|
| v003：FP32/整数直接编码E4M3，避开SDK FP64路径 | FP8/Quantized |
| v004＋v007/v009/v010：hidden分块与实测tile/threads选择 | 按shape和变体分派 |
| v020：hidden-first grid | FP8/Quantized的h7168、prefill |
| v022：编码与写回分开 | FP8/Quantized的tiny、prefill |
| v025：8行寄存器预取 | FP8/Quantized的h3072、h7168 |

当前正式实现未使用shared中转或硬件async；本次未新增优化或采集新的Roofline。

## 测量与复现

- 两版先完成JIT，使用相同输入和预分配输出；只统计device kernel时间。
- 项目 `bench_kernel`：10次warmup、50repeat×3trials，CUPTI，每次L2flush。
- 外层5轮交替两版先后顺序，各自取中位数；random路由，seed=1235。
- tiny/h3072/h7168克隆输入；prefill因项目1GiB克隆阈值不克隆，两版条件相同。
- 16组全部逐字节输出一致，全部使用CUPTI；单次配对复测不是严格统计置信区间。

```bash
cd /data/TileOPs-Metax
source /data/sc16g-recovery-20261004/env.sh
python profiles/v026_SUM/scripts/benchmark.py
python profiles/v026_SUM/scripts/render_summary.py
```

重跑会覆盖本目录结果，先保留需要的历史数据。

- [16组原始CSV](../raw/comparison_16g.csv)：包含实际分派配置、grid、vector_store、prefetch_rows等。
- `raw/*_samples.json`：每版各5轮原始时延、计时方式与克隆状态。
- [v000源码快照](../codegen/v000.py)、[v025源码快照](../codegen/v025.py)。
- [源码来源与协议](../meta/source.json)、[软件环境](../meta/environment.json)、[GPU状态](../meta/gpu_before.txt)、[运行日志](../logs/benchmark.log)。
'''
report=r/'analysis/README.md';notes_marker='\n### 计算方法'
previous=report.read_text() if report.exists() else ''
if notes_marker in previous:text+='\n'+notes_marker+previous.split(notes_marker,1)[1]
report.write_text(text)
p=repo/'profiles/README.md';s=p.read_text();needle='\n## 后续约定';row='| [v026_SUM](v026_SUM/analysis/README.md) | v000与当前v025的16组同轮配对加速比总结 |\n';assert needle in s
if row not in s:s=s.replace(needle,row+needle)
s=s.replace('\n\n| [v026_SUM]','\n| [v026_SUM]')
p.write_text(s)
(r/'meta/completion.json').write_text(json.dumps({'cases':16,'all_byte_equal':True,'all_timing_cupti':True,'baseline':'v000','current':'v025','current_commit':source['current_commit'],'summary_only':True},indent=2)+'\n')
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(r.rglob('*'))if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts};(r/'meta/artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
bounds=r/'scripts/bandwidth_bounds.py'
if bounds.exists():
 subprocess.run([sys.executable,str(bounds)],check=True,capture_output=True,text=True)
print('SUMMARY_CREATED')
for v in variants:print(v,[(w,round(float(lookup[v,w]['speedup']),4))for w in workloads])
