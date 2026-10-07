from pathlib import Path
import csv,json,hashlib
r=Path(__file__).resolve().parents[1]
rows=[]
for z in csv.DictReader((r/'raw/comparison_16g.csv').open()):
 t,k,h=map(int,[z['num_tokens'],z['num_topk'],z['hidden']]);sf=z['variant']in['FP8','Quantized'];xsf=z['variant']in['XSF','Quantized']
 input_bytes=t*k*h*2;output_bytes=t*h*(1 if sf else 2);weights_bytes=t*k*4;positions_bytes=t*k*4;x_sf_bytes=t*k*4 if xsf else 0;sf_bytes=4 if sf else 0
 q=input_bytes+output_bytes+weights_bytes+positions_bytes+x_sf_bytes+sf_bytes
 reference_bw=1.50;ideal=q/1.8432e6;reference=q/(reference_bw*1e6);current=float(z['current_us'])
 rows.append(dict(variant=z['variant'],workload=z['workload'],input_bytes=input_bytes,output_bytes=output_bytes,weights_bytes=weights_bytes,positions_bytes=positions_bytes,x_sf_bytes=x_sf_bytes,sf_bytes=sf_bytes,modeled_bytes=q,roofline_bandwidth_TBs=1.8432,roofline_ideal_us=ideal,probe_reference_TBs=reference_bw,probe_reference_us=reference,current_us=current,reference_attainment_pct=100*reference/current))
with(r/'raw/bandwidth_bound_comparison_16g.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
text='''## 带宽模型下界与当前耗时

下面的“下界”均依赖流量模型。假设每个路由槽的expert输出从HBM读取一次、最终输出写一次，weights/positions/x_sf各读一次。未计hidden分块造成的重复元数据读取；重复路由、缓存及实际事务粒度可使实际HBM流量不同，因此不是严格证明的算法下限。

设T=token数，K=top-k，H=hidden，所有本表输入每元素2字节：

```text
Q = 2*T*K*H                         # expert输入
  + (1 if FP8/Quantized else 2)*T*H # 输出
  + 8*T*K                           # FP32 weights + int32 positions
  + (4*T*K if XSF/Quantized else 0) # FP32 x_sf
  + (4 if FP8/Quantized else 0)     # FP32 sf
时间(μs) = Q / (带宽(TB/s) * 1,000,000)
```

- **Roofline理想时间**：用现有mcProfiler模型的标称HBM屋顶1.8432 TB/s，表示该流量模型下仅传输数据的乐观时间，不表示本容器一定能达到。
- **统一带宽参考时间**：按用户指定，全部变体统一采用1.50 TB/s。这是选定的比较基准；[v027复测](../../v027_DIS:v026/analysis/README.md)的最高独立复测中位数约1.492 TB/s，不能把1.50标成已经实测证明的硬上限。
- **参考达成率** = 统一带宽参考时间 / 当前耗时。它等于模型有效带宽 / 1.50 TB/s，**不是mcProfiler HBM utilization**。超过100%不截断。

| 变体 | workload | Roofline理想 μs | 1.50 TB/s参考 μs | 当前v025 μs | 参考达成率 |
|---|---|---:|---:|---:|---:|
'''
for z in rows:text+=f"| {z['variant']} | {z['workload']} | {z['roofline_ideal_us']:.3f} | {z['probe_reference_us']:.3f} | {z['current_us']:.3f} | {z['reference_attainment_pct']:.2f}% |\n"
text+='\n### 如何理解\n\n'
text+='- **tiny**：带宽参考时间约0.03 μs，当前约3.6～3.8 μs。CTA数量、设备执行固定成本和指令延迟不能忽略；这不意味着可以实现百倍加速，CUPTI不含CPU launch开销。\n'
text+='- **Base/XSF的h7168、prefill，以及FP8/Quantized的prefill**：非常接近1.50 TB/s参考，但不能据此证明达到绝对极限。\n'
lookup={(z['variant'],z['workload']):z for z in rows}
for variant,workload in [('FP8','h3072'),('FP8','h7168'),('Quantized','h3072'),('Quantized','h7168')]:
 z=lookup[variant,workload];gap=z['current_us']-z['probe_reference_us'];pct=100-z['reference_attainment_pct']
 text+=f"- **{variant} {workload}**：参考达成率{z['reference_attainment_pct']:.2f}%，与参考耗时差{gap:.2f} μs，占当前耗时{pct:.2f}%。\n"
prefill=[z for z in rows if z['workload']=='prefill']
low=min(z['reference_attainment_pct']for z in prefill);high=max(z['reference_attainment_pct']for z in prefill)
text+='\n### prefill 的优化是否已经收敛\n\n'
text+=f'按统一1.50 TB/s参考，四个prefill变体的达成率为 **{low:.2f}%～{high:.2f}%**；流量模型下剩余耗时差约 **{100-high:.2f}%～{100-low:.2f}%**。\n\n'
text+='在当前sc-16g实例、random路由和现有测试协议下，prefill已接近实测带宽参考，现有算法与接口下的性能优化基本收敛，可以作为当前阶段的停止点。1.50 TB/s不是已证明的硬上限，因此不能宣称达到这个shape的绝对极限；改变数据流或融合前后算子可能产生新的空间。\n'

text+='\n这些差距不保证全部能够通过优化消除；缓存、重复路由和真实事务流量也可能改变模型有效带宽。\n\n[完整字节数和计算CSV](../raw/bandwidth_bound_comparison_16g.csv)、[计算脚本](../scripts/bandwidth_bounds.py)。本次仅计算已有数据，不重跑GPU。\n'

p=r/'analysis/README.md';s=p.read_text();marker='\n## 带宽模型下界与当前耗时';notes_marker='\n### 计算方法'
notes=notes_marker+s.split(notes_marker,1)[1] if notes_marker in s else ''
notes=notes.replace('但是实际只有 1.49 TB/s 左右，所以实测参考大概是 41.895 μs','本次统一用 1.50 TB/s 作为比较参考，参考时间大概是 41.615 μs')
prefix=s.split(marker,1)[0].split(notes_marker,1)[0].rstrip()
s=prefix+'\n\n'+text
if notes:s+='\n'+notes.lstrip('\n')
p.write_text(s)
(r/'meta/bandwidth_bound_model.json').write_text(json.dumps({'roofline_bandwidth_TBs':1.8432,'probe_bandwidth_TBs':{v:1.50 for v in ['Base','XSF','FP8','Quantized']},'reference_basis':'user-specified uniform 1.50 TB/s; not a measured hard limit','strict_lower_bound_proven':False,'modeled_reads':'one x row per valid routing slot, metadata once per slot; output once','input_element_bytes':2,'data_source':'raw/comparison_16g.csv','cases':16},indent=2)+'\n')
hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(r.rglob('*'))if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts};(r/'meta/artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(text)
