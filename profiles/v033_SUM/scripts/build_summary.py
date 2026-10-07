from pathlib import Path
import csv

root = Path(__file__).resolve().parents[1]
with (root / 'raw/v026_comparison_16g.csv').open() as f:
    baseline = list(csv.DictReader(f))
with (root / 'raw/v026_bandwidth_bound_comparison_16g.csv').open() as f:
    bounds = {(x['variant'], x['workload']): x for x in csv.DictReader(f)}
with (root / 'raw/v032_formal_tiny_comparison_16g.csv').open() as f:
    tiny = {x['variant']: x for x in csv.DictReader(f)}
rows = []
for x in baseline:
    variant, workload = x['variant'], x['workload']
    old = float(x['baseline_us'])
    current = float(tiny[variant]['current_us']) if workload == 'tiny' else float(x['current_us'])
    byte_count = int(bounds[variant, workload]['modeled_bytes'])
    lower = byte_count / 1.5e6
    rows.append(dict(variant=variant, workload=workload,
        num_tokens=int(x['num_tokens']), num_topk=int(x['num_topk']), hidden=int(x['hidden']),
        dtype=x['dtype'], baseline_us=old, current_us=current,
        speedup=old / current, modeled_bytes=byte_count,
        bandwidth_reference_TBs=1.5, bandwidth_lower_bound_us=lower,
        lower_bound_attainment_pct=100 * lower / current,
        baseline_source='v026_comparison_16g.csv',
        current_source='v032_formal_tiny_comparison_16g.csv' if workload == 'tiny' else 'v026_comparison_16g.csv',
        paired_baseline_current=workload != 'tiny'))
with (root / 'raw/summary_16g.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
table = ['| 变体 | Workload | v000 | 当前 | 加速比 | 带宽理想下限 | 下限达成率 |',
         '|---|---|---:|---:|---:|---:|---:|']
for x in rows:
    precision = 5 if x['workload'] == 'tiny' else 3
    table.append(f"| {x['variant']} | {x['workload']} | {x['baseline_us']:.3f} | {x['current_us']:.3f} | {x['speedup']:.3f}× | {x['bandwidth_lower_bound_us']:.{precision}f} | {x['lower_bound_attainment_pct']:.2f}% |")
report = """# v033：当前正式版本与 v000 的性能及带宽下限总结

## 数据范围

当前正式实现以提交 `037a8028` 为准：Base/XSF/FP8 的加权 FP16 tiny 使用 v030 两线程协作，Quantized tiny 保留 v029；其他 workload 保持 v025 及之前已接入的优化。v031、v032 未接入正式 kernel。

tiny=T32/K2/H256、FP16；h3072=T512/K8/H3072，h7168=T512/K8/H7168，prefill=T4096/K8/H7168，后三组输入为 BF16。

本次汇总使用已有测量，没有重新运行 benchmark：

- v000 基线和非 tiny 的十二组当前耗时取自 v026 的同轮配对测量。
- 四个 tiny 当前耗时取自 v032 第二种打包实验中的正式实现控制组，即原数据的 `current_us`，不是未采用候选的 `candidate_us`。
- tiny 的 v000 与当前值来自不同轮测量，其加速比属于参考对比。
- 项目计时口径为 CUPTI、L2 flush、10 次 warmup、50 repeat × 3 trials；tiny/h3072/h7168 克隆输入，prefill 不克隆。这里比较的是设备 kernel 时长。

## 计算口径

加速比 = v000 耗时 / 当前耗时。

带宽理想下限（μs）= 模型数据量（字节）/ 1,500,000（字节/μs）。带宽参考为 **1.50 TB/s**，采用十进制单位。

下限达成率 = 带宽理想下限耗时 / 当前耗时 × 100%。越接近 100%，越接近这一参考下限。

所有 workload 使用相同的字节模型，按每个有效路由输入读取一次、输出写入一次、元数据读取一次计算：

```text
Q = 2*T*K*H + output_bytes_per_element*T*H + 8*T*K
    + (4*T*K，XSF/Quantized 使用 x_sf 时)
    + (4，FP8/Quantized 使用 sf 时)
```

Base/XSF 输出每元素 2 字节，FP8/Quantized 输出每元素 1 字节；positions 和 weights 各 4 字节。该模型是语义数据量参考，并非 profiler 测得的实际 HBM 流量。

## 十六组结果

时间单位均为 **μs**。计算使用原始精度，表中数值仅作显示舍入。

""" + '\n'.join(table) + """

## 结论与适用范围

- prefill 四组下限达成率为 98.82%–99.94%，已非常接近 1.50 TB/s 参考与当前字节模型给出的带宽下限；这支持实际性能接近收敛，不能证明已达到绝对硬件极限。
- h7168 为 92.47%–99.74%；h3072 为 80.00%–91.20%。
- tiny 为 0.75%–0.99%。纯带宽下限没有计入固定执行成本、有限并行度和访存依赖延迟，不适合据此判断 tiny 优化质量，也不代表存在百倍可实现的加速空间。
- 1.50 TB/s 是先前大工作集探针得到的实测带宽参考，不是针对每一种访问模式证明的硬上限。缓存、访存形态和测量波动都会影响比较。

## 数据与复现

[完整精度汇总 CSV](../raw/summary_16g.csv)。输入 CSV 已在本版本 raw 目录冻结，来源与哈希见 [source.json](../meta/source.json)。

仅重新计算并生成本报告，不运行 GPU：

```bash
cd /data/TileOPs-Metax
python profiles/v033_SUM/scripts/build_summary.py
```
"""
(root / 'analysis/README.md').write_text(report)
print('已生成 v033_SUM 的16组汇总与报告')
