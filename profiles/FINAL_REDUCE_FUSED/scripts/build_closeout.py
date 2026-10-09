from pathlib import Path
import csv, json

root=Path(__file__).resolve().parents[1]
with (root/'raw/summary_16g.csv').open() as f:
    rows=list(csv.DictReader(f))
by={(x['variant'],x['workload']):x for x in rows}
for x in rows:
    x['baseline_us']=float(x['baseline_us'])
    x['current_us']=float(x['current_us'])
    x['modeled_bytes']=int(x['modeled_bytes'])
    x['bandwidth_lower_bound_us']=x['modeled_bytes']/1.5e6
    x['speedup']=x['baseline_us']/x['current_us']
    x['lower_bound_attainment_pct']=100*x['bandwidth_lower_bound_us']/x['current_us']
    x['latency_reduction_pct']=100*(1-x['current_us']/x['baseline_us'])
    x['saved_vs_baseline_us']=x['baseline_us']-x['current_us']
    x['gap_to_bandwidth_lower_us']=x['current_us']-x['bandwidth_lower_bound_us']
    w,v=x['workload'],x['variant']
    h=int(x['hidden']);t=int(x['num_tokens'])
    sf=v in ('FP8','Quantized')
    if w=='tiny':
        tile=256;threads=512 if v=='FP8' else 256
        impl='v029 local' if v=='Quantized' else 'v030 adjacent pair'
        grid_x,grid_y=t,1;prefetch=0
    elif sf:
        tile=1024 if w=='prefill' else 512;threads=128
        impl='general';prefetch=0 if w=='prefill' else 8
        grid_x,grid_y=(h//tile,t) if w in ('h7168','prefill') else (t,h//tile)
    else:
        tile=h;threads=256;impl='general';prefetch=0;grid_x,grid_y=t,1
    x.update(tile_hidden=tile,num_threads=threads,grid_x=grid_x,grid_y=grid_y,
        cta_count=grid_x*grid_y,implementation=impl,prefetch_rows=prefetch)
# Native HBM Usage comes from mcProfiler RoofLine, not semantic-byte estimates.
native_usage={}
native_csv=root/'raw/hbm_native/usage_16g.csv'
if native_csv.exists():
    with native_csv.open()as f:
        native_usage={(x['version'],x['variant'],x['workload']):x for x in csv.DictReader(f)}
for x in rows:
    x['baseline_lower_bound_attainment_pct']=100*x['bandwidth_lower_bound_us']/x['baseline_us']
    variant=x['variant'].lower()
    baseline_native=native_usage.get(('v000',variant,x['workload']))
    final_native=native_usage.get(('final',variant,x['workload']))
    x['baseline_hbm_usage_native_pct']=float(baseline_native['hbm_usage_pct'])if baseline_native else ''
    x['current_hbm_usage_native_pct']=float(final_native['hbm_usage_pct'])if final_native else ''
    x['baseline_hbm_native_source']=baseline_native['source_record']if baseline_native else ''
    x['current_hbm_native_source']=final_native['source_record']if final_native else ''
    x['hbm_native_roof_GBs']=float(final_native['native_roof_GBs'])if final_native else float(baseline_native['native_roof_GBs'])if baseline_native else ''
    x['hbm_report_reference_GBs']=1500.0
    x['baseline_hbm_bandwidth_native_GBs']=float(baseline_native['native_bandwidth_GBs'])if baseline_native else ''
    x['current_hbm_bandwidth_native_GBs']=float(final_native['native_bandwidth_GBs'])if final_native else ''
    x['baseline_hbm_usage_1p5_pct']=100*float(baseline_native['native_bandwidth_GBs'])/1500 if baseline_native else ''
    x['current_hbm_usage_1p5_pct']=100*float(final_native['native_bandwidth_GBs'])/1500 if final_native else ''
    x['current_hbm_usage_projected_1p5_pct']=x['baseline_hbm_usage_1p5_pct']*x['speedup'] if baseline_native else ''

def native_pct(value):
    return f'{value:.2f}%'if value!=''else '未取得可靠计数'

with (root/'raw/closeout_16g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
native_reference_rows=[]
for x in native_usage.values():
    y=dict(x)
    y['report_reference_GBs']=1500.0
    y['hbm_usage_reference_1p5_pct']=100*float(x['native_bandwidth_GBs'])/1500
    native_reference_rows.append(y)
if native_reference_rows:
    with (root/'raw/hbm_native/usage_reference_1p5_16g.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(native_reference_rows[0]));writer.writeheader();writer.writerows(native_reference_rows)
    index_path=root/'analysis/hbm_native/README.md'
    if index_path.exists():
        import os
        text=index_path.read_text()
        begin=text.index('| 变体 | Workload |')
        end=text.index('\n\n',begin)
        table=['| 变体 | Workload | v000 HBM usage（1.50 TB/s） | 终盘 HBM usage（1.50 TB/s） | 终盘 HBM usage（推算） | v000图 | 终盘图 |',
               '|---|---|---:|---:|---:|---|---|']
        for var,label in [('base','Base'),('xsf','XSF'),('fp8','FP8'),('quantized','Quantized')]:
            for work in ['tiny','h3072','h7168','prefill']:
                a=native_usage.get(('v000',var,work));b=native_usage.get(('final',var,work))
                if not(a and b):continue
                first=100*float(a['native_bandwidth_GBs'])/1500;last=100*float(b['native_bandwidth_GBs'])/1500
                final_row=by[label,work];projected=first*final_row['speedup']
                links=[os.path.relpath(root.parents[1]/x['image'],index_path.parent)for x in [a,b]]
                table.append(f'| {label} | {work} | {first:.2f}% | {last:.2f}% | {projected:.2f}% | [图]({links[0]}) | [图]({links[1]}) |')
        text=text[:begin]+'\n'.join(table)+text[end:]
        text=text.replace('原生HBM usage读取工具的 `case_bandwith/MAX_Bandwith`，屋顶1843.2 GB/s；不是用算法字节除以benchmark runtime估算。',
            '文档HBM usage表以1.50 TB/s为参考：`100 × native case_bandwith /1500`，等价于原生百分比乘1.2288。原始CSV与PNG保留1843.2 GB/s工具屋顶。它使用native带宽，不是算法字节/benchmark runtime估算。')
        if '超过100%'not in text:
            text=text.replace('## 采集口径与数据边界','## 采集口径与数据边界\n\n1.50 TB/s是近似实测参考，换算值可以略超过100%，保留实际数值。图中的百分比仍是原生1.8432 TB/s口径，与本表换算值不同。')
        index_path.write_text(text)

meta=json.loads((root/'meta/source.json').read_text())

def record(v,w):
    x=by[v,w]
    return (f"{x['current_us']:.3f} μs，较 v000 加速 {x['speedup']:.3f}×、"
            f"耗时下降 {x['latency_reduction_pct']:.2f}%；下限达成率 "
            f"{x['lower_bound_attainment_pct']:.2f}%")

def gap(v,w):return f"{by[v,w]['gap_to_bandwidth_lower_us']:.3f} μs"

perf=['| 变体 | Workload | v000 μs | 终盘 μs | 加速比 | 耗时下降 | 1.50 TB/s 下限 μs | v000 下限达成率 | 终盘下限达成率 | v000 HBM usage（1.50 TB/s） | 终盘 HBM usage（1.50 TB/s） | 终盘 HBM usage（推算，1.50 TB/s） |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
    p=5 if x['workload']=='tiny' else 3
    perf.append(f"| {x['variant']} | {x['workload']} | {x['baseline_us']:.3f} | {x['current_us']:.3f} | **{x['speedup']:.3f}×** | {x['latency_reduction_pct']:.2f}% | {x['bandwidth_lower_bound_us']:.{p}f} | {x['baseline_lower_bound_attainment_pct']:.2f}% | **{x['lower_bound_attainment_pct']:.2f}%** | {native_pct(x['baseline_hbm_usage_1p5_pct'])} | {native_pct(x['current_hbm_usage_1p5_pct'])} | {native_pct(x['current_hbm_usage_projected_1p5_pct'])} |")

methods={
('Base','tiny'):'v030：相邻两线程分担 K=2 输入读取；按 K 顺序累加；每对两列；输入/输出均合并为 4B',
('Base','h3072'):'v009/v010：每 token 一个 CTA，整行 256 线程，按 shape 固定配置；普通向量访存',
('Base','h7168'):'同 Base/h3072，整行 tile 改为 7168',
('Base','prefill'):'同 Base/h7168，token 数增至 4096',
('XSF','tiny'):'同 Base/tiny；语义上额外乘 x_sf[pos]',
('XSF','h3072'):'同 Base/h3072；语义上额外乘 x_sf[pos]',
('XSF','h7168'):'同 Base/h7168；语义上额外乘 x_sf[pos]',
('XSF','prefill'):'同 Base/prefill；语义上额外乘 x_sf[pos]',
('FP8','tiny'):'v003 的直接 FP32→E4M3 编码 + v030 两线程协作；每对一列、512 线程；单元素读取/写回',
('FP8','h3072'):'v003 编码 + v004/v007/v010 hidden 分块与配置选择 + v025 深度 8 输入寄存器预取',
('FP8','h7168'):'同 FP8/h3072，另有 v020 hidden-first grid；14 个 hidden 块/token',
('FP8','prefill'):'v003 编码 + hidden 分块/配置选择 + v020 hidden-first grid + v022 先编码后集中 8B 写回；不启用深度 8 预取',
('Quantized','tiny'):'v003 编码 + v029 local：每线程一列，K=2 展开，positions/weights 成对载入；不使用 shuffle',
('Quantized','h3072'):'同 FP8/h3072；语义上额外乘 x_sf[pos]',
('Quantized','h7168'):'同 FP8/h7168；语义上额外乘 x_sf[pos]',
('Quantized','prefill'):'同 FP8/prefill；语义上额外乘 x_sf[pos]',
}
config=['| 变体 | Workload | tile / 线程 | grid(x,y) | CTA 数 | 终盘优化与相同项引用 |',
        '|---|---|---|---|---:|---|']
for x in rows:
    config.append(f"| {x['variant']} | {x['workload']} | {x['tile_hidden']} / {x['num_threads']} | ({x['grid_x']}, {x['grid_y']}) | {x['cta_count']:,} | {methods[x['variant'],x['workload']]} |")

report=f"""# FINAL — reduce_fused 封盘报告（restart / sc-16g）

## 1. 封盘范围与版本边界

本报告只总结 **restart 分支已接入的正式 kernel**，以 `{meta['reference_commit']}` 为版本锚点。该提交的正式性能实现与 `037a8028` 相同；`931c10a2` 保存了 v032 失败实验及 v033 总结。v000 naive 基线为 `ffc084d94b4efa66e763fdcc0a130905c18ed836`。

所有引用数据、源码和版本记录均从该 restart 提交的 Git tree 提取。不引用其他分支的代码、实验或数据，也不依据磁盘上额外目录的 ACC/TRIAL 标签认定正式优化。本报告作为独立封盘记录保存于 **FINAL_REDUCE_FUSED**，不使用迭代版本编号；封盘标签为 **reduce-fused-final**。

封盘 runtime 沿用已有 benchmark，未修改 kernel 或重新运行正确性测试。为补充 HBM usage，复用 v021 的 v000 Base/FP8 八组 native profile，并于 2026-10-09 补齐另外二十四组；32条原生记录均已核对，历史异常记录保留但不用于最终表格。封盘基准日期为2026-10-08，本次native HBM补表完成日期为2026-10-09；runtime与native指标各自的采集时间和口径分别记录。

### 最终判断

- **大 workload：**prefill 四组下限达成率 98.82%–99.94%；Base/XSF h7168 达 99.35%–99.74%。在当前实例及字节模型下，继续追求大幅收益的依据不足，可以作为本轮封盘点。
- **收益最大的组合：**FP8/h7168 2.197×，Quantized/h7168 2.097×；FP8/h3072 1.985×，Quantized/h3072 1.898×。
- **仍有参考差距的组合：**FP8/Quantized h3072 达成率 84.36%/80.00%，不能把整个算子概括为所有 shape 均达到极限。
- **tiny：**终盘约 3.34–3.70 μs，较 v000 加速 1.102×–1.325×。此规模应看设备时延和同规模对照，不能用不足 1% 的纯带宽达成率评价优化质量。
- 这是当前调用方式、shape、精度要求和实例条件下的工程封盘，不是证明不存在更快实现。

## 2. 算子语义、四个变体与四种规模

令 `p=positions[token,k]`。有效路由先读取 `x[p,h]`，乘路由权重；启用 x_sf 时，系数再乘 `x_sf[p]`；按 K 顺序使用 FP32 累加。`p<0` 的路由跳过。启用 sf 时，最终乘全局标量 sf，转换成 E4M3FN FP8 输出。

| 变体 | weights | x_sf | 全局 sf | 输出 |
|---|---|---|---|---|
| Base | 使用 | 不使用 | 不使用 | 与输入 dtype 相同 |
| XSF | 使用 | 使用，每个展开行一个缩放因子 | 不使用 | 与输入 dtype 相同 |
| FP8 | 使用 | 不使用 | 使用，最终输出缩放 | FP8 E4M3FN |
| Quantized | 使用 | 使用 | 使用 | FP8 E4M3FN |

weights 在本报告全部使用。sf/x_sf 是算子语义，不是优化手段；FP8/Quantized 的明显收益来自改进转换和执行组织，不能说“加上 sf 就变快”。sf 路径同时改变输出 dtype，输入仍为 FP16/BF16。

| Workload | T | K | H | 输入 | Base/XSF 输出 | FP8/Quantized 输出 |
|---|---:|---:|---:|---|---|---|
| tiny | 32 | 2 | 256 | FP16 | FP16 | FP8 |
| h3072 | 512 | 8 | 3072 | BF16 | BF16 | FP8 |
| h7168 | 512 | 8 | 7168 | BF16 | BF16 | FP8 |
| prefill | 4096 | 8 | 7168 | BF16 | BF16 | FP8 |

## 3. 最终十六组性能

加速比 = v000 耗时 / 终盘耗时；耗时下降 = (1 − 终盘耗时 / v000 耗时) × 100%。下限达成率的口径见第 4 节。计算使用原始精度，显示数值经过舍入。

"""+'\n'.join(perf)+f"""

### 新增三列的口径与采集来源

- **v000 下限达成率** = 同一模型的 1.50 TB/s 下限耗时 / v000 benchmark runtime × 100%。
- **HBM usage（1.50 TB/s）** = `100 × RoofLine.data.case_bandwith / 1500`。使用工具原生实测带宽，以统一参考1.50 TB/s重新归一化；等价于原生百分比乘以 `1.8432 / 1.50 = 1.2288`。流量与周期仍来自native counter，不使用语义Q/runtime替代。原生百分比及1843.2 GB/s工具屋顶保留在原始CSV和PNG中。
- v000 Base/FP8 的八组复用 [v021 原始 profile](../../v021_SUM/analysis/README.md)；另外二十四组已用封盘源码及原naive快照于2026-10-09补齐，均通过几何、流量与周期尺度核对。为使批量采集可按 case 区分，编译时只重命名 kernel symbol，生成设备代码在归一化该符号后与原代码相同。
- native采集使用seed=1235、同地址和显式start/stop。复用的八组以及两个先取得有效计数的大case使用10次预热；剩余22组在JIT准备完成后，每组只调用一次，不在采集进程内执行目标预热，以使trace与指标重放顺序对应。没有逐次L2 flush或输入克隆。它与表中的 benchmark 不是同一次采集，不能用两列差值直接归因。tiny 特别容易受缓存及计时口径影响。
- 此前异常采集未用于最终表格；2026-10-09补齐的24条均逐组核对WORKGROUPS、流量及周期尺度。HBM usage的带宽仍依赖工具RoofLine的计数与时钟模型。1.50 TB/s是近似实测参考，换算值可以略超过100%，不截断，也不能当作已证明的硬上限。
- [32组 native 指标及原始记录路径](../raw/hbm_native/usage_16g.csv)。新增采集及历史失败记录在 `raw/hbm_native/`，命令与状态在 `meta/hbm_native/`。24张有效新图以`verified_`前缀保存在`analysis/hbm_native/`；复用八图链接到v021，完整图索引见 [32组Roofline对照](hbm_native/README.md)。没有`verified_`前缀的旧诊断图不用于最终表格。

### 终盘 HBM usage 推算列

新增列按 `v000 HBM usage（1.50 TB/s）× benchmark加速比` 计算，使用未舍入的原始数据。

```text
U_final_projected = U_v000_native_1p5 * (t_v000_benchmark / t_final_benchmark)
```

该推算假设两版实际HBM搬运字节量相同，带宽参考保持1.50 TB/s，且将benchmark加速比例应用到v000的native带宽基准上。它用于展示在这些假设下的终盘比例，不是独立profile实测；不能再用这列反过来证明benchmark加速。原终盘实测列继续展示其各自采集时的观察值，不能把跨批次、不同预热/计时的两列直接当成严格优化对照。推算超过100%时保留数值。

### 数据来源和比较边界

- 非 tiny 十二组：v026 的同轮 v000/正式 v025 配对测量。后续 restart 的正式代码改动只改变目标 tiny 分派，这十二组沿用已存测量，不能称为今天新测的终盘值。
- tiny 四组：v000 取 v026；正式终盘值取 v032 第二种打包实验中的 **current_us 控制组**。没有使用失败候选的 candidate_us，也没有挑选每组历史最低值。tiny 加速比属于跨轮参考，没有新的同轮 v000/终盘全量复测或置信区间。
- 测量机器为 sc-16g、MetaX C500 的 16GB 配额实例；历史记录为 25% compute 配额。v026 保存的 Torch 为 `2.8.0+metax3.7.1.3`，不能把后续重建环境版本替换成测量时版本。
- 项目 `bench_kernel`：10 warmup，50 repeat × 3 trials，CUPTI 设备 kernel 时间，L2 flush，外层配对交替测量；v026、v032 都记录了五轮。tiny/h3072/h7168 按项目规则克隆输入；prefill 不克隆，基线与候选条件一致。
- 该 runtime 不等于包含 Python、CPU 提交、排队和同步的端到端调用时间；缓存条件变化也可能改变实际部署表现。

[完整精度、配置、绝对节省和下限差距 CSV](../raw/closeout_16g.csv)。冻结输入在 raw；来源提交及哈希在 [source.json](../meta/source.json)。

## 4. 1.50 TB/s 下限如何计算、怎样使用

### 4.1 字节模型

输入每元素 2B；Base/XSF 输出每元素 2B，FP8/Quantized 输出每元素 1B。positions 和 weights 各为 4B。对于本表的有效路由数据量模型：

```text
Q = 2*T*K*H                         # expert 输入
  + output_bytes_per_element*T*H    # 输出
  + 4*T*K                           # positions
  + 4*T*K                           # weights
  + 4*T*K（仅 XSF / Quantized）      # x_sf
  + 4（仅 FP8 / Quantized）          # 全局 sf

带宽理想下限(μs) = Q / 1,500,000
下限达成率 = 带宽理想下限 / 终盘 runtime × 100%
```

例如 FP8/h3072：Q=26,771,460B，下限 17.84764 μs；实际 21.15584 μs，达成率 84.36%。Base/prefill：Q=528,744,448B，下限 352.49630 μs，实际 352.70144 μs，达成率 99.94%。

### 4.2 为什么采用 1.50 TB/s

[v027 的大工作集独立复测](../../v027_DIS:v026/analysis/README.md)确认：1:1 copy 约 1.4645–1.4652 TB/s；8:1 读写探针约 1.4472–1.4660 TB/s；16:1 探针 1GiB 中位数为 1.4920 TB/s、2GiB 为 1.4742 TB/s。此前 v023 的 16:1 1GiB 结果约 1.4979 TB/s。因此统一采用约 **1.50 TB/s 的高位实测参考**。

算法主要输入/输出比例：Base/XSF 的 K=8 大 workload 约 8:1，FP8/Quantized 约 16:1。不同比例探针的结果已经说明带宽并非只由一个固定数字决定。1.50 TB/s 不是已证明对每一种访问模式都不可超过的硬上限。

初始 native Roofline 曾使用 2048B/cycle × 0.9GHz = 1.8432 TB/s 的屋顶参考。本报告的带宽理想下限与下限达成率按用户指定的 1.50 TB/s 计算；HBM usage列读取native RoofLine实测带宽，并同样以1.50 TB/s重新归一化。两种达成率使用相同分母，但前者来自模型字节/benchmark时间，后者来自native计数/周期，因此不一定相等。原生图仍采用工具的1.8432 TB/s屋顶。

### 4.3 模型的限制

- 这是按算法语义字节数得到的带宽参考，不是 profiler 实测 HBM usage。真实 HBM 字节还受缓存、重复路由、metadata 的跨 CTA 重读、事务粒度等因素影响。
- hidden 分块后的多个 CTA 不复用同一个 x 元素，但会重复 position/weight/x_sf 等 metadata；字节模型按语义读取一次，不把全部逻辑重读直接视为 HBM 流量。
- 纯 Q/B 没有包含有限并行度、依赖链、指令执行和固定设备成本。tiny 下限 0.028–0.033 μs 并不现实；不足 1% 不表示还有百倍可实现加速。
- 接近 100% 支持“接近当前带宽参考”，不能证明绝对最优。超过 100% 时首先检查流量模型、带宽参考、缓存和计时口径。

## 5. 终盘配置与每组采用的优化

以下均为正式默认配置；用户手动 config 或未测 shape/dtype/无权重场景有回退逻辑，不在本表封盘范围内。grid 是 block 索引与任务的映射，不是硬件实际调度顺序。

"""+'\n'.join(config)+f"""

### 5.1 Base/XSF 大 workload：整行 256 线程

v009 比较整行与 hidden 分块后，选择每 token 一个 CTA、256 线程；v010 将按 shape/变体选择配置接入正式包装层。h3072/h7168/prefill 的 tile 分别为 3072/7168/7168。这些路径没有采用 FP8 编码、hidden-first grid、深度 8 预取或 shared 中转。

加大 CTA 数不是普遍有效的方向。Base/XSF 在大 shape 的整行配置已经接近带宽参考，继续分块增加 CTA 和 metadata 工作，历史扫描没有超过所选整行配置。具体代价分解未通过新硬件计数验证。

### 5.2 FP8/Quantized：直接编码与 hidden 分块

v003 用 FP32/uint32 位运算直接产生 E4M3FN 字节，替代 SDK 通用 FP32→FP64→FP8 软件路径；处理 RNE 最近偶数舍入、非正规数、SATFINITE 饱和、符号与 NaN。没有降低累加精度，也没有改成先分别舍入两个乘积再相加。

v004 允许多个 CTA 分担一行 hidden；v007/v010 选择已测 tile/线程。终盘 h3072 为 6 块/token，h7168 为 14 块/token，prefill 为 7 块/token。编码成本与每 CTA 的工作量下降，CTA 数增多；单项优化收益不能与不同轮其他收益直接相乘。

### 5.3 h7168/prefill：hidden-first grid

v020 将 FP8/Quantized 的 grid 从 (token,hidden块) 改为 (hidden块,token)，使相邻展平 block 编号更集中到同 token 的相邻列块。不搬移输入，不减少必需 x 字节，不增加 shared。

h7168 历史地址分析显示：前 14 个 block 涉及不同 x 行由 112 行变为 8 行，相邻同槽位块起点差的中位数由 16,930,816B 变为 1,024B；必需 x 读取量仍为 114,688B。它改善请求组织的解释有地址和配对性能支持，但不能直接称为已证实 L2 命中率提升或实际调度顺序固定。h3072 未接入此项，因为同轮比较没有收益。

### 5.4 h3072/h7168：深度 8 输入寄存器预取

v025 先将 8 个 expert 输入列块组织读入 `[8,tile_hidden]` 的输入 dtype fragment，再按原 K 顺序加权累加。它是普通 global→register 软件预取，没有 shared、async ticket 或 CTA barrier。编译器会调度最终指令，不能只凭 TileLang 顺序保证硬件完成全部加载后才执行第一个 FMA。

历史资源查询中寄存器/thread 从 12 增至 22，private/shared 均为 0，原生资源允许 CTA/AP 上限仍为 16；此上限不是实测驻留数。本项在 h3072/h7168 的 FP8/Quantized 上有独立复测收益，且 h7168 上能与 grid 优化叠加。prefill 未采用此项。

### 5.5 prefill：先编码，再集中写回

v022 将逐段编码/写回改为先生成寄存器 fragment 中的 FP8 字节，再集中写回。所选 prefill 几何每线程有 8 个 FP8，机器 IR 从两条 4B store 变为一条 8B store；输入仍为一次 16B 读取。编码展开也同时变化，不能把所有收益只归因于 store 宽度。

终盘 tiny 已由 v029/v030 helper 替换。虽然生产分派仍保留 vector_store 标志，不能据此声称 tiny 仍执行 v022 的 8B 写回：FP8 tiny 每输出线程一列，Quantized tiny 每线程一列，实际是单字节输出。v022 的 8B 写回主要对应终盘 prefill。

### 5.6 tiny：两线程协作与独立 local 两条路径

Base/XSF：相邻 lane 配对，两个线程分别读 expert0/expert1，shuffle 交换位置、系数与 FP32 输入；expert0 线程按原顺序完成累加并写回。256 线程形成 128 对，每对处理两列；生成代码已合并两个相邻 FP16 为一次 4B 读取，输出也是 4B 写回。

FP8：同样两线程协作，但 512 线程形成 256 对，每对一列。历史比较中该几何优于 256 线程候选，因此没有统一缩成 256 线程。终盘每线程读一个 FP16，leader 编码并写一个 FP8。

Quantized：v029 的 local 实现更合适，256 线程、每线程一列，K=2 展开；position/weight 成对载入，无 shuffle。两线程方案没有确认超过它，因此没有为了统一四个变体而强行采用 pair。

生产入口：[reduce_fused.py](../../../tileops/kernels/moe/reduce_fused.py)，私有 tiny 工厂：[reduce_fused_tiny.py](../../../tileops/kernels/moe/reduce_fused_tiny.py)。封盘源码副本在 codegen，哈希见 meta。

## 6. 每个 workload 的终盘分析

本节分别给出十六组判断；相同优化不重复展开，明确引用第 5 节的对应组合。计数器未确认的细分原因只作为解释，不包装成事实。

### 6.1 Base

**Base/tiny。** {record('Base','tiny')}。采用第 5 节 Base/tiny 配置，两线程分担 K=2 读取是终盘主要新增手段。相较旧独立线程路径，减少了单线程串行处理两行的工作，但引入 shuffle；收益由测量确认，不能精确分摊为访存等待减少多少。约 3.34 μs 应看同规模时延，纯带宽下限不适用于这里。相邻两列已合并读取，不能再把“改成一次 4B 读”当作尚未做过的优化。

**Base/h3072。** {record('Base','h3072')}。整行 256 线程，距参考下限约 {gap('Base','h3072')}。已有约 91% 的参考达成率，但不如 h7168/prefill 接近；较小工作规模、有限并发或指令依赖可能使满带宽较难实现。历史 hidden 分块未超过整行方案，暂不能据这 9% 差距指定某个未确认瓶颈。

**Base/h7168。** {record('Base','h7168')}。优化手段同 Base/h3072，仅 tile=7168；距参考下限约 {gap('Base','h7168')}。v000 已很接近最终表现，整体加速仅约 0.6%，符合原路径已经高效的结果。在现有模型下已基本收敛，微小差距可能与测量波动同量级，不能把它当作保证可获得的收益。

**Base/prefill。** {record('Base','prefill')}。优化手段同 Base/h7168，T 增至 4096；距参考下限约 {gap('Base','prefill')}。大工作量能更充分维持带宽，当前值几乎贴住 1.50 TB/s 模型参考。本轮可以停止局部调参；若要进一步明显降低端到端成本，更需要上游融合或减少算法/物化流量，而不是期待 shared 缓存一个只消费一次的 x 元素。

### 6.2 XSF

**XSF/tiny。** {record('XSF','tiny')}。优化同 Base/tiny，差别是系数包含 x_sf[pos]。它比 Base 同轮约多 0.020 μs，但 tiny 的几十 ns 差不能独立解释为 x_sf 的准确开销；当前两者总体接近。终盘保留 pair256；v032 合并有效标志与输入的试验反而明显变慢，未接入。

**XSF/h3072。** {record('XSF','h3072')}。优化同 Base/h3072，距参考下限约 {gap('XSF','h3072')}。额外 x_sf 读取在模型中只有 16,384B，并增加每路由缩放系数乘法，相较约 28MB 总流量较小。XSF 本轮甚至略快于 Base，说明不能用跨变体时延差直接断定缩放没有成本；配置和测量条件下它们属于相近性能水平。

**XSF/h7168。** {record('XSF','h7168')}。优化同 Base/h7168，距参考下限约 {gap('XSF','h7168')}。加入 x_sf 后仍保持约 99% 达成率，不需要为此引入额外 shared 或分块。最终提升仍小，主要结论是该路径已接近带宽参考，而不是所有尝试都应得到更大收益。

**XSF/prefill。** {record('XSF','prefill')}。优化同 Base/prefill，距参考下限约 {gap('XSF','prefill')}。尽管同时有额外缩放读取和乘法，达成率仍为 99.42%。该结果与 Base/prefill 一起支持大规模非 FP8 路径在本轮接近收敛；不构成针对全部访问模式的最优性证明。

### 6.3 FP8

**FP8/tiny。** {record('FP8','tiny')}。采用 v003 直接编码与 v030 pair512，一列/线程对；总体节省约 {by['FP8','tiny']['saved_vs_baseline_us']:.3f} μs。编码优化在小数据上也减少实际指令工作，但无法消除设备固定成本。FP8 输出字节更少仍比 Base 慢，说明 tiny 不能按总字节多少推断耗时。四线程协作与打包减少 shuffle 没有进一步收益，终盘保留所选 512 线程。

**FP8/h3072。** {record('FP8','h3072')}。采用直接编码、tile512/threads128 分块及深度 8 寄存器预取，未采用 hidden-first grid 和 v022 独立写回路径。距参考下限约 {gap('FP8','h3072')}，本组仍有明显模型差距。直接编码和分块已使 v000 时延接近减半，但较多 CTA、依赖等待或编码工作仍可能限制有效吞吐；当前数据不能认定具体哪一项主导，也不能承诺剩余差距全部可消除。

**FP8/h7168。** {record('FP8','h7168')}。优化同 FP8/h3072，加 v020 hidden-first grid；距参考下限约 {gap('FP8','h7168')}。这是最终相对 v000 加速最大的一组：旧通用转换成本被削减，分块、grid 与预取共同改善执行组织。它已达到 96.58% 的参考达成率；终盘不把不同版本单项加速比相乘来归因，也不声称每项贡献已被完全分离。

**FP8/prefill。** {record('FP8','prefill')}。采用 tile1024/threads128、hidden-first grid 与集中 8B 写回，不启用深度 8 预取；距参考下限约 {gap('FP8','prefill')}。v000 大规模运行本来就更接近吞吐受限，所以总加速 1.280×，不像 h7168 超过 2×。当前 99.52% 达成率支持本组合可以封盘，继续扩大写回宽度的历史尝试没有超越正式方案。

### 6.4 Quantized

**Quantized/tiny。** {record('Quantized','tiny')}。采用 v003 编码与 v029 local256，与 FP8/tiny 的 pair512 不同。单线程自己处理两个 expert，避免跨线程 shuffle；系数包含 x_sf。历史两线程候选没有稳定超过 local，因此不应仅因 FP8 采用协作而同步替换此路径。v000 相比节省约 {by['Quantized','tiny']['saved_vs_baseline_us']:.3f} μs，仍应以时延而非带宽达成率判断。

**Quantized/h3072。** {record('Quantized','h3072')}。优化同 FP8/h3072，额外启用 x_sf；距参考下限约 {gap('Quantized','h3072')}。这是大 workload 中参考达成率最低的一组，不能声称已到极限。x_sf 引入依赖于 position 的读取和系数乘法，可能增加等待，但相对 FP8 本轮额外约 1.167 μs 不能全归给一条乘法；若未来重开优化，这是较有数据依据的观察对象。

**Quantized/h7168。** {record('Quantized','h7168')}。优化同 FP8/h7168，额外启用 x_sf；距参考下限约 {gap('Quantized','h7168')}。总加速已超过 2×，但达成率 92.47% 仍低于对应 FP8。已有同轮证据支持 grid 与预取组合有效；没有最终可靠 stall 分解，所以不将余下差距归咎于 shared 缺失、private spill 或某个确定流水线问题。

**Quantized/prefill。** {record('Quantized','prefill')}。优化同 FP8/prefill，额外启用 x_sf；距参考下限约 {gap('Quantized','prefill')}。尽管两个缩放因子和 FP8 编码均启用，参考达成率仍达 98.82%。它是 prefill 四组中距参考下限稍远的一组，但剩余差距很小；微调收益需要新同轮测量证明，当前足以支持本轮工程封盘。

## 7. 终盘 profile、资源与正确性的证据边界

### 已有可使用的证据

- v003：直接编码避开通用 FP64 转换；v004：hidden 分块改变 CTA 几何；v020：grid 映射与真实 positions 地址统计；v022：prefill 机器 IR 的 2×4B→1×8B 写回；v025：寄存器预取的时延与资源对照。
- v025 查询的四个 SF h3072/h7168 组合：22 registers/thread，private=0，shared=0；原生资源允许 CTA/AP 上限仍为 16。它是容量查询，不是 GPU 时间线上实际驻留量。
- v030 对应 tiny 资源查询：Base/XSF/FP8 分别为 8/10/8 registers/thread，Quantized 保留 v029 的 6；private=0、shared=0。这是相关版本编译代码的记录，不是今天重测。
- 现有正确性记录：v025、v030 等已接入实验记录了 131 个现有测试通过；v030 正式接入四 tiny 的输出对照通过；v032 当前/候选在 random 与 padded 路由下逐字节一致。本次封盘未重新执行这些测试，不把既有验证说成新验证。

### 不用于终盘归因的结果

v020 的部分 mcProfiler 采集出现目标写入应约 3.67MB、计数却约 189MB 等异常，没有将它们用于计算最终带宽或等待归因。最终表的下限达成率来自 Q/runtime 模型；新增 HBM usage 来自各 case 的独立 native RoofLine 计数。两者都以1.50 TB/s为参考，但使用不同流量及计时口径，不能把模型值贴成profiler实测值。

曾发现 runtime 的 CYCLE_TRACE_MODE 开关，但没有确认可用的独立逐段 cycle trace 接口；没有据此生成“每一段耗时”或精准瓶颈占比。

## 8. 未采用方案与封盘依据

| 方向 | restart 中相关版本 | 保留结论 |
|---|---|---|
| shared 中转与同步 | v006/v012/v014/v015 | 当前 x 元素通常只供对应输出列消费一次；shared 不减少必需输入流量。部分任务分配实验有局部小收益，但未成为正式默认路径 |
| 动态局部索引 | v014 的初版比较 | private 与额外写入有代码/资源证据；改为直接 global position 后消除。失败不能泛化为所有 shared 方法都不行 |
| 更宽 FP8 输出 | v024 | 128-bit 写回候选未超过正式 v022，prefill 保留已证实的 8B 写回 |
| tiny 四线程协作 | v031 | 四组均未超过最终参考路径；K=2 缺少足够 expert 工作分摊，新增通信/分工代价不值得 |
| tiny 有效标志与输入打包 | v032 | 少一次 shuffle 但增加打包/解包，Base/XSF 变慢、FP8 基本持平，未接入 |

v016 的预取方向后来由 v025 在当前正式基础上组合验证并接入，不能简单把 v016 的 DEP 标签解释成预取完全无效。ACC 表示已接入，不代表该目录所有历史候选和所有 shape 都被采用。

这些实验支持“已经覆盖多种常见局部手段”的判断，不证明所有算法和布局空间都被穷举。针对某一组合无收益，也不排除改变上游数据组织、精度约束或融合边界后出现新收益。

## 9. tiny 的评价与封盘界限

[v028 同几何空 kernel](../../v028_DIS:v026/analysis/README.md)测到：32CTA×256 线程约 2.028 μs，32CTA×128 线程约 2.002 μs。统一以约 2 μs 作参考，当前 tiny 的空 kernel/完整 runtime 比值约 54%–60%。

它是 profiler 设备事件时间，不是 CPU launch 耗时，也不是可以严格扣除的固定成本；FP8 当前 512 线程没有对应空 kernel 同几何新测。不能说已精确证明完整 tiny 有 60% 是启动，其余 40% 全可优化。

tiny 封盘依据是：当前同规模真实时延、已接入候选的正确性与配对结果、后续四线程/打包未超越，以及现有编译器已经做的访存合并。仍可能存在小幅改善，但要取得较大端到端收益，融合相邻算子、减少独立调用或改变上游物化边界通常比继续追逐纯 Q/B 更有依据；本报告没有实施或验证这些新方向。

## 10. 后续边界与复现入口

本轮完成十六组工程封盘。若重开单 kernel 优化，优先看仍有模型差距的 Quantized/h3072、FP8/h3072，其次 Quantized/h7168；这只是依据表格选择观察对象，不是已确认可实现收益。若部署实例、缓存条件、SDK/编译器、shape、精度或上游融合方式变化，应重新建立基线。

正式入口与工作负载：

- [YAML manifest](../../../tileops/manifest/moe.yaml)：四变体和四种规模。
- [输入生成](../../../workloads/moe.py)。
- [正式 benchmark](../../../benchmarks/ops/bench_moe_reduce_fused.py)。
- [现有正确性测试](../../../tests/ops/test_moe_reduce_fused.py)。

仅重新计算 CSV 并生成本封盘文档，不调用 GPU：

```bash
cd /data/TileOPs-Metax
python profiles/FINAL_REDUCE_FUSED/scripts/build_closeout.py
```

需要以后重新测量时，使用项目环境及原 benchmark 协议，并保存为新实验，不覆盖本次冻结的数据。完整历史入口见 [profiles 索引](../../README.md)。
"""
# Arithmetic Intensity: same semantic FLOP/byte model as Op.eval_roofline().
ai_rows=[]
for x in rows:
    t,k,h=int(x['num_tokens']),int(x['num_topk']),int(x['hidden'])
    with_sf=x['variant'] in ('FP8','Quantized')
    with_xsf=x['variant'] in ('XSF','Quantized')
    reduce_flops=2*t*k*h
    xsf_flops=t*k if with_xsf else 0
    sf_flops=t*h if with_sf else 0
    flops=reduce_flops+xsf_flops+sf_flops
    input_bytes=2*t*k*h
    output_bytes=(1 if with_sf else 2)*t*h
    weights_bytes=4*t*k
    positions_bytes=4*t*k
    xsf_bytes=4*t*k if with_xsf else 0
    sf_bytes=4 if with_sf else 0
    byte_count=input_bytes+output_bytes+weights_bytes+positions_bytes+xsf_bytes+sf_bytes
    ai_rows.append(dict(variant=x['variant'],workload=x['workload'],
        num_tokens=t,num_topk=k,hidden=h,
        reduction_flops=reduce_flops,xsf_flops=xsf_flops,sf_flops=sf_flops,total_flops=flops,
        input_bytes=input_bytes,output_bytes=output_bytes,weights_bytes=weights_bytes,
        positions_bytes=positions_bytes,x_sf_bytes=xsf_bytes,sf_bytes=sf_bytes,
        total_bytes=byte_count,arithmetic_intensity_FLOP_per_byte=flops/byte_count))
with (root/'raw/arithmetic_intensity_16g.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(ai_rows[0]));writer.writeheader();writer.writerows(ai_rows)
ai_table=['| 变体 | Workload | FLOPs | 模型 Bytes | AI（FLOP/B） |',
          '|---|---|---:|---:|---:|']
for x in ai_rows:
    ai_table.append(f"| {x['variant']} | {x['workload']} | {x['total_flops']:,} | {x['total_bytes']:,} | **{x['arithmetic_intensity_FLOP_per_byte']:.6f}** |")
report += """
## 11. Arithmetic Intensity（AI，算术强度）推导

Arithmetic Intensity = FLOPs / Bytes，单位为 FLOP/B，表示每传输一个字节对应多少浮点运算。这里 AI 指算术强度。

### 11.1 统计口径

沿用正式 Op 的 `eval_roofline()` 语义模型，假设所有 K 个路由槽有效；输入读取和输出写入各按一次计算，metadata 按语义读取一次。这与第 4 节带宽下限的字节模型一致，不能当作实测 HBM 事务字节数。

FLOPs 统计归约和缩放：一次乘法计 1 FLOP，一次加法计 1 FLOP，一次 FMA 计 2 FLOPs。FP8 编码、索引、分支、转换等辅助指令未完整计入此语义 FLOP 模型；尤其整数位操作不作为 FLOPs。该口径适合比较算法算术强度，不表示已统计全部机器指令成本。

AI 不由 runtime 或 1.50 TB/s 决定；带宽参考用于计算 Q/B 下限，AI 则只由上述运算量和字节模型决定。

### 11.2 FLOPs 推导

```text
归约：T*H 个输出，每个输出累加 K 个加权贡献
      K 次乘加，每次 2 FLOPs
F_reduce = 2*T*K*H

x_sf：先计算 weights[token,k] * x_sf[pos]
      每个路由槽一次乘法，随后该系数用于所有 H 列
F_xsf = T*K（仅 XSF / Quantized）

sf：归约完成后，每个输出元素乘 sf[0]
F_sf = T*H（仅 FP8 / Quantized）

总 FLOPs = 2*T*K*H + I_xsf*T*K + I_sf*T*H
```

其中 I_xsf/I_sf 是是否启用该缩放的 0/1 标志。

### 11.3 Bytes 推导

```text
输入 x：        2*T*K*H
输出 out：      output_bytes_per_element*T*H
weights：       4*T*K
positions：     4*T*K
x_sf：          4*T*K（仅 XSF / Quantized）
sf：            4（仅 FP8 / Quantized）

总 Bytes = 2*T*K*H + output_bytes_per_element*T*H
         + 8*T*K + I_xsf*4*T*K + I_sf*4
```

本报告输入均为 FP16/BF16，每元素 2B；Base/XSF 输出为 2B，FP8/Quantized 输出为 1B。

### 11.4 十六组结果

"""+'\n'.join(ai_table)+"""

### 11.5 完整例子：Quantized/h3072

T=512、K=8、H=3072，启用 x_sf 和 sf，FP8 输出。

```text
FLOPs = 2*512*8*3072 + 512*8 + 512*3072
      = 25,165,824 + 4,096 + 1,572,864
      = 26,742,784

Bytes = 2*512*8*3072 + 1*512*3072
      + 8*512*8 + 4*512*8 + 4
      = 25,165,824 + 1,572,864 + 32,768 + 16,384 + 4
      = 26,787,844

AI = 26,742,784 / 26,787,844
   = 0.998318 FLOP/B
```

### 11.6 如何解读

- Base/XSF 的大 workload 约为 0.89 FLOP/B；FP8/Quantized 约为 1.00 FLOP/B。FP8 输出更少，同时有最终 sf 乘法，因此算法算术强度稍高。
- x_sf 每个路由槽只增加一次系数乘法，但增加 4B 的缩放读取。因此同规模 XSF/Quantized 的 AI 略低于对应 Base/FP8。
- T 同比扩大时，绝大部分 FLOPs 和 Bytes 同比增加，所以 h7168 与 prefill 的 AI 几乎相同，实际运行效率却可能不同。
- 所有组的语义 AI 都不超过 1 FLOP/B，支持大规模路径以访存为主要观察方向；但 FP8 软件编码成本不由这个数充分表达，tiny 也不能仅据 AI 判断为已充分达到带宽上限。

[完整 FLOPs、字节分项与 AI CSV](../raw/arithmetic_intensity_16g.csv)。
"""

(root/'analysis/README.md').write_text(report)
print('已生成封盘报告，16组性能、16组配置与16组逐项分析已保存')
