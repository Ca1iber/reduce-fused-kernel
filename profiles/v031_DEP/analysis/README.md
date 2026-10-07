# v031：tiny 四线程列协作实验

## 分工与范围

仅测T32/K2/H256、FP16的四个tiny。每组四个线程：role0读取expert0并负责最终累加/输出，role1读取expert1，role2准备expert0的weight×x_sf，role3准备expert1的weight×x_sf；没有x_sf时只读取weight。通过warp内shuffle交换数据及系数，保持原K顺序和FP8编码。

固定32CTA，扫描256/512/1024线程：每组分别处理4/2/1列。比较相邻四lane与四个16-lane区间两种分配，共24配置。四线程均有任务，没有把多出来的线程当空闲填充。

按要求只在计时前核对一次输出，不运行大测试集、其他shape或额外资源校验。24配置输出均与正式v029逐字节一致。

## 测量

同进程与正式v029、v030双线程方案对比；固定seed1235、相同输入和预分配输出。项目bench_kernel，CUPTI、L2flush、输入克隆、10warmup、50repeat×3trials，外层5轮轮换/反转顺序，取中位数。

v030对照：Base/XSF使用相邻配对256线程，FP8/Quantized使用相邻配对512线程。Quantized的v030对照只是候选，不是正式采用版本。

## 各变体相对v030表现最好的四线程配置

| 变体 | 布局/线程 | v029 μs | v030 μs | 四线程 μs | 比v030变慢 |
|---|---|---:|---:|---:|---:|
| Base | adjacent/256 | 3.651 | 3.354 | 3.574 | 6.56% |
| XSF | adjacent/256 | 3.558 | 3.389 | 3.907 | 15.26% |
| FP8 | adjacent/256 | 3.681 | 3.615 | 4.024 | 11.33% |
| Quantized | adjacent/256 | 3.676 | 3.651 | 4.434 | 21.46% |

## 所有四线程配置耗时

| 布局 | threads | 每组列数 | Base μs | XSF μs | FP8 μs | Quantized μs |
|---|---:|---:|---:|---:|---:|---:|
| adjacent | 256 | 4 | 3.574 | 3.907 | 4.024 | 4.434 |
| adjacent | 512 | 2 | 3.584 | 3.999 | 4.019 | 4.460 |
| adjacent | 1024 | 1 | 3.886 | 4.357 | 4.096 | 4.521 |
| striped16 | 256 | 4 | 3.579 | 3.968 | 4.229 | 4.654 |
| striped16 | 512 | 2 | 3.907 | 4.332 | 4.265 | 4.700 |
| striped16 | 1024 | 1 | 3.907 | 4.352 | 4.070 | 4.531 |

## 结论

本轮四线程分工没有超过双线程方案，四个变体相对v030的最佳配置仍分别慢约6.6%、15.3%、11.3%、21.5%，不接入正式kernel，保留为v031_DEP。

K=2时，双线程已经分别覆盖两个expert。增加两线程只能继续拆分数据与系数准备；本实现增加了角色分支和shuffle，输出仍由一个线程完成。相同warp中不同角色的指令路径可能串行执行，不能把逻辑分工等同于四倍独立并行。这与性能退化一致；本轮未做硬件计数归因，不宣称排除了所有四线程实现。

[原始CSV](../raw/comparison_16g.csv)、[TileLang实现](../scripts/tiny_quad.py)、[运行脚本](../scripts/benchmark.py)、[运行日志](../logs/benchmark.log)。各配置五轮原始数据在 `raw/*_samples.json`，生成代码在 `codegen/`。

```bash
cd /data/TileOPs-Metax
source /data/sc16g-recovery-20261004/env.sh
python profiles/v031_DEP/scripts/benchmark.py
```

重跑覆盖数据；脚本用当前正式kernel作v029对照，复测应记录当时正式源码。当前正式源码与tiny helper快照保存在codegen/，历史绝对时间不混用。
