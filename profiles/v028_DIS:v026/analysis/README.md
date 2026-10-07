# v028：tiny 同几何空 kernel 开销测量

## 范围与方法

sc-16g、当前25% compute切片、16000MiB配额。tiny=T32/K2/H256、FP16；正式Base/XSF使用32CTA×256线程，FP8/Quantized使用32CTA×128线程。

原生MACA定义实际发射的 `__global__ tiny_empty`，主体为空；保留六个pointer参数槽和两个int维度，参数不读取。通过编译后的共享库发射，不把空Python调用或被省略的TileLang调用当空kernel。所有原生launch及同步成功，项目profiler识别到实际设备kernel事件。

同一进程按线程配置分别配对空kernel与正式tiny，五轮交替先后顺序。项目bench_kernel：10warmup、50repeat×3trials、CUPTI、L2flush。表中统计为五轮的“trial平均时间中位数”的中位数，不是单次调用P50。完整tiny使用相同随机输入、预分配输出；没有修改正式kernel。

## Profiler device kernel 时间

| 对照 | CTA数 | 线程/CTA | 中位时间 μs | 各轮范围 μs |
|---|---:|---:|---:|---:|
| empty_256 | 32 | 256 | 2.028 | 2.007～2.043 |
| Base | 32 | 256 | 3.615 | 3.589～3.625 |
| XSF | 32 | 256 | 3.579 | 3.569～3.599 |
| empty_128 | 32 | 128 | 2.002 | 1.992～2.007 |
| FP8 | 32 | 128 | 3.886 | 3.871～3.907 |
| Quantized | 32 | 128 | 3.732 | 3.722～3.779 |

## 无profiler的原生event批量计时

每配置100次warmup；每批128/1024/8192次launch，各五轮。event括住整个原生C++提交循环；CPU计时包含launch与mcGetLastError检查，可能受队列回压影响。wall计时到event完成，包括等待。下表均为每批时间除以次数，再取五轮中位数。

| 线程/CTA | 每批次数 | event平均 μs/次 | CPU提交循环 μs/次 | wall完成平均 μs/次 |
|---|---:|---:|---:|---:|
| 128 | 128 | 6.130 | 5.865 | 6.774 |
| 128 | 1024 | 5.992 | 6.160 | 6.614 |
| 128 | 8192 | 7.113 | 7.213 | 8.541 |
| 256 | 128 | 6.424 | 6.146 | 6.999 |
| 256 | 1024 | 5.766 | 5.922 | 6.380 |
| 256 | 8192 | 7.204 | 7.299 | 8.635 |

## 解释

- 当前条件下，32CTA的空kernel profiler device时间约 **2.0 μs**；128/256线程差异很小。
- 同轮完整tiny约 **3.6～3.9 μs**。这比0.03 μs的带宽理想时间更有参考意义：即使没有有效访存和计算，实际设备kernel事件仍需要约2 μs。
- 2 μs包含空kernel在该启动几何及profiler条件下的设备执行成本，不能解释成CPU launch时间，不能证明任何32CTA实现都不可能更快。
- event批量平均约5～7 μs/次，与CPU提交循环平均同量级；因此该口径包含CPU提交空隙和运行时调度影响，不能用它替代2 μs的内核事件时长。增大批量也没有自动消除这种影响。
- 完整kernel减去空kernel的差值可辅助定位工作成本，但不是可严格剥离的固定开销，不能据此把其余部分全部当作可优化空间。
- 缓存、时钟、切片调度和采集工具可能影响微秒级结果。正式优化仍使用同轮、相同计时方法对照，最终独立复测。

[内核时间CSV](../raw/profiler_comparison_16g.csv)、[原生event原始CSV](../raw/native_events_16g.csv)、[批量统计](native_event_summary_16g.json)、[空kernel源码](../scripts/empty_probe.cpp)、[完整运行日志](../logs/run.log)。Profiler每轮样本在 `raw/profiler_samples_threads*.json`，环境与源码哈希在 `meta/`。

## 复现

```bash
cd /data/TileOPs-Metax
source /data/sc16g-recovery-20261004/env.sh
python profiles/v028_DIS:v026/scripts/run.py
```

重跑覆盖本目录的测量结果。
