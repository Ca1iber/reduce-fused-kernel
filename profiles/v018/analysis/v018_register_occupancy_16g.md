# v018：寄存器预取与 AP 驻留容量验证（16g）

## 结论

当前实例原生runtime报告：**每AP有131,072个32位寄存器=512KiB，最大2048线程、16CTA、64KiB shared；warp64线程，共104AP。**
FP8 h7168当前生产配置tile512/128线程：12寄存器/线程，编译使用数换算1536寄存器/CTA=6KiB；v016深度8预取为22/线程、2816/CTA=11KiB。两者资源容量都允许16CTA/AP；长驻留探针均测到全卡1664存活CTA，与104×16一致。
256线程、tile512的FP8对照：8→12寄存器/线程，2048→3072/CTA；两者都是8CTA/AP，探针832=104×8。
**不能推广成寄存器预取永不降低驻留**：Base/XSF整行tile7168/256线程从52→130寄存器/线程，13312→33280寄存器/CTA，原生API上限8→3。Base长驻留探针也从832降到312。
另实现v015的沿hidden双fragment候选：39寄存器/线程、4992/CTA、19.5KiB，shared0、容量16CTA/AP；shared双缓冲版12寄存器/线程但shared16KiB，容量4CTA/AP。

## 1. 上版本遗留问题

v016记录了12→22寄存器/线程和private0，但没有计算每CTA/AP资源，也没有实验区分“寄存器变多”与“驻留下降”。v015沿hidden的双fragment未测试，此次为回答该替代方案的资源压力，单独实现group2双fragment。
本轮资源核验以FP8 T512,K8,H7168,BF16为主，包含128/256线程、tile512/1024、深度2/4/8、Base/XSF整行及Quantized。没有更改生产kernel。

## 2. 当前硬件与计数口径

[原生设备属性](../raw/device_properties.json)由SDK C++ `mcGetDeviceProperties`读取：

| 属性 | 当前值 |
|---|---:|
| AP数 | 104 |
| warp线程数 | 64 |
| 每AP 32位寄存器数 | 131072 |
| 每AP寄存器容量 | 524288 bytes = 512 KiB |
| 每AP最大线程数 | 2048 |
| 每AP最大CTA数 | 16 |
| 每AP shared | 65536 bytes |
| runtime compute major/minor | 10/0 |

SDK `mc_runtime_types.h` 的 `regsPerMultiprocessor` 注释明确单位是32位寄存器。各shader的 `MC_FUNC_ATTRIBUTE_NUM_REGS` 与mcTracer `registers_per_thread`逐项一致。
**表中的每CTA使用数=编译计数×线程数，不是已获得硬件对齐后预留量的精确读数。** 硬件可能按粒度/分区分配。资源上限以原生 `mcModuleOccupancyMaxActiveBlocksPerMultiprocessor` 交叉核验，不以简单除法取代它。
SDK通用 `mc_occupancy.h` 计算器对当前native major10返回status2，不适配此架构；其0输出没有用于结论，也没有把major改成CUDA兼容值来冒充C500计算。
机器、时间、源码hash、git及mx-smi查询保存在meta。16G/compute配额没有被机械乘到AP寄存器、线程或CTA上限。

## 3. 计算过程

资源容量的手算模型（未显式建模分配粒度/子分区）：

```text
CTA/AP ≤ min(
    硬件最大CTA数,
    floor(每AP最大线程数 / CTA线程数),
    floor(每AP寄存器数 / (寄存器每线程 × CTA线程数)),
    floor(每APshared容量 / CTA的shared用量)
)
```

shared为0时忽略最后一项。本轮16个配置手算结果均与原生module API相符；这不能证明所有配置都无需考虑分配粒度。

FP8 tile512、128线程：

```text
基线：12×128 = 1536寄存器 = 6144 bytes
预取：22×128 = 2816寄存器 = 11264 bytes
增加：1280寄存器 = 5120 bytes = 5KiB/CTA

仅寄存器允许的CTA：floor(131072/1536)=85 → floor(131072/2816)=46
线程数允许的CTA：2048/128=16
硬件CTA上限：16
最终：min(85,16,16)=16 → min(46,16,16)=16
```

Base整行、256线程：

```text
基线：52×256=13312寄存器=52KiB/CTA
预取：130×256=33280寄存器=130KiB/CTA
寄存器上限：floor(131072/13312)=9 → floor(131072/33280)=3
线程上限：2048/256=8
最终：8 → 3 CTA/AP
```

hidden双fragment：39×128=4992寄存器=19.5KiB，寄存器上限26，大于线程/CTA上限16。
hidden双shared：65536/16384=4CTA/AP，shared是限制项。
完整计算在[occupancy_calculations.json](occupancy_calculations.json)。

## 4. 实际编译资源与原生查询

| kernel配置 | 线程/CTA | 寄存器/线程 | 使用寄存器/CTA（32位） | 使用容量/CTA | shared/CTA | 可驻留CTA/AP |
| --- | --- | --- | --- | --- | --- | --- |
| fp8_b512_t128 | 128 | 12 | 1536 | 6 KiB | 0 KiB | 16 |
| fp8_reg2_b512_t128 | 128 | 15 | 1920 | 7.5 KiB | 0 KiB | 16 |
| fp8_reg4_b512_t128 | 128 | 19 | 2432 | 9.5 KiB | 0 KiB | 16 |
| fp8_reg8_b512_t128 | 128 | 22 | 2816 | 11 KiB | 0 KiB | 16 |
| fp8_b512_t256 | 256 | 8 | 2048 | 8 KiB | 0 KiB | 8 |
| fp8_reg8_b512_t256 | 256 | 12 | 3072 | 12 KiB | 0 KiB | 8 |
| fp8_b1024_t128 | 128 | 20 | 2560 | 10 KiB | 0 KiB | 16 |
| fp8_reg8_b1024_t128 | 128 | 42 | 5376 | 21 KiB | 0 KiB | 16 |
| base_full_t256 | 256 | 52 | 13312 | 52 KiB | 0 KiB | 8 |
| base_reg8_full_t256 | 256 | 130 | 33280 | 130 KiB | 0 KiB | 3 |
| xsf_full_t256 | 256 | 52 | 13312 | 52 KiB | 0 KiB | 8 |
| xsf_reg8_full_t256 | 256 | 130 | 33280 | 130 KiB | 0 KiB | 3 |
| quant_b512_t128 | 128 | 12 | 1536 | 6 KiB | 0 KiB | 16 |
| quant_reg8_b512_t128 | 128 | 22 | 2816 | 11 KiB | 0 KiB | 16 |
| hidden_double_fragment | 128 | 39 | 4992 | 19.5 KiB | 0 KiB | 16 |
| hidden_double_shared | 128 | 12 | 1536 | 6 KiB | 16 KiB | 4 |

全部16个原始kernel在mcTracer和module API中 private=0，候选输出与同几何直接读参考逐字节一致（10warmup后一次、ROI后一次）。这只覆盖此处BF16/H7168/workload，不声称新doublefragment已完成全部131用例或适合生产。
[trace原始文件](../raw/trace_resources)、[资源查询原始JSON](../raw/kernel_resources.json)、[CSV](../raw/occupancy_summary_16g.csv)均保存。
trace每配置先有一次baseline校验，随后10warmup+1ROI；192条目标事件按16×12核对，统计取每组后11条，没有把输入生成kernel资源混入。

## 5. 驻留实验验证

只调用occupancy API属于资源预测，因此又用真实生成代码构造**有限时长的长驻留探针**：

1. 每CTA线程0进入时atomic加1，全卡atomicMax记录峰值。
2. 全CTA同步，原数据保持/读取后用clock64等待有限cycles，让CTA生命周期足够长。
3. 原计算和输出完成、全CTA同步后，线程0减1；结束计数必须回到0。
4. 1,000,000和5,000,000 cycles分别3次，48次launch；没有让CTA彼此等待、没有跨CTA全局死锁屏障。
5. 各探针都重新查询其自己的寄存器和occupancy属性，并检查输出前16个元素。原版完整输出由Python采集脚本另做逐字节校验。

| 原始配置 | 探针寄存器/线程 | 探针API上限 | 全卡峰值CTA（6次） | 峰值/104 AP |
| --- | --- | --- | --- | --- |
| fp8_b512_t128 | 12 | 16 | 1664,1664,1664,1664,1664,1664 | 16.0 |
| fp8_reg8_b512_t128 | 22 | 16 | 1664,1664,1664,1664,1664,1664 | 16.0 |
| fp8_b512_t256 | 8 | 8 | 832,832,832,832,832,832 | 8.0 |
| fp8_reg8_b512_t256 | 12 | 8 | 832,832,832,832,832,832 | 8.0 |
| base_full_t256 | 52 | 8 | 832,832,832,832,832,832 | 8.0 |
| base_reg8_full_t256 | 142 | 3 | 312,312,312,312,312,312 | 3.0 |
| hidden_double_fragment | 39 | 16 | 1664,1662,1663,1664,1664,1662 | 16.0 |
| hidden_double_shared | 12 | 4 | 416,416,416,416,416,416 | 4.0 |

所有run live_after=0、wrong_sample=0；private_bytes=0。
Base整行预取的仪器化使寄存器/线程由130→142，原版和探针API上限都为3；因此3CTA结果验证这一资源档位，**不是无改动原kernel的瞬时测量**。其余探针寄存器计数与对应原版一致。
hidden doublefragment有少量1662/1663样本，最大1664，与容量上限一致，全部样本保留。

**探针测的是全卡逻辑存活CTA峰值；除以104得到全卡平均的峰值对应量，没有读取每个AP的逐时刻ID/轨迹。** 结果与原生容量API/硬件上限一致，支持16/8/3/4这些驻留容量。它不能证明原短kernel全程每AP都保持这些数量，也不消除sGPU调度/抢占的影响。
当T512、Base整行只有512CTA时，即使所有CTA同时存活，全卡平均也不超过512/104≈4.92CTA/AP；资源容量8并不意味着这个workload能让每AP都驻留8。Base探针用T4096保证grid足够大，并明确记录这个改变。

## 6. 具体落地与复现

工作目录 `/data/TileOPs-Metax`，环境 `source /data/sc16g-recovery-20261004/env.sh`。

- [collect_cases.py](../scripts/collect_cases.py)：当前原kernel/预取/双缓冲的完整输出校验和11次trace。
- [resource_query.cpp](../scripts/resource_query.cpp)：原生属性、原kernel NUM_REGS与module occupancy。
- [query_all.py](../scripts/query_all.py)：按相同O3/lineinfo/xcore1000参数构建mcbin并查询16配置；命令原样在raw JSON。
- [hidden_fragment.py](../scripts/hidden_fragment.py)：v015 group2，stage_x改为fragment，mode3使用普通global→fragment load，无shared/ticket等待/CTA同步；源方案另保留mode2对照。
- [residency_probe.cpp](../scripts/residency_probe.cpp)、[run_residency.py](../scripts/run_residency.py)：有限长驻留计数。
- codegen保留原始和hold版源码/二进制；[hold_specs.json](../meta/hold_specs.json)记录实际参数顺序、grid、block及数据规模。

```bash
python profiles/v018/scripts/run_collect.py
python profiles/v018/scripts/query_all.py
python profiles/v018/scripts/run_residency.py
```

实际GPU阶段按顺序执行。query_all中的模块加载没有执行GPU计算。没有下载、更新、修改SDK，也未更改sGPU调度、功耗或AP设置。

## 7. 实验总结

- 小tile的v016预取增加寄存器使用，却没有降低16CTA/AP（128线程）或8CTA/AP（256线程）的容量；本轮长驻留计数与API吻合。
- 若在整行tile7168上做深度8预取，寄存器会成为限制：Base/XSF从8降到3；不能只看prefetch_depth而不看tile和threads。
- 把v015双缓冲放在fragment中，本次实际编译39寄存器/线程，容量16；shared双缓冲容量4，资源权衡已经量化。下文补充了FP8 h7168的正式配对benchmark；其它变体/workload尚未测性能。
- 寄存器使用量、资源允许的最大驻留量、实际短kernel逐时刻驻留量是三个量。本文分别列出编译/查询/探针证据，没有将mcTracer的百分比或累计wave数当成achieved occupancy。

## 补充：双fragment性能尝试（FP8 h7168）

T512,K8,H7168,BF16→E4M3FN；相同输入逐字节一致。项目bench_kernel：10warmup、50repeat×3trials、CUPTI、L2flush；五个方案交替五外层轮次，取中位数。此次input_cloned=true，全部原始样本保留。

| 方案 | 时间µs | 对v010加速比 | 延迟降低 |
|---|---:|---:|---:|
| v010 | 48.266 | 1.0000× | +0.00% |
| v016_register8 | 44.662 | 1.0807× | +7.47% |
| group2_direct | 50.043 | 0.9645× | -3.68% |
| v015_shared_double | 63.708 | 0.7576× | -31.99% |
| v018_fragment_double | 46.100 | 1.0470× | +4.49% |

双fragment比v010加速4.70%，延迟降低4.49%；比同几何group2_direct加速约8.55%。它明显好于shared双缓冲，但仍慢于v016的K方向深度8寄存器预取。这是单一workload的性能结论，不能推到全部变体/shape；没有接入生产dispatch。

[原始五轮数据](../raw/pilot_benchmark_fp8_h7168_16g.csv)、[脚本](../scripts/benchmark_hidden_fragments.py)。复现：`python profiles/v018/scripts/benchmark_hidden_fragments.py --workload h7168 --variant fp8`。

## 后续完整覆盖

现已补齐四变体×四workloads，共16组、80行五方案对照，每方案五外层轮次；新双fragment通过现有131用例。完整结果及范围说明见[16组完整benchmark报告](v018_full_benchmark_16g.md)。前文单例pilot作为当时记录保留，不用它代表全量结论。
