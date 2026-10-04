# v015：hidden 双缓冲与分组流水线（16g）

## 1. 上版本遗留问题

v012 的 K-ring 每次只与当前 expert 的少量 FMA 重叠，窗口短。v013 删 CTA 同步或换 warp 同步均错误，不能采用其速度。检验能否用整个 hidden 块的归约和 FP8 编码隐藏下一块读取。

## 2. 问题原因分析

一 CTA 处理整行可复用 metadata，但减少 CTA 数。以512列为块，v010 是512×14=7168 CTA；整行stream只有512 CTA，硬件报告104 multiprocessor。直接stream控制已经84.762µs，比48.236µs基线慢，说明不能只归因于 async/shared。
shared双缓冲16KiB，在shared64KiB/MP的容量模型下最多4 CTA；128线程/CTA对应最多512线程，而 runtime MP上限2048。这是容量上界推算，**不是测得25% occupancy，也不是容器compute quota**。实际还受分配粒度及其它资源影响。

## 3. 本版本解决方案

两槽shared `[2,K,B]`；wait当前块全部K ticket、CTA同步后，先发起下一hidden块全部K读取，接着当前块归约和编码。保持 K 次序、路由有效性和原编码，不删除必要同步。
控制：直接读取mode0、同步doublebuffer mode1、真实nativeasync mode2。随后每CTA只负责2/4块，恢复grid并隔离并行度因素。

## 4. 具体落地策略

[hidden_kernel.py](../scripts/hidden_kernel.py)::get_hidden_kernel 与 [grouped_hidden_kernel.py](../scripts/grouped_hidden_kernel.py)，tile512/128 threads；ticket `[2,K]` 用 uint64 IR+b64vectype backend注解。
初版缺少预取 pos 上界的 `T.assume`，safe memory pass 在 async rewrite 报错；补上与原kernel相同的合法输入契约后通过。失败源 [hidden_missing_assume.py](../codegen/hidden_missing_assume.py) 保留。
已测 FP8 h7168 全部候选与基线逐字节一致，profile脚本再复核；**没有为此失败候选声称跑完整131用例**。

## 5. Benchmark 对比

机器 sc-16g / C500，容器 96933d7d09ab，16G、25% compute quota；MACA 3.7.1.5。
工作目录 `/data/TileOPs-Metax`，环境 `source /data/sc16g-recovery-20261004/env.sh`。
当前正式源为 v010，SHA256 `c6b5dcb13fa171ace058b3f0a50d8361a439eb3335ad4596c539273859831596`。
本轮候选独立保存在 profiles 中。h3072=(T512,K8,H3072)、h7168=(T512,K8,H7168)、prefill=(T4096,K8,H7168)，输入均 BF16。
Base/XSF 输出 BF16；FP8/Quantized 输出 E4M3FN。只改变加载/调度，保留路由判断、K 累加顺序和 v003 编码。
正式计时用项目 `bench_kernel`：10 warmup、50 repeat×3 trials、L2 flush、CUPTI；再交替方案顺序取外层中位数。
trace 是同应用内 10 warmup+1 ROI 的资源/时间辅助证据，不与正式计时混算。
`Compute Instructions` 包含控制等指令，不等于 FLOPs；`Total Cycles` 和 stall 是工具聚合值，不能直接当 wall time。
`Achieved waves`/`Dispatched waves` 是总量，不能当驻留 occupancy。prefill 的 inputs_cloned=false 是项目内存阈值策略。


整行stream：

| 方案 | 正式 µs | 相对直接基线 | 延迟变化 µs |
| --- | --- | --- | --- |
| baseline | 48.236 | 1.0000× | +0.000 |
| stream_direct | 84.762 | 0.5691× | +36.526 |
| stream_sync | 158.208 | 0.3049× | +109.972 |
| stream_async | 84.265 | 0.5724× | +36.029 |

分组控制：

| 方案 | 正式 µs | 相对直接基线 | 延迟变化 µs |
| --- | --- | --- | --- |
| baseline | 47.903 | 1.0000× | +0.000 |
| group2_direct | 49.905 | 0.9599× | +2.002 |
| group2_async | 63.601 | 0.7532× | +15.698 |
| group4_direct | 48.727 | 0.9831× | +0.824 |
| group4_async | 61.722 | 0.7761× | +13.819 |

与同步双缓冲158.208相比，真实async84.265µs有明显改善，但仍未击败正式基线。
恢复grid之后 group2/4直接读49.905/48.727µs已接近基线47.903，而async仍63.601/61.722µs，证明已测几何的额外shared/ticket/同步成本没有被隐藏。
各组来自各自配对run，不跨run拼一个“精确加速比”；三外层样本在CSV保留。
命令：`python profiles/v015/scripts/initial_compare.py`，`python profiles/v015/scripts/compare_grouped.py`。

## 6. Profile 指标变化

| 方案 | 同次 trace µs | 寄存器/线程 | shared bytes | private_total | grid | threads |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 50.688 | 12 | 0 | 0 | 512×14 | 128 |
| stream_direct | 84.736 | 14 | 0 | 0 | 512×1 | 128 |
| stream_sync | 158.464 | 14 | 16384 | 0 | 512×1 | 128 |
| stream_async | 85.760 | 14 | 16384 | 0 | 512×1 | 128 |
| group2_direct | 52.480 | 12 | 0 | 0 | 512×7 | 128 |
| group2_async | 63.744 | 12 | 16384 | 0 | 512×7 | 128 |
| group4_async | 62.208 | 14 | 16384 | 0 | 512×4 | 128 |

| 方案 | 总指令 | 内存指令 | 读 bytes | 写 bytes | VLS pipeline stall | WSM stall | shared load/store | efficiency/conflict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stream_async | 4988329 | 245936 | 58842528 | 3670336 | 469645 | 56784 | 113344/0 | 100.0%/0.0 |
| group2_async | 5332038 | 276900 | 58871744 | 3670368 | 562517 | 56720 | 113600/0 | 100.0%/0.0 |

原始独立目标JSON在 `raw/native_*/tool_output/1_reduce_fused_kernel_kernel.txt.json`，有效流量均接近输入58.72MB/输出3.67MB。global→shared专用copy不走普通shared store计数，因此store=0不等于shared未写。
资源 private_total=0，shared访问efficiency100%、平均conflict0：未见spill/bank conflict证据。较高shared容量、CTA数量变化、额外shared读取和控制才是该已测方案的代价。
具体驻留率和等待隐藏比例没有直接硬件measurement，不由模型或stall总数伪造；资源和配对控制足以限定这里的失败原因，而不排除其它producer/consumer设计。
profile重跑：`python profiles/v015/scripts/collect_profiles.py`。采用relative mcTracer odname；JSON存在及目标事件数量已核实，不仅依赖exit0。

## 7. 实验总结

这套整行/2块/4块hidden双缓冲不采用：正确但比正式基线慢。async相对同结构sync有效，证明copy路径确实可用；失败不能外推所有pipeline不可行。
v014 warp协作和v016寄存器预取已经给出反例。这里记录失败范围、控制实验和硬件资源约束，不宣称无论如何不可能。
