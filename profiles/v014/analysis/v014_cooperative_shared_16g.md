# v014：warp 协作 shared gather（16g）

## 1. 上版本遗留问题

v006 单 buffer 每 K 两次 CTA 同步没有收益；v012 的真实 async K-ring 也慢于 v010。两者都沿用全 CTA 顺序读取每个 expert，没有检验改变生产者任务分配。

## 2. 问题原因分析

当前直接读取每个 x 元素供对应输出列使用一次，shared 缓存不减少算法必需的输入读取。shared 的价值应检验访存任务分配和数据布局，而不能只期待缓存复用。
初版按 warp 动态索引 `topk_to_pos_local[expert]`，设备 trace 报 private_total=36，global write 26,011,200 bytes，远多于输出 3,670,016 bytes；同一几何改为直接读 global position 后 private_total=0、write=3,670,464。动态局部数组访问引入 private 存储是有代码和资源支持的结论，不能把该失败归咎于 shared 本身。

## 3. 本版本解决方案

物理 warp=64；让不同 warp 读取不同 expert。每 lane 向量读 8 个 BF16，512 列块的 global 读取连续。shared 保存原 BF16，不先算乘积；全部 producer 完成后一次 CTA 同步，consumer 按原 K 顺序累加。

## 4. 具体落地策略

- 初版：[cooperative_kernel.py](../scripts/cooperative_kernel.py)；修正版：[cooperative_global_positions.py](../scripts/cooperative_global_positions.py)::get_cooperative_kernel。
- 扫描 tile512/1024、threads128/256/512，并保留全 CTA 逐 K staging 控制。
- 最佳已测配置 tile1024/512 threads：8 warp 对应 K8；shared=K×B×2=16KiB。理论 shared64KiB 允许 4 CTA，4×512=2048 threads，与 runtime thread 上限相等。这只是资源容量模型，不是 measured occupancy。
- 全部 12 组逐字节对比；最佳 factory 跑现有 131 用例，全部通过。factory 只对 BF16、H≥1024 且 H 整除1024使用此方案，其余走原函数；**不声称 131 用例全部都跑了 shared 路径**。见 [validate_best.py](../scripts/validate_best.py)、[JUnit](../raw/correctness_best.xml)。

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


下面为独立确认，5 个外层顺序轮次；包含当前最佳生产配置和相同 tile1024/512 的 direct 控制。

| 变体 | workload | 正式基线 µs | 同几何直接读 µs | 候选 µs | 加速比 | 减少 µs | 延迟降低 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | h3072 | 20.844 | 29.240 | 22.564 | 0.9238× | -1.720 | -8.25% |
| xsf | h3072 | 20.741 | 29.399 | 27.090 | 0.7656× | -6.349 | -30.61% |
| fp8 | h3072 | 22.789 | 29.763 | 22.840 | 0.9978× | -0.051 | -0.22% |
| quantized | h3072 | 23.020 | 30.198 | 27.581 | 0.8346× | -4.562 | -19.82% |
| base | h7168 | 44.329 | 63.334 | 49.582 | 0.8941× | -5.253 | -11.85% |
| xsf | h7168 | 44.339 | 63.816 | 54.564 | 0.8126× | -10.225 | -23.06% |
| fp8 | h7168 | 48.338 | 63.181 | 47.324 | 1.0214× | +1.014 | +2.10% |
| quantized | h7168 | 48.481 | 64.108 | 54.620 | 0.8876× | -6.139 | -12.66% |
| base | prefill | 351.887 | 498.463 | 379.034 | 0.9284× | -27.146 | -7.71% |
| xsf | prefill | 353.971 | 504.346 | 443.756 | 0.7977× | -89.784 | -25.36% |
| fp8 | prefill | 356.541 | 493.665 | 359.132 | 0.9928× | -2.591 | -0.73% |
| quantized | prefill | 359.475 | 503.772 | 449.987 | 0.7989× | -90.511 | -25.18% |

[完整样本](../raw/confirmation_16g.csv) 保留异常值，包括 current 个别 >100µs 样本，没有删除后重算。
初步 expanded 曾出现假大收益，确认以五轮数据为准。FP8 h7168 的候选五个样本均约47.22–47.52µs，而基线中位48.338µs；约2.14%速度提升、2.10%延迟降低。其它11组不推广。
重跑命令：`python profiles/v014/scripts/run_confirmation.py`；重跑前应另存原 CSV，脚本追加结果。

## 6. Profile 指标变化

| 方案 | 同次 trace µs | 寄存器/线程 | shared bytes | private_total | grid | threads |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 50.176 | 12 | 0 | 0 | 512×14 | 128 |
| direct1024t512 | 63.232 | 8 | 0 | 0 | 512×7 | 512 |
| warp512_local | 158.208 | 12 | 8192 | 36 | 512×14 | 256 |
| warp512_global | 51.968 | 10 | 8192 | 0 | 512×14 | 256 |
| warp1024_global | 47.872 | 14 | 16384 | 0 | 512×7 | 512 |

| 方案 | 总指令 | 内存指令 | 读 bytes | 写 bytes | VLS pipeline stall | WSM stall | shared load/store |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3962542 | 241450 | 58835584 | 3670464 | 192151 | 0 | 0/0 |
| direct1024t512 | 5252526 | 482736 | 58870464 | 3670464 | 383441 | 0 | 0/0 |
| warp512_local | 7155722 | 880276 | 62072832 | 26011200 | 2851288 | 404303 | 227168/56792 |
| warp512_global | 6019728 | 568080 | 58763904 | 3670464 | 408433 | 459520 | 227232/56808 |
| warp1024_global | 5565452 | 539640 | 58859072 | 3670464 | 806330 | 479752 | 227136/56800 |

目标计数来自 `raw/native_*/tool_output/1_period0.txt.json`；**report.txt.json 是全应用 aggregate，不能用来评价目标 kernel**。精确来源及字段见 [profile_comparison.json](profile_comparison.json)。
所有 shared 版本 reported access efficiency100%、平均 conflict cycles0；没有 bank conflict 导致本次变慢的证据。
修正局部索引后，private 存储、额外 global 写出和 VLS stall 都下降，构成强对照证据。
最佳 shared 方案仍比正式基线执行更多指令和聚合 stall，但端到端更快；总量不能替代关键路径、并发或实际时长。当前证据支持任务重排有效，不足以量化每个硬件机制贡献。
mcTracer 原始 JSON 在 `raw/trace_comparison`；生成 C++ 在 `codegen/profile_*.cu`。

## 7. 实验总结

shared 方向可行：至少 FP8 h7168 已出现小而可复现的收益。收益来自这套协作读取/布局/几何整体，不能说单独加 shared 就会加速。
Base、XSF、Quantized 和 prefill 在已测配置中没有收益；动态局部数组、线程配置和额外搬运比 bank conflict 更值得优先检查。
v016 寄存器预取在同一 FP8 workload 获得更大收益，因此 shared 候选作为可行性证据保存，不取代更快方案。


## 补充说明

按照分析来说，输入 x 中的某个元素 x[pos, hidden] 只会一个 CTA 读取一次，并不会被复用，因此按理来说没有必要硬上 smem 再读取，这样反而会增加额外开销
但如果 smem 能支持更好的分工，例如让不同 warp 并行搬不同 expert，再交给计算线程，就仍可能有收益。v014 的 FP8 h7168 实测约快了 2.1%


## codex 回答

让不同warp分担不同expert读取任务嘛，那这么做的目的是？

| 分配方式 | 每个 warp 要读什么 |
|---|---|
| 所有 warp 一起处理每个 expert | 读 8 个 expert，每个读其中 128 列 |
| 每个 warp 独立负责 expert | 读 2 个 expert，每个读完整的 512 列 |
每个 warp 搬的数据量相同：
8 × 128 = 2 × 512
但读取的组织变了：

- **逐 expert 的读取轮次从 8 轮变成 2 轮**，减少地址计算、分支和逐轮等待。
- 每线程一次搬更多连续元素，我们用的是 **8 个 BF16，也就是 16 字节的向量读取**。
- 不同 warp 的 expert 读取可以独立推进，希望更充分地利用访存吞吐。

只看**搬运线程取 `pos`** 这一步：

路由位置本来就保存在一张 global 表里。

**原来：**

原始 global 表
    ↓ 复制
局部表（本想放寄存器，实测出现 private 开销）
    ↓ 按 expert 查找
pos

**改后：**

原始 global 表
    ↓ 直接按 expert 查找
pos

**两种写法的 `expert` 都是动态的。** 改动只是让这一步直接查原始表，绕过那份产生 private 开销的局部表。

当时想通过局部表加速，实际编译结果却增加了开销，所以直接查原始表反而更好。

## 个人复述

我来复述一下原来的方法和v014使用的方法，请你细细评判。场景是这样的，一个CTA计算一行的一部分列，以256个线程，k=8为例，那么4个warp需要取x的8行。之前的做法是，每个warp取所有八行我们要的部分的1/4，这样子每个warp都拿到了8个小部分，我们觉得这样不太好，希望每个warp拿到更加连续的一部分，所以改成了每个warp取两行，这样拿到的东西就更加连续，但是warp之间数据是不共享的，所以这一部分需要额外的shared memory来存