# v016：2/4/8 行寄存器预取（16g）

## 1. 上版本遗留问题

v005 只测两行交替预取；v012_DEP/v015 又表明 shared 和同步可能抵消重叠收益。两行失败不能判断更深的寄存器预取不可行。

## 2. 问题原因分析

直接版本逐 K 执行 load→转换→加权累加；后续 expert 的独立读取没有在源码中提前组织。K=8 时可先请求多行，增大独立访存窗口；代价是存活输入更多，寄存器增加，可能降低驻留或发生 private 溢出。
这条路线检验加载调度/指令级并行；它不减少必需 x 流量，也不提高算术强度。

## 3. 本版本解决方案

用输入 dtype 的 register fragment `[prefetch_rows,tile_hidden]`。先预取 depth 行，消费 k 后请求 k+depth 到同槽；depth=8,K=8 时全部行先加载，再按原 K 顺序累加，无 shared/CTA 同步。
这是寄存器中的软件预取；不把它叫成 global→shared 硬件 async copy。

## 4. 具体落地策略

[get_prefetch_kernel](../scripts/prefetch_kernel.py) 增加 prefetch_rows=2/4/8。h3072、h7168 的 SF 路径保持 tile512/128，与当前生产几何相同；prefill 对照 tile1024/128。Base/XSF 的 current 是整行256线程，因此同时保留同几何 direct 控制，避免把几何影响混入预取效果。
生成 C++ `profile_register8.cu` 明确先出现 `first<8` 的8B向量读取，再出现 K 循环，BF16 packed 转换保留。这里只验证生成源结构，**没有以 C++ 顺序假定最终机器指令严格顺序**。
[validate.py](../scripts/validate.py) 将现有测试 factory 替换为深度8，在全部131用例通过；[JUnit](../raw/correctness_register8.xml) tests131/failures0/errors0/skipped0。覆盖不同 dtype、四变体、无效路由等现有条件，benchmark另有逐字节对照。

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


初步 h7168 同几何测试：direct48.328µs，depth2 48.353，depth4 49.628，depth8 44.698。只有 depth8 获得收益。
12组扩展对照：

| 变体 | workload | 正式基线 µs | 同几何直接读 µs | 候选 µs | 加速比 | 减少 µs | 延迟降低 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | h3072 | 20.751 | 21.939 | 20.813 | 0.9970× | -0.061 | -0.30% |
| xsf | h3072 | 20.685 | 21.990 | 21.540 | 0.9603× | -0.855 | -4.13% |
| fp8 | h3072 | 22.738 | 22.712 | 21.125 | 1.0763× | +1.613 | +7.09% |
| quantized | h3072 | 22.907 | 22.912 | 22.328 | 1.0259× | +0.579 | +2.53% |
| base | h7168 | 44.017 | 49.393 | 46.582 | 0.9449× | -2.565 | -5.83% |
| xsf | h7168 | 44.134 | 49.715 | 47.247 | 0.9341× | -3.113 | -7.05% |
| fp8 | h7168 | 48.148 | 48.169 | 44.421 | 1.0839× | +3.727 | +7.74% |
| quantized | h7168 | 48.579 | 48.594 | 45.527 | 1.0670× | +3.052 | +6.28% |
| base | prefill | 351.078 | 365.916 | 367.191 | 0.9561× | -16.113 | -4.59% |
| xsf | prefill | 353.480 | 369.439 | 370.058 | 0.9552× | -16.579 | -4.69% |
| fp8 | prefill | 354.724 | 354.714 | 359.137 | 0.9877× | -4.413 | -1.24% |
| quantized | prefill | 357.504 | 357.289 | 362.849 | 0.9853× | -5.345 | -1.50% |

独立进程重新生成输入、5外层轮次确认四个有收益的SF组合：

| 变体 | workload | 正式基线 µs | 同几何直接读 µs | 候选 µs | 加速比 | 减少 µs | 延迟降低 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp8 | h3072 | 22.738 | 22.738 | 21.074 | 1.0790× | +1.664 | +7.32% |
| quantized | h3072 | 22.958 | 22.943 | 22.359 | 1.0268× | +0.599 | +2.61% |
| fp8 | h7168 | 48.067 | 48.072 | 44.273 | 1.0857× | +3.794 | +7.89% |
| quantized | h7168 | 48.573 | 48.543 | 45.466 | 1.0684× | +3.108 | +6.40% |

速度提升与延迟降低分开列出；例如 FP8 h7168 加速8.57%，延迟降低7.89%。全样本及计时口径在 CSV measurements 中。
命令：`python profiles/v016_DEP/scripts/run_confirmation.py`；完整扩展：`python profiles/v016_DEP/scripts/run_expanded.py`。脚本追加 CSV，重跑前另存原结果。

## 6. Profile 指标变化

| 方案 | 同次 trace µs | 寄存器/线程 | shared bytes | private_total | grid | threads |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 50.688 | 12 | 0 | 0 | 512×14 | 128 |
| register2 | 51.456 | 15 | 0 | 0 | 512×14 | 128 |
| register4 | 51.968 | 19 | 0 | 0 | 512×14 | 128 |
| register8 | 46.592 | 22 | 0 | 0 | 512×14 | 128 |

同应用资源显示 depth8 registers/thread 12→22，shared/private 均0，grid/block不变；没有发现该配置溢出。
本轮原始 register2/4/8 大事件集计数与零shared代码、算法流量冲突，已明确作废，见 [native_validity.json](../meta/native_validity.json)。不拿这些异常 stall/shared 数据解释收益。
隔离 `--custom --per-kernel --kernelnames reduce_fused_kernel_kernel --counts 1`，仅采5个核心指标后的深度8结果有效：

| 方案 | 总指令 | 计算类指令 | 内存类指令 | 读 bytes | 写 bytes |
| --- | --- | --- | --- | --- | --- |
| baseline | 3999397 | 3757961 | 241436 | 58835328 | 3670464 |
| register8 | 4318990 | 4120258 | 198732 | 58798432 | 3670336 |

正式理论输入58,720,256bytes，输出3,670,016bytes；重采 read58,798,432、write3,670,336相差约0.13%/0.009%，与目标流量一致。说明收益不是少读专家输出或少写结果。
内存类指令下降约17.7%、总指令增加约8.0%，因此“指令少所以快”也不是充分解释；提前独立读取是实现和计时共同支持的机制，缺乏有效候选 stall 计数来进一步分解。
原始重采 [per-kernel JSON](../raw/native_register8_minimal/tool_output/1_reduce_fused_kernel_kernel.txt.json)、[命令](../meta/recollection_job.json)、[汇总](profile_comparison.json) 均保留。baseline来自同版本有效目标周期数据，不用全应用aggregate。

## 7. 实验总结

更深的寄存器预取可行，否定“两行失败因此流水线没用”的外推。收益确认在 SF=true、T512、K8、BF16、H3072/7168；其它测到的组合无收益，不推广。
正式文件仍是 v010；候选源码、131用例结果、12组数据和独立确认都已保存，可按这些shape选择调用。正式集成不属于本次可行性证明的前提。
