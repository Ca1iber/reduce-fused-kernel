# 2026-10-09：正式版本16组 benchmark 复测

restart 分支，提交 f9c1a213。直接运行现有 benchmarks/ops/bench_moe_reduce_fused.py；16 passed，未运行其他测试或采图。随机输入使用seed1235；10 warmup、50 repeat × 3 trials，L2 flush，项目CUPTI计时。prefill按项目规则不克隆输入。

| 变体 | Workload | 原封盘 μs | 本次 μs | 耗时变化 |
|---|---|---:|---:|---:|
| Base | tiny | 3.343 | 3.343 | +0.00% |
| Base | h3072 | 20.818 | 20.731 | -0.42% |
| Base | h7168 | 44.175 | 44.083 | -0.21% |
| Base | prefill | 352.701 | 350.618 | -0.59% |
| XSF | tiny | 3.364 | 3.405 | +1.22% |
| XSF | h3072 | 20.731 | 20.644 | -0.42% |
| XSF | h7168 | 44.360 | 44.278 | -0.18% |
| XSF | prefill | 354.627 | 352.348 | -0.64% |
| FP8 | tiny | 3.640 | 3.630 | -0.28% |
| FP8 | h3072 | 21.156 | 21.064 | -0.44% |
| FP8 | h7168 | 43.090 | 43.167 | +0.18% |
| FP8 | prefill | 334.515 | 334.003 | -0.15% |
| Quantized | tiny | 3.697 | 3.671 | -0.69% |
| Quantized | h3072 | 22.323 | 22.272 | -0.23% |
| Quantized | h7168 | 45.015 | 45.107 | +0.20% |
| Quantized | prefill | 337.004 | 336.020 | -0.29% |

本次结果与原封盘值接近，没有复现此前mcProfiler报告的数十倍周期差。支持当前正式kernel实际性能基本保持稳定；它不能单独定位mcProfiler异常计数的原因。原封盘值来自历史配对多轮测量，本次是一遍现有benchmark复测，小幅差异不视为新的优化收益。

[完整精度CSV](../raw/benchmark_rerun_20261009T105946/comparison_16g.csv)、[16个测试结果XML](../raw/benchmark_rerun_20261009T105946/results.xml)、[运行状态](../raw/benchmark_rerun_20261009T105946/status.json)。
