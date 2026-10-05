# v008：联合调整 threads 和 hidden tile（sc-16g）

## 改动与范围

v007 固定 128 线程；本轮在同一 FP8 编码、K 归约顺序、输入和输出逻辑下，把 threads 参数化。
首轮测试 64/128/256 线程 × tile512/1024；补测 64线程/tile256，并重新测 128线程/tile512、1024。
Base/XSF、tiny 继续使用 v004；正式 kernel 文件未改动。本轮仅保存参数实验。

## 方法

机器 sc-16g，容器与 GPU 信息见 `meta/runtime.json`。
使用项目 bench_kernel：10 warmup、50 repeats × 3 trials、L2 flush、cupti GPU timeline。
首轮六配置分三轮旋转顺序，每个配置三次结果取中位数；补测三配置也旋转顺序三轮。
输入 seed=1235；h3072/h7168 按项目规则逐次克隆输入，prefill 因内存阈值复用地址。
每组计时前，与 v004 输出逐字节检查。编译不计入 kernel 延迟。
两轮各自重测相同的 128线程基线；小于约 1% 的差别不作为明确优化收益。

## 首轮结果（μs，越小越好）

| 变体 | workload | 64线程 / tile512 | 64线程 / tile1024 | 128线程 / tile512 | 128线程 / tile1024 | 256线程 / tile512 | 256线程 / tile1024 |
|---|---|---:|---:|---:|---:|---:|---:|
| fp8 | h3072 | 23.782 | 24.878 | 22.733 | 23.695 | 29.353 | 22.840 |
| fp8 | h7168 | 48.461 | 48.952 | 48.169 | 48.809 | 61.727 | 48.527 |
| fp8 | prefill | 357.852 | 360.617 | 361.856 | 354.918 | 493.885 | 357.996 |
| quantized | h3072 | 23.813 | 24.878 | 23.009 | 23.741 | 29.788 | 23.086 |
| quantized | h7168 | 48.532 | 49.034 | 48.573 | 48.845 | 62.659 | 48.788 |
| quantized | prefill | 360.873 | 364.129 | 364.841 | 357.837 | 501.683 | 361.472 |

## 补测结果（μs，同轮基线）

| 变体 | workload | 64线程 / tile256 | 128线程 / tile512 | 128线程 / tile1024 |
|---|---|---:|---:|---:|
| fp8 | h3072 | 22.825 | 22.784 | 23.767 |
| fp8 | h7168 | 47.964 | 48.118 | 48.742 |
| fp8 | prefill | 365.681 | 361.672 | 354.632 |
| quantized | h3072 | 23.050 | 22.999 | 23.711 |
| quantized | h7168 | 48.804 | 48.512 | 48.870 |
| quantized | prefill | 370.729 | 365.399 | 357.586 |

## 判断

- 改线程数没有超过上一轮最佳参数的明确、稳定收益；128 线程仍是合适的默认选择。
- h3072：128线程/tile512 优于 v004 的 128线程/tile1024。256线程/tile1024 接近，但没有明显超过它。
- h7168：大多数配置接近；补测 64线程/tile256 仅在 FP8 上比同轮 128线程/tile512 快约 0.3%，Quantized 反而较慢，因此不将它视为明确的新优化。
- prefill：128线程/tile1024 在两轮都最快。减小 tile、增加线程都没有改善这个 workload。
- 256线程/tile512 明显变慢；threads 和 tile 需要联合选择，不能单独追求更多线程。

## 正确性

36 个首轮参数/workload 组合、18 个补测组合均与 v004 输出逐字节一致。
七个不同参数组合分别运行现有测试，共 7×131 次通过（不是 917 个不同用例）。

- threads=64, tile=512：131 passed。
- threads=64, tile=1024：131 passed。
- threads=128, tile=512：131 passed。
- threads=128, tile=1024：131 passed。
- threads=256, tile=512：131 passed。
- threads=256, tile=1024：131 passed。
- threads=64, tile=256：131 passed。

## 证据范围与复现

本轮没有采集 mcProfiler/mcTracer 资源计数器，不能把计时差异确定归因于某个寄存器、占用率或 HBM 指标。
源码快照：`codegen/baseline_v004.py`，哈希与提交：`meta/source.json`。
首轮原始值：`raw/threads_tile_sweep_16g.csv`；补测：`raw/extra_64x256_16g.csv`。
在仓库根目录使用既有 MACA_PATH/PYTHONPATH 环境：
`/opt/conda/bin/python profiles/v008_DIS:v007/scripts/run.py`，然后
`/opt/conda/bin/python profiles/v008_DIS:v007/scripts/run_extra.py`。
