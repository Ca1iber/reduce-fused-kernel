# v007：固定 128 线程，比较 hidden tile size

只改变 v004 的 tile size，比较 256、512、1024；FP8 编码、K 归约顺序和线程数相同。
正式 kernel 文件未改动；本轮记录参数对比结果。

## 计时方法

使用项目 bench_kernel：10 warmup、50 repeats × 3 trials、L2 flush；GPU timeline（cupti）计时。
每组依次测 1024/512/256、256/1024/512、512/256/1024，各 tile 的三次结果取中位数。
固定 seed=1235，所有输入相同；计时前 18 组输出与 v004 逐字节一致。
h3072/h7168 使用 per-iteration 输入克隆；prefill 按项目内存阈值使用相同地址，三种 tile 协议一致。
本轮没有运行 mcProfiler；不能据此把变慢原因确定为某个硬件瓶颈。

## 结果

| 变体 | workload | tile 256 μs | tile 512 μs | tile 1024 μs | 本轮最快 |
|---|---|---:|---:|---:|---|
| fp8 | h3072 | 29.379 | 22.764 | 23.757 | 512 |
| fp8 | h7168 | 61.932 | 48.184 | 48.886 | 512 |
| fp8 | prefill | 502.405 | 361.923 | 354.888 | 1024 |
| quantized | h3072 | 29.763 | 22.989 | 23.823 | 512 |
| quantized | h7168 | 62.920 | 48.532 | 48.840 | 512 |
| quantized | prefill | 510.822 | 364.872 | 357.361 | 1024 |

## 判断

- h3072：512 比 1024 快约 3.6%～4.4%，值得作为这个 workload 的参数选择。
- h7168：512 比 1024 快约 0.6%～1.5%，收益较小。
- prefill：1024 比 512 快约 2%；不能把所有 workload 都统一改成 512。
- 256：全部变慢。更多 CTA 并不保证更快；tile 越小，每个 CTA 的工作减少，同时 CTA 总数、路由/权重的重复读取和调度工作增加。这是解释方向，未通过硬件计数器确认各因素占比。

## 正确性

- tile 256：现有 131 个正确性用例全部通过。
- tile 512：现有 131 个正确性用例全部通过。
- tile 1024：现有 131 个正确性用例全部通过。

原始计时及各轮值：`raw/tile_sweep.csv`。
复现：在仓库根目录使用此前相同的 MACA_PATH/PYTHONPATH 环境，运行
`/opt/conda/bin/python profiles/v007/scripts/run.py`。
基线源码与哈希：`codegen/baseline_v004.py`、`meta/source.json`。
