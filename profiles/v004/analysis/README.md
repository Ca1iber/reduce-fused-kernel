# v004 实验记录（sc-16g）

## 改动

FP8/Quantized 的 hidden 分块：1024/512/256 选择可整除块；每 token 多个 CTA，tiny 与 Base/XSF 走 v003 路径。

## 配对结果

原项目计时：10 warmup，50 repeats×3 trials；A-B-B-A-A-B，分别取三次统计值的中位数；相同输入、L2 flush。benchmark 前逐字节与基线对照一致。

| 变体 | workload | 基线 μs | 试验 μs | 加速比 |
|---|---|---:|---:|---:|
| fp8 | tiny | 3.927 | 3.932 | 0.999× |
| fp8 | h3072 | 25.902 | 23.757 | 1.090× |
| fp8 | h7168 | 55.813 | 48.476 | 1.151× |
| fp8 | prefill | 363.612 | 354.662 | 1.025× |
| quantized | tiny | 4.070 | 4.050 | 1.005× |
| quantized | h3072 | 25.907 | 23.788 | 1.089× |
| quantized | h7168 | 55.660 | 48.763 | 1.141× |
| quantized | prefill | 363.750 | 357.432 | 1.018× |

## 判断

六个实际改变的大 workload 均有收益；tiny 使用相同函数，极小差值视为波动。已接入正式文件，131个正确性用例、16个现有benchmark全部通过。

131 个现有正确性用例通过。来源：raw/paired_benchmark_16g.csv、raw/correctness.xml。

## mcTracer

以下短 trace 各11次，包含warmup、不逐次flush；不是正式benchmark。

| 版本 | 中位 μs | registers/thread | shared bytes | grid |
|---|---:|---:|---:|---|
| naive | 97.536 | 96 | 0 | {'x': 512, 'y': 1, 'z': 1} |
| v003 | 58.880 | 96 | 0 | {'x': 512, 'y': 1, 'z': 1} |
| v004 | 49.920 | 20 | 0 | {'x': 512, 'y': 7, 'z': 1} |

mcProfiler 的正式v003任务超时（180秒），因此没有声称完成新的native Roofline。已保存日志及部分原始数据；实际v003/v004的资源与时长对比来自mcTracer。
