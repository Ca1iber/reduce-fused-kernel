# v006 实验记录（sc-16g）

## 改动

在 v004 分块基础上，将每行输入先拷到 shared memory，CTA 同步后计算，再同步重用。

## 配对结果

原项目计时：10 warmup，50 repeats×3 trials；A-B-B-A-A-B，分别取三次统计值的中位数；相同输入、L2 flush。benchmark 前逐字节与基线对照一致。

| 变体 | workload | 基线 μs | 试验 μs | 加速比 |
|---|---|---:|---:|---:|
| fp8 | tiny | 3.917 | 3.932 | 0.996× |
| fp8 | h3072 | 23.788 | 24.038 | 0.990× |
| fp8 | h7168 | 48.430 | 49.874 | 0.971× |
| fp8 | prefill | 353.925 | 356.178 | 0.994× |
| quantized | tiny | 4.091 | 4.101 | 0.998× |
| quantized | h3072 | 23.798 | 24.428 | 0.974× |
| quantized | h7168 | 48.799 | 51.236 | 0.952× |
| quantized | prefill | 358.026 | 361.395 | 0.991× |

## 判断

所有实际改变的大 workload 都没有收益，未接入正式 kernel。保留试验、131用例和逐字节对比结果。

131 个现有正确性用例通过。来源：raw/paired_benchmark_16g.csv、raw/correctness.xml。

## mcTracer

以下短 trace 各11次，包含warmup、不逐次flush；不是正式benchmark。

| 版本 | 中位 μs | registers/thread | shared bytes | grid |
|---|---:|---:|---:|---|
| v004 | 51.200 | 20 | 0 | {'x': 512, 'y': 7, 'z': 1} |
| v006 | 52.224 | 18 | 2048 | {'x': 512, 'y': 7, 'z': 1} |

该算子每个输入元素只供一个输出列使用，没有跨线程复用；这个配置增加共享内存往返和每个K的同步。它没有收益，不代表所有shared memory方案都无效。
