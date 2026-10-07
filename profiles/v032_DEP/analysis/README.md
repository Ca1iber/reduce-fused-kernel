# v032：tiny 的输入与有效标志打包

## 目的与改动

以已接入的 v030 为控制组，尝试减少一次 warp shuffle。只处理 FP16、T=32、K=2、H=256 的 tiny；Base/XSF 保持 256 线程，FP8 保持 512 线程，每个 CTA 对应一个 token。

原实现分别交换另一线程的 position、系数和输入值。接收线程只需要知道 position 是否有效，因此把有效标志与输入一起放进一个 32 位数据中交换，系数仍单独交换。Base/XSF 每对线程处理两列，shuffle 从 4 次减到 3 次；FP8 每对线程处理一列，从 3 次减到 2 次。仍按 expert0、expert1 的顺序执行 FP32 累加。

尝试了两种编码：

1. **FP16 位打包**：低 16 位保存 FP16 输入，第 16 位保存有效标志；接收端解包后转成 FP32。
2. **FP32 位打包**：保留原来的 FP16→FP32 转换，在转换后数值的最低位保存有效标志；接收端清掉最低位，恢复输入。有限 FP16 数转换成 FP32 后，该位原本为零。

Quantized 正式 tiny 使用 v029 的每线程独立计算实现，没有 shuffle，两轮均作为未改动控制组。

## 测量方法

使用项目现有 bench_kernel：CUPTI 计时、L2 flush、克隆输入，10 次预热、50 次重复、3 次 trial，外层 5 轮交替测量控制组与候选。两种方案各测四个 tiny；random 与带负 position 的 padded 路由均与正式实现逐字节一致。未运行额外测试套件或其他 shape。两个表分别来自不同轮测量，比较时使用各自同轮控制组。

## 结果

正值表示候选更快，负值表示候选更慢；时间单位为 μs。

### FP16 位打包

| 变体 | 当前正式版本 | 打包候选 | 时间下降 |
|---|---:|---:|---:|
| Base | 3.384320 | 3.491840 | -3.18% |
| XSF | 3.384320 | 3.722240 | -9.98% |
| FP8 | 3.609600 | 3.609600 | 0.00% |
| Quantized（未改动） | 3.701760 | 3.691520 | 0.28% |

数据：[comparison_16g.csv](../raw/comparison_16g.csv)。

### FP32 位打包

| 变体 | 当前正式版本 | 打包候选 | 时间下降 |
|---|---:|---:|---:|
| Base | 3.343360 | 3.430400 | -2.60% |
| XSF | 3.363840 | 3.717120 | -10.50% |
| FP8 | 3.640320 | 3.645440 | -0.14% |
| Quantized（未改动） | 3.696640 | 3.681280 | 0.42% |

数据：[floatbits_comparison_16g.csv](../raw/floatbits_comparison_16g.csv)。Quantized 的时间差来自测量波动，不能归为优化收益。

## 结论

两种打包均未获得收益，归类为 DEP，未接入正式 kernel。Base/XSF 明显变慢，FP8 基本持平。

减少 shuffle 同时增加了打包、解包位运算，并改变指令依赖和调度。第一种还把对端输入的 FP16→FP32 转换移到输出线程；第二种保留转换路径后仍未改善。当前只有时延与生成代码证据，没有新增硬件等待计数，不能据此断言某一种 stall 是具体原因。

## 文件

- [tiny_packed.py](../scripts/tiny_packed.py)、[benchmark.py](../scripts/benchmark.py)：FP16 位打包及测量脚本。
- [tiny_floatbits.py](../scripts/tiny_floatbits.py)、[benchmark_floatbits.py](../scripts/benchmark_floatbits.py)：FP32 位打包及测量脚本。
- codegen：正式代码快照与候选生成代码。
- raw：汇总 CSV 与各轮样本。
- logs：编译与测量日志。
- meta：来源版本与实验完成状态。
