# v003：用 FP32/uint32 加速 FP8 输出（sc-16g）

## 改了什么

只修改 tileops/kernels/moe/reduce_fused.py 的 E4M3 输出分支：避开 SDK 的 FP32→FP64→通用 FP8 软件转换，用 FP32 和 uint32 位运算完成 RNE 编码。原归约顺序、128 线程、输入和 Op 接口保留。Base/XSF 使用原分支。

保留原 SDK 的 SATFINITE 语义：溢出/Inf 饱和、NaN 规范化、signed zero 与 nearest-even 舍入。使用 TileLang 原语，没有注入 C/C++，没有 FP16 中间近似。

## 验证

- 开发候选：202,752 次编码逐字节对照，0 差异；131 个现有正确性用例通过。
- 接入正式文件后：131 个现有正确性用例通过，16 个现有 benchmark 用例通过。
- 最终生成代码不再调用 __maca_cvt_float_to_fp8 / float2_to_fp8x2。Base H7168 的生成代码与改动前一致。
- 原始结果位于 raw/，实际接入源码、生成代码与补丁位于 codegen/。

## 配对性能结果

使用项目 bench_kernel：10 warmup、50 repeat×3 trials、L2 flush，A-B-B-A 顺序；每次记录计时方法和 input-clone 标志。以下为保守比较：取两次中较快的基线、较慢的候选，避免抬高加速比。开发候选与正式文件实现相同转换机制，下面数据来自开发候选；正式文件完整结果另存 final_benchmark_16g.csv。

| 变体 | Workload | 基线 μs | 候选 μs | 保守加速比 |
|---|---|---:|---:|---:|
| fp8 | tiny | 4.905 | 4.004 | 1.23× |
| fp8 | h3072 | 45.578 | 26.573 | 1.72× |
| fp8 | h7168 | 94.070 | 55.834 | 1.68× |
| fp8 | prefill | 430.264 | 364.068 | 1.18× |
| quantized | tiny | 4.792 | 4.132 | 1.16× |
| quantized | h3072 | 45.507 | 26.650 | 1.71× |
| quantized | h7168 | 94.531 | 55.690 | 1.70× |
| quantized | prefill | 431.601 | 363.960 | 1.19× |

8 组都有收益。h7168 约 1.7×，prefill 约 1.18×，tiny 约 1.2×。FP8 tiny 的首次基线测量与 h7168 的末次基线有明显波动，所以没有采用两次平均后的较高加速比。

最终现有 16 组 benchmark 也有单轮波动，例如 FP8/h7168 为 71.5 μs，而开发配对约 55.8 μs；XSF/prefill 也出现较高延迟。因此这些加速比不是保证值，保留完整原始测量。代码相同的 Base/XSF 不能把单轮差值解释成优化收益。

## Profiling 证据

- mcTracer：H7168 短 trace 中，基线 97.280 μs，候选 58.368 μs。各 11 次，包含 warmup、没有逐次清 L2，不用它替代正式 benchmark。
- mcProfiler：计算指令 7,351,174→2,909,100（减少约60%）；访存指令 68,952→76,050（增加约10%）。生成代码的输出写回从8字节变为4字节，与访存指令增加相符。
- mcTracer 的两者寄存器都是96/thread、private memory为0、mtreg occupancy字段都是18%；没有证据声称本轮减少了寄存器或spill。
- 汇编导出尝试因 mxcc 不接受 -S 失败，没有用汇编结果作判断。

## 工具与 cycle_trace

mcTracer 可用，位于 /opt/maca/bin/mcTracer；mcProfiler 可用。runtime 中发现 CYCLE_TRACE_MODE 开关，但没有找到独立 cycle_trace 命令或 TileLang 接口。设为1后的实际输出仍是同类 kernel 时间线，没有确认可用的算子内部逐段 cycle 数据。尝试和原始输出保留在 raw/cycle_trace_attempt。

## 结论

减少了FP8转换的通用64位工作，指令和配对耗时都下降；结果正确，已接入正式 kernel。下一轮可研究新路径的4字节写回，但本轮没有继续扩展改动。

数据与脚本全部在 profiles/v003，另备份到 /data/sc16g-recovery-20261004/profiles/v003。
