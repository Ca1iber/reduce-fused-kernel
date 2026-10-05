# 正式 v003 的补充 profile（基于395df34）

这次使用实际 v003 源码快照，而不是开发候选。与 naive、v004 在同一进程、同一输入上各运行11次。

| 实现 | GPU中位 μs | registers/thread | grid |
|---|---:|---:|---|
| naive | 97.536 | 96 | {'x': 512, 'y': 1, 'z': 1} |
| v003 | 58.880 | 96 | {'x': 512, 'y': 1, 'z': 1} |
| v004 | 49.920 | 20 | {'x': 512, 'y': 7, 'z': 1} |

正式v003与naive的主要差异仍是输出编码：避开SDK通用FP64转换；归约顺序、128线程、一个token一个CTA不变。生成代码在 codegen/formal_v003_h7168.cu，快照在 formal_v003_395df34.py。

此次mcProfiler添加RoofLine的采集超时，部分数据在 raw/formal_395df34_mcprofiler；没有把不完整采集写成成功。完整mcTracer位于v004_ACC/raw/trace，精确执行版本对应metadata及代码快照。

补充：CLI超时后回收到了单kernel的dumped_result与三张RoofLine PNG。执行流程没有正常完成；恢复出的计数保存在上述raw目录，可独立查看。


## 回收数据的有效性限制

回收的 native RoofLine 记录 all_memacs=259,606,944 bytes、during=1,152,689；
按1.125GHz换算约1024.6μs，与这个case约62.4MB语义流量和mcTracer约58.9μs明显不符。
可能涉及异常窗口或多次回放累计，具体原因未确认。
因此这份超时采集的native计数与PNG仅作为问题记录，不用于带宽、上限或优化收益结论。
正式v003的有效profile比较采用同次mcTracer数据和生成代码。
