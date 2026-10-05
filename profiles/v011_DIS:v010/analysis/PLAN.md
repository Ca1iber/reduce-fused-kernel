# v011：流水线 / shared memory 探索的证据基线

v005只测双寄存器预取，v006只测单buffer、每K两次CTA同步；不能据此推出整个方向不可行。
先验证当前v010基线、工具采集窗口和MACA后端能力，再迭代机制。

当前runtime报告warp_size=64，shared_memory_per_block和per_multiprocessor均65536 bytes。
TileLang提供MACA async copy/local barrier接口；编译、执行和reduce_fused收益均需单独验证。
旧mcProfiler命令没有显式kernelnames过滤，本轮custom加精确过滤，并检查计数是否包括输入生成。
后续试验既比较同tile/threads控制组，也比较当前正式配置。

## 机制与可证伪预测

1. K寄存器流水线：不同预取距离/深度/调度，检查load前移、寄存器、spill、等待变化。
2. K shared pipeline：同步与真实async分开，双/多buffer，最少必要同步，确认wait对应目标buffer。
3. shared协作读取：64线程warp分担expert读取，shared中转原输入，保留K累加顺序；检查并发与smem驻留代价。
4. hidden流水线：一CTA处理多列块，尝试下一块访存与当前归约/FP8编码重叠。
5. shared元数据/输出整理：明确共享复用对象与同步成本。

profile记录时长、grid/block、寄存器、private/spill、shared bytes、load/store宽度/顺序、barrier/wait，以及有效native访存/stall/shared冲突计数。
不通过sanity check的计数保留但不用于结论。
正式文件保持v010，直到新方案正确且有可复现收益。
最终区分已测配置、硬件/编译器限制与剩余可能，不用有限实验宣称排除所有实现。
