# v013：减少CTA同步的反证实验

依据生成地址公式，尝试将async ring2/ring3的CTA同步删除，或改为warp同步。
两类版本均出现输出字节不一致，不使用它们的性能数字、不接入正式kernel。
控制版本仍正确。地址相同不足以保证该async/shared路径的可见性、编译器排序与buffer生命周期；具体导致差异的机制尚未隔离，不能声称已证明某一硬件原因。
本轮warp运行时baseline约92us，对比前轮约48us表明环境/工具状态波动；不跨轮比较速度。
保留所有失败源码、生成设备代码、日志以及meta/*failures.json。
下一步保留CTA同步，探索warp协作gather和hidden方向更长的计算重叠窗口。

## v017 已定位根因

后续最小反例与最终机器指令证实：当前MXCC对条件async-copy的ticket依赖/等待生成不正确，wait后仍可能读到未搬完的shared旧值。第二次copy的条件分支足以触发，单warp、无slot复用、纯整数copy均可复现。额外CTA barrier补充更严格访存等待；强制每次MEM后等待的无CTA版本通过现有131用例。详见[v017根因报告](../../v017_DIS:v013/analysis/v017_barrier_root_cause_16g.md)。上文“原因尚未隔离”为当时状态。
