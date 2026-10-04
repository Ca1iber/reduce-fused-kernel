# v013：减少CTA同步的反证实验

依据生成地址公式，尝试将async ring2/ring3的CTA同步删除，或改为warp同步。
两类版本均出现输出字节不一致，不使用它们的性能数字、不接入正式kernel。
控制版本仍正确。地址相同不足以保证该async/shared路径的可见性、编译器排序与buffer生命周期；具体导致差异的机制尚未隔离，不能声称已证明某一硬件原因。
本轮warp运行时baseline约92us，对比前轮约48us表明环境/工具状态波动；不跨轮比较速度。
保留所有失败源码、生成设备代码、日志以及meta/*failures.json。
下一步保留CTA同步，探索warp协作gather和hidden方向更长的计算重叠窗口。
