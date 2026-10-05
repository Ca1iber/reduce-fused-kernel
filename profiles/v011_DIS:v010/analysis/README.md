# v011：基线与工具能力审计（sc-16g）

正式源7cd99af/c6b5dcb13fa171ace058b3f0a50d8361a439eb3335ad4596c539273859831596，容器96933d7d09ab。
K8、T512、H7168、BF16；FP8/Quantized正式tile512/128线程，Base/XSF整行256线程。

mcProfiler custom+显式kernelnames过滤正常完成，读取58,833,600 bytes，写入3,670,464 bytes。
目标kernel最小输入58,720,256、输出3,670,016 bytes，分别相差约0.193%/0.012%。
该流量与目标kernel一致，没有旧采集约4倍流量的输入生成污染现象。
新命令同时改了采集窗口和过滤，因此暂不归因于某一个旧参数。

Total Instructions 3,963,583；Compute 3,722,083；Memory 241,500。
ISU报告vls_pipeline_stall=192,561，vls_wdata_stall=79,984，wsm_stall=0，valu_stall=0。
这些是工具聚合计数，仅用于相同配置的前后对照，未把Total Cycles当作wall-time或占用率。
mcTracer资源：12 registers/thread，private=0，shared=0，grid512×14，block128。
详细11个warm/ROI事件时长在analysis/baseline_trace_summary.json；正式计时为项目cupti、L2flush协议，在raw/baseline_bench_*.json。
单次未配对的Base计时50.14us、XSF44.26us显示环境/运行波动；后续判断采用同进程配对，不跨单独run下结论。

## 工具问题与处理

第一次mcTracer把absolute odname拼到cwd后导致输出路径错误但退出0；改用relative odname重采，11个目标事件已验证存在。
原T.alloc_maca_barrier接口缺barrier_type，后端codegen报错。
用已有T.alloc_buffer(scope=local.barrier, annotations={barrier_type:b128vectype})表达同一个类型后，生成memcpy_async<16>和wait，8192个FP16元素逐字节拷贝验证通过。
源码、失败日志、成功probe及设备代码均保留。未修改SDK/编译器或注入外部设备函数。
异步拷贝可执行不等于reduce_fused会加速，下一版本测试具体流水线。

后续机制与预测见PLAN.md。正式kernel保持v010。

## 时间口径限制

当前mcTracer 11个目标事件中位82.432us，而独立项目cupti基线为48.220us。两个运行的环境和工具介入不同，不能把这两个绝对时间混用。后续速度判定以同进程项目配对benchmark为准；mcTracer用于同次采集中的资源和执行顺序对照，mtreg_occupancy不解释为achieved occupancy。
