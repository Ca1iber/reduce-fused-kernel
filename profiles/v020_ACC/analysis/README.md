# v020：交换 grid 维度顺序

## 改了什么

基于当前正式 v010，仅交换一行 grid 维度和 block 索引绑定：

```python
# v010
with T.Kernel(num_tokens, hidden // tile_hidden, threads=num_threads) as (pid_token, pid_hidden):
# v020
with T.Kernel(hidden // tile_hidden, num_tokens, threads=num_threads) as (pid_hidden, pid_token):
```

连续 block 编号由不同 token 的同一 hidden 块，改为同一 token 的相邻 hidden 块。实际执行顺序仍由硬件调度决定。tile、threads、CTA 总数、K 累加顺序和 FP8 编码不变；没有增加 shared memory，也没有重排或额外复制输入。

## 八组配对 benchmark

FP8/Quantized × tiny/h3072/h7168/prefill，共八组。tiny 为 T32/K2/H256/FP16，其余输入为 BF16；h3072/h7168 为 T512/K8，prefill 为 T4096/K8。项目 bench_kernel：10 warmup、50 repeat×3 trials、L2 flush、CUPTI；外层五轮交替先后顺序，取中位数。计时前全部输出逐字节一致。tiny/h3072/h7168 克隆输入；prefill 按项目阈值不克隆。

| 变体 | workload | tile/threads | v010 μs | 交换 grid μs | 加速比 | 耗时下降 |
|---|---|---|---:|---:|---:|---:|
| fp8 | tiny | 256/128 | 3.988 | 4.040 | 0.9873× | -1.28% |
| fp8 | h3072 | 512/128 | 22.758 | 22.845 | 0.9962× | -0.38% |
| fp8 | h7168 | 512/128 | 48.389 | 46.807 | 1.0338× | +3.27% |
| fp8 | prefill | 1024/128 | 354.621 | 346.854 | 1.0224× | +2.19% |
| quantized | tiny | 256/128 | 4.004 | 3.901 | 1.0262× | +2.56% |
| quantized | h3072 | 512/128 | 23.132 | 23.219 | 0.9963× | -0.38% |
| quantized | h7168 | 512/128 | 48.758 | 47.171 | 1.0336× | +3.26% |
| quantized | prefill | 1024/128 | 358.339 | 348.595 | 1.0280× | +2.72% |

## 判断与限制

- tiny：补测 FP8 3.988→4.040 μs（慢 1.28%），Quantized 4.004→3.901 μs（快 2.56%）。两组均逐字节一致。这说明单 hidden 块也应实测；本轮方向不一致，未进行独立复测，不将其概括为稳定的 tiny 收益。
- h3072：两种变体都慢约 0.4%，没有收益。
- h7168：两种变体的五轮中位耗时下降约 3.3%。
- prefill：FP8/Quantized 中位耗时分别下降约 2.2%/2.7%。
- 部分 baseline 和候选样本都有明显波动，全部保留，未删异常值。例如 FP8 h7168 的候选首轮 124.923 μs，其余四轮约 46.8 μs；Quantized prefill 的候选末轮 569.093 μs，其余四轮约 348～349 μs。此结论来自五轮中位数，尚无第二组独立完整复测。
- 地址映射已量化，mcProfiler 原始计数异常，不能据此断言 HBM/L2 利用率提高；原因和证据边界见下节。
- 本轮性能覆盖 FP8/Quantized 的全部四个 workload；Base/XSF 未跑性能比较。此前以 tiny 只有一个 hidden 块为由省略测量，是预判，现已补齐。

Quantized h7168 首次进程在写完所有样本和结果后以 SIGKILL 退出。容器 OOM 计数为 1、历史峰值达到 32 GiB 限额，但没有采集该次启动前的 OOM 计数，不能严格断言本次终止原因。原始记录保留在 raw/terminated_run。随后限制宿主线程/内存 arena，在计时之外释放空闲宿主堆内存，重新完成该组，退出码为 0；表中使用这次正常完成的结果。后续两组 prefill 同样正常完成。

## 交换 grid 改变了什么，为什么可能更快（16g）

### 1. 编号与任务的对应关系

以 h7168 为例：T=512、K=8、H=7168、tile=512、threads=128，每 token 分成 14 个 hidden 块，两版均为 7168 个 CTA。
定义展平 block 编号 `n = blockIdx.x + gridDim.x * blockIdx.y`：

```text
v010 grid=(512,14)：token=n%512，hidden块=n//512
    (token0,块0) → (token1,块0) → … → (token511,块0) → (token0,块1)

v020 grid=(14,512)：token=n//14，hidden块=n%14
    (token0,块0) → (token0,块1) → … → (token0,块13) → (token1,块0)
```

这是编号到任务的映射，不是实际执行时间线。硬件不保证 CTA 严格按此编号顺序执行，也不保证相邻编号驻留在同一 AP。

### 2. 真实路由下的输入地址变化

BF16 每元素 2 字节。把 x 起点记为 0，同一个 expert 槽位 k 的输入块起始地址为：

```text
地址 = (positions[token,k] * 7168 + hidden块 * 512) * 2
```

v010 的相邻编号切换 token，positions 通常不同，输入行地址跳跃。v020 在同 token 的 14 个块内保持 positions 相同，每块起始地址只增加 1024 字节。

用本轮 seed=1235 的真实 positions 计算得到下表。“地址跨度”比较的是相邻展平编号、同一个 expert 槽位的绝对起始地址差，不是实际请求的时间间隔；MB 使用十进制单位。

| 指标（h7168） | v010 | 交换后 |
|---|---:|---:|
| 前 14 个 block 对应 token 数 | 14 | 1 |
| 前 14 个 block 涉及不同 x 行数 | 112 | 8 |
| 前 14 个 block 必需 x 读取量 | 114,688 B | 114,688 B |
| 地址跨度均值 | 19.49 MB | 1.39 MB |
| 地址跨度中位数 | 16,930,816 B | 1,024 B |
| 相邻编号对应同 token 的比例 | 0% | 92.87% |

[真实 positions](../raw/profile_positions_h7168_v010.json)、[交换版 positions](../raw/profile_positions_h7168_grid_hidden_first.json) 两份一致；[地址统计与计算口径](address_locality_16g.json)。

前 14 个 CTA 仍读取 114,688 字节，只是从 112 行的各一段，变成 8 行各自的 14 个连续段。读取的是不同列，**没有复用同一个 x 元素，也没有少读 expert 输出**。

重复的是 metadata：每 CTA 都读取 8 个 position 和 8 个 weight，逻辑上共 64 字节。前 14 个 CTA 的这部分逻辑读取仍为 896 字节，但涉及的唯一 metadata 从 14 个 token 的 896 字节变成一个 token 的 64 字节。这提供了 cache 重用条件；不等于已经测出了 cache 命中率提高。Quantized 的 x_sf 同样会在同 token 的不同 hidden 块间重复读取。

### 3. 每个 warp 的读取和计算保持原样

h7168 生成代码每线程读取 4 个 BF16=8 字节，一个 64 线程 warp 对应连续的 512 字节；每线程写 4 个 FP8。**原来的 warp 读取就已经连续。**此次没有扩大向量宽度，改变的是不同 CTA 对应的行和列范围。

两版 CTA 总数、128 线程、tile、K 累加顺序、FP8 编码都不变。八组生成 C++ 在对调 `blockIdx.x/y` 后逐字一致，没有增加重排、shared 往返或减少算术工作。这里比较的是生成 C++，不是最终机器指令。

### 4. 不同 workload 的变化

| workload | 旧 grid → 新 grid | CTA 总数 | 相邻编号同 token 的比例：旧 → 新 | 同 token/槽位的相邻块起点差 |
|---|---|---:|---:|---:|
| tiny | (32, 1) → (1, 32) | 32 | 0% → 0.00% | 512 B |
| h3072 | (512, 6) → (6, 512) | 3,072 | 0% → 83.36% | 1,024 B |
| h7168 | (512, 14) → (14, 512) | 7,168 | 0% → 92.87% | 1,024 B |
| prefill | (4096, 7) → (7, 4096) | 28,672 | 0% → 85.72% | 2,048 B |

[映射计算](grid_mapping_16g.json)。比例描述编号映射，不是实际 AP 调度顺序。

- h7168 的 14 块使相邻编号更集中到同 token；本轮两种变体耗时下降约 3.3%。
- prefill 每 token 7 块，仍出现约 2.2%～2.7% 的下降。
- h3072 每 token 6 块，编号映射同样更集中，却慢约 0.4%。地址集中不是保证收益的充分条件；并发请求的分布和调度代价也可能改变。
- tiny 每 token 只有一块，展平编号对应的 `(token,hidden块)` 序列两版完全相同，没有上述地址集中变化。因此不能用这个解释归因 tiny 的小幅正负差异。

### 5. 相同数据量下，完成速度提高了多少

只计算必需 x 读取与 out 写回（不含 metadata）：`数据量=T*K*H*2+T*H*1`，有效吞吐=数据量/benchmark 耗时。
这是算法数据量推导的有效吞吐，**不是实测 HBM 吞吐或利用率**。两版的分子相同。

| 变体/workload | 必需 x+out MB | v010 有效 TB/s | 交换后有效 TB/s | 有效吞吐提高 |
|---|---:|---:|---:|---:|
| fp8/h7168 | 62.390 | 1.289 | 1.333 | 3.38% |
| fp8/prefill | 499.122 | 1.407 | 1.439 | 2.24% |
| quantized/h7168 | 62.390 | 1.280 | 1.323 | 3.36% |
| quantized/prefill | 499.122 | 1.393 | 1.432 | 2.80% |

### 6. mcProfiler 采集结果与证据边界

对 FP8 h7168 做了前后采集，但得到的计数不能作为可靠的原因证据。两版必需 x 读取均为 58,720,256 B，输出均为 3,670,016 B；而采集出现如下矛盾：

| 采集 | Global Memory Read B | Global Memory Write B | 处理 |
|---|---:|---:|---|
| 七指标 v010 | 653,916,928 | 150,976,320 | 不用于收益归因 |
| 最小五指标 v010 | 59,862,656 | 3,670,368 | 不用于收益归因 |
| 最小五指标交换版 | 60,317,856 | 189,213,344 | 不用于收益归因 |

七指标采集的读写均远超目标数据量；改用此前验证过的五指标最小采集后，交换版仍报告约 189 MB 写出，约为实际输出的 51.6 倍。退出码为 0 不代表计数有效。可能混入其它工作或计数窗口异常，具体原因尚未定位。没有使用这些数据计算实际 HBM 带宽，也没有用异常 stall 计数解释性能。

[七指标原始结果](../raw/profile_h7168_v010/tool_output/1_reduce_fused_kernel_kernel_dumped_result.json)、[最小 v010 结果](../raw/profile_minimal_h7168_v010/tool_output/1_reduce_fused_kernel_kernel_dumped_result.json)、[最小交换版结果](../raw/profile_minimal_h7168_grid_hidden_first/tool_output/1_reduce_fused_kernel_kernel_dumped_result.json)、[采集命令和退出状态](../meta/profile_minimal_status.json)。

### 7. 当前能说明的收益原因

**已经确认：**这次改动改变了 block 编号对应的访存任务，显著集中同 token 的输入行与 metadata；没有减少必需 x 流量或改变生成 C++ 中的计算/向量读取结构。配对 benchmark 在 h7168/prefill 上测出了小幅中位数收益。

**合理解释：**若邻近编号的 CTA 在相近时间执行，更集中的请求可能改善内存系统的服务效率，并有利于 metadata cache 重用。这是实现与地址数据支持的解释方向。

**尚未确认：**不能确定收益具体由 L2 命中、地址转换、内存通道/分区分布或其它调度因素贡献了多少；也没有有效计数证明访存等待下降。因此不能把“cache 命中率提高”或“HBM 流量减少”写成已验证结论。

### 8. mcTracer：资源占用没有变化

同进程采集两版 FP8 h7168，每版 10 warmup+1 ROI，共 11 个目标事件。

| 指标 | v010 | 交换 grid |
|---|---:|---:|
| grid | 512×14 | 14×512 |
| CTA 总数 | 7168 | 7168 |
| threads/CTA | 128 | 128 |
| registers/thread | 12 | 12 |
| static/dynamic shared | 0/0 | 0/0 |
| private_total | 0 | 0 |

[原始 trace](../raw/trace_grid/grid-149790.json)、[资源汇总](trace_grid_summary_16g.json)。资源字段确认没有寄存器、shared 或 private 的减少；不使用 mcTracer 的 occupancy 字段推导实际驻留。trace 未沿用 benchmark 的逐次 L2 flush/输入克隆协议，辅助时间不用于替代配对 benchmark。

## 正确性与接入状态

现有 131 项正确性测试全部通过（80.45 秒），failures/errors/skipped 均为 0；见 [JUnit 原始结果](../raw/correctness.xml)。八组 benchmark 另有逐字节对照。正式 kernel 已按下述范围接入，目录改为 ACC。

八组生成 C++ 在交换 `blockIdx.x/y` 后逐字完全相同；除 block 索引绑定外，没有其它生成源改动。本轮未比较最终机器指令。

## 正式接入（2026-10-06）

外层 `MoeReduceFusedKernel` 选择 `grid_hidden_first`，传入 JIT 工厂；该 bool 在编译时决定 grid 和 block 索引绑定。
仅加权 FP8/Quantized、BF16、K=8、H=7168 的以下默认配置启用：

| T | tile/threads | grid |
|---|---|---|
| 512（h7168） | 512/128 | (14,512) |
| 4096（prefill） | 1024/128 | (7,4096) |

Base/XSF、tiny、h3072、其它 shape/dtype、无权重路径以及不同的手动 tile/threads 几何使用原 grid。直接调用 JIT 工厂时，新增参数 `grid_hidden_first` 默认 False，保留原调用行为。

本报告的八组性能数据和 131 项正确性结果属于前面的实验采集；此次接入仅做源码检查，未重跑测试或 benchmark。benchmark 脚本已固定加载 `codegen/baseline_v010.py`，避免正式接入后把新 grid 当作旧基线。

## 文件

- [完整样本与结果](../raw/comparison.csv)
- [唯一源码改动](../codegen/grid_order.patch)
- [生成代码对比](source_comparison.json)
- [benchmark 脚本](../scripts/benchmark.py)
- [现有测试入口](../scripts/validate.py)
- [运行状态](../meta/status.json)

在仓库根目录加载 `source /data/sc16g-recovery-20261004/env.sh`，再运行 `python profiles/v020_ACC/scripts/benchmark.py --variant fp8 --workload h7168` 可复现单组。脚本会覆盖该组 CSV；重测前请另存已有结果。
