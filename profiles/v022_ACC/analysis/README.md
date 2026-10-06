# v022：向量化读写检查与独立 FP8 写回（sc-16g）

## 1. 上版本遗留问题

实验基线为 v020_ACC；本版本现已接入正式 kernel。需要确认 TileLang/MXCC 是否已经向量化，以及是否还存在可以合并的全局读写。
本轮检查四变体、四 workload 的 16 份正式生成 C++；另生成 8 份 FP8/Quantized 候选，选取代表情况导出 MXCC 最后的机器 IR。实验采集时未改正式 kernel；接入结果见末尾。

## 2. 问题原因分析：编译器已经做了什么

下表宽度指**每线程一次访问**，不是整个 warp 的合并事务大小。Base/XSF 的 BF16 大 workload 每线程处理 12 或 28 个元素，因此会重复多次 8 字节访问。

| 路径 | workload | 输入读取 | 输出写回 | 设备机器 IR 证据 |
|---|---|---|---|---|
| Base/XSF | tiny | 2 B，1 个 FP16 | 2 B，1 个 FP16 | Base：`LDG_U16` / `STG_B16_SADDR` |
| Base/XSF | h3072/h7168/prefill | 8 B，4 个 BF16 | 8 B，4 个 BF16 | h7168：`LDG_B64` / `STG_B64` |
| FP8/Quantized | tiny | 4 B，2 个 FP16 | 2 B，2 个 FP8 | FP8：`LDG_B32` / `STG_B16` |
| FP8/Quantized | h3072/h7168 | 8 B，4 个 BF16 | 4 B，4 个 FP8 | `LDG_B64` / `STG_B32` |
| FP8/Quantized | prefill | 16 B，8 个 BF16 | **两次 4 B**，共 8 个 FP8 | `LDG_B128` / **2×`STG_B32`** |

结论：输入和大部分输出已经向量化；并不是每个元素都各发一条全局读写指令。prefill 的输入一次读 16 字节，输出却分两次写 4 字节，是本轮的具体改进目标。

TileLang 的循环向量化与 MACA codegen 源码位置、哈希和相关行记录在 [编译器源码证据](../meta/compiler_sources.json)。MACA codegen 也有 256 位 global load/store 的 helper 路径；不能把当前看到的 128 位读取直接当成硬件宽度上限，本轮没有验证 256 位对应几条硬件指令。

## 3. 本版本解决方案

将 FP8 编码和最终 global 写回分开：先把本线程的编码结果放入 `encoded_fragment`，再由 `T.copy` 整体写出。

```python
encoded_fragment = T.alloc_fragment((tile_hidden,), out_dtype)
# 原 FP8 编码逻辑、K 顺序不变
encoded_fragment[i] = T.reinterpret(final_encoded.astype("uint8"), out_dtype)
T.copy(encoded_fragment, out[pid_token, start:end], coalesced_width=copy_width)
```

这里的 fragment 是寄存器中的结果，不是 shared 中转。prefill 每线程有 8 个 FP8，因此可以一次写 8 字节；h3072/h7168 每线程只有 4 个，tiny 只有 2 个，宽度分别限定为 4/2。强行全部指定 8 会在 tiny 上触发布局错误，失败源码和状态已保留。

## 4. 具体落地与编译验证

- [候选源码](../scripts/vector_store.py)：保留正式 v020 的 grid、tile、threads、路由判断、累加顺序与 FP8 数值逻辑。
- [编译驱动](../scripts/compile_candidates.py)：生成 C++，再用 MXCC `-O3 -print-after=stack-frame-layout` 导出最后机器 IR；这是最终后端机器 IR，不是二进制反汇编。
- prefill 基线是两条 `STG_B32`，候选是一条 `STG_B64`；两版均有 8 条静态展开的 `LDG_B128` 输入读取。静态指令出现次数不能直接当全 kernel 的动态计数。
- h7168 候选仍是 `LDG_B64`/`STG_B32`，没有扩大实际读写宽度。

### MXCC 会不会自己合并两次写回？

会。本轮增加了一个**仅静态编译**的控制：保留候选中先编码全部 8 个结果的结构，把最后一次 8 字节 C++ 写回人为写成相邻两次 4 字节写回。MXCC 最后的机器 IR 仍生成一条 `STG_B64`。

因此，后端具备自动合并能力。原来的结构是“编码 4 个→写 4 字节→再编码 4 个→再写 4 字节”；改成“先编码 8 个→集中写回”后，编译器有机会合并。候选还改变了编码循环的展开方式：基线的内层 vec 循环没有显式 unroll，候选的 8 元素编码循环有 `#pragma unroll`。**不能把全部性能收益都归因于 store 宽度。**

[静态控制与结果](store_merge_control_16g.json)、[控制 C++](../codegen/fp8_prefill_store32_control.cu)、[控制机器 IR](../codegen/fp8_prefill_store32_control_final.mir)。此 C++ 控制未用于 GPU benchmark，实际候选全部来自 TileLang 源码。

### 每个 warp 的写回地址也更集中

prefill 每线程负责 8 个 FP8。一个 64 线程 warp 对应连续 512 字节输出：

```text
旧第一次 STG_B32：thread t 写 [8t, 8t+3]
旧第二次 STG_B32：thread t 写 [8t+4, 8t+7]
新一次   STG_B64：thread t 写 [8t, 8t+7]
```

旧的每次 store 写 256 字节有效数据，分散在 512 字节范围中；新的一次写满连续 512 字节。全 kernel 输出字节数没有减少。是否减少真实 L2/HBM 事务，还需要有效计数，不能仅凭这个地址模型断言。

## 5. Benchmark 对比

sc-16g，当前容器与 GPU 快照见 meta；正式基线为 v020。确认 v021 Roofline 已完成、无本实例其它采集进程后顺序测量。
输入 seed=1235；tiny 为 T32/K2/H256/FP16，h3072/h7168 为 T512/K8/BF16，prefill 为 T4096/K8/H7168/BF16。tile/threads 分别为 256/128、512/128、512/128、1024/128。

项目 `bench_kernel`：10 warmup、50 repeat×3 trials、CUPTI、逐次 L2 flush；外层五轮交替先后顺序取中位数。tiny/h3072/h7168 克隆输入，prefill 按项目阈值不克隆。所有组计时前与正式基线逐字节一致。

| 变体 | workload | v020 μs | 候选 μs | 加速比 | 耗时下降 |
|---|---|---:|---:|---:|---:|
| fp8 | tiny | 3.953 | 3.758 | 1.0518× | +4.92% |
| fp8 | h3072 | 22.794 | 23.526 | 0.9689× | -3.21% |
| fp8 | h7168 | 47.739 | 48.742 | 0.9794× | -2.10% |
| fp8 | prefill | 355.651 | 338.417 | 1.0509× | +4.85% |
| quantized | tiny | 16.676 | 15.939 | 1.0463× | +4.42% |
| quantized | h3072 | 22.984 | 23.562 | 0.9754× | -2.52% |
| quantized | h7168 | 47.084 | 48.026 | 0.9804× | -2.00% |
| quantized | prefill | 355.656 | 342.221 | 1.0393× | +3.78% |

量化 tiny 首轮前两轮约 4 μs，后三轮两版都升到约 16 μs；保留所有样本，不能据此把中位数差异当成稳定优化。初次 tiny 进程启动时遭 SIGKILL，容器 OOM 计数增加，随后正常补测。其它六组结果没有重跑或覆盖。

### 独立进程复测

prefill 重新生成输入，在独立进程用同样五轮协议复测。tiny 在另一个新进程内依次复测两变体。初轮与复测文件分开保存。

| 变体 | workload | 复测 v020 μs | 复测候选 μs | 加速比 | 耗时下降 |
|---|---|---:|---:|---:|---:|
| fp8 | tiny | 3.942 | 3.763 | 1.0476× | +4.55% |
| fp8 | prefill | 359.342 | 338.109 | 1.0628× | +5.91% |
| quantized | tiny | 3.937 | 3.907 | 1.0079× | +0.78% |
| quantized | prefill | 354.468 | 340.035 | 1.0424× | +4.07% |

[初轮八组完整样本](../raw/comparison_16g.csv)。复测：[FP8 prefill](../raw/confirmation_fp8_prefill.csv)、[Quantized prefill](../raw/confirmation_quantized_prefill.csv)、[FP8 tiny](../raw/confirmation_fp8_tiny.csv)、[Quantized tiny](../raw/confirmation_quantized_tiny.csv)。

## 6. 指令与资源变化

prefill 原生 SDK module 属性与 occupancy API 查询结果：

| 指标 | 正式 v020 | 写回候选 |
|---|---:|---:|
| 输入指令宽度 | LDG_B128（16 B） | LDG_B128（16 B） |
| 输出指令 | 2×STG_B32 | 1×STG_B64 |
| registers/thread（两变体） | 20 | 20 |
| private bytes | 0 | 0 |
| static/dynamic shared | 0/0 | 0/0 |
| 原生 API 允许的 CTA/AP | 16 | 16 |

没有增加 private、shared 或寄存器数；原生资源容量上限未降低。CTA/AP 是资源允许上限，不是实际驻留时间线。SDK 通用 header occupancy 查询仍返回 status=2，不用它的 0 值作结论。

[机器读写指令](machine_vectorization_16g.json)、[原生资源查询](../raw/resources_16g.json)。没有使用 mcProfiler 的不可靠跨窗口计数，也没有把减少一条 store 等同于实际 HBM 字节减少。

现有 `tests/ops/test_moe_reduce_fused.py`：**131 passed，75.43 秒**，failures/errors/skipped 均为 0；[JUnit](../raw/correctness.xml)。

## 7. 实验总结

- 编译器已经自动完成了输入向量读取和多数输出向量写回，不能简单再把所有访问改成“向量化”就获得收益。
- prefill 的两种 SF 路径有可复现收益：首轮耗时下降 4.85%/3.78%，独立复测下降 5.91%/4.07%。更宽 store 已在机器 IR 中确认，但编码循环展开也改变，收益不能完全归因于 store。
- FP8 tiny 首轮/复测下降 4.92%/4.55%，但写回宽度仍为 2 B，应视作编码与写回代码组织变化的收益；它的绝对节省约 0.18～0.19 μs。
- Quantized tiny 复测仅下降 0.78%，首轮还有明显环境阶段变化，暂不视为稳定收益。
- h3072/h7168 的两种变体均慢约 2%～3%，不推广到所有 SF workload。
- 实验阶段推荐 prefill 两变体和 FP8 tiny；用户随后明确要求将 Quantized tiny 也接入。现已接入四个组合，目录标记 ACC，接入验证见下节。

复现：加载 `source /data/sc16g-recovery-20261004/env.sh`，运行 `python profiles/v022_ACC/scripts/benchmark.py --variant fp8 --workload prefill`；完整现有测试入口为 `scripts/validate.py`。测速脚本会覆盖单组 CSV，重测前请另存旧结果。

## 补充：单条 global 向量读写宽度

静态编译 64/128/256/512 位复制，在当前 C500/xcore1000、MXCC -O3 编译路径得到：

| 源码每线程数据宽度 | 机器 IR load | 机器 IR store |
|---|---|---|
| 64 bit | 1×LDG_B64 | 1×STG_B64 |
| 128 bit | 1×LDG_B128 | 1×STG_B128 |
| 256 bit | 2×LDG_B128 | 2×STG_B128 |
| 512 bit | 4×LDG_B128 | 4×STG_B128 |

在这条普通全局读写编译路径中，128 bit=16 B/线程可由单条访存指令完成；更宽源码向量拆成多条。它描述此工具链的实际降低结果，不是对所有内存空间和特殊指令的完整 ISA 上限声明。

[探针源码](../codegen/vector_width_probe.cu)、[最后机器 IR](../codegen/vector_width_probe_final.mir)、[指令汇总及编译命令](vector_width_probe_16g.json)。未运行 GPU kernel。

## 正式接入：tiny/prefill × FP8/Quantized

用户明确要求四个组合全部接入，包括此前收益较小的 Quantized tiny。正式文件为 `tileops/kernels/moe/reduce_fused.py`：

- JIT 工厂增加 `vector_store=False` 编译参数。直接调用未传该参数时保留原写回；开启时编码到 fragment 后 `T.copy`。
- 包装层 `MoeReduceFusedKernel.vector_store` 仅对带权重、SF=true 的 `(T32,K2,H256,FP16)` 和 `(T4096,K8,H7168,BF16)` 开启。两个 with_x_sf 状态都开启，覆盖 FP8/Quantized。
- 必须使用已测 tile/threads（tiny256/128，prefill1024/128）。自定义几何、未测 shape/dtype、无权重都回退。
- 保留 v020 的 grid 选型：prefill 使用 hidden-first，tiny 使用原顺序。

16 组自动分派检查选中恰好 4 组；其余 12 组的生成 C++ 与接入前逐字一致。另有 5 个回退配置检查。

### 接入后与 v020 的配对结果

同样为五轮中位数、项目 CUPTI/L2 flush 协议，八组均逐字节一致。

| 变体 | workload | 启用 vector_store | v020 μs | 正式 v022 μs | 耗时下降 |
|---|---|---|---:|---:|---:|
| fp8 | tiny | True | 4.081 | 3.738 | +8.41% |
| fp8 | h3072 | False | 22.804 | 22.830 | -0.11% |
| fp8 | h7168 | False | 46.551 | 46.536 | +0.03% |
| fp8 | prefill | True | 346.097 | 334.449 | +3.37% |
| quantized | tiny | True | 3.891 | 3.717 | +4.47% |
| quantized | h3072 | False | 22.938 | 22.994 | -0.25% |
| quantized | h7168 | False | 46.956 | 47.386 | -0.92% |
| quantized | prefill | True | 347.878 | 335.718 | +3.50% |

未选中的 h3072/h7168 代码完全相同，其小幅耗时差不作为代码改动的收益或回退。Quantized prefill 进程曾在导入阶段 SIGKILL，失败日志和状态保留；随后在单进程内顺序完成剩余 benchmark 和测试，减少重复导入。

正式接入后的现有测试：**131 passed，37.05 秒**，failures/errors/skipped 为 0。

[正式源码快照](../codegen/integrated_formal.py)、[接入前 v020](../codegen/pre_integration_formal.py)、[分派检查](../meta/integration_dispatch.json)、[八组完整样本](../raw/integrated_comparison_16g.csv)、[正式测试 JUnit](../raw/integrated_correctness.xml)、[运行状态](../meta/integration_status.json)。
