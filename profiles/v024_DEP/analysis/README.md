# v024：128-bit FP8 写回尝试（sc-16g）

## 1. 目标与配置

以已接入的 v022 为基线，尝试每线程一次写 16 个 FP8，即 128 bit。
大 workload 使用 tile1024、64线程，因此每线程有16个结果；保留各 workload 原来的 grid 编号方向。tiny 使用 tile256/64线程，每线程只有4个结果，实际为32-bit写回，不能把它标成真正128-bit。

每组比较三条路径：当前 v022；相同 tile/threads/显式连续布局的直接写回控制；先编码到寄存器 fragment 再整体写回的候选。

## 2. 第一版：仅改线程数与 copy_width 没有真正得到128-bit

把 copy_width 上限从8改成16，线程从128减到64后，每线程确实有16个结果，但自动布局把它们分成两个不相邻的8元素片段。
prefill 的 thread0 拿到列0～7与512～519，thread1拿到8～15与520～527，所以仍生成2×STG_B64。指令核验为16×LDG_B128、2×STG_B64；36寄存器/线程，private=0。

这是“有16个值”与“有16个连续值”的区别。第一版的数据和代码保留在 [自动布局原始结果](../raw/auto_layout/comparison_16g.csv)、[自动布局资源与指令](auto_layout_resource_16g.json)，没有把它当成128-bit方案的性能。

## 3. 真正128-bit：显式安排连续16个元素

在 reduced_fragment 与 encoded_fragment 上使用相同的 fragment 布局：

```python
elements_per_thread = tile_hidden // num_threads
T.Fragment(
    (tile_hidden,),
    forward_thread_fn=lambda i: i // elements_per_thread,
    forward_index_fn=lambda i: i % elements_per_thread,
)
```

对大 workload，thread0持有列0～15，thread1持有16～31，以此类推。再用 coalesced_width=16 的 T.copy 写回。
生成 C++ 的目的地址为 `out + row_offset + threadIdx.x*16`，使用 `fp8_e4_16_t`；MXCC最后机器IR确认**1×STG_B128**。输入32字节/线程/expert拆成2×LDG_B128，K=8共16条静态展开的LDG_B128。

[候选源码](../scripts/store128_contiguous.py)、[prefill生成C++](../codegen/fp8_prefill_store128.cu)、[最后机器IR](../codegen/fp8_prefill_store128_final.mir)、[静态确认](contiguous128_static.json)。

## 4. 配对 benchmark

FP8/Quantized全部四个workload，共8组，每组3个kernel，共24行。seed=1235；tiny为T32/K2/H256/FP16，h3072/h7168为T512/K8/BF16，prefill为T4096/K8/H7168/BF16。
项目bench_kernel：10warmup、50repeat×3trials、CUPTI、L2flush，外层5轮轮换/反转顺序取中位数。tiny/h3072/h7168克隆输入，prefill按项目阈值不克隆。全部输出与v022逐字节一致。

| 变体 | workload | v022 μs | 同几何直接写回 μs | 候选 μs | 候选/v022加速比 | 候选慢于v022 |
|---|---|---:|---:|---:|---:|---:|
| fp8 | tiny | 3.763 | 4.403 | 3.932 | 0.9570× | +4.49% |
| fp8 | h3072 | 22.758 | 25.262 | 23.516 | 0.9678× | +3.33% |
| fp8 | h7168 | 46.582 | 57.533 | 48.236 | 0.9657× | +3.55% |
| fp8 | prefill | 334.244 | 450.514 | 347.704 | 0.9613× | +4.03% |
| quantized | tiny | 3.758 | 4.342 | 3.881 | 0.9683× | +3.27% |
| quantized | h3072 | 22.932 | 25.457 | 23.547 | 0.9739× | +2.68% |
| quantized | h7168 | 46.894 | 58.260 | 47.764 | 0.9818× | +1.86% |
| quantized | prefill | 335.749 | 456.463 | 349.036 | 0.9619× | +3.96% |

**本轮候选没有超过当前v022。**prefill两变体慢约4%，其它组合也慢。tiny这一行仅是64线程/32-bit写回的对照，不是128-bit。
同几何控制也使用相同连续布局；直接写回的prefill为4×STG_B32，候选为1×STG_B128，候选明显快于这个控制。但循环展开和编码组织也改变，不能把这部分差异全部归因于store宽度。

[24行结果与全部五轮样本](../raw/comparison_16g.csv)、[benchmark入口](../scripts/benchmark_contiguous.py)。

## 5. 指令、寄存器与容量

| prefill指标（两变体） | v022 | 同几何直接控制 | 真128-bit候选 |
|---|---:|---:|---:|
| tile/threads | 1024/128 | 1024/64 | 1024/64 |
| 每线程结果数 | 8 | 16 | 16 |
| 每expert输入load | 1×LDG_B128 | 2×LDG_B128 | 2×LDG_B128 |
| 每线程输出store | 1×STG_B64 | 4×STG_B32 | 1×STG_B128 |
| registers/thread | 20 | 34 | 34 |
| private bytes | 0 | 0 | 0 |
| static/dynamic shared | 0/0 | 0/0 | 0/0 |
| 原生module API返回CTA/AP容量 | 16 | 32 | 32 |

[原生API和机器指令原始结果](resources_and_instructions_16g.json)。没有private/shared中转，也没有证据称寄存器溢出。

**容量口径限制：**SDK device property 的 max_blocks_per_ap字段仍为16，但原生module occupancy API对64线程返回32；保留这两条原始信息，未做64线程的长驻留探针来裁决实际最大CTA。这里的32是API返回值，不是实际驻留时间线。按该API估算，128×16与64×32均为2048线程，即32个64线程warp，不能因为线程数减半就断言resident warp容量减半。

h3072候选只有512×3=1536个CTA，即使全部同时存活，全实例也只发出1536个warp；v022有512×6=3072个CTA、每CTA两个warp。候选grid更小，属于可能影响并发覆盖的几何变化。prefill/h7168的grid足够大，不能用同一解释直接归因。

## 6. 为什么更宽不一定更快

输出总字节数和归约数学没有减少。为了让16个FP8连续归属于同一线程，线程数、寄存器存活集合以及输入请求组织都改变了。
prefill基线每warp的128-bit输入load对应连续1024字节；候选每warp的一次load取每线程16元素中的前8个，读取1024字节有效数据，分散在2048字节范围中，再由下一次load读取后8个。地址范围的密度从100%变成50%，但是否增加实际L2/HBM事务，未采集有效计数，不能断言。
同样，寄存器20→34、每线程编码8→16、每CTA warp数2→1都可能影响调度/依赖；资源允许容量没有显示下降，也没有实测stall数据区分各项贡献。

因此本轮证明“真实128-bit写回可以生成且计算正确”，并证明**1024/64＋连续16布局这组方案不如v022**，不代表所有128-bit方案都无效。

## 7. 正确性与结论

候选SF路径运行现有测试，Base/XSF回退正式实现：**131项全部通过**，failures/errors/skipped为0。八组另有逐字节对照。自动布局第一版同样通过131项，但实际写回为64-bit，不混用两版数据。

[JUnit](../raw/correctness.xml)、[验证入口](../scripts/validate_contiguous.py)、[运行状态](../meta/status.json)。正式kernel保持v022，v024保留为DEP。

复现：加载 `source /data/sc16g-recovery-20261004/env.sh` 后运行 `python profiles/v024_DEP/scripts/run_contiguous.py`。旧的 `run.py` 对应第一版自动布局。两者会写同名采集文件，重测前请另存已有结果。
