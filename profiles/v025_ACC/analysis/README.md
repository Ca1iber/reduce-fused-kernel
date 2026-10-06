# v025：当前v022与8行寄存器预取组合（sc-16g）

## 1. 目标

v016曾在FP8/Quantized的h3072/h7168上确认寄存器预取收益，但尚未接入。当前正式v022已包含v020的h7168 hidden-first grid，因此重新验证预取收益与grid优化能否叠加。

## 2. 改动

基于当前正式源码，增加输入dtype的寄存器fragment `[8,tile_hidden]`：先读取8个expert输入，再按原K顺序加权累加。仍判断pos>=0，无效路由不读x也不消费该slot；保留当前FP8编码。

这是普通global→register的软件预取，不是global→shared硬件async，没有shared或ticket。K=8时全部8行预先组织读取。源码表达了加载阶段前移，不能仅凭源码顺序断言最终硬件严格先完成全部读取再执行首个FMA。

## 3. 对照与配置

四组：FP8/Quantized × h3072/h7168。均为T512、K8、BF16、tile512、128线程；与v022几何、数据类型和数学顺序相同。

- h3072：v022为token-first grid，候选保留此grid。
- h7168：v022为hidden-first grid，候选保留此grid；另测“预取＋旧token-first grid”控制。
- 当前v022在这四组没有启用独立vector_store，候选也保持该状态；本轮不改变tiny/prefill。

[候选TileLang](../scripts/prefetch_combined.py)、[固定v022基线](../codegen/baseline_v022.py)。实验阶段保留固定基线；正式接入见第8节。

## 4. Benchmark

项目bench_kernel：10warmup、50repeat×3trials、CUPTI、逐次L2flush、输入克隆；外层五轮轮换/反转顺序取中位数。seed=1235；全部输出逐字节与v022一致。另一个新进程重新生成输入，按同协议独立复测四组。

| 变体 | workload | 初轮v022 μs | 初轮组合 μs | 复测v022 μs | 复测组合 μs | 复测加速比 | 复测耗时下降 |
|---|---|---:|---:|---:|---:|---:|---:|
| fp8 | h3072 | 22.784 | 21.156 | 22.789 | 21.125 | 1.0788× | 7.30% |
| quantized | h3072 | 22.968 | 22.308 | 22.994 | 22.282 | 1.0319× | 3.10% |
| fp8 | h7168 | 46.669 | 43.156 | 46.648 | 43.228 | 1.0791× | 7.33% |
| quantized | h7168 | 46.971 | 44.938 | 46.950 | 44.974 | 1.0439× | 4.21% |

[初轮全部样本](../raw/comparison_16g.csv)、[独立复测全部样本](../raw/confirmation_comparison_16g.csv)、[benchmark脚本](../scripts/benchmark.py)。两轮分开保存，没有挑选或合并较快样本。

## 5. h7168：两项优化能否叠加

用同一次独立复测中的旧grid预取控制比较：

| 变体 | v022（当前grid，无预取） μs | 预取＋旧grid μs | 预取＋当前grid μs | 当前grid在预取基础上的耗时下降 |
|---|---:|---:|---:|---:|
| fp8 | 46.648 | 44.334 | 43.228 | 2.49% |
| quantized | 46.950 | 45.537 | 44.974 | 1.24% |

本轮两项可以叠加：当前grid＋预取均快于只有预取的旧grid。这里来自同轮对照，不把v016旧数据与v020旧数据的加速比直接相乘。

## 6. 指令、资源与正确性

| 指标（h3072/h7168，已查询组合） | v022 | 预取组合 |
|---|---:|---:|
| registers/thread | 12 | 22 |
| private bytes | 0 | 0 |
| static/dynamic shared | 0/0 | 0/0 |
| 原生API允许CTA/AP | 16 | 16 |
| x读取指令宽度 | LDG_B64 | LDG_B64 |
| 输出写回 | STG_B32 | STG_B32 |

K=8的机器IR仍有8条静态展开的LDG_B64与1条STG_B32；输入和输出必需字节数未减少。这里的静态出现次数不等于全kernel动态计数。元数据的均匀load组合也有变化。

寄存器增加但没有private/shared，原生资源容量未降低；该CTA/AP值是资源允许上限，不是实测驻留时间线。本轮没有用mcProfiler等待计数精确拆解收益。

[原生资源和最后机器IR指令](resources_and_instructions_16g.json)。

现有测试**131 passed**，failures/errors/skipped为0。验证脚本把全部SF路径换成候选，实际构造候选SF kernel共63次，覆盖FP16/BF16/FP32、不同K、无效路由等原有测试条件；Base/XSF回退正式实现。这避免了仅运行基线回退却声称测试候选的情况。

[JUnit](../raw/correctness.xml)、[候选覆盖计数和参数](../meta/validation.json)、[验证脚本](../scripts/validate.py)。初次按目标shape过滤的验证另外保留，最后结论使用扩大SF覆盖后的结果。

## 7. 结论

四个目标组合均有可复现收益：独立复测耗时下降约7.3%、3.1%、7.3%、4.2%。h7168上，寄存器预取与当前grid优化可以叠加。

已将这四个组合接入正式kernel，目录改为ACC，接入验证见第8节。复现命令：加载恢复环境后运行 `python profiles/v025_ACC/scripts/benchmark.py --variant fp8 --workload h7168`；四组独立复测用 `scripts/confirm.py`。脚本会覆盖单组文件，重测前另存结果。

## 8. 正式接入验证

正式入口 `tileops/kernels/moe/reduce_fused.py` 新增 `prefetch_rows`，默认0；wrapper仅对加权FP8/Quantized、BF16、T512/K8/H3072或H7168且tile512/threads128时选择8。保留grid、vector_store和K累加顺序。其他shape、dtype、无权重或手动改变几何均关闭预取。

16组自动分派检查中恰好4组启用预取；其余12组生成的C++与接入前逐字一致。额外5组回退条件检查通过。正式入口运行现有测试：**131 passed**，0 failures/errors/skipped。8组SF输出均与接入前逐字节一致。

接入后重新按相同CUPTI/L2flush协议配对测试，目标四组如下：

| 变体 | workload | 接入前 μs | 正式接入 μs | 耗时下降 |
|---|---|---:|---:|---:|
| fp8 | h3072 | 22.748 | 21.089 | 7.29% |
| fp8 | h7168 | 46.705 | 43.203 | 7.50% |
| quantized | h3072 | 22.994 | 22.354 | 2.78% |
| quantized | h7168 | 47.063 | 45.123 | 4.12% |

[正式接入8组配对结果](../raw/integrated_comparison_16g.csv)、[正式测试JUnit](../raw/integrated_correctness.xml)、[自动分派检查](../meta/integration_dispatch.json)、[接入验证脚本](../scripts/integrate.py)、[正式源码快照](../codegen/integrated_formal.py)。
