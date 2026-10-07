# v029：tiny 的 CTA、线程和任务分配实验（sc-16g）

## 1. 问题与假设

tiny固定T32/K2/H256、FP16。当前正式v025每token一个CTA，共32CTA；Base/XSF为256线程，FP8/Quantized为128线程。v028测得同几何空kernel约2 μs，完整tiny约3.6～3.9 μs。

假设：一个CTA处理多个token或减少线程，可能降低设备派发/调度成本；代价是更少CTA并行、每线程更多元素和指令。空kernel比例不能证明减少CTA一定有效，因此使用完整kernel测量判断。

## 2. 实现与控制组

- **formal_threads**：直接使用固定v025工厂，仅改变threads=32/64/128/256，保持32CTA及tiny原有写回路径。
- **static_group**：tokens/CTA=1/2/4/8，对应32/16/8/4CTA；每种threads=32/64/128/256。显式二维fragment布局分配token与连续列，位置/权重按token复制到负责该token的线程。
- **local_vector**：同样的token/线程分配，但明确使用每线程local数组和Vectorized加载/写出。K索引和元素索引展开，避免把标量访存代码生成差异误判成CTA数量的效果。该前端local数组是否成为物理private存储，由原生资源查询验证。

两个新实现均保留K累加顺序、负位置屏蔽及v003的FP8编码，无shared或async。性能版本只针对T32完整tile；正确性验证还使用同模板的动态token尾部保护版本。

初版动态尾部版本保留在 `codegen/tiny_group_initial.py` 与 `raw/guarded_initial_sweep_16g.csv`。初版FP8临时变量与输出fragment重名导致解析失败，已修正；原失败日志和状态保留。随后完成80组正式线程/固定shape扫描、32组显式向量实现扫描，最终两轮共112组均通过五种路由的逐字节核对。初版启动中断另保存在日志中，不据此推断硬件或性能问题。

## 3. 测量协议

同进程、相同输入、预分配输出；seed=1235，性能用random路由。每个配置另核对identity、padded、duplicates、all-invalid，共五种路由，与固定v025输出逐字节一致。

项目bench_kernel：CUPTI、L2flush、输入克隆、10warmup、50repeat×3trials，外层五轮交替两版顺序。扫描后对最佳、最佳较少CTA和最佳较少线程候选重新测20个配对轮次；保存各轮原始样本，按配对轮次均值差bootstrap 5000次得到95%参考区间。中位耗时是轮次统计，不是单次调用P50；该区间描述本次条件下的稳定性，不能替代跨环境验证。

## 4. 二十轮复测结果

正的耗时下降表示变快；负值表示变慢。

### 各变体扫描选出的最佳候选

| 变体 | 实现 | CTA数 | threads | v025 μs | 候选 μs | 耗时下降 | 配对平均节省 μs的95%区间 |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | local_vector | 32 | 128 | 3.612 | 3.599 | 0.35% | [-0.0044, 0.0128] |
| XSF | static_group | 32 | 256 | 3.581 | 3.548 | 0.93% | [0.0195, 0.0394] |
| FP8 | static_group | 32 | 256 | 3.768 | 3.709 | 1.56% | [0.0548, 0.0704] |
| Quantized | local_vector | 32 | 256 | 3.873 | 3.686 | 4.82% | [0.1846, 0.1976] |

Base区间包含0，没有确定收益；XSF约0.93%属于小幅局部实现收益，CTA和threads均未减少。FP8/Quantized有可复现收益，但最佳候选都保持32CTA，并将threads从128增加到256。两者实现也有差别，不能把全部收益归因于线程数一个因素。

### 最佳较少CTA候选

| 变体 | CTA数 | threads | v025 μs | 候选 μs | 耗时下降 |
|---|---:|---:|---:|---:|---:|
| Base | 16 | 128 | 3.604 | 3.717 | -3.13% |
| XSF | 16 | 256 | 3.584 | 3.704 | -3.36% |
| FP8 | 16 | 256 | 3.768 | 3.973 | -5.43% |
| Quantized | 16 | 256 | 3.881 | 3.994 | -2.90% |

四个较少CTA候选均变慢，约2.9%～5.4%。本轮没有证据支持为tiny减少CTA。减少线程的候选也没有稳定收益：Base的128线程候选不确定，XSF的128线程、FP8/Quantized的64线程复测均变慢。

## 5. 寄存器与private资源

原生模块属性查询；这些是编译资源，不是实测驻留时间线。

| kernel | threads | registers/thread | private bytes | shared bytes |
|---|---:|---:|---:|---:|
| base_local_vector_g1_t128 | 128 | 7 | 0 | 0 |
| base_local_vector_g2_t128 | 128 | 14 | 0 | 0 |
| formal_base | 256 | 6 | 0 | 0 |
| xsf_static_group_g1_t256 | 256 | 6 | 0 | 0 |
| xsf_local_vector_g2_t256 | 256 | 14 | 0 | 0 |
| xsf_local_vector_g1_t128 | 128 | 7 | 0 | 0 |
| formal_xsf | 256 | 6 | 0 | 0 |
| fp8_static_group_g1_t256 | 256 | 6 | 0 | 0 |
| fp8_local_vector_g2_t256 | 256 | 12 | 0 | 0 |
| fp8_local_vector_g1_t64 | 64 | 10 | 0 | 0 |
| formal_fp8 | 128 | 8 | 0 | 0 |
| quantized_local_vector_g1_t256 | 256 | 6 | 0 | 0 |
| quantized_local_vector_g2_t256 | 256 | 14 | 0 | 0 |
| quantized_local_vector_g1_t64 | 64 | 10 | 0 | 0 |
| formal_quantized | 128 | 7 | 0 | 0 |

15个查询对象均private=0、shared=0，不能将较少CTA方案的失败归因于private spill。减少线程会增加每线程列数，例如256/128/64/32线程、每CTA一个token时为1/2/4/8列；减少CTA还会将任务集中到更少CTA。这与完整kernel变慢的观察一致，但本轮没有采集新的访存等待计数，未精确分摊各原因。SDK header占用率接口对本架构返回不支持，其0值不解释为实际占用率。

## 6. 正确性与结论

现有测试 **131 passed**，0 failures/errors/skipped。对H256/K2覆盖选中的候选模板，共87次候选kernel构造，包含FP16/BF16/FP32、动态token尾部与无效路由；其余shape保持正式实现。T32性能特化版本另通过前述五种路由逐字节核对。

本轮结论：

- 不采用减少CTA方案；减少线程没有明确优势。
- Base保持当前实现；XSF的小幅候选收益保留为实验记录。
- FP8/Quantized的32CTA、256线程候选值得后续接入验证，耗时下降约1.56%/4.82%。
- XSF/FP8/Quantized的tiny候选已正式接入；Base保留原实现，目录改为 `v029_ACC`。接入后的结果见下节。

## 7. 复现与数据

```bash
cd /data/TileOPs-Metax
source /data/sc16g-recovery-20261004/env.sh
python profiles/v029_ACC/scripts/sweep_static.py
python profiles/v029_ACC/scripts/sweep_local.py
python profiles/v029_ACC/scripts/confirm_validate.py
```

重跑覆盖本版本结果，先保留需要的原始数据。

- [112组原始配置数据：正式线程/固定shape](../raw/static_sweep_16g.csv)、[显式向量读写](../raw/local_sweep_16g.csv)。
- [20轮确认结果](../raw/confirmation_16g.json)，逐轮时延在 `raw/confirm_*_samples.json`。
- [正确性JUnit](../raw/selected_correctness.xml)、[候选覆盖计数](../meta/selected_validation.json)。
- [原生资源](../raw/kernel_resources.json)、[原始v025](../codegen/baseline_v025.py)。
- `codegen/`保存各配置生成代码与确认对象的原生代码对象；`logs/`保存成功、失败和中断记录。

## 8. 正式接入

正式入口 `tileops/kernels/moe/reduce_fused.py` 对加权T32/K2/H256、FP16进行分派：XSF/FP8使用fragment实现，Quantized使用per-thread local实现，均32CTA、256线程。Base保留原实现。私有helper位于 `tileops/kernels/moe/reduce_fused_tiny.py`，正式代码不依赖profiles。

四个tiny同轮配对结果（CUPTI，20轮）：

| 变体 | 接入前 μs | 接入后 μs | 耗时下降 |
|---|---:|---:|---:|
| Base | 3.615 | 3.612 | 0.07% |
| XSF | 3.607 | 3.569 | 1.06% |
| FP8 | 3.850 | 3.692 | 4.12% |
| Quantized | 3.712 | 3.656 | 1.52% |

Base生成代码未变，其数值差异为测量波动。其余13组未选中workload生成代码与接入前一致；现有131项测试通过；四个tiny的六种路由共24组输出逐字节一致。

[接入性能CSV](../raw/integrated_comparison_16g.csv)、[正式源码快照](../codegen/integrated_formal.py)、[tiny源码快照](../codegen/integrated_tiny.py)。XSF约1%的收益较小，FP8/Quantized的幅度随运行条件变化，保留本轮与历史配对结果。
