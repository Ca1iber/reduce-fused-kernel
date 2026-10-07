# v030：tiny 两线程协作处理一列（sc-16g）

## 1. 目标与假设

T32/K2/H256、FP16、32CTA。K=2时，两个线程分别读取两个expert的输入及系数，尝试减少串行输入加载等待。代价是warp内通信、更多warp或半数线程只参与加载而不执行最后的编码。

## 2. 分工与精度

- **adjacent**：相邻lane配对，expert=threadIdx%2，partner=lane^1。
- **halfwarp**：64线程warp的两个32线程半区配对，expert=(lane//32)，partner=lane^32。
- 512线程：256对，每对一列；256线程：128对，每对两列；128线程：64对，每对四列。每一列都由同一对线程协作。
- 先各自加载原始x与系数。通过TileLang公开shuffle原语交换原始FP32值、系数和有效位置，然后expert0线程仍按expert0→expert1顺序完成两次累加，避免先分别舍入两个乘积再相加造成精度变化。
- 只有expert0线程编码、写回。SF编码保持v003；无shared、无CTA barrier、无async。
- shuffle使用64位完整warp mask、width64；初版把-1直接构造成uint64常量导致构图失败，已改为正的0xFFFFFFFFFFFFFFFF，失败源码和日志保留。这是接口常量构造问题，不是GPU执行错误。

[TileLang候选](../scripts/tiny_pair.py)。只针对tiny；正式kernel尚未修改。

## 3. 协议与正确性

24配置（4变体×2配对布局×3线程数），每配置先与v025进行random、identity、padded、duplicates、all-invalid五种路由的逐字节核对，全部通过，再进行性能测量。

项目bench_kernel：CUPTI、L2flush、输入克隆、预分配输出、10warmup、50repeat×3trials，扫描外层5轮交替顺序。选择候选后，新进程重新生成seed1235输入，与正式v025及v029最佳候选同轮比较20轮，轮换/反转三个实现顺序；按配对轮次均值差bootstrap5000次给出95%参考区间。

现有测试 **131 passed**，0 failures/errors/skipped；H256/K2使用选中配对模板的动态token版本，共87次候选构造，覆盖FP16/BF16/FP32、不同token数与无效路由。固定T32性能版本另外经过五种路由逐字节核对。

## 4. 首次独立20轮复测

正的耗时下降表示变快。

| 变体 | 布局/线程 | v025 μs | v029 μs | v030 μs | 比v025耗时下降 | 比v029耗时下降 |
|---|---|---:|---:|---:|---:|---:|
| Base | adjacent/256 | 3.599 | 3.633 | 3.382 | 6.05% | 6.91% |
| XSF | adjacent/256 | 3.589 | 3.574 | 3.382 | 5.78% | 5.37% |
| FP8 | adjacent/512 | 3.732 | 3.671 | 3.584 | 3.98% | 2.37% |
| Quantized | adjacent/256 | 3.727 | 3.656 | 3.988 | -7.01% | -9.10% |

Base、XSF、FP8在独立复测中都优于v025及v029，配对节省时间区间均大于0。

| 变体 | 对v025平均节省 μs的95%区间 | 对v029平均节省 μs的95%区间 |
|---|---:|---:|
| Base | [0.2081, 0.2263] | [0.2383, 0.2614] |
| XSF | [0.2053, 0.2196] | [0.1889, 0.2089] |
| FP8 | [0.1341, 0.1556] | [0.0727, 0.0963] |
| Quantized | [-0.2724, -0.2560] | [-0.3415, -0.3259] |

## 5. Quantized相近配置补测

扫描中256线程和512线程的相对收益只差约0.13个百分点。第一次复测的256线程结果退化，因此另外用新进程将v025、v029、pair256、pair512同轮测20轮，避免只根据一个配置否定全部方向。原两轮数据均保留。

| 实现 | 补测中位 μs |
|---|---:|
| v025 | 3.743 |
| v029 | 3.656 |
| pair256 | 3.668 |
| pair512 | 3.661 |

| 两线程候选 | 比v025耗时下降 | 比v029耗时下降 |
|---|---:|---:|
| pair256 | 1.98% | -0.35% |
| pair512 | 2.19% | -0.14% |

补测显示Quantized可以比v025快约2%，但没有证明优于v029最佳单线程列实现；512线程对v029的配对区间包含0。256线程两次独立测量波动较大，生成设备代码逐字一致，具体原因尚未定位。这里不把某一轮更快的结果当作稳定收益，Quantized继续推荐v029候选。

## 6. 资源与解释

原生资源查询：

| kernel | threads | registers/thread | private bytes | shared bytes |
|---|---:|---:|---:|---:|
| base_v025 | 256 | 6 | 0 | 0 |
| base_v029 | 128 | 7 | 0 | 0 |
| base_v030 | 256 | 8 | 0 | 0 |
| xsf_v025 | 256 | 6 | 0 | 0 |
| xsf_v029 | 256 | 6 | 0 | 0 |
| xsf_v030 | 256 | 10 | 0 | 0 |
| fp8_v025 | 128 | 8 | 0 | 0 |
| fp8_v029 | 256 | 6 | 0 | 0 |
| fp8_v030 | 512 | 8 | 0 | 0 |
| quantized_v025 | 128 | 7 | 0 | 0 |
| quantized_v029 | 256 | 6 | 0 | 0 |
| quantized_v030 | 256 | 10 | 0 | 0 |

12个查询对象均private=0、shared=0。原生允许驻留上限是资源容量，不能当实测驻留数量；SDK header接口对该架构不支持，其0值不解释成占用率。

已验证改变了expert加载任务分配，并确实使用warp内shuffle；没有减少输入/输出必需字节。缩短访存依赖等待是机制假设，尚未用新的stall计数精确归因。增加线程和列编码任务分配也同时发生，不能把全部收益只归给shuffle或某一个因素。

## 7. 结论

- 两线程协作可行，并保持输出精度与K累加顺序。
- Base/XSF采用相邻配对256线程候选值得后续接入，分别比正式版快约6.05%/5.78%。
- FP8采用相邻配对512线程候选值得后续接入，比正式版快约3.98%，比v029快约2.37%。
- Quantized没有确认超过v029的收益，推荐保留v029单线程列候选。
- Base/XSF/FP8已接入正式kernel，Quantized保留v029；目录改为 `v030_ACC`，接入结果见第9节。

## 8. 复现与原始数据

```bash
cd /data/TileOPs-Metax
source /data/sc16g-recovery-20261004/env.sh
python profiles/v030_ACC/scripts/sweep.py
python profiles/v030_ACC/scripts/confirm_validate.py
python profiles/v030_ACC/scripts/recheck_quantized.py
```

重跑覆盖本目录结果；与v029对照依赖其scripts及meta/selected.json。生成的设备代码已逐项保存。

- [24配置扫描](../raw/sweep_16g.csv)、[20轮同轮确认](../raw/confirmation_16g.json)。
- [Quantized相近配置额外复测](../raw/quantized_close_configs_recheck.json)，逐轮样本在相邻samples文件。
- [正确性JUnit](../raw/correctness.xml)、[候选覆盖](../meta/validation.json)、[原生资源](../raw/resources.json)。
- [固定v025源码](../codegen/baseline_v025.py)；每个实现生成代码在 `codegen/`，成功和初版失败日志在 `logs/`。

## 9. 正式接入及四个tiny结果

正式入口对加权T32/K2/H256、FP16选择v030：Base/XSF使用32CTA×256线程，FP8使用32CTA×512线程，均相邻lane配对；Quantized保留v029的32CTA×256线程local实现。生产实现位于 `tileops/kernels/moe/reduce_fused_tiny.py`，分派位于 `tileops/kernels/moe/reduce_fused.py`。512线程仅允许已测FP8 tiny；自定义256线程FP8仍有v029路径。

按用户要求，仅对四个tiny核对一次随机输入的逐字节输出，再进行项目配对benchmark；不追加其他shape、大测试集或额外资源校验。四组输出均一致。

| 变体 | v029 μs | 接入后 μs | 耗时下降 |
|---|---:|---:|---:|
| Base | 3.615 | 3.369 | 6.80% |
| XSF | 3.574 | 3.400 | 4.87% |
| FP8 | 3.692 | 3.640 | 1.39% |
| Quantized | 3.676 | 3.651 | 0.70% |

Quantized实现保持原样，其差异属于运行波动；不宣称新收益。CUPTI、L2flush、克隆输入、10warmup/50repeat×3trials，外层5轮交替顺序。

[四组接入CSV](../raw/integrated_comparison_16g.csv)、[接入计时脚本](../scripts/benchmark_integrated.py)、[正式入口快照](../codegen/integrated_formal.py)、[tiny实现快照](../codegen/integrated_tiny.py)。
