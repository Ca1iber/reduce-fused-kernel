# 流水线与 shared memory：v005_DEP/v006 后续复盘（16g）

## 结论

两条路线都没有被 v005_DEP/v006 的失败排除。本轮找到了可复现的成功配置，因此不能再写“在这个 kernel 上无论如何都不可行”。

- shared：v014 warp 分担 expert 读取、消除动态局部position索引，tile1024/512threads；FP8 h7168 48.338→47.324µs，1.0214×。
- 软件预取：v016 depth8、tile512/128threads；FP8 h3072 22.738→21.074µs，1.0790×；h7168 48.067→44.273µs，1.0857×。Quantized 分别1.0268×、1.0684×。
- 真实global→shared硬件async已跑通；K-ring及已测hidden双缓冲还没击败正式最佳配置。这一子结论和寄存器软件预取的成功分开记录。

## 为什么原来判断太早

v005只尝试两个register fragment，v006只尝试逐K单buffer+两次CTA同步；并没有验证预取深度、任务分配、动态局部数组、真正async以及CTA数量。失败只能支持“当时这一个实现没有收益”。
输入一次读取并不意味着 shared 永远无用：warp协作版本不减少必需HBM字节，而是改变谁读取、如何安排并发、何时由consumer使用。shared也不是免费：搬运、同步和容量仍可能抵消收益。
memory bound是总体约束；当读取请求组织、并行度和依赖链尚未达到最佳时，仍能在不减少流量的条件下提升有效带宽。

## 本轮实验覆盖

| 版本 | 改变 | 已有证据 | 工程判断 |
|---|---|---|---|
| v011 | 基线和工具能力审计 | 准确kernel过滤、58.8MB/3.67MB；native async probe bytes一致 | 旧混合计数不可用于归因，API可用 |
| v012 | single/sync ring2/async ring2/3 | 配对计时、native计数、trace；async2/3各131用例 | async减少WSM等待，但总成本未赢baseline |
| v013 | 删CTA sync/换warp sync | 两种均有逐字节错误，控制正确 | 不使用错误版本的性能数字 |
| v014 | warp协作、global position、tile512/1024和threads128/256/512 | 原始private36/异常写入→private0/正常流量；12组+五轮独立确认 | FP8 h7168约2.1%成功，其余不推广 |
| v015 | hidden doublebuffer、2/4块分组 | 直接/sync/async控制、byte一致、资源及有效native计数 | 小grid和shared容量/额外工作有代价，已测方案不采用 |
| v016 | depth2/4/8 register prefetch | 全131通过、12组+五轮独立确认；有效最小native重采 | depth8在4个SF T512组合成功 |

[v011审计](../../v011_DIS:v010/analysis/README.md)、[v012详细计数](../../v012_DEP/analysis/profile_comparison.json)、[v013错误](../../v013_DEP/analysis/README.md)、[v014报告](../../v014_DEP/analysis/v014_cooperative_shared_16g.md)、[v015报告](../../v015_DEP/analysis/v015_hidden_pipeline_16g.md)、[v016报告](v016_register_prefetch_16g.md)。

## Profile 支持了什么，没支持什么

1. v012 sync2 WSM stall358782→async2 56760，shared普通store113632→0；来自专用copy机制，shared load仍约113k，流量不减。
2. v014动态局部position版private_total36、write26.0MB；修正后private0、write3.67MB，VLS stall显著下降。不能拿一次坏布局代表shared方向。
3. v015同次trace direct whole-token84.736µs/512CTA，baseline50.688µs/7168CTA；group2_direct52.480µs/3584CTA，async63.744µs/shared16KiB。控制证明几何与shared的影响不同。
4. v016 registers12→22而private/shared均0；有效目标traffic仍58.8MB/3.67MB，证明不是通过漏算/减少必需输入获得收益。
5. 没有使用trace的occupancy字段推导实际驻留；没有把总stall、总wave或ComputeInstructions当百分比、FLOPs或时长。
6. v016最初大事件集计数矛盾，保留且标为invalid；成功重采只支持核心计数，不能拿未验证的stall解释收益。所有计数只取独立目标或目标周期JSON，不取全应用report aggregate。

## 适用范围与保留问题

成功证据限定 sc-16g 当前runtime、这些shape/dtype及现有测试覆盖。没有证明所有输入分布、K值、设备和SDK都获益；单次环境异常原始样本保留。
shared元数据、输出重排、其它warp producer/consumer和更大参数搜索仍可能研究；本轮已发现两条路线的反例，不需要穷举它们来证明“不可能”。
当前最佳正式文件保持v010。实验候选各自完整在 scripts 下，可重复运行；没有擅自把对其它workload会退步的方案全局替换。
本次用户目标是重新验证两条路线可行性及提供证据；已有收益、独立确认、正确性、profile和失败复盘完成了这一目标。生产dispatch集成可独立进行，其正确性/benchmark需基于最终实现再验证。

## 重复实验

工作目录 `/data/TileOPs-Metax`，先 `source /data/sc16g-recovery-20261004/env.sh`。
顺序执行 `python profiles/v014_DEP/scripts/run_confirmation.py`、`python profiles/v016_DEP/scripts/run_confirmation.py`，每次重跑另存原CSV，避免追加结果误聚合。
`python profiles/v016_DEP/scripts/validate.py` 和 `python profiles/v014_DEP/scripts/validate_best.py` 跑现有测试；后者有dtype/shape fallback，详见报告。
`python profiles/v015_DEP/scripts/collect_profiles.py`、v014_DEP/v016对应collector用于资源及计数；v016旧大事件集已知不可靠，采用 `recollect_register8.py` 的最小独立事件集。
逐字节校验和正式benchmark不与native profiler同时运行。本轮保存目录为 `/data/TileOPs-Metax/profiles/v011` 至 `v016`。
