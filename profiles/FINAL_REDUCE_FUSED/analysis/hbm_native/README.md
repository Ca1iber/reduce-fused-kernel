# HBM 原生Roofline：已完成32组对照

**完成日期：2026-10-09。8组复用，24组新增；正式源码仅来自restart。**

| 变体 | Workload | v000 HBM usage（1.50 TB/s） | 终盘 HBM usage（1.50 TB/s） | 终盘 HBM usage（推算） | v000图 | 终盘图 |
|---|---|---:|---:|---:|---|---|
| Base | tiny | 1.49% | 1.72% | 1.65% | [图](../../../v021_SUM/v000-roofline/v000_base_tiny_T32_K2_H256.png) | [图](verified_hbm_final_base_tiny.png) |
| Base | h3072 | 96.69% | 89.62% | 98.62% | [图](../../../v021_SUM/v000-roofline/v000_base_h3072_T512_K8_H3072.png) | [图](verified_hbm_final_base_h3072.png) |
| Base | h7168 | 103.72% | 100.22% | 104.37% | [图](../../../v021_SUM/v000-roofline/v000_base_h7168_T512_K8_H7168.png) | [图](verified_hbm_final_base_h7168.png) |
| Base | prefill | 99.34% | 100.78% | 101.48% | [图](../../../v021_SUM/v000-roofline/v000_base_prefill_T4096_K8_H7168.png) | [图](verified_hbm_final_base_prefill.png) |
| XSF | tiny | 1.53% | 1.70% | 1.68% | [图](verified_hbm_v000_xsf_tiny.png) | [图](verified_hbm_final_xsf_tiny.png) |
| XSF | h3072 | 83.28% | 87.04% | 85.65% | [图](verified_hbm_v000_xsf_h3072.png) | [图](verified_hbm_final_xsf_h3072.png) |
| XSF | h7168 | 100.10% | 100.19% | 100.74% | [图](verified_hbm_v000_xsf_h7168.png) | [图](verified_hbm_final_xsf_h7168.png) |
| XSF | prefill | 98.41% | 100.53% | 100.18% | [图](verified_hbm_v000_xsf_prefill.png) | [图](verified_hbm_final_xsf_prefill.png) |
| FP8 | tiny | 0.90% | 1.20% | 1.20% | [图](../../../v021_SUM/v000-roofline/v000_fp8_tiny_T32_K2_H256.png) | [图](verified_hbm_final_fp8_tiny.png) |
| FP8 | h3072 | 43.95% | 91.72% | 87.24% | [图](../../../v021_SUM/v000-roofline/v000_fp8_h3072_T512_K8_H3072.png) | [图](verified_hbm_final_fp8_h3072.png) |
| FP8 | h7168 | 44.92% | 89.68% | 98.71% | [图](../../../v021_SUM/v000-roofline/v000_fp8_h7168_T512_K8_H7168.png) | [图](verified_hbm_final_fp8_h7168.png) |
| FP8 | prefill | 78.66% | 99.74% | 100.72% | [图](../../../v021_SUM/v000-roofline/v000_fp8_prefill_T4096_K8_H7168.png) | [图](verified_hbm_final_fp8_prefill.png) |
| Quantized | tiny | 0.91% | 1.39% | 1.16% | [图](verified_hbm_v000_quantized_tiny.png) | [图](verified_hbm_final_quantized_tiny.png) |
| Quantized | h3072 | 41.49% | 75.81% | 78.76% | [图](verified_hbm_v000_quantized_h3072.png) | [图](verified_hbm_final_quantized_h3072.png) |
| Quantized | h7168 | 44.75% | 92.81% | 93.84% | [图](verified_hbm_v000_quantized_h7168.png) | [图](verified_hbm_final_quantized_h7168.png) |
| Quantized | prefill | 77.58% | 99.81% | 100.11% | [图](verified_hbm_v000_quantized_prefill.png) | [图](verified_hbm_final_quantized_prefill.png) |

## 采集口径与数据边界

1.50 TB/s是近似实测参考，换算值可以略超过100%，保留实际数值。图中的百分比仍是原生1.8432 TB/s口径，与本表换算值不同。

- 文档HBM usage表以1.50 TB/s为参考：`100 × native case_bandwith /1500`，等价于原生百分比乘1.2288。原始CSV与PNG保留1843.2 GB/s工具屋顶。它使用native带宽，不是算法字节/benchmark runtime估算。
- 原有v000 Base/FP8八组直接复用v021；没有重采覆盖这些有效记录。
- 新采的终盘Base/h3072、FP8/h3072使用10次预热；其余22组在采集进程内不执行目标预热，每个不同名称的目标kernel只调用一次，使trace和指标重放顺序一致。
- 采集前完成JIT/source准备，输入seed1235，同地址；没有benchmark逐次L2 flush和输入克隆。新旧profile预热和日期不同，不能把两列差值当作隔离出的单项优化收益。
- 逐组核对WORKGROUPS、native字节量和周期尺度；本次32条均通过，原始记录与哈希见 [usage CSV](../../raw/hbm_native/usage_16g.csv)，核对记录见 [ingestion status](../../meta/hbm_native/ingestion_status.json)。
- native周期换算与benchmark不完全相同；正式性能结论以独立benchmark为准，2026-10-09十六组复测均通过且基本维持封盘性能。

## 历史失败采集

此前采集出现32CTA被计成182076CTA、微秒级目标被计成毫秒、HBM usage超过48000%等异常。失败原始数据保留，未混入以上32条。今天单组采集先恢复正常；批量程序移除采集进程内目标预热后，22组的事件与重放对齐并取得可核对结果。
此前异常的全部根因尚未通过因果隔离验证；不能把运营商线路、7%GPU利用率或共享负载直接认定为原因。
