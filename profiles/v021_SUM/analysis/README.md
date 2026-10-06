# v021：v000/v020 的 Base 与 FP8 Roofline

已完成 16 张 mcProfiler 原生单 kernel Roofline：每版 Base/FP8 × tiny/h3072/h7168/prefill，各八张。图片位于对应版本文件夹根目录，文件名注明版本、变体、workload 和 shape。

## 图片索引

| 变体 | workload | v000 | v020 |
|---|---|---|---|
| base | tiny | [v000](../v000-roofline/v000_base_tiny_T32_K2_H256.png) | [v020](../v020-roofline/v020_base_tiny_T32_K2_H256.png) |
| base | h3072 | [v000](../v000-roofline/v000_base_h3072_T512_K8_H3072.png) | [v020](../v020-roofline/v020_base_h3072_T512_K8_H3072.png) |
| base | h7168 | [v000](../v000-roofline/v000_base_h7168_T512_K8_H7168.png) | [v020](../v020-roofline/v020_base_h7168_T512_K8_H7168.png) |
| base | prefill | [v000](../v000-roofline/v000_base_prefill_T4096_K8_H7168.png) | [v020](../v020-roofline/v020_base_prefill_T4096_K8_H7168.png) |
| fp8 | tiny | [v000](../v000-roofline/v000_fp8_tiny_T32_K2_H256.png) | [v020](../v020-roofline/v020_fp8_tiny_T32_K2_H256.png) |
| fp8 | h3072 | [v000](../v000-roofline/v000_fp8_h3072_T512_K8_H3072.png) | [v020](../v020-roofline/v020_fp8_h3072_T512_K8_H3072.png) |
| fp8 | h7168 | [v000](../v000-roofline/v000_fp8_h7168_T512_K8_H7168.png) | [v020](../v020-roofline/v020_fp8_h7168_T512_K8_H7168.png) |
| fp8 | prefill | [v000](../v000-roofline/v000_fp8_prefill_T4096_K8_H7168.png) | [v020](../v020-roofline/v020_fp8_prefill_T4096_K8_H7168.png) |

## 采集口径

- v000 为保存的原始 naive 源码；v020 为包含按 shape 选择 grid 的正式源码快照，版本与哈希见 [sources.json](../meta/sources.json)。
- seed=1235，10 次 warmup；从显式 profiler.start 标记开始采集一轮目标 kernel。该 native 采集使用相同输入地址，不沿用 benchmark 的逐次输入克隆/L2 flush 协议。tiny 工作集很小，需考虑 warm cache。
- 单 kernel 图根据工具对应记录提取；16 个 PNG 的完整性、对应 shape 与原始 RoofLine 记录已核对，见 [image_check.json](../meta/image_check.json)。
- 原生屋顶仍是工具模型参考，不等于当前实例实测上限；native ops/traffic 口径不能直接等同 v001 的算法 FLOPs/最小字节数。
- 各版本目录的 raw、logs、codegen 分别保留原始结果、日志和设备代码。v000 Base prefill 首次出图失败，失败记录保留，第二次采集成功；展示图来自成功重采。

[采集状态与命令](../meta/status.json)。
