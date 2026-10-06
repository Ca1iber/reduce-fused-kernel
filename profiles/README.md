# Profile 目录约定

所有 Profile 脚本、原始数据和分析结果按实验顺序放在这里。版本目录必须带状态后缀：

- `vxxx_ACC`：已接入正式 kernel 的优化，包括后来已接入的参数选择。
- `vxxx_DEP`：未采用的优化实验；保留代码、数据和失败记录。
- `vxxx_DIS:vyyy`：针对 vyyy 的延伸讨论、验证或分析。
- `vxxx_SUM`：总结（Summary），用于汇总多个版本的结果与结论。

`v000` 仅表示最初的 naive 基线，不另建目录。DIS 后的版本号表示讨论对象，不带其目录后缀。

## 当前版本

| 目录 | 内容与分类依据 |
|---|---|
| [v001_DIS:v000](v001_DIS:v000/analysis/README.md) | naive 的初始 Roofline |
| [v002_DIS:v000](v002_DIS:v000/analysis/README.md) | naive 的 mcProfiler Roofline 与带宽分析 |
| [v003_ACC](v003_ACC/analysis/README.md) | 直接 FP32→FP8 编码，已接入 |
| [v004_ACC](v004_ACC/analysis/README.md) | hidden 分块，已接入 |
| [v005_DEP](v005_DEP/analysis/README.md) | 两行寄存器预取，未采用 |
| [v006_DEP](v006_DEP/analysis/README.md) | 单 shared buffer，未采用 |
| [v007_ACC](v007_ACC/analysis/README.md) | tile 参数选择，已在 v010 接入 |
| [v008_DIS:v007](v008_DIS:v007/analysis/README.md) | 对 v007 的 threads/tile 延伸验证，无新方案接入 |
| [v009_ACC](v009_ACC/analysis/README.md) | Base/XSF 参数选择，已在 v010 接入 |
| [v010_ACC](v010_ACC/analysis/README.md) | 按 shape/变体选择正式配置，已接入 |
| [v011_DIS:v010](v011_DIS:v010/analysis/README.md) | 正式 v010 的工具与基线审计 |
| [v012_DEP](v012_DEP/analysis/README.md) | async K-ring，未采用 |
| [v013_DEP](v013_DEP/analysis/README.md) | 删除/替换 CTA 同步，正确性失败 |
| [v014_DEP](v014_DEP/analysis/README.md) | warp 协作 shared gather，未接入 |
| [v015_DEP](v015_DEP/analysis/README.md) | hidden shared 双缓冲，未采用 |
| [v016_DEP](v016_DEP/analysis/README.md) | 更深寄存器预取，有收益但未接入 |
| [v017_DIS:v013](v017_DIS:v013/analysis/README.md) | 定位 v013 同步错误的根因 |
| [v018_DIS:v016](v018_DIS:v016/analysis/README.md) | 围绕 v016 的寄存器、驻留与双 fragment 对照 |
| [v019_SUM](v019_SUM/analysis/README.md) | v000 与 v010 的 16 组加速比总结 |
| [v020_ACC](v020_ACC/analysis/README.md) | h7168/prefill 的 FP8/Quantized 采用 hidden-first grid，已接入 |
| [v021_SUM](v021_SUM/analysis/README.md) | v000/v020 的 Base/FP8 八组原生 Roofline 对照 |
| [v022_ACC](v022_ACC/analysis/README.md) | 向量写回已接入 FP8/Quantized 的 tiny/prefill 四组合 |
| [v023_DIS:v021](v023_DIS:v021/analysis/README.md) | 当前实例 1:1、8:1、16:1 实测带宽，解释 Roofline 屋顶 |

## 后续约定

- 新版本编号为已有最大编号加一，编号包含所有 ACC、DEP、DIS、SUM 目录。
- 新优化实验在接入前使用 DEP；接入正式 kernel 后改为 ACC，并更新引用。
- 延伸讨论使用 DIS，并注明讨论对象版本；总结使用 SUM。未来仍按本约定命名。
- 改名不删除旧实验，也不改变历史 benchmark 的版本标签。

每个版本按用途分类：

```text
profiles/
  v003_ACC/
    scripts/    # 采集、启动、解析脚本
    raw/        # profiler/trace 原始输出
    analysis/   # 分析、汇总、图表
    codegen/    # 设备代码、IR、汇编
    logs/       # 运行和错误日志
    meta/       # 环境、参数、命令、源码版本与哈希
  v005_DEP/
    ...
  v017_DIS:v013/
    ...
  v019_SUM/
    analysis/
      README.md # 总结入口；数据等附件按上述分类存放
```

同一版本的多个 workload 放在对应分类下的 case 子目录。meta 记录 Git commit、改动、源码哈希、shape/dtype、命令、工具版本和退出码。analysis 写明目的、证据、结论与未确认问题。失败采集同样保留。历史原始数据和采集元数据保留采集时的路径；当前复现命令、脚本导入路径和文档链接使用新目录名。
