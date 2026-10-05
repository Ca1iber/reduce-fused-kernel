# v010：正式接入按参数选择 launch 配置（sc-16g）

## 改动

`tileops/kernels/moe/reduce_fused.py` 中增加 `_MEASURED_CONFIGS` 和 `MoeReduceFusedKernel.default_config`。
包装层按 T/K/H/dtype 查表；with_sf=False 选择 Base/XSF 配置，True 选择 FP8/Quantized 配置。
仅加权归约、已实测的四个 shape/dtype 命中表；其他 shape、dtype 或无权重配置继续使用 v004 默认值。
JIT 工厂接收可选 tile_hidden、num_threads；包装层把选好的配置传进去。
直接按原七个参数调用 JIT 工厂，仍得到 v004 的默认配置。

| workload（T/K/H/input dtype） | Base/XSF tile / threads | FP8/Quantized tile / threads |
|---|---|---|
| tiny：32/2/256/FP16 | 256 / 256 | 256 / 128 |
| h3072：512/8/3072/BF16 | 3072 / 256 | 512 / 128 |
| h7168：512/8/7168/BF16 | 7168 / 256 | 512 / 128 |
| prefill：4096/8/7168/BF16 | 7168 / 256 | 1024 / 128 |

## 手动配置

现有 Kernel 的 config 参数已接通，例如：

```python
kernel = MoeReduceFusedKernel(
    512, 8, 3072, torch.bfloat16,
    config={"tile_hidden": 1024, "num_threads": 128},
)
print(kernel.config)
```

config 可只覆盖一个字段，其余使用当前 shape 的默认配置。
tile 必须是能整除 hidden 的正256倍数，threads 支持此次实验的64/128/256。
Op 接口保持现有调用方式，自动选型发生在内部 Kernel 创建时。

## 验证

- 配置检查：16组配对计时均确认实际 kernel.config 命中上述表。
- 未测 shape、dtype、无权重回退，以及手动覆盖、非法配置检查通过（meta/dispatch_check.json）。
- 现有 tests/ops/test_moe_reduce_fused.py：131 passed。
- 现有 benchmarks/ops/bench_moe_reduce_fused.py：16 passed。
- 16个正式 wrapper 输出与v004源码快照逐字节一致。

## 与 v004 配对计时

旧版与新版 Kernel wrapper 接收相同输入和预分配输出。
项目 bench_kernel：10 warmup、50 repeats×3 trials、L2 flush、cupti GPU timeline。
按A-B-B-A-A-B测量，分别取三次结果中位数。每个workload单独进程。
tiny/h3072/h7168使用项目输入克隆规则；prefill复用地址，旧/新协议一致。

| 变体 | workload | tile | threads | v004 μs | 接入后 μs | 加速比 |
|---|---|---:|---:|---:|---:|---:|
| base | tiny | 256 | 256 | 3.697 | 3.610 | 1.024× |
| xsf | tiny | 256 | 256 | 3.686 | 3.599 | 1.024× |
| fp8 | tiny | 256 | 128 | 3.912 | 3.917 | 0.999× |
| quantized | tiny | 256 | 128 | 3.922 | 3.917 | 1.001× |
| base | h3072 | 3072 | 256 | 21.320 | 20.777 | 1.026× |
| xsf | h3072 | 3072 | 256 | 21.376 | 20.751 | 1.030× |
| fp8 | h3072 | 512 | 128 | 23.844 | 22.779 | 1.047× |
| quantized | h3072 | 512 | 128 | 23.823 | 22.948 | 1.038× |
| base | h7168 | 7168 | 256 | 44.355 | 44.037 | 1.007× |
| xsf | h7168 | 7168 | 256 | 44.488 | 44.201 | 1.006× |
| fp8 | h7168 | 512 | 128 | 48.415 | 47.821 | 1.012× |
| quantized | h7168 | 512 | 128 | 48.538 | 48.169 | 1.008× |
| base | prefill | 7168 | 256 | 357.914 | 350.868 | 1.020× |
| xsf | prefill | 7168 | 256 | 358.354 | 352.927 | 1.015× |
| fp8 | prefill | 1024 | 128 | 354.417 | 354.396 | 1.000× |
| quantized | prefill | 1024 | 128 | 358.180 | 358.963 | 0.998× |

## 判断与复现

实测配置已接入正式代码。约1%以内的差异按小收益或计时波动处理；保持相同配置的FP8/Quantized tiny、prefill不宣称有优化收益。
本轮没有新增硬件counter采集，性能原因沿用此前实验假设，不据此声称占用率或HBM指标发生特定变化。
原始值：raw/paired_selection_16g.csv；基线与正式代码快照：codegen/；源码SHA：meta/source.json。
仓库根目录加载现有环境后运行 `python profiles/v010_ACC/scripts/run.py`。
若重测，先另存旧paired CSV，避免在同一个原始结果文件中混入重复轮次。
