"""Build Chinese experiment reports from checked benchmark and profile artifacts."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[2]
def read(v,name):return json.loads((R/{'v001': 'v001_DIS:v000', 'v002': 'v002_DIS:v000', 'v003': 'v003_ACC', 'v004': 'v004_ACC', 'v005': 'v005_DEP', 'v006': 'v006_DEP', 'v007': 'v007_ACC', 'v008': 'v008_DIS:v007', 'v009': 'v009_ACC', 'v010': 'v010_ACC', 'v011': 'v011_DIS:v010', 'v012': 'v012_DEP', 'v013': 'v013_DEP', 'v014': 'v014_DEP', 'v015': 'v015_DEP', 'v016': 'v016_DEP', 'v017': 'v017_DIS:v013', 'v018': 'v018_DIS:v016'}[v]/'analysis'/name).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def bench_table(v,name,candidate):
 rows=list(csv.DictReader((R/{'v001': 'v001_DIS:v000', 'v002': 'v002_DIS:v000', 'v003': 'v003_ACC', 'v004': 'v004_ACC', 'v005': 'v005_DEP', 'v006': 'v006_DEP', 'v007': 'v007_ACC', 'v008': 'v008_DIS:v007', 'v009': 'v009_ACC', 'v010': 'v010_ACC', 'v011': 'v011_DIS:v010', 'v012': 'v012_DEP', 'v013': 'v013_DEP', 'v014': 'v014_DEP', 'v015': 'v015_DEP', 'v016': 'v016_DEP', 'v017': 'v017_DIS:v013', 'v018': 'v018_DIS:v016'}[v]/'raw'/name).open()));groups={}
 for x in rows:groups.setdefault((x.get('variant','fp8'),x.get('workload','h7168')), {})[x['name']]=x
 out=[]
 for (var,w),d in groups.items():
  a=float(d.get('current',d.get('baseline'))['latency_us']);b=float(d[candidate]['latency_us'])
  control=float(d.get('direct_geometry',d.get('current',d.get('baseline')))['latency_us'])
  out.append([var,w,f'{a:.3f}',f'{control:.3f}',f'{b:.3f}',f'{a/b:.4f}×',f'{a-b:+.3f}',f'{(a-b)/a*100:+.2f}%'])
 return table(['变体','workload','正式基线 µs','同几何直接读 µs','候选 µs','加速比','减少 µs','延迟降低'],out)
def resource_table(v):
 return table(['方案','同次 trace µs','寄存器/线程','shared bytes','private_total','grid','threads'],[[z['name'],f"{z['trace_median_us']:.3f}",z['resources']['registers_per_thread'],z['resources']['dynamic_shared']+z['resources']['static_shared'],z['resources']['private_total'],f"{z['grid']['x']}×{z['grid']['y']}",z['block']['x']] for z in read(v,'trace_summary.json')])
common="""机器 sc-16g / C500，容器 96933d7d09ab，16G、25% compute quota；MACA 3.7.1.5。
工作目录 `/data/TileOPs-Metax`，环境 `source /data/sc16g-recovery-20261004/env.sh`。
当前正式源为 v010，SHA256 `c6b5dcb13fa171ace058b3f0a50d8361a439eb3335ad4596c539273859831596`。
本轮候选独立保存在 profiles 中。h3072=(T512,K8,H3072)、h7168=(T512,K8,H7168)、prefill=(T4096,K8,H7168)，输入均 BF16。
Base/XSF 输出 BF16；FP8/Quantized 输出 E4M3FN。只改变加载/调度，保留路由判断、K 累加顺序和 v003 编码。
正式计时用项目 `bench_kernel`：10 warmup、50 repeat×3 trials、L2 flush、CUPTI；再交替方案顺序取外层中位数。
trace 是同应用内 10 warmup+1 ROI 的资源/时间辅助证据，不与正式计时混算。
`Compute Instructions` 包含控制等指令，不等于 FLOPs；`Total Cycles` 和 stall 是工具聚合值，不能直接当 wall time。
`Achieved waves`/`Dispatched waves` 是总量，不能当驻留 occupancy。prefill 的 inputs_cloned=false 是项目内存阈值策略。
"""
s14=read('v014','profile_comparison.json')
nt=table(['方案','总指令','内存指令','读 bytes','写 bytes','VLS pipeline stall','WSM stall','shared load/store'],[[z['name'],int(z['counters']['Total Instructions']),int(z['counters']['Memory Instructions']),int(z['counters']['Global Memory Read bytes']),int(z['counters']['Global Memory Write bytes']),int(z['counters']['ISU stall cycles layout']['vls_pipeline_stall']),int(z['counters']['ISU stall cycles layout']['wsm_stall']),f"{int(z['counters']['load instructions'])}/{int(z['counters']['store instructions'])}"] for z in s14])
a=f"""# v014：warp 协作 shared gather（16g）

## 1. 上版本遗留问题

v006 单 buffer 每 K 两次 CTA 同步没有收益；v012 的真实 async K-ring 也慢于 v010。两者都沿用全 CTA 顺序读取每个 expert，没有检验改变生产者任务分配。

## 2. 问题原因分析

当前直接读取每个 x 元素供对应输出列使用一次，shared 缓存不减少算法必需的输入读取。shared 的价值应检验访存任务分配和数据布局，而不能只期待缓存复用。
初版按 warp 动态索引 `topk_to_pos_local[expert]`，设备 trace 报 private_total=36，global write 26,011,200 bytes，远多于输出 3,670,016 bytes；同一几何改为直接读 global position 后 private_total=0、write=3,670,464。动态局部数组访问引入 private 存储是有代码和资源支持的结论，不能把该失败归咎于 shared 本身。

## 3. 本版本解决方案

物理 warp=64；让不同 warp 读取不同 expert。每 lane 向量读 8 个 BF16，512 列块的 global 读取连续。shared 保存原 BF16，不先算乘积；全部 producer 完成后一次 CTA 同步，consumer 按原 K 顺序累加。

## 4. 具体落地策略

- 初版：[cooperative_kernel.py](../scripts/cooperative_kernel.py)；修正版：[cooperative_global_positions.py](../scripts/cooperative_global_positions.py)::get_cooperative_kernel。
- 扫描 tile512/1024、threads128/256/512，并保留全 CTA 逐 K staging 控制。
- 最佳已测配置 tile1024/512 threads：8 warp 对应 K8；shared=K×B×2=16KiB。理论 shared64KiB 允许 4 CTA，4×512=2048 threads，与 runtime thread 上限相等。这只是资源容量模型，不是 measured occupancy。
- 全部 12 组逐字节对比；最佳 factory 跑现有 131 用例，全部通过。factory 只对 BF16、H≥1024 且 H 整除1024使用此方案，其余走原函数；**不声称 131 用例全部都跑了 shared 路径**。见 [validate_best.py](../scripts/validate_best.py)、[JUnit](../raw/correctness_best.xml)。

## 5. Benchmark 对比

{common}

下面为独立确认，5 个外层顺序轮次；包含当前最佳生产配置和相同 tile1024/512 的 direct 控制。

{bench_table('v014','confirmation_16g.csv','cooperative')}

[完整样本](../raw/confirmation_16g.csv) 保留异常值，包括 current 个别 >100µs 样本，没有删除后重算。
初步 expanded 曾出现假大收益，确认以五轮数据为准。FP8 h7168 的候选五个样本均约47.22–47.52µs，而基线中位48.338µs；约2.14%速度提升、2.10%延迟降低。其它11组不推广。
重跑命令：`python profiles/v014_DEP/scripts/run_confirmation.py`；重跑前应另存原 CSV，脚本追加结果。

## 6. Profile 指标变化

{resource_table('v014')}

{nt}

目标计数来自 `raw/native_*/tool_output/1_period0.txt.json`；**report.txt.json 是全应用 aggregate，不能用来评价目标 kernel**。精确来源及字段见 [profile_comparison.json](profile_comparison.json)。
所有 shared 版本 reported access efficiency100%、平均 conflict cycles0；没有 bank conflict 导致本次变慢的证据。
修正局部索引后，private 存储、额外 global 写出和 VLS stall 都下降，构成强对照证据。
最佳 shared 方案仍比正式基线执行更多指令和聚合 stall，但端到端更快；总量不能替代关键路径、并发或实际时长。当前证据支持任务重排有效，不足以量化每个硬件机制贡献。
mcTracer 原始 JSON 在 `raw/trace_comparison`；生成 C++ 在 `codegen/profile_*.cu`。

## 7. 实验总结

shared 方向可行：至少 FP8 h7168 已出现小而可复现的收益。收益来自这套协作读取/布局/几何整体，不能说单独加 shared 就会加速。
Base、XSF、Quantized 和 prefill 在已测配置中没有收益；动态局部数组、线程配置和额外搬运比 bank conflict 更值得优先检查。
v016 寄存器预取在同一 FP8 workload 获得更大收益，因此 shared 候选作为可行性证据保存，不取代更快方案。
"""
(R/'v014_DEP/analysis/v014_cooperative_shared_16g.md').write_text(a)
(R/'v014_DEP/analysis/README.md').write_text('# v014\n\n[完整实验报告](v014_cooperative_shared_16g.md)\n')
s16=read('v016','profile_comparison.json')
nt=table(['方案','总指令','计算类指令','内存类指令','读 bytes','写 bytes'],[[z['name'],int(z['counters']['Total Instructions']),int(z['counters']['Compute Instructions']),int(z['counters']['Memory Instructions']),int(z['counters']['Global Memory Read bytes']),int(z['counters']['Global Memory Write bytes'])] for z in s16 if 'counters' in z])
a=f"""# v016：2/4/8 行寄存器预取（16g）

## 1. 上版本遗留问题

v005 只测两行交替预取；v012_DEP/v015 又表明 shared 和同步可能抵消重叠收益。两行失败不能判断更深的寄存器预取不可行。

## 2. 问题原因分析

直接版本逐 K 执行 load→转换→加权累加；后续 expert 的独立读取没有在源码中提前组织。K=8 时可先请求多行，增大独立访存窗口；代价是存活输入更多，寄存器增加，可能降低驻留或发生 private 溢出。
这条路线检验加载调度/指令级并行；它不减少必需 x 流量，也不提高算术强度。

## 3. 本版本解决方案

用输入 dtype 的 register fragment `[prefetch_rows,tile_hidden]`。先预取 depth 行，消费 k 后请求 k+depth 到同槽；depth=8,K=8 时全部行先加载，再按原 K 顺序累加，无 shared/CTA 同步。
这是寄存器中的软件预取；不把它叫成 global→shared 硬件 async copy。

## 4. 具体落地策略

[get_prefetch_kernel](../scripts/prefetch_kernel.py) 增加 prefetch_rows=2/4/8。h3072、h7168 的 SF 路径保持 tile512/128，与当前生产几何相同；prefill 对照 tile1024/128。Base/XSF 的 current 是整行256线程，因此同时保留同几何 direct 控制，避免把几何影响混入预取效果。
生成 C++ `profile_register8.cu` 明确先出现 `first<8` 的8B向量读取，再出现 K 循环，BF16 packed 转换保留。这里只验证生成源结构，**没有以 C++ 顺序假定最终机器指令严格顺序**。
[validate.py](../scripts/validate.py) 将现有测试 factory 替换为深度8，在全部131用例通过；[JUnit](../raw/correctness_register8.xml) tests131/failures0/errors0/skipped0。覆盖不同 dtype、四变体、无效路由等现有条件，benchmark另有逐字节对照。

## 5. Benchmark 对比

{common}

初步 h7168 同几何测试：direct48.328µs，depth2 48.353，depth4 49.628，depth8 44.698。只有 depth8 获得收益。
12组扩展对照：

{bench_table('v016','expanded_register8_16g.csv','register8')}

独立进程重新生成输入、5外层轮次确认四个有收益的SF组合：

{bench_table('v016','confirmation_register8_16g.csv','register8')}

速度提升与延迟降低分开列出；例如 FP8 h7168 加速8.57%，延迟降低7.89%。全样本及计时口径在 CSV measurements 中。
命令：`python profiles/v016_DEP/scripts/run_confirmation.py`；完整扩展：`python profiles/v016_DEP/scripts/run_expanded.py`。脚本追加 CSV，重跑前另存原结果。

## 6. Profile 指标变化

{resource_table('v016')}

同应用资源显示 depth8 registers/thread 12→22，shared/private 均0，grid/block不变；没有发现该配置溢出。
本轮原始 register2/4/8 大事件集计数与零shared代码、算法流量冲突，已明确作废，见 [native_validity.json](../meta/native_validity.json)。不拿这些异常 stall/shared 数据解释收益。
隔离 `--custom --per-kernel --kernelnames reduce_fused_kernel_kernel --counts 1`，仅采5个核心指标后的深度8结果有效：

{nt}

正式理论输入58,720,256bytes，输出3,670,016bytes；重采 read58,798,432、write3,670,336相差约0.13%/0.009%，与目标流量一致。说明收益不是少读专家输出或少写结果。
内存类指令下降约17.7%、总指令增加约8.0%，因此“指令少所以快”也不是充分解释；提前独立读取是实现和计时共同支持的机制，缺乏有效候选 stall 计数来进一步分解。
原始重采 [per-kernel JSON](../raw/native_register8_minimal/tool_output/1_reduce_fused_kernel_kernel.txt.json)、[命令](../meta/recollection_job.json)、[汇总](profile_comparison.json) 均保留。baseline来自同版本有效目标周期数据，不用全应用aggregate。

## 7. 实验总结

更深的寄存器预取可行，否定“两行失败因此流水线没用”的外推。收益确认在 SF=true、T512、K8、BF16、H3072/7168；其它测到的组合无收益，不推广。
正式文件仍是 v010；候选源码、131用例结果、12组数据和独立确认都已保存，可按这些shape选择调用。正式集成不属于本次可行性证明的前提。
"""
(R/'v016_DEP/analysis/v016_register_prefetch_16g.md').write_text(a)
(R/'v016_DEP/analysis/README.md').write_text('# v016\n\n[完整实验报告](v016_register_prefetch_16g.md)\n\n[两条路线总复盘](pipeline_shared_conclusions_16g.md)\n')
print('wrote v014_DEP/v016 reports')

status=json.loads((R/'v015_DEP/meta/profile_status.json').read_text())
if status['stage']!='done':
 print('v015 profile still live; final reports will be built after completion')
 raise SystemExit(0)
import re
def native(p):
 d=json.loads(p.read_text());out={}
 for section in d.values():
  if not isinstance(section,list):continue
  for x in section:
   if not isinstance(x,dict) or 'name' not in x:continue
   v=x['value']
   if isinstance(v,dict):out[x['name']]=v.get('data',v)
   else:
    m=re.match(r'[-+]?\d+(?:\.\d+)?',str(v).replace(',',''))
    out[x['name']]=float(m.group()) if m else v
 return out
s15=read('v015','trace_summary.json')
for z in s15:
 if z['name'] in ('stream_async','group2_async'):
  f=R/f"v015_DEP/raw/native_{z['name']}/tool_output/1_reduce_fused_kernel_kernel.txt.json"
  z['counters']=native(f)
  c=z['counters']
  assert abs(c['Global Memory Read bytes']/58720256-1)<0.01,(z['name'],c)
  assert abs(c['Global Memory Write bytes']/3670016-1)<0.01,(z['name'],c)
  z['native_validity']='target_traffic_consistent'
(R/'v015_DEP/analysis/profile_comparison.json').write_text(json.dumps(s15,indent=2)+'\n')
nt=table(['方案','总指令','内存指令','读 bytes','写 bytes','VLS pipeline stall','WSM stall','shared load/store','efficiency/conflict'],[[z['name'],int(z['counters']['Total Instructions']),int(z['counters']['Memory Instructions']),int(z['counters']['Global Memory Read bytes']),int(z['counters']['Global Memory Write bytes']),int(z['counters']['ISU stall cycles layout']['vls_pipeline_stall']),int(z['counters']['ISU stall cycles layout']['wsm_stall']),f"{int(z['counters']['load instructions'])}/{int(z['counters']['store instructions'])}",f"{z['counters']['shared memory access efficiency']}%/{z['counters']['average conflict cycles per instruction']}"] for z in s15 if 'counters'in z])
def simple_bench(name):
 rows=list(csv.DictReader((R/'v015_DEP/raw'/name).open()));base=float(rows[0]['latency_us'])
 return table(['方案','正式 µs','相对直接基线','延迟变化 µs'],[[z['name'],f"{float(z['latency_us']):.3f}",f"{base/float(z['latency_us']):.4f}×",f"{float(z['latency_us'])-base:+.3f}"] for z in rows])
a=f"""# v015：hidden 双缓冲与分组流水线（16g）

## 1. 上版本遗留问题

v012 的 K-ring 每次只与当前 expert 的少量 FMA 重叠，窗口短。v013 删 CTA 同步或换 warp 同步均错误，不能采用其速度。检验能否用整个 hidden 块的归约和 FP8 编码隐藏下一块读取。

## 2. 问题原因分析

一 CTA 处理整行可复用 metadata，但减少 CTA 数。以512列为块，v010 是512×14=7168 CTA；整行stream只有512 CTA，硬件报告104 multiprocessor。直接stream控制已经84.762µs，比48.236µs基线慢，说明不能只归因于 async/shared。
shared双缓冲16KiB，在shared64KiB/MP的容量模型下最多4 CTA；128线程/CTA对应最多512线程，而 runtime MP上限2048。这是容量上界推算，**不是测得25% occupancy，也不是容器compute quota**。实际还受分配粒度及其它资源影响。

## 3. 本版本解决方案

两槽shared `[2,K,B]`；wait当前块全部K ticket、CTA同步后，先发起下一hidden块全部K读取，接着当前块归约和编码。保持 K 次序、路由有效性和原编码，不删除必要同步。
控制：直接读取mode0、同步doublebuffer mode1、真实nativeasync mode2。随后每CTA只负责2/4块，恢复grid并隔离并行度因素。

## 4. 具体落地策略

[hidden_kernel.py](../scripts/hidden_kernel.py)::get_hidden_kernel 与 [grouped_hidden_kernel.py](../scripts/grouped_hidden_kernel.py)，tile512/128 threads；ticket `[2,K]` 用 uint64 IR+b64vectype backend注解。
初版缺少预取 pos 上界的 `T.assume`，safe memory pass 在 async rewrite 报错；补上与原kernel相同的合法输入契约后通过。失败源 [hidden_missing_assume.py](../codegen/hidden_missing_assume.py) 保留。
已测 FP8 h7168 全部候选与基线逐字节一致，profile脚本再复核；**没有为此失败候选声称跑完整131用例**。

## 5. Benchmark 对比

{common}

整行stream：

{simple_bench('initial_fp8_h7168_16g.csv')}

分组控制：

{simple_bench('grouped_fp8_h7168_16g.csv')}

与同步双缓冲158.208相比，真实async84.265µs有明显改善，但仍未击败正式基线。
恢复grid之后 group2/4直接读49.905/48.727µs已接近基线47.903，而async仍63.601/61.722µs，证明已测几何的额外shared/ticket/同步成本没有被隐藏。
各组来自各自配对run，不跨run拼一个“精确加速比”；三外层样本在CSV保留。
命令：`python profiles/v015_DEP/scripts/initial_compare.py`，`python profiles/v015_DEP/scripts/compare_grouped.py`。

## 6. Profile 指标变化

{resource_table('v015')}

{nt}

原始独立目标JSON在 `raw/native_*/tool_output/1_reduce_fused_kernel_kernel.txt.json`，有效流量均接近输入58.72MB/输出3.67MB。global→shared专用copy不走普通shared store计数，因此store=0不等于shared未写。
资源 private_total=0，shared访问efficiency100%、平均conflict0：未见spill/bank conflict证据。较高shared容量、CTA数量变化、额外shared读取和控制才是该已测方案的代价。
具体驻留率和等待隐藏比例没有直接硬件measurement，不由模型或stall总数伪造；资源和配对控制足以限定这里的失败原因，而不排除其它producer/consumer设计。
profile重跑：`python profiles/v015_DEP/scripts/collect_profiles.py`。采用relative mcTracer odname；JSON存在及目标事件数量已核实，不仅依赖exit0。

## 7. 实验总结

这套整行/2块/4块hidden双缓冲不采用：正确但比正式基线慢。async相对同结构sync有效，证明copy路径确实可用；失败不能外推所有pipeline不可行。
v014 warp协作和v016寄存器预取已经给出反例。这里记录失败范围、控制实验和硬件资源约束，不宣称无论如何不可能。
"""
(R/'v015_DEP/analysis/v015_hidden_pipeline_16g.md').write_text(a)
(R/'v015_DEP/analysis/README.md').write_text('# v015\n\n[完整实验报告](v015_hidden_pipeline_16g.md)\n')
a="""# 流水线与 shared memory：v005_DEP/v006 后续复盘（16g）

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
"""
(R/'v016_DEP/analysis/pipeline_shared_conclusions_16g.md').write_text(a)
print('wrote v015 and complete conclusions')
