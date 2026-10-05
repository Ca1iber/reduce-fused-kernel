# v017：删除 CTA barrier 后出错的根因（16g）

## 结论

**当前 MXCC 工具链对有条件执行的异步拷贝，其 ticket 完成依赖/访存等待生成不正确；`wait(ticket)` 会在目标 global→shared 拷贝仍未完成时继续。后续 shared load 因而读到旧值。额外 `T.sync_threads()` 生成更严格的访存完成等待，掩盖了这一问题。**

本轮最小反例进一步定位到第二次 copy 的条件分支：只给 copy B 加条件也能触发；只给 copy A 或 wait 加条件不触发。无须浮点归约、FP8 编码、缓冲区复用或跨 warp 共享就能复现。
“每线程读自己拷的4个BF16”在地址层面是对的；错误的是把源代码里的 ticket wait 当成已可靠兑现的完成保证。

## 1. 上版本遗留问题

v013 无 CTA / warp 同步均出现 byte mismatch，但仅留泛化错误，没有记录首次错误的 expert/数据内容。后来 warp 实验覆盖了同名 codegen 文件，旧 `async_ring*_owner.cu` 已是 `__syncwarp()` 版本，不能当原无同步版本分析。本轮从各脚本源重新生成，每个变体独立命名。
此前解释曾猜 shared 可见性、slot 复用；这些只是待验证假设，此报告替代“具体原因未知”的结论。

## 2. 问题原因分析：最小反例及机器指令

### 2.1 排除算法和线程间复用

- 原 reduce pipeline 比较 K1/K2/K8、两/三缓冲、FP32/FP8输出、none/warp/CTA/fence，共48配置，每个3次。K1全正确；K2无 CTA 已错，K2没有覆盖旧slot，排除slot复用是必要触发条件。
- 64线程单warp、tile256、K2：无同步与warp同步均错，CTA版本5次全正确。排除“必须等另一个warp读完”是本次根因。
- 优化 LLVM IR 中 async request/wait/shared load 仍保持该逻辑顺序；没有在这一阶段找到 load 被移到 wait 前的证据。

### 2.2 只有整数copy也能读到旧值

[conditional_copy.py](../scripts/conditional_copy.py)：7168个CTA，每CTA128线程，shared两行256个int32。每线程copy2个int32=8B，与原配置每线程4个BF16=8B相同；两slot的间隔都1024 bytes。先将shared填`-999`并CTA同步，再发两次copy、wait A、输出A、wait B、输出B。
所有运行输入p=[0,1]，copy都实际执行，逻辑上每线程只读自己负责的位置，没有slot复用。

失败的5次运行，A全部正确，B错误数量依次为：585312, 495200, 439520, 507456, 450656。
**所有错误元素都等于初始化值 -999，wrong_B == poison_count**。这是拷贝还没覆盖目标shared就读取它的直接证据，而非舍入差别或误读其他expert。
同代码去掉条件copy，7168CTA五次都正确；加CTA barrier也五次都正确。单CTA各组合都正确，说明小探针会因延迟不足而漏掉问题；不能拿单copy探针证明复杂条件pipeline可靠。

### 2.3 条件分支单因素隔离

[mask_copy.py](../scripts/mask_copy.py)及[mask_individual.py](../scripts/mask_individual.py)按bit分别控制copy A/B和wait A/B是否放在条件分支，所有运行输入均有效。某些诊断配置不保证无效路由行为，不能用于生产。

| mask | 条件 copy A | 条件 copy B | 条件 wait A | 条件 wait B | 第二行错误元素数（5次） |
| --- | --- | --- | --- | --- | --- |
| 0 | False | False | False | False | 0, 0, 0, 0, 0 |
| 1 | True | False | False | False | 0, 0, 0, 0, 0 |
| 2 | False | True | False | False | 614048, 525984, 554208, 485440, 516928 |
| 3 | True | True | False | False | 572800, 531584, 562880, 552384, 588800 |
| 4 | False | False | True | False | 0, 0, 0, 0, 0 |
| 7 | True | True | True | False | 513632, 495168, 522976, 526368, 506592 |
| 8 | False | False | False | True | 0, 0, 0, 0, 0 |
| 11 | True | True | False | True | 608768, 566912, 591104, 564960, 584544 |
| 12 | False | False | True | True | 0, 0, 0, 0, 0 |
| 15 | True | True | True | True | 474016, 489280, 466432, 485120, 474624 |

**mask2（只条件化copy B）已足以触发，mask1（只条件化copy A）5次全部正确；wait条件不是必要因素。**

### 2.4 后端等待指令不正确

SDK安装源码：`/opt/maca/mxgpu_llvm/lib/clang/19/include/__clang_maca_device_functions.h`：
- 3775–3780：`memcpy_async<8>`调用`__builtin_mxc_ldg_b64_bsm`。
- 3837–3845：`barrier_arrive_and_wait`默认scope为`thread_scope_block`，调用`__builtin_mxc_barrier_and_wait2`。所以它本身也生成block `BARRIER`；之前简单讲成“每线程凭据”不够精确。
- 57–62、73–75：`__syncthreads()`是release fence + block barrier + acquire fence。额外效果包含访存等待，不能只当线程到齐。

直接用与JIT相同的`-O3 -lineinfo`和xcore1000目标编译，抓取MetaXGPU insert-arrivecnts前后及最终machine dump：

| 最小版本 | 源码中的第二次wait | 最终指令观察 | 正确性 |
|---|---|---|---|
| 无条件copy | wait B存在 | `ARRIVE 65` + `BARRIER`，然后LDS B | 正确 |
| copy A/B都有条件 | wait B存在 | `ARRIVE 66` + `BARRIER`，然后LDS B | 错，读到-999 |
| 只copy B有条件(mask2) | wait B无条件存在 | 第二次只剩`BARRIER`，未插入对应global完成等待，然后LDS B | 错，读到-999 |
| 有条件copy + CTA | wait B存在，后接syncthreads | 额外`ARRIVE 4160` + `BARRIER` + `ARRIVE 4096`，然后LDS B | 正确 |

以上数值是直接观察的编码。按编译器GVM等待计数和强制等待对照解释，66比65允许更多未完成请求；mask2甚至漏掉所需等待。没有依赖自制ISA规范推导硬件理论值。
指令和source debug-location将错误wait对应回SDK第3845行及生成源实际调用。关键excerpt见[全条件copy](conditional_b7168_if1_order0_cta0_instructions.txt)、[无条件copy](conditional_b7168_if0_order0_cta0_instructions.txt)、[只条件copy B](mask2_instructions.txt)。原dump在codegen，精确命令在[matched_mir_commands.json](../meta/matched_mir_commands.json)。
原reduce K2的无CTA版本第二次生成`ARRIVE 65`；加CTA后该处成为更严格的等待。与toy的绝对编码不同，不跨kernel把某数字当固定ticket编号。

这里定位到的是**MXCC的条件async-copy ticket依赖/等待插入问题**，没有取得MXCC后端实现源码，因此不声称已定位到vendor内部C++函数的具体错误行。最小反例、最终指令和修复对照足以说明原kernel为何出错。

## 3. 本版本解决方案：用诊断开关验证因果

TileLang现有`TL_DEVICE_COMPILE_FLAGS`传入`-mllvm --metaxgpu-force-arrive`，强制每次MEM指令后等待。保留原路由/归约/编码，sync_mode=0，无额外CTA barrier。
该开关让原K2/K8五次均正确；两缓冲版本现有131用例全过；三缓冲四变体和含无效路由均5次逐字节正确。
**这只是排障workaround，访存被强制等待，会牺牲pipeline重叠；没有将其作为优化接入正式kernel。**

## 4. 具体落地策略及复现

目录 `/data/TileOPs-Metax/profiles/v017`，机器sc-16g、容器96933d7d09ab、SDK3.7.1.5、MXCC/Clang19、xcore1000。环境先`source /data/sc16g-recovery-20261004/env.sh`。

- `python profiles/v017_DIS:v013/scripts/reproduce.py`：原kernel48配置，分别保存codegen，排除FP8/复用。
- `python profiles/v017_DIS:v013/scripts/conditional_copy.py`：带初始化哨兵的整数最小反例。
- `python profiles/v017_DIS:v013/scripts/mask_copy.py`及`mask_individual.py`：逐分支隔离。
- `python profiles/v017_DIS:v013/scripts/one_warp.py`：跨warp假设反证。
- `python profiles/v017_DIS:v013/scripts/diagnose_options.py`：fence/sharedbarrier、MD、O0、强制等待对照。
- `python profiles/v017_DIS:v013/scripts/validate_workaround.py`：existing131测试；`verify_ring3.py`：三缓冲四变体及无效路由。

以上GPU任务顺序执行，没有同时跑profile/benchmark。编译器dump只在CPU执行，不下载工具、不修改SDK。两个最初shell命令引用错误在执行前即失败，已修正；storemd=-1因uint参数拒绝，没有将其当作有效MD实验。

## 5. 正确性对比结果

本轮定位原因，不测速度，不用“正确但慢”遮盖问题。

| 单因素改变 | K2 错误元素数（5次） | K8 错误元素数（5次） |
| --- | --- | --- |
| sharedbarrier | 3665710, 3666421, 3667003, 3666447, 3587459 | 3670016, 3670016, 3670016, 3670016, 3670016 |
| forcezero | 0, 0, 0, 0, 0 | 3667153, 3663942, 3664313, 3664012, 3663933 |
| forcearrive | 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0 |
| restoremd | 3667386, 3666504, 3666494, 3666385, 3666217 | 3670016, 3670016, 3670016, 3670016, 3670016 |
| O0 | 3667394, 3666491, 3665898, 3666356, 3666369 | 3670016, 3670016, 3670016, 3670015, 3670016 |
| forcebsm | 3667378, 3666368, 3666378, 3666298, 3666325 | 3670016, 3670016, 3670016, 3670016, 3670016 |
| forcegvm | 3667232, 3666342, 3666340, 3666458, 3666370 | 3670016, 3670016, 3670016, 3670016, 3670016 |

forcezero只把已有ARRIVE阈值设为0，K2成功但K8仍失败；forcearrive额外在各MEM后插入等待，两者不可混为一谈。debug-counter的forcebsm/gvm在此安装版本未修复，不据此认定对应硬件队列无关。
[131用例JUnit](../raw/correctness_workaround.xml) tests131/failures0/errors0/skipped0；[三缓冲数据](../raw/ring3_workaround.json)8种组合×5次，全部wrong_bytes0。没有把后者说成三缓冲也跑了全部131用例。

## 6. Profile/代码证据与被排除的解释

- [reproduce.json](../raw/reproduce.json)：原kernel FP32输出也失败，排除FP8编码。
- [conditional_copy.json](../raw/conditional_copy.json)：无归约整数copy、无slot复用仍失败，读到完整哨兵值。
- [mask_copy.json](../raw/mask_copy.json)、[单copy条件](../raw/mask_individual.json)：只第二次copy的条件分支足够触发。
- [one_warp.json](../raw/one_warp.json)：仅一warp仍错，warp同步不能修复。
- [instruction_comparison.json](instruction_comparison.json)：最终wait缺失/过宽，额外CTA插入完整访存等待。
- [options_results.json](../raw/options_results.json)：仅改变后端等待策略即可恢复正确；MD restore、普通fence、sharedbarrier、O0均未修复本实例。

没有用roofline/HBM利用率解释正确性问题，也没有将源码顺序等同机器层可靠等待。

## 7. 实验总结

地址映射和每线程所有权本身没有被本次实验否定；**源代码里的ticket wait没有被当前后端可靠翻译成目标拷贝的完成等待**，才是删除额外CTA后出错的直接原因。额外CTA起到了保守访存等待的补救作用。
当前生产v010不含此async pipeline，无需为此改动它。探索版在默认工具链下继续保留CTA barrier；若未来修编译器或写精确的等待封装，必须重新检验最终指令和正确性，不能只删除同步。
本结论限定当前SDK/编译器和已测控制流；未断言所有版本或所有async接口都存在相同问题。
