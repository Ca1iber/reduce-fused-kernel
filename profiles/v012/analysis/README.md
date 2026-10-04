# v012：真实K异步shared ring（sc-16g）

## 机制

同tile512/128线程，比较直接读、原单shared双barrier、同步ring2、真实async ring2/ring3。
使用TileLang maca_async_copy，生成真实memcpy_async<8>；每轮wait当前ticket，CTA同步后发起未来拷贝，再按原K顺序累加。
同步点同时保证shared可见性及上一轮slot读完。输入仍BF16，不预先舍入加权乘积。

## 编译问题

void类型的barrier array在此循环触发SIGFPE，isolated exit136，未发生OOM。
改用uint64 IR类型并保留backend b64vectype注解后可编译执行。单copy探针已有同类注解修复；没有改SDK/编译器。
失败和成功源码均保留，见codegen和meta/compiler_crash.json。

## 计时与正确性

FP8 T512 K8 H7168 BF16，同轮项目cupti / L2flush / 10warmup / 50repeat×3trials，各方案再测三轮取中位数。
直接读47.862us；单shared51.205；同步ring2 50.770；async ring2 49.531；async ring3 50.381。
所有输出逐字节一致；async ring2和ring3各131个现有正确性用例通过（包括4变体/3dtype/无效路由）。
第三轮baseline57.267、single shared53.903是明显波动；保留原值，三轮中位数结果只是初步方向比较，不作为最终稳定提升。

## Profile对照

同一mcTracer进程每方案11次，寄存器/动态shared/private：
直接读12/0/0；单shared12/1024/0；同步ring2 14/2048/0；async ring2 14/2048/0；async ring3 12/3072/0。
对应trace中位49.920、53.248、52.480、51.456、53.504us；只与同次trace比较，不混作正式计时。

native custom+kernel过滤，各方案global read约58.8MB、write约3.67MB，与目标kernel一致。
直接读Total Instructions约3.96M，Memory约0.241M；同步ring2约5.15M/0.426M；async ring2约5.91M/0.312M。
同步ring2 shared load/store约113632/113632；async ring2约113584/0。
async从专用copy路径写shared，store计数0不意味着没有shared写入。
WSM stall同步ring2约358782，async ring2约56760；vls_pipeline stall两者均约192k。
所有shared方案reported efficiency100%、平均冲突cycles0，本次没有bank-conflict造成变慢的证据。
总体async确实降低shared store/WSM等待，但增加的计算/控制类指令和shared load仍使它慢于直接读取。
不能把Compute Instructions等同FP32 FLOPs，不能把Achieved waves/Dispatched waves总数当驻留占用率。
完整各方案数值、原始计数和资源保存在analysis/profile_comparison.json与raw。

## 结论与后续

K-ring比简单同步shared更有效，但此配置尚未击败正式基线。该结果不能排除hidden流水线、warp协作、shared元数据/输出整理等机制。
生成代码保留packed BF16转换，未验证原先猜测的scalar化；继续测试必要同步与更合理任务分配。
正式kernel保持v010。原始工作副本和结果均在/data，后续继续。
