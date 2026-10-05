# v016：更深寄存器预取

v005仅2-row局部预取，v012真实async K环形仍比基线慢。
此轮比较2/4/8-row输入寄存器预取，8-row先发起全部K读取再归约，保持原K运算顺序。
消除shared搬运和CTA同步，检查设备代码是否保留前移load、register/private代价、与当前同几何的正式计时。
第一次仅FP8 h7168，后续依profile决定参数/全workload覆盖，不以单配置否定方向。
