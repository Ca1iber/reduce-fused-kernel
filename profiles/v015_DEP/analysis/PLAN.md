# v015：hidden方向真正重叠

一CTA处理一个token，按hidden块流式执行，减少重复metadata与整行累加寄存器。
控制0直接读每块，控制1同步双缓冲，候选2预取下一hidden块的全部K输入，在当前块归约+FP8编码期间完成拷贝。
与K-ring不同，重叠窗口覆盖整个归约和FP8 epilogue。仍保留CTA同步以保证数据可见性和slot复用。
预测隐藏访存延迟但shared/ticket容量可能降低驻留，并且CTA总数下降；同时采集资源、spill和指令变化。
先比较512列/128线程，后续按资源证据调节块与producer/consumer任务分配。
