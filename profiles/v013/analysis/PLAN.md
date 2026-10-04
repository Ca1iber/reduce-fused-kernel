# v013：消除不必要的跨线程同步

v012设备代码显示tile512/threads128的copy dst与消费load均为stage_slot*512+threadIdx.x*4，每线程只消费自己拷贝的4个BF16。
因此对应ticket wait后不需要CTA barrier，slot重用也由同线程顺序保证。
保持相同tile/threads/stages，以owner_only开关做对照。
预测减少同步/控制指令及等待，减少与同步双buffer之间的差异。
生成代码地址映射若改变、出现跨线程读写、正确性不一致，方案不成立；不能把此条件外推到任意layout。
