# v012：K shared ring 的同步/异步对照

v011已经验证MCA异步拷贝可正确执行。保持tile512、128线程，先对FP8 H7168进行同几何比较。
控制组：v010；旧式single shared两次CTA同步；同步双缓冲；真实async两缓冲/三缓冲。
每轮消费前wait当前ticket，一次CTA同步兼顾当前数据可见性和上一轮slot复用安全。
消费前发起S-1距离的未来拷贝，保持K累加顺序。
预测async相对sync环形缓冲降低数据等待；更深prefetch增大in-flight量，但也增smem和ticket寄存器。
可证伪：生成代码没有copy-before-current-FMA；正确性不一致；同几何时延不降或native访存等待不改善。
初步结果不能否定hidden方向流水线、warp协作/shared元数据/输出重排；后续继续覆盖。
