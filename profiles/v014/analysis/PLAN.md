# v014：warp协作gather + shared输入缓存

C500 warp_size=64；分别使用2/4个warp（128/256线程）分担K个expert的读取。
每warp以16byte连续读取一个expert行的512列块；完成所有K输入后一次CTA同步，按原K顺序归约。
shared仅存输入dtype，不预先计算加权贡献，保持原FP32运算顺序。
控制组all-K staging仍由全部线程逐K加载，隔离warp任务分配的效果。
预测每warp的global-read依赖链缩短，但smem容量与消费端布局可能限制收益；比较当前直接读与同几何控制组。
