import tilelang
from tilelang import language as T

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_tiny_group_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf, tokens_per_cta=1, num_threads=128):
    assert num_threads % tokens_per_cta == 0
    threads_per_token = num_threads // tokens_per_cta
    assert hidden % threads_per_token == 0
    elements_per_thread = hidden // threads_per_token
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"
    vector_store = use_fp8_bits
    copy_width = min(8, elements_per_thread)
    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')

    @T.prim_func
    def tiny_group(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_cta), threads=num_threads) as pid:
            reduced = T.alloc_fragment((tokens_per_cta, hidden), T.float32)
            encoded_values = T.alloc_fragment((tokens_per_cta, hidden), out_dtype)
            positions = T.alloc_fragment((tokens_per_cta, num_topk), T.int32)
            weights = T.alloc_fragment((tokens_per_cta, num_topk), T.float32)
            sf_var = T.alloc_var(T.float32)
            T.annotate_layout({
                reduced: T.Fragment((tokens_per_cta, hidden),
                    forward_thread_fn=lambda r, i: r * threads_per_token + i // elements_per_thread,
                    forward_index_fn=lambda r, i: i % elements_per_thread),
                encoded_values: T.Fragment((tokens_per_cta, hidden),
                    forward_thread_fn=lambda r, i: r * threads_per_token + i // elements_per_thread,
                    forward_index_fn=lambda r, i: i % elements_per_thread),
                positions: T.Fragment((tokens_per_cta, num_topk), replicate=threads_per_token,
                    forward_thread_fn=lambda r, k, rep: r * threads_per_token + rep,
                    forward_index_fn=lambda r, k: k),
                weights: T.Fragment((tokens_per_cta, num_topk), replicate=threads_per_token,
                    forward_thread_fn=lambda r, k, rep: r * threads_per_token + rep,
                    forward_index_fn=lambda r, k: k),
            })
            T.clear(reduced)
            if with_sf:
                sf_var = sf[0]
            T.copy(token_topk_to_pos[pid * tokens_per_cta:(pid + 1) * tokens_per_cta, :], positions)
            if with_weights:
                T.copy(topk_weights[pid * tokens_per_cta:(pid + 1) * tokens_per_cta, :], weights)
            for k in T.unroll(num_topk):
                for r, i in T.Parallel(tokens_per_cta, hidden):
                    token = pid * tokens_per_cta + r
                    if token < num_tokens:
                        pos = positions[r, k]
                        T.assume(pos < num_expanded_tokens)
                        if pos >= 0:
                            scale = T.alloc_var(T.float32)
                            scale = 1
                            if with_weights:
                                scale = weights[r, k]
                            if with_x_sf:
                                scale *= x_sf[pos]
                            reduced[r, i] += x[pos, i] * scale
            for r, i in T.Parallel(tokens_per_cta, hidden):
                token = pid * tokens_per_cta + r
                if token < num_tokens:
                    if use_fp8_bits:
                        value = reduced[r, i] * sf_var
                        raw = T.reinterpret(value, "uint32")
                        # magnitude 清除掉 FP32 最高位的符号，得到绝对值的编码
                        # sign 把符号从 PF32 的第 31 位, 搬到 FP8 的第 7 位, 结果是 0 或者 0x80
                        magnitude = raw & T.uint32(0x7fffffff)
                        sign = (raw >> 24) & T.uint32(0x80)
                        # Keep three mantissa bits, rounding nearest with even ties.
                        # FP32 有 23 位小数, 而 FP8 只有 3 位, 所以需要舍弃 20 位
                        # 这个加法是在根据被丢掉的位决定是否进一位
                        # &1 处理恰好在中间的情况, 按照最近偶数规则舍入
                        # 整体算出一个普通大小的数对应的 FP8 编码, 暂时
                        normal = (
                            (magnitude + T.uint32(0x7ffff)
                             + ((magnitude >> 20) & T.uint32(1))) >> 20
                        ) - T.uint32(0x3c0)
                        # E4M3 subnormals have a 2^-9 step. The FP32 bias add
                        # rounds to that step without the SDK's FP64 conversion.
                        # FP8 非正规数的编码
                        denormal = T.reinterpret(
                            T.reinterpret(magnitude, "float32") + T.float32(16384.0),
                            "uint32",
                        ) - T.uint32(0x46800000)
                        # 如果特别小用非正规编码, 如果特别大饱和到 FP8 最大有限值 448, 普通范围就用正规数编码
                        encoded = T.Select(
                            magnitude < T.uint32(0x3c800000), denormal,
                            T.Select(magnitude >= T.uint32(0x43e00000),
                                     T.uint32(0x7e), normal),
                        )
                        # Match the original SDK SATFINITE conversion: clamp
                        # overflow/Inf, canonicalize NaN, preserve signed zero.
                        # 识别 NaN, 给其他结果拼回正负号
                        final_encoded = T.Select(
                            magnitude > T.uint32(0x7f800000),
                            T.uint32(0x7f), encoded | sign,
                        )
                        if vector_store:
                            encoded_values[r, i] = T.reinterpret(
                                final_encoded.astype("uint8"), out_dtype,
                            )
                        else:
                            out[pid * tokens_per_cta + r, i] = T.reinterpret(
                                final_encoded.astype("uint8"), out_dtype,
                            )
                    else:
                        out[pid * tokens_per_cta + r, i] = T.Select(
                            with_sf, reduced[r, i] * sf_var, reduced[r, i],
                        )
            if vector_store:
                T.copy(encoded_values, out[pid * tokens_per_cta:(pid + 1) * tokens_per_cta, :], coalesced_width=copy_width)
    return tiny_group
