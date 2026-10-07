import tilelang
from tilelang import language as T

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_tiny_local_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf, tokens_per_cta=1, num_threads=128, num_tokens_hint=32):
    assert num_threads % tokens_per_cta == 0
    threads_per_token = num_threads // tokens_per_cta
    assert hidden % threads_per_token == 0
    elements_per_thread = hidden // threads_per_token
    assert elements_per_thread <= 8
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"
    vector_store = use_fp8_bits
    num_tokens = T.dynamic('num_tokens') if num_tokens_hint is None else num_tokens_hint
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    guard_tokens = num_tokens_hint is None or num_tokens_hint % tokens_per_cta != 0

    @T.prim_func
    def tiny_local(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, tokens_per_cta), threads=num_threads) as pid:
            tx = T.get_thread_binding()
            token = pid * tokens_per_cta + tx // threads_per_token
            col = (tx % threads_per_token) * elements_per_thread
            reduced = T.alloc_local((elements_per_thread,), T.float32)
            loaded = T.alloc_local((elements_per_thread,), in_dtype)
            encoded_values = T.alloc_local((elements_per_thread,), out_dtype)
            positions = T.alloc_local((num_topk,), T.int32)
            weights = T.alloc_local((num_topk,), T.float32)
            sf_var = T.alloc_var(T.float32)
            scale = T.alloc_var(T.float32)
            pos = T.alloc_var(T.int32)
            if not guard_tokens or token < num_tokens:
                for j in T.unroll(elements_per_thread):
                    reduced[j] = T.float32(0)
                if with_sf:
                    sf_var = sf[0]
                for k in T.Vectorized(num_topk):
                    positions[k] = token_topk_to_pos[token, k]
                    if with_weights:
                        weights[k] = topk_weights[token, k]
                for k in T.unroll(num_topk):
                    pos = positions[k]
                    T.assume(pos < num_expanded_tokens)
                    if pos >= 0:
                        scale = 1
                        if with_weights:
                            scale = weights[k]
                        if with_x_sf:
                            scale *= x_sf[pos]
                        for j in T.Vectorized(elements_per_thread):
                            loaded[j] = x[pos, col + j]
                        for j in T.unroll(elements_per_thread):
                            reduced[j] += loaded[j] * scale
                for j in T.unroll(elements_per_thread):
                    if use_fp8_bits:
                        value = reduced[j] * sf_var
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
                            encoded_values[j] = T.reinterpret(
                                final_encoded.astype("uint8"), out_dtype,
                            )
                        else:
                            encoded_values[j] = T.reinterpret(
                                final_encoded.astype("uint8"), out_dtype,
                            )
                    else:
                        encoded_values[j] = T.Select(
                            with_sf, reduced[j] * sf_var, reduced[j],
                        )
                for j in T.Vectorized(elements_per_thread):
                    out[token, col + j] = encoded_values[j]
    return tiny_local
