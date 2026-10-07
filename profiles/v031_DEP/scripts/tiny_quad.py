import tilelang
from tilelang import language as T
from tilelang.language.builtin import shfl_sync

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_quad_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf, num_threads=1024, role_layout="adjacent"):
    assert num_topk == 2
    assert num_threads in (256,512,1024)
    elements_per_quad = hidden // (num_threads // 4)
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"
    vector_store = use_fp8_bits

    @T.prim_func
    def tiny_quad(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(32, 2), T.float32],
        token_topk_to_pos: T.Tensor[(32, 2), T.int32],
        out: T.Tensor[(32, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(32, threads=num_threads) as token:
            tx = T.get_thread_binding()
            role = tx % 4 if role_layout == "adjacent" else (tx % 64) // 16
            expert = role % 2
            column_id = tx // 4 if role_layout == "adjacent" else (tx // 64) * 16 + tx % 16
            group_lane = ((tx % 64) // 4) * 4 if role_layout == "adjacent" else tx % 16
            lane_step = 1 if role_layout == "adjacent" else 16
            loaded = T.alloc_local((elements_per_quad,), T.float32)
            reduced = T.alloc_local((elements_per_quad,), T.float32)
            encoded_values = T.alloc_local((elements_per_quad,), out_dtype)
            pos = T.alloc_var(T.int32)
            scale = T.alloc_var(T.float32)
            scale0 = T.alloc_var(T.float32)
            scale1 = T.alloc_var(T.float32)
            pos1 = T.alloc_var(T.int32)
            value1 = T.alloc_var(T.float32)
            sf_var = T.alloc_var(T.float32)
            scale = 0
            for j in T.unroll(elements_per_quad):
                loaded[j] = T.float32(0)
            if with_sf:
                sf_var = sf[0]
            pos = token_topk_to_pos[token, expert]
            T.assume(pos < num_expanded_tokens)
            if pos >= 0:
                if role >= 2:
                    scale = 1
                    if with_weights:
                        scale = topk_weights[token, expert]
                    if with_x_sf:
                        scale *= x_sf[pos]
                if role < 2:
                    for j in T.Vectorized(elements_per_quad):
                        loaded[j] = x[pos, column_id * elements_per_quad + j].astype("float32")
            scale0 = shfl_sync(scale, group_lane + 2 * lane_step, width=64, mask=0xFFFFFFFFFFFFFFFF)
            scale1 = shfl_sync(scale, group_lane + 3 * lane_step, width=64, mask=0xFFFFFFFFFFFFFFFF)
            pos1 = shfl_sync(pos, group_lane + lane_step, width=64, mask=0xFFFFFFFFFFFFFFFF)
            for j in T.unroll(elements_per_quad):
                value1 = shfl_sync(loaded[j], group_lane + lane_step, width=64, mask=0xFFFFFFFFFFFFFFFF)
                if role == 0:
                    reduced[j] = T.float32(0)
                    if pos >= 0:
                        reduced[j] += loaded[j] * scale0
                    if pos1 >= 0:
                        reduced[j] += value1 * scale1
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
            if role == 0:
                for j in T.Vectorized(elements_per_quad):
                    out[token, column_id * elements_per_quad + j] = encoded_values[j]
    return tiny_quad
