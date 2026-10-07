import tilelang
from tilelang import language as T
from tilelang.language.builtin import shfl_sync

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_pair_kernel(hidden, num_topk, in_dtype, out_dtype, with_sf, with_weights, with_x_sf, num_threads=512, pair_layout="adjacent", num_tokens_hint=32):
    assert num_topk == 2
    assert num_threads in (128,256,512)
    columns_per_cta = num_threads // 2
    assert hidden % columns_per_cta == 0
    elements_per_pair = hidden // columns_per_cta
    num_tokens = T.dynamic('num_tokens') if num_tokens_hint is None else num_tokens_hint
    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"
    vector_store = use_fp8_bits
    guard_tokens = num_tokens_hint is None

    @T.prim_func
    def tiny_pair(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as token:
            tx = T.get_thread_binding()
            expert = tx % 2 if pair_layout == "adjacent" else (tx % 64) // 32
            column_id = tx // 2 if pair_layout == "adjacent" else (tx // 64) * 32 + tx % 32
            peer_lane = (tx % 64) ^ (1 if pair_layout == "adjacent" else 32)
            loaded = T.alloc_local((elements_per_pair,), T.float32)
            reduced = T.alloc_local((elements_per_pair,), T.float32)
            encoded_values = T.alloc_local((elements_per_pair,), out_dtype)
            scale = T.alloc_var(T.float32)
            pos = T.alloc_var(T.int32)
            peer_pos = T.alloc_var(T.int32)
            peer_scale = T.alloc_var(T.float32)
            peer_value = T.alloc_var(T.float32)
            sf_var = T.alloc_var(T.float32)
            if not guard_tokens or token < num_tokens:
                scale = 0
                for j in T.unroll(elements_per_pair):
                    loaded[j] = T.float32(0)
                if with_sf:
                    sf_var = sf[0]
                pos = token_topk_to_pos[token, expert]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    scale = 1
                    if with_weights:
                        scale = topk_weights[token, expert]
                    if with_x_sf:
                        scale *= x_sf[pos]
                    for j in T.Vectorized(elements_per_pair):
                        loaded[j] = x[pos, column_id * elements_per_pair + j].astype("float32")
                # All 64 lanes reach each shuffle. Each pair stays within a warp.
                peer_pos = shfl_sync(pos, peer_lane, width=64, mask=0xFFFFFFFFFFFFFFFF)
                peer_scale = shfl_sync(scale, peer_lane, width=64, mask=0xFFFFFFFFFFFFFFFF)
                for j in T.unroll(elements_per_pair):
                    peer_value = shfl_sync(loaded[j], peer_lane, width=64, mask=0xFFFFFFFFFFFFFFFF)
                    if expert == 0:
                        # Preserve the baseline's expert order and FP32 FMA rounding.
                        reduced[j] = T.float32(0)
                        if pos >= 0:
                            reduced[j] += loaded[j] * scale
                        if peer_pos >= 0:
                            reduced[j] += peer_value * peer_scale
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
                if expert == 0:
                    for j in T.Vectorized(elements_per_pair):
                        out[token, column_id * elements_per_pair + j] = encoded_values[j]
    return tiny_pair
