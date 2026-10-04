"""FP32-to-E4M3 conversion expressed with TileLang bit operations."""
import tilelang
from tilelang import language as T

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_candidate(hidden: int, num_topk: int, in_dtype: T.dtype,
                  out_dtype: T.dtype, with_sf: bool,
                  with_weights: bool, with_x_sf: bool):
    num_tokens = T.dynamic("num_tokens")
    expanded = T.dynamic("expanded")
    @T.prim_func
    def reduce_fused_candidate(
        x: T.Tensor[(expanded, hidden), in_dtype],
        weights: T.Tensor[(num_tokens, num_topk), T.float32],
        positions: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(expanded,), T.float32],
    ):
        with T.Kernel(num_tokens, threads=128) as token:
            acc = T.alloc_fragment((hidden,), T.float32)
            w = T.alloc_fragment((num_topk,), T.float32)
            p = T.alloc_fragment((num_topk,), T.int32)
            sf_value = T.alloc_var(T.float32)
            T.clear(acc)
            if with_sf:
                sf_value = sf[0]
            if with_weights:
                T.copy(weights[token, :], w)
            T.copy(positions[token, :], p)
            for k in T.unroll(num_topk):
                pos = p[k]
                T.assume(pos < expanded)
                if pos >= 0:
                    scale = T.alloc_var(T.float32)
                    scale = 1
                    if with_weights:
                        scale = w[k]
                    if with_x_sf:
                        scale *= x_sf[pos]
                    for i in T.Parallel(hidden):
                        acc[i] += x[pos, i] * scale
            for i in T.Parallel(hidden):
                if with_sf:
                    value = acc[i] * sf_value
                    raw = T.reinterpret("uint32", value)
                    mag = raw & T.uint32(0x7fffffff)
                    sign = (raw >> 24) & T.uint32(0x80)
                    # RNE: retain three fraction bits with an even tie.
                    normal = ((mag + T.uint32(0x7ffff)
                               + ((mag >> 20) & T.uint32(1))) >> 20) - T.uint32(0x3c0)
                    # For E4M3 subnormals, an FP32 add rounds to units of 2^-9.
                    denormal = (T.reinterpret("uint32", T.reinterpret("float32", mag)
                                             + T.float32(16384.0)) - T.uint32(0x46800000))
                    encoded = T.Select(mag < T.uint32(0x3c800000), denormal,
                                       T.Select(mag >= T.uint32(0x43e00000),
                                                T.uint32(0x7e), normal))
                    final_encoded = T.Select(mag > T.uint32(0x7f800000),
                                       T.uint32(0x7f), encoded | sign)
                    out[token, i] = T.reinterpret(out_dtype, final_encoded.astype("uint8"))
                else:
                    out[token, i] = acc[i]
    return reduce_fused_candidate
