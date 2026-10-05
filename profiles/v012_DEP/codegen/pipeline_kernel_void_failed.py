"""MoE reduction with direct FP32-to-E4M3 output encoding.

Copied from TileKernels-Metax tile_kernels/moe/reduce_fused_kernel.py
at source commit 0266ab740980de7dc03a828b8259cd73d100c2eb.
The FP8 path preserves SDK SATFINITE rounding with FP32/uint32 operations.
"""

from typing import Optional

import tilelang
import torch
from tilelang import language as T

from tileops.kernels.buffer_utils import tensors_overlap
from tileops.kernels.kernel_base import Kernel

__all__ = ["get_pipeline_kernel"]

# sc-16g measurements: (tile_hidden, num_threads) for Base/XSF and FP8/Quantized.
_MEASURED_CONFIGS = {
    (32, 2, 256, torch.float16): ((256, 256), (256, 128)),
    (512, 8, 3072, torch.bfloat16): ((3072, 256), (512, 128)),
    (512, 8, 7168, torch.bfloat16): ((7168, 256), (512, 128)),
    (4096, 8, 7168, torch.bfloat16): ((7168, 256), (1024, 128)),
}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_pipeline_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    with_sf: bool,
    with_weights: bool,
    with_x_sf: bool,
    tile_hidden: Optional[int] = None,
    num_threads: int = 128,
    stages: int = 2,
    asynchronous: bool = True,
    old_two_barriers: bool = False,
):
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"
    if tile_hidden is None:
        tile_hidden = (
            1024 if hidden % 1024 == 0 else 512 if hidden % 512 == 0 else 256
        ) if use_fp8_bits and hidden > 1024 else hidden
    split_hidden = tile_hidden != hidden

    elems_per_thread = tile_hidden // num_threads
    input_bytes = 4 if str(in_dtype) == "float32" else 2
    width_bytes = elems_per_thread * input_bytes
    assert width_bytes in (4, 8, 16), "probe supports one native copy per thread"
    barrier_type = "b32vectype" if width_bytes == 4 else "b64vectype" if width_bytes == 8 else "b128vectype"
    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')

    @T.prim_func
    def reduce_fused_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(num_tokens, hidden // tile_hidden, threads=num_threads) as (pid_token, pid_hidden):
            reduced_fragment = T.alloc_fragment((tile_hidden,), T.float32)
            topk_weights_local = T.alloc_fragment((num_topk,), T.float32)
            topk_to_pos_local = T.alloc_fragment((num_topk,), T.int32)
            sf_var = T.alloc_var(T.float32)

            stage_x = T.alloc_shared((stages, tile_hidden), in_dtype)
            tickets = T.alloc_buffer((stages,), "void", scope="local.barrier", annotations={"barrier_type": barrier_type})
            T.clear(reduced_fragment)
            if with_sf:
                sf_var = sf[0]
            if with_weights:
                T.copy(topk_weights[pid_token, :], topk_weights_local)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)

            # Initial lookahead: S-1 rows. Invalid routes neither copy nor wait.
            for prefetch in T.unroll(stages - 1):
                if prefetch < num_topk:
                    first_pos = topk_to_pos_local[prefetch]
                    T.assume(first_pos < num_expanded_tokens)
                    if first_pos >= 0:
                        if asynchronous:
                            T.maca_async_copy(x[first_pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden], stage_x[prefetch, :], barrier=tickets[prefetch], annotations={"coalesced_width": elems_per_thread})
                        else:
                            T.copy(x[first_pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden], stage_x[prefetch, :], coalesced_width=elems_per_thread)
            for k in T.unroll(num_topk):
                pos = topk_to_pos_local[k]
                T.assume(pos < num_expanded_tokens)
                if stages == 1:
                    if pos >= 0:
                        T.copy(x[pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden], stage_x[0, :], coalesced_width=elems_per_thread)
                if asynchronous:
                    if pos >= 0:
                        T.maca_barrier_arrive_and_wait(tickets[k % stages])
                # This also protects reuse of the previous iteration's buffer.
                T.sync_threads()
                if stages > 1:
                    if k + stages - 1 < num_topk:
                        next_pos = topk_to_pos_local[k + stages - 1]
                        T.assume(next_pos < num_expanded_tokens)
                        if next_pos >= 0:
                            if asynchronous:
                                T.maca_async_copy(x[next_pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden], stage_x[(k + stages - 1) % stages, :], barrier=tickets[(k + stages - 1) % stages], annotations={"coalesced_width": elems_per_thread})
                            else:
                                T.copy(x[next_pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden], stage_x[(k + stages - 1) % stages, :], coalesced_width=elems_per_thread)
                if pos >= 0:
                    s = T.alloc_var(T.float32)
                    s = 1
                    if with_weights:
                        s = topk_weights_local[k]
                    if with_x_sf:
                        s *= x_sf[pos]
                    for i in T.Parallel(tile_hidden):
                        reduced_fragment[i] += stage_x[k % stages, i] * s
                if old_two_barriers:
                    T.sync_threads()

            for i in T.Parallel(tile_hidden):
                output_index = pid_hidden * tile_hidden + i if split_hidden else i
                if use_fp8_bits:
                    value = reduced_fragment[i] * sf_var
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
                    out[pid_token, output_index] = T.reinterpret(
                        final_encoded.astype("uint8"), out_dtype,
                    )
                else:
                    out[pid_token, output_index] = T.Select(
                        with_sf, reduced_fragment[i] * sf_var, reduced_fragment[i],
                    )

    return reduce_fused_kernel
