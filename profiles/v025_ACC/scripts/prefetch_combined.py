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

__all__ = ["MoeReduceFusedKernel"]

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
def get_reduce_fused_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    with_sf: bool,
    with_weights: bool,
    with_x_sf: bool,
    tile_hidden: Optional[int] = None,
    num_threads: int = 128,
    grid_hidden_first: bool = False,
    vector_store: bool = False,
    prefetch_rows: int = 8,
):
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"
    if tile_hidden is None:
        tile_hidden = (
            1024 if hidden % 1024 == 0 else 512 if hidden % 512 == 0 else 256
        ) if use_fp8_bits and hidden > 1024 else hidden
    split_hidden = tile_hidden != hidden
    vector_store = vector_store and use_fp8_bits
    copy_width = 1
    while copy_width < 8 and tile_hidden % (num_threads * copy_width * 2) == 0:
        copy_width *= 2

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
        with T.Kernel(
            hidden // tile_hidden if grid_hidden_first else num_tokens,
            num_tokens if grid_hidden_first else hidden // tile_hidden,
            threads=num_threads,
        ) as (pid_x, pid_y):
            pid_token = pid_y if grid_hidden_first else pid_x
            pid_hidden = pid_x if grid_hidden_first else pid_y
            reduced_fragment = T.alloc_fragment((tile_hidden,), T.float32)
            topk_weights_local = T.alloc_fragment((num_topk,), T.float32)
            topk_to_pos_local = T.alloc_fragment((num_topk,), T.int32)
            sf_var = T.alloc_var(T.float32)
            encoded_fragment = T.alloc_fragment((tile_hidden,), out_dtype)
            prefetched = T.alloc_fragment((prefetch_rows, tile_hidden), in_dtype)

            T.clear(reduced_fragment)
            if with_sf:
                sf_var = sf[0]
            if with_weights:
                T.copy(topk_weights[pid_token, :], topk_weights_local)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)

            for first in T.unroll(prefetch_rows):
                if first < num_topk:
                    pos = topk_to_pos_local[first]
                    T.assume(pos < num_expanded_tokens)
                    if pos >= 0:
                        T.copy(
                            x[pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden],
                            prefetched[first, :],
                        )
            for k in T.unroll(num_topk):
                pos = topk_to_pos_local[k]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    s = T.alloc_var(T.float32)
                    s = 1
                    if with_weights:
                        s = topk_weights_local[k]
                    if with_x_sf:
                        s *= x_sf[pos]
                    for i in T.Parallel(tile_hidden):
                        reduced_fragment[i] += prefetched[k % prefetch_rows, i] * s
                if k + prefetch_rows < num_topk:
                    next_pos = topk_to_pos_local[k + prefetch_rows]
                    T.assume(next_pos < num_expanded_tokens)
                    if next_pos >= 0:
                        T.copy(
                            x[next_pos, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden],
                            prefetched[k % prefetch_rows, :],
                        )

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
                    if vector_store:
                        encoded_fragment[i] = T.reinterpret(
                            final_encoded.astype("uint8"), out_dtype,
                        )
                    else:
                        out[pid_token, output_index] = T.reinterpret(
                            final_encoded.astype("uint8"), out_dtype,
                        )
                else:
                    out[pid_token, output_index] = T.Select(
                        with_sf, reduced_fragment[i] * sf_var, reduced_fragment[i],
                    )

            if vector_store:
                T.copy(
                    encoded_fragment,
                    out[pid_token, pid_hidden * tile_hidden:(pid_hidden + 1) * tile_hidden],
                    coalesced_width=copy_width,
                )

    return reduce_fused_kernel


class MoeReduceFusedKernel(Kernel):
    """Dispatch measured launch configs and retain v004 defaults for other shapes.

    The two scale flags select Base, WithXsf, FP8, or Quantized behavior.
    Token and expanded-row counts remain dynamic in the device kernel.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        num_topk: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
        *,
        with_weights: bool = True,
        with_sf: bool = False,
        with_x_sf: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if num_tokens < 0 or num_topk <= 0:
            raise ValueError("num_tokens must be nonnegative and num_topk positive")
        if hidden <= 0 or hidden % 256 != 0:
            raise ValueError("hidden must be a positive multiple of 256")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported input dtype: {dtype}")

        self.num_tokens = num_tokens
        self.num_topk = num_topk
        self.hidden = hidden
        self.dtype = dtype
        self.with_weights = with_weights
        self.with_sf = with_sf
        self.with_x_sf = with_x_sf
        self.out_dtype = torch.float8_e4m3fn if with_sf else dtype
        self.init_config(config, tune=False)
        tile_hidden = self.config["tile_hidden"]
        num_threads = self.config["num_threads"]
        if (type(tile_hidden) is not int or tile_hidden <= 0
                or tile_hidden % 256 != 0 or hidden % tile_hidden != 0):
            raise ValueError("tile_hidden must be a positive multiple of 256 dividing hidden")
        if type(num_threads) is not int or num_threads not in (64, 128, 256):
            raise ValueError("num_threads must be 64, 128, or 256")
        # v020: use hidden-first numbering only for the measured SF cases
        # and their measured launch geometry; custom geometry keeps v010.
        measured_key = (num_tokens, num_topk, hidden, dtype)
        self.grid_hidden_first = (
            with_sf and with_weights
            and dtype == torch.bfloat16
            and (num_tokens, num_topk, hidden) in (
                (512, 8, 7168), (4096, 8, 7168),
            )
            and (tile_hidden, num_threads) == _MEASURED_CONFIGS[measured_key][1]
        )
        # v022: separate FP8 encoding from global writeback for the
        # measured tiny/prefill shapes and their measured launch geometry.
        self.vector_store = (
            with_sf and with_weights
            and measured_key in (
                (32, 2, 256, torch.float16),
                (4096, 8, 7168, torch.bfloat16),
            )
            and (tile_hidden, num_threads) == _MEASURED_CONFIGS[measured_key][1]
        )
        self.kernel = get_reduce_fused_kernel(
            hidden,
            num_topk,
            T.dtype(dtype),
            T.dtype(self.out_dtype),
            with_sf,
            with_weights,
            with_x_sf,
            tile_hidden=tile_hidden,
            num_threads=num_threads,
            grid_hidden_first=self.grid_hidden_first,
            vector_store=self.vector_store,
        )

    @property
    def default_config(self) -> dict[str, int]:
        key = (self.num_tokens, self.num_topk, self.hidden, self.dtype)
        measured = _MEASURED_CONFIGS.get(key) if self.with_weights else None
        if measured is not None:
            tile_hidden, num_threads = measured[int(self.with_sf)]
        else:
            # Preserve v004 for unmeasured shapes, dtypes, or unweighted reduction.
            hidden = self.hidden
            tile_hidden = (
                1024 if hidden % 1024 == 0 else 512 if hidden % 512 == 0 else 256
            ) if self.with_sf and hidden > 1024 else hidden
            num_threads = 128
        return {"tile_hidden": tile_hidden, "num_threads": num_threads}

    @staticmethod
    def _check_tensor(name, tensor, shape, dtype, device) -> None:
        if tensor is None:
            raise ValueError(f"{name} is required for this kernel configuration")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        x_sf: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return [T, H] output, optionally using a caller-provided buffer.

        Nonnegative positions must be less than x.shape[0]. The device
        kernel retains the source assumption and skips negative positions.
        """
        if x.ndim != 2 or not x.is_cuda:
            raise ValueError("x must be a two-dimensional GPU tensor")
        device = x.device
        num_expanded_tokens = x.shape[0]
        self._check_tensor("x", x, (num_expanded_tokens, self.hidden), self.dtype, device)
        routing_shape = (self.num_tokens, self.num_topk)
        self._check_tensor(
            "token_topk_to_pos", token_topk_to_pos, routing_shape, torch.int32, device
        )
        if self.with_weights:
            self._check_tensor(
                "topk_weights", topk_weights, routing_shape, torch.float32, device
            )
        if self.with_x_sf:
            self._check_tensor(
                "x_sf", x_sf, (num_expanded_tokens,), torch.float32, device
            )
        if self.with_sf:
            self._check_tensor("sf", sf, (1,), torch.float32, device)

        output_shape = (self.num_tokens, self.hidden)
        if out is None:
            out = torch.empty(output_shape, dtype=self.out_dtype, device=device)
        else:
            self._check_tensor("out", out, output_shape, self.out_dtype, device)
            inputs = [x, token_topk_to_pos]
            if self.with_weights:
                inputs.append(topk_weights)
            if self.with_x_sf:
                inputs.append(x_sf)
            if self.with_sf:
                inputs.append(sf)
            if any(tensors_overlap(out, tensor) for tensor in inputs):
                raise ValueError("out must not overlap an input tensor")

        if self.num_tokens == 0:
            return out

        # The source GPU signature has fixed tensor slots. Supply unused
        # buffers for disabled branches; their contents are never read.
        if not self.with_weights:
            topk_weights = torch.empty(routing_shape, dtype=torch.float32, device=device)
        if not self.with_x_sf:
            x_sf = torch.empty((num_expanded_tokens,), dtype=torch.float32, device=device)
        if not self.with_sf:
            sf = torch.empty((1,), dtype=torch.float32, device=device)

        # Preserve the original device argument order, which differs from
        # this wrapper's public argument order.
        self.kernel(x, topk_weights, token_topk_to_pos, out, sf, x_sf)
        return out
