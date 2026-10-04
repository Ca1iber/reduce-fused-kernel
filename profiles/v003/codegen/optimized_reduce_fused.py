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
):
    num_threads = 128
    use_fp8_bits = with_sf and str(out_dtype) == "float8_e4m3fn"

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
        with T.Kernel(num_tokens, threads=num_threads) as (pid_token,):
            reduced_fragment = T.alloc_fragment((hidden,), T.float32)
            topk_weights_local = T.alloc_fragment((num_topk,), T.float32)
            topk_to_pos_local = T.alloc_fragment((num_topk,), T.int32)
            sf_var = T.alloc_var(T.float32)

            T.clear(reduced_fragment)
            if with_sf:
                sf_var = sf[0]
            if with_weights:
                T.copy(topk_weights[pid_token, :], topk_weights_local)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)

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
                    for i in T.Parallel(hidden):
                        reduced_fragment[i] += x[pos, i] * s

            for i in T.Parallel(hidden):
                if use_fp8_bits:
                    value = reduced_fragment[i] * sf_var
                    raw = T.reinterpret(value, "uint32")
                    magnitude = raw & T.uint32(0x7fffffff)
                    sign = (raw >> 24) & T.uint32(0x80)
                    # Keep three mantissa bits, rounding nearest with even ties.
                    normal = (
                        (magnitude + T.uint32(0x7ffff)
                         + ((magnitude >> 20) & T.uint32(1))) >> 20
                    ) - T.uint32(0x3c0)
                    # E4M3 subnormals have a 2^-9 step. The FP32 bias add
                    # rounds to that step without the SDK's FP64 conversion.
                    denormal = T.reinterpret(
                        T.reinterpret(magnitude, "float32") + T.float32(16384.0),
                        "uint32",
                    ) - T.uint32(0x46800000)
                    encoded = T.Select(
                        magnitude < T.uint32(0x3c800000), denormal,
                        T.Select(magnitude >= T.uint32(0x43e00000),
                                 T.uint32(0x7e), normal),
                    )
                    # Match the original SDK SATFINITE conversion: clamp
                    # overflow/Inf, canonicalize NaN, preserve signed zero.
                    final_encoded = T.Select(
                        magnitude > T.uint32(0x7f800000),
                        T.uint32(0x7f), encoded | sign,
                    )
                    out[pid_token, i] = T.reinterpret(
                        final_encoded.astype("uint8"), out_dtype,
                    )
                else:
                    out[pid_token, i] = T.Select(
                        with_sf, reduced_fragment[i] * sf_var, reduced_fragment[i],
                    )

    return reduce_fused_kernel


class MoeReduceFusedKernel(Kernel):
    """Wrap the original one-block-per-token kernel for TileOPs.

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
        self.kernel = get_reduce_fused_kernel(
            hidden,
            num_topk,
            T.dtype(dtype),
            T.dtype(self.out_dtype),
            with_sf,
            with_weights,
            with_x_sf,
        )

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
