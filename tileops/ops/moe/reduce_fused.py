"""MoE expert-output reduction ops sharing the source naive kernel."""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import MoeReduceFusedKernel
from tileops.ops.op_base import Op

__all__ = [
    "MoeReduceFusedFp8FwdOp",
    "MoeReduceFusedFwdOp",
    "MoeReduceFusedQuantizedFwdOp",
    "MoeReduceFusedWithXsfFwdOp",
]


class MoeReduceFusedFwdOp(Op):
    """Gather rows and reduce K weighted contributions in FP32.

    Input x is [E, H]; positions and weights are [T, K]. E is independent
    of T*K. Valid positions must be less than E, and -1 means skip.
    Output is [T, H] with the same dtype as x.
    """

    _with_x_sf = False
    _with_sf = False

    def __init__(
        self,
        num_tokens: int,
        num_topk: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        if num_tokens <= 0 or num_topk <= 0:
            raise ValueError("num_tokens and num_topk must be positive")
        if hidden <= 0 or hidden % 256 != 0:
            raise ValueError("hidden must be a positive multiple of 256")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported input dtype: {dtype}")
        self.num_tokens = num_tokens
        self.num_topk = num_topk
        self.hidden = hidden
        self.dtype = dtype
        self.out_dtype = torch.float8_e4m3fn if self._with_sf else dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["reduce_fused_kernel"](
            num_tokens, num_topk, hidden, dtype,
            with_weights=True,
            with_x_sf=self._with_x_sf,
            with_sf=self._with_sf,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"reduce_fused_kernel": MoeReduceFusedKernel}

    def _infer_output_shapes(
        self, x_shape, token_topk_to_pos_shape, topk_weights_shape
    ) -> dict[str, tuple[int, ...]]:
        if len(x_shape) != 2 or len(token_topk_to_pos_shape) != 2:
            raise ValueError("x and token_topk_to_pos must have rank 2")
        if tuple(topk_weights_shape) != tuple(token_topk_to_pos_shape):
            raise ValueError("weights and positions must have the same shape")
        return {"out": (token_topk_to_pos_shape[0], x_shape[1])}

    def _validate_dtypes(self, x, token_topk_to_pos, topk_weights) -> None:
        if x.dtype != self.dtype:
            raise ValueError(f"x dtype must be {self.dtype}, got {x.dtype}")
        if token_topk_to_pos.dtype != torch.int32:
            raise ValueError("token_topk_to_pos dtype must be int32")
        if topk_weights.dtype != torch.float32:
            raise ValueError("topk_weights dtype must be float32")

    @staticmethod
    def _validate_scale_dtype(name, scale) -> None:
        if scale.dtype != torch.float32:
            raise ValueError(f"{name} dtype must be float32")

    def _validate_shapes(self, x, token_topk_to_pos, topk_weights) -> None:
        if x.ndim != 2 or x.shape[1] != self.hidden:
            raise ValueError(f"x must have shape [E, {self.hidden}]")
        shape = (self.num_tokens, self.num_topk)
        for name, tensor in (
            ("token_topk_to_pos", token_topk_to_pos),
            ("topk_weights", topk_weights),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")

    def eval_roofline(self) -> tuple[int, int]:
        """All K slots valid; minimum semantic traffic, excluding casts."""
        slots = self.num_tokens * self.num_topk
        elements = self.num_tokens * self.hidden
        flops = 2 * slots * self.hidden
        if self._with_x_sf:
            flops += slots
        if self._with_sf:
            flops += elements
        elem_bytes = torch.finfo(self.dtype).bits // 8
        input_bytes = slots * self.hidden * elem_bytes
        output_bytes = elements * (1 if self._with_sf else elem_bytes)
        metadata_bytes = slots * (12 if self._with_x_sf else 8)
        total_bytes = input_bytes + output_bytes + metadata_bytes
        if self._with_sf:
            total_bytes += 4
        return flops, total_bytes

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(x, token_topk_to_pos, topk_weights)
        self._validate_shapes(x, token_topk_to_pos, topk_weights)
        return self.kernel(x, token_topk_to_pos, topk_weights)


class MoeReduceFusedWithXsfFwdOp(MoeReduceFusedFwdOp):
    """Multiply each expert row by x_sf[pos] before weighted reduction."""

    _with_x_sf = True

    def _infer_output_shapes(
        self, x_shape, token_topk_to_pos_shape, topk_weights_shape, x_sf_shape
    ) -> dict[str, tuple[int, ...]]:
        return super()._infer_output_shapes(
            x_shape, token_topk_to_pos_shape, topk_weights_shape
        )

    def _validate_dtypes(self, x, token_topk_to_pos, topk_weights, x_sf) -> None:
        super()._validate_dtypes(x, token_topk_to_pos, topk_weights)
        self._validate_scale_dtype("x_sf", x_sf)

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        topk_weights: torch.Tensor,
        x_sf: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(x, token_topk_to_pos, topk_weights, x_sf)
        self._validate_shapes(x, token_topk_to_pos, topk_weights)
        return self.kernel(x, token_topk_to_pos, topk_weights, x_sf=x_sf)


class MoeReduceFusedFp8FwdOp(MoeReduceFusedFwdOp):
    """Multiply the reduced output by sf[0], then cast to FP8 E4M3FN."""

    _with_sf = True

    def _infer_output_shapes(
        self, x_shape, token_topk_to_pos_shape, topk_weights_shape, sf_shape
    ) -> dict[str, tuple[int, ...]]:
        return super()._infer_output_shapes(
            x_shape, token_topk_to_pos_shape, topk_weights_shape
        )

    def _validate_dtypes(self, x, token_topk_to_pos, topk_weights, sf) -> None:
        super()._validate_dtypes(x, token_topk_to_pos, topk_weights)
        self._validate_scale_dtype("sf", sf)

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        topk_weights: torch.Tensor,
        sf: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(x, token_topk_to_pos, topk_weights, sf)
        self._validate_shapes(x, token_topk_to_pos, topk_weights)
        return self.kernel(x, token_topk_to_pos, topk_weights, sf=sf)


class MoeReduceFusedQuantizedFwdOp(MoeReduceFusedFwdOp):
    """Apply input x_sf and output sf, with FP8 E4M3FN output."""

    _with_x_sf = True
    _with_sf = True

    def _infer_output_shapes(
        self, x_shape, token_topk_to_pos_shape, topk_weights_shape, x_sf_shape, sf_shape
    ) -> dict[str, tuple[int, ...]]:
        return super()._infer_output_shapes(
            x_shape, token_topk_to_pos_shape, topk_weights_shape
        )

    def _validate_dtypes(self, x, token_topk_to_pos, topk_weights, x_sf, sf) -> None:
        super()._validate_dtypes(x, token_topk_to_pos, topk_weights)
        self._validate_scale_dtype("x_sf", x_sf)
        self._validate_scale_dtype("sf", sf)

    def forward(
        self,
        x: torch.Tensor,
        token_topk_to_pos: torch.Tensor,
        topk_weights: torch.Tensor,
        x_sf: torch.Tensor,
        sf: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(x, token_topk_to_pos, topk_weights, x_sf, sf)
        self._validate_shapes(x, token_topk_to_pos, topk_weights)
        return self.kernel(x, token_topk_to_pos, topk_weights, x_sf=x_sf, sf=sf)
