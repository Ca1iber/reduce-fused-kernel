"""Manifest-driven naive reduce_fused benchmark against vectorized PyTorch.

Both implementations consume the same inputs and use ManifestBenchmark:
10 warmup iterations, 50 repeats x 3 trials, L2 flush, and kernel timing.
Index conversion, masking, gather, scales, reduction, and casts are included
inside the PyTorch timed callable. JIT compilation is completed beforehand.
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.moe import (
    MoeReduceFusedFp8FwdOp,
    MoeReduceFusedFwdOp,
    MoeReduceFusedQuantizedFwdOp,
    MoeReduceFusedWithXsfFwdOp,
)
from workloads.moe import MoeReduceFusedWorkload

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reduce_fused requires a GPU"
)

_OPS = {
    "MoeReduceFusedFwdOp": (MoeReduceFusedFwdOp, False, False),
    "MoeReduceFusedWithXsfFwdOp": (MoeReduceFusedWithXsfFwdOp, True, False),
    "MoeReduceFusedFp8FwdOp": (MoeReduceFusedFp8FwdOp, False, True),
    "MoeReduceFusedQuantizedFwdOp": (MoeReduceFusedQuantizedFwdOp, True, True),
}


_MANIFEST_WORKLOADS = {
    "MoeReduceFusedFwdOp": load_workloads("MoeReduceFusedFwdOp"),
    "MoeReduceFusedWithXsfFwdOp": load_workloads("MoeReduceFusedWithXsfFwdOp"),
    "MoeReduceFusedFp8FwdOp": load_workloads("MoeReduceFusedFp8FwdOp"),
    "MoeReduceFusedQuantizedFwdOp": load_workloads("MoeReduceFusedQuantizedFwdOp"),
}


def _manifest_params():
    params = []
    for op_name, workloads in _MANIFEST_WORKLOADS.items():
        for workload in workloads:
            label = workload["label"]
            for dtype_name in workload["dtypes"]:
                params.append(pytest.param(
                    op_name, label,
                    workload["num_tokens"], workload["num_topk"], workload["hidden"],
                    getattr(torch, dtype_name),
                    id=f"{op_name}-{label}-{dtype_name}",
                ))
    return params


def _make_benchmark(op_name, op, workload):
    if op_name == "MoeReduceFusedFwdOp":
        return ManifestBenchmark("MoeReduceFusedFwdOp", op, workload)
    if op_name == "MoeReduceFusedWithXsfFwdOp":
        return ManifestBenchmark("MoeReduceFusedWithXsfFwdOp", op, workload)
    if op_name == "MoeReduceFusedFp8FwdOp":
        return ManifestBenchmark("MoeReduceFusedFp8FwdOp", op, workload)
    if op_name == "MoeReduceFusedQuantizedFwdOp":
        return ManifestBenchmark("MoeReduceFusedQuantizedFwdOp", op, workload)
    raise ValueError(f"unknown reduce_fused op: {op_name}")


def _torch_reduce_fused(x, positions, weights, *, x_sf=None, sf=None):
    """Independent vectorized baseline; no cached input-derived intermediates."""
    valid = positions >= 0
    safe_positions = positions.clamp(min=0).long()
    gathered = x[safe_positions].float()
    scales = weights
    if x_sf is not None:
        scales = scales * x_sf[safe_positions]
    contributions = gathered * scales.unsqueeze(-1)
    contributions = torch.where(valid.unsqueeze(-1), contributions, 0.0)
    output = contributions.sum(dim=1, dtype=torch.float32)
    if sf is not None:
        output = output * sf[0]
    return output.to(torch.float8_e4m3fn if sf is not None else x.dtype)


@pytest.mark.parametrize(
    "op_name,label,num_tokens,num_topk,hidden,dtype", _manifest_params()
)
def test_moe_reduce_fused_bench(
    op_name: str,
    label: str,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    dtype: torch.dtype,
) -> None:
    op_class, with_x_sf, with_sf = _OPS[op_name]
    route = "random"
    workload = MoeReduceFusedWorkload(
        num_tokens, num_topk, hidden, dtype,
        with_x_sf=with_x_sf, with_sf=with_sf, route=route,
    )
    inputs = workload.gen_inputs()
    op = op_class(num_tokens, num_topk, hidden, dtype)
    bm = _make_benchmark(op_name, op, workload)

    def baseline_fn(*values):
        x, positions, weights = values[:3]
        kwargs = {}
        if with_x_sf:
            kwargs["x_sf"] = values[3]
        if with_sf:
            kwargs["sf"] = values[4 if with_x_sf else 3]
        return _torch_reduce_fused(x, positions, weights, **kwargs)

    # Warm up/JIT and verify both paths before collecting any timings.
    with torch.no_grad():
        actual = op(*inputs)
        expected = baseline_fn(*inputs)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        rtol = 0.125 if with_sf else 1.6e-2 if dtype == torch.bfloat16 else 1e-3
        atol = 2 ** -9 if with_sf else 1e-4 if dtype == torch.bfloat16 else 1e-5
        if dtype == torch.float32 and not with_sf:
            rtol, atol = 1e-5, 1e-6
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=False
        )
    del actual, expected
    torch.cuda.synchronize()

    params = {
        "label": label, "num_tokens": num_tokens, "num_topk": num_topk,
        "hidden": hidden, "dtype": dtype, "route": route,
    }
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, params, result, tag="tileops")

    result_torch = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, params, result_torch, tag="torch-ref")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vvs"]))
