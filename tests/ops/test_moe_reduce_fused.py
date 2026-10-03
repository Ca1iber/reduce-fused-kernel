"""Correctness, routing, scaling, and input-contract tests for reduce_fused."""

import pytest
import torch

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

_VARIANTS = {
    "base": MoeReduceFusedFwdOp,
    "xsf": MoeReduceFusedWithXsfFwdOp,
    "fp8": MoeReduceFusedFp8FwdOp,
    "quantized": MoeReduceFusedQuantizedFwdOp,
}
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_ROUTES = ("identity", "random", "padded", "duplicates", "all-invalid")


def _reference_reduce_fused(x, positions, weights, *, x_sf=None, sf=None):
    """Independent vectorized gather and reduction, accumulated in FP32."""
    valid = positions >= 0
    safe_positions = positions.clamp(min=0).long()
    gathered = x[safe_positions].float()
    scales = weights
    if x_sf is not None:
        scales = scales * x_sf[safe_positions]
    contributions = gathered * scales.unsqueeze(-1)
    # Use selection, not zero multiplication: invalid rows may contain NaN.
    contributions = torch.where(valid.unsqueeze(-1), contributions, 0.0)
    result = contributions.sum(dim=1, dtype=torch.float32)
    if sf is not None:
        result = result * sf[0]
    output_dtype = torch.float8_e4m3fn if sf is not None else x.dtype
    return result.to(output_dtype)


def _make_inputs(num_tokens, num_topk, hidden, dtype, route="random"):
    workload = MoeReduceFusedWorkload(
        num_tokens, num_topk, hidden, dtype,
        with_x_sf=True, with_sf=True, route=route,
    )
    x, positions, weights, x_sf, sf = workload.gen_inputs()
    return {
        "x": x, "positions": positions, "weights": weights,
        "x_sf": x_sf, "sf": sf,
    }

def _scale_kwargs(variant, data):
    kwargs = {}
    if variant in ("xsf", "quantized"):
        kwargs["x_sf"] = data["x_sf"]
    if variant in ("fp8", "quantized"):
        kwargs["sf"] = data["sf"]
    return kwargs


def _run_and_compare(op, variant, data):
    kwargs = _scale_kwargs(variant, data)
    output = op(data["x"], data["positions"], data["weights"], **kwargs)
    expected = _reference_reduce_fused(
        data["x"], data["positions"], data["weights"], **kwargs
    )
    _assert_output(output, expected)
    return output


def _assert_output(output, expected):
    assert output.shape == expected.shape
    assert output.dtype == expected.dtype
    assert output.device == expected.device
    if expected.dtype == torch.float8_e4m3fn:
        # Different FP32 reduction orders can straddle a rounding boundary.
        # Permit one FP8 code step, including the minimum subnormal step.
        rtol, atol = 0.125, 2 ** -9
    elif expected.dtype == torch.bfloat16:
        rtol, atol = 1.6e-2, 1e-4
    elif expected.dtype == torch.float16:
        rtol, atol = 1e-3, 1e-5
    else:
        rtol, atol = 1e-5, 1e-6
    torch.testing.assert_close(
        output.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=False
    )


def _correctness_cases():
    cases = []
    # All smoke cases precede full cases, as required by tests/conftest.py.
    for variant in _VARIANTS:
        for dtype in _DTYPES:
            for route in _ROUTES:
                cases.append(pytest.param(
                    variant, 4, 2, 256, dtype, route,
                    marks=pytest.mark.smoke,
                    id=f"{variant}-{dtype}-{route}",
                ))
    for variant in _VARIANTS:
        for tokens, topk, hidden in (
            (1, 8, 256),
            (7, 1, 256),
            (7, 9, 512),
            (512, 8, 3072),
            (512, 8, 7168),
            (4096, 8, 7168),
        ):
            cases.append(pytest.param(
                variant, tokens, topk, hidden, torch.bfloat16, "padded",
                marks=pytest.mark.full,
                id=f"{variant}-t{tokens}-k{topk}-h{hidden}-bf16",
            ))
    return cases


@pytest.mark.parametrize(
    "variant,num_tokens,num_topk,hidden,dtype,route", _correctness_cases()
)
def test_moe_reduce_fused_correctness(
    variant, num_tokens, num_topk, hidden, dtype, route
):
    data = _make_inputs(num_tokens, num_topk, hidden, dtype, route)
    op = _VARIANTS[variant](num_tokens, num_topk, hidden, dtype)
    output = _run_and_compare(op, variant, data)
    invalid_tokens = (data["positions"] < 0).all(dim=1)
    if invalid_tokens.any().item():
        assert torch.count_nonzero(output.float()[invalid_tokens]).item() == 0


@pytest.mark.smoke
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_moe_reduce_fused_known_result(variant):
    """Hand-computed oracle detects wrong weights, scale indices, or normalization."""
    x = torch.tensor([1.0, 2.0, 4.0], device="cuda")[:, None].repeat(1, 256)
    data = {
        "x": x,
        "positions": torch.tensor([[2, 0], [1, -1]], device="cuda", dtype=torch.int32),
        "weights": torch.tensor([[0.25, 0.75], [2.0, 5.0]], device="cuda"),
        "x_sf": torch.tensor([2.0, 3.0, 5.0], device="cuda"),
        "sf": torch.tensor([0.5], device="cuda"),
    }
    values = {
        "base": [1.75, 4.0],
        "xsf": [6.5, 12.0],
        "fp8": [0.875, 2.0],
        "quantized": [3.25, 6.0],
    }
    expected = torch.tensor(values[variant], device="cuda")[:, None].repeat(1, 256)
    op = _VARIANTS[variant](2, 2, 256, torch.float32)
    output = _run_and_compare(op, variant, data)
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_moe_reduce_fused_invalid_slots_do_not_propagate_nan(variant):
    data = _make_inputs(3, 2, 256, torch.float32)
    data["positions"] = torch.tensor(
        [[1, -1], [-1, -1], [2, 1]], device="cuda", dtype=torch.int32
    )
    data["x"][0] = float("nan")
    data["x_sf"][0] = float("nan")
    data["weights"][data["positions"] < 0] = float("nan")
    op = _VARIANTS[variant](3, 2, 256, torch.float32)
    output = _run_and_compare(op, variant, data)
    assert torch.isfinite(output.float()).all().item()
    assert torch.count_nonzero(output.float()[1]).item() == 0


@pytest.mark.smoke
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_moe_reduce_fused_reuse_with_different_expanded_rows(variant):
    """One Op must handle both E < T*K and E > T*K without retaining inputs."""
    op = _VARIANTS[variant](4, 2, 256, torch.float32)
    for route in ("duplicates", "random"):
        data = _make_inputs(4, 2, 256, torch.float32, route)
        _run_and_compare(op, variant, data)


@pytest.mark.smoke
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize(
    "num_tokens,num_topk,hidden,dtype,message",
    [
        pytest.param(0, 2, 256, torch.float32, "positive", id="zero-tokens"),
        pytest.param(4, 0, 256, torch.float32, "positive", id="zero-topk"),
        pytest.param(4, 2, 0, torch.float32, "multiple", id="zero-hidden"),
        pytest.param(4, 2, 128, torch.float32, "multiple", id="unaligned-hidden"),
        pytest.param(4, 2, 256, torch.int32, "dtype", id="integer-input"),
    ],
)
def test_moe_reduce_fused_invalid_constructor(
    variant, num_tokens, num_topk, hidden, dtype, message
):
    with pytest.raises(ValueError, match=message):
        _VARIANTS[variant](num_tokens, num_topk, hidden, dtype)


def _noncontiguous(tensor):
    return tensor.t().contiguous().t()


_BAD_INPUTS = [
    pytest.param("x", lambda x: x.to(torch.float64), "x dtype", id="x-dtype"),
    pytest.param("positions", lambda x: x.long(), "int32", id="positions-dtype"),
    pytest.param("weights", lambda x: x.half(), "float32", id="weights-dtype"),
    pytest.param("x_sf", lambda x: x.half(), "float32", id="xsf-dtype"),
    pytest.param("sf", lambda x: x.half(), "float32", id="sf-dtype"),
    pytest.param("x", lambda x: x.reshape(-1), "shape", id="x-rank"),
    pytest.param("x", lambda x: x[:, :128], "shape", id="hidden-mismatch"),
    pytest.param("positions", lambda x: x[:1], "shape", id="tokens-mismatch"),
    pytest.param("weights", lambda x: x[:, :1], "shape", id="topk-mismatch"),
    pytest.param("x_sf", lambda x: x[:1], "shape", id="xsf-shape"),
    pytest.param("sf", lambda x: x.repeat(2), "shape", id="sf-shape"),
    pytest.param("x", _noncontiguous, "contiguous", id="x-layout"),
    pytest.param("weights", _noncontiguous, "contiguous", id="weights-layout"),
    pytest.param("x", lambda x: x.cpu(), "GPU", id="x-device"),
    pytest.param("sf", lambda x: x.cpu(), "must be on", id="sf-device"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("name,mutation,message", _BAD_INPUTS)
def test_moe_reduce_fused_invalid_inputs(name, mutation, message):
    data = _make_inputs(4, 2, 256, torch.float32)
    data[name] = mutation(data[name])
    op = MoeReduceFusedQuantizedFwdOp(4, 2, 256, torch.float32)
    with pytest.raises(ValueError, match=message):
        op(
            data["x"], data["positions"], data["weights"],
            x_sf=data["x_sf"], sf=data["sf"],
        )
