"""Profile one naive reduce_fused variant/workload with mcProfiler."""
import argparse

import torch

from tileops.ops.moe import (
    MoeReduceFusedFwdOp, MoeReduceFusedWithXsfFwdOp,
    MoeReduceFusedFp8FwdOp, MoeReduceFusedQuantizedFwdOp,
)
from workloads.moe import MoeReduceFusedWorkload

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["base", "xsf", "fp8", "quantized"], required=True)
parser.add_argument("--tokens", type=int, required=True)
parser.add_argument("--hidden", type=int, required=True)
args = parser.parse_args()
classes = {
    "base": MoeReduceFusedFwdOp, "xsf": MoeReduceFusedWithXsfFwdOp,
    "fp8": MoeReduceFusedFp8FwdOp, "quantized": MoeReduceFusedQuantizedFwdOp,
}
torch.manual_seed(1235)
workload = MoeReduceFusedWorkload(
    args.tokens, 8, args.hidden, torch.bfloat16,
    with_x_sf=args.variant in ("xsf", "quantized"),
    with_sf=args.variant in ("fp8", "quantized"),
)
inputs = workload.gen_inputs()
op = classes[args.variant](args.tokens, 8, args.hidden, torch.bfloat16)
for _ in range(10):
    op(*inputs)
torch.cuda.synchronize()
torch.cuda.profiler.start()
op(*inputs)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"PROFILE_DONE {args.variant} T={args.tokens} K=8 H={args.hidden}", flush=True)
