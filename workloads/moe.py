import torch

from workloads.workload_base import WorkloadBase


class FusedTopKTest(WorkloadBase):
    def __init__(self, num_tokens, num_experts, top_k, scoring_func, renormalize, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        torch.manual_seed(42)
        return (torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device="cuda"),)


class MoePermuteTest(WorkloadBase):

    def __init__(self, total_tokens, top_k, num_experts, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.randn(
            self.total_tokens, self.hidden_size, dtype=self.dtype, device="cuda"
        )
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device="cuda",
        )
        return hidden_states, topk_ids


class MoePermuteAlignTest(WorkloadBase):

    def __init__(self, total_tokens: int, top_k: int, num_experts: int, block_size: int):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size

    def gen_inputs(self) -> tuple[torch.Tensor]:
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device="cuda",
        )
        return (topk_ids,)


class MoeUnpermuteTest(WorkloadBase):

    def __init__(self, total_tokens, top_k, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = self.total_tokens * self.top_k
        mm2_pad = torch.randn(numel, self.hidden_size, dtype=self.dtype, device="cuda")
        # fwd_idx: simulate a valid mapping: random shuffle of [0, numel)
        fwd_idx = torch.randperm(numel, dtype=torch.int32, device="cuda")
        topk_weights = torch.rand(
            self.total_tokens, self.top_k, dtype=torch.float32, device="cuda"
        )
        return mm2_pad, fwd_idx, topk_weights


class MoeReduceFusedWorkload(WorkloadBase):
    """Shared input generation for reduce_fused tests and benchmarks.

    gen_inputs() follows each Op's public input order: x, positions,
    weights, optional x_sf, optional sf. The default route is a random
    gather with all K slots valid and E > T*K.
    """

    def __init__(
        self,
        num_tokens: int,
        num_topk: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
        *,
        with_x_sf: bool = False,
        with_sf: bool = False,
        route: str = "random",
    ):
        if num_tokens <= 0 or num_topk <= 0 or hidden <= 0:
            raise ValueError("num_tokens, num_topk, and hidden must be positive")
        if route not in ("identity", "random", "padded", "duplicates", "all-invalid"):
            raise ValueError(f"unsupported routing mode: {route}")
        self.num_tokens = num_tokens
        self.num_topk = num_topk
        self.hidden = hidden
        self.dtype = dtype
        self.with_x_sf = with_x_sf
        self.with_sf = with_sf
        self.route = route
        slots = num_tokens * num_topk
        self.num_expanded_tokens = (
            slots if route == "identity"
            else max(1, slots // 2) if route == "duplicates"
            else slots + 3
        )
        self.shape = (self.num_expanded_tokens, hidden)

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        slots = self.num_tokens * self.num_topk
        expanded = self.num_expanded_tokens
        if self.route == "duplicates":
            positions = torch.arange(slots, device="cuda", dtype=torch.int32) % expanded
        elif self.route == "identity":
            positions = torch.arange(slots, device="cuda", dtype=torch.int32)
        elif self.route == "all-invalid":
            positions = torch.full((slots,), -1, device="cuda", dtype=torch.int32)
        else:
            positions = torch.randperm(expanded, device="cuda")[:slots].to(torch.int32)
            positions[-1] = expanded - 1
        positions = positions.reshape(self.num_tokens, self.num_topk)
        if self.route == "padded":
            positions.reshape(-1)[::3] = -1
            positions[0] = -1
        x = torch.randn(expanded, self.hidden, device="cuda", dtype=self.dtype) * 0.1
        weights = torch.rand(
            self.num_tokens, self.num_topk, device="cuda", dtype=torch.float32
        )
        inputs = (x, positions, weights)
        if self.with_x_sf:
            x_sf = torch.linspace(
                -0.75, 1.25, expanded, device="cuda", dtype=torch.float32
            )
            inputs += (x_sf,)
        if self.with_sf:
            inputs += (torch.tensor([0.75], device="cuda", dtype=torch.float32),)
        return inputs
