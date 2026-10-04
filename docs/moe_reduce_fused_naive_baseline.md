# MoE reduce_fused naive baseline

Date: 2026-10-03. Source: TileKernels-Metax commit 0266ab740980de7dc03a828b8259cd73d100c2eb.

Four Op variants use the original one-block-per-token, 128-thread GPU computation.

Validation:

- 131 correctness cases passed after shared input generation was introduced.
- 16 benchmark cases passed; 32 naive/PyTorch measurements used torch.profiler GPU-timeline timing (cupti).
- Targeted strict Manifest validation passed. Four warnings concern synthetic H=5 mock inputs violating H%256==0.
- Python syntax and whitespace checked. The extra pre-commit tool download was cancelled.

Environment: C500, 25% compute, 16000 MiB sGPU; MACA 3.7.1.5; PyTorch 2.8.0+metax3.7.1.3; TileLang /opt/tilelang-metax-v0.1.10.

Protocol: 10 warmup, 50 repeats x 3 trials, L2 flush, compilation excluded. Random valid routing. Prefill skips input clones under the shared 1 GiB clone-pool limit for both implementations.

Raw results, exact source snapshot, environment, and runner:
/root/reduce_fused_benchmark_runs/naive_20261003_084101/

Later cleanup covers Manifest metadata, static benchmark discovery, shape-inference layering, and formatting. GPU computation was not optimized. Measurements were not repeated after cleanup.

Reproduce from the repository root with MACA_PATH=/opt/maca and PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:

    python scripts/validate_manifest.py --check-op MoeReduceFusedFwdOp --strict
    python -m pytest -q tests/ops/test_moe_reduce_fused.py
    python -m pytest -q benchmarks/ops/bench_moe_reduce_fused.py
