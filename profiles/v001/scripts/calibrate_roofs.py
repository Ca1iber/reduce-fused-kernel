"""Calibrate empirical streaming-memory and FP32 vector reference roofs."""
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import tilelang
import tilelang.language as T
import torch

from benchmarks.benchmark_base import bench_kernel, _bench_meta

ROOT = Path(__file__).resolve().parent.parent


@tilelang.jit(
    out_idx=[],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def fp32_probe(n: int, block: int, iterations: int):
    @T.prim_func
    def fp32_vector_probe(
        x: T.Tensor([n], "float32"),
        factor: T.Tensor([n], "float32"),
        bias: T.Tensor([n], "float32"),
        out: T.Tensor([n], "float32"),
    ):
        with T.Kernel(n // block, threads=128) as pid:
            acc = T.alloc_fragment([block], "float32")
            mul = T.alloc_fragment([block], "float32")
            add = T.alloc_fragment([block], "float32")
            T.copy(x[pid * block:(pid + 1) * block], acc)
            T.copy(factor[pid * block:(pid + 1) * block], mul)
            T.copy(bias[pid * block:(pid + 1) * block], add)
            for rep in T.serial(iterations):
                for i in T.Parallel(block):
                    acc[i] = acc[i] * mul[i] + add[i]
            T.copy(acc, out[pid * block:(pid + 1) * block])
    return fp32_vector_probe


def measure(fn, name, count, kind, settings):
    latency = bench_kernel(fn, n_warmup=10, n_repeat=50, n_trials=3)
    row = {
        "name": name, "kind": kind, **settings,
        "latency_ms": latency,
        "timing": getattr(_bench_meta, "timing", "unknown"),
        "work_per_call": count,
        "rate": count / latency * 1e-9,
        "unit": "TB/s" if kind == "bandwidth" else "TFLOP/s",
        "addresses": "fixed; native L2 flush before each iteration",
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    torch.manual_seed(1235)
    props = torch.cuda.get_device_properties(0)
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__,
        "tilelang": tilelang.__version__, "tilelang_path": tilelang.__file__,
        "device_name": props.name, "visible_memory_bytes": props.total_memory,
        "visible_multiprocessors": props.multi_processor_count,
        "warmup": 10, "repeat": 50, "trials": 3,
        "scope": "current sc-16g; empirical references, not guaranteed peaks",
    }
    (ROOT / "meta/calibration_environment.json").write_text(json.dumps(meta, indent=2))
    records = []
    # 1 GiB arrays are much larger than L2; copy counts read+write,
    # add(out=) counts two reads+one write.
    n = (1 << 30) // 4
    x = torch.ones(n, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    out = torch.empty_like(x)
    records.append(measure(
        lambda: out.copy_(x), "copy_1GiB", 2 * n * 4,
        "bandwidth", {"array_bytes": n * 4, "pattern": "1 read + 1 write"},
    ))
    records.append(measure(
        lambda: torch.add(x, y, out=out), "add_1GiB", 3 * n * 4,
        "bandwidth", {"array_bytes": n * 4, "pattern": "2 reads + 1 write"},
    ))
    del x, y, out
    gc.collect()
    torch.cuda.empty_cache()
    # Multiple independent FP32 accumulators amortize loop dependency.
    # Two configurations provide a limited sweep, not a proof of peak.
    for per_thread in (8, 16):
        block = 128 * per_thread
        n = 4096 * block
        iterations = 2048
        x = torch.full((n,), 0.5, device="cuda", dtype=torch.float32)
        factor = torch.full_like(x, 0.999)
        bias = torch.full_like(x, 0.001)
        out = torch.empty_like(x)
        kernel = fp32_probe(n, block, iterations)
        name = f"fp32_{per_thread}acc"
        (ROOT / "codegen" / (name + ".cu")).write_text(kernel.get_kernel_source())
        kernel(x, factor, bias, out)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all().item()
        sample = float(out[0].item())
        assert 0.5 < sample < 1.01, sample
        records.append(measure(
            lambda: kernel(x, factor, bias, out), name,
            2 * n * iterations, "compute",
            {"elements": n, "block": block, "iterations": iterations,
             "independent_accumulators_per_thread": per_thread,
             "operation": "FP32 multiply-add; no GEMM/MMA"},
        ))
        del x, factor, bias, out, kernel
        gc.collect()
        torch.cuda.empty_cache()
    roofs = {
        "bandwidth_tbps": max(r["rate"] for r in records if r["kind"] == "bandwidth"),
        "fp32_tflops": max(r["rate"] for r in records if r["kind"] == "compute"),
        "records": records,
        "scope": "empirical current-instance references; limited probe sweep",
        "bytes_convention": "one read/write per streaming array element",
        "compute_convention": "two FLOPs per FP32 multiply-add",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "raw/roofs.json").write_text(json.dumps(roofs, indent=2) + "\n")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT / "meta/calibration_script.sha256").write_text(script_hash + "\n")
    print("CALIBRATION_DONE", flush=True)


if __name__ == "__main__":
    main()
