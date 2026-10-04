"""Collect 12 native mcProfiler RoofLine images with descriptive filenames."""
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
TOOL = Path("/opt/mcProfiler-ubuntu18.04")
REPO = ROOT.parents[1]
WORKLOADS = [("h3072", 512, 3072), ("h7168_decode", 512, 7168)]
for variant in ("xsf", "fp8", "quantized"):
    for workload, tokens, hidden in WORKLOADS:
        name = f"{variant}_{workload}_T{tokens}_K8_H{hidden}"
        image = ROOT / "analysis" / (name + ".png")
        if image.exists():
            print("EXISTS", name, flush=True)
            continue
        # Reuse the already verified first capture.
        first = ROOT / "analysis/base_h3072_T512_K8_H3072.png"
        if variant == "base" and workload == "h3072" and first.exists():
            print("EXISTS", name, flush=True)
            continue
        raw = ROOT / "raw" / name
        raw.mkdir(parents=True, exist_ok=True)
        command = (
            "env MACA_PATH=/opt/maca PYTHONDONTWRITEBYTECODE=1 "
            f"PYTHONPATH=/opt/tilelang-metax-v0.1.10:{REPO} "
            f"MCTX_TARGET_PROFILE_PATH={raw} /opt/conda/bin/python "
            f"{ROOT}/scripts/profile_case.py --variant {variant} --tokens {tokens} --hidden {hidden}"
        )
        args = [
            str(TOOL / "mcProfiler"), "perf_exec", "--cmdline", command,
            "--kernelname", "reduce_fused_kernel_kernel", "--casename", name,
            "--cwd", str(raw), "--per-kernel", "--profile-from-start", "0",
            "--counts", "1",
            "--metrics", "RoofLine", "Total Cycles", "WORKGROUPS",
        ]
        log = ROOT / "logs" / (name + ".log")
        print("START", name, flush=True)
        with log.open("w") as f:
            result = subprocess.run(args, cwd=TOOL, stdout=f, stderr=subprocess.STDOUT, timeout=420)
        content = log.read_text(errors="replace")
        matches = re.findall(r"output path is: (\S+)", content)
        if not matches:
            raise RuntimeError(f"No mcProfiler output path; see {log}")
        output = Path(matches[-1])
        kernels = list(output.glob("*reduce_fused_kernel_kernel_dumped_result.json"))
        pictures = sorted(output.glob("RoofLine*.png"))
        if result.returncode or len(kernels) != 1 or len(pictures) != 3:
            raise RuntimeError(f"Incomplete capture {name}: rc={result.returncode}, kernels={len(kernels)}, images={len(pictures)}; see {log}")
        shutil.copytree(output, raw / "mcprofiler_output", dirs_exist_ok=True)
        # Native exporter order: process report, single-kernel report, final process report.
        shutil.copy2(pictures[1], image)
        print("SAVED", image.name, flush=True)
print("DONE: 6 variant RoofLine PNG files plus the 3 existing Base files", flush=True)
