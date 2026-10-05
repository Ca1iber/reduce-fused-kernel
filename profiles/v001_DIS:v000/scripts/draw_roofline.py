"""Build a standalone Plotly roofline from existing latency and calibrated roofs."""
import ast
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
PLOTLY = Path("/opt/mcProfiler-ubuntu18.04/tools/js_code/plotly-2.24.1.min.js")
COLORS = {"Base": "#0072B2", "WithXsf": "#009E73", "FP8": "#D55E00", "Quantized": "#CC79A7"}
FAMILIES = {
    "MoeReduceFusedFwdOp": "Base",
    "MoeReduceFusedWithXsfFwdOp": "WithXsf",
    "MoeReduceFusedFp8FwdOp": "FP8",
    "MoeReduceFusedQuantizedFwdOp": "Quantized",
}
SYMBOLS = {"tiny": "circle", "h3072": "square", "h7168": "diamond", "prefill": "triangle-up"}


def main():
    roofs = json.loads((ROOT / "raw/roofs.json").read_text())
    manifest = yaml.safe_load((REPO / "tileops/manifest/moe.yaml").read_text())
    with (ROOT / "raw/benchmark.csv").open() as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    assert len(rows) == 16, len(rows)
    points = []
    for row in rows:
        op = row["op"]
        dtype = row["dtype"].removeprefix("torch.")
        env = {
            "num_tokens": int(row["T"]), "num_topk": int(row["K"]),
            "hidden": int(row["H"]), "elem_bytes": {"float16": 2, "bfloat16": 2, "float32": 4}[dtype],
        }
        spec = manifest[op]["roofline"]
        for name, expression in spec.get("vars", {}).items():
            env[name] = eval(expression, {"__builtins__": {}}, env)
        flops = eval(spec["flops"], {"__builtins__": {}}, env)
        traffic = eval(spec["bytes"], {"__builtins__": {}}, env)
        latency = float(row["naive_ms"])
        case = row["label"].split("-")[0]
        points.append({
            "variant": FAMILIES[op], "op": op, "label": row["label"], "case": case,
            "T": int(row["T"]), "K": int(row["K"]), "H": int(row["H"]),
            "dtype": dtype, "flops": flops, "semantic_bytes": traffic,
            "ai_flop_per_byte": flops / traffic,
            "latency_ms": latency, "performance_tflops": flops / latency * 1e-9,
            "semantic_tbps": traffic / latency * 1e-9,
            "timing": row["naive_timing"],
        })
    with (ROOT / "analysis/roofline_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    bw, peak = roofs["bandwidth_tbps"], roofs["fp32_tflops"]
    ridge = peak / bw
    xmin, xmax = 0.1, 10 ** math.ceil(math.log10(ridge * 8))
    xs = [10 ** (math.log10(xmin) + i / 199 * math.log10(xmax / xmin)) for i in range(200)]
    ys = [min(peak, bw * x) for x in xs]
    traces = [
        {"type": "scatter", "mode": "lines", "x": xs, "y": ys,
         "name": "Empirical reference roof", "line": {"color": "#263238", "width": 2.5},
         "hovertemplate": "AI=%{x:.3f}<br>Reference=%{y:.3f} TFLOP/s<extra></extra>"},
        {"type": "scatter", "mode": "lines", "x": [xmin, xmax], "y": [peak, peak],
         "name": f"FP32 probe: {peak:.2f} TFLOP/s",
         "line": {"color": "#78909c", "width": 1.2, "dash": "dot"}},
        {"type": "scatter", "mode": "lines", "x": xs, "y": ys,
         "xaxis": "x2", "yaxis": "y2", "showlegend": False,
         "line": {"color": "#263238", "width": 2.5}},
    ]
    for variant, color in COLORS.items():
        data = [p for p in points if p["variant"] == variant]
        tips = [
            f'{p["variant"]} / {p["label"]}<br>T={p["T"]}, K={p["K"]}, H={p["H"]}, {p["dtype"]}'
            f'<br>AI={p["ai_flop_per_byte"]:.6f} FLOP/B'
            f'<br>Performance={p["performance_tflops"]:.6f} TFLOP/s'
            f'<br>Latency={p["latency_ms"] * 1000:.3f} us'
            f'<br>Semantic traffic={p["semantic_bytes"] / 1e6:.3f} MB'
            for p in data
        ]
        trace = {
            "type": "scatter", "mode": "markers", "name": variant,
            "legendgroup": variant,
            "x": [p["ai_flop_per_byte"] for p in data],
            "y": [p["performance_tflops"] for p in data],
            "marker": {"color": color, "size": 12, "opacity": 0.8,
                       "symbol": [SYMBOLS[p["case"]] for p in data],
                       "line": {"color": "white", "width": 0.8}},
            "text": tips, "hovertemplate": "%{text}<extra></extra>",
        }
        traces.append(trace)
        traces.append({**trace, "xaxis": "x2", "yaxis": "y2", "showlegend": False})
    amin = min(p["ai_flop_per_byte"] for p in points)
    amax = max(p["ai_flop_per_byte"] for p in points)
    pmin = min(p["performance_tflops"] for p in points)
    pmax = max(p["performance_tflops"] for p in points)
    layout = {
        "template": "plotly_white", "height": 640,
        "margin": {"l": 80, "r": 45, "t": 95, "b": 90},
        "font": {"family": "Arial, sans-serif", "size": 13},
        "legend": {"orientation": "h", "x": 0, "y": 1.16},
        "xaxis": {"type": "log", "domain": [0, 0.57],
                  "range": [math.log10(xmin), math.log10(xmax)],
                  "title": "Arithmetic intensity (FLOP/byte)"},
        "yaxis": {"type": "log", "range": [math.log10(pmin * 0.5), math.log10(peak * 1.3)],
                  "title": "Achieved FP32 FLOP rate (TFLOP/s)"},
        "xaxis2": {"domain": [0.68, 1], "anchor": "y2", "range": [amin * 0.95, amax * 1.04],
                   "title": "AI: workload region"},
        "yaxis2": {"type": "log", "anchor": "x2",
                   "range": [math.log10(pmin * 0.5), math.log10(max(pmax, bw * amax) * 1.3)]},
        "annotations": [
            {"xref": "paper", "yref": "paper", "x": 0.25, "y": 1.06,
             "text": "Full empirical roofline", "showarrow": False},
            {"xref": "paper", "yref": "paper", "x": 0.86, "y": 1.06,
             "text": "Workload zoom", "showarrow": False},
            {"xref": "paper", "yref": "paper", "x": 0, "y": -0.19,
             "text": "circle: tiny | square: H3072 | diamond: H7168 decode | triangle: prefill",
             "showarrow": False, "xanchor": "left"},
        ],
    }
    config = {"responsive": True, "displaylogo": False,
              "toImageButtonOptions": {"format": "svg", "filename": "reduce_fused_roofline_v001"}}
    table = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.6f}</td><td>{:.6f}</td><td>{:.3f}</td></tr>".format(
            html.escape(p["variant"]), html.escape(p["label"]),
            p["ai_flop_per_byte"], p["performance_tflops"], p["latency_ms"] * 1000
        ) for p in points
    )
    js = PLOTLY.read_text().replace("</script", "<\\/script")
    payload = json.dumps([traces, layout, config]).replace("</", "<\\/")
    page = """<!doctype html><html><head><meta charset="utf-8">
<title>reduce_fused roofline v001</title><style>
body{font:15px Arial,sans-serif;margin:24px auto;max-width:1500px;padding:0 20px;color:#263238}
p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
.note{background:#f1f5f9;padding:12px 16px;border-radius:8px}
</style></head><body><h1>Naive reduce_fused: roofline v001</h1>
<p>C500 sc-16g, 25% compute quota, 16000 MiB. All 16 points reuse the saved benchmark.</p>
<div class="note">ROOF_NOTE</div><div id="chart"></div>
<p>Colors distinguish variants; marker shapes distinguish workload sizes. Hover for exact values.
The camera button exports SVG. No network connection or external package is required.</p>
<p>Bytes are minimum semantic traffic from the Manifest, not measured HBM traffic.
The roofs are limited-sweep empirical references, not guaranteed hardware peaks.
Small crossings of the streaming roof may reflect access-pattern, model, or measurement differences.
FP8 cast/index/branch work is not fully represented by FP32 FLOP counting.</p>
<table><thead><tr><th>Variant</th><th>Case</th><th>AI (FLOP/B)</th><th>TFLOP/s</th><th>Latency (us)</th></tr></thead>
<tbody>TABLE_ROWS</tbody></table><script>PLOTLY_JS</script><script>
const [data,layout,config]=PLOT_DATA;Plotly.newPlot("chart",data,layout,config);
</script></body></html>"""
    note = f"Measured reference roofs: streaming {bw:.3f} TB/s; FP32 multiply-add {peak:.2f} TFLOP/s; ridge {ridge:.2f} FLOP/B."
    page = page.replace("ROOF_NOTE", note).replace("TABLE_ROWS", table).replace("PLOTLY_JS", js).replace("PLOT_DATA", payload)
    output = ROOT / "analysis/roofline.html"
    output.write_text(page)
    (ROOT / "analysis/plot_data.json").write_text(json.dumps({"data": traces, "layout": layout, "config": config}, indent=2))
    formula = {name: manifest[name]["roofline"] for name in FAMILIES}
    meta = {"formulas": formula, "source_latency_file": "raw/benchmark.csv",
            "calibrated_roofs": "raw/roofs.json", "points": len(points),
            "bandwidth_tbps": bw, "fp32_tflops": peak, "ridge_flop_per_byte": ridge,
            "plotly_path": str(PLOTLY),
            "html_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT / "meta/roofline_model.json").write_text(json.dumps(meta, indent=2))
    summary = f"""# Roofline v001

16 saved naive timings, no rerun of the 16 workload matrix.
Current-instance reference roofs: {bw:.3f} TB/s and {peak:.2f} TFLOP/s.
Ridge: {ridge:.2f} FLOP/B. Workload AI range: {amin:.3f} to {amax:.3f} FLOP/B.

Open roofline.html offline. The camera button exports SVG.
Formulas and peak sources are in ../meta/roofline_model.json.
Raw calibration and its limited sweep are in ../raw/roofs.json.

Interpretation: large Base/WithXsf cases are close to the streaming
reference. FP8/Quantized cases fall further below it; this alone does not
identify the cause or establish achievable speedup. Tiny cases have low
throughput and limited work. Use targeted Profile to distinguish memory,
conversion, dependency, and scheduling costs.

Limits: semantic traffic is not actual hardware traffic; compute probes
do not establish the exhaustive FP32 peak; streaming copy/triad access
patterns differ from gather-reduce. Calibration and operator timings were
recorded at different times on the same instance.
"""
    (ROOT / "analysis/README.md").write_text(summary)
    print(f"ROOFLINE_DONE points={len(points)} file={output}")


if __name__ == "__main__":
    main()
