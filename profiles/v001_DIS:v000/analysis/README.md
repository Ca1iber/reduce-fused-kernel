# Roofline v001

16 saved naive timings, no rerun of the 16 workload matrix.
Current-instance reference roofs: 1.482 TB/s and 22.15 TFLOP/s.
Ridge: 14.94 FLOP/B. Workload AI range: 0.658 to 0.999 FLOP/B.

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
