#!/bin/sh
cd /data/TileOPs-Metax/profiles/v003_ACC/raw/mctracer
export MACA_PATH=/opt/maca
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/data/TileOPs-Metax
/opt/maca/bin/mcTracer --mctx /opt/conda/bin/python /data/TileOPs-Metax/profiles/v003_ACC/scripts/trace_pair.py
