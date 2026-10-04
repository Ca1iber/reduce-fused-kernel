#!/bin/sh
cd /root/TileOPs-Metax/profiles/v003/raw/mctracer
export MACA_PATH=/opt/maca
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/root/TileOPs-Metax
/opt/maca/bin/mcTracer --mctx /opt/conda/bin/python /root/TileOPs-Metax/profiles/v003/scripts/trace_pair.py
