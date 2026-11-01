#!/usr/bin/env bash
# 安巡 Agent · 启动（macOS / Linux）
# 自动选后端：有 GPU 用 GPU，没有就用 CPU 版 ONNX
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
exec $PY src/anxun/app.py "$@"
