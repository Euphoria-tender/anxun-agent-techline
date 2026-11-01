#!/usr/bin/env bash
# 安巡 Agent · 安装（CPU 版，macOS / Linux / Windows-GitBash 通用）
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  安巡 Agent · 安装（CPU 版，任意电脑）"
echo "============================================================"

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "[x] 没找到 python3"; exit 1; }

echo "[1/3] $($PY -V 2>&1)"

echo "[2/3] 安装依赖（onnxruntime / opencv / numpy，约 80 MB）..."
$PY -m pip install -q --upgrade pip
if ! $PY -m pip install -q -r requirements-cpu.txt; then
  echo "[x] 依赖安装失败。国内网络可加镜像："
  echo "    \$PY -m pip install -r requirements-cpu.txt -i https://mirrors.aliyun.com/pypi/simple/"
  exit 1
fi

echo "[3/3] 检查模型权重..."
if [ -f models/onnx/best.onnx ]; then
  echo "    已找到 models/onnx/ ，共 55 MB"
else
  echo "[x] 缺 models/onnx/best.onnx"
  echo "    从项目仓库把 models/onnx/ 整个目录拷过来，"
  echo "    或在装了 torch 的机器上运行：$PY src/anxun/export_onnx.py"
  exit 1
fi

echo
echo "============================================================"
echo "  装完了。启动：./run.sh        （或 $PY src/anxun/app.py）"
echo "============================================================"
