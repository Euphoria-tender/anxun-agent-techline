# -*- coding: utf-8 -*-
"""
导出 ONNX 权重（换机器 / 换输入尺寸时跑一次）
===========================================

    python src/anxun/export_onnx.py

产物：
    models/onnx/yolov8n.onnx   人体检测   ~13 MB
    models/onnx/best.onnx      安全帽     ~43 MB
    models/onnx/manifest.json  类别名 + 输入尺寸

为什么要导：`.pt` 要装 torch（本机那个 cu130 轮子 1.99 GB），
换台电脑基本装不上。ONNX 两个模型合计 55 MB，`onnxruntime` 只有 15 MB，
CPU 上还比 torch 快 2.6 倍（53.5 ms vs 138 ms，同一台机器实测）。

注意：导出需要 torch（一次性），运行不需要。
没有 torch 的机器直接拷 `models/onnx/` 整个目录过去就行。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "onnx"

SOURCES = [("yolov8n", ROOT / "models" / "yolov8n.pt"),
           ("best", ROOT / "models" / "best.pt")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="已存在也重新导出")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"需要先装 ultralytics 才能导出：{e}")
        print("没有 torch 的机器不用导，直接把 models/onnx/ 目录拷过去即可。")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for stem, pt in SOURCES:
        dst = OUT / f"{stem}.onnx"
        if not pt.exists():
            print(f"跳过 {pt.name}（不存在）")
            continue
        if dst.exists() and not args.force:
            print(f"已存在 {dst.name}，跳过（--force 可强制重导）")
        else:
            m = YOLO(str(pt))
            p = Path(m.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                              simplify=False, dynamic=False, device="cpu"))
            shutil.move(str(p), dst)
        m = YOLO(str(pt))
        manifest[stem] = {"names": {int(k): str(v) for k, v in m.names.items()},
                          "imgsz": args.imgsz}
        print(f"{pt.name:18} -> {dst.name:14} {dst.stat().st_size / 1e6:6.1f} MB  "
              f"类别 {len(m.names)}")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.glob("*.onnx"))
    print(f"\n总计 {total / 1e6:.1f} MB -> {OUT}")
    print("换机器只需拷这个目录 + pip install onnxruntime opencv-python numpy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
