# -*- coding: utf-8 -*-
"""
YOLO 各版本架构开销实测
=======================

为什么只测速度不测精度：镜像上没有干净的 COCO 权重（都是各领域微调版），
但**架构的算子开销与是否训练无关**——同一份 yaml 建出来的网络，参数量、FLOPs、
前向耗时是确定的。所以这份表回答的是「同规模下哪个架构更省」，不是「谁更准」。

    python tools/bench_arch.py

精度那部分在 README 里用另一套证据说明：我们的瓶颈是 helmet 权重（AP 0.579），
不是架构——换版本必须重训，重训前谈精度没有意义。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "arch_bench.json"

ARCHS = [
    "yolov8n", "yolov8s",
    "yolo11n", "yolo11s",
    "yolo12n", "yolo12s",
    "yolo26n", "yolo26s",
]
IMGSZ = 640
REPEAT = 15


def main() -> int:
    import numpy as np
    from ultralytics import YOLO

    img = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    rows = []

    for name in ARCHS:
        try:
            m = YOLO(f"{name}.yaml")          # 只用结构，不需要权重
            model = m.model
            params = sum(p.numel() for p in model.parameters()) / 1e6
            flops = None
            try:
                from ultralytics.utils.torch_utils import get_flops
                flops = get_flops(model, IMGSZ)
            except Exception:
                pass

            m.predict(img, imgsz=IMGSZ, verbose=False)      # 预热
            ts = []
            for _ in range(REPEAT):
                t0 = time.perf_counter()
                m.predict(img, imgsz=IMGSZ, verbose=False)
                ts.append((time.perf_counter() - t0) * 1000)
            ms = round(statistics.median(ts), 1)
            row = {"arch": name, "params_m": round(params, 2),
                   "flops_g": round(flops, 1) if flops else None,
                   "cpu_ms": ms, "p90_ms": round(sorted(ts)[int(len(ts) * 0.9)], 1)}
            rows.append(row)
            print(f"{name:<10} {params:5.2f}M  {str(row['flops_g']):>6}G  {ms:7.1f} ms")
        except Exception as e:
            print(f"{name:<10} 跳过：{type(e).__name__}: {e}")

    if not rows:
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"imgsz": IMGSZ, "repeat": REPEAT, "device": "CPU",
                              "note": "随机初始化权重，只测架构开销，不含精度",
                              "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
