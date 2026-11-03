# -*- coding: utf-8 -*-
"""
批处理基准：batch 多大最划算
============================

README 6.2 节那张表（batch 1/2/4/8/16 → 19.5/15.2/13.7/14.8/14.1 ms）是 v3.0.0
**双模型**时期测的。v3.1.0 改单模型后，前向次数从 2 次变 1 次，这张表必须重测——
否则「batch=4 最优、每帧 13.7 ms」这句话就是拿旧结论套新架构。

做法：同一张图复制 N 份凑一批，`predict_many` 一次前向，测「总耗时 / 每帧耗时」。
每档跑 3 次取中位，避开首次 warmup 和偶发抖动。

    python tools/bench_batch.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anxun.detectors import build_detector

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--image", default="")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data" / "batch_bench.json"))
    args = ap.parse_args()

    import cv2

    img_path = Path(args.image) if args.image else next(
        (ROOT / "datasets/ppe_test/test/images").glob("*.jpg"))
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"读不到图：{img_path}")

    det, note = build_detector(args.weights)
    print(note)
    # DualDetector 有 detect_batch，YoloDetector 只有 predict_many，两者都是「一次前向吃 N 帧」
    run_many = getattr(det, "detect_batch", None) or det.predict_many

    # warmup：第一次前向含显存分配与 kernel 编译，不计入
    for _ in range(3):
        det.detect_frame(frame)

    sizes = [int(x) for x in args.batches.split(",")]
    rows = []
    print(f"\n{'batch':>6}{'总耗时':>12}{'每帧':>11}{'相对':>9}")
    base = None
    for b in sizes:
        frames = [frame] * b
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            run_many(frames)
            times.append((time.perf_counter() - t0) * 1000)
        total = statistics.median(times)
        per = total / b
        if base is None:
            base = per
        rows.append({"batch": b, "total_ms": round(total, 1),
                     "per_frame_ms": round(per, 1)})
        print(f"{b:>6}{total:>10.1f} ms{per:>9.1f} ms{base / per:>8.2f}×")

    best = min(rows, key=lambda r: r["per_frame_ms"])
    print(f"\n最优点：batch={best['batch']} → 每帧 {best['per_frame_ms']} ms")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
