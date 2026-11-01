# -*- coding: utf-8 -*-
"""
对比：单级全图检测 vs 两级级联（头部 ROI 精查）
用同一套 100 张测试集，只改感知层，判定口径完全不变。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun.cascade import CascadeDetector          # noqa: E402
from anxun.detectors import DualDetector, pick_device  # noqa: E402
from anxun.evaluate_real import run_eval           # noqa: E402

DATA = ROOT / "datasets" / "ppe_test" / "test"
PERSON = ROOT / "models" / "yolov8n.pt"
PPE = ROOT / "models" / "best.pt"


def show(tag, r):
    res = r["results"]
    print(f"{tag:<34} mAP {res['mAP50']:.3f}  误报 {res['fp_rate']:>5.1f}%  "
          f"漏报 {res['fn_rate']:>5.1f}%  {res['latency']['mean_ms']:>6.1f}ms  "
          f"TP{res['confusion']['tp']:>3} FN{res['confusion']['fn']:>3} "
          f"FP{res['confusion']['fp']:>2} TN{res['confusion']['tn']:>3}")
    return res


def main():
    dev = pick_device()
    print(f"device = {dev}\n")
    base = DualDetector(str(PERSON), str(PPE), device=dev)
    print("== 基线：一次全图前向 ==")
    b = show("baseline (imgsz 640)", run_eval(DATA, base, verbose=False))

    print("\n== 级联：头部 < N px 的人做 ROI 精查 ==")
    rows = []
    for head_px in (48, 64, 80, 96, 120, 160):
        det = CascadeDetector(DualDetector(str(PERSON), str(PPE), device=dev),
                              head_px=head_px, head_imgsz=320, conf=0.15)
        r = show(f"cascade head_px={head_px}", run_eval(DATA, det, verbose=False))
        rows.append((head_px, r))

    print("\n== 精查输入尺寸（固定 head_px=96）==")
    for himg in (256, 320, 448):
        det = CascadeDetector(DualDetector(str(PERSON), str(PPE), device=dev),
                              head_px=96, head_imgsz=himg, conf=0.15)
        show(f"cascade head_px=96 imgsz={himg}", run_eval(DATA, det, verbose=False))

    best = min(rows, key=lambda x: (x[1]["fn_rate"], x[1]["fp_rate"]))
    print(f"\n基线  : 漏报 {b['fn_rate']:.1f}%  误报 {b['fp_rate']:.1f}%  {b['latency']['mean_ms']:.1f}ms")
    print(f"最优  : 漏报 {best[1]['fn_rate']:.1f}%  误报 {best[1]['fp_rate']:.1f}%  "
          f"{best[1]['latency']['mean_ms']:.1f}ms  (head_px={best[0]})")


if __name__ == "__main__":
    main()
