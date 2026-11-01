# -*- coding: utf-8 -*-
"""
在「1080p 模拟集」上验证级联感知。

为什么需要这份数据：原始测试集是 640×640 原生图，全图推理时没有缩放，
头部就是 30~50px。级联把它放大到 320 反而离开了训练分布，所以测不出收益。
现场是 1080p 摄像头压到 640——真丢的是 3 倍细节。

做法：把 640 图放大到 1920（YOLO 标注归一化，不用改）。
  baseline：1920 → imgsz 640，细节被压掉，和真实部署同构
  cascade ：全图 640 定位人 → 头部在 1920 原图上裁出来，细节没丢
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun.cascade import CascadeDetector              # noqa: E402
from anxun.detectors import DualDetector, pick_device  # noqa: E402
from anxun.evaluate_real import run_eval               # noqa: E402

PERSON = ROOT / "models" / "yolov8n.pt"
PPE = ROOT / "models" / "best.pt"


def show(tag, r):
    res = r["results"]
    print(f"{tag:<44} 误报 {res['fp_rate']:>5.1f}%  漏报 {res['fn_rate']:>5.1f}%  "
          f"{res['latency']['mean_ms']:>6.1f}ms  "
          f"TP{res['confusion']['tp']:>3} FN{res['confusion']['fn']:>3} "
          f"FP{res['confusion']['fp']:>2} TN{res['confusion']['tn']:>3}")
    return res


def main():
    dev = pick_device()
    print(f"device = {dev}")
    for name, data in (("640 原始集", ROOT / "datasets/ppe_test/test"),
                       ("1920 模拟 1080p", ROOT / "datasets/ppe_test/test1080")):
        print(f"\n===== {name} =====")
        base = DualDetector(str(PERSON), str(PPE), device=dev)
        b = show("baseline  imgsz=640", run_eval(data, base, verbose=False))
        for mode in ("replace", "union"):
            for head_px in (72, 96, 128):
                det = CascadeDetector(
                    DualDetector(str(PERSON), str(PPE), device=dev),
                    head_px=head_px, head_imgsz=320, conf=0.15, mode=mode)
                show(f"cascade  mode={mode} head_px={head_px}", run_eval(data, det, verbose=False))
        print(f"  基线对照：漏报 {b['fn_rate']:.1f}% / 误报 {b['fp_rate']:.1f}%")


if __name__ == "__main__":
    main()
