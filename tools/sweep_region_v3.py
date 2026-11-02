# -*- coding: utf-8 -*-
"""
判定区域扫参（v3 中心点口径专用）
=================================

一次推理，多组区域参数复用同一批框——只重算判定，不重跑模型。

为什么单独写：`tools/sweep_ppe_geom.py` 是 v2.x 的（单模型 + IoU 判据 +
旧两类权重），v3 换双模型和中心点判据后它已经不适用。

`docs/15` 附录 B.3 只扫过**收紧**方向（0.22 / 0.18 / 0.15），
从未试过**放宽**（0.30 以上）。而 6 例 B2 误报的标注框中心 y 都在
0.02–0.20，说明区域本身盖得住头——放宽能不能把「区域外高置信框」也纳进来，
是最后一个没验的方向。

    python tools/sweep_region_v3.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anxun.rules_engine as R
from anxun.detectors import DualDetector, pick_device
from anxun.evaluate_real import iou_box, list_images, load_gt, read_names, truth_of
from anxun.metrics import business_rates
from anxun.rules_engine import judge_frame, min_person_area

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    det = DualDetector(args.person_weights, args.weights,
                       imgsz=640, conf=0.25, device=pick_device(""))
    from anxun.evaluate_real import warmup
    warmup(det)

    data = Path(args.data)
    img_dir = data / "images" if (data / "images").exists() else data
    lbl_dir = data / "labels" if (data / "labels").exists() else data
    names = read_names(data)
    files = list_images(img_dir, args.limit or None)

    # ---- 一次性缓存：框 + 真值匹配关系 ----
    cache = []
    for fp in files:
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(lbl_dir / (fp.stem + ".txt"), names, W, H)
        res = det.detect_image(fp)
        preds = res["boxes"]
        gt_persons = [g for g in gt if g["label"] == "person"]
        pred_persons = [b for b in preds if b.label == "person"]
        pairs, used = [], set()
        for pi, pbox in enumerate(pred_persons):
            best, bv = None, 0.5
            for gi, g in enumerate(gt_persons):
                if gi in used:
                    continue
                v = iou_box({"x1": pbox.x1, "y1": pbox.y1, "x2": pbox.x2, "y2": pbox.y2}, g)
                if v > bv:
                    best, bv = gi, v
            if best is None:
                continue
            used.add(best)
            worn = truth_of(gt, gt_persons[best])
            if worn is not None:
                pairs.append((pi, best, worn))
        truth = [{"person_id": f"G{gi}", "equip": "helmet", "worn": w,
                  "matched": gi in used}
                 for gi, w in ((gi, truth_of(gt, g)) for gi, g in enumerate(gt_persons))
                 if w is not None]
        cache.append({"preds": preds, "pairs": pairs, "truth": truth,
                      "area_min": min_person_area(W, H), "image_id": fp.stem})
    print(f"缓存 {len(cache)} 张（含框与真值匹配）\n")

    base = {"HEAD_REGION_RATIO": R.HEAD_REGION_RATIO, "HEAD_WIDTH_RATIO": R.HEAD_WIDTH_RATIO}

    def measure(rr: float, wr: float) -> dict:
        R.HEAD_REGION_RATIO, R.HEAD_WIDTH_RATIO = rr, wr
        samples = []
        for c in cache:
            verdicts = judge_frame(c["preds"], ["helmet"], area_min=c["area_min"])
            judged = []
            for pi, gi, _w in c["pairs"]:
                vs = [v for v in verdicts if v.person_idx == pi and v.equip == "helmet"]
                if vs:
                    judged.append({"person_id": f"G{gi}", "equip": "helmet",
                                   "status": vs[0].status})
            samples.append({"image_id": c["image_id"], "truth": c["truth"],
                            "judged": judged})
        r = business_rates(samples, scope="matched")
        return {"fp": r["fp_rate"], "fn": r["fn_rate"], "cm": r["confusion"]}

    b = measure(base["HEAD_REGION_RATIO"], base["HEAD_WIDTH_RATIO"])
    print(f"基线 {base['HEAD_REGION_RATIO']}/{base['HEAD_WIDTH_RATIO']}："
          f"误报 {b['fp']}%  漏报 {b['fn']}%  {b['cm']}\n")

    print(f"{'region':>8}{'width':>8}{'误报':>9}{'漏报':>9}   fp/fn")
    rows = []
    for rr, wr in itertools.product((0.26, 0.30, 0.34, 0.38, 0.44),
                                    (0.70, 0.80, 0.90)):
        r = measure(rr, wr)
        rows.append((rr, wr, r))
        print(f"{rr:>8.2f}{wr:>8.2f}{r['fp']:>8.2f}%{r['fn']:>8.2f}%   "
              f"{r['cm']['fp']}/{r['cm']['fn']}")

    best = min(rows, key=lambda x: (x[2]["fn"], x[2]["fp"]))
    print(f"\n漏报优先的最优：region {best[0]}/{best[1]} → "
          f"误报 {best[2]['fp']}%  漏报 {best[2]['fn']}%")

    R.HEAD_REGION_RATIO, R.HEAD_WIDTH_RATIO = base["HEAD_REGION_RATIO"], base["HEAD_WIDTH_RATIO"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
