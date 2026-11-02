# -*- coding: utf-8 -*-
"""
几何阈值扫参（v2.x IoU 判据专用）
=================================

⚠️ **v3.0.0 起 helmet 改用「中心点」判据**，`HELMET_IOU_MIN` / `CONTAIN_MIN`
   不再参与 helmet 判定——扫这两个参数不会有任何变化（实测 15 组指标一格没动）。

   v3 要扫参请用：
     - `tools/sweep_region_v3.py`     判定区域（0.26–0.44 × 0.70–0.90）
     - `tools/sweep_ppe_geom.py --stage conf`  置信度

   本脚本保留，因为 vest 仍走 IoU 判据，而且要留 v2.x 的实验证据。

背景：微调后的模型 helmet AP 从 0.5645 涨到 0.7376，但业务指标反而变差。
查下来是**框的几何变了**——新模型的帽框更紧（面积/头部区域 0.60 vs 0.86），
与头部区域的 IoU 从 0.410 掉到 0.291，而 `HELMET_IOU_MIN=0.25` 是照旧模型调的。

置信度阈值（CONF_FINAL）只决定「敢不敢下结论」，几何阈值决定「这个框算不算戴在头上」。
换模型后两套都要重扫，只扫前者会漏掉真正的失配点。

    python src/anxun/sweep_geom.py                        # 默认扫当前权重
    python src/anxun/sweep_geom.py --ppe runs/ft_mix/weights/best.pt

检测只推理一次（缓存），后面纯算阈值，几秒出结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image                                     # noqa: E402

from anxun import rules_engine as R                       # noqa: E402
from anxun.detectors import DualDetector, pick_device     # noqa: E402
from anxun.evaluate_real import (list_images, load_gt,    # noqa: E402
                                 read_names, iou_box, truth_of)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "ppe_test" / "test"


def collect(det) -> list[dict]:
    """把 100 张图的检测结果 + 真值一次性算好，之后扫阈值不再推理。"""
    names = read_names(DATA)
    out = []
    for fp in list_images(DATA / "images"):
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(DATA / "labels" / f"{fp.stem}.txt", names, W, H)
        preds = det.detect_image(fp)["boxes"]
        out.append({"preds": preds, "gt": gt, "W": W, "H": H})
    return out


def score(cache: list[dict], conf_final: float, iou_min: float,
          contain_min: float) -> tuple[float, float, int, int]:
    fp = n_v = fn = n_c = 0
    for row in cache:
        preds, gt = row["preds"], row["gt"]
        vs = R.judge_frame(preds, ["helmet"], area_min=R.min_person_area(row["W"], row["H"]))
        pp = [b for b in preds if b.label == "person"]
        gp = [g for g in gt if g["label"] == "person"]
        used = set()
        for pi, pb in enumerate(pp):
            best, bv = None, 0.5
            for gi, g in enumerate(gp):
                if gi in used:
                    continue
                v = iou_box({"x1": pb.x1, "y1": pb.y1, "x2": pb.x2, "y2": pb.y2}, g)
                if v > bv:
                    best, bv = gi, v
            if best is None:
                continue
            used.add(best)
            worn = truth_of(gt, gp[best])
            if worn is None:
                continue
            it = [v for v in vs if v.person_idx == pi]
            if not it:
                continue
            pv = it[0].status == "未佩戴"
            if worn:
                n_c += 1
                fp += pv
            else:
                n_v += 1
                fn += (not pv)
    return (fp / n_c * 100 if n_c else 0, fn / n_v * 100 if n_v else 0, fp, fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppe", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--device", default="")
    ap.add_argument("--conf-final", type=float, default=R.CONF_FINAL,
                    help="固定置信度阈值，只扫几何")
    args = ap.parse_args()

    det = DualDetector(args.person, args.ppe, device=pick_device(args.device))
    print(f"PPE 权重：{Path(args.ppe).name}   CONF_FINAL 固定 {args.conf_final}")
    cache = collect(det)

    rows = []
    base = (R.CONF_FINAL, R.HELMET_IOU_MIN, R.CONTAIN_MIN, R.MAX_EQUIP_RATIO)
    R.CONF_FINAL = args.conf_final
    # 0.28 是 2026-11-02 复扫选中的点，必须在网格里，否则前沿复现不出交付值
    for iou in (0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40):
        for ct in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            for mr in (1.6, 3.0, 6.0):
                R.HELMET_IOU_MIN, R.CONTAIN_MIN, R.MAX_EQUIP_RATIO = iou, ct, mr
                fpr, fnr, fp, fn = score(cache, args.conf_final, iou, ct)
                rows.append({"iou": iou, "contain": ct, "max_ratio": mr,
                             "fp": round(fpr, 1), "fn": round(fnr, 1),
                             "fp_n": fp, "fn_n": fn})
    R.CONF_FINAL, R.HELMET_IOU_MIN, R.CONTAIN_MIN, R.MAX_EQUIP_RATIO = base

    # 帕累托前沿：误报和漏报都不比已选中的差
    rows.sort(key=lambda r: (r["fn"], r["fp"]))
    print(f"\n{'IOU':>6}{'CONTAIN':>9}{'MAXR':>6}{'误报':>8}{'漏报':>8}{'FP':>4}{'FN':>4}")
    front = []
    for r in rows:
        if any(r["fp"] >= s["fp"] and r["fn"] >= s["fn"] for s in front):
            continue
        front.append(r)
        print(f"{r['iou']:>6.2f}{r['contain']:>9.2f}{r['max_ratio']:>6.1f}"
              f"{r['fp']:>7.1f}%{r['fn']:>7.1f}%{r['fp_n']:>4}{r['fn_n']:>4}")

    best = min(rows, key=lambda r: (r["fn"], r["fp"]))
    print(f"\n漏报最优：IOU {best['iou']} / CONTAIN {best['contain']} / "
          f"MAXR {best['max_ratio']} → 误报 {best['fp']}% 漏报 {best['fn']}%")

    out = ROOT / "data" / "sweep_geom.json"
    out.write_text(json.dumps(
        {"ppe": Path(args.ppe).name, "conf_final": args.conf_final,
         "rows": rows, "pareto": front}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
