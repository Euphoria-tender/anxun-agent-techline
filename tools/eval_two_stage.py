# -*- coding: utf-8 -*-
"""
两级判定实验：全图粗判 + 头部放大复核
=====================================

v3.0.0 的 10 例误报里，6 例是「框在头部区域内、conf 不到 0.59」。这些人的
person 框宽度只有 44–93 px，头在 640 输入下只有 20–30 px——不是看不见，是太模糊。

两条路已经试过、否掉：
  - 全图提高分辨率：960 → 误报 18.09%，1280 → 34.09%（模型在 640 训练，放大即出分布）
  - 单纯降阈值：conf 0.55 → 误报 8.16% 但漏报 19.35%（拿漏报换误报，不划算）

本脚本验证第三条：**只对判为「未佩戴」的人，把他的头裁出来放大到 640 再跑一次**。
`tools/probe_head_zoom.py` 已证明 10 例里 3 例能被放大救回（置信度 0.58→0.74、0.25→0.81）。

这里做全量 100 张，看四个指标的净变化，以及耗时代价。

    python tools/eval_two_stage.py                    # 默认 refine_conf=0.59
    python tools/eval_two_stage.py --refine-conf 0.45 # 扫更宽的翻案门槛
    python tools/eval_two_stage.py --side-scale 2.6   # 裁更大一圈的上下文

口径与 evaluate_real 完全一致（同一个 run 逻辑、同一套真值推导），
唯一差别是多了一步复核，所以数字可以直接对比。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anxun.rules_engine as R
from anxun.detectors import DualDetector, pick_device
from anxun.evaluate_real import (TRUTH_HEAD_REGION, TRUTH_HEAD_WIDTH,
                                 iou_box, list_images, load_gt, read_names, truth_of)
from anxun.metrics import evaluate
from anxun.rules_engine import Box, judge_frame, min_person_area

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT = ROOT / "data" / "metrics_two_stage.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--refine-conf", type=float, default=0.59,
                    help="放大复核时，多少置信度才翻案为「已佩戴」")
    ap.add_argument("--side-scale", type=float, default=2.2,
                    help="裁剪正方形边长 = max(区域宽, 区域高) × 本值")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    det = DualDetector(args.person_weights, args.weights,
                       imgsz=args.imgsz, conf=args.conf, device=pick_device(""))
    data = Path(args.data)
    img_dir = data / "images" if (data / "images").exists() else data
    lbl_dir = data / "labels" if (data / "labels").exists() else data
    names = read_names(data)
    files = list_images(img_dir, args.limit or None)

    samples, flips, n_refine = [], 0, 0
    t_start = time.perf_counter()

    for i, fp in enumerate(files, 1):
        t0 = time.perf_counter()
        with Image.open(fp) as im:
            im = im.convert("RGB")
            W, H = im.size
            gt = load_gt(lbl_dir / (fp.stem + ".txt"), names, W, H)
            res = det.detect_image(fp)
            preds = res["boxes"]
            verdicts = judge_frame(preds, ["helmet"], area_min=min_person_area(W, H))
            persons = [b for b in preds if b.label == "person"]

            # ---- 二级：只复核判为「未佩戴」的人 ----
            status_override = {}
            for v in verdicts:
                if v.equip != "helmet" or v.status != "未佩戴":
                    continue
                if v.person_idx >= len(persons):
                    continue
                pbox = persons[v.person_idx]
                region = pbox.top_region(R.HEAD_REGION_RATIO, R.HEAD_WIDTH_RATIO)
                side = max(region.w, region.h) * args.side_scale
                half = side / 2
                x1 = max(0, int(region.cx - half))
                y1 = max(0, int(region.cy - half))
                x2 = min(W, int(region.cx + half))
                y2 = min(H, int(region.cy + half))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crop = im.crop((x1, y1, x2, y2)).resize((640, 640), Image.BICUBIC)
                z = det.ppe.detect_frame(np.array(crop))
                n_refine += 1
                best = max((b.conf for b in z["boxes"] if b.label == "helmet"), default=0.0)
                if best >= args.refine_conf:
                    status_override[v.person_idx] = ("已佩戴", best)
                    flips += 1

        # ---- 组装样本（与 run_eval 同构）----
        pred_persons = persons
        gt_persons = [g for g in gt if g["label"] == "person"]
        judged, matched_gi, used = [], set(), set()
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
            matched_gi.add(best)
            if truth_of(gt, gt_persons[best]) is None:
                continue
            vs = [v for v in verdicts if v.person_idx == pi and v.equip == "helmet"]
            if not vs:
                continue
            status = status_override.get(pi, (vs[0].status, None))[0]
            judged.append({"person_id": f"G{best}", "equip": "helmet", "status": status})

        truth = []
        for gi, g in enumerate(gt_persons):
            worn = truth_of(gt, g)
            if worn is None:
                continue
            truth.append({"person_id": f"G{gi}", "equip": "helmet", "worn": worn,
                          "matched": gi in matched_gi})

        samples.append({
            "image_id": fp.stem,
            "gt_boxes": [{"label": g["label"], **{k: g[k] for k in ("x1", "y1", "x2", "y2")}}
                         for g in gt if g["label"] in ("person", "helmet")],
            "pred_boxes": [b.to_dict() for b in preds if b.label in ("person", "helmet")],
            "truth": truth, "judged": judged,
            "latency_ms": res["infer_ms"],
        })
        if i % 25 == 0:
            print(f"  {i}/{len(files)} ... 复核 {n_refine} 次，翻案 {flips} 例")

    res = evaluate(samples, classes=("person", "helmet"), scope="matched")
    wall = time.perf_counter() - t_start

    print("=" * 62)
    print(f"一级 640 + 头部放大复核（refine_conf={args.refine_conf}, "
          f"side×{args.side_scale}）")
    print(f"mAP@0.5   {res['mAP50']}   逐类 "
          f"{ {k: v['ap'] for k, v in res['per_class'].items()} }")
    print(f"误报率     {res['fp_rate']}%   (合规 {res['n_compliant']} 人次)")
    print(f"漏报率     {res['fn_rate']}%   (违规 {res['n_violating']} 人次)")
    print(f"混淆矩阵   {res['confusion']}")
    print(f"二级复核   {n_refine} 次 / 翻案 {flips} 例")
    print(f"单帧耗时   {res['latency']['mean_ms']} ms（仅一级推理）+ "
          f"平均 {wall / len(files) * 1000:.1f} ms 端到端（含复裁与二级）")
    print(f"-> {args.out}")

    payload = {
        "kind": "model-measured-two-stage",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"imgsz": args.imgsz, "conf": args.conf,
                   "refine_conf": args.refine_conf, "side_scale": args.side_scale,
                   "rules_profile": R.active_profile(), "metric_scope": "matched"},
        "refine": {"n_refine": n_refine, "n_flips": flips},
        "results": res,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
