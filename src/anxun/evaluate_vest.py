# -*- coding: utf-8 -*-
"""
反光背心（vest）在**独立测试集**上的评测
==========================================

为什么单独写一个：`evaluate_real.py` 的口径是围绕 helmet 建的（头部区域、四态、时序），
vest 的判定区域是上身（person 框 20%~70% 高度），真值推导也不同，硬塞进同一个函数会把两件事搅混。

口径与 helmet 完全对齐：
  - 检测级：逐类 mAP（vest / no_vest）
  - 业务级：误报率（穿了被判没穿）/ 漏报率（没穿被判穿了）
  - 真值推导：上身区域里有 vest 框 = 合规，只有 no_vest = 违规，都没有 = 不进统计

    python src/anxun/evaluate_vest.py

⚠️ vest 模型的训练集是 Chapian 的 train 分片，这里评的是 **test 分片**（100 张全程未参与）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image                                     # noqa: E402

from anxun import rules_engine as R                       # noqa: E402
from anxun.rules_engine import Box                           # noqa: E402
from anxun.detectors import DualDetector, pick_device     # noqa: E402
from anxun.evaluate_real import (list_images, load_gt,     # noqa: E402
                                 read_names, iou_box)
from anxun.metrics import evaluate                        # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "ppe_test" / "test"


def torso_region(person: dict) -> Box:
    p = Box("person", 1.0, person["x1"], person["y1"], person["x2"], person["y2"])
    return Box("torso", 1.0, p.x1, p.y1 + p.h * 0.20, p.x2, p.y1 + p.h * 0.70)


def truth_vest(gt: list[dict], person: dict) -> bool | None:
    """上身区域里有 vest = 穿了，只有 no_vest = 没穿，都没有 = None。"""
    reg = torso_region(person)
    worn, seen = False, False
    for g in gt:
        if g["label"] not in ("vest", "novest"):
            continue
        cx, cy = (g["x1"] + g["x2"]) / 2, (g["y1"] + g["y2"]) / 2
        if reg.x1 <= cx <= reg.x2 and reg.y1 <= cy <= reg.y2:
            seen = True
            worn = worn or g["label"] == "vest"
    return worn if seen else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--person", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--vest", default=str(ROOT / "models" / "vest.pt"))
    ap.add_argument("--helmet", default=str(ROOT / "models" / "best.pt"),
                    help="helmet 权重也一起加载，用来确认「加了一个模型不影响原有判定」")
    ap.add_argument("--no-helmet", action="store_true", help="只加载 vest，不加载 helmet")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    vw = Path(args.vest)
    if not vw.exists():
        print(f"找不到 vest 权重：{vw}\n"
              f"v3.0.0 起背心已并入 models/best.pt 的五类权重，本脚本只用于\n"
              f"「独立背心模型」的对照实验，日常评测不需要它。\n"
              f"确实要独立模型：python src/anxun/finetune.py "
              f"--data datasets/vest/data.yaml --init models/yolov8n.pt --name vest_ft")
        return 2

    det = DualDetector(args.person, "" if args.no_helmet else args.helmet,
                       vest_weights=str(vw), device=pick_device(args.device))
    print(f"模型：{det.name}  imgsz=640")
    det.detect_frame(__import__("numpy").zeros((640, 640, 3), dtype=__import__("numpy").uint8))

    data = Path(args.data)
    names = read_names(data)
    files = list_images(data / "images")

    samples = []
    lat = []
    fp = fn = n_c = n_v = 0
    for i, fp_ in enumerate(files):
        with Image.open(fp_) as im:
            W, H = im.size
        gt = load_gt(data / "labels" / f"{fp_.stem}.txt", names, W, H)
        res = det.detect_image(fp_)
        lat.append(res["infer_ms"])
        preds = res["boxes"]

        # 检测级：只统计 vest / no_vest
        gtb = [g for g in gt if g["label"] in ("vest", "novest")]
        prb = [b.to_dict() for b in preds if b.label in ("vest", "novest")]

        # 业务级：按人判。judged/truth 要按 evaluate_real 的口径填，
        # 否则 metrics.business_rates 没有输入，误报漏报只能是 None。
        verdicts = R.judge_frame(preds, ["vest"], area_min=R.min_person_area(W, H))
        judged, truth = [], []
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
            worn = truth_vest(gt, gp[best])
            if worn is None:
                continue
            it = [v for v in verdicts if v.person_idx == pi]
            if not it:
                continue
            pv = it[0].status == "未佩戴"
            judged.append({"person_id": f"G{best}", "equip": "vest",
                           "status": it[0].status})
            truth.append({"person_id": f"G{best}", "equip": "vest", "worn": worn})
            if worn:
                n_c += 1
                fp += pv
            else:
                n_v += 1
                fn += (not pv)
        samples.append({"image_id": fp_.stem,
                        "gt_boxes": [{"label": g["label"], **{k: g[k] for k in ("x1", "y1", "x2", "y2")}}
                                     for g in gtb],
                        "pred_boxes": prb, "truth": truth, "judged": judged,
                        "latency_ms": res["infer_ms"]})
        if i % 50 == 0:
            print(f"  {i}/{len(files)} ...")

    r = evaluate(samples, classes=("vest", "novest"))
    lat.sort()
    print("=" * 62)
    print(f"图片 {len(files)}   vest 检测框 {sum(len(s['pred_boxes']) for s in samples)}")
    print(f"mAP@0.5   {r['mAP50']}   逐类 "
          f"{ {k: v['ap'] for k, v in r['per_class'].items()} }")
    print(f"误报率    {r['fp_rate']}%   (合规 {n_c} 人次)   穿了被判没穿")
    print(f"漏报率    {r['fn_rate']}%   (违规 {n_v} 人次)   没穿被判穿了")
    print(f"混淆矩阵   {r['confusion']}")
    print(f"单帧耗时   {round(sum(lat) / len(lat), 1)} ms（三模型串联）")
    out = ROOT / "data" / "metrics_vest.json"
    out.write_text(json.dumps(
        {"kind": "model-measured",
         "config": {"models": det.name, "vest": Path(args.vest).name,
                    "device": f"GPU（torch）"},
         "dataset": {"name": "Chapian/PPE_detection · test", "license": "CC BY 4.0",
                     "images": len(files), "n_compliant": n_c, "n_violating": n_v,
                     "eval_note": "vest 训练用的是 train 分片，本页全部数字来自未参与的 test 分片"},
         "results": r}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
