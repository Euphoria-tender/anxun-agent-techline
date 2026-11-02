# -*- coding: utf-8 -*-
"""
安巡 Agent · 第二数据源独立验证
================================

**这个脚本只回答一个问题：helmet 检测在「没参与过训练的 706 张」上还准不准。**

为什么单开一个脚本，不复用 `evaluate_real.py`：
706 张（`toborobot/Safety-helmet-dataset` 的 test split）只有 helmet / nohelmet 两类标注，
**没有 person**。而 `evaluate_real.py` 的业务指标（误报率 / 漏报率）是「人×装备」口径，
必须有 person 真值才能算。硬套会得到两套不可比的数字。

所以口径写死在这里：

    能算：helmet 单类 AP@0.5、框级 precision / recall / F1、单帧耗时
    不能算：人次级误报率 6.12% / 漏报率 17.20%、三类 mAP 0.6808

**两个口径不并列、不混算、不加总。**

与主口径的可比性（`data/metrics_real.json` → `results.per_class.helmet`）：
同一 `models/best.pt`、同一 `imgsz 640`、同一 `conf 0.25`、同一 `map_at_50` 实现、
同一 IoU 0.5、同样是「helmet 对背景」单类口径（真值里的 nohelmet 框不参与 helmet 的 AP）。

    python tools/eval_second_source.py
    python tools/eval_second_source.py --limit 100 --device cpu

产出 `data/metrics_second_source.json`。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anxun.detectors import YoloDetector, pick_device          # noqa: E402
from anxun.metrics import map_at_50                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "helmet_p2" / "test"
WEIGHTS = ROOT / "models" / "best.pt"
OUT = ROOT / "data" / "metrics_second_source.json"

# 落盘后的类别顺序，由 tools/fetch_helmet_p2.py 的 NAMES 决定（REMAP 已重排过）
NAMES = ["helmet", "nohelmet"]


def load_gt(label_path: Path, w: int, h: int) -> list[dict]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls >= len(NAMES):
            continue
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        out.append({"label": NAMES[cls],
                    "x1": (cx - bw / 2) * w, "y1": (cy - bh / 2) * h,
                    "x2": (cx + bw / 2) * w, "y2": (cy + bh / 2) * h})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from PIL import Image

    data = Path(args.data)
    img_dir = data / "images" if (data / "images").exists() else data
    lbl_dir = data / "labels" if (data / "labels").exists() else data
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"没有图片：{img_dir}", file=sys.stderr)
        return 2
    if args.limit:
        step = max(1, len(files) // args.limit)
        files = files[::step][:args.limit]

    det = YoloDetector(args.weights, imgsz=args.imgsz, conf=args.conf,
                       device=pick_device(args.device))

    samples, lat = [], []
    n_gt_box = n_pred_box = 0
    empty_gt = 0

    for i, fp in enumerate(files, 1):
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(lbl_dir / (fp.stem + ".txt"), W, H)
        if not gt:
            empty_gt += 1
        res = det.detect_image(fp)
        preds = [b for b in res["boxes"] if b.label in ("helmet", "nohelmet")]
        n_gt_box += len(gt)
        n_pred_box += len(preds)
        lat.append(res["infer_ms"])
        samples.append({
            "image_id": fp.stem,
            "gt_boxes": gt,
            "pred_boxes": [b.to_dict() for b in preds],
        })
        if i % 100 == 0:
            print(f"  {i}/{len(files)} ...")

    # 只算 helmet：与主口径 results.per_class.helmet 同口径（nohelmet 不进 helmet 的 AP）
    helmet = map_at_50(samples, classes=("helmet",))["helmet"]
    nohelmet = map_at_50(samples, classes=("nohelmet",))["nohelmet"]

    def pr(x: dict) -> dict:
        tp, fp_, npos = x["tp"], x["fp"], x["npos"]
        p = tp / (tp + fp_) if (tp + fp_) else None
        r = tp / npos if npos else None
        f1 = (2 * p * r / (p + r)) if (p and r) else None
        return {"tp": tp, "fp": fp_, "npos": npos,
                "precision": round(p, 4) if p is not None else None,
                "recall": round(r, 4) if r is not None else None,
                "f1": round(f1, 4) if f1 is not None else None}

    payload = {
        "kind": "model-measured",
        "scope": "detection-only",
        "comparable_to": "data/metrics_real.json → results.per_class.helmet",
        "not_comparable": [
            "人次级误报率 / 漏报率（6.12% / 17.20%）——本集无 person 标注",
            "三类 mAP（0.6808）——本集只有 helmet / nohelmet 两类",
        ],
        "config": {
            "weights": Path(args.weights).name,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": pick_device(args.device) or "auto",
        },
        "dataset": {
            "name": "toborobot/Safety-helmet-dataset · test",
            "split": "Phat_project-3/test",
            "source": "https://hf-mirror.com/datasets/toborobot/Safety-helmet-dataset",
            "license": "MIT",
            "images": len(files),
            "boxes_gt": n_gt_box,
            "boxes_pred": n_pred_box,
            "images_without_gt": empty_gt,
            "note": ("训练只用该数据源的 train 1200 + valid 400，本 split 从未参与训练；"
                     "但与该训练集同源，所以它是「同源独立留出集」，不是跨域评测。"),
        },
        "results": {
            "n_images": len(files),
            "n_boxes": n_pred_box,
            "per_class": {
                "helmet": {**{k: v for k, v in helmet.items() if k != "curve"}, **pr(helmet)},
                "nohelmet": {**{k: v for k, v in nohelmet.items() if k != "curve"}, **pr(nohelmet)},
            },
            "latency_ms": {
                "mean": round(statistics.fmean(lat), 2),
                "p50": round(statistics.median(lat), 2),
            },
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    h = payload["results"]["per_class"]["helmet"]
    print(f"\n706 张 / helmet 单类：AP@0.5 = {h['ap']}  "
          f"P = {h['precision']}  R = {h['recall']}  F1 = {h['f1']}")
    print(f"  真值框 {h['npos']} / 命中 {h['tp']} / 误检 {h['fp']}  "
          f"（对比 100 张主口径：npos 146 / tp 122 / fp 131 / AP 0.5645）")
    print(f"  单帧耗时 mean {payload['results']['latency_ms']['mean']} ms")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
