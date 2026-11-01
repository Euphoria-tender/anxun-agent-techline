# -*- coding: utf-8 -*-
"""
安巡 Agent · 判定阈值扫参
=========================

模型不变、检测框不变，只改**判定口径**：
    CONF_FINAL  —— 置信度 ≥ 这个值才下结论
    CONF_REVIEW —— 低于这个值直接丢弃（不进人工）

为什么值得单独扫：漏报 38.7% 里 53% 是「灰区转人工」，也就是说**阈值把结论卡住了**，
不是模型没看见。把 CONF_FINAL 降一点，一部分灰区就能定论——代价是误报上升。
这条曲线必须画出来才知道拐点在哪。

    python src/anxun/sweep_rules.py

产物 data/sweep_rules.json。注意：阈值是在测试集上选的，属于乐观估计，
真实上线要在现场数据上复核（README 里已标注）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anxun import rules_engine as R
from anxun.detectors import DualDetector, pick_device
from anxun.evaluate_real import iou_box, list_images, load_gt, read_names, truth_of

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT = ROOT / "data" / "sweep_rules.json"


def cache_detections(det, limit=None) -> list[dict]:
    """只推理一次，把 gt 与 pred 都存下来，后面反复重算判定。"""
    from PIL import Image
    img_dir = DATA / "images"
    lbl_dir = DATA / "labels"
    names = read_names(DATA)
    out = []
    for fp in list_images(img_dir, limit):
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(lbl_dir / (fp.stem + ".txt"), names, W, H)
        res = det.detect_image(fp)
        out.append({"id": fp.stem, "W": W, "H": H, "gt": gt, "preds": res["boxes"]})
    return out


def score(cache: list[dict]) -> dict:
    """按当前模块常量重算一遍误报/漏报（口径与 evaluate_real 完全一致）。"""
    n_compliant = n_violating = fp = fn = 0
    for s in cache:
        verdicts = R.judge_frame(s["preds"], ["helmet"], area_min=R.min_person_area(s["W"], s["H"]))
        pred_persons = [b for b in s["preds"] if b.label == "person"]
        gt_persons = [g for g in s["gt"] if g["label"] == "person"]
        used = set()
        for pi, pb in enumerate(pred_persons):
            best, bv = None, 0.5
            for gi, g in enumerate(gt_persons):
                if gi in used:
                    continue
                v = iou_box({"x1": pb.x1, "y1": pb.y1, "x2": pb.x2, "y2": pb.y2}, g)
                if v > bv:
                    best, bv = gi, v
            if best is None:
                continue
            used.add(best)
            worn = truth_of(s["gt"], gt_persons[best])
            if worn is None:
                continue
            vs = [v for v in verdicts if v.person_idx == pi and v.equip == "helmet"]
            if not vs:
                continue
            pred_v = (vs[0].status == "未佩戴")
            if worn:
                n_compliant += 1
                fp += pred_v
            else:
                n_violating += 1
                fn += (not pred_v)
    return {
        "fp": fp, "fn": fn,
        "fp_rate": round(fp / n_compliant * 100, 2) if n_compliant else None,
        "fn_rate": round(fn / n_violating * 100, 2) if n_violating else None,
        "n_compliant": n_compliant, "n_violating": n_violating,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="默认 CPU，避免和训练抢 GPU")
    ap.add_argument("--ppe", default=str(ROOT / "models" / "best.pt"), help="PPE 权重")
    ap.add_argument("--person", default=str(ROOT / "models" / "yolov8n.pt"))
    args = ap.parse_args()

    det = DualDetector(args.person, args.ppe, device=pick_device(args.device))
    print(f"PPE 权重：{Path(args.ppe).name}")
    print("缓存检测结果（只推理一次）...")
    cache = cache_detections(det, args.limit or None)
    print(f"  {len(cache)} 张")

    base = (R.CONF_FINAL, R.CONF_REVIEW)
    rows = []
    for cf in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        # 0.15 / 0.20 是交付配置用的复核下限，必须在网格里，否则基线行复现不出来
        for cr in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            if cr >= cf:
                continue
            R.CONF_FINAL, R.CONF_REVIEW = cf, cr
            m = score(cache)
            rows.append({"conf_final": cf, "conf_review": cr, **m,
                         "current": (cf, cr) == base})
    R.CONF_FINAL, R.CONF_REVIEW = base       # 还原，别把模块常量改坏

    # 漏报优先排序（安全场景漏报后果更重），同漏报看误报
    rows.sort(key=lambda r: (r["fn"], r["fp"]))

    payload = {
        "kind": "rules-sweep",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Chapian/PPE_detection · test（100 张）",
        "ppe_weights": Path(args.ppe).name,
        "baseline": {"conf_final": base[0], "conf_review": base[1]},
        "note": "阈值在测试集上选，属乐观估计；现场数据到位后需复核。",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'CONF_FINAL':>11}{'CONF_REVIEW':>13}{'误报率':>10}{'漏报率':>10}{'FP':>5}{'FN':>5}")
    print("-" * 66)
    for r in rows[:12]:
        mark = "  ← 当前" if r["current"] else ""
        print(f"{r['conf_final']:>11.2f}{r['conf_review']:>13.2f}"
              f"{r['fp_rate']:>9.1f}%{r['fn_rate']:>9.1f}%{r['fp']:>5}{r['fn']:>5}{mark}")
    print("=" * 66)
    best = rows[0]
    print(f"漏报最低：CONF_FINAL={best['conf_final']} / CONF_REVIEW={best['conf_review']} "
          f"→ 漏报 {best['fn_rate']}% 误报 {best['fp_rate']}%")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
