# -*- coding: utf-8 -*-
"""
目标可达性扫描：四个指标能不能同时达标
=====================================

目标（立项时定的）：mAP ≥ 0.85 / 误报 ≤ 10% / 漏报 ≤ 5% / 单帧 ≤ 50 ms。

mAP 与耗时由模型决定，调参动不了；能动的只有判定层。本脚本扫 `CONF_FINAL`
全区间，回答一个问题：**存不存在一组阈值，让 (fp, fn) 同时落进目标框**。

同时把漏报拆成两笔账，因为它们是两回事：

- **严格漏报**：系统明确说「已佩戴」，实际没戴 → 系统骗了人
- **未定论**：系统说「无法判定 / 待复核」 → 转人工，是降级不是漏

`business_rates()` 现在把两者都算进 fn（见 metrics.py 第 132–143 行：
只有 `st == "未佩戴"` 才算判违规，其余一律计入 fn）。这个口径偏严，
但对安全场景不算错——系统没告警就是没告警。本脚本两个数都报，不替谁遮掩。

    python tools/probe_target_reach.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anxun.rules_engine as R
from anxun.detectors import DualDetector, pick_device
from anxun.evaluate_real import iou_box, list_images, load_gt, read_names, truth_of
from anxun.rules_engine import judge_frame, min_person_area

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"

# 目标（来自立项文档）
TARGET_FP = 10.0          # ≤ 10%
TARGET_FN = 5.0           # ≤ 5%


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
        preds = det.detect_image(fp)["boxes"]
        gt_persons = [g for g in gt if g["label"] == "person"]
        pred_persons = [b for b in preds if b.label == "person"]
        pairs, used = [], set()
        for pi, pbox in enumerate(pred_persons):
            best, bv = None, 0.5
            for gi, g in enumerate(gt_persons):
                if gi in used:
                    continue
                v = iou_box({"x1": pbox.x1, "y1": pbox.y1,
                             "x2": pbox.x2, "y2": pbox.y2}, g)
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
                      "area_min": min_person_area(W, H)})
    print(f"缓存 {len(cache)} 张（含框与真值匹配）\n")

    def measure(conf_final: float, conf_review: float) -> dict:
        """按给定阈值重算判定，返回两套漏报口径。"""
        R.CONF_FINAL, R.CONF_REVIEW = conf_final, conf_review
        tp = fn = fp = tn = 0
        undecided = 0
        fn_strict = 0
        for c in cache:
            verdicts = judge_frame(c["preds"], ["helmet"], area_min=c["area_min"])
            got = {}
            for pi, gi, _w in c["pairs"]:
                vs = [v for v in verdicts if v.person_idx == pi and v.equip == "helmet"]
                if vs:
                    got[gi] = vs[0].status
            for t in c["truth"]:
                if not t["matched"]:
                    continue
                st = got.get(int(t["person_id"][1:]), "无法判定")
                truth_v = not t["worn"]
                pred_v = (st == "未佩戴")
                if truth_v:
                    if pred_v:
                        tp += 1
                    else:
                        fn += 1
                        if st != "已佩戴":
                            undecided += 1
                        else:
                            fn_strict += 1
                else:
                    if pred_v:
                        fp += 1
                    else:
                        tn += 1
        n_c, n_v = fp + tn, fn + tp
        return {
            "fp": round(fp / n_c * 100, 2) if n_c else None,
            "fn": round(fn / n_v * 100, 2) if n_v else None,
            "fp_n": fp, "fn_n": fn, "tn_n": tn, "tp_n": tp,
            "undecided": undecided, "fn_strict": fn_strict,
            "fn_strict_pct": round(fn_strict / n_v * 100, 2) if n_v else None,
            "n_v": n_v, "n_c": n_c,
        }

    base = (R.CONF_FINAL, R.CONF_REVIEW)
    print("=" * 78)
    print("一、当前口径拆分（conf_final {:.2f} / conf_review {:.2f}）".format(*base))
    print("=" * 78)
    b = measure(*base)
    print(f"  违规 {b['n_v']} 人次 · 合规 {b['n_c']} 人次")
    print(f"  混淆矩阵      tp {b['tp_n']} · fn {b['fn_n']} · fp {b['fp_n']} · tn {b['tn_n']}")
    print(f"  漏报（现口径）{b['fn']}%   ← 含「无法判定/待复核」")
    print(f"  漏报（严格）  {b['fn_strict_pct']}%   ← 只算「系统说已佩戴」")
    print(f"  未定论        {b['undecided']} 例（{b['undecided']/b['n_v']*100:.1f}%）转人工")

    print()
    print("=" * 78)
    print("二、扫 CONF_FINAL：能不能让 (误报, 漏报) 同时进目标框")
    print(f"    目标：误报 ≤ {TARGET_FP}%（≤9 例）· 漏报 ≤ {TARGET_FN}%（≤4 例）")
    print("=" * 78)
    print(f"{'conf':>6}{'误报':>9}{'漏报':>9}{'严格漏报':>10}{'未定论':>8}   fp/fn  达标?")
    hit = []
    for i in range(12):
        cf = round(0.30 + i * 0.05, 2)
        r = measure(cf, base[1])
        ok_fp = r["fp"] is not None and r["fp"] <= TARGET_FP
        ok_fn = r["fn"] is not None and r["fn"] <= TARGET_FN
        mark = "✅ 达标" if (ok_fp and ok_fn) else (
            f"{'fp✓' if ok_fp else 'fp✗'} {'fn✓' if ok_fn else 'fn✗'}")
        print(f"{cf:>6.2f}{r['fp']:>8.2f}%{r['fn']:>8.2f}%{r['fn_strict_pct']:>9.2f}%"
              f"{r['undecided']:>8}   {r['fp_n']}/{r['fn_n']}  {mark}")
        if ok_fp and ok_fn:
            hit.append(cf)

    print()
    print("=" * 78)
    print("三、结论")
    print("=" * 78)
    if hit:
        print(f"  存在达标点：conf_final = {hit}")
    else:
        print("  ❌ 扫描区间内不存在同时满足 (误报≤10%, 漏报≤5%) 的阈值。")
        print("     fp 与 fn 是同一条权衡曲线的两端：压一头必抬另一头。")
        print("     要同时达标，只能让曲线整体内移 —— 即模型变强，不是调参。")

    R.CONF_FINAL, R.CONF_REVIEW = base
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
