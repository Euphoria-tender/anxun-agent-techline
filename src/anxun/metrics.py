# -*- coding: utf-8 -*-
"""
安巡 Agent · 指标评测
=====================

四个指标，一套口径，一条命令：

    mAP@0.5        检测层：框准不准
    误报率          业务层：合规的人被冤枉了多少
    漏报率          业务层：违规的人被放过了多少
    单帧耗时        工程层：能不能实时

误报率/漏报率按**人×装备**计，不按框计——评委关心的是「有没有冤枉人」，不是框的 F1。

评测数据格式（与 YOLO 一致，换成真数据集不用改代码）：

    samples = [{
        "image_id": "self_workshop-a_0031",
        "gt_boxes":  [{"label":"person","x1":..,"y1":..,"x2":..,"y2":..}, ...],
        "pred_boxes":[{"label":"person","conf":0.93,"x1":..,...}, ...],
        "truth":  [{"person_id":"P1","equip":"helmet","worn":true}, ...],
        "judged": [{"person_id":"P1","equip":"helmet","status":"已佩戴"}, ...],
        "latency_ms": 4.2,
    }, ...]
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence


# ============================================================
# 一、mAP@0.5

def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ub = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def map_at_50(samples: Sequence[dict], classes: Sequence[str] = ("person", "helmet"),
              iou_thr: float = 0.5) -> dict:
    """标准 mAP@0.5：逐类别贪心匹配，按置信度降序算 PR 曲线。"""
    out = {}
    for cls in classes:
        npos = sum(1 for s in samples for g in s["gt_boxes"] if g["label"] == cls)
        if npos == 0:
            out[cls] = {"ap": None, "npos": 0, "tp": 0, "fp": 0, "curve": []}
            continue

        records = []  # (conf, is_tp)
        for s in samples:
            gts = [g for g in s["gt_boxes"] if g["label"] == cls]
            preds = sorted([p for p in s["pred_boxes"] if p["label"] == cls],
                           key=lambda p: p["conf"], reverse=True)
            used = set()
            for p in preds:
                best_i, best_v = -1, 0.0
                for i, g in enumerate(gts):
                    if i in used:
                        continue
                    v = _iou(p, g)
                    if v > best_v:
                        best_i, best_v = i, v
                if best_v >= iou_thr and best_i >= 0:
                    used.add(best_i)
                    records.append((p["conf"], True))
                else:
                    records.append((p["conf"], False))

        records.sort(key=lambda r: r[0], reverse=True)
        tp = fp = 0
        curve = []
        for conf, ok in records:
            tp += ok
            fp += (not ok)
            curve.append({"conf": round(conf, 3),
                          "recall": round(tp / npos, 4),
                          "precision": round(tp / (tp + fp), 4)})

        # 全点插值 AP
        mrec = [0.0] + [c["recall"] for c in curve] + [1.0]
        mpre = [1.0] + [c["precision"] for c in curve] + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        out[cls] = {"ap": round(ap, 4), "npos": npos, "tp": tp, "fp": fp,
                    "curve": curve}
    aps = [v["ap"] for v in out.values() if v["ap"] is not None]
    out["mAP"] = round(sum(aps) / len(aps), 4) if aps else None
    return out


# ============================================================
# 二、业务层：误报率 / 漏报率 / 混淆矩阵

def business_rates(samples: Sequence[dict]) -> dict:
    """
    误报率 = 实际合规却被判「未佩戴」的人×装备 / 实际合规总数
    漏报率 = 实际违规却没有被判「未佩戴」的人×装备 / 实际违规总数
    """
    fp = n_compliant = 0
    fn = n_violating = 0
    cm = defaultdict(int)     # (真值, 判定) -> 次数
    fn_cases = []

    for s in samples:
        judged = {(j["person_id"], j["equip"]): j["status"] for j in s["judged"]}
        for t in s["truth"]:
            key = (t["person_id"], t["equip"])
            st = judged.get(key, "无法判定")
            truth_v = not t["worn"]
            pred_v = (st == "未佩戴")
            cm[("违规" if truth_v else "合规", "判违规" if pred_v else "未判违规")] += 1
            if not truth_v:
                n_compliant += 1
                if pred_v:
                    fp += 1
            else:
                n_violating += 1
                if not pred_v:
                    fn += 1
                    fn_cases.append({"image_id": s["image_id"], "person": t["person_id"],
                                     "equip": t["equip"], "judged": st,
                                     "why": s.get("truth_note", "")})

    return {
        "fp_rate": round(fp / n_compliant * 100, 2) if n_compliant else None,
        "fn_rate": round(fn / n_violating * 100, 2) if n_violating else None,
        "n_compliant": n_compliant,
        "n_violating": n_violating,
        "collateral": {"合规判违规": fp, "违规判违规": n_violating - fn},
        "confusion": {
            "tp": cm[("违规", "判违规")],
            "fn": cm[("违规", "未判违规")],
            "fp": cm[("合规", "判违规")],
            "tn": cm[("合规", "未判违规")],
        },
        "fn_cases": fn_cases[:12],
    }


# ============================================================
# 三、耗时

def latency(samples: Sequence[dict], real_ms: float | None = None) -> dict:
    xs = sorted(s.get("latency_ms", 0.0) for s in samples)
    if not xs:
        return {"mean_ms": None, "p95_ms": None, "n": 0}
    p95 = xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]
    return {
        "mean_ms": round(sum(xs) / len(xs), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(xs[-1], 2),
        "n": len(xs),
        "pipeline_mean_ms": round(real_ms, 2) if real_ms else None,
    }


# ============================================================
# 四、汇总

def evaluate(samples: Sequence[dict], classes=("person", "helmet"),
             real_ms: float | None = None) -> dict:
    det = map_at_50(samples, classes)
    biz = business_rates(samples)
    lat = latency(samples, real_ms)
    return {
        "n_images": len(samples),
        "n_boxes": sum(len(s["pred_boxes"]) for s in samples),
        "mAP50": det.get("mAP"),
        "per_class": {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                      for k, v in det.items() if k != "mAP"},
        "curves": {k: v.get("curve", []) for k, v in det.items() if k != "mAP"},
        "fp_rate": biz["fp_rate"],
        "fn_rate": biz["fn_rate"],
        "confusion": biz["confusion"],
        "n_compliant": biz["n_compliant"],
        "n_violating": biz["n_violating"],
        "fn_cases": biz["fn_cases"],
        "latency": lat,
        "targets": {"mAP50": 0.85, "fp_rate": 10.0, "fn_rate": 5.0, "latency_ms": 50.0},
    }
