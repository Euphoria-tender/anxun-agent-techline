# -*- coding: utf-8 -*-
"""
视频级（时序聚合后）误报/漏报
=============================

为什么单独算：单帧评测是 100 张静态图，每张独立判定；真实部署是视频，
`aggregate_temporal` 要求**连续 3 帧结论一致才定论**。这两件事的指标不一样。

素材：`datasets/site_pan.mp4`（34.8 s，整幅照片匀速平移的合成视频），
真值 `data/site_pan_truth.json` 由生成器给出。
账本：`data/events_pan_v3.json`（v3.0.0 权重 + 中心点判据）。

**匹配方式（第一版踩过的坑）**：最初只按时间窗对齐，结果 12 个人全被配成
「未佩戴」、误报 100%。原因是画面里同时有多个人，轨迹在时间上完全重叠——
账本里没有位置信息就分不清谁是谁。现在账本带 `bbox`，改为
「时间窗筛候选 + 框 IoU 定人」，并对合成素材的平移量做补偿。

    python tools/eval_temporal.py
    python tools/eval_temporal.py --ledger data/events_pan.json   # 旧账本无 bbox，自动退回时间窗
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "data" / "site_pan_truth.json"
LEDGER = ROOT / "data" / "events_pan_v3.json"


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_person(bbox, gt_boxes, pan_px: float) -> tuple[int, float]:
    """在候选真值人里找与账本框最像的那个（含平移补偿）。"""
    best_i, best_v = -1, 0.0
    for i, g in enumerate(gt_boxes):
        # 合成素材是整幅平移，人物在画面里的 x 会偏移 ±pan_px
        for dx in (0.0, -pan_px, pan_px, -pan_px / 2, pan_px / 2):
            v = iou(bbox, [g[0] + dx, g[1], g[2] + dx, g[3]])
            if v > best_v:
                best_i, best_v = i, v
    return best_i, best_v


def pick_verdict(rows: list[dict]) -> str:
    """同一人可能有多条状态记录，取最保守的结论。"""
    statuses = [r["status"] for r in rows]
    for s in ("未佩戴", "待复核", "已佩戴", "无法判定"):
        if s in statuses:
            return s
    return "无记录"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(TRUTH))
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--window", type=float, default=1.0, help="时间窗容差（秒）")
    ap.add_argument("--iou-min", type=float, default=0.25, help="空间匹配的 IoU 下限")
    ap.add_argument("--equip", default="helmet", choices=("helmet", "vest"),
                    help="按装备分开算：这批素材 helmet 全合规（只能测误报），"
                         "vest 大多没穿（能测漏报），两类不能合并统计")
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    led = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    persons = truth["persons"]
    W, H = truth["size"]
    pan = float(truth.get("pan_px", 0))
    rows = [r for r in led["ledger"] if r["equip"] == args.equip]

    has_bbox = any("bbox" in r for r in rows)
    print(f"真值 {len(persons)} 人 · 账本 {args.equip} {len(rows)} 条 · "
          f"匹配方式：{'时间窗 + 框 IoU' if has_bbox else '仅时间窗（旧账本无 bbox）'}\n")

    # 真值框转像素
    gt_px = [[p["box_norm"][0] * W, p["box_norm"][1] * H,
              p["box_norm"][2] * W, p["box_norm"][3] * H] for p in persons]

    assigned: dict[int, list[dict]] = {i: [] for i in range(len(persons))}
    unmatched = 0
    for r in rows:
        t = r["t_video_s"]
        cand = [i for i, p in enumerate(persons)
                if p["t_enter"] - args.window <= t <= p["t_exit"] + args.window]
        if not cand:
            unmatched += 1
            continue
        if has_bbox and "bbox" in r:
            sub = [gt_px[i] for i in cand]
            k, score = match_person(r["bbox"], sub, pan)
            if k >= 0 and score >= args.iou_min:
                r = dict(r); r["_match_iou"] = round(score, 3)
                assigned[cand[k]].append(r)
            else:
                unmatched += 1
        else:
            # 旧账本：只能按时间窗，多人重叠时归属不明，统一记到第一个
            assigned[cand[0]].append(r)

    tp = fn = fp = tn = 0
    undecided = []
    detail = []
    for i, p in enumerate(persons):
        verdict = pick_verdict(assigned[i])
        real_worn = (p["equip"] == "已佩戴") if args.equip == "helmet" else (p.get("vest") is True)
        truth_label = (p["equip"] if args.equip == "helmet"
                       else ("穿了" if p.get("vest") is True else "没穿"))
        pred_v = verdict == "未佩戴"
        if verdict in ("无记录", "无法判定", "待复核"):
            undecided.append((p["id"], p["equip"], verdict))
            detail.append({"id": p["id"], "truth": truth_label, "verdict": verdict,
                           "n_records": len(assigned[i]), "result": "未定论"})
            continue
        if real_worn and pred_v:
            fp += 1; res = "误报"
        elif (not real_worn) and (not pred_v):
            fn += 1; res = "漏报"
        elif real_worn:
            tn += 1; res = "正确(合规)"
        else:
            tp += 1; res = "正确(违规)"
        detail.append({"id": p["id"], "truth": truth_label, "verdict": verdict,
                       "n_records": len(assigned[i]), "result": res,
                       "match_iou": [r.get("_match_iou") for r in assigned[i]],
                       "conf": [r["conf"] for r in assigned[i]]})

    for d in detail:
        flag = "  ←" if d["result"] in ("误报", "漏报") else ""
        extra = ""
        if d.get("match_iou"):
            extra = f" IoU {d['match_iou']} conf {d['conf']}"
        print(f"  {d['id']:<6} 真值 {d['truth']:<4} 系统 {d['verdict']:<5}"
              f" 记录{d['n_records']} {d['result']}{flag}{extra}")

    n_c, n_v = fp + tn, fn + tp
    print(f"\n可判人次：合规 {n_c} · 违规 {n_v}；未定论 {len(undecided)}；"
          f"账本未匹配 {unmatched} 条")
    print(f"混淆矩阵：tp {tp} · fn {fn} · fp {fp} · tn {tn}")
    if n_c:
        print(f"**视频级误报率：{fp / n_c * 100:.2f}%**  ({fp}/{n_c})")
    if n_v:
        print(f"**视频级漏报率：{fn / n_v * 100:.2f}%**  ({fn}/{n_v})")

    out = {
        "kind": "temporal-verdict", "equip": args.equip,
        "source": truth["source"], "ledger": Path(args.ledger).name,
        "window_s": args.window, "iou_min": args.iou_min, "match": "spatial" if has_bbox else "time-only",
        "n_persons": len(persons), "n_compliant": n_c, "n_violating": n_v,
        "n_undecided": len(undecided), "n_unmatched_records": unmatched,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "fp_rate": round(fp / n_c * 100, 2) if n_c else None,
        "fn_rate": round(fn / n_v * 100, 2) if n_v else None,
        "detail": detail,
    }
    p = ROOT / "data" / f"temporal_verdict_{args.equip}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
