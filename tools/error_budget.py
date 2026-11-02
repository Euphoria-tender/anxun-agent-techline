# -*- coding: utf-8 -*-
"""
漏报误差预算：把「漏报率 17.20%」拆到每一例
==========================================

漏报率 = 16/93（matched 口径）。README 里有一张手写的归因表（10 / 4 / 2），
但那是**人工数的**，没有脚本可复现。这个脚本把它变成可复现的数字：

    对每个 truth（worn=False，即真值违规）但 judged 不是「未佩戴」的人次，按判定结果归类：
      「已佩戴」  -> 模型看错：裸头被框成安全帽（规则层无解）
      「待复核」  -> 灰区：置信度落在 [0.20, 0.30)
      「无法判定」-> 小目标 / 半身 / 头部不在判定区（设计取舍）
      缺失        -> 未判定（不该出现，出现就是 bug）

    python tools/error_budget.py
    python tools/error_budget.py --out data/error_budget.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anxun.detectors import DualDetector, pick_device            # noqa: E402
from anxun.evaluate_real import run_eval, warmup                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT = ROOT / "data" / "error_budget.json"

CATS = {
    "已佩戴": ("model_confused", "模型看错：裸头被框成安全帽"),
    "待复核": ("grey_zone", "灰区：置信度落在 [0.20, 0.30)"),
    "无法判定": ("undecidable", "无法判定：小目标 / 半身 / 头部不在判定区"),
}
LABELS = {c: label for c, label in CATS.values()}
LABELS["missing"] = "未判定（异常，应排查）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    det = DualDetector(args.person_weights, args.weights, imgsz=640, conf=0.25,
                       device=pick_device(""))
    warmup(det)
    out = run_eval(args.data, det, limit=args.limit or None, scope="matched")
    samples = out["samples"]

    counter: Counter[str] = Counter()
    cases: list[dict] = []
    n_violating = 0

    for s in samples:
        judged = {(j["person_id"], j["equip"]): j["status"] for j in s["judged"]}
        for t in s["truth"]:
            if t["worn"]:                       # 只看真值违规的人
                continue
            if not t.get("matched", True):      # matched 口径：没检出的人不进分母
                continue
            n_violating += 1
            st = judged.get((t["person_id"], t["equip"]), "缺失")
            if st == "未佩戴":                  # 判对了，不是漏报
                continue
            cat, why = CATS.get(st, ("missing", f"未判定（{st}）"))
            counter[cat] += 1
            cases.append({"image_id": s["image_id"], "person_id": t["person_id"],
                          "status": st, "category": cat, "why": why})

    n_miss = sum(counter.values())
    payload = {
        "kind": "error-budget",
        "scope": "matched（只统计已检出的人，与 headline 17.20% 同口径）",
        "dataset": {"name": "Chapian/PPE_detection · test", "images": out["images"]},
        "violating_total": n_violating,
        "missed_total": n_miss,
        "fn_rate_recomputed": round(n_miss / n_violating * 100, 2) if n_violating else None,
        "breakdown": [
            {"category": k, "count": v,
             "share": round(v / n_miss, 4) if n_miss else None,
             "label": LABELS.get(k, k)}
            for k, v in counter.most_common()
        ],
        "cases": cases,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"违规人次 {n_violating}，漏报 {n_miss}，重算漏报率 "
          f"{payload['fn_rate_recomputed']}%（交付值 17.20%）\n")
    for b in payload["breakdown"]:
        print(f"  {b['label']:<32} {b['count']:>3} 例  {b['share']*100:>5.1f}%")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
