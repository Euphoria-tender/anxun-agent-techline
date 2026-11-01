# -*- coding: utf-8 -*-
"""
安巡 Agent · 配置扫参
=====================

现场是**实时监控视频巡查**，所以不能只看「哪个模型更准」，
要看「在撑得住多路视频的算力下，哪个最准」。这个脚本就是做这个权衡的。

    python src/anxun/sweep.py

对每个配置跑同一个 100 张真实测试集，输出：

    mAP@0.5 / helmet AP / 误报率 / 漏报率 / 单帧耗时 / 单进程能带几路（按 1fps 抽帧）

结论会写进 data/sweep.json，并用 --apply 把最优配置落到 README 的推荐参数里。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anxun.detectors import DualDetector
from anxun.evaluate_real import run_eval, warmup

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT = ROOT / "data" / "sweep.json"

PERSON = ROOT / "models" / "yolov8n.pt"

# (标签, 安全帽权重, imgsz, conf)
CONFIGS = [
    ("s @480",  "models/best.pt",      480, 0.25),
    ("s @640",  "models/best.pt",      640, 0.25),
    ("s @960",  "models/best.pt",      960, 0.25),
    ("m @480",  "models/hardhat_m.pt", 480, 0.25),
    ("m @640",  "models/hardhat_m.pt", 640, 0.25),
    ("m @960",  "models/hardhat_m.pt", 960, 0.25),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试用）")
    ap.add_argument("--configs", default="", help="逗号分隔的标签过滤，如 's @640,m @640'")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"找不到数据集：{DATA}")
        return 2

    wanted = [c.strip() for c in args.configs.split(",")] if args.configs else None
    rows = []
    for label, w, imgsz, conf in CONFIGS:
        if wanted and label not in wanted:
            continue
        wp = ROOT / w
        if not wp.exists():
            print(f"跳过 {label}：缺权重 {w}")
            continue

        print(f"\n▶ {label}  ({Path(w).name}, imgsz={imgsz}, conf={conf})")
        det = DualDetector(str(PERSON), str(wp), imgsz=imgsz, conf=conf)
        t0 = time.perf_counter()
        warmup(det)                      # 预热不计入
        out = run_eval(DATA, det, limit=args.limit or None)
        wall = time.perf_counter() - t0
        r = out["results"]
        ms = r["latency"]["mean_ms"]
        row = {
            "label": label,
            "ppe_weights": Path(w).name,
            "imgsz": imgsz,
            "conf": conf,
            "mAP50": r["mAP50"],
            "helmet_ap": r["per_class"].get("helmet", {}).get("ap"),
            "person_ap": r["per_class"].get("person", {}).get("ap"),
            "fp_rate": r["fp_rate"],
            "fn_rate": r["fn_rate"],
            "ms": ms,
            "p95_ms": r["latency"]["p95_ms"],
            "fps": round(1000 / ms, 2) if ms else None,
            # 每路摄像头按 1 fps 抽帧，单进程能带几路（留 30% 余量给调度与后处理）
            "cameras_1fps": int(1000 / ms * 0.7) if ms else None,
            "wall_s": round(wall, 1),
        }
        rows.append(row)
        print(f"  mAP {row['mAP50']}  helmet {row['helmet_ap']}  "
              f"FP {row['fp_rate']}%  FN {row['fn_rate']}%  "
              f"{ms} ms  ≈{row['cameras_1fps']} 路/进程")

    if not rows:
        print("没有可跑的配置")
        return 1

    # 按「精度够用 + 吞吐最高」排序：先卡 mAP≥0.70，再挑最快
    ok = [r for r in rows if (r["mAP50"] or 0) >= 0.70]
    best = min(ok or rows, key=lambda r: r["ms"])

    payload = {
        "kind": "sweep",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Chapian/PPE_detection · test (100 张)",
        "person_model": PERSON.name,
        "rows": rows,
        "recommended": best["label"],
        "note": ("cameras_1fps = 1000/单帧耗时 × 0.7（30% 余量给调度与后处理）。"
                 "按每路 1 fps 抽帧估算，实际按业务要求的抽帧率折算。"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'配置':<9}{'mAP':>7}{'helmet':>8}{'误报':>8}{'漏报':>8}{'耗时':>9}{'路/进程':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:<9}{r['mAP50']:>7.3f}{(r['helmet_ap'] or 0):>8.3f}"
              f"{r['fp_rate']:>7.1f}%{r['fn_rate']:>7.1f}%{r['ms']:>7.1f}ms"
              f"{r['cameras_1fps']:>8}")
    print("=" * 78)
    print(f"推荐：{best['label']}（在 mAP≥0.70 的配置里耗时最低）")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
