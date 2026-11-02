# -*- coding: utf-8 -*-
"""
安巡 Agent · 判定几何阈值扫描（ppe 系列模型）
=============================================

为什么单独写一个：`sweep_geom.py` 是给老的三模型链路写的，扫的是
`iou_min / contain_min` 两个维度，且权重写死 `models/best.pt`。
ppe1206 这一轮暴露的问题不在那两个维度——新模型的头盔框比老模型更贴框，
但落在「person 框上部 30% × 横向 44%」的头部区域之外，`contain_min=0.7`
直接判不达标，于是「戴了帽子」被判成「没戴」。

所以要扫的是**区域定义**本身：

    head_region_ratio   纵向取 person 框上部多少（默认 0.30）
    head_width_ratio    横向取 person 框中心多少宽（默认 0.44）
    iou_min             装备框与区域的最小 IoU（默认 0.28）
    contain_min         装备框落在区域内的最小占比（默认 0.70）

    python tools/sweep_ppe_geom.py --weights runs/ppe_mix_n/weights/best.pt \
        --data datasets/ppe_test/test --stage region

输出表格按「漏报优先」排序：安全场景下放过违规比冤枉合规严重。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun import rules_engine as R                            # noqa: E402
from anxun.detectors import YoloDetector, pick_device          # noqa: E402
from anxun.evaluate_real import run_eval, warmup               # noqa: E402


def measure(det, data: Path) -> dict:
    r = run_eval(data, det, verbose=False, scope="matched")["results"]
    m = r["scope_detail"]["matched"]
    e = r["scope_detail"]["e2e"]
    return {"fp": m["fp_rate"], "fn": m["fn_rate"],
            "efp": e["fp_rate"], "efn": e["fn_rate"],
            "tp": m["confusion"]["tp"], "fn_n": m["confusion"]["fn"],
            "fp_n": m["confusion"]["fp"], "cov": r["coverage"]["rate"],
            "map": r["mAP50"], "helmet": r["per_class"]["helmet"]["ap"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/ppe_mix_n/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/ppe_test/test"))
    ap.add_argument("--stage", default="region",
                    choices=("region", "geom", "conf", "fine"))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    det = YoloDetector(args.weights, device=pick_device(args.device))
    warmup(det)
    data = Path(args.data)

    base = {"HEAD_REGION_RATIO": R.HEAD_REGION_RATIO, "HEAD_WIDTH_RATIO": R.HEAD_WIDTH_RATIO,
            "HELMET_IOU_MIN": R.HELMET_IOU_MIN, "CONTAIN_MIN": R.CONTAIN_MIN,
            "CONF_FINAL": R.CONF_FINAL}
    current = dict(base)

    def run(**kw) -> dict:
        current.update(kw)
        for k, v in current.items():
            setattr(R, k, v)
        return measure(det, data)

    b = run()
    print(f"基线（当前 measured 口径）：误报 {b['fp']:.2f}% 漏报 {b['fn']:.2f}% "
          f"tp={b['tp']} fn={b['fn_n']} fp={b['fp_n']} 覆盖 {b['cov']:.2f}% "
          f"helmetAP {b['helmet']:.4f}\n")

    rows = []
    if args.stage == "region":
        # 第一步：只动区域定义，看方向。iou/contain 保持 measured 值。
        for rr, wr in itertools.product((0.30, 0.35, 0.40, 0.45, 0.55),
                                        (0.44, 0.55, 0.65, 0.80)):
            r = run(HEAD_REGION_RATIO=rr, HEAD_WIDTH_RATIO=wr)
            rows.append((rr, wr, current["HELMET_IOU_MIN"], current["CONTAIN_MIN"], r))

    elif args.stage == "geom":
        for iou, con in itertools.product((0.28, 0.20, 0.14, 0.10),
                                          (0.70, 0.60, 0.50, 0.40)):
            r = run(HELMET_IOU_MIN=iou, CONTAIN_MIN=con)
            rows.append((current["HEAD_REGION_RATIO"], current["HEAD_WIDTH_RATIO"], iou, con, r))

    elif args.stage == "conf":
        for cf in (0.30, 0.40, 0.50, 0.60, 0.70):
            r = run(CONF_FINAL=cf)
            rows.append((current["HEAD_REGION_RATIO"], current["HEAD_WIDTH_RATIO"],
                         current["HELMET_IOU_MIN"], current["CONTAIN_MIN"], r))

    else:  # fine：在已选定的区域参数附近做 iou×contain 二维细扫
        for iou, con in itertools.product((0.28, 0.24, 0.20, 0.16),
                                          (0.70, 0.60, 0.50)):
            r = run(HELMET_IOU_MIN=iou, CONTAIN_MIN=con)
            rows.append((current["HEAD_REGION_RATIO"], current["HEAD_WIDTH_RATIO"], iou, con, r))

    rows.sort(key=lambda t: (t[4]["fn"], t[4]["fp"]))
    print(f"{'region':>7}{'width':>7}{'iou':>7}{'contain':>9}"
          f"{'误报%':>8}{'漏报%':>8}{'tp':>4}{'fn':>4}{'fp':>5}{'覆盖%':>8}{'helmetAP':>10}")
    for a, c, d, e, r in rows:
        print(f"{a:>7.2f}{c:>7.2f}{d:>7.2f}{e:>9.2f}"
              f"{r['fp']:>8.2f}{r['fn']:>8.2f}{r['tp']:>4}{r['fn_n']:>4}{r['fp_n']:>5}"
              f"{r['cov']:>8.2f}{r['helmet']:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
