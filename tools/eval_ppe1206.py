# -*- coding: utf-8 -*-
"""
安巡 Agent · ppe1206 单模型评测
===============================

`evaluate_real.py` 和 `evaluate_vest.py` 都是给「三模型串联」写的：前者只出
person/helmet 的四个指标，后者硬编码 DualDetector（人体+安全帽+背心三份权重）。
单模型一次出全部属性，两者都不适用，所以这里做一次合并评测：

    helmet / person   -> 复用 evaluate_real.run_eval（同一套 GT 映射与判定口径）
    vest / novest     -> 复用 evaluate_vest 的躯干区域真值推导

两段跑在同一个检测器上，**共用一次推理**（YoloDetector 自带按路径缓存），
所以 helmet 和 vest 的数字来自同一批预测框，不存在「两套推理各说各话」。

    python tools/eval_ppe1206.py --weights runs/ppe1206_n/weights/best.pt \
        --data datasets/ppe1206/test --out data/metrics_ppe1206.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))          # anxun 包在 src/ 下，不在仓库根

from anxun import rules_engine as R                          # noqa: E402
from anxun.detectors import DualDetector, YoloDetector, pick_device  # noqa: E402
from anxun.evaluate_real import (iou_box, list_images, load_gt,  # noqa: E402
                                 read_names, run_eval, warmup)
from anxun.evaluate_vest import torso_region, truth_vest      # noqa: E402
from anxun.metrics import evaluate                            # noqa: E402
from PIL import Image                                         # noqa: E402


def eval_vest(data: Path, det) -> dict:
    """vest 的业务级指标：上身区域有 vest 框=穿了，只有 no-vest=没穿，都没有=不进统计。"""
    names = read_names(data)
    files = list_images(data / "images")
    samples = []
    n_c = n_v = 0

    for fp in files:
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(data / "labels" / f"{fp.stem}.txt", names, W, H)
        res = det.detect_image(fp)
        preds = res["boxes"]

        gtb = [g for g in gt if g["label"] in ("vest", "novest")]
        prb = [b.to_dict() for b in preds if b.label in ("vest", "novest")]

        verdicts = R.judge_frame(preds, ["vest"], area_min=R.min_person_area(W, H))
        pp = [b for b in preds if b.label == "person"]
        gp = [g for g in gt if g["label"] == "person"]

        judged, truth, used = [], [], set()
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
            else:
                n_v += 1

        samples.append({"image_id": fp.stem, "gt_boxes": gtb, "pred_boxes": prb,
                        "truth": truth, "judged": judged,
                        "latency_ms": res["infer_ms"]})

    r = evaluate(samples, classes=("vest", "novest"))
    return {"mAP50": r["mAP50"],
            "per_class": {k: v["ap"] for k, v in r["per_class"].items()},
            "fp_rate": r["fp_rate"], "fn_rate": r["fn_rate"],
            "n_compliant": n_c, "n_violating": n_v,
            "confusion": r["confusion"],
            "boxes": sum(len(s["pred_boxes"]) for s in samples)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/ppe1206_n/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/ppe1206/test"))
    ap.add_argument("--out", default=str(ROOT / "data/metrics_ppe1206.json"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="结果里标注这是哪套数据，便于跨域对比")
    ap.add_argument("--person-weights", default="",
                    help="给了就走「三模型串联」：人体 + 安全帽 +（可选）背心，"
                         "用来和新单模型在同一批图上做同口径对比")
    ap.add_argument("--vest-weights", default="", help="配合 --person-weights 使用")
    ap.add_argument("--profile", default="",
                    help="判定阈值口径（measured / conservative / v3）；留空用 config 默认")
    args = ap.parse_args()

    if args.profile:
        R.apply_profile(args.profile)

    w = Path(args.weights)
    if not w.exists():
        print(f"找不到权重：{w}")
        return 2

    dev = pick_device(args.device)
    if args.person_weights:
        pw = Path(args.person_weights)
        vw = args.vest_weights or ""
        det = DualDetector(str(pw), str(w), vest_weights=vw,
                           imgsz=args.imgsz, conf=args.conf, device=dev)
        arch = f"三模型串联（{pw.name} + {w.name}" + (f" + {Path(vw).name}" if vw else "") + "）"
    else:
        det = YoloDetector(str(w), imgsz=args.imgsz, conf=args.conf, device=dev)
        arch = f"单模型（{w.name}）"
    data = Path(args.data)
    print(f"架构：{arch}")
    print(f"数据：{data}")
    warmup(det)

    out = run_eval(data, det, limit=args.limit or None, verbose=True, scope="matched")
    h = out["results"]
    v = eval_vest(data, det)

    payload = {
        "kind": "model-measured",
        "config": {"weights": w.name, "imgsz": args.imgsz, "conf": args.conf,
                   "device": f"GPU（torch）" if str(dev).startswith("cuda") else "CPU",
                   "rules_profile": R.active_profile(), "metric_scope": "matched",
                   "arch": arch, "single_model": not bool(args.person_weights)},
        "dataset": {"name": args.tag or data.name,
                    "images": out["images"],
                    "persons_gt": out["persons_gt"],
                    "persons_pred": out["persons_pred"],
                    "eval_note": "测试集未参与训练与选轮次"},
        "helmet": {
            "mAP50": h["mAP50"],
            "per_class": {k: v_["ap"] for k, v_ in h["per_class"].items()},
            "fp_rate": h["fp_rate"], "fn_rate": h["fn_rate"],
            "n_compliant": h["n_compliant"], "n_violating": h["n_violating"],
            "confusion": h["confusion"], "coverage": h["coverage"],
            "scope_detail": h["scope_detail"],
            "latency": h["latency"],
        },
        "vest": v,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print("=" * 64)
    print(f"helmet  mAP@0.5 {h['mAP50']}   逐类 {payload['helmet']['per_class']}")
    print(f"        误报 {h['fp_rate']}%（合规 {h['n_compliant']} 人次）  "
          f"漏报 {h['fn_rate']}%（违规 {h['n_violating']} 人次）")
    print(f"        覆盖率 {h['coverage']['rate']}%  "
          f"单帧 {h['latency']['mean_ms'] if isinstance(h['latency'], dict) else h['latency']} ms")
    print(f"vest    mAP@0.5 {v['mAP50']}   逐类 {v['per_class']}")
    print(f"        误报 {v['fp_rate']}%（合规 {v['n_compliant']} 人次）  "
          f"漏报 {v['fn_rate']}%（违规 {v['n_violating']} 人次）")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
