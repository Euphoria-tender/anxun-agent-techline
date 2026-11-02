# -*- coding: utf-8 -*-
"""
安巡 Agent · 真实数据评测
=========================

四个指标的**唯一口径**，训练/换模型/调参之后都用它复核：

    mAP@0.5     检测层：框准不准
    误报率       业务层：合规的人被冤枉了多少
    漏报率       业务层：违规的人被放过了多少
    单帧耗时     工程层：能不能撑住实时巡查

    python src/anxun/evaluate_real.py --data datasets/ppe_test/test

真值从标注来（Hardhat / NO-Hardhat），判定从引擎来，两者不共享规则，
所以误报漏报衡量的是整条链路，不是自己跟自己比。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anxun.detectors import DualDetector, YoloDetector, pick_device
from anxun.metrics import evaluate
from anxun.rules_engine import Box, judge_frame, min_person_area

# 真值推导专用的头部区域，**刻意不跟随判定口径**。
# 为什么：truth_of 要回答的是「标注里这个人头上到底有没有帽子框」，这是数据事实，
# 不该随判定阈值变。原来直接用 rules_engine.HEAD_REGION_RATIO，既在 import 时被快照
# （切换 profile 不更新），又让真值与判定共享同一套几何——等于自己给自己出题。
# 钉死成 v2.2.0 measured 口径的值，历史上所有数字都不变，换口径后分母也仍然可比。
TRUTH_HEAD_REGION = 0.30
TRUTH_HEAD_WIDTH = 0.44

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "metrics_real.json"      # torch 后端的交付指标
OUT_ONNX = ROOT / "data" / "metrics_onnx.json"  # CPU / ONNX 后端的交付指标


def _default_out(backend: str) -> Path:
    """按后端决定默认落盘文件，避免两条链路互相覆盖。"""
    return OUT if backend == "torch" else OUT_ONNX

# 真值类别 → 引擎类别。键统一成「小写 + 连字符」，不同数据集的写法都能对上。
GT_MAP = {
    "person": "person", "worker": "person", "people": "person",
    "hardhat": "helmet", "helmet": "helmet", "safety-helmet": "helmet",
    "no-hardhat": "nohelmet", "no-helmet": "nohelmet", "no helmet": "nohelmet",
    "head": "nohelmet", "bare-head": "nohelmet",
    # vest 两类：helmet 评测用不到（truth_of 只认 helmet/nohelmet，map_at_50 只算
    # person/helmet），但 evaluate_vest.py 要用。加进来不影响 helmet 的任何数字。
    "vest": "vest", "safety-vest": "vest",
    "no-vest": "novest", "no_vest": "novest", "no safety vest": "novest",
}


# ============================================================
# 一、读取与映射

def read_names(data_dir: Path) -> list[str] | None:
    for name in ("data.yaml", "dataset.yaml"):
        p = data_dir / name
        if not p.exists():
            p = data_dir.parent / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            try:
                import yaml
                d = yaml.safe_load(text)
                if isinstance(d.get("names"), list):
                    return [str(x) for x in d["names"]]
                if isinstance(d.get("names"), dict):
                    return [str(d["names"][k]) for k in sorted(d["names"], key=int)]
            except Exception:
                pass
            import re
            m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
            if m:
                return [x.strip().strip("'\"") for x in m.group(1).split(",")]
    return None


def load_gt(label_path: Path, names: list[str], w: int, h: int) -> list[dict]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        raw = names[cls] if cls < len(names) else str(cls)
        norm = raw.strip().lower().replace("_", "-").replace(" ", "-")
        label = GT_MAP.get(norm) or GT_MAP.get(raw.strip().lower())
        if label is None:
            continue
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        out.append({"label": label,
                    "x1": (cx - bw / 2) * w, "y1": (cy - bh / 2) * h,
                    "x2": (cx + bw / 2) * w, "y2": (cy + bh / 2) * h})
    return out


def truth_of(gt: list[dict], person: dict) -> bool | None:
    """
    按标注判断这个人戴没戴：只看落在头部区域里的装备框。
    有 hardhat 算戴，只有 no-hardhat 算没戴，两者都没有返回 None（不进统计）。
    """
    p = Box("person", 1.0, person["x1"], person["y1"], person["x2"], person["y2"])
    head = p.top_region(TRUTH_HEAD_REGION, TRUTH_HEAD_WIDTH)
    worn, seen = False, False
    for g in gt:
        if g["label"] not in ("helmet", "nohelmet"):
            continue
        cx = (g["x1"] + g["x2"]) / 2
        cy = (g["y1"] + g["y2"]) / 2
        if head.x1 <= cx <= head.x2 and head.y1 <= cy <= head.y2:
            seen = True
            if g["label"] == "helmet":
                worn = True
    return worn if seen else None


def iou_box(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ub = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (ua + ub - inter)


# ============================================================
# 二、跑一遍评测（sweep.py 也调这个）

def list_images(img_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if limit and len(files) > limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    return files


def warmup(det) -> None:
    """第一次推理要 1~2 秒（内存分配），不能算进 p95。"""
    from PIL import Image
    w = Path(tempfile.gettempdir()) / "anxun_warmup.jpg"
    if not w.exists():
        Image.new("RGB", (640, 640), (30, 30, 30)).save(w)
    det.detect_image(w)


def run_eval(data_dir: str | Path, det, limit: int | None = None,
             verbose: bool = False, scope: str = "matched") -> dict:
    """在指定数据集上评测一个检测器，返回 samples + 指标。

    scope="matched"：误报/漏报只统计「被检出的人」，衡量判定层；
    scope="e2e"    ：连没检出的人一起算，衡量整条链路。
    JSON 里两个数都有，headline 用哪个由 --metric-scope 决定。
    """
    from PIL import Image
    data = Path(data_dir)
    img_dir = data / "images" if (data / "images").exists() else data
    lbl_dir = data / "labels" if (data / "labels").exists() else data
    names = read_names(data)
    if not names:
        raise ValueError("找不到 data.yaml 里的 names，无法映射类别")

    files = list_images(img_dir, limit)
    samples = []
    n_person_gt = n_person_pred = 0

    for i, fp in enumerate(files, 1):
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(lbl_dir / (fp.stem + ".txt"), names, W, H)
        res = det.detect_image(fp)
        preds = res["boxes"]
        verdicts = judge_frame(preds, ["helmet"], area_min=min_person_area(W, H))

        pred_persons = [b for b in preds if b.label == "person"]
        gt_persons = [g for g in gt if g["label"] == "person"]
        n_person_gt += len(gt_persons)
        n_person_pred += len(pred_persons)

        judged, matched_gi = [], set()
        used = set()
        for pi, pbox in enumerate(pred_persons):
            best, bv = None, 0.5
            for gi, g in enumerate(gt_persons):
                if gi in used:
                    continue
                v = iou_box({"x1": pbox.x1, "y1": pbox.y1, "x2": pbox.x2, "y2": pbox.y2}, g)
                if v > bv:
                    best, bv = gi, v
            if best is None:
                continue
            used.add(best)
            matched_gi.add(best)
            worn = truth_of(gt, gt_persons[best])
            if worn is None:
                continue
            vs = [v for v in verdicts if v.person_idx == pi and v.equip == "helmet"]
            if not vs:
                continue
            judged.append({"person_id": f"G{best}", "equip": "helmet",
                           "status": vs[0].status})

        # 真值里「能判」的人一律记录，不管有没有被检出（matched 标记）。
        # 只记已检出的人，等于把检测层面漏掉的人从漏报率的分母里悄悄删掉：
        # 数字会好看，但「系统一共漏了多少违规」这个问题的答案就没了。
        # 两套口径都能从这一份数据算出来，见 metrics.business_rates(scope=...)。
        truth = []
        for gi, g in enumerate(gt_persons):
            worn = truth_of(gt, g)
            if worn is None:
                continue
            truth.append({"person_id": f"G{gi}", "equip": "helmet", "worn": worn,
                          "matched": gi in matched_gi})

        samples.append({
            "image_id": fp.stem,
            "gt_boxes": [{"label": g["label"], **{k: g[k] for k in ("x1", "y1", "x2", "y2")}}
                         for g in gt if g["label"] in ("person", "helmet")],
            "pred_boxes": [b.to_dict() for b in preds if b.label in ("person", "helmet")],
            "truth": truth,
            "judged": judged,
            "latency_ms": res["infer_ms"],
        })
        if verbose and i % 50 == 0:
            print(f"  {i}/{len(files)} ...")

    if not samples:
        raise ValueError("没有可用样本")

    return {
        "images": len(files),
        "persons_gt": n_person_gt,
        "persons_pred": n_person_pred,
        "results": evaluate(samples, classes=("person", "helmet"), scope=scope),
        # 逐例明细留给误差预算用（tools/error_budget.py）；不写进交付 JSON。
        "samples": samples,
    }


# ============================================================
# 三、命令行

def _device_note(det) -> str:
    import torch
    dev = pick_device(getattr(getattr(det, "person", det), "device", "") or "")
    tag = "GPU" if str(dev).startswith("cuda") else "CPU"
    name = torch.cuda.get_device_name(0) if tag == "GPU" else ""
    return f"{tag} {dev} {name}".strip() + f"（torch {torch.__version__}）"


def _rules_profile() -> str:
    """这组指标是哪套阈值跑出来的。没有它，两套口径的数字会被混着引用。"""
    import anxun.rules_engine as R
    return R.active_profile()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "ppe_test" / "test"))
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="")
    ap.add_argument("--backend", default="torch", choices=("torch", "onnx"),
                    help="onnx = 不依赖 torch 的 CPU 后端（可移植版）")
    ap.add_argument("--out", default="",
                    help="结果 json 路径；留空按后端落盘（torch → data/metrics_real.json，"
                         "onnx → data/metrics_onnx.json）。写另一个后端的交付文件需加 --force")
    ap.add_argument("--vest", default="", help="额外加载 vest 权重，用来验证「加模型不影响 helmet 判定」")
    ap.add_argument("--metric-scope", default="matched", choices=("matched", "e2e"),
                    help="误报/漏报的口径：matched=只算已检出的人（默认，衡量判定层）；"
                         "e2e=连没检出的人一起算（衡量整条链路）。"
                         "两种结果都会写进 JSON，这里只决定 headline 用哪个")
    ap.add_argument("--profile", default="", choices=("", "measured", "conservative"),
                    help="判定阈值口径；留空用 config 默认（measured）。"
                         "报数只能用 measured，换口径的结果不能和交付数字比")
    ap.add_argument("--force", action="store_true",
                    help="允许把非 torch 后端的结果写进 metrics_real.json（默认拒绝）")
    args = ap.parse_args()

    # 落盘前先把「会不会覆盖别人的交付指标」判掉，别等跑完整套推理才拒绝。
    out_path = Path(args.out) if args.out else _default_out(args.backend)
    if args.backend != "torch" and out_path.resolve() == OUT.resolve() and not args.force:
        # 原来 --backend onnx 默认也写 metrics_real.json，等于用 CPU 的结果
        # 覆盖 GPU 交付指标，而且没有任何提示。踩过一次，加个闸。
        print(f"拒绝写入：{out_path} 是 torch 后端的交付指标，"
              f"--backend {args.backend} 的结果请写 "
              f"{_default_out(args.backend).name}（确实要覆盖就加 --force）")
        return 2

    if args.profile:
        import anxun.rules_engine as R
        R.apply_profile(args.profile)

    if args.backend == "onnx":
        from anxun.onnx_backend import build_onnx_detector
        onnx_dir = ROOT / "models" / "onnx"
        det, model_name = build_onnx_detector(
            conf=args.conf, imgsz=args.imgsz,
            ppe=str(onnx_dir / "best.onnx"), person=str(onnx_dir / "yolov8n.onnx"))
    else:
        dev = pick_device(args.device)
        vest = str(args.vest) if args.vest else ""
        if vest and not Path(vest).exists():
            print(f"找不到 vest 权重：{vest}")
            return 2
        if Path(args.person_weights).exists():
            det = DualDetector(args.person_weights, args.weights, vest_weights=vest,
                               imgsz=args.imgsz, conf=args.conf, device=dev)
            model_name = f"{Path(args.person_weights).name} + {Path(args.weights).name}"
            if vest:
                model_name += f" + {Path(vest).name}"
        else:
            det = YoloDetector(args.weights, imgsz=args.imgsz, conf=args.conf, device=dev)
            model_name = Path(args.weights).name
    print(f"模型：{model_name}  imgsz={args.imgsz}  conf={args.conf}")
    warmup(det)

    out = run_eval(args.data, det, limit=args.limit or None, verbose=True,
                   scope=args.metric_scope)
    res = out["results"]
    payload = {
        "kind": "model-measured",
        "config": {"weights": model_name, "imgsz": args.imgsz, "conf": args.conf,
                   "backend": args.backend,
                   "rules_profile": _rules_profile(),
                   "metric_scope": args.metric_scope,
                   "device": "CPU（onnxruntime，不依赖 torch）" if args.backend == "onnx"
                             else _device_note(det)},
        "dataset": {
            "name": "Chapian/PPE_detection · test",
            "source": "https://hf-mirror.com/datasets/Chapian/PPE_detection",
            "license": "CC BY 4.0",
            "images": out["images"],
            "persons_gt": out["persons_gt"],
            "persons_pred": out["persons_pred"],
            "eval_note": ("真值佩戴状态由标注框推导：Hardhat 落在头部区域=合规，"
                          "只检出 NO-Hardhat=违规；两者都无则跳过。判定由引擎给出。"),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": res,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 62)
    print(f"mAP@0.5   {res['mAP50']}   逐类 "
          f"{ {k: v['ap'] for k, v in res['per_class'].items()} }")
    print(f"误报率     {res['fp_rate']}%   (合规 {res['n_compliant']} 人次)")
    print(f"漏报率     {res['fn_rate']}%   (违规 {res['n_violating']} 人次)")
    print(f"单帧耗时   {res['latency']['mean_ms']} ms (p95 {res['latency']['p95_ms']})")
    print(f"混淆矩阵   {res['confusion']}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
