# -*- coding: utf-8 -*-
"""
误报逐例归因：合规的人为什么被判「未佩戴」
==========================================

v3.0.0 的误报率 10.20%（98 个合规人次里 10 例）是唯一破线的指标。
但「10 例」背后至少是四种完全不同的原因，对应四种完全不同的解法——
不先分清就调参，等于闭眼改阈值。

    B1  模型没检出          头部区域周边压根没有 helmet 框
    B2  检出但置信度不够    有框，中心点在区域内，conf < CONF_FINAL
    B3  框在但中心点出区    有 conf 达标的框，中心点落在头部区域外
    B4  面积闸门挡掉        框面积 > max_ratio × 区域面积
    B5  判定逻辑异常        有完全达标的框却没判「已佩戴」（不该出现，出现就是 bug）

输出：
    data/fp_attribution.json     逐例明细 + 分类汇总
    runs/_review_fp/*.jpg        每例裁图（person 外扩 35%，画 GT/预测/区域）

用法：
    python tools/attr_fp.py                    # 全量 100 张
    python tools/attr_fp.py --limit 20         # 快速冒烟
    python tools/attr_fp.py --imgsz 960        # 换输入尺寸再跑一遍做对照
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anxun.rules_engine as R
from anxun.detectors import DualDetector, pick_device
from anxun.evaluate_real import (TRUTH_HEAD_REGION, TRUTH_HEAD_WIDTH, iou_box,
                                 list_images, load_gt, read_names, truth_of)
from anxun.rules_engine import Box, judge_frame, min_person_area

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT_JSON = ROOT / "data" / "fp_attribution.json"
OUT_FIG = ROOT / "runs" / "_review_fp"


def classify(person: Box, region: Box, helmets: list[Box], cfg: dict,
             conf_final: float, person_conf: float) -> tuple[str, dict]:
    """给一例误报定因。

    只按「几何 + 置信度」分，不掺语义——语义判断留给人工看图那一步。
    """
    max_ratio = cfg["max_ratio"]
    info: dict = {"n_helmet_boxes": len(helmets), "best_conf_in_region": None,
                  "best_conf_out_region": None, "max_area_ratio": None}

    in_region, out_region, blocked = [], [], []
    for b in helmets:
        ratio = b.area / region.area if region.area > 0 else 0.0
        info["max_area_ratio"] = max(info["max_area_ratio"] or 0.0, round(ratio, 2))
        if ratio > max_ratio:
            blocked.append(b)
            continue

        cx_in = region.x1 <= b.cx <= region.x2 and region.y1 <= b.cy <= region.y2
        if cx_in:
            in_region.append(b)
            if info["best_conf_in_region"] is None or b.conf > info["best_conf_in_region"]:
                info["best_conf_in_region"] = round(b.conf, 3)
        else:
            out_region.append(b)
            if info["best_conf_out_region"] is None or b.conf > info["best_conf_out_region"]:
                info["best_conf_out_region"] = round(b.conf, 3)
            # 偏移量：中心点相对区域的归一化距离，>1 表示差了一整个区域宽度
            dx = 0.0 if region.x1 <= b.cx <= region.x2 else min(
                abs(b.cx - region.x1), abs(b.cx - region.x2)) / max(region.w, 1e-6)
            dy = 0.0 if region.y1 <= b.cy <= region.y2 else min(
                abs(b.cy - region.y1), abs(b.cy - region.y2)) / max(region.h, 1e-6)
            info.setdefault("offsets", []).append(
                {"conf": round(b.conf, 3), "dx": round(dx, 2), "dy": round(dy, 2)})

    # B5：完全达标却没判已佩戴 —— 出现就是判定链路的 bug，不是模型问题
    solid = [b for b in in_region if b.conf >= conf_final]
    if solid:
        info["solid_conf"] = round(max(b.conf for b in solid), 3)
        return "B5", info

    if in_region:
        return "B2", info          # 中心点在区域内，但没一个够 CONF_FINAL
    if out_region:
        return "B3", info          # 有够置信的框，位置偏了
    if blocked:
        info["blocked_conf"] = round(max(b.conf for b in blocked), 3)
        return "B4", info
    return "B1", info              # 一个 helmet 框都没有


def draw_case(img_path: Path, person: Box, region: Box,
              gt_helmets: list[Box], pred_helmets: list[Box],
              out_path: Path, tag: str) -> bool:
    """裁 person 外扩 35%，画 GT 绿 / 预测红 / 头部区域黄 / person 白。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        pad_x, pad_y = person.w * 0.35, person.h * 0.15
        x1 = max(0, int(person.x1 - pad_x))
        y1 = max(0, int(person.y1 - pad_y))
        x2 = min(W, int(person.x2 + pad_x))
        y2 = min(H, int(person.y2 + pad_y))
        crop = im.crop((x1, y1, x2, y2))
        d = ImageDraw.Draw(crop)

        def rect(b: Box, color: tuple, width: int = 3) -> None:
            d.rectangle([b.x1 - x1, b.y1 - y1, b.x2 - x1, b.y2 - y1],
                        outline=color, width=width)

        for g in gt_helmets:
            rect(g, (0, 210, 90), 3)          # 绿：标注头盔
        for b in pred_helmets:
            c = (255, 70, 70) if b.conf >= R.CONF_FINAL else (255, 165, 0)
            rect(b, c, 2)                     # 红：达标预测 / 橙：低置信
            d.text((b.x1 - x1 + 2, b.y1 - y1 + 2), f"{b.conf:.2f}",
                   fill=(255, 255, 255))
        rect(region, (255, 210, 0), 2)        # 黄：头部判定区域
        rect(person, (255, 255, 255), 1)      # 白：人员框

        scale = 460 / max(crop.width, 1)
        crop = crop.resize((460, max(1, int(crop.height * scale))))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=92)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", default="", help="阈值口径，留空用 config 默认（v3）")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if args.profile:
        R.apply_profile(args.profile)
    print(f"口径 {R.active_profile()}  conf_final={R.CONF_FINAL}  "
          f"region={R.HEAD_REGION_RATIO}/{R.HEAD_WIDTH_RATIO}")

    det = DualDetector(args.person_weights, args.weights,
                       imgsz=args.imgsz, conf=args.conf, device=pick_device(""))
    data = Path(args.data)
    img_dir = data / "images" if (data / "images").exists() else data
    lbl_dir = data / "labels" if (data / "labels").exists() else data
    names = read_names(data)
    if not names:
        raise SystemExit("找不到 data.yaml 的 names")

    files = list_images(img_dir, args.limit or None)
    cfg = R.equip_cfg("helmet")
    cases: list[dict] = []
    cause_counter: Counter = Counter()

    from PIL import Image

    for i, fp in enumerate(files, 1):
        with Image.open(fp) as im:
            W, H = im.size
        gt = load_gt(lbl_dir / (fp.stem + ".txt"), names, W, H)
        res = det.detect_image(fp)
        preds = res["boxes"]
        verdicts = judge_frame(preds, ["helmet"], area_min=min_person_area(W, H))

        gt_persons = [g for g in gt if g["label"] == "person"]
        pred_persons = [b for b in preds if b.label == "person"]
        gt_helmets_all = [g for g in gt if g["label"] == "helmet"]

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

            worn = truth_of(gt, gt_persons[best])
            if worn is not True:
                continue                       # 只看合规的人
            vs = [v for v in verdicts if v.person_idx == pi and v.equip == "helmet"]
            if not vs or vs[0].status != "未佩戴":
                continue                       # 只看真的被判违规的

            region = pbox.top_region(R.HEAD_REGION_RATIO, R.HEAD_WIDTH_RATIO)
            helmets = [b for b in preds if b.label == "helmet"]
            cause, info = classify(pbox, region, helmets, cfg, R.CONF_FINAL, pbox.conf)

            # 标注头盔框：只留中心点落在真值区域内的（与 truth_of 同口径）
            t_person = gt_persons[best]
            tp = Box("person", 1.0, t_person["x1"], t_person["y1"],
                     t_person["x2"], t_person["y2"])
            t_head = tp.top_region(TRUTH_HEAD_REGION, TRUTH_HEAD_WIDTH)
            gt_h = [Box("helmet", 1.0, g["x1"], g["y1"], g["x2"], g["y2"])
                    for g in gt_helmets_all
                    if t_head.x1 <= (g["x1"] + g["x2"]) / 2 <= t_head.x2
                    and t_head.y1 <= (g["y1"] + g["y2"]) / 2 <= t_head.y2]

            case = {
                "image_id": fp.stem,
                "image": str(fp.relative_to(ROOT)).replace("\\", "/"),
                "person": {k: round(getattr(pbox, k), 1)
                           for k in ("x1", "y1", "x2", "y2", "w", "h")},
                "person_conf": round(pbox.conf, 3),
                "verdict": {"status": vs[0].status, "conf": round(vs[0].conf, 3),
                            "reason": vs[0].reason},
                "gt_helmet_boxes": [{"x1": round(g.x1, 1), "y1": round(g.y1, 1),
                                     "x2": round(g.x2, 1), "y2": round(g.y2, 1)}
                                    for g in gt_h],
                "cause": cause,
                **info,
            }
            if not args.no_fig:
                fig = OUT_FIG / f"fp_{len(cases):02d}_{fp.stem}.jpg"
                case["fig"] = str(fig.relative_to(ROOT)).replace("\\", "/")
                draw_case(fp, pbox, region, gt_h,
                          [b for b in helmets
                           if b.x2 > pbox.x1 and b.x1 < pbox.x2
                           and b.y2 > pbox.y1 and b.y1 < pbox.y2],
                          fig, cause)
            cases.append(case)
            cause_counter[cause] += 1
        if i % 25 == 0:
            print(f"  {i}/{len(files)} ... 累计误报 {len(cases)}")

    out = {
        "kind": "fp-attribution",
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "config": {"imgsz": args.imgsz, "conf": args.conf, "profile": R.active_profile(),
                   "conf_final": R.CONF_FINAL, "region": [R.HEAD_REGION_RATIO,
                                                          R.HEAD_WIDTH_RATIO]},
        "summary": {"n_images": len(files), "n_fp": len(cases),
                    "by_cause": dict(sorted(cause_counter.items()))},
        "legend": {
            "B1": "模型没检出：头部区域周边没有 helmet 框",
            "B2": "检出但置信度不够：框在区域内，conf < conf_final",
            "B3": "框够置信但中心点出区：几何/区域估计问题",
            "B4": "面积闸门挡掉：框面积 > max_ratio × 区域面积",
            "B5": "判定逻辑异常：有完全达标的框却没判已佩戴（bug）",
        },
        "cases": cases,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f"误报 {len(cases)} 例 / {len(files)} 张")
    for k in sorted(cause_counter):
        print(f"  {k}  {cause_counter[k]:>2} 例  {out['legend'][k]}")
    print(f"-> {OUT_JSON}")
    if not args.no_fig:
        print(f"-> {OUT_FIG}/   （绿=标注头盔 红=达标预测 橙=低置信 黄=头部区域）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
