# -*- coding: utf-8 -*-
"""生成「同一批人真实速度移动」的评测素材：整幅照片在画布上平移。

为什么不用「贴人像」
--------------------
第一版把 person 框从原图裁下来贴到空车间背景上，两个问题：
  1. 裁切带原背景块，边缘肉眼可见，检测框每帧抖动 → 轨迹暴涨（真值 6 人切出 101 条轨）；
  2. 两人交叉段必然重叠，检测混乱。
根因是"贴图"破坏了照片的光照与透视一致性，检测器不在训练分布内。

本脚本改成整图平移：**一张完整照片**在更大的画布上匀速滑动。
画面内容原封不动，检测器看到的就是它训练时见过的真实照片，框稳定；
而人的绝对位置随画面移动，位移完全可控——这正是要测的变量。

能测什么 / 测不了什么
--------------------
能测：给定抽帧率下，真实移动速度会不会导致断轨（这是判定覆盖率的唯一决定因素）
      · 帧差在不同运动速度下的分布
      · 「画面内所有人一起移动」时的判定链路完整性
测不了：交叉场景的 ID switch（整图平移不会有两人相向而行的情形）
      · 真实域差（人像仍来自公开数据集）
定位：**评测专用素材，不进 PPT/视频**

用法
----
    python tools/make_motion_video.py                       # 出 datasets/site_pan.mp4
    python tools/make_motion_video.py --pan-px 320 --pan-s 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL_IMG = ROOT / "datasets" / "ppe_test" / "test" / "images"
POOL_LBL = ROOT / "datasets" / "ppe_test" / "test" / "labels"

CLS = {"boots": 0, "gloves": 1, "hardhat": 2, "no_boots": 3, "no_gloves": 4,
       "no_hardhat": 5, "no_vest": 6, "person": 7, "vest": 8}

SCENE_W, SCENE_H = 1280, 720          # 单幅照片铺满的高度


# ---------------------------------------------------------------- 读标注

def read_yolo(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int, int]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, int((cx - bw / 2) * img_w), int((cy - bh / 2) * img_h),
                    int((cx + bw / 2) * img_w), int((cy + bh / 2) * img_h)))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pick_scenes(min_persons: int = 1, max_persons: int = 4) -> list[dict]:
    """挑几张"人不多、框够大、佩戴状态可判"的照片，每张当一个场景。"""
    out = []
    for img_path in sorted(POOL_IMG.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_yolo(POOL_LBL / (img_path.stem + ".txt"), w, h)
        persons = [b for b in boxes if b[0] == CLS["person"]]
        hats = [b for b in boxes if b[0] in (CLS["hardhat"], CLS["no_hardhat"])]
        vests = [b for b in boxes if b[0] in (CLS["vest"], CLS["no_vest"])]
        if not (min_persons <= len(persons) <= max_persons):
            continue
        recs = []
        ok = True
        for pb in persons:
            x1, y1, x2, y2 = pb[1:5]
            bw, bh = x2 - x1, y2 - y1
            if bh < 120:                                  # 太远的人缩放到 720 后只剩几十像素
                ok = False
                break

            def nearest(cands):
                best, bid = 0.0, None
                for cb in cands:
                    v = iou(pb[1:5], cb[1:5])
                    if v > best:
                        best, bid = v, cb[0]
                return (bid, best) if best >= 0.05 else (None, best)

            hat_id, _ = nearest(hats)
            if hat_id is None:
                ok = False
                break                                     # 帽子状态不可知，真值写不出来
            vest_id, _ = nearest(vests)
            recs.append({
                "box": [round(v / d, 4) for v, d in zip(pb[1:5], (w, h, w, h))],
                "equip": "已佩戴" if hat_id == CLS["hardhat"] else "未佩戴",
                # 背心状态：None = 没标注。数据集 vest 只有 51 个框 / 295 个人，
                # 绝大多数人没背心，所以「合规」的人必须两样都有。
                "vest": None if vest_id is None else (vest_id == CLS["vest"]),
            })
        if ok and recs:
            # 场景分级：全合规（每人帽+衣都有）/ 部分合规 / 无合规
            n = len(recs)
            good = sum(1 for r in recs if r["equip"] == "已佩戴" and r["vest"] is True)
            out.append({"img": img_path.name, "persons": recs,
                        "n": n, "good": good,
                        "tier": "all_ok" if good == n else ("some_ok" if good else "none_ok")})
    return out


# ---------------------------------------------------------------- 背景

def background(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (72, 74, 76)
    for x in range(0, w, 160):
        cv2.line(img, (x, 0), (x, h), (64, 66, 68), 1)
    cv2.putText(img, "BAY-01 / CAM-02", (24, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (110, 110, 108), 2, cv2.LINE_AA)
    return img


def load_scene(rec: dict) -> np.ndarray:
    """读图并铺成 SCENE_W × SCENE_H（保持比例，居中裁切）。"""
    img = cv2.imread(str(POOL_IMG / rec["img"]))
    h, w = img.shape[:2]
    s = max(SCENE_W / w, SCENE_H / h)
    img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    h2, w2 = img.shape[:2]
    x0, y0 = (w2 - SCENE_W) // 2, (h2 - SCENE_H) // 2
    return img[y0:y0 + SCENE_H, x0:x0 + SCENE_W]


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets" / "site_pan.mp4"))
    ap.add_argument("--truth", default=str(ROOT / "data" / "site_pan_truth.json"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--pan-px", type=int, default=320, help="每段平移的总像素")
    ap.add_argument("--pan-s", type=float, default=2.0, help="平移耗时（秒）")
    ap.add_argument("--hold-s", type=float, default=2.0, help="平移后静止时长（秒）")
    ap.add_argument("--gap-s", type=float, default=0.8,
                    help="段间空画面时长（秒）。必须大于 Tracker 的 max_miss÷fps，"
                         "否则上一场景的人会被下一场景继承——实测 T3 横跨 4 个场景")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--violation-scenes", type=float, default=0.0,
                    help="含「头盔未佩戴」场景的占比（0~1）。默认 0 保持旧行为；\n                         测漏报必须调到 0.3 以上——旧采样按文件名顺序取，\n                         12 个人头盔全是合规的，只测得了误报")
    ap.add_argument("--canvas-w", type=int, default=1920)
    args = ap.parse_args()

    scenes = pick_scenes()
    # 采样规则：**按数据集真实比例，不人为拔高**
    #
    # 数据集的真实分布是：120 个够大的人里只有 14 个「帽 + 背心」都合规（11.7%），
    # 因为 vest 只有 51 个框 / 295 个人。所以合规率天然就低——这是数据集的性质，
    # 不是模型的问题。把合规场景人为堆到一半，跑出来的数好看但经不起问。
    #
    # 这里保证至少 1 个全合规场景（否则合规率恒为 0，等于采样错误），
    # 其余按文件顺序取，整体比例贴近 11.7%。
    # 分层采样：按「头盔是否违规」分开挑，而不是按文件名顺序。
    #
    # 旧做法（all_ok[:1] + others[:n-1]）按文件顺序取，采到的 12 个人头盔全是合规的，
    # 只能测误报、测不了漏报——而漏报才是安全场景真正在意的那个数。
    viol = [s for s in scenes if any(p["equip"] == "未佩戴" for p in s["persons"])]
    comp = [s for s in scenes if all(p["equip"] == "已佩戴" for p in s["persons"])]
    n_viol = max(0, round(args.scenes * args.violation_scenes))
    n_comp = args.scenes - n_viol
    if args.violation_scenes > 0 and n_viol == 0:
        n_viol, n_comp = 1, args.scenes - 1
    if len(viol) < n_viol or len(comp) < n_comp:
        raise SystemExit(f"场景不足：需违规 {n_viol}/有 {len(viol)}，"
                         f"需合规 {n_comp}/有 {len(comp)}")
    picked = viol[:n_viol] + comp[:n_comp]
    if not picked:
        raise SystemExit("没挑出任何场景")

    tot = sum(s["n"] for s in picked)
    good = sum(s["good"] for s in picked)
    print(f"   采样 {len(picked)} 场景 · {tot} 人物 · 其中帽+衣全合规 {good} "
          f"（数据集整体 11.7%）")

    W, H = args.canvas_w, SCENE_H
    seg_s = args.pan_s + args.hold_s + args.gap_s
    lead_s = 3.0
    total_s = lead_s + seg_s * len(picked) + 3.0
    frames = int(round(total_s * args.fps))

    bg = background(W, H)
    out = Path(args.out)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    # 真值：每个人物记场景、原图归一化框、装备、进出时间
    truth = {"source": out.name, "fps": args.fps, "size": [W, H], "seconds": round(total_s, 2),
             "truth_source": "generator", "canvas": "整幅照片匀速平移",
             "pan_px": args.pan_px, "pan_s": args.pan_s, "persons": []}
    for i, sc in enumerate(picked):
        t0 = lead_s + seg_s * i
        for j, p in enumerate(sc["persons"]):
            truth["persons"].append({
                "id": f"S{i + 1}P{j + 1}", "img": sc["img"], "equip": p["equip"],
                "vest": p.get("vest"),           # None = 数据集没标背心
                "fully_ok": p["equip"] == "已佩戴" and p.get("vest") is True,
                "box_norm": p["box"], "t_enter": round(t0, 2),
                "t_exit": round(t0 + seg_s, 2),
            })

    scene_cache = {rec["img"]: load_scene(rec) for rec in picked}
    rng = np.random.default_rng(7)

    for fi in range(frames):
        t = fi / args.fps
        frame = bg.copy()

        idx = int((t - lead_s) // seg_s) if t >= lead_s else -1
        if 0 <= idx < len(picked):
            sc = picked[idx]
            img = scene_cache[sc["img"]]
            p_t = t - (lead_s + seg_s * idx)
            if p_t > args.pan_s + args.hold_s:
                vw.write(np.clip(frame.astype(np.int16) + rng.integers(-1, 2, (H, W, 1),
                              dtype=np.int16), 0, 255).astype(np.uint8))
                continue                       # 段间空画面：让上一段的轨迹自然消亡
            # 0..pan_s 平移，之后静止
            p = min(1.0, p_t / max(1e-6, args.pan_s))
            # 平移量取 min(pan_px, 可用余量)；偶数段向左、奇数段向右
            span = max(0, min(args.pan_px, W - SCENE_W))
            x = int(round(span * (1 - p))) if idx % 2 == 0 else int(round(span * p))
            frame[0:SCENE_H, x:x + SCENE_W] = img

        if args.fps and fi % 1 == 0:
            n = rng.integers(-1, 2, (H, W, 1), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8)
        vw.write(frame)

    vw.release()
    Path(args.truth).parent.mkdir(parents=True, exist_ok=True)
    Path(args.truth).write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"-> {out}  {frames} 帧 / {total_s:.0f}s @ {args.fps}fps  {W}x{H}")
    print(f"   场景 {len(picked)} 段 · 每段 平移 {args.pan_px}px/{args.pan_s}s + 静止 {args.hold_s}s")
    ok_n = sum(1 for p in truth["persons"] if p["fully_ok"])
    print(f"   人物 {len(truth['persons'])} · 全合规 {ok_n} "
          f"（真值 -> {args.truth}，truth_source=generator）")
    for p in truth["persons"]:
        print(f"     {p['id']:>6}  {p['equip']}  {p['t_enter']:5.1f}–{p['t_exit']:5.1f}s  {p['img'][:28]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
