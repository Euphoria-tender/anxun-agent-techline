# -*- coding: utf-8 -*-
"""
安巡 Agent · 从 Roboflow COCO 全量重建五类训练集
================================================

背景：v3.0.0 的五类权重用 ppe1206（1206 张，997/119/90）训练，那份原图已经丢失。
但源数据集还在——`Chapian/ppev3` 的 `PPE DETECTION.v3i.coco.zip`（Roboflow
`ppe-detection-9p0pu` v3，CC BY 4.0），2896 张、32498 框、9 类、已 resize 到 640。

**关键：必须先剔除评测集**。ppe_test 的 200 张（train 100 + test 100）全部来自
这个源集（test 里 65 张重名、train 100 张全重名）。不剔除就是训练集里有测试图，
指标再高也不可信。

产出：
    datasets/ppe_full/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt
      data.yaml            nc=5，顺序与现役权重一致

类别映射（与 models/best.pt 的 names 完全一致，detector 的 LABEL_MAP 不用改）：
    helmet / no-helmet / no-vest / person / vest
  丢弃 boots / gloves / no_boots / no_gloves 与 Roboflow 的 PPE 汇总类。

    python tools/build_ppe_full.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 与 models/best.pt 的 names 严格一致
NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]

# COCO 类名 -> 五类下标；未列出的类别整条丢弃
MAP = {
    "hardhat": 0,
    "no_hardhat": 1,
    "no_vest": 2,
    "person": 3,
    "vest": 4,
}

YAML = """# 安巡 Agent · ppe_full（Roboflow ppe-detection-9p0pu v3 全量，已剔除评测集）
# 源: https://universe.roboflow.com/.../ppe-detection-9p0pu  · CC BY 4.0
# 由 tools/build_ppe_full.py 生成，不要手改
train: train/images
val: valid/images
nc: {nc}
names: {names}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "_dl" / "ppev3" / "train"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "ppe_full"))
    ap.add_argument("--exclude", default=str(ROOT / "datasets" / "ppe_test"),
                    help="排除该目录下所有 split 的图片名，防止训练/评测泄漏")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    ann = json.loads((src / "_annotations.coco.json").read_text(encoding="utf-8"))

    # ---- 1. 收集要排除的图片名（评测集 train + test + valid）----
    exclude: set[str] = set()
    for sub in Path(args.exclude).glob("*/images"):
        exclude |= {p.name for p in sub.glob("*.*")}
    print(f"排除评测集图片：{len(exclude)} 张")

    # ---- 2. 按图聚合标注 ----
    by_img: dict[int, list[dict]] = defaultdict(list)
    dropped_cats = Counter()
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        by_img[a["image_id"]].append(a)

    imgs = sorted(ann["images"], key=lambda x: x["file_name"])
    kept, skipped_leak, skipped_empty = [], 0, 0
    for im in imgs:
        fname = im["file_name"]
        if fname in exclude:
            skipped_leak += 1
            continue
        rows = []
        W, H = im["width"], im["height"]
        for a in by_img.get(im["id"], []):
            cname = next(c["name"] for c in ann["categories"] if c["id"] == a["category_id"])
            if cname not in MAP:
                dropped_cats[cname] += 1
                continue
            x, y, w, h = a["bbox"]
            # COCO 左上角+宽高 -> YOLO 归一化中心点；越界一律夹到 [0,1]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(W), x + w), min(float(H), y + h)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            rows.append((MAP[cname], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                         (x2 - x1) / W, (y2 - y1) / H))
        if not rows:
            skipped_empty += 1
            continue
        kept.append((im, rows))

    print(f"源集 {len(imgs)} 张 -> 可用 {len(kept)} 张"
          f"（剔除泄漏 {skipped_leak} / 无有效框 {skipped_empty}）")

    # ---- 3. 划分 train / val（按图，不按框）----
    random.Random(args.seed).shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio))
    splits = {"valid": kept[:n_val], "train": kept[n_val:]}

    # ---- 4. 落盘 ----
    if out.exists():
        shutil.rmtree(out)
    for split, items in splits.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for im, rows in items:
            name = im["file_name"]
            shutil.copy2(src / name, out / split / "images" / name)
            with (out / split / "labels" / (Path(name).stem + ".txt")).open(
                    "w", encoding="utf-8") as f:
                for cid, cx, cy, bw, bh in rows:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    (out / "data.yaml").write_text(
        YAML.format(nc=len(NAMES), names=NAMES), encoding="utf-8")

    # ---- 5. 统计 ----
    print()
    for split, items in splits.items():
        c = Counter(r[0] for _im, rows in items for r in rows)
        tot = sum(c.values())
        print(f"{split}: {len(items)} 张 / {tot} 框")
        for i, n in enumerate(NAMES):
            print(f"   {n:<12} {c.get(i, 0):>6}")
    if dropped_cats:
        print("\n丢弃的类别（框数）:", dict(dropped_cats.most_common()))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
