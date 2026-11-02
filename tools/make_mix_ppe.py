# -*- coding: utf-8 -*-
"""
安巡 Agent · 混编训练集（ppe1206 + ppe_test）
=============================================

为什么要混编而不是替换：ppe1206（997 张）单独训出来的模型，检测层更强
（helmet AP 0.862 vs 0.785），但业务层塌了——它学到的 person 框法与
现有判定链（用 COCO 人体框推头部区域）不兼容，IoU>0.5 匹配不上，
判定链直接跳过这些人，e2e 漏报 22% → 50%。

混编的假设：让模型同时见过两种标注风格，旧域的框法不至于丢。
如果旧测试集上业务指标能回到 6% / 17% 一线，同时 helmet AP 涨上去，
这条路才算成立。

组成：
    train = ppe1206/train (997) + ppe_test/train (100，9 类重映射为 5 类)
    val   = ppe1206/valid (119)
    test  = 不在这里出，评测时分别用 ppe1206/test 与 ppe_test/test

    python tools/make_mix_ppe.py
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "datasets" / "ppe1206"
OLD = ROOT / "datasets" / "ppe_test"
OUT = ROOT / "datasets" / "mix_ppe"

NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]

# 旧集 9 类 -> 新集 5 类；None 表示丢弃（手套/靴子不在判定链里）
OLD_MAP = {
    0: None,        # boots
    1: None,        # gloves
    2: 0,           # hardhat    -> helmet
    3: None,        # no_boots
    4: None,        # no_gloves
    5: 1,           # no_hardhat -> no-helmet
    6: 2,           # no_vest    -> no-vest
    7: 3,           # person     -> person
    8: 4,           # vest       -> vest
}


def link_or_copy(src: Path, dst: Path) -> None:
    """同分区用硬链接（省空间且快），跨分区退回复制。"""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def put(img: Path, lbl: Path, split: str, rename: str | None, remap: dict | None):
    """把一张图和它的标签放进目标 split，返回实际写入的框数。"""
    di = OUT / split / "images"
    dl = OUT / split / "labels"
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    stem = rename or img.stem
    link_or_copy(img, di / f"{stem}{img.suffix}")

    rows = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if remap is not None:
                if c not in remap or remap[c] is None:
                    continue
                c = remap[c]
            rows.append(f"{c} " + " ".join(p[1:5]))
    (dl / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""),
                                    encoding="utf-8")
    return len(rows)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {"train": Counter(), "valid": Counter()}

    for img in sorted((NEW / "train" / "images").glob("*.jpg")):
        n = put(img, NEW / "train" / "labels" / f"{img.stem}.txt", "train", None, None)
        stats["train"]["images"] += 1
        stats["train"]["boxes"] += n

    for img in sorted((NEW / "valid" / "images").glob("*.jpg")):
        n = put(img, NEW / "valid" / "labels" / f"{img.stem}.txt", "valid", None, None)
        stats["valid"]["images"] += 1
        stats["valid"]["boxes"] += n

    for img in sorted((OLD / "train" / "images").glob("*.jpg")):
        # 旧集文件名是 000017_jpg.rf.xxx，加前缀避免与新集混淆
        n = put(img, OLD / "train" / "labels" / f"{img.stem}.txt", "train",
                f"old_{img.stem}", OLD_MAP)
        stats["train"]["images"] += 1
        stats["train"]["boxes"] += n
        stats["train"]["old"] += 1

    (OUT / "data.yaml").write_text(
        "# 混编训练集：ppe1206 train/valid + ppe_test train（重映射为 5 类）\n"
        "# 路径相对本文件所在目录\n"
        f"train: train/images\nval: valid/images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES!r}\n", encoding="utf-8")

    print(f"输出：{OUT}")
    for k, v in stats.items():
        extra = f"（其中旧集 {v['old']} 张）" if v.get("old") else ""
        print(f"  {k:5s} {v['images']:5d} 张 / {v['boxes']:5d} 框 {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
