# -*- coding: utf-8 -*-
"""
构建反光背心（vest）二分类训练集
=================================

为什么单独做：规则引擎早就支持 vest（上身区域判定 + L/S 风险矩阵 + 整改动作），
但一直没有对应权重，所以「未穿戴反光背心」这条判定实际是空转的。

数据来源：Chapian/PPE_detection 的 train 分片，100 张里有 75 个 vest + 237 个 no_vest。
反光背心是大目标、特征强（高可见度条纹），这个量级够用——和安全帽那种 30px 小目标不同。

    python tools/make_vest.py

产出 datasets/vest/{images,labels}，类别 0 = vest，1 = no_vest。
其余类别（person / helmet / gloves ...）一律丢弃：这个模型只管背心。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "datasets" / "ppe_test" / "train"
OUT = ROOT / "datasets" / "vest"
SRC_NAMES = ["boots", "gloves", "hardhat", "no_boots", "no_gloves",
             "no_hardhat", "no_vest", "person", "vest"]

KEEP = {"vest": 0, "no-vest": 1}


def main() -> int:
    img_dir = SRC / "images"
    lbl_dir = SRC / "labels"
    if not img_dir.exists():
        print("缺训练集：python tools/fetch_ppe_test.py --splits train")
        return 2

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    # 80/20 切分：val 必须是训练时没见过的图。
    # 第一版 val 和 train 是同一批，模型选择等于在训练集上做，
    # 选出来的必然是最过拟合的那一轮——实测 mAP50 从 0.995（训练集）掉到 0.415（测试集）。
    files = sorted(img_dir.glob("*.jpg"))
    val_idx = set(range(0, len(files), 5))          # 每 5 张抽 1 张做验证

    n_img = n_box = n_empty = 0
    for i, img in enumerate(files):
        lp = lbl_dir / f"{img.stem}.txt"
        rows = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                name = SRC_NAMES[int(float(p[0]))].replace("_", "-")
                if name in KEEP:
                    rows.append(f"{KEEP[name]} {' '.join(p[1:5])}")
        split = "val" if i in val_idx else "train"
        shutil.copy2(img, OUT / split / "images" / img.name)
        (OUT / split / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(rows), encoding="utf-8")
        n_img += 1
        n_box += len(rows)
        n_empty += (0 if rows else 1)

    # 路径一律写成「相对 data.yaml 所在目录」：ultralytics 在没有 path 字段时
    # 以 yaml 所在目录为根，换机器 / 换盘符都不用改文件（旧版写死了绝对路径 D:/...）。
    (OUT / "data.yaml").write_text(
        "train: train/images\nval: val/images\n"
        "nc: 2\nnames: ['vest', 'no_vest']\n", encoding="utf-8")

    print(f"图片 {n_img}（无背心框的负样本 {n_empty}）  框 {n_box}  "
          f"训练 {n_img - len(val_idx)} / 验证 {len(val_idx)}")
    print(f"-> {OUT / 'data.yaml'}")
    print("下一步：python src/anxun/finetune.py --data datasets/vest/data.yaml "
          "--init models/yolov8n.pt --name vest_ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
