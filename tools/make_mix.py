# -*- coding: utf-8 -*-
"""
合并两个数据源，产出训练用 yaml
================================

    datasets/helmet_p2   toborobot（1200 训 / 400 验）  ← 量大，但境外工地
    datasets/ppe_test    Chapian（100 训 / 100 测）      ← 量小，但和我们同域

为什么要混：只用量大的会把域带偏，只用量小的又会重蹈「80 张训不出置信度」的覆辙。
混起来用小数据把域拉住，用大数据把置信度撑住。

    python tools/make_mix.py

产出 datasets/mix/data.yaml，`train` 用列表形式指向多个图片目录——
YOLO 原生支持，标注目录靠把 images 换成 labels 自动定位，不用复制文件。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "datasets" / "ppe_test"
P2 = ROOT / "datasets" / "helmet_p2"
MIX = ROOT / "datasets" / "mix"
WORK = ROOT / "datasets" / "ppe_mix"

# Chapian 的 9 类 -> 我们的 2 类
KEEP = {"hardhat": 0, "no-hardhat": 1}
SRC_NAMES = ["boots", "gloves", "hardhat", "no_boots", "no_gloves",
             "no_hardhat", "no_vest", "person", "vest"]


def remap_chapian() -> dict:
    """Chapian train -> datasets/ppe_mix/{images,labels}（2 类）。"""
    src_img = SRC / "train" / "images"
    src_lbl = SRC / "train" / "labels"
    if not src_img.exists():
        raise SystemExit("缺 Chapian 训练集：python tools/fetch_ppe_test.py --splits train")

    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "images").mkdir(parents=True, exist_ok=True)
    (WORK / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "boxes": 0}
    for img in sorted(src_img.glob("*.jpg")):
        lp = src_lbl / f"{img.stem}.txt"
        rows = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                name = SRC_NAMES[int(float(p[0]))].replace("_", "-")
                if name in KEEP:
                    rows.append(f"{KEEP[name]} {' '.join(p[1:5])}")
        shutil.copy2(img, WORK / "images" / img.name)
        (WORK / "labels" / f"{img.stem}.txt").write_text("\n".join(rows), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(rows)
    return stats


def main() -> int:
    st = remap_chapian()
    print(f"Chapian 并入：{st['images']} 张 / {st['boxes']} 框 -> {WORK.relative_to(ROOT)}")

    MIX.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        """相对 data.yaml 所在目录的路径——换机器 / 换盘符都不用改 yaml。"""
        return Path(os.path.relpath(p, MIX)).as_posix()

    trains = []
    if (P2 / "train" / "images").exists():
        n = len(list((P2 / "train" / "images").glob("*.jpg")))
        trains.append(rel(P2 / "train" / "images"))
        print(f"toborobot   ：{n} 张")
    trains.append(rel(WORK / "images"))

    val = rel(P2 / "valid" / "images")
    yaml = MIX / "data.yaml"
    yaml.write_text(
        "train:\n" + "".join(f"  - {t}\n" for t in trains) +
        f"val: {val}\n"
        f"nc: 2\nnames: ['helmet', 'nohelmet']\n", encoding="utf-8")

    print(f"\n-> {yaml.relative_to(ROOT)}")
    print(yaml.read_text(encoding="utf-8"))
    print("下一步：python src/anxun/finetune.py --data datasets/mix/data.yaml --epochs 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
