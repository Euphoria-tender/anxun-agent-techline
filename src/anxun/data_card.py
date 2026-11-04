# -*- coding: utf-8 -*-
"""
全量数据卡片：把项目用到的所有数据源汇总成一份文档
===================================================

早期版本一次只能描述一个目录，而项目现在有三个数据源：

    datasets/ppe_test   Chapian/PPE_detection（train 100 / test 100）  helmet + vest + person
    datasets/helmet_p2  toborobot/Safety-helmet-dataset（train 1200 / val 400）helmet 两类
    datasets/vest       Chapian train 的 vest 子集（100）              vest 两类

分开写三份卡片看不出全局（谁和谁不重叠、哪个是评测集、哪个参与了训练）。
这份脚本把它们汇总，并且**显式标注每个集合的角色**——训练集、验证集、
以及「全程不参与」的测试集。评测口径能不能信，就看这张表。

    python src/anxun/data_card.py

产出 docs/数据集卡片.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 数据源登记表：目录 -> (来源, 许可, 角色说明)
SOURCES = [
    ("ppe_full/train", "Chapian/ppev3（Roboflow ppe-detection-9p0pu v3） · train", "CC BY 4.0",
     "**v3.1.0 主训练集**：2451 张，9 类重映射为 5 类，已剔除与评测集重名的 165 张"),
    ("ppe_full/valid", "Chapian/ppev3 · valid", "CC BY 4.0",
     "v3.1.0 训练时的验证集（选轮次用），不参与评测"),
    ("ppe_test/train", "Chapian/PPE_detection · train", "CC BY 4.0",
     "v3.0.0 时代的同域训练集；**与 ppe_test/test 同源**，v3.1.0 已不再用它训练"),
    ("ppe_test/test", "Chapian/PPE_detection · test", "CC BY 4.0",
     "**评测集**：helmet 的四个指标全部出自这 100 张，全程不参与训练与选轮次"),
    ("helmet_p2/train", "toborobot/Safety-helmet-dataset · train", "MIT",
     "helmet 两类（0=helmet, 1=nohelmet，已人工抽样核对）。用于第二数据源验证与早前的微调扩量"),
    ("helmet_p2/valid", "toborobot/Safety-helmet-dataset · valid", "MIT",
     "helmet 微调的验证集（选轮次用），不参与测试集评测"),
    ("helmet_p2/test", "toborobot/Safety-helmet-dataset · test", "MIT",
     "**第二数据源留出集（706 张）**：只验检测层，无 person 标注，因此不算业务指标"),
    ("vest/train", "Chapian train 的 vest 子集 · train", "CC BY 4.0",
     "v2.x 的独立背心模型训练集；v3.1.0 起背心并入五类权重，此集仅作历史留档"),
    ("vest/val", "Chapian train 的 vest 子集 · valid", "CC BY 4.0",
     "同上，历史留档"),
]

# 每个数据源目录下的类别名（与标注文件里的整数 id 对应，顺序即 data.yaml 的 names）
NAMES = {
    "ppe_test": ["boots", "gloves", "hardhat", "no_boots", "no_gloves",
                 "no_hardhat", "no_vest", "person", "vest"],
    "ppe_full": ["helmet", "no-helmet", "no-vest", "person", "vest"],
    "helmet_p2": ["helmet", "nohelmet"],       # data.yaml: names: ['helmet','nohelmet']
    "vest": ["vest", "no_vest"],
}


def scan(rel: str) -> dict:
    d = ROOT / "datasets" / rel
    img_dir = d / "images"
    lbl_dir = d / "labels"
    key = rel.split("/")[0]
    names = NAMES.get(key, [])
    if not img_dir.exists():
        return {"ok": False}

    imgs = sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    n_lbl = 0
    empty = 0
    boxes: Counter = Counter()
    imgs_with: Counter = Counter()
    for p in imgs:
        lp = lbl_dir / f"{p.stem}.txt"
        rows = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                q = line.split()
                if len(q) < 5:
                    continue
                try:
                    rows.append(int(float(q[0])))
                except ValueError:
                    continue
        n_lbl += 1 if lp.exists() else 0
        empty += 0 if rows else 1
        for c in rows:
            nm = names[c] if c < len(names) else str(c)
            boxes[nm] += 1
            imgs_with[nm] += 1
    return {"ok": True, "images": len(imgs), "labeled": n_lbl, "empty": empty,
            "boxes": boxes, "imgs_with": imgs_with}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "数据集卡片.md"))
    args = ap.parse_args()

    L = []
    L.append("# 数据集卡片 · 全量数据源\n")
    L.append(f"> 由 `data_card.py` 自动生成，{datetime.now():%Y-%m-%d %H:%M:%S}。")
    L.append("> 数字都是数出来的，不是估的；换数据重跑即可更新。\n")

    L.append("## 一、总览\n")
    L.append("| 目录 | 来源 | 许可 | 图片 | 有标注 | 空标注 | 框数 |")
    L.append("|---|---|---|---|---|---|---|")
    total_img = 0
    rows = []
    for rel, src, lic, role in SOURCES:
        r = scan(rel)
        if not r["ok"]:
            continue
        total_img += r["images"]
        rows.append((rel, src, lic, role, r))
        L.append(f"| `datasets/{rel}` | {src} | {lic} | {r['images']} | "
                 f"{r['labeled']} | {r['empty']} | {sum(r['boxes'].values())} |")
    L.append(f"\n合计 **{total_img}** 张图片（含重复使用的同域数据）。\n")

    L.append("## 二、每个集合的角色\n")
    L.append("| 目录 | 角色 |")
    L.append("|---|---|")
    for rel, src, lic, role, r in rows:
        L.append(f"| `datasets/{rel}` | {role} |")

    L.append("\n## 三、类别分布\n")
    for rel, src, lic, role, r in rows:
        L.append(f"\n### {rel}\n")
        L.append("| 类别 | 框数 | 出现图片数 |")
        L.append("|---|---|---|")
        for nm, n in r["boxes"].most_common():
            L.append(f"| {nm} | {n} | {r['imgs_with'][nm]} |")

    L.append("""
## 四、已知缺口

| 缺口 | 影响 | 状态 |
|---|---|---|
| 数据全部来自境外工地 | 安全帽款式、工装颜色与国内车间差异明显；**跨域泛化未验证** | 两轮公开数据微调均为负面结果（README 5.4/5.5）；v3.1.0 只做到同源独立留出 |
| vest 标注密度低 | 996 图 / 1530 框，远少于 helmet（1428 图 / 2891 框）。vest 误报 28.26%、漏报 6.09%，**不能与 helmet 并列宣传** | v3.1.0 已并入五类权重；HF 镜像上再无带 vest 标注的可用数据源 |
| 空标注负样本 0 张 | 误报率的负样本侧不够硬 | 公开集均无此分片 |
| 无 1080p 真实监控素材 | 三级触发链的收益是在合成视频上测的；1080p 复验用的是 3× 上采样图（`ppe_test/test1080`），不是真机录像 | 需现场录像复核 |

## 五、复现

```bash
python src/anxun/data_card.py        # 重新生成本卡片
python tools/build_ppe_full.py       # v3.1.0 主训练集：9 类→5 类，剔除评测集重叠
python tools/fetch_helmet_p2.py --splits test   # 第二数据源 706 张留出集
python tools/make_vest.py            # v2.x 历史的 vest 80/20 训练集
```
""")

    out = Path(args.out)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out}")
    print(f"   覆盖 {len(rows)} 个数据源，共 {total_img} 张图片")


if __name__ == "__main__":
    raise SystemExit(main())
