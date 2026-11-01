# -*- coding: utf-8 -*-
"""
下载第二个安全帽数据源：toborobot/Safety-helmet-dataset
==========================================================

为什么要它：第一轮微调只用 80 张，AP 从 0.579 涨到 0.766 但业务指标反而跌——
数据量太小，置信度整体漂移，真帽子掉到检测阈值以下。**这是数据问题，不是训练问题。**

这个数据集：4916 训练 / 1413 验证 / 706 测试，MIT 许可，平均框面积 1.9%（小目标，与我们同构）。
类别已人工抽样核对：**0 = 没戴帽，1 = 戴了安全帽**。

    python tools/fetch_helmet_p2.py --limit 2500

产出：datasets/helmet_p2/{train,val}/{images,labels}，类别已重映射为
      0 = helmet, 1 = nohelmet（与 models/best.pt 的 Hardhat / NO-Hardhat 对齐）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "helmet_p2"
REPO = "toborobot/Safety-helmet-dataset"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main"
API = f"https://hf-mirror.com/api/datasets/{REPO}"

# 原类别 -> 我们的 2 类（已人工核对：0=没戴帽，1=戴帽）
REMAP = {1: 0, 0: 1}
NAMES = ["helmet", "nohelmet"]


def siblings() -> list[str]:
    req = urllib.request.Request(API, headers={"User-Agent": "anxun/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=40))
    return [s["rfilename"] for s in d.get("siblings", [])]


def fetch(rel: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 200:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        r = subprocess.run(["curl", "-sL", "-m", "60", "-o", str(dst),
                            f"{BASE}/{rel}"], capture_output=True)
        if dst.exists() and dst.stat().st_size > 200:
            return True
    return False


def fetch_text(rel: str) -> str:
    """直接把标注内容拉进内存，不落临时文件。"""
    for _ in range(3):
        r = subprocess.run(["curl", "-sL", "-m", "40", f"{BASE}/{rel}"],
                           capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout.decode("utf-8", errors="ignore")
    return ""


# 仓库里的分片名和我们目录名不一致：仓库用 valid，我们落成 val
SPLIT_DIR = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def do_split(split: str, files: list[str], limit: int, workers: int) -> dict:
    sd = SPLIT_DIR.get(split, split)
    imgs = [f for f in files if f"/{sd}/images/" in f]
    lbls = {Path(f).stem: f for f in files if f"/{sd}/labels/" in f}
    imgs = imgs[:limit] if limit else imgs

    jobs = [(f, OUT / split / "images" / Path(f).name) for f in imgs]
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, rel, dst): rel for rel, dst in jobs}
        for fu in as_completed(futs):
            ok, fail = ok + (1 if fu.result() else 0), fail + (0 if fu.result() else 1)

    # 标注：并行拉取 + 内存里重写类别
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    def one(stem_rel):
        stem, rel = stem_rel
        dst = OUT / split / "labels" / f"{stem}.txt"
        txt = fetch_text(rel)
        rows = []
        for line in txt.splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            try:
                c = int(float(p[0]))
            except ValueError:
                continue
            if c not in REMAP:
                continue
            rows.append(f"{REMAP[c]} {' '.join(p[1:5])}")
        dst.write_text("\n".join(rows), encoding="utf-8")
        return len(rows)

    pairs = [(Path(f).stem, lbls[Path(f).stem]) for f in imgs if Path(f).stem in lbls]
    boxes = empty = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, pr): pr for pr in pairs}
        for fu in as_completed(futs):
            n = fu.result()
            boxes += n
            empty += (0 if n else 1)
    # 有图没标注的也补一个空文件，YOLO 会当负样本
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    for f in imgs:
        lp = OUT / split / "labels" / f"{Path(f).stem}.txt"
        if not lp.exists():
            lp.write_text("", encoding="utf-8")
            empty += 1
    return {"images": ok, "failed": fail, "boxes": boxes, "empty": empty}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每个分片最多取几张（0=全部）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--splits", default="train,valid")
    args = ap.parse_args()

    print("拉取文件清单 ...")
    files = siblings()
    print(f"仓库共 {len(files)} 个文件")

    summary = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        st = do_split(split, files, args.limit, args.workers)
        summary[split] = st
        print(f"{split:<6} 图 {st['images']:>5}（失败 {st['failed']}）  "
              f"框 {st['boxes']:>6}  空标注 {st['empty']}")

    yaml = OUT / "data.yaml"
    yaml.write_text(
        f"path: {OUT.as_posix()}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")
    print(f"\n-> {yaml}")
    print("下一步：python src/anxun/finetune.py --data datasets/helmet_p2/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
