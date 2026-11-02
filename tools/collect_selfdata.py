# -*- coding: utf-8 -*-
"""
现场视频 → 可标注帧
===================

现场部署时，摄像头录像要变成能训练的标注集，中间隔着「抽帧 + 去重 + 建目录」三步。
这个脚本把它从一件写代码的事，变成一件跑命令的事：

    # 1. 把现场视频放进 datasets/raw/（mp4/mov/avi 都行）
    python tools/collect_selfdata.py --src datasets/raw

    # 2. 用任意 YOLO 标注工具打开 datasets/self/2026-11-02/images 标注
    #    （labelImg / Label Studio / Ultralytics HUB 都认这套目录）

    # 3. 标完核对一遍覆盖度
    python tools/collect_selfdata.py --check datasets/self/2026-11-02

设计取舍
--------
- **只抽帧，不自动标注。** 自动标注会把模型的错误固化成真值，测出来的指标是自己骗自己。
- **按秒抽 + 感知去重**：固定机位站着不动的十秒，抽 10 帧等于 1 帧。
  用 8×8 灰度均值哈希去重，阈值可调；宁可多留，别漏掉换姿势的瞬间。
- **默认不并入测试集。** 现场帧只做训练/验证；测试集必须保持「从未参与训练」，
  这是四个指标可信的前提。要评估新域的效果，单独建一个集合评，别混进 ppe_test。
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NAMES = ["boots", "gloves", "hardhat", "no_boots", "no_gloves",
         "no_hardhat", "no_vest", "person", "vest"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def imread(path: Path):
    """Windows 上 cv2.imread 遇到非 ASCII 路径（用户名「华硕」）会静默返回 None。"""
    import cv2
    import numpy as np
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite(path: Path, img) -> bool:
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def dhash(img, size: int = 8) -> int:
    """差值哈希：8×8 灰度 + 水平相邻比较，64 bit。"""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect(src: Path, out: Path, every: float, min_gap: int, limit: int) -> int:
    import cv2

    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        print(f"没找到视频：{src}（支持 {sorted(VIDEO_EXT)}）")
        return 2

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    rows, hashes, kept = [], [], 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"打不开：{v.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps * every)))
        idx = 0
        while kept < limit or limit <= 0:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                h = dhash(frame)
                if all(hamming(h, x) >= min_gap for x in hashes):
                    hashes.append(h)
                    kept += 1
                    name = f"{v.stem}_{idx:06d}"
                    imwrite(out / "images" / f"{name}.jpg", frame)
                    # 空标注：YOLO 把「有图无框」当负样本，正好用来压误报
                    (out / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
                    rows.append({"image": f"{name}.jpg", "video": v.name,
                                 "frame": idx, "t_s": round(idx / fps, 2),
                                 "w": frame.shape[1], "h": frame.shape[0]})
            idx += 1
        cap.release()
        print(f"  {v.name:<28} 抽到 {sum(1 for r in rows if r['video'] == v.name)} 帧")

    if not rows:
        print("一帧都没抽到：调大 --every 或调小 --min-gap")
        return 1

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    (out / "data.yaml").write_text(
        "train: images\nval: images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")

    print(f"\n共 {kept} 帧 -> {out}")
    print("下一步：")
    print(f"  1) 用 YOLO 标注工具打开 {out / 'images'}，标注写到同名的 labels/*.txt")
    print(f"  2) 标完核对：python tools/collect_selfdata.py --check {out}")
    print("  3) 训练：把 data.yaml 的 val 指向留出的验证集，再跑 finetune.py"
          "（现场帧只做训练/验证，别并进 ppe_test）")
    return 0


def check(target: Path) -> int:
    """核对标注覆盖度：有多少张已标、多少张还是空标注、类别分布。"""
    imgs = sorted((target / "images").glob("*.jpg"))
    if not imgs:
        print(f"{target}/images 里没有图")
        return 2
    labeled, empty, boxes = 0, 0, 0
    names = []
    dy = target / "data.yaml"
    if dy.exists():
        import re
        m = re.search(r"names:\s*\[(.*?)\]", dy.read_text(encoding="utf-8"), re.S)
        if m:
            names = [s.strip().strip("'\"") for s in m.group(1).split(",")]
    cls = {}
    for p in imgs:
        lp = target / "labels" / f"{p.stem}.txt"
        rows = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()] \
            if lp.exists() else []
        if rows:
            labeled += 1
            for ln in rows:
                c = int(float(ln.split()[0]))
                cls[c] = cls.get(c, 0) + 1
                boxes += 1
        else:
            empty += 1

    print(f"图 {len(imgs)} 张 · 有标注 {labeled} · 空标注 {empty} · 框 {boxes}")
    if cls and names:
        for c, n in sorted(cls.items()):
            label = names[c] if c < len(names) else f"类{c}"
            print(f"  {label:<12} {n} 框")
    if empty == len(imgs):
        print("⚠️ 全是空标注：要么还没标，要么这批帧里确实没有人——"
              "空标注不是失败，是负样本，能压误报；但不要整批都空。")
    elif labeled < len(imgs) * 0.3:
        print("⚠️ 已标比例 < 30%：先把人物多的帧标完再训练，否则学不到东西。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="现场视频抽帧 → 可标注目录")
    ap.add_argument("--src", default=str(ROOT / "datasets" / "raw"),
                    help="视频所在目录（递归查找）")
    ap.add_argument("--out", default="", help="输出目录，默认 datasets/self/<今天>")
    ap.add_argument("--every", type=float, default=2.0, help="每几秒抽一帧（默认 2）")
    ap.add_argument("--min-gap", type=int, default=6,
                    help="去重阈值：汉明距离小于它就算重复（64 bit，默认 6；调大=留更多帧）")
    ap.add_argument("--limit", type=int, default=0, help="最多抽多少帧（0=不限）")
    ap.add_argument("--check", default="", help="只核对某个已抽好的目录")
    args = ap.parse_args()

    if args.check:
        return check(Path(args.check).resolve())

    src = Path(args.src).resolve()
    if not src.exists():
        print(f"目录不存在：{src}\n先把视频放进去（手机拍的 .mp4/.mov 直接丢进去就行）")
        return 2
    out = Path(args.out).resolve() if args.out else \
        ROOT / "datasets" / "self" / datetime.now().strftime("%Y-%m-%d")
    return collect(src, out, args.every, args.min_gap, args.limit)


if __name__ == "__main__":
    sys.exit(main())
