# -*- coding: utf-8 -*-
"""
安巡 Agent · ppe1206 单模型训练
===============================

背景：v2.2.0 之前是「三模型串联」——COCO 人体 + 第三方安全帽 + 自训背心。
两个硬伤：① 安全帽权重是第三方在境外工地训的，在自有测试集 AP 只有 0.565；
② 背心只有 100 张训练图，误报 27.3%。

ppe1206 这个集一次给齐了 person / helmet / no-helmet / vest / no-vest 五类，
998 张训练图，所以可以训**一个模型直接出全部属性**，把串联链路压成单模型：

    - 不用再跨模型做 person↔helmet 的 IoU 配对（那是误差最大的一环）
    - 部署包从 3 个权重 105 MB 降到 1 个
    - 反向证据（no-helmet / no-vest）由模型自己给，不再靠"找不到就算没戴"

训练集/验证集/测试集是数据集自带的切分，**测试集全程不参与训练和选轮次**，
指标用 evaluate_real.py 在同一口径下出。

    python tools/train_ppe1206.py --arch n --epochs 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[1]   # tools/ 的上一级才是 techline 根
DATA = ROOT / "datasets" / "ppe1206"
RUNS = ROOT / "runs"

NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]


def check_data(data: Path) -> dict:
    """数一遍数据，缺图缺标注直接报错，别等训到一半才发现。"""
    stats: dict[str, dict] = {}
    for split in ("train", "valid", "test"):
        img_dir = data / split / "images"
        lbl_dir = data / split / "labels"
        if not img_dir.exists():
            if split == "test":
                continue
            raise SystemExit(f"缺目录：{img_dir}")
        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        missing = [p.name for p in imgs if not (lbl_dir / (p.stem + ".txt")).exists()]
        if missing:
            raise SystemExit(f"{split} 有 {len(missing)} 张图没有标注，例：{missing[:3]}")
        stats[split] = {"images": len(imgs)}
    return stats


def fix_yaml(data: Path) -> Path:
    """
    修 data.yaml 的路径。

    数据集自带的版本写的是 `train: ../train/images`，但 ultralytics 按 **yaml 所在目录**
    解析相对路径，这么写会指到 datasets/train/images，直接找不到。改成不带 ../ 的版本。
    原文件留一份 .orig 备份，避免反复覆盖。
    """
    yml = data / "data.yaml"
    text = yml.read_text(encoding="utf-8")
    if "../" not in text:
        return yml

    backup = data / "data.yaml.orig"
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    fixed = (f"# 安巡 Agent · ppe（施工安全 PPE 检测）\n"
             f"# 路径相对本文件所在目录，不写 path 字段，换机器不用改\n"
             f"train: train/images\n"
             f"val: valid/images\n"
             f"test: test/images\n"
             f"nc: {len(NAMES)}\n"
             f"names: {NAMES!r}\n")
    yml.write_text(fixed, encoding="utf-8")
    print(f"已修正 data.yaml 路径（原文件备份为 data.yaml.orig）")
    return yml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="n", choices=("n", "s", "m"),
                    help="YOLOv8 规模；n 对齐现有 CPU 46ms 的部署指标")
    ap.add_argument("--init", default="", help="起点权重；留空用 models/yolov8n.pt")
    ap.add_argument("--data", default="datasets/ppe1206",
                    help="数据集目录（需含 train/valid 与 data.yaml）")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="", help="run 名；留空 ppe1206_<arch>")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--cache", default="ram", choices=("ram", "disk", "none"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    stats = check_data(data_dir)
    print(f"数据集：{data_dir}")
    print("数据体检：")
    for k, v in stats.items():
        print(f"  {k:5s} {v['images']:5d} 张")
    yml = fix_yaml(data_dir)

    init = Path(args.init) if args.init else ROOT / "models" / "yolov8n.pt"
    if args.arch != "n" and not args.init:
        init = Path(f"yolov8{args.arch}.pt")     # 本地没有，让 ultralytics 自己下
    if str(init) and Path(init).exists() is False and "/" not in str(init):
        print(f"本地无 {init}，训练时由 ultralytics 自动下载")

    name = args.name or f"ppe1206_{args.arch}"
    print(f"\n起点权重：{init}\nrun 名：{name}\nepochs：{args.epochs}\n")

    from ultralytics import YOLO
    model = YOLO(str(init))

    # 数据量从 100 张涨到 998 张，过拟合风险已经下来，增强用「中等」而不是 strong：
    # 强增强是给 100 张的小集硬撑泛化用的，这里再用会把 no-helmet 这种长尾类打散。
    model.train(
        data=str(yml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=0,                    # Windows 下多进程 dataloader 容易卡
        lr0=args.lr0, lrf=0.01, cos_lr=True,
        warmup_epochs=3, patience=args.patience,
        freeze=args.freeze or None,
        mosaic=1.0, close_mosaic=10,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.4, fliplr=0.5,
        mixup=0.0, weight_decay=0.0005,
        cache=False if args.cache == "none" else args.cache,
        project=str(RUNS), name=name, exist_ok=True, verbose=True,
    )

    best = RUNS / name / "weights" / "best.pt"
    print(f"\n训练完成：{best}")

    # 把训练侧的验证指标留一份，方便和测试集对比看有没有过拟合
    mpath = RUNS / name / "results.csv"
    if mpath.exists():
        rows = mpath.read_text(encoding="utf-8").strip().splitlines()
        head = [h.strip() for h in rows[0].split(",")]
        last = [c.strip() for c in rows[-1].split(",")]
        rec = dict(zip(head, last))
        keep = {k: rec[k] for k in rec
                if k in ("epoch", "metrics/precision(B)", "metrics/recall(B)",
                         "metrics/mAP50(B)", "metrics/mAP50-95(B)")}
        (RUNS / name / "summary.json").write_text(
            json.dumps({"arch": args.arch, "epochs": args.epochs,
                        "imgsz": args.imgsz, "init": str(init),
                        "final_val": keep}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"验证集（最后一轮）：{keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
