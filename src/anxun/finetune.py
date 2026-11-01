# -*- coding: utf-8 -*-
"""
安巡 Agent · 安全帽模型微调
==========================

为什么必须微调：现有 helmet 权重是第三方在境外工地训的，在我们这批现场图上
helmet AP 只有 0.579，36 例漏报里 53% 是灰区转人工——这是域不匹配，不是结构问题。

两个关键决定（都为了让 100 张训练图能榨出最大价值）：

1. **压缩成 2 类**（hardhat / no_hardhat）。原数据集 9 类，如果按 9 类训，
   分类头要重新初始化、100 张图还要分掉一半容量给手套靴子。压成 2 类后
   nc 与预训练权重完全一致，整网（含检测头）可以直接迁移。

2. **训练集内部再切 80/20**。验证集只用于选最优轮次，**测试集全程不参与**，
   最后用 evaluate_real 在 100 张独立测试图上出指标——不给自己刷分的机会。

    python src/anxun/finetune.py --epochs 50

产出 runs/finetune/weights/best.pt，并自动跑一次测试集评估 -> data/metrics_finetuned.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets" / "ppe_test"
WORK = ROOT / "datasets" / "ppe_ft"
RUNS = ROOT / "runs"

# 原数据集类别 → 我们的 2 类
KEEP = {"hardhat": 0, "no-hardhat": 1}
SRC_NAMES = ["boots", "gloves", "hardhat", "no_boots", "no_gloves",
             "no_hardhat", "no_vest", "person", "vest"]


def remap(split: str, keep_every: int = 0) -> dict:
    """把某个分片重映射成 2 类，写进 work 目录。keep_every=5 表示每 5 张抽 1 张做 val。"""
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"
    stats = {"images": 0, "boxes": 0, "empty": 0, "val": 0}

    for i, img in enumerate(sorted(src_img.glob("*.jpg"))):
        lp = src_lbl / (img.stem + ".txt")
        rows = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                name = SRC_NAMES[int(float(p[0]))].replace("_", "-")
                if name not in KEEP:
                    continue
                rows.append(f"{KEEP[name]} {' '.join(p[1:5])}")

        dst_split = "val" if (keep_every and i % keep_every == 0) else "train"
        (WORK / "images" / dst_split).mkdir(parents=True, exist_ok=True)
        (WORK / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, WORK / "images" / dst_split / img.name)
        (WORK / "labels" / dst_split / (img.stem + ".txt")).write_text(
            "\n".join(rows), encoding="utf-8")

        stats["images"] += 1
        stats["boxes"] += len(rows)
        stats["empty"] += (0 if rows else 1)
        stats["val"] += (1 if dst_split == "val" else 0)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--init", default=str(ROOT / "models" / "best.pt"),
                    help="从哪个权重开始微调（同 2 类，整网迁移）")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--no-eval", action="store_true")
    # 数据来源：给了 yaml 就直接用它（多数据源合并后的产物）；不给则走原来的
    # 「Chapian train 内部切 80/20」——那条路只适合 100 张的规模。
    ap.add_argument("--data", default="",
                    help="数据集 yaml（如 datasets/helmet_p2/data.yaml）")
    # 超参：小数据和大数据要的不是一套东西
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--mosaic", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前 N 层；只想训练头就设 10")
    # 小数据抗过拟合：数据越少，增强越重、权重衰减越大
    ap.add_argument("--aug", default="normal", choices=("normal", "strong"),
                    help="strong = 重增强 + 大 weight decay，100 张量级专用")
    args = ap.parse_args()

    if args.data:
        data_yaml = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() \
            else Path(args.data)
        if not data_yaml.exists():
            print(f"找不到数据集 yaml：{data_yaml}")
            return 2
        src_note = str(data_yaml.relative_to(ROOT)) if str(data_yaml).startswith(str(ROOT)) \
            else str(data_yaml)
        print(f"数据集：{src_note}（已按 0=helmet / 1=nohelmet 重映射）")
    else:
        if not (SRC / "train" / "images").exists():
            print("缺训练集：先跑 python tools/fetch_ppe_test.py --splits train")
            return 2
        if WORK.exists():
            shutil.rmtree(WORK)
        st = remap("train", keep_every=5)      # 80 训练 / 20 验证
        print(f"训练集重映射完成：{st['images']} 张（val {st['val']}）、"
              f"{st['boxes']} 个框、空标注 {st['empty']} 张")
        # 不写 path：以 yaml 所在目录为根，路径相对它，换机器不用改
        (WORK / "data.yaml").write_text(
            "train: images/train\nval: images/val\n"
            "nc: 2\nnames: ['hardhat', 'no_hardhat']\n", encoding="utf-8")
        data_yaml = WORK / "data.yaml"

    from ultralytics import YOLO
    model = YOLO(args.init)
    print(f"起点权重：{Path(args.init).name}（nc={len(model.names)}）")

    aug = dict(mosaic=args.mosaic, close_mosaic=max(5, args.epochs // 6),
               hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
               degrees=0.0, translate=0.1, scale=0.3, fliplr=0.5,
               weight_decay=0.0005)
    if args.aug == "strong":
        # 第一版 vest 训练集 mAP 0.995 / 测试集 0.415，典型过拟合。
        # 治它不靠更多数据（没有），靠把增强拉满 + 权重衰减加大 + 冻结骨干更久。
        aug.update(mosaic=1.0, scale=0.5, translate=0.2, hsv_s=0.7, hsv_v=0.5,
                   degrees=5.0, shear=2.0, perspective=0.0002,
                   mixup=0.1, copy_paste=0.1, weight_decay=0.001)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=0,                 # Windows 下多进程 dataloader 容易卡
        lr0=args.lr0, lrf=0.01, cos_lr=True,
        warmup_epochs=3, patience=args.patience,
        freeze=args.freeze or None,
        **aug,
        project=str(RUNS), name=args.name, exist_ok=True, verbose=True,
    )

    best = RUNS / args.name / "weights" / "best.pt"
    print(f"\n训练完成：{best}")

    if args.no_eval or not best.exists():
        return 0

    # ---- 在独立的 100 张测试图上评估（与旧权重同口径）----
    from anxun.detectors import DualDetector, pick_device
    from anxun.evaluate_real import run_eval, warmup

    det = DualDetector(str(ROOT / "models" / "yolov8n.pt"), str(best),
                       imgsz=args.imgsz, device=pick_device())
    warmup(det)
    out = run_eval(SRC / "test", det)
    r = out["results"]
    payload = {
        "kind": "model-measured",
        "config": {"weights": f"yolov8n.pt + {best.name}", "imgsz": args.imgsz,
                   "conf": 0.25, "device": f"GPU（torch {__import__('torch').__version__}）",
                   "finetuned_on": src_note,
                   "epochs": args.epochs},
        "dataset": {"name": "Chapian/PPE_detection · test", "license": "CC BY 4.0",
                    "images": out["images"], "persons_gt": out["persons_gt"],
                    "persons_pred": out["persons_pred"],
                    "eval_note": "测试集全程未参与训练与选轮次"},
        "results": r,
    }
    (ROOT / "data" / "metrics_finetuned.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 62)
    print(f"微调后（独立测试集）")
    print(f"mAP@0.5   {r['mAP50']}   逐类 { {k: v['ap'] for k, v in r['per_class'].items()} }")
    print(f"误报率     {r['fp_rate']}%   漏报率 {r['fn_rate']}%")
    print(f"混淆矩阵   {r['confusion']}")
    print(f"-> data/metrics_finetuned.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
