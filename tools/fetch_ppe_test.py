# -*- coding: utf-8 -*-
"""
安巡 Agent · 拉取带标注的 PPE 验证集
===================================

数据集：Chapian/PPE_detection（Roboflow PPE Detection v3，CC BY 4.0）
用途：真实照片上的四指标评测，不做训练

为什么不用 keremberke/hard-hat-detection：
    那个包的 valid.zip 里**只有图片没有标注**（3962 张 jpg，0 个 txt），
    没法算 mAP 和误报漏报。这份 200 张的测试集是标准 YOLO 结构，图片标签齐全。
    数据量小是缺点，但"有标注"比"量大"重要——没有真值就没有指标。

    python tools/fetch_ppe_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "Chapian/PPE_detection"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/"
API = f"https://hf-mirror.com/api/datasets/{REPO}"
OUT = Path(__file__).resolve().parents[1] / "datasets" / "ppe_test"
SPLITS = ("test",)

YAML = """# Chapian/PPE_detection · Roboflow PPE Detection v3 · CC BY 4.0
# 来源: https://universe.roboflow.com/developing-automatic-surveillance-system-for-personal-protective-equipment-compliance-in-real-construction-sites-using-deep-neural-network-models/ppe-detection-9p0pu/dataset/3
train: images
val: images
test: images
nc: 9
names: ['boots', 'gloves', 'hardhat', 'no_boots', 'no_gloves', 'no_hardhat', 'no_vest', 'person', 'vest']
"""


def siblings() -> list[str]:
    raw = subprocess.run(["curl", "-sL", "-m", "40", API],
                         capture_output=True).stdout.decode("utf-8", "ignore")
    data = json.loads(raw)
    return [s["rfilename"] for s in data.get("siblings", [])]


def fetch(rel: str) -> str:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return f"skip {rel}"
    url = BASE + urllib.parse.quote(rel)
    # 用 curl 而不是 urllib：镜像站会挡 urllib 的请求头（403），curl 正常
    rc = subprocess.run(
        ["curl", "-sL", "-m", "90", "--retry", "2", "-o", str(dst), url],
        capture_output=True).returncode
    if rc != 0 or not dst.exists() or dst.stat().st_size == 0:
        if dst.exists():
            dst.unlink()
        return f"FAIL {rel}"
    return f"ok   {rel}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test", help="逗号分隔，如 test,train")
    splits = tuple(s.strip() for s in ap.parse_args().splits.split(","))
    files = [f for f in siblings() if f.startswith(splits)]
    imgs = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    lbls = [f for f in files if f.endswith(".txt")]
    print(f"待下载：{len(imgs)} 张图 + {len(lbls)} 个标注")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.yaml").write_text(YAML, encoding="utf-8")

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for msg in ex.map(fetch, imgs + lbls):
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(imgs) + len(lbls)}")

    n_img = sum(len(list((OUT / s / "images").glob("*.*"))) for s in splits)
    n_lbl = sum(len(list((OUT / s / "labels").glob("*.txt"))) for s in splits)
    print(f"完成：{n_img} 张图 / {n_lbl} 个标注 -> {OUT}")
    return 0 if n_img and n_lbl else 1


if __name__ == "__main__":
    raise SystemExit(main())
