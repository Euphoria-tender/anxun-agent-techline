# -*- coding: utf-8 -*-
"""
补齐 helmet_p2 缺失/空白的标注文件
===================================

背景：并发拉取被 hf-mirror 限流时会静默失败（curl 返回 0 但 body 为空），
于是留下大量「有图无标注」和「空标注」。这个脚本按图反查标注，缺的、空的都补。

    python tools/repair_labels.py

判定标准：文件不存在，或存在但解析不出任何有效框，都算需要补。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "helmet_p2"
REPO = "toborobot/Safety-helmet-dataset"
API = f"https://hf-mirror.com/api/datasets/{REPO}"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main"
REMAP = {1: 0, 0: 1}          # 1=戴帽 -> 0(helmet)；0=没戴 -> 1(nohelmet)
SPLIT_DIR = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

# 429 时镜像返回 HTML 错误页。不校验就会 parse 出 0 行，被当成「真空标注」写盘。
_HTML_MARKS = ("<!DOCTYPE", "<HTML", "访问频率限制", "Error 429")


def _looks_like_html(text: str) -> bool:
    head = text[:400].upper()
    return any(m in head for m in _HTML_MARKS)


def siblings() -> dict[str, str]:
    req = urllib.request.Request(API, headers={"User-Agent": "anxun/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=40))
    return {Path(s["rfilename"]).stem: s["rfilename"]
            for s in d.get("siblings", []) if s["rfilename"].endswith(".txt")}


def parse(body: str) -> list[str]:
    rows = []
    for line in body.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = int(float(p[0]))
        except ValueError:
            continue
        if c in REMAP:
            rows.append(f"{REMAP[c]} {' '.join(p[1:5])}")
    return rows


def fetch_one(rel: str, tries: int = 5) -> list[str] | None:
    """返回 None = 拉取失败（不能写盘）；返回 [] = 远端确实是空标注（负样本）。"""
    for i in range(tries):
        r = subprocess.run(["curl", "-sL", "-m", "45", "-w", "\n%{http_code}",
                            f"{BASE}/{rel}"], capture_output=True)
        out = r.stdout.decode("utf-8", "ignore")
        body, _, code = out.rpartition("\n")
        if code.strip() == "200" and not _looks_like_html(body):
            return parse(body)
        time.sleep(0.3 * (i + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3,
                    help="并发数。镜像限流，3 是稳妥值；调大只会得到更多空文件")
    args = ap.parse_args()

    files = siblings()
    print(f"仓库标注文件 {len(files)} 个")

    # 本地目录名和仓库分片名可能不同（我们叫 valid，仓库也叫 valid；但脚本参数里写 val）
    local = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    for split in ("train", "valid", "test"):
        sd = SPLIT_DIR[split]
        img_dir = OUT / split / "images"
        lbl_dir = OUT / split / "labels"
        if not img_dir.exists():
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)

        todo = []
        for img in sorted(img_dir.glob("*.jpg")):
            lp = lbl_dir / f"{img.stem}.txt"
            if not lp.exists() or lp.stat().st_size == 0:
                rel = files.get(img.stem)
                if rel:
                    todo.append((rel, lp))

        print(f"{split}: 待补 {len(todo)} 个标注")
        fixed = failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_one, rel): (rel, lp) for rel, lp in todo}
            for i, fu in enumerate(as_completed(futs)):
                rel, lp = futs[fu]
                rows = fu.result()
                if rows is not None:
                    lp.write_text("\n".join(rows), encoding="utf-8")
                    fixed += 1
                else:
                    failed += 1
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)}  已补 {fixed} 失败 {failed}")

        fs = list(lbl_dir.glob("*.txt"))
        ne = [f for f in fs if f.stat().st_size > 0]
        tot = sum(len(f.read_text().splitlines()) for f in ne)
        print(f"{split}: 标注 {len(fs)}，非空 {len(ne)}，总框 {tot}，本次补 {fixed}")

    print(f"\n-> {OUT / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
