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
import time
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
    """拉全量文件清单。14072 个文件，JSON 有几 MB——urllib 的 timeout 是按单次 read 算的，
    大响应会在 read 阶段超时（实测），改用 curl 一次取完 + 重试。"""
    last = ""
    for i in range(3):
        r = subprocess.run(["curl", "-sL", "-m", "180", API], capture_output=True)
        if r.returncode == 0 and r.stdout:
            try:
                d = json.loads(r.stdout.decode("utf-8", errors="ignore"))
                sib = [s["rfilename"] for s in d.get("siblings", [])]
                if sib:
                    return sib
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
        else:
            last = f"curl rc={r.returncode}"
        print(f"  清单第 {i + 1} 次失败（{last}），重试 ...")
        time.sleep(3)
    raise RuntimeError(f"拉文件清单失败：{last}")


# 429 限流时 hf-mirror 返回的是**HTML 错误页**（HTTP 200 之外的 429），
# 旧的 fetch 只判文件大小、fetch_text 只判 rc==0——都会把错误页当成功：
#   - 图片：HTML 被存成 .jpg，PIL 打不开
#   - 标注：HTML 按行解析，每行字段不够被跳过，最后写出**空文件**，看起来像负样本
# 实测 706 张里有 307 张就是这样被写成空标注的。所以这里必须校验状态码 + 内容特征。
_HTML_MARKS = ("<!DOCTYPE", "<HTML", "访问频率限制", "Error 429")


def _looks_like_html(text: str) -> bool:
    head = text[:400].upper()
    return any(m in head for m in _HTML_MARKS)


def fetch(rel: str, dst: Path) -> bool:
    """下图片。先落 .tmp，校验确为 JPEG/PNG 再原子替换，避免 429 错误页混进来。"""
    if dst.exists() and dst.stat().st_size > 1024:
        with open(dst, "rb") as f:
            if f.read(3) in (b"\xff\xd8\xff", b"\x89PN"):
                return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for i in range(6):
        r = subprocess.run(["curl", "-sL", "-m", "90", "-o", str(tmp),
                            f"{BASE}/{rel}"], capture_output=True)
        if tmp.exists() and tmp.stat().st_size > 1024:
            with open(tmp, "rb") as f:
                magic = f.read(3)
            if magic in (b"\xff\xd8\xff", b"\x89PN"):
                tmp.replace(dst)
                return True
        tmp.unlink(missing_ok=True)
        time.sleep(1.5 * (i + 1))          # 429 退避
    return False


def fetch_text(rel: str) -> str | None:
    """拉标注内容到内存。返回 None = 拉取失败（调用方绝不能把它当空标注写盘）。"""
    for i in range(6):
        r = subprocess.run(["curl", "-sL", "-m", "60", "-w", "\n%{http_code}",
                            f"{BASE}/{rel}"], capture_output=True)
        out = r.stdout.decode("utf-8", errors="ignore")
        body, _, code = out.rpartition("\n")
        code = code.strip()
        if code == "200" and not _looks_like_html(body):
            return body
        time.sleep(1.5 * (i + 1))
    return None


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
        # 已有非空标注直接复用（重跑只为补失败的那批，不重复消耗配额）
        if dst.exists() and dst.stat().st_size > 0:
            return len([l for l in dst.read_text(encoding="utf-8").splitlines()
                        if len(l.split()) >= 5])
        txt = fetch_text(rel)
        if txt is None:                 # 拉取失败：绝不写空文件冒充负样本
            return -1
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
    boxes = empty = lbl_fail = 0
    failed_stems: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, pr): pr for pr in pairs}
        for fu in as_completed(futs):
            n = fu.result()
            if n < 0:
                lbl_fail += 1
                failed_stems.add(futs[fu][0])
            else:
                boxes += n
                empty += (0 if n else 1)
    # 远端**确实没有**标注文件的图，补空文件（YOLO 会当负样本）；
    # 拉取失败的图不补，让它们留空以便下次重跑时重拉。
    for f in imgs:
        stem = Path(f).stem
        lp = OUT / split / "labels" / f"{stem}.txt"
        if not lp.exists() and stem not in failed_stems:
            lp.write_text("", encoding="utf-8")
            empty += 1
    return {"images": ok, "failed": fail, "boxes": boxes, "empty": empty,
            "labels_failed": lbl_fail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每个分片最多取几张（0=全部）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--no-yaml", action="store_true",
                    help="只拉分片、不写 data.yaml（单拉 test 时用，避免写出指向不存在 train/val 的 yaml）")
    args = ap.parse_args()

    print("拉取文件清单 ...")
    files = siblings()
    print(f"仓库共 {len(files)} 个文件")

    summary = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        st = do_split(split, files, args.limit, args.workers)
        summary[split] = st
        print(f"{split:<6} 图 {st['images']:>5}（失败 {st['failed']}）  "
              f"框 {st['boxes']:>6}  空标注 {st['empty']}  "
              f"标注拉取失败 {st.get('labels_failed', 0)}")

    if args.no_yaml:
        print("\n--no-yaml：跳过 data.yaml")
        return 0

    yaml = OUT / "data.yaml"
    # 不写 path：ultralytics 以 yaml 所在目录为根，路径相对它即可，换机器不用改
    yaml.write_text(
        f"train: train/images\nval: val/images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n", encoding="utf-8")
    print(f"\n-> {yaml}")
    print("下一步：python src/anxun/finetune.py --data datasets/helmet_p2/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
