# -*- coding: utf-8 -*-
"""
第二数据源标注体检（706 张，2 类）
==================================

只做数值核对，不改任何文件：框数、类别比、框几何、越界、退化框、空标注。
目的是在跑指标之前先知道「标注能不能用」——标注坏了，AP 掉点就分不清是模型的锅还是数据的锅。

    python tools/probe_second_source_labels.py
    python tools/probe_second_source_labels.py --dump 8      # 另外导出 8 张带框图供肉眼看
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "helmet_p2" / "test"
NAMES = ["helmet", "nohelmet"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--dump", type=int, default=0, help="导出 N 张带框预览图到 _audit/")
    args = ap.parse_args()

    data = Path(args.data)
    img_dir = data / "images" if (data / "images").exists() else data
    lbl_dir = data / "labels" if (data / "labels").exists() else data

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        print(f"没有图片：{img_dir}")
        return 2

    cls_count = {n: 0 for n in NAMES}
    per_img = []
    widths, heights, ratios, areas = [], [], [], []
    oob = degenerate = empty = missing = 0
    problems = []

    for fp in imgs:
        lp = lbl_dir / (fp.stem + ".txt")
        if not lp.exists():
            missing += 1
            per_img.append(0)
            continue
        n = 0
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            try:
                c = int(float(p[0]))
                cx, cy, bw, bh = (float(x) for x in p[1:5])
            except ValueError:
                continue
            if c < 0 or c >= len(NAMES):
                problems.append(f"{fp.name}: 类别 {c} 越界")
                continue
            cls_count[NAMES[c]] += 1
            n += 1
            widths.append(bw)
            heights.append(bh)
            areas.append(bw * bh)
            if bh > 0:
                ratios.append(bw / bh)
            if bw <= 0.002 or bh <= 0.002:
                degenerate += 1
            if cx - bw / 2 < -0.01 or cy - bh / 2 < -0.01 \
                    or cx + bw / 2 > 1.01 or cy + bh / 2 > 1.01:
                oob += 1
        if n == 0:
            empty += 1
        per_img.append(n)

    def q(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        i = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
        return round(ys[i], 4)

    total = sum(cls_count.values())

    # 图片尺寸：Roboflow 导出常把原图缩到统一边长，若被拉伸，归一化框还是对的，
    # 但检测器看到的人形会变形——这直接影响 AP，必须先量出来。
    from PIL import Image
    sizes = []
    for fp in imgs[:400]:
        try:
            with Image.open(fp) as im:
                sizes.append(im.size)
        except Exception:
            pass
    from collections import Counter
    size_top = Counter(f"{w}x{h}" for w, h in sizes).most_common(5)

    report = {
        "images": len(imgs),
        "image_sizes_top": size_top,
        "labels_missing": missing,
        "images_empty": empty,
        "boxes_total": total,
        "per_class": cls_count,
        "helmet_ratio": round(cls_count["helmet"] / total, 4) if total else None,
        "boxes_per_image": {
            "min": min(per_img) if per_img else 0,
            "p50": round(statistics.median(per_img), 1) if per_img else 0,
            "max": max(per_img) if per_img else 0,
            "mean": round(statistics.fmean(per_img), 2) if per_img else 0,
            "zero": sum(1 for n in per_img if n == 0),
        },
        "box_geometry": {
            "w_p50": q(widths, 0.5), "w_p95": q(widths, 0.95),
            "h_p50": q(heights, 0.5), "h_p95": q(heights, 0.95),
            "area_p50": q(areas, 0.5),
            "aspect_p50": q(ratios, 0.5),
        },
        "boxes_out_of_bounds": oob,
        "boxes_degenerate": degenerate,
        "problems": problems[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.dump:
        from PIL import Image, ImageDraw
        outdir = ROOT.parent.parent / "_audit" / "second_source"
        outdir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(imgs) // args.dump)
        picked = imgs[::step][:args.dump]
        for fp in picked:
            with Image.open(fp) as im:
                im = im.convert("RGB")
                W, H = im.size
                d = ImageDraw.Draw(im)
                lp = lbl_dir / (fp.stem + ".txt")
                if lp.exists():
                    for line in lp.read_text(encoding="utf-8").splitlines():
                        p = line.split()
                        if len(p) < 5:
                            continue
                        c = int(float(p[0]))
                        cx, cy, bw, bh = (float(x) for x in p[1:5])
                        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                        col = (0, 200, 0) if NAMES[c] == "helmet" else (255, 140, 0)
                        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
                        d.text((x1 + 3, max(0, y1 - 12)), NAMES[c], fill=col)
                im.save(outdir / f"{fp.stem}.jpg", quality=85)
        print(f"\n-> {len(picked)} 张带框预览：{outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
