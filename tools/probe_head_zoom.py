# -*- coding: utf-8 -*-
"""
局部放大复核可行性探针
======================

v3.0.0 的 10 例误报里，6 例是「框在头部区域内、但 conf 不到 0.59」。
这些人的 person 框宽度只有 44–93 px，头在 640 输入下只有 20–30 px——
模型不是看不见，是看得太模糊。

直接提高全图推理分辨率（960/1280）已被否掉：模型在 640 上训练，
放大推理等于把目标推出训练分布，误报反而涨到 18%/34%。

那么换个问法：**只把「未佩戴」那个人的头裁出来、放大到 640 再跑一次**，
置信度会不会上去？

本脚本只做可行性验证，不接主线：
  - 输入：data/fp_attribution.json 里的 10 例误报
  - 处理：裁 person 头区（区域外扩 1.8 倍）→ resize 640 → 单模型推理
  - 输出：原置信度 vs 放大后置信度

判据：若多例能从 <0.59 抬到 >=0.59，就值得做成两级判定。

    python tools/probe_head_zoom.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anxun.rules_engine as R
from anxun.rules_engine import Box

ROOT = Path(__file__).resolve().parents[1]
ATTR = ROOT / "data" / "fp_attribution.json"
OUT = ROOT / "data" / "head_zoom_probe.json"


def main() -> int:
    from PIL import Image
    from ultralytics import YOLO

    if not ATTR.exists():
        print(f"缺少 {ATTR}，先跑 tools/attr_fp.py")
        return 2
    d = json.loads(ATTR.read_text(encoding="utf-8"))

    model = YOLO(str(ROOT / "models" / "best.pt"))
    names = model.names  # {idx: name}
    print(f"模型类别：{list(names.values())}")

    rows = []
    for i, c in enumerate(d["cases"]):
        p = c["person"]
        person = Box("person", c["person_conf"], p["x1"], p["y1"], p["x2"], p["y2"])
        # 头部区域：判定口径的 0.26/0.70，再外扩留上下文
        # 必须裁正方形：非等比拉伸会改变头的形状，模型在 640 上训的是自然比例
        img_path = ROOT / c["image"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            region = person.top_region(R.HEAD_REGION_RATIO, R.HEAD_WIDTH_RATIO)
            cx, cy = region.cx, region.cy
            side = max(region.w, region.h) * 2.2
            half = side / 2
            x1 = max(0, int(cx - half))
            y1 = max(0, int(cy - half))
            x2 = min(W, int(cx + half))
            y2 = min(H, int(cy + half))
            crop = im.crop((x1, y1, x2, y2))
            if crop.width < 8 or crop.height < 8:
                continue
            crop = crop.resize((640, 640), Image.BICUBIC)
        res = model.predict(crop, imgsz=640, conf=0.25, verbose=False)[0]

        best = None
        for b in res.boxes:
            cls = names[int(b.cls)]
            if cls.lower() != "helmet":
                continue
            conf = float(b.conf)
            if best is None or conf > best:
                best = conf

        rows.append({
            "case": i,
            "cause": c["cause"],
            "person_w": round(p["w"]),
            "person_h": round(p["h"]),
            "conf_full_image": c.get("best_conf_in_region"),
            "conf_head_zoom": round(best, 3) if best is not None else None,
        })
        old = c.get("best_conf_in_region")
        new = rows[-1]["conf_head_zoom"]
        flag = ""
        if new is not None and new >= R.CONF_FINAL and (old is None or old < R.CONF_FINAL):
            flag = "  ← 翻案"
        print(f"case{i:<3}{c['cause']:<4} person {p['w']:>4.0f}x{p['h']:<4.0f} "
              f"全图 {str(old):>6}  放大 {str(new):>6}{flag}")

    out = {"kind": "head-zoom-probe", "conf_final": R.CONF_FINAL,
           "crop_scale": 1.8, "rows": rows}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    saved = sum(1 for r in rows
                if r["conf_head_zoom"] is not None and r["conf_head_zoom"] >= R.CONF_FINAL
                and (r["conf_full_image"] is None or r["conf_full_image"] < R.CONF_FINAL))
    print(f"\n{len(rows)} 例里 {saved} 例能被放大救回（conf >= {R.CONF_FINAL}）")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
