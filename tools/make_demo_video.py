# -*- coding: utf-8 -*-
"""
生成一段模拟监控视频，用于验证视频巡查链路。
素材是 datasets/scenes 里的 4 张现场照片，每张停留 4 秒，叠时间戳。

    python tools/make_demo_video.py -> datasets/demo.mp4

说明：这是**验证流水线用**的合成流，不是真实现场录像。
真实现场只需要把 --source 换成摄像头 RTSP 地址或录像文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "datasets" / "scenes"
OUT = ROOT / "datasets" / "demo.mp4"
FPS = 10
HOLD_S = 4


def main() -> int:
    meta = json.loads((SCENES / "scenes.json").read_text(encoding="utf-8"))
    files = [(s["scene_id"], SCENES / s["file"]) for s in meta if (SCENES / s["file"]).exists()]
    if not files:
        print("没有场景图片")
        return 1

    frame0 = cv2.imread(str(files[0][1]))
    h, w = frame0.shape[:2]
    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    n = 0
    for sid, p in files:
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        for k in range(FPS * HOLD_S):
            frame = img.copy()
            t = n / FPS
            cv2.putText(frame, f"CAM-DEMO  {t:05.1f}s  {sid}", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"CAM-DEMO  {t:05.1f}s  {sid}", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(frame)
            n += 1
    vw.release()
    print(f"{OUT}  {n} 帧  {n / FPS:.1f}s  {w}x{h}  ({len(files)} 个场景 × {HOLD_S}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
