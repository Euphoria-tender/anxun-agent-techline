# -*- coding: utf-8 -*-
"""
生成一段「接近现场」的监控视频：固定机位，绝大部分时间是空画面。

为什么要这个：demo.mp4 里每一帧都有人，只能验证「有人时判得对不对」，
验证不了「没人时省不省」。真实车间摄像头 80% 以上的时间是空的——
人员门控（没人不跑 PPE）和帧差跳帧（不动不推理）的价值都在这里。

    python tools/make_site_video.py --out datasets/site_demo.mp4 --seconds 60 --occupy 0.15

场景假设写在输出里：占用率 15%（两个 6 秒的人员通过窗口），这是小样本车间白班的典型值。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "datasets" / "ppe_test" / "test" / "images"


def background(w: int, h: int) -> np.ndarray:
    """画一个空车间：水泥地 + 地脚线 + 两台设备 + 天花板灯管。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (86, 88, 90)
    cv2.rectangle(img, (0, int(h * 0.62)), (w, h), (108, 108, 106), -1)   # 地面
    cv2.line(img, (0, int(h * 0.62)), (w, int(h * 0.62)), (70, 70, 70), 3)
    for x in (int(w * 0.14), int(w * 0.62)):
        cv2.rectangle(img, (x, int(h * 0.36)), (x + int(w * 0.16), int(h * 0.62)),
                      (64, 72, 80), -1)
        cv2.rectangle(img, (x + 6, int(h * 0.38)), (x + int(w * 0.16) - 6, int(h * 0.46)),
                      (44, 52, 60), -1)
    for x in range(int(w * 0.08), w, int(w * 0.28)):
        cv2.rectangle(img, (x, 12), (x + int(w * 0.18), 26), (150, 150, 145), -1)
    cv2.putText(img, "BAY-01", (24, int(h * 0.58)), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (120, 120, 118), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets" / "site_demo.mp4"))
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--occupy", type=float, default=0.15, help="有人出现的时长占比")
    ap.add_argument("--size", default="1280x720")
    args = ap.parse_args()

    w, h = (int(x) for x in args.size.lower().split("x"))
    frames = args.seconds * args.fps
    bg = background(w, h)
    rng = np.random.default_rng(7)

    files = sorted(POOL.glob("*.jpg"))
    if not files:
        print(f"缺少素材：{POOL}")
        return 2

    # 占用窗口：按占用率均分到时间轴上
    win = max(1, int(round(frames * args.occupy / 2)))
    centers = [int(frames * 0.22), int(frames * 0.62)]
    occupied = set()
    for c in centers:
        occupied.update(range(max(0, c - win // 2), min(frames, c + win // 2)))

    out = Path(args.out)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    k = 0
    for i in range(frames):
        frame = bg.copy()
        # 轻微的传感器噪声：让静止画面「几乎不变但不完全一样」，更贴近真实
        noise = rng.integers(-2, 3, (h, w, 1), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if i in occupied:
            img = cv2.imread(str(files[k % len(files)]))
            k += 1
            if img is not None:
                # 贴到画面中下部，模拟人走进来（尺寸不缩放，保持原始分辨率）
                ih, iw = img.shape[:2]
                x = (w - iw) // 2
                y = max(0, int(h * 0.62) - ih)
                frame[y:y + min(ih, h - y), x:x + min(iw, w - x)] = \
                    img[:min(ih, h - y), :min(iw, w - x)]

        vw.write(frame)
    vw.release()

    real = len(occupied) / frames
    print(f"-> {out}  {frames} 帧 / {args.seconds}s @ {args.fps}fps")
    print(f"   有人帧 {len(occupied)}（设计占用率 {args.occupy:.0%}，实际 {real:.0%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
