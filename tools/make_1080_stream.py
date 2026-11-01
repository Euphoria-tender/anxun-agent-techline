"""把 1080p 真实照片素材串成监控流视频，用于复验三级触发层。

背景
----
三级触发（抽帧 / 帧差 / 人员门控）此前只在 `site_demo.mp4` 上验证过，
那是合成的动画画面（640x640、15% 占用率）。真实部署是 1080p 摄像头，
两个变量都变了：分辨率（压缩噪声量级）和画面内容（真实照片、无规律运动）。
阈值 3.0 是按合成动画定的，在 1080p 上未必还成立——本工具就是为复验这个。

产出两个视频
-----------
1. `real1080.mp4`        静态切换：每张照片停留 15 帧（25fps 下 0.6s）。
                          模拟"固定机位先后看到的不同场景"，帧差只在切换瞬间非零。
2. `real1080_jitter.mp4` 同上，但每帧加 ±1.5px 随机平移。
                          真实固定机位一定有轻微晃动（风、振动），
                          帧差不会恒为 0——这是更接近现场的版本。

标注不丢：YOLO 标注是归一化坐标，与分辨率无关，直接用 `datasets/ppe_test/test/labels`。
告警一致性因此可以按 evaluate_real 的真值口径核对，这是合成动画做不到的。

用法
----
    python tools/make_1080_stream.py [--src DIR] [--hold 15] [--jitter 1.5]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "datasets" / "ppe_test" / "test1080" / "images"
OUT = ROOT / "datasets"

FPS = 25
W, H = 1920, 1920


def _read(p: Path) -> np.ndarray:
    """cv2.imread 在 Windows 非 ASCII 路径（用户名含中文）会静默失败。"""
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图片：{p}")
    return img


def jitter(img: np.ndarray, rnd: random.Random, amp: float) -> np.ndarray:
    """随机平移 + 轻微亮度扰动，模拟固定机位的固有晃动与压缩噪声。"""
    dx, dy = rnd.uniform(-amp, amp), rnd.uniform(-amp, amp)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         borderMode=cv2.BORDER_REPLICATE)
    # 亮度 ±2/255：模拟真实编码噪声，让帧差不为 0 但很小
    gain = 1.0 + rnd.uniform(-0.008, 0.008)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def build(src: Path, hold: int, amp: float, out: Path, n_images: int = 0) -> dict:
    files = sorted(src.glob("*.jpg"))
    if not files:
        raise RuntimeError(f"{src} 里没有图片")
    if n_images:
        files = files[:n_images]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"打不开输出视频：{out}")

    rnd = random.Random(1080)
    for f in files:
        img = _read(f)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        for _ in range(hold):
            frame = jitter(img, rnd, amp) if amp > 0 else img
            vw.write(frame)
    vw.release()

    frames = len(files) * hold
    return {"images": len(files), "frames": frames, "seconds": frames / FPS,
            "size_mb": round(out.stat().st_size / 1e6, 1), "jitter_px": amp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--images", type=int, default=20,
                    help="取前 N 张（按文件名排序，已覆盖 1~6 人分布）")
    ap.add_argument("--hold-sec", type=float, default=5.0,
                    help="每张照片停留秒数。真实固定机位一个画面通常持续数分钟，"
                         "5 秒是保守下限；第一版用 0.6 秒，结果更像幻灯片，测不出触发层")
    ap.add_argument("--jitter", type=float, default=1.5, help="随机平移幅度(px)，0=纯静态")
    args = ap.parse_args()

    hold = max(1, int(round(FPS * args.hold_sec)))
    src = Path(args.src)
    for name, amp in ((f"real1080.mp4", 0.0), (f"real1080_jitter.mp4", args.jitter)):
        info = build(src, hold, amp, OUT / name, args.images)
        tag = "纯静态切换" if amp == 0 else f"±{amp}px 机位抖动"
        print(f"{name}  {tag}  {info['images']} 张 × {hold} 帧 = "
              f"{info['frames']} 帧 / {info['seconds']:.1f}s  {info['size_mb']} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
