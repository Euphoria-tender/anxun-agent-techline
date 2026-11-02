"""直接测「一个人走过半秒」时，全幅帧差到底是多少。

这是 v2.1.0 把 --motion 从 10.0 降到 3.0 的依据。跑法：

    python tools/probe_frame_diff.py


为什么不从视频里测
------------------
- 原来的 site_motion（贴人像）检测不稳，已废弃；
- 现在的 site_pan（整图平移）整幅画面都在动，帧差 P50=32.9，远高于真实场景；
- 真实厂区是**固定机位、背景静止、人移动**，帧差只来自人覆盖的那一小块。

所以这里不走视频，直接在真实照片上做「把 person 区域平移一段」的差分：
两帧 = 原图 + 人像平移后的图。背景不变，只有人动，这就是真实场景的帧差来源。

位移取「框宽 × 1.18」——正常步速 1.2 m/s、0.5 秒位移 0.6 m，
人宽约 0.5 m，两者相除约 1.18 倍框宽。这个比值也和几何推导一致：
位移 106 px / 框宽 90 px ≈ 1.18。

输出：各分位数 + 阈值命中率，直接回答「10.0 会不会跳过走动帧」。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "datasets/ppe_test/test/images"
LBL = ROOT / "datasets/ppe_test/test/labels"
CLS_PERSON = 7
SHIFT_RATIO = 1.18          # 位移 / 框宽
SAMPLE = (160, 160)         # stream.py 的帧差用 160x160 灰度均值


def persons_of(label_path: Path, w: int, h: int):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5 or int(p[0]) != CLS_PERSON:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                    int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return out


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(cv2.resize(a, SAMPLE), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, SAMPLE), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def main() -> int:
    diffs, ratios = [], []
    for img_path in sorted(IMG.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = persons_of(LBL / (img_path.stem + ".txt"), w, h)
        if not boxes:
            continue
        # 每张图只取最大的那个人，避免多人叠加把帧差抬高
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        bw = x2 - x1
        shift = int(round(bw * SHIFT_RATIO))
        if shift < 8 or x1 + shift + bw > w:
            continue
        cur = img.copy()
        # 背景抹平（用周围像素填），再把同一块人像贴到右移后的位置
        patch = img[y1:y2, x1:x2].copy()
        cur[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (31, 31), 0)
        cur[y1:y2, x1 + shift:x2 + shift] = patch
        diffs.append(frame_diff(img, cur))
        ratios.append(shift / w)

    d = np.array(diffs)
    print(f"样本 {len(d)} 张（每张取最大的一人，位移 = 框宽 × {SHIFT_RATIO}）")
    print(f"位移占画面宽度 P50 = {np.percentile(ratios, 50):.1%}")
    print(f"帧差  P10 {np.percentile(d,10):.2f} | P50 {np.percentile(d,50):.2f} | "
          f"P90 {np.percentile(d,90):.2f} | max {d.max():.2f}")
    for th in (3.0, 5.0, 10.0):
        print(f"  阈值 {th:>4}：会跳过 {float((d < th).mean()):>6.1%} 的走动帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
