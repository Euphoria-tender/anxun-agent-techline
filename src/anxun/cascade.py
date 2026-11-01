# -*- coding: utf-8 -*-
"""
安巡 Agent · 两级级联感知
=========================

问题：安全帽是个小目标。在 640×640 的输入里，一个中等距离的人只有 100~200px 高，
头部区域（上 30%）只有 30~60px，扣掉安全帽本身（约头宽的 60%），实际像素只有 20~40px。
模型在这个尺度上分不清「帽子」和「头发/安全帽颜色相近的背景」——实测 65% 的漏报属于这一类。

做法：两级级联。

    L1  全图一次前向：人体模型定位所有人 + PPE 模型给出全图装备框（粗）
    L2  只对「头部低于 head_px」的人，裁出头部 ROI，放大到 head_imgsz 再跑一次 PPE（精）
    L3  精查结果替换 L1 在该 ROI 内的装备框

代价是可控的：只对小人做精查，且多张 ROI 合成一个 batch 一次前向。
一帧里 5 个人通常只有 2~3 个需要精查，批量跑约 8~12 ms（RTX 5060）。

为什么不是直接把全图 imgsz 提到 960：
    模型在 640 上训的，全图放大等于把目标推出训练分布，实测掉 3.4 个点、慢 2.2 倍（见 sweep.py）。
    级联只放大「需要看清的那一小块」，等效分辨率上去了，但整个画面仍在训练分布内。
"""

from __future__ import annotations

import time
from pathlib import Path

from .rules_engine import Box, HEAD_REGION_RATIO, HEAD_WIDTH_RATIO


def _read_bgr(path: str | Path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"读不到图片：{path}")
    return img


class CascadeDetector:
    """
    在任意检测器外面套一层「头部精查」。

        det = CascadeDetector(DualDetector(person, ppe), head_px=72)

    base 只要求实现 detect_image / detect_frame 并返回 {"boxes", "infer_ms"}。
    """

    name = "cascade"
    kind = "model"

    def __init__(self, base, head_px: float = 72.0, pad: float = 1.9,
                 head_imgsz: int = 320, conf: float = 0.15,
                 max_batch: int = 16, accept_pad: float = 1.15,
                 mode: str = "replace"):
        self.base = base
        # 对外暴露 person / ppe，供上层读设备、权重名
        self.person = getattr(base, "person", base)
        self.ppe = getattr(base, "ppe", base)
        self.head_px = head_px          # 头部高度低于此值（像素）才精查
        self.pad = pad                  # ROI 相对头部区域的放大倍数（给模型上下文）
        self.head_imgsz = head_imgsz    # 精查时的输入尺寸
        self.conf = conf                # 精查用更低阈值，交给规则引擎按置信度分档
        self.max_batch = max_batch
        self.accept_pad = accept_pad    # 精查框中心须落在头部区域×此倍数内才采纳
        # replace：ROI 内的旧框被精查结果替换（信精查）
        # union  ：旧框保留，精查结果追加（信证据，交给规则引擎按 IoU 择优）
        self.mode = mode

    # ---------------- 对外 ----------------

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        base = self.base.detect_image(image_path, frame_idx)
        return self._refine(base, _read_bgr(image_path))

    def detect_frame(self, frame) -> dict:
        base = self.base.detect_frame(frame)
        return self._refine(base, frame)

    def detect(self, scene: dict, frame_idx: int = 0) -> dict:
        return self.detect_image(scene["image"], frame_idx)

    # ---------------- 内部 ----------------

    @staticmethod
    def _head_region(p: Box) -> tuple[float, float, float, float]:
        r = p.top_region(HEAD_REGION_RATIO, HEAD_WIDTH_RATIO)
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        return cx, cy, r.w, r.h

    def _roi(self, p: Box, W: int, H: int):
        """以头部中心为基准切一个正方形 ROI，给模型足够上下文。"""
        cx, cy, hw, hh = self._head_region(p)
        side = max(hw, hh, 12.0) * self.pad
        x1 = max(0.0, cx - side / 2)
        y1 = max(0.0, cy - side / 2)
        x2 = min(float(W), cx + side / 2)
        y2 = min(float(H), cy + side / 2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return x1, y1, x2, y2

    def _accept(self, p: Box, cx: float, cy: float) -> bool:
        """精查框的中心是否落在这个人的头部附近。防止把邻边人的帽子抢过来。"""
        hx, hy, hw, hh = self._head_region(p)
        return (abs(cx - hx) <= hw * self.accept_pad / 2
                and abs(cy - hy) <= hh * self.accept_pad / 2)

    def _refine(self, base: dict, img) -> dict:
        t0 = time.perf_counter()
        H, W = img.shape[:2]
        persons = [b for b in base["boxes"] if b.label == "person"]
        ppe = [b for b in base["boxes"] if b.label != "person"]

        jobs = []   # (person, roi)
        for p in persons:
            if p.h * HEAD_REGION_RATIO >= self.head_px:
                continue                      # 本来就够大，不必精查
            roi = self._roi(p, W, H)
            if roi:
                jobs.append((p, roi))

        info = {"refined": len(jobs), "persons": len(persons), "boxes_before": len(ppe),
                "boxes_after": len(ppe), "ms": 0.0}
        if not jobs:
            base = dict(base)
            base["cascade"] = info
            return base

        crops, spans = [], []
        for p, (x1, y1, x2, y2) in jobs:
            crops.append(img[int(y1):int(y2), int(x1):int(x2)])
            spans.append((p, x1, y1, x2, y2))

        # 批量前向：一次喂多张 ROI，比逐张省 2~3 倍
        refined: list[Box] = []
        for s in range(0, len(crops), self.max_batch):
            chunk = crops[s:s + self.max_batch]
            results = self.ppe.model.predict(
                chunk, imgsz=self.head_imgsz, conf=self.conf,
                verbose=False, device=getattr(self.ppe, "device", "") or None)
            for (p, ox, oy, _, _), res in zip(spans[s:s + self.max_batch], results):
                for xyxy, c, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                       res.boxes.conf.tolist()):
                    raw = str(self.ppe.names[int(c)])
                    label = self.ppe.LABEL_MAP.get(raw)
                    if label not in ("helmet", "nohelmet"):
                        continue
                    bx1, by1, bx2, by2 = (xyxy[0] + ox, xyxy[1] + oy,
                                          xyxy[2] + ox, xyxy[3] + oy)
                    if not self._accept(p, (bx1 + bx2) / 2, (by1 + by2) / 2):
                        continue
                    refined.append(Box(label, float(cf), bx1, by1, bx2, by2))

        # 替换：干掉落在任一 ROI 内的旧装备框，换成精查结果
        def in_any_roi(b: Box) -> bool:
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            for _, x1, y1, x2, y2 in spans:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return True
            return False

        kept = [b for b in ppe if not (b.label in ("helmet", "nohelmet") and in_any_roi(b))]
        if self.mode == "union":
            kept = ppe                     # 保留原有证据，精查只做加法
        out = dict(base)
        out["boxes"] = [b for b in base["boxes"] if b.label == "person"] + kept + refined
        out["infer_ms"] = round(base["infer_ms"] + (time.perf_counter() - t0) * 1000, 2)
        out["backend"] = self.name
        info.update(boxes_after=len(kept) + len(refined),
                    ms=round((time.perf_counter() - t0) * 1000, 2))
        out["cascade"] = info
        return out
