# -*- coding: utf-8 -*-
"""
安巡 Agent · 多模型证据融合（WBF）
==================================

现状：单个头盔模型在 640 全图上 helmet AP 0.579，漏报 24.7%。
换更大的模型（yolov8m）掉 14 个点，微调后 AP 涨 32% 但业务指标反而跌——
原因都是「单点判断」：一个模型说戴就是戴，说没戴就是没戴，错一次没法纠正。

做法：多个模型各自出框，按 WBF（Weighted Boxes Fusion）合并成一组框，再交规则引擎。
    同一位置多个模型都检出 -> 置信度上调（共识）
    只有某个模型检出      -> 保留但按该模型权重打折（孤证）

这是「感知层多证据」，和规则引擎里的「正反证据」是两层不同的东西：
    前者解决「看见没看见」，后者解决「看见了算不算」。

参考：Solovyev et al., Weighted Boxes Fusion, 2021
"""

from __future__ import annotations

import time
from pathlib import Path

from .detectors import YoloDetector, pick_device
from .rules_engine import Box, iou


def _fuse_clusters(clusters: list[list[tuple[Box, float]]],
                   iou_thr: float, conf_boost: float) -> list[Box]:
    """
    clusters 里每个簇是 [(box, weight), ...]。
    坐标按置信度×模型权重加权平均；置信度取加权均值，多模型共识时上调。
    """
    out: list[Box] = []
    for items in clusters:
        n = len(items)
        tw = sum(w for _, w in items) or 1.0
        x1 = sum(b.x1 * b.conf * w for b, w in items) / sum(b.conf * w for b, w in items)
        y1 = sum(b.y1 * b.conf * w for b, w in items) / sum(b.conf * w for b, w in items)
        x2 = sum(b.x2 * b.conf * w for b, w in items) / sum(b.conf * w for b, w in items)
        y2 = sum(b.y2 * b.conf * w for b, w in items) / sum(b.conf * w for b, w in items)
        conf = sum(b.conf * w for b, w in items) / tw
        if n > 1:
            conf = min(0.99, conf * (1.0 + conf_boost * (n - 1)))
        out.append(Box(items[0][0].label, float(conf), x1, y1, x2, y2))
    return out


class FusionDetector:
    """
    人体模型 + 多个 PPE 模型 → WBF 融合。

        det = FusionDetector("models/yolov8n.pt",
                             [("models/best.pt", 1.0),
                              ("runs/finetune/weights/best.pt", 0.7)])
    """

    name = "fusion"
    kind = "model"

    def __init__(self, person_weights: str, ppe_specs: list[tuple[str, float]],
                 imgsz: int = 640, conf: float = 0.25, device: str = "",
                 iou_thr: float = 0.55, conf_boost: float = 0.06):
        dev = pick_device(device)
        self.person = YoloDetector(person_weights, imgsz=imgsz, conf=conf, device=dev)
        self.ppes = [(YoloDetector(w, imgsz=imgsz, conf=conf, device=dev), wt)
                     for w, wt in ppe_specs]
        self.device = dev
        self.iou_thr = iou_thr
        self.conf_boost = conf_boost
        self._cache: dict[tuple[str, int], dict] = {}

    # ---------- 内部 ----------

    def _merge_ppe(self, results: list[dict]) -> list[Box]:
        """
        加权框融合：按置信度从高到低扫，落进已有簇就并入，否则新开一簇。
        同标签才允许并簇。
        """
        flat: list[tuple[Box, float]] = []
        for r, (_, wt) in zip(results, self.ppes):
            for b in r["boxes"]:
                if b.label in ("helmet", "nohelmet", "vest", "goggles"):
                    flat.append((b, wt))
        flat.sort(key=lambda x: -x[0].conf)

        clusters: list[list[tuple[Box, float]]] = []
        for b, w in flat:
            hit = None
            for c in clusters:
                if c[0][0].label == b.label and iou(c[0][0], b) >= self.iou_thr:
                    hit = c
                    break
            if hit is None:
                clusters.append([(b, w)])
            else:
                hit.append((b, w))
        return _fuse_clusters(clusters, self.iou_thr, self.conf_boost)

    # ---------- 对外 ----------

    def _run(self, source) -> dict:
        t0 = time.perf_counter()
        a = self.person._predict(source)
        parts = [det._predict(source) for det, _ in self.ppes]
        boxes = list(a["boxes"]) + self._merge_ppe(parts)
        return {"boxes": boxes, "infer_ms": round((time.perf_counter() - t0) * 1000, 2),
                "backend": self.name, "kind": self.kind,
                "width": a.get("width"), "height": a.get("height"),
                "dropped": {k: v for p in parts for k, v in p.get("dropped", {}).items()}}

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        key = (str(image_path), frame_idx)
        if key not in self._cache:
            self._cache[key] = self._run(str(image_path))
        return self._cache[key]

    def detect_frame(self, frame) -> dict:
        return self._run(frame)

    def detect(self, scene: dict, frame_idx: int = 0) -> dict:
        return self.detect_image(scene["image"], frame_idx)
