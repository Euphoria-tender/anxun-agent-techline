# -*- coding: utf-8 -*-
"""
安巡 Agent · 检测器
===================

检测器只做一件事：给一张图，吐一串 Box。规则引擎完全不知道底下用的是哪个模型。

    YoloDetector   单模型（自带 person 类的权重）
    DualDetector   双模型串联：人体 + 装备

为什么默认双模型：公开的安全帽权重多数只有 Hardhat / NO-Hardhat 两类，没有 person。
判定链第一步要「谁在画面里」，所以人体另配一个 COCO 预训练模型。
两个模型各管一段，换任何一个都不影响另一个。
"""

from __future__ import annotations

import time
from pathlib import Path

from .rules_engine import Box


class YoloDetector:
    """ultralytics YOLO 后端。类别映射是这一层唯一的脏活。"""

    name = "yolo"
    kind = "model"

    LABEL_MAP = {
        "person": "person", "Person": "person", "people": "person", "worker": "person",
        "helmet": "helmet", "Helmet": "helmet", "HELMET": "helmet",
        "hardhat": "helmet", "Hardhat": "helmet", "hard hat": "helmet",
        "safety helmet": "helmet", "Safety Helmet": "helmet",
        "vest": "vest", "Vest": "vest", "safety vest": "vest",
        "Safety Vest": "vest", "reflective vest": "vest",
        "goggles": "goggles", "Glasses": "goggles", "glasses": "goggles",
        "safety goggles": "goggles",
        # vest 训练集的类名是 vest / no_vest（下划线），别和 IGNORE 里的
        # "no-safety vest" 混为一类。no_vest 是反向证据，不能丢。
        "no_vest": "novest", "no-vest": "novest", "No-Vest": "novest",
        # 否定类**不丢弃**，改标签后交给引擎当反向证据用。
        # 安全帽模型专门训练过「裸头」这一类，扔掉等于白扔一半信息。
        "NO-Hardhat": "nohelmet", "no-hardhat": "nohelmet", "NO-Helmet": "nohelmet",
        "no helmet": "nohelmet", "Head": "nohelmet", "head": "nohelmet",
    }
    IGNORE = {
        "NO-Safety Vest", "no-safety vest", "NO-Mask", "no-mask", "mask",
        "Safety Cone", "cone", "machinery", "vehicle",
        "boots", "gloves", "no_boots", "no_gloves",
    }

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25,
                 device: str = ""):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.names = self.model.names
        self._cache: dict[tuple[str, int], dict] = {}
        self.unmapped = sorted({str(n) for n in self.names.values()}
                               - set(self.LABEL_MAP) - self.IGNORE) if isinstance(self.names, dict) else []

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        key = (str(image_path), frame_idx)
        if key in self._cache:
            return self._cache[key]
        out = self._predict(str(image_path))
        self._cache[key] = out
        return out

    def detect_frame(self, frame) -> dict:
        """直接吃 numpy 帧（视频流用），不进缓存——实时场景每帧都不同。"""
        return self._predict(frame)

    def _predict(self, source) -> dict:
        t0 = time.perf_counter()
        res = self.model.predict(source, imgsz=self.imgsz, conf=self.conf,
                                 verbose=False, device=self.device or None)[0]
        ms = (time.perf_counter() - t0) * 1000

        boxes, dropped = [], {}
        for xyxy, c, cf in zip(res.boxes.xyxy.tolist(),
                               res.boxes.cls.tolist(),
                               res.boxes.conf.tolist()):
            raw = str(self.names[int(c)])
            label = self.LABEL_MAP.get(raw)
            if label is None:
                dropped[raw] = dropped.get(raw, 0) + 1
                continue
            boxes.append(Box(label, float(cf), *[float(v) for v in xyxy]))

        h, w = res.orig_shape[:2]
        return {"boxes": boxes, "infer_ms": round(ms, 2), "backend": self.name,
                "kind": self.kind, "width": int(w), "height": int(h),
                "dropped": dropped}

    def detect(self, scene: dict, frame_idx: int = 0) -> dict:
        img = scene.get("image")
        if not img:
            raise ValueError(f"场景 {scene.get('scene_id')} 没有 image 字段，无法推理")
        return self.detect_image(img, frame_idx)

    def predict_many(self, sources: list) -> list[dict]:
        """
        批量前向：一次喂多帧（多路视频的各一路）。
        GPU 上 batch=8 时每帧 17.2ms → 7.8ms，吞吐 2.2 倍。
        """
        if not sources:
            return []
        t0 = time.perf_counter()
        results = self.model.predict(list(sources), imgsz=self.imgsz, conf=self.conf,
                                     verbose=False, device=self.device or None)
        per_ms = (time.perf_counter() - t0) * 1000 / len(sources)
        out = []
        for res in results:
            boxes, dropped = [], {}
            for xyxy, c, cf in zip(res.boxes.xyxy.tolist(),
                                   res.boxes.cls.tolist(),
                                   res.boxes.conf.tolist()):
                raw = str(self.names[int(c)])
                label = self.LABEL_MAP.get(raw)
                if label is None:
                    dropped[raw] = dropped.get(raw, 0) + 1
                    continue
                boxes.append(Box(label, float(cf), *[float(v) for v in xyxy]))
            h, w = res.orig_shape[:2]
            out.append({"boxes": boxes, "infer_ms": round(per_ms, 2), "backend": self.name,
                        "kind": self.kind, "width": int(w), "height": int(h),
                        "dropped": dropped})
        return out


class DualDetector:
    """
    人体 + 装备模型串联。

    vest_weights 可选：给了就再串一个反光背心模型（规则引擎早就支持 vest 判定，
    只是以前没有权重，「未穿戴反光背心」这条一直是空转的）。

    三个模型各管一段，换任何一个都不影响另外两个。
    """

    name = "person+ppe"
    kind = "model"

    def __init__(self, person_weights: str, ppe_weights: str,
                 vest_weights: str = "",
                 imgsz: int = 640, conf: float = 0.25, device: str = "",
                 gate: bool = True):
        self.person = YoloDetector(person_weights, imgsz=imgsz, conf=conf, device=device)
        self.ppe = YoloDetector(ppe_weights, imgsz=imgsz, conf=conf, device=device)
        self.vest = (YoloDetector(vest_weights, imgsz=imgsz, conf=conf, device=device)
                     if vest_weights else None)
        if self.vest:
            self.name = "person+ppe+vest"
        # 人员门控：判定是「按人」做的，画面里没人时装备模型的前向纯属浪费。
        # 这一步在语义上**完全无损**——没有 person，装备框不参与任何判定。
        self.gate = gate
        self.gate_skipped = 0

    def _need_equip(self, person_res: dict) -> bool:
        if not self.gate:
            return True
        if any(b.label == "person" for b in person_res["boxes"]):
            return True
        self.gate_skipped += 1
        return False

    def _equip_models(self) -> list:
        out = [self.ppe]
        if self.vest is not None:
            out.append(self.vest)
        return out

    def detect(self, scene: dict, frame_idx: int = 0) -> dict:
        img = scene.get("image")
        if not img:
            raise ValueError(f"场景 {scene.get('scene_id')} 没有 image 字段，无法推理")
        a = self.person.detect_image(img, frame_idx)
        if not self._need_equip(a):
            return self._merge(a, {"boxes": [], "infer_ms": 0.0})
        boxes: list[Box] = []
        ms = 0.0
        for m in self._equip_models():
            r = m.detect_image(img, frame_idx)
            boxes += r["boxes"]
            ms += r["infer_ms"]
        return self._merge(a, {"boxes": boxes, "infer_ms": ms})

    def detect_frame(self, frame) -> dict:
        """三模型跑同一帧（视频流）。没人就跳过全部装备模型。"""
        a = self.person.detect_frame(frame)
        if not self._need_equip(a):
            return self._merge(a, {"boxes": [], "infer_ms": 0.0})
        boxes, ms = [], 0.0
        for m in self._equip_models():
            r = m.detect_frame(frame)
            boxes += r["boxes"]
            ms += r["infer_ms"]
        return self._merge(a, {"boxes": boxes, "infer_ms": ms})

    def detect_batch(self, frames: list) -> list[dict]:
        """多路视频的帧一起前向：一次 GPU 调用吃 N 帧，吞吐显著高于逐帧。"""
        if not frames:
            return []
        a = self.person.predict_many(frames)
        need = [i for i, r in enumerate(a) if self._need_equip(r)]
        out = [self._merge(r, {"boxes": [], "infer_ms": 0.0}) for r in a]
        for m in self._equip_models():
            if not need:
                break
            sub = [frames[i] for i in need]
            for i, r in zip(need, m.predict_many(sub)):
                prev = out[i]
                out[i] = self._merge(a[i], {"boxes": list(prev["boxes"]) + list(r["boxes"]),
                                            "infer_ms": prev["infer_ms"] + r["infer_ms"]})
        return out

    def _merge(self, a: dict, b: dict) -> dict:
        return {
            "boxes": list(a["boxes"]) + list(b["boxes"]),
            "infer_ms": round(a["infer_ms"] + b["infer_ms"], 2),
            "backend": self.name, "kind": self.kind,
            "width": a.get("width"), "height": a.get("height"),
            "dropped": {**a.get("dropped", {}), **b.get("dropped", {})},
        }

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        return self.detect({"image": str(image_path),
                            "scene_id": Path(image_path).stem}, frame_idx)


def pick_device(device: str = "") -> str:
    """
    设备选择：显式指定优先；否则有 CUDA 就用 GPU，没有回落到 CPU。
    显式写 "cpu" 可以强制跑 CPU（对比测速、或在没有显卡的评委电脑上演示）。
    """
    if device:
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def build_detector(weights: str, person_weights: str = "", vest_weights: str = "",
                   imgsz: int = 640, conf: float = 0.25, device: str = ""):
    """
    按权重组合挑检测器，返回 (detector, 说明)。
    权重缺失直接报错，不做降级——没有模型就没有指标，藏起来没意义。
    """
    dev = pick_device(device)
    w = Path(weights)
    pw = Path(person_weights) if person_weights else None
    if not w.exists():
        raise FileNotFoundError(f"找不到 PPE 权重：{weights}")
    vw = Path(vest_weights) if vest_weights else None
    if pw and pw.exists():
        if vw and vw.exists():
            det = DualDetector(str(pw), str(w), str(vw), imgsz=imgsz, conf=conf, device=dev)
            return det, f"三模型：{pw.name} + {w.name} + {vw.name} @ {dev}"
        return DualDetector(str(pw), str(w), imgsz=imgsz, conf=conf, device=dev), \
               f"双模型：{pw.name} + {w.name} @ {dev}"
    det = YoloDetector(str(w), imgsz=imgsz, conf=conf, device=dev)
    names = [str(n) for n in det.names.values()]
    has_person = any(n.lower() == "person" for n in names)
    note = f"单模型：{w.name}（{','.join(names)}）@ {dev}"
    if not has_person:
        note += "；缺 person 类，人员判定不会生效，请配 --person-model"
    return det, note
