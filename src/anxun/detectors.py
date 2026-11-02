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

# 引擎类别 -> 模型输出里能对应的全部写法（正向证据 + 反向证据）。
# 用来回答「这个检测器到底能判哪几类装备」，见 available_equips()。
EQUIP_LABELS = {
    "helmet": ("helmet", "nohelmet"),
    "vest": ("vest", "novest"),
    "goggles": ("goggles",),
}


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
        # 同上：无连字符/无下划线写法也收进来
        "novest": "novest", "no vest": "novest",
        # 否定类**不丢弃**，改标签后交给引擎当反向证据用。
        # 安全帽模型专门训练过「裸头」这一类，扔掉等于白扔一半信息。
        "NO-Hardhat": "nohelmet", "no-hardhat": "nohelmet", "NO-Helmet": "nohelmet",
        "no helmet": "nohelmet", "Head": "nohelmet", "head": "nohelmet",
        # 无连字符写法：helmet_p2 数据集的 data.yaml 就是 ['helmet', 'nohelmet']。
        # 精确匹配的字典查不到就会把整类框丢掉，静默变成「没有反向证据」——
        # 漏报会虚高，而且不报错。v2.2 时期在这批数据上训过模型，一直没发现。
        "nohelmet": "nohelmet",
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
        self.cache_max = 256      # 长跑服务：缓存不设上限会一路涨
        self.unmapped = sorted({str(n) for n in self.names.values()}
                               - set(self.LABEL_MAP) - self.IGNORE) if isinstance(self.names, dict) else []

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        key = (str(image_path), frame_idx)
        if key in self._cache:
            return self._cache[key]
        if len(self._cache) >= self.cache_max:
            self._cache.clear()
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
                 gate: bool = True, drop_equip_person: bool = True):
        # 单模型模式：person 与装备权重是同一个文件（v3 的五类权重自带 person）。
        # 只加载一次、只前向一次，person 与装备框来自同一次推理。
        #
        # 为什么不能走"双模型但传同一个文件"：那条路会前向两次，然后 `_merge`
        # 把 person 模型的**全部**框（含 helmet/vest）和装备模型的框拼起来——
        # 装备框被算两遍。实测 helmet 框 2 个变成 4 个，fp 从 27 涨到 189，
        # helmet AP 从 0.87 掉到 0.45。这不是模型问题，是拼框逻辑的假设失效：
        # 它默认「person 模型只出 person」，而 yolov8n 恰好如此。
        self.single = (str(Path(person_weights).resolve())
                       == str(Path(ppe_weights).resolve()))
        if self.single:
            self.ppe = YoloDetector(ppe_weights, imgsz=imgsz, conf=conf, device=device)
            self.person = self.ppe
            self.vest = None
            self.name = "ppe+person"
            self.drop_equip_person = False
            self.gate = False
            self.gate_skipped = 0
            return

        self.person = YoloDetector(person_weights, imgsz=imgsz, conf=conf, device=device)
        self.ppe = YoloDetector(ppe_weights, imgsz=imgsz, conf=conf, device=device)
        self.vest = (YoloDetector(vest_weights, imgsz=imgsz, conf=conf, device=device)
                     if vest_weights else None)
        if self.vest:
            self.name = "person+ppe+vest"
        # 装备模型如果自带 person 类（v3 的五类权重就带），它给出的 person 会和
        # 专用人体模型的框重复。判定是「按人」做的，同一个人出现两次会被算两遍，
        # 误报率直接翻倍。person 一律只认人体模型那一路。
        # 老的两类权重（Hardhat / Hardhat-NO）本来就没有 person 类，此开关对它们无影响。
        self.drop_equip_person = drop_equip_person
        # 人员门控：判定是「按人」做的，画面里没人时装备模型的前向纯属浪费。
        # 这一步在语义上**完全无损**——没有 person，装备框不参与任何判定。
        self.gate = gate
        self.gate_skipped = 0

    def _equip_boxes(self, r: dict) -> list:
        """取装备模型的框；按需剔掉 person，避免和人体模型重复计数。"""
        if not self.drop_equip_person:
            return r["boxes"]
        return [b for b in r["boxes"] if b.label != "person"]

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
        if self.single:      # 一次前向already包含全部类别
            return self._merge(a, {"boxes": [], "infer_ms": 0.0})
        if not self._need_equip(a):
            return self._merge(a, {"boxes": [], "infer_ms": 0.0})
        boxes: list[Box] = []
        ms = 0.0
        for m in self._equip_models():
            r = m.detect_image(img, frame_idx)
            boxes += self._equip_boxes(r)
            ms += r["infer_ms"]
        return self._merge(a, {"boxes": boxes, "infer_ms": ms})

    def detect_frame(self, frame) -> dict:
        """三模型跑同一帧（视频流）。没人就跳过全部装备模型。"""
        a = self.person.detect_frame(frame)
        if self.single:      # 单模型：一次前向已含全部类别
            return self._merge(a, {"boxes": [], "infer_ms": 0.0})
        if not self._need_equip(a):
            return self._merge(a, {"boxes": [], "infer_ms": 0.0})
        boxes, ms = [], 0.0
        for m in self._equip_models():
            r = m.detect_frame(frame)
            boxes += self._equip_boxes(r)
            ms += r["infer_ms"]
        return self._merge(a, {"boxes": boxes, "infer_ms": ms})

    def detect_batch(self, frames: list) -> list[dict]:
        """多路视频的帧一起前向：一次 GPU 调用吃 N 帧，吞吐显著高于逐帧。"""
        if not frames:
            return []
        a = self.person.predict_many(frames)
        if self.single:      # 单模型：一批帧一次前向，框已完整
            return [self._merge(r, {"boxes": [], "infer_ms": 0.0}) for r in a]
        need = [i for i, r in enumerate(a) if self._need_equip(r)]
        out = [self._merge(r, {"boxes": [], "infer_ms": 0.0}) for r in a]
        for m in self._equip_models():
            if not need:
                break
            sub = [frames[i] for i in need]
            for i, r in zip(need, m.predict_many(sub)):
                prev = out[i]
                out[i] = self._merge(a[i], {"boxes": list(prev["boxes"]) + self._equip_boxes(r),
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


def available_equips(det) -> list[str] | None:
    """这个检测器**实际能产出**的装备类别。

    为什么要问模型而不是问配置：`catalog` 里写的是「想判什么」，但权重缺失时
    模型根本吐不出这一类框，规则引擎会把「一个框都没有」判成「没戴」——
    一个人「没穿反光背心」的假告警能刷满整屏。这里按模型真实类别求交集：
    缺模型就少判一类，绝不拿「没检出」冒充「没佩戴」。

    返回 None 表示问不出来（自定义 / 桩检测器没有 names），调用方应保持原配置。
    """
    seen, found = set(), False
    stack = [det]
    while stack:
        d = stack.pop()
        names = getattr(d, "names", None)
        if names:
            found = True
            for n in names.values():
                lab = YoloDetector.LABEL_MAP.get(str(n))
                if lab:
                    seen.add(lab)
        for attr in ("person", "ppe", "vest"):
            sub = getattr(d, attr, None)
            if sub is not None and sub is not det:
                stack.append(sub)
    if not found:
        return None
    return [e for e, labs in EQUIP_LABELS.items() if seen & set(labs)]


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
                   imgsz: int = 640, conf: float = 0.25, device: str = "",
                   prefer_own_person: bool = True):
    """
    按权重组合挑检测器，返回 (detector, 说明)。
    权重缺失直接报错，不做降级——没有模型就没有指标，藏起来没意义。

    **PPE 权重自带 person 类时优先用自己的**（v3.1.0 起默认走这条）：

    - 同一次前向就能拿到 person 与装备框，耗时减半（22.9 → 11.4 ms）；
    - 人框与装备框出自同一模型、同一套坐标，头部区域推算不会错位；
    - 实测 person AP 0.846，高于外挂 yolov8n 的 0.797。

    外挂人体模型只在 PPE 权重没有 person 类时才需要（老的两类权重点这样）。
    显式想用外部人体模型时传 `prefer_own_person=False`。
    """
    dev = pick_device(device)
    w = Path(weights)
    if not w.exists():
        raise FileNotFoundError(f"找不到 PPE 权重：{weights}")

    det = YoloDetector(str(w), imgsz=imgsz, conf=conf, device=dev)
    names = [str(n) for n in det.names.values()]
    has_person = any(n.lower() == "person" for n in names)

    if has_person and prefer_own_person:
        return det, f"单模型：{w.name}（{','.join(names)}）@ {dev}"

    vw = Path(vest_weights) if vest_weights else None
    pw = Path(person_weights) if person_weights else None
    if pw and pw.exists():
        if vw and vw.exists():
            det = DualDetector(str(pw), str(w), str(vw), imgsz=imgsz, conf=conf, device=dev)
            return det, f"三模型：{pw.name} + {w.name} + {vw.name} @ {dev}"
        return DualDetector(str(pw), str(w), imgsz=imgsz, conf=conf, device=dev), \
               f"双模型：{pw.name} + {w.name} @ {dev}"

    note = f"单模型：{w.name}（{','.join(names)}）@ {dev}"
    if not has_person:
        note += "；缺 person 类，人员判定不会生效，请配 --person-model"
    return det, note
