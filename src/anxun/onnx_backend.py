# -*- coding: utf-8 -*-
"""
安巡 Agent · ONNX 推理后端（不依赖 torch）
==========================================

为什么要有这个后端
------------------
原来跑推理要装 `torch 2.14.1+cu130`，光这一个轮子 **1.99 GB**，还是 cp313 + Blackwell 专用。
换一台电脑（没显卡 / Python 版本不同 / macOS / Linux）基本装不上，等于项目搬不走。

ONNX 版把依赖压到这一层：

    onnxruntime   ~15 MB   （CPU 版，任何平台都有轮子）
    numpy         ~20 MB
    opencv        ~40 MB
    models/onnx   ~55 MB   （两个模型）
    ------------------------
    合计 ≈130 MB，纯 CPU，不碰 CUDA

接口与 `detectors.YoloDetector` **完全一致**（detect_image / detect_frame / predict_many），
所以上层 pipeline、stream、scheduler、evaluate 一行都不用改。

实现要点
--------
1. **letterbox**：等比缩放 + 灰边填充到 640×640，与 ultralytics 训练时一致；
   不能直接 resize，否则长宽比变了，框会系统性偏移。
2. **解码**：YOLOv8 输出 (1, 4+nc, N)，前 4 行是 cx,cy,w,h，其余是各类别概率（导出时已过 sigmoid）。
   不需要再做 sigmoid，也不能做——做了会把 0.9 变成 0.71。
3. **NMS**：逐类别做，与 ultralytics 的 `agnostic=False` 对齐。
4. **回映射**：先减 padding 再除缩放比，得到原图坐标。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .rules_engine import Box

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "onnx"


# ============================================================
# 预处理 / 解码 / NMS（纯 numpy + cv2）

def letterbox(img: np.ndarray, size: int = 640, color: int = 114):
    """等比缩放并居中填充。返回 (图, 缩放比, (左填充, 上填充))。"""
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) == (size, size):
        return img, r, (0, 0)          # 已经是目标尺寸：不 resize 也不拷贝
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, r, (left, top)


def to_tensor(img: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> RGB 1CHW float32 [0,1]。"""
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    x *= np.float32(1.0 / 255.0)       # 原地缩放，少一次 5 MB 的分配
    return x[None, ...]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.7) -> np.ndarray:
    """单类别 NMS，返回保留索引。boxes 为 xyxy。"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode(output: np.ndarray, conf: float, iou_thr: float,
           ratio: float, pad: tuple[int, int], n_classes: int):
    """
    output: (1, 4+nc, N) 或 (4+nc, N)
    返回 [(cls_id, conf, x1, y1, x2, y2), ...]，坐标已映射回原图。
    """
    pred = output[0] if output.ndim == 3 else output
    if pred.shape[0] != 4 + n_classes and pred.shape[1] == 4 + n_classes:
        pred = pred.T
    boxes_cxcywh = pred[:4].T                  # (N, 4)
    cls_scores = pred[4:4 + n_classes].T       # (N, nc)
    cls_id = cls_scores.argmax(axis=1)
    score = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
    m = score >= conf
    if not m.any():
        return []
    b = boxes_cxcywh[m]
    s = score[m]
    c = cls_id[m]

    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    out = []
    for k in np.unique(c):
        idx = np.where(c == k)[0]
        keep = idx[nms(xyxy[idx], s[idx], iou_thr)]
        for i in np.atleast_1d(keep):
            x1, y1, x2, y2 = xyxy[i]
            out.append((int(k), float(s[i]),
                        (x1 - pad[0]) / ratio, (y1 - pad[1]) / ratio,
                        (x2 - pad[0]) / ratio, (y2 - pad[1]) / ratio))
    return out


# ============================================================
# 检测器

class OnnxDetector:
    """与 YoloDetector 同接口，底层是 onnxruntime。"""

    name = "onnx"
    kind = "model"

    def __init__(self, onnx_path: str | Path, manifest: dict | None = None,
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.7,
                 device: str = "", threads: int = 0):
        import onnxruntime as ort
        from .detectors import YoloDetector      # 复用同一份类别映射

        self.path = Path(onnx_path)
        if not self.path.exists():
            raise FileNotFoundError(f"找不到 ONNX 权重：{self.path}")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device                      # 保留字段，ONNX 侧只做 CPU
        self.LABEL_MAP = YoloDetector.LABEL_MAP
        self.IGNORE = YoloDetector.IGNORE

        # 类别名来自导出时写下的 manifest，不猜
        stem = self.path.stem
        man = manifest if manifest is not None else _load_manifest()
        info = man.get(stem) or {}
        names = info.get("names") or {}
        self.names = {int(k): v for k, v in names.items()}
        self.imgsz = int(info.get("imgsz", imgsz))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        # ⚠️ 关键一行：关掉线程自旋。
        # ORT 默认让每个 session 的计算线程忙等（spin）而不是睡眠。
        # 我们有两个 session（人体 + 装备）交替跑，24 核就被 48 个自旋线程抢爆：
        # 实测 888 ms → 49 ms，18 倍差距。这是本文件里最重要的一行。
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        providers = _pick_providers(self.device)
        self.sess = ort.InferenceSession(str(self.path), so, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.input_name = self.sess.get_inputs()[0].name
        self.unmapped = sorted({str(n) for n in self.names.values()}
                               - set(self.LABEL_MAP) - self.IGNORE)
        self._cache: dict[tuple[str, int], dict] = {}
        # 服务是长跑进程：缓存不设上限，换一批图就一路涨到内存告警。
        # 缓存只负责「同一张图重复请求不重算」，清空不改变任何结果。
        self.cache_max = 256

    # ---------- 内部 ----------

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """letterbox + 归一化，返回 (输入张量, 缩放比, 填充)。

        单独拆出来是为了复用：三个模型 imgsz 相同时，同一张图的张量完全一样，
        以前每个模型各算一遍（0.5 + 2.8 ms × 3），纯属重复劳动。
        """
        lb, r, pad = letterbox(frame, self.imgsz)
        return to_tensor(lb), r, pad

    def infer_tensor(self, tensor: np.ndarray, ratio: float, pad: tuple[int, int],
                     shape: tuple[int, int]) -> dict:
        """对已预处理的张量做前向 + 解码。shape 是原图 (h, w)。"""
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: tensor})[0]
        dets = decode(out, self.conf, self.iou, ratio, pad, len(self.names))
        boxes, dropped = [], {}
        for k, s, x1, y1, x2, y2 in dets:
            raw = str(self.names.get(k, k))
            label = self.LABEL_MAP.get(raw)
            if label is None:
                dropped[raw] = dropped.get(raw, 0) + 1
                continue
            boxes.append(Box(label, float(s), float(x1), float(y1),
                             float(x2), float(y2)))
        h, w = shape
        return {"boxes": boxes, "infer_ms": round((time.perf_counter() - t0) * 1000, 2),
                "backend": self.name, "kind": self.kind,
                "width": int(w), "height": int(h), "dropped": dropped}

    def _infer(self, frame: np.ndarray) -> dict:
        """单模型完整一次推理（含预处理）。infer_ms 与改动前口径一致。"""
        t0 = time.perf_counter()
        tensor, r, pad = self.preprocess(frame)
        out = self.infer_tensor(tensor, r, pad, frame.shape[:2])
        out["infer_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    # ---------- 对外 ----------

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        key = (str(image_path), frame_idx)
        if key not in self._cache:
            if len(self._cache) >= self.cache_max:
                self._cache.clear()
            self._cache[key] = self._infer(_read(image_path))
        return self._cache[key]

    def detect_frame(self, frame) -> dict:
        return self._infer(frame)

    def predict_many(self, sources: list) -> list[dict]:
        """ONNX 静态 batch=1，这里顺序跑。CPU 上批处理本来也没有收益（见 README 6.2）。"""
        if not sources:
            return []
        return [self._infer(s) for s in sources]

    def detect(self, scene: dict, frame_idx: int = 0) -> dict:
        return self.detect_image(scene["image"], frame_idx)


class OnnxDualDetector:
    """
    人体 + 装备，ONNX 模型串联。门控、批前向接口与 DualDetector 保持一致。

    vest_onnx 可选：反光背心模型。给了就是三模型，不加也不影响原有行为。

    两条路径的执行策略不同，是有意的：

      detect_image（单图 / HTTP 接口 / 评测）—— **三模型并发**，延迟优先。
        单张图没有「空画面」可省，串行跑等于三个模型互相排队；
        并发之后 wall time 从 ≈70 ms 降到 ≈47 ms，输出的框逐位不变
        （同一批权重、同一份预处理，只是不互相等了）。

      detect_frame（视频流）—— 先人体、再门控、再串行跑装备，吞吐优先。
        视频里大部分帧没人，门控省下的前向比并发更值钱；这条路径的耗时
        不是指标口径（见 README「三级触发」一节）。

    环境变量 ANXUN_ONNX_PARALLEL=0 可强制退回串行（对比测速 / 排障用）。
    """

    name = "onnx+onnx"
    kind = "model"

    def __init__(self, person_onnx: str | Path, ppe_onnx: str | Path,
                 vest_onnx: str | Path = "",
                 imgsz: int = 640, conf: float = 0.25, device: str = "",
                 gate: bool = True, parallel: bool | None = None):
        self.person = OnnxDetector(person_onnx, imgsz=imgsz, conf=conf, device=device)
        self.ppe = OnnxDetector(ppe_onnx, imgsz=imgsz, conf=conf, device=device)
        self.vest = (OnnxDetector(vest_onnx, imgsz=imgsz, conf=conf, device=device)
                     if vest_onnx else None)
        if self.vest:
            self.name = "onnx+onnx+onnx"
        self.gate = gate
        self.gate_skipped = 0

        # 并发只在核够多时划算：单模型本来就能吃满十几线程，核少了三个模型只会互相抢。
        # 实测 24 线程收益最大，8 线程时并发与串行基本持平，4 线程反而更慢。
        if parallel is None:
            parallel = (os.cpu_count() or 1) >= 12
            if os.environ.get("ANXUN_ONNX_PARALLEL", "1") == "0":
                parallel = False
        self.parallel = bool(parallel)
        self._pool = (ThreadPoolExecutor(max_workers=1 + len(self._equip_models()),
                                         thread_name_prefix="anxun-onnx")
                      if self.parallel else None)

        self._cache: dict[tuple[str, int], dict] = {}
        self.cache_max = 256

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

    def _shared_prep(self) -> bool:
        """三个模型输入尺寸一致时，同一张图的预处理结果可以共用。"""
        return all(m.imgsz == self.person.imgsz for m in self._equip_models())

    def _pack(self, a: dict, rest: list, ms: float) -> dict:
        boxes = list(a["boxes"])
        dropped = dict(a.get("dropped") or {})
        for r in rest:
            boxes.extend(r["boxes"])
            dropped.update(r.get("dropped") or {})
        return {"boxes": boxes, "infer_ms": round(ms, 2),
                "backend": self.name, "kind": self.kind,
                "width": a.get("width"), "height": a.get("height"),
                "dropped": dropped}

    def _run(self, frame, parallel: bool) -> dict:
        t0 = time.perf_counter()
        models = [self.person] + self._equip_models()
        if self._shared_prep():
            prep = self.person.preprocess(frame)          # 只算一次
            jobs = [(m, prep) for m in models]
        else:
            jobs = [(m, m.preprocess(frame)) for m in models]
        shape = frame.shape[:2]

        if parallel and self._pool is not None:
            futs = [self._pool.submit(m.infer_tensor, *p, shape) for m, p in jobs]
            res = [f.result() for f in futs]
            a = res[0]
            wall = (time.perf_counter() - t0) * 1000
            # 门控照旧生效：没人就把装备结果丢掉。并发只是「白算」，结论不变。
            if not self._need_equip(a):
                return self._pack(a, [], wall)
            return self._pack(a, res[1:], wall)

        a = self.person.infer_tensor(*jobs[0][1], shape)
        if not self._need_equip(a):
            return self._pack(a, [], (time.perf_counter() - t0) * 1000)
        rest = [m.infer_tensor(*p, shape) for m, p in jobs[1:]]
        return self._pack(a, rest, (time.perf_counter() - t0) * 1000)

    def detect_image(self, image_path: str | Path, frame_idx: int = 0) -> dict:
        key = (str(image_path), frame_idx)
        if key in self._cache:
            return self._cache[key]
        if len(self._cache) >= self.cache_max:
            self._cache.clear()
        out = self._run(_read(image_path), parallel=True)
        self._cache[key] = out
        return out

    def detect_frame(self, frame) -> dict:
        return self._run(frame, parallel=False)

    def detect_batch(self, frames: list) -> list[dict]:
        return [self.detect_frame(f) for f in frames]

    def detect(self, scene: dict, frame_idx: int = 0) -> dict:
        return self.detect_image(scene["image"], frame_idx)


# ============================================================

def _pick_providers(device: str) -> list[str]:
    """
    设备选择：装了 onnxruntime-gpu 且驱动能做，就用 CUDA；否则回落 CPU。

    `onnxruntime`（CPU 版）和 `onnxruntime-gpu` 只能装一个——
    同一份代码，装哪个就跑哪个，不用改一行。
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    want_cuda = (device or "").lower().startswith("cuda") or (device or "") == "auto"
    if want_cuda and "CUDAExecutionProvider" in avail:
        # 显存不足时 ORT 会自动回落到 CPUExecutionProvider，不会直接崩
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_manifest() -> dict:
    p = MODEL_DIR / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _read(path) -> np.ndarray:
    """
    读图。不用 cv2.imread —— 它在 Windows 上遇到非 ASCII 路径（比如用户名「华硕」）
    会静默返回 None，然后报一个「读不到图片」的假错。imdecode 没这个问题。
    """
    import cv2
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(f"读不到图片：{path}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"图片解码失败：{path}")
    return img


def build_onnx_detector(conf: float = 0.25, imgsz: int = 640,
                        ppe: str = "", person: str = "", vest: str = "",
                        device: str = "auto") -> tuple[object, str]:
    """按目录里的 ONNX 权重挑组合，返回 (detector, 说明)。device=auto 时自动挑 CUDA/CPU。"""
    ppe_p = Path(ppe) if ppe else MODEL_DIR / "best.onnx"
    per_p = Path(person) if person else MODEL_DIR / "yolov8n.onnx"
    ves_p = Path(vest) if vest else MODEL_DIR / "vest.onnx"
    if not ppe_p.exists():
        raise FileNotFoundError(
            f"找不到 ONNX 权重：{ppe_p}\n先导出：python src/anxun/export_onnx.py")
    dev = device or "auto"
    if per_p.exists():
        if ves_p.exists():
            det = OnnxDualDetector(per_p, ppe_p, ves_p, imgsz=imgsz, conf=conf, device=dev)
            return det, (f"ONNX 三模型：{per_p.name} + {ppe_p.name} + {ves_p.name} "
                         f"@ {det.person.provider}")
        det = OnnxDualDetector(per_p, ppe_p, imgsz=imgsz, conf=conf, device=dev)
        return det, f"ONNX 双模型：{per_p.name} + {ppe_p.name} @ {det.person.provider}"
    det = OnnxDetector(ppe_p, imgsz=imgsz, conf=conf, device=dev)
    return det, f"ONNX 单模型：{ppe_p.name} @ {det.provider}（缺人体模型，人员判定不会生效）"
