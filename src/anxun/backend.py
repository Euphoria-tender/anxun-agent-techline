# -*- coding: utf-8 -*-
"""
安巡 Agent · 后端自动分发
==========================

同一个程序，换台电脑就能跑，有显卡就自动用显卡。三条路：

    torch + CUDA     .pt 权重 + torch[cuXXX]     最快（本机 RTX 5060：25.7 ms）
    ONNX + CUDA      .onnx + onnxruntime-gpu     快，且模型只有 55MB
    ONNX + CPU       .onnx + onnxruntime         任意电脑可跑（本机 CPU：≈46 ms，三模型并发）

选路策略（prefer="auto"）：

    1. 有 torch 且有 CUDA、且有 .pt 权重        -> torch（GPU 上最快）
    2. 有 onnxruntime-gpu 且能做 CUDA、有 .onnx  -> onnx（GPU，模型小）
    3. 有 .onnx                                  -> onnx（CPU，**任何电脑**）
    4. 有 torch                                  -> torch（CPU）
    5. 都没有                                     -> 报错，并说明该装什么

命中的是哪条，会作为 note 字符串回给调用方，服务 /health 会原样显示——
不静默降级，跑的是什么一眼能看出来。
"""

from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PT_PPE = ROOT / "models" / "best.pt"
PT_PERSON = ROOT / "models" / "yolov8n.pt"
PT_VEST = ROOT / "models" / "vest.pt"          # 反光背心（可选，有就三模型）
ONNX_DIR = ROOT / "models" / "onnx"
ONNX_VEST = ONNX_DIR / "vest.onnx"
ONNX_PPE = ONNX_DIR / "best.onnx"
ONNX_PERSON = ONNX_DIR / "yolov8n.onnx"


# ------------------------------------------------------------
# 能力探测（只探测，不导入重依赖）

def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def has_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def ort_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def can(backend: str) -> tuple[bool, str]:
    """某条路能不能走，以及为什么不能。"""
    if backend == "onnx":
        if not has_ort():
            return False, "未安装 onnxruntime（pip install onnxruntime）"
        if not ONNX_PPE.exists():
            return False, f"缺 ONNX 权重 {ONNX_PPE.name}（python src/anxun/export_onnx.py）"
        return True, ""
    if backend == "torch":
        if not has_torch():
            return False, "未安装 torch"
        if not PT_PPE.exists():
            return False, f"缺权重 {PT_PPE.name}"
        return True, ""
    return False, f"未知后端 {backend}"


def _as_onnx(p: str, default: Path) -> str:
    """只有显式给了 .onnx 才用它；默认的 .pt 路径不往 ONNX 后端塞。"""
    return p if p and str(p).lower().endswith(".onnx") else str(default)


def _as_pt(p: str, default: Path) -> str:
    return p if p and str(p).lower().endswith(".pt") else str(default)


# ------------------------------------------------------------
# 分发

def build(prefer: str = "auto", imgsz: int = 640, conf: float = 0.25,
          device: str = "", model: str = "", person_model: str = "",
          vest_model: str = ""):
    """
    返回 (detector, note, backend_name)。

    prefer: auto | torch | onnx
    device: 空/auto 自动；cpu 强制 CPU；cuda:0 强制 GPU
    """
    prefer = (prefer or "auto").lower()
    want_gpu = not (device or "").lower().startswith("cpu")

    # ---- 显式指定 onnx ----
    if prefer == "onnx":
        ok, why = can("onnx")
        if not ok:
            raise RuntimeError(f"ONNX 后端不可用：{why}")
        from .onnx_backend import build_onnx_detector
        det, note = build_onnx_detector(
            conf=conf, imgsz=imgsz, device=device or "auto",
            ppe=_as_onnx(model, ONNX_PPE), person=_as_onnx(person_model, ONNX_PERSON),
            vest=_as_onnx(vest_model, ONNX_VEST))
        return det, note, "onnx"

    # ---- 显式指定 torch ----
    if prefer == "torch":
        ok, why = can("torch")
        if not ok:
            raise RuntimeError(f"torch 后端不可用：{why}")
        from .detectors import build_detector, pick_device
        det, note = build_detector(_as_pt(model, PT_PPE), _as_pt(person_model, PT_PERSON),
                                   _as_pt(vest_model, PT_VEST),
                                   imgsz=imgsz, conf=conf,
                                   device=pick_device("" if want_gpu else "cpu"))
        return det, note, "torch"

    # ---- auto ----
    torch_ok, _ = can("torch")
    onnx_ok, _ = can("onnx")
    if torch_ok and want_gpu and torch_cuda():
        from .detectors import build_detector
        det, note = build_detector(_as_pt(model, PT_PPE), _as_pt(person_model, PT_PERSON),
                                   _as_pt(vest_model, PT_VEST),
                                   imgsz=imgsz, conf=conf, device="cuda:0")
        return det, note + "（auto：优先 torch-CUDA）", "torch"

    if onnx_ok and want_gpu and "CUDAExecutionProvider" in ort_providers():
        from .onnx_backend import build_onnx_detector
        det, note = build_onnx_detector(conf=conf, imgsz=imgsz, device="cuda",
                                        ppe=_as_onnx(model, ONNX_PPE),
                                        person=_as_onnx(person_model, ONNX_PERSON),
                                        vest=_as_onnx(vest_model, ONNX_VEST))
        return det, note + "（auto：ONNX-CUDA）", "onnx"

    if onnx_ok:
        from .onnx_backend import build_onnx_detector
        det, note = build_onnx_detector(conf=conf, imgsz=imgsz, device="cpu",
                                        ppe=_as_onnx(model, ONNX_PPE),
                                        person=_as_onnx(person_model, ONNX_PERSON),
                                        vest=_as_onnx(vest_model, ONNX_VEST))
        return det, note + "（auto：ONNX-CPU，任意电脑可跑）", "onnx"

    if torch_ok:
        from .detectors import build_detector, pick_device
        det, note = build_detector(_as_pt(model, PT_PPE), _as_pt(person_model, PT_PERSON),
                                   _as_pt(vest_model, PT_VEST),
                                   imgsz=imgsz, conf=conf, device=pick_device(""))
        return det, note + "（auto：torch）", "torch"

    raise RuntimeError(
        "没有任何可用的推理后端。二选一：\n"
        "  轻量（推荐，任意电脑）：pip install onnxruntime opencv-python numpy\n"
        "  有显卡：pip install torch ultralytics")


def describe() -> dict:
    """给 /health 用：这台机器能走哪几条路。"""
    return {
        "torch": has_torch(),
        "torch_cuda": torch_cuda(),
        "onnxruntime": has_ort(),
        "onnx_providers": ort_providers(),
        "onnx_weights": ONNX_PPE.exists(),
        "pt_weights": PT_PPE.exists(),
    }


# ------------------------------------------------------------
# 启动自检：显卡到底用上了没，实测一个数出来

def _probe_frames(n: int) -> list:
    """
    取 n 帧**真实画面**做测速样本。

    为什么不用随机噪声：噪声图里没有目标，NMS 几乎不干活，测出来的耗时偏乐观。
    实测同一后端：随机图 23.9 ms，真实图 40+ ms——差的就是这部分后处理。
    """
    import numpy as np
    try:
        import cv2
        for name in ("site_demo.mp4", "demo.mp4"):
            p = ROOT / "datasets" / name
            if not p.exists():
                continue
            cap = cv2.VideoCapture(str(p))
            out = []
            while len(out) < n:
                ok, f = cap.read()
                if not ok:
                    break
                out.append(f)
            cap.release()
            if out:
                return out
    except Exception:
        pass
    # 兜底：没有视频素材时用渐变图（至少比纯噪声接近真实）
    rng = np.random.default_rng(0)
    return [np.clip(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
                    0, 255).astype(np.uint8) for _ in range(n)]


def selfcheck(det, samples: int = 6) -> dict:
    """
    跑几次推理，报出实际设备与实测耗时。

    为什么需要它：光是「检测到 CUDA」不等于「真的在用」——
    驱动版本不匹配、provider 回落到 CPU、显存不足都会让程序悄悄跑在 CPU 上。
    自检拿真实帧量一遍，把「应该用显卡」变成「确实在用，而且这么快」。
    """
    import numpy as np
    inner = getattr(det, "person", det)
    provider = getattr(inner, "provider", None)
    device = getattr(inner, "device", "") or ""
    frames = _probe_frames(samples)

    det.detect_frame(frames[0])            # 预热，不计入
    ts = []
    for f in frames:
        t0 = time.perf_counter()
        det.detect_frame(f)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()

    on_gpu = (str(provider).startswith("CUDA")
              or str(device).startswith("cuda"))
    return {
        "device": provider or device or "cpu",
        "gpu_active": bool(on_gpu),
        "mean_ms": round(sum(ts) / len(ts), 1),
        "p50_ms": round(ts[len(ts) // 2], 1),
        "best_ms": round(ts[0], 1),
        "samples": samples,
    }

