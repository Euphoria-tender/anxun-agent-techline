# -*- coding: utf-8 -*-
"""
安巡 Agent · 推理服务
=====================

技术线的「部署」这一项，就这一个文件：模型封成 HTTP，业务层怎么调都行。

    python src/anxun/app.py
    -> http://127.0.0.1:8765

接口
----
    GET  /health                        服务与模型状态
    GET  /api/scenes                    样例场景列表
    POST /api/inspect   {scene_id}      跑完整链路：检测 → 四态 → 定级 → 工单
    POST /detect_b64    {image_base64}  只做检测，给扣子插件/外部系统用
    POST /detect_b64    {image_path}    同上，传本地图片路径
    GET  /api/metrics                   最近一次评测结果
    GET  /docs/<file>                   技术页等静态文档

设计取舍：只用标准库。比赛现场断网、评委电脑没装依赖时，
「python 一敲就能跑」比「技术栈先进」值钱。
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import tempfile
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anxun.catalog import REGION_MAP, SCENES, SCENE_MAP, public_scene
from anxun import backend as backend_mod
from anxun.pipeline import inspect, thresholds

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "data" / "metrics_real.json"

STATE = {"detector": None, "note": "", "boot_at": datetime.now()}


def load_metrics() -> dict | None:
    if not METRICS.exists():
        return None
    try:
        return json.loads(METRICS.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_inspect(scene_id: str) -> dict:
    scene = SCENE_MAP.get(scene_id)
    if not scene:
        raise ValueError(f"未知场景 {scene_id}，可用：{[s['scene_id'] for s in SCENES]}")
    return inspect(scene, REGION_MAP[scene["region_id"]], STATE["detector"])


class Handler(BaseHTTPRequestHandler):
    server_version = "AnxunAgent/1.0"

    def log_message(self, fmt, *args):
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)

    # ---------- helpers ----------
    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            return self._json({"error": "not found"}, 404)
        raw = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---------- routes ----------
    def do_GET(self):
        p = urlparse(self.path).path
        try:
            if p == "/health":
                inner = getattr(STATE["detector"], "person", STATE["detector"])
                prov = getattr(inner, "provider", None)          # ONNX 时是 CPU/CUDA 提供者
                dev = getattr(inner, "device", "") or ""
                return self._json({
                    "status": "ok", "backend": STATE["detector"].name,
                    "backend_kind": STATE.get("backend", ""),
                    "note": STATE["note"], "scenes": len(SCENES),
                    "device": prov or dev or "cpu",
                    "selfcheck": STATE.get("selfcheck"),
                    "available": backend_mod.describe(),
                    "uptime_s": (datetime.now() - STATE["boot_at"]).seconds,
                })
            if p == "/api/scenes":
                return self._json({"scenes": [public_scene(s) for s in SCENES],
                                   "thresholds": thresholds()})
            if p == "/api/metrics":
                return self._json(load_metrics() or {"kind": "empty"})
            if p.startswith("/docs/"):
                target = (ROOT / "docs" / unquote(p[len("/docs/"):])).resolve()
                if not str(target).startswith(str((ROOT / "docs").resolve())):
                    return self._json({"error": "forbidden"}, 403)
                return self._file(target)
            return self._json({"error": "not found", "path": p}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            body = self._body()
            if p == "/api/inspect":
                return self._json(run_inspect(body.get("scene_id", "")))
            if p == "/detect_b64":
                return self._json(self._detect(body))
            return self._json({"error": "not found", "path": p}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": str(e)}, 500)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    # ---------- 检测接口 ----------
    def _detect(self, body: dict) -> dict:
        """
        只做检测，不做判定。给外部系统（扣子插件等）用。
        三种入参任选：image_base64 / image_path / scene_id。
        返回结构与本文件顶部注释一致，与 Box 严格对齐。
        """
        b64 = body.get("image_base64")
        path = body.get("image_path")
        sid = body.get("scene_id")
        tmp = None

        if b64:
            if "," in b64[:100]:                 # 兼容 data URI
                b64 = b64.split(",", 1)[1]
            tmp = Path(tempfile.gettempdir()) / f"anxun_{int(time.time() * 1000)}.jpg"
            tmp.write_bytes(base64.b64decode(b64))
            path = str(tmp)
        elif not path and sid:
            scene = SCENE_MAP.get(sid)
            if not scene:
                return {"error": f"未知场景 {sid}"}
            path = scene["image"]
        elif path:
            resolved = Path(path).resolve()
            if not str(resolved).startswith(str(ROOT)):
                return {"error": "image_path 必须在项目目录内"}
            path = str(resolved)

        if not path or not Path(path).exists():
            return {"error": "需要 image_base64 / image_path / scene_id 之一"}

        det = STATE["detector"].detect({"image": path, "scene_id": Path(path).stem}, 0)
        if tmp and tmp.exists():
            tmp.unlink()
        return {
            "width": det.get("width"), "height": det.get("height"),
            "infer_ms": det["infer_ms"],
            "backend": det["backend"],
            "boxes": [b.to_dict() for b in det["boxes"]],
        }


def main():
    ap = argparse.ArgumentParser(description="安巡 Agent 推理服务")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=str(ROOT / "models" / "best.pt"),
                    help="PPE 模型（安全帽）路径")
    ap.add_argument("--person-model", default=str(ROOT / "models" / "yolov8n.pt"),
                    help="人体检测模型路径，与 --model 组成双模型")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "onnx"),
                    help="auto=有显卡用显卡；onnx=不依赖 torch，任意电脑可跑")
    ap.add_argument("--device", default="", help="cpu 强制 CPU；留空自动")
    ap.add_argument("--vest-model", default="", help="反光背心权重；留空则 models/vest.pt 存在就用")
    ap.add_argument("--profile", default="", choices=("", "measured", "conservative"),
                    help="判定阈值口径；留空读 ANXUN_RULES_PROFILE 或 config 默认值（measured）。"
                         "演示想要四态齐全就用 conservative")
    args = ap.parse_args()

    if args.profile:
        from anxun import rules_engine
        snap = rules_engine.apply_profile(args.profile)
        print(f"阈值口径：{snap['label']}（{snap['profile']}）"
              f"  conf {snap['conf_review']}/{snap['conf_final']}"
              f"  IoU {snap['iou_min']}")

    try:
        det, note, bname = backend_mod.build(
            prefer=args.backend, imgsz=args.imgsz, conf=args.conf,
            device=args.device, model=args.model, person_model=args.person_model,
            vest_model=args.vest_model)
    except Exception as e:
        print(f"启动失败：{e}")
        return 2

    STATE["detector"], STATE["note"], STATE["backend"] = det, note, bname
    STATE["selfcheck"] = backend_mod.selfcheck(det)
    _sc = STATE["selfcheck"]
    print(f"设备：{_sc['device']}  GPU{'已启用' if _sc['gpu_active'] else '未启用'}"
          f"  实测 {_sc['mean_ms']} ms/帧（p50 {_sc['p50_ms']}）")
    if not _sc["gpu_active"]:
        print("提示：本机可用后端 " + str(backend_mod.describe())
              + "\n      想让 GPU 生效：装 torch 或 onnxruntime-gpu；或加 --device cuda:0")

    # 预热：CUDA 的第一次推理要走 cuDNN 选核，要 2~3 秒。
    # 不预热的话，第一路请求会卡在那儿，现场看着就像"服务挂了"。
    if "cuda" in note.lower() or "CUDA" in note:
        import time as _t
        import numpy as _np
        t0 = _t.perf_counter()
        det.detect_frame(_np.zeros((640, 640, 3), dtype=_np.uint8))
        print(f"GPU 预热完成（{(_t.perf_counter() - t0) * 1000:.0f} ms）")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"安巡 Agent 推理服务  ->  http://{args.host}:{args.port}")
    print(f"检测后端：{det.name}（{note}）")
    print(f"可选后端能力：{backend_mod.describe()}")
    print(f"样例场景：{len(SCENES)} 个")
    print("Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
