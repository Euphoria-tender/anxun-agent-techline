# -*- coding: utf-8 -*-
"""
安巡 Agent · 多路批处理调度
============================

原来的问题是「一路一帧、一帧一次前向」：GPU 一次只算一张图，显存和算力都空着。
实测 batch=8 时每帧 17.2ms → 7.8ms，吞吐 2.2 倍——同样的卡能多看一倍路数。

架构上的改动是**把推理和判定拆开**：

    读取线程池        采样、跳帧、帧差判静止      （每路独立）
        ↓
    批处理推理        多路帧凑一批，一次前向      （共享 GPU）
        ↓
    判定状态机        StreamPipeline，每路一个     （每路独立）

拆开之后两件事都变简单了：
- 推理层只关心「给我 N 帧，还我 N 组框」，不知道什么叫违规；
- 判定层只吃「框 + 时间戳」，不知道底下是 GPU 还是 CPU、批多大。

用法
----
    python src/anxun/scheduler.py --sources rtsp://cam1,rtsp://cam2 --fps 1
    python src/anxun/scheduler.py --sources demo.mp4,demo.mp4 --fps 2 --batch 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anxun.catalog import REGION_MAP                       # noqa: E402
from anxun import backend as backend_mod                 # noqa: E402
from anxun.stream import StreamPipeline                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class StreamReader:
    """
    一路视频源。负责按目标抽帧率跳帧、帧差判静止。
    它是「生产者」，不做任何推理。
    """

    def __init__(self, sid: str, source: str, region, fps: float = 2.0,
                 max_seconds: float | None = None, motion_threshold: float = 10.0):
        import cv2
        self.sid = sid
        self.source = source
        self.region = region
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开视频源：{source}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.step = max(1, int(round(self.src_fps / max(0.1, fps))))
        self.max_seconds = max_seconds
        self.motion_threshold = motion_threshold
        self.pipe = StreamPipeline(sid, region, ("helmet",), 60.0, verbose=False)
        self.read = 0
        self.last_boxes = None
        self.size = (1920, 1080)
        self._prev_gray = None
        self.active = True

    def _still(self, frame) -> bool:
        if self.motion_threshold <= 0:
            return False
        import cv2
        gray = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False
        diff = float(cv2.absdiff(gray, self._prev_gray).mean())
        self._prev_gray = gray
        return diff < self.motion_threshold

    def next_sample(self):
        """返回下一个采样帧与视频时间；到结尾返回 None。"""
        while self.active:
            ok, frame = self.cap.read()
            if not ok:
                self.active = False
                self.cap.release()
                return None
            self.read += 1
            if self.read % self.step:
                continue
            t = (self.read - 1) / self.src_fps
            if self.max_seconds and t > self.max_seconds:
                self.active = False
                self.cap.release()
                return None
            return frame, t
        return None


class MultiStreamScheduler:
    """
    多路锁步批处理：每一拍从所有活着的路各取一帧，凑成一批一次前向。

    锁步（lockstep）而不是异步队列，是为了**可复现**：
    同样的输入必定得到同样的帧序和同样的事件，演示和评测才能对上账。
    真上现场时把 next_sample 换成一个有超时的队列即可，判定层一行不用改。
    """

    def __init__(self, detector, batch: int = 8, verbose: bool = True):
        self.det = detector
        self.batch = max(1, batch)
        self.verbose = verbose
        self.readers: list[StreamReader] = []
        self.stats = {"samples": 0, "inferred": 0, "skipped": 0, "ticks": 0,
                      "batch_max": 0, "infer_ms_total": 0.0,
                      "wall_s": 0.0, "events": 0}

    def add(self, reader: StreamReader) -> "MultiStreamScheduler":
        self.readers.append(reader)
        return self

    def run(self) -> dict:
        t0 = time.perf_counter()
        while True:
            batch: list[tuple[StreamReader, object, float]] = []
            deferred: list[tuple[StreamReader, float]] = []

            for r in self.readers:
                if not r.active:
                    continue
                item = r.next_sample()
                if item is None:
                    continue
                frame, t = item
                self.stats["samples"] += 1
                if r.last_boxes is not None and r._still(frame):
                    self.stats["skipped"] += 1
                    deferred.append((r, t))       # 静止：不推理，复用上一帧结果
                else:
                    batch.append((r, frame, t))

            if not batch and not deferred:
                break

            self.stats["ticks"] += 1

            # 静止的先把时序窗口推进（人不动也得能判出没戴帽）
            for r, t in deferred:
                r.pipe.post(r.last_boxes, r.size[0], r.size[1], t)

            # 成批推理
            for s in range(0, len(batch), self.batch):
                chunk = batch[s:s + self.batch]
                results = self.det.detect_batch([f for _, f, _ in chunk])
                self.stats["batch_max"] = max(self.stats["batch_max"], len(chunk))
                for (r, frame, t), res in zip(chunk, results):
                    self.stats["inferred"] += 1
                    self.stats["infer_ms_total"] += res["infer_ms"]
                    r.last_boxes = res["boxes"]
                    r.size = (res.get("width") or 1920, res.get("height") or 1080)
                    r.pipe.post(res["boxes"], r.size[0], r.size[1], t)

        wall = time.perf_counter() - t0
        events = [e for r in self.readers for e in r.pipe.events]
        self.stats.update(
            wall_s=round(wall, 2),
            events=len(events),
            avg_infer_ms=round(self.stats["infer_ms_total"] / max(1, self.stats["inferred"]), 1),
            gate_skipped=getattr(self.det, "gate_skipped", 0),
            throughput_fps=round(self.stats["inferred"] / wall, 1) if wall else 0,
        )
        self.events = events
        return self.stats


def main() -> int:
    ap = argparse.ArgumentParser(description="安巡 Agent 多路视频巡查")
    ap.add_argument("--sources", required=True, help="逗号分隔的视频源（文件或 RTSP）")
    ap.add_argument("--regions", default="", help="逗号分隔的区域编号，缺省按 B-01 循环")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8, help="批大小，默认 8（GPU 最优）")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--motion", type=float, default=10.0,
                    help="帧差阈值（160x160 灰度均值差）。1080p 复验后从 3.0 提到 10.0")
    ap.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-model", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "onnx"))
    ap.add_argument("--device", default="", help="cpu 强制 CPU；留空自动")
    ap.add_argument("--vest-model", default="", help="反光背心权重；留空自动")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
    regs = [s.strip() for s in args.regions.split(",") if s.strip()] or ["B-01"]
    regions = [REGION_MAP.get(r) or list(REGION_MAP.values())[0] for r in regs]

    det, note, _ = backend_mod.build(prefer=args.backend, imgsz=args.imgsz,
                                     device=args.device,
                                     model=args.model, person_model=args.person_model,
                                     vest_model=args.vest_model)
    sc = backend_mod.selfcheck(det)
    print(f"设备：{sc['device']}  GPU{'已启用' if sc['gpu_active'] else '未启用'}"
          f"  实测 {sc['mean_ms']} ms/帧")
    print(f"检测后端：{det.name}（{note}）  路数：{len(srcs)}  batch={args.batch}")

    import numpy as np
    det.detect_frame(np.zeros((640, 640, 3), dtype=np.uint8))   # 预热

    sch = MultiStreamScheduler(det, batch=args.batch)
    for i, s in enumerate(srcs):
        sch.add(StreamReader(f"S{i + 1}", s, regions[i % len(regions)],
                             fps=args.fps, max_seconds=args.max_seconds or None,
                             motion_threshold=args.motion))
    stats = sch.run()

    print("\n" + "=" * 62)
    print(f"采样 {stats['samples']} · 推理 {stats['inferred']} · 静止跳过 {stats['skipped']} · "
          f"无人跳过 PPE {stats.get('gate_skipped', 0)} · 拍数 {stats['ticks']}（最大批 {stats['batch_max']}）")
    print(f"平均每帧推理 {stats['avg_infer_ms']} ms · 吞吐 {stats['throughput_fps']} fps · "
          f"实耗 {stats['wall_s']}s")
    print(f"告警事件 {stats['events']} 条")
    for e in sch.events:
        print(f"  {e['stream']}  {e['t_video_s']:>5.1f}s  {e['track']}  "
              f"{e['grade']}R{e['R']}  {e['violation']}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"sources": srcs, "batch": args.batch, "fps": args.fps,
             "stats": stats, "events": sch.events}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
