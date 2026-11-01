# -*- coding: utf-8 -*-
"""
安巡 Agent · 视频巡查
=====================

现场要的是**盯着监控视频**，不是一张张看照片。这个文件把整条链路接到视频流上：

    抽帧 → 双模型检测 → 四态判定 → 跨帧跟踪 → 3 帧定论 → 事件去重 → 输出

三个针对视频的关键设计
----------------------
1. **抽帧，不逐帧**：25fps 全跑既没必要也跑不动。默认 2fps，一秒看两帧足够定论。
   连 3 帧一致才判违规，等价于「稳定 1.5 秒」——比单帧告警稳得多。
2. **跨帧跟踪**：同一个人在多帧之间要有身份，否则每帧都是一张新面孔，
   时序聚合就无从谈起。这里用 IoU 贪心匹配，轻量、够用。
3. **事件去重**：同一个人同一个违规，冷却期内只报一次。否则一个人站 10 分钟，
   会生成 600 条告警——现场最烦的就是这个。

用法
----
    python src/anxun/stream.py --source demo.mp4 --fps 2 --region B-01
    python src/anxun/stream.py --source rtsp://user:pass@10.0.0.8/stream1 --fps 1

输出事件流（JSON）到 stdout 或 --out 文件，不落盘工单——派单是系统线的事。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anxun.catalog import REGION_MAP
from anxun import backend as backend_mod
from anxun.detectors import available_equips
from anxun.rules_engine import (Box, Verdict, aggregate_temporal, build_event,
                                judge_frame, make_ticket, min_person_area)

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 一、跟踪：给每帧里的人一个稳定身份

def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua, ub = a.area, b.area
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


class Track:
    __slots__ = ("tid", "box", "verdicts", "miss", "last_event_at")

    def __init__(self, tid: int, box: Box):
        self.tid = tid
        self.box = box
        self.verdicts: deque[list[Verdict]] = deque(maxlen=3)
        self.miss = 0                  # 连续未匹配次数
        self.last_event_at: dict[str, float] = {}

    def push(self, verdicts: list[Verdict]) -> None:
        self.verdicts.append(verdicts)


class Tracker:
    """IoU 贪心匹配。轻量到可以在 CPU 上跟着 25fps 跑，够用就好。"""

    def __init__(self, iou_min: float = 0.30, max_miss: int = 2):
        self.iou_min = iou_min
        self.max_miss = max_miss
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, persons: list[Box]) -> list[tuple[Track, int]]:
        # 老轨先按 IoU 贪心分配
        pairs = []
        for tid, tr in self.tracks.items():
            for i, p in enumerate(persons):
                v = _iou(tr.box, p)
                if v >= self.iou_min:
                    pairs.append((v, tid, i))
        pairs.sort(reverse=True)

        used_t, used_p, matched = set(), set(), {}
        for v, tid, i in pairs:
            if tid in used_t or i in used_p:
                continue
            used_t.add(tid)
            used_p.add(i)
            tr = self.tracks[tid]
            tr.box = persons[i]
            tr.miss = 0
            matched[tid] = i

        # 没匹配上的老轨算丢失，连丢 2 次就删
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].miss += 1
                if self.tracks[tid].miss > self.max_miss:
                    del self.tracks[tid]

        # 新出现的人开新轨
        for i, p in enumerate(persons):
            if i in used_p:
                continue
            tid = self._next
            self._next += 1
            self.tracks[tid] = Track(tid, p)
            matched[tid] = i

        return [(self.tracks[t], matched[t]) for t in matched]


# ============================================================
# 二、每路的判定状态机（与推理解耦）
#
# 这一层不碰模型：只吃「一帧的检测框 + 时间戳」，吐「事件」。
# 解耦之后，推理可以是逐帧的（VideoInspector），也可以是
# 多路攒批的（scheduler.py），判定行为完全一致。

class StreamPipeline:
    """一路视频的判定状态机：跟踪 → 四态判定 → 三帧定论 → 冷却去重。"""

    def __init__(self, sid: str, region, equips: tuple[str, ...] = ("helmet",),
                 cooldown_s: float = 60.0, verbose: bool = True,
                 ledger: bool = True):
        self.sid = sid
        self.region = region
        self.equips = equips
        self.cooldown_s = cooldown_s
        self.verbose = verbose
        self.tracker = Tracker()
        self.events: list[dict] = []
        # ---- v2.0 判定账本 ----
        # v1.x 只在 is_violation 时落一条事件，合规的人一条记录都不留，
        # 于是「合规率」没有分母，「降噪率」没有分子（去重前有多少帧判了违规）。
        # 这里把每一次定论都记下来，违规与否一样对待。
        self.ledger_on = ledger
        self.ledger: list[dict] = []
        self.raw_alarms = 0        # 逐帧判违规的次数（冷却去重前）
        self.persons_total = 0     # 进入画面的轨迹数（人次，不等于独立人数）
        self._last: dict[tuple[int, str], dict] = {}
        self._seen: set[int] = set()

    def post(self, boxes: list[Box], width: int, height: int,
             t_video: float) -> list[dict]:
        area_min = min_person_area(width, height)
        verdicts = judge_frame(boxes, self.equips, area_min=area_min)
        persons = [b for b in boxes if b.label == "person"]
        tracks = self.tracker.update(persons)

        out = []
        for tr, idx in tracks:
            if tr.tid not in self._seen:
                self._seen.add(tr.tid)
                self.persons_total += 1
            mine = [v for v in verdicts if v.person_idx == idx]
            mine = [Verdict(v.person_idx, v.equip, v.status, v.conf, v.reason,
                            v.is_violation, person_id=f"T{tr.tid}", iou=v.iou)
                    for v in mine]
            tr.push(mine)

            if len(tr.verdicts) < 3:
                continue
            # 复用引擎里的时序聚合：三帧一致才定论
            final = aggregate_temporal(list(tr.verdicts), need=3)
            for v in final:
                # 账本先记：合规、违规、待复核、无法判定一视同仁
                if self.ledger_on:
                    self._book(tr, v, t_video)
                if not v.is_violation:
                    continue
                self.raw_alarms += 1
                key = v.equip
                last = tr.last_event_at.get(key, -1e9)
                if t_video - last < self.cooldown_s:
                    continue
                tr.last_event_at[key] = t_video
                ev = build_event(self.region, v.equip, datetime.now(), v.conf,
                                 f"{self.sid} track T{tr.tid} @ {t_video:.1f}s",
                                 event_id=f"EV-{datetime.now():%Y%m%d%H%M%S}-"
                                          f"{self.region.region_id}-{v.equip}-{self.sid}-T{tr.tid}")
                d = _event_dict(ev, v, tr, t_video)
                d["stream"] = self.sid
                self.events.append(d)
                out.append(d)
                if self.verbose:
                    print(f"  ⚠ {self.sid} {t_video:6.1f}s  T{tr.tid}  {d['grade']}R{d['R']}  "
                          f"{d['violation']}  置信度 {v.conf:.2f}")
        return out

    def _book(self, tr, v, t_video: float) -> None:
        """记一条账：同一人同一装备，状态变了才新增，否则只延长观测窗口。

        为什么不逐帧记 —— 2fps × 60s = 120 条/路；一个 20 路的园区跑一天就是
        345 万条，账本自己先成了负担。按「状态变化」记账，条数等于这个人换了
        几次状态，量级才对得上运营口径。

        t_video_s 是该状态**第一次**被定论的时刻，t_last 是最后一次仍成立的时刻，
        两个都在，才能算「这个人违规持续了多久」。
        """
        key = (tr.tid, v.equip)
        rec = self._last.get(key)
        if rec is not None and rec["status"] == v.status:
            rec["t_last"] = round(t_video, 1)
            rec["observations"] += 1
            return
        rec = {
            "t_video_s": round(t_video, 1),
            "t_last": round(t_video, 1),
            "track": f"T{tr.tid}",
            "equip": v.equip,
            "status": v.status,
            "conf": round(v.conf, 3),
            "violation": v.is_violation,
            "reason": v.reason,
            "region": self.region.name,
            "stream": self.sid,
            "observations": 1,
        }
        self.ledger.append(rec)
        self._last[key] = rec


# ============================================================
# 三、巡查器（单路，逐帧推理）

class VideoInspector:
    def __init__(self, detector, region, fps: float = 2.0,
                 equips: tuple[str, ...] = ("helmet", "vest"),
                 cooldown_s: float = 60.0, motion_threshold: float = 10.0,
                 verbose: bool = True):
        self.det = detector
        self.region = region
        self.interval = 1.0 / max(0.1, fps)
        # 判定范围要跟模型能力对齐：vest 权重缺失时模型不产背心框，
        # 硬判「未穿戴」等于把所有路过的人都报一遍。
        can = available_equips(detector)
        self.dropped_equips: tuple[str, ...] = ()
        if can is not None:
            self.dropped_equips = tuple(e for e in equips if e not in can)
            equips = tuple(e for e in equips if e in can)
        self.equips = equips
        self.cooldown_s = cooldown_s
        # 3.0 -> 10.0（2026-11-01，1080p 真实素材复验后调整）
        # 原值按 640 合成动画定。换成 1920x1920 真实照片 + ±1.5px 机位抖动后，
        # 帧差中位数从 ~0 抬到 3.86 —— 真实固定机位的轻微晃动（风、振动）本身就
        # 超过 3.0，触发层的帧差判据在现场会失效：实测节省从 90% 掉到 39%。
        # 阈值扫到 10.0：抖动版节省回到 86%，告警与全量完全一致（miss=0）。
        # 边界：场景切换帧差 88+，人走动的帧差远大于机位晃动的 3.86，10.0 有余量。
        # ⚠️ 复验素材是静态照片，没有真实人体运动；上限需现场录像确认。
        self.motion_threshold = motion_threshold   # 0 = 关闭帧差跳帧
        self.verbose = verbose
        if self.dropped_equips and verbose:
            print(f"  [warn] 缺权重组件，跳过判定：{', '.join(self.dropped_equips)}")
        self.pipe = StreamPipeline("main", region, equips, cooldown_s, verbose)
        self.events = self.pipe.events          # 兼容旧引用
        self._prev_gray = None
        self._last_boxes = None
        self._last_size = (1920, 1080)
        self.stats = {"frames_read": 0, "frames_processed": 0, "frames_skipped": 0,
                      "gate_skipped": 0, "infer_ms_total": 0.0, "events": 0,
                      "raw_alarms": 0, "persons_total": 0, "ledger": 0,
                      "realtime_factor": 0.0}

    # ---------- 帧差：静止画面直接跳过 ----------
    def _still(self, frame) -> bool:
        """
        固定摄像头下，画面大部分时间是空的（没人经过）。
        灰度均值差小于阈值就认为画面没变，跳过推理，省掉这一帧的全部算力。
        跳过时**不动跟踪状态**——人还在原地站着，轨迹和时序窗口都要保留。
        """
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

    # ---------- 单帧 ----------
    def _process(self, frame, t_video: float) -> list[dict]:
        """
        静止画面不推理，但**复用上一帧的检测结果继续推进时序窗口**。

        这是踩过的坑：一开始跳过就整帧丢掉，结果画面不动时三帧永远凑不齐，
        一条告警都出不来。人站在那儿不动，也得能判出他没戴帽。
        """
        still = self._still(frame)
        if still and self._last_boxes is not None:
            det = {"boxes": self._last_boxes, "infer_ms": 0.0,
                   "width": self._last_size[0], "height": self._last_size[1]}
            self.stats["frames_skipped"] += 1
        else:
            det = self.det.detect_frame(frame)
            self._last_boxes = det["boxes"]
            self._last_size = (det.get("width") or 1920, det.get("height") or 1080)
            self.stats["infer_ms_total"] += det["infer_ms"]
            self.stats["frames_processed"] += 1

        boxes: list[Box] = det["boxes"]
        width = det.get("width") or 1920
        height = det.get("height") or 1080
        out = self.pipe.post(boxes, width, height, t_video)
        self.stats["events"] += len(out)
        return out

    # ---------- 整个视频 ----------
    def run(self, source: str, max_seconds: float | None = None,
            source_fps: float | None = None) -> dict:
        import cv2
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"打不开视频源：{source}")
        fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if self.verbose:
            print(f"视频：{Path(source).name}  {fps:.0f}fps  "
                  f"{(total / fps if fps else 0):.1f}s  抽帧 {1 / self.interval:.1f}fps")

        next_t, t0 = 0.0, time.perf_counter()
        step = max(1, int(round(fps * self.interval)))   # 每读 step 帧处理一帧
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.stats["frames_read"] += 1
            t_video = i / fps
            if max_seconds and t_video > max_seconds:
                break
            if t_video + 1e-6 >= next_t:
                self._process(frame, t_video)
                next_t += self.interval
            i += 1
        cap.release()

        wall = time.perf_counter() - t0
        video_s = (i / fps) if fps else 0
        self.stats["realtime_factor"] = round(video_s / wall, 2) if wall else None
        self.stats["gate_skipped"] = getattr(self.det, "gate_skipped", 0)
        self.stats["wall_s"] = round(wall, 1)
        self.stats["video_s"] = round(video_s, 1)
        self.stats["avg_infer_ms"] = (round(self.stats["infer_ms_total"]
                                            / max(1, self.stats["frames_processed"]), 1))
        self.stats["raw_alarms"] = self.pipe.raw_alarms
        self.stats["persons_total"] = self.pipe.persons_total
        self.stats["ledger"] = len(self.pipe.ledger)
        return self.stats


def _event_dict(ev, v, tr, t_video: float) -> dict:
    t = make_ticket(ev, affected=1)
    return {
        "t_video_s": round(t_video, 1),
        "track": f"T{tr.tid}",
        "equip": v.equip,
        "violation": t["violation"],
        "conf": round(v.conf, 3),
        "L": ev.L, "S": ev.S, "R": ev.R,
        "grade": ev.grade,
        "action": ev.action,
        "assignee": t["assignee"],
        "region": ev.region.name,
        "region_id": ev.region.region_id,
    }


# ============================================================
# 三、命令行

def main() -> int:
    ap = argparse.ArgumentParser(description="安巡 Agent 视频巡查")
    ap.add_argument("--source", required=True, help="视频文件路径或 RTSP 地址")
    ap.add_argument("--region", default="B-01", help="区域编号，决定 L/S 与责任人")
    ap.add_argument("--fps", type=float, default=2.0, help="抽帧频率（默认 2）")
    ap.add_argument("--cooldown", type=float, default=60.0, help="同一人同一违规的告警冷却秒数")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--motion", type=float, default=10.0,
                    help="帧差阈值（160x160 灰度均值差），低于此值跳过推理（0=关闭）。"
                         "1080p 真实素材复验后从 3.0 提到 10.0，见 VideoInspector 注释")
    ap.add_argument("--no-gate", action="store_true",
                    help="关闭人员门控（没人也跑 PPE），用于全量对照")
    ap.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--person-model", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "onnx"))
    ap.add_argument("--device", default="", help="cpu 强制 CPU；留空自动")
    ap.add_argument("--vest-model", default="", help="反光背心权重；留空自动")
    ap.add_argument("--out", default="", help="事件流输出 json 路径")
    ap.add_argument("--no-ledger", action="store_true",
                    help="不记判定账本（退回 v1.x 行为，只输出违规事件）")
    args = ap.parse_args()

    region = REGION_MAP.get(args.region)
    if not region:
        print(f"未知区域 {args.region}，可选 {list(REGION_MAP)}")
        return 2

    det, note, _ = backend_mod.build(prefer=args.backend, imgsz=args.imgsz,
                                     device=args.device,
                                     model=args.model, person_model=args.person_model,
                                     vest_model=args.vest_model)
    if getattr(det, "gate", None) is not None:
        det.gate = not args.no_gate
    sc = backend_mod.selfcheck(det)
    print(f"设备：{sc['device']}  GPU{'已启用' if sc['gpu_active'] else '未启用'}"
          f"  实测 {sc['mean_ms']} ms/帧")
    print(f"检测后端：{det.name}（{note}）  区域：{region.name}")

    # 预热：第一次推理含内存分配，不计入统计
    import numpy as _np
    det.detect_frame(_np.zeros((640, 640, 3), dtype=_np.uint8))

    insp = VideoInspector(det, region, fps=args.fps, cooldown_s=args.cooldown,
                          motion_threshold=args.motion)
    insp.pipe.ledger_on = not args.no_ledger
    stats = insp.run(args.source, max_seconds=args.max_seconds or None)

    print("\n" + "=" * 62)
    print(f"读帧 {stats['frames_read']} · 推理 {stats['frames_processed']} · "
          f"静止跳过 {stats['frames_skipped']} · 无人跳过 PPE {stats.get('gate_skipped', 0)} · "
          f"平均推理 {stats['avg_infer_ms']} ms")
    print(f"视频 {stats['video_s']}s / 实耗 {stats['wall_s']}s  "
          f"→ 实时倍速 {stats['realtime_factor']}×")
    print(f"告警事件 {stats['events']} 条"
          f"（去重前逐帧 {stats['raw_alarms']} 次 · 画面人次 {stats['persons_total']} · "
          f"账本 {stats['ledger']} 条）")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"source": args.source, "region": region.name, "fps": args.fps,
             "stats": stats, "events": insp.events,
             "ledger": insp.pipe.ledger}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
