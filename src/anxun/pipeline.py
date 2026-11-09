# -*- coding: utf-8 -*-
"""
安巡 Agent · 编排流水线
=======================

一次巡检 = 抽帧 → 检测 → 四态判定 → 时序聚合 → 风险定级 → 工单。

技术线的职责止于「产出结构化结果」：判定、定级、工单都是确定性的，
谁来消费（控制台 / 扣子工作流 / 报告生成）都能拿到同一份 JSON。

单张照片只有 1 帧，**不做时序聚合**——静态图没有时序信息，硬凑等于编证据。
视频流进来才走 3 帧。
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .rules_engine import (Box, Region, Event, aggregate_temporal, build_event,
                           judge_frame, make_ticket, grade_of, GRADE_HEX,
                           min_person_area, profile_snapshot, L_TABLE, S_MAP)

FRAME_COUNT = 3          # 视频流的时序窗口：连续 3 帧一致才定论
GRADE_ORDER = {"蓝色": 1, "黄色": 2, "橙色": 3, "红色": 4}


def thresholds() -> dict:
    """把当前生效的阈值原样吐出去，答辩时可以直接指着屏幕念依据。

    数据源是 config/thresholds.json（经 rules_engine.profile_snapshot()），
    不在本文件里另抄一份——抄一份就是下一个「两套阈值」。
    """
    snap = profile_snapshot()
    snap["temporal_need"] = FRAME_COUNT
    return snap


def inspect(scene: dict, region: Region, detector,
            ts: datetime | None = None,
            frames: int | None = None,
            equips: Sequence[str] | None = None) -> dict:
    """跑完整条链路，返回结构化结果（不含落盘、不含展示层）。

    equips 显式传入时以它为准（调用方按模型实际能力过滤后的结果）；
    不传才用场景里写的。见 detectors.available_equips()。
    """
    ts = ts or datetime.now()
    if equips is None:
        # 场景写的是「想判什么」，模型决定「能判什么」。这里兜一道底：
        # 调用方忘了过滤时，缺权重也不会把「一个框都没有」判成「没佩戴」。
        equips = list(scene.get("equips", []))
        try:
            from .detectors import available_equips
            can = available_equips(detector)
        except Exception:
            can = None
        if can is not None:
            equips = [e for e in equips if e in can]
    else:
        equips = list(equips)
    images = list(scene.get("images") or [])
    requested = int(frames or scene.get("frames", 1))
    # 只有一张图时，帧数硬凑成 3 只会把同一张图推理三遍，
    # 「连续 3 帧一致」必然成立——那是编证据，不是时序确认。
    # 真要多帧就把多张图的路径写进 scene["images"]，否则一律按单帧处理。
    if requested > 1 and not images:
        requested = 1
    if images:
        requested = min(requested, len(images))
    requested = max(1, requested)          # frames=0 也得留一帧，否则下面取不到 last

    per_frame = []
    for f in range(requested):
        det = (detector.detect_image(images[f], f) if images
               else detector.detect(scene, f))
        boxes: list[Box] = det["boxes"]
        # 面积阈值按实际画面尺寸换算，不拿 1080p 的绝对值套所有分辨率
        fw = det.get("width") or scene.get("width") or 1280
        fh = det.get("height") or scene.get("height") or 720
        area_min = min_person_area(fw, fh)
        verdicts = judge_frame(boxes, equips, area_min=area_min) if equips else []
        per_frame.append({
            "idx": f,
            "boxes": [b.to_dict() for b in boxes],
            "verdicts": [v.to_dict() for v in verdicts],
            "infer_ms": det["infer_ms"],
            "frame_w": fw, "frame_h": fh, "area_min": round(area_min),
            "_raw_verdicts": verdicts,
            "_raw_boxes": boxes,
        })

    last_boxes = per_frame[-1]["_raw_boxes"]
    last_verdicts = per_frame[-1]["_raw_verdicts"]

    if requested <= 1 or not equips:
        final = last_verdicts if equips else []
    else:
        final = aggregate_temporal([p["_raw_verdicts"] for p in per_frame], need=requested)

    # ---- 按装备类型归并违规，一类违规一张工单 ----
    events, tickets = [], []
    grouped: dict[str, list] = {}
    for v in final:
        if v.is_violation:
            grouped.setdefault(v.equip, []).append(v)

    for equip, vs in grouped.items():
        ev = build_event(region, equip, ts, max(v.conf for v in vs),
                         f"{scene['scene_id']} · frame {requested - 1} · {len(vs)} 人",
                         event_id=f"EV-{ts:%Y%m%d%H%M%S}-{region.region_id}-{equip}-{_suffix(scene)}")
        events.append(_event_dict(ev))
        tickets.append(make_ticket(ev, affected=len(vs)))

    for t in tickets:
        t["scene_id"] = scene["scene_id"]

    persons = []
    for i, b in enumerate([b for b in last_boxes if b.label == "person"]):
        pid = f"P{i + 1}"
        items = [v.to_dict() for v in final if v.person_id == pid]
        persons.append({"id": pid, "box": b.to_dict(), "items": items,
                        "worst": _worst(items)})

    for p in per_frame:
        p.pop("_raw_verdicts", None)
        p.pop("_raw_boxes", None)

    return {
        "scene": {k: v for k, v in scene.items()
                  if k in ("scene_id", "camera", "title", "hint", "region_id", "equips")},
        "region": region.to_dict(),
        "ts": ts.isoformat(timespec="seconds"),
        "frames": per_frame,
        "temporal": {
            "used": requested,                  # 实际参与判定的帧数
            "requested": int(frames or scene.get("frames", 1)),
            "aggregated": requested > 1,        # 没聚合就别在界面上写「3 帧一致」
            "source": "images" if images else "single",
        },
        "equips": equips,                        # 实际判定的装备（已按模型能力过滤）
        "final": [v.to_dict() for v in final],
        "persons": persons,
        "events": events,
        "tickets": tickets,
        "summary": _summary(final, events, persons),
        "thresholds": thresholds(),
        "detector": {"backend": detector.name, "kind": detector.kind},
    }


def _suffix(scene: dict) -> str:
    """场景后缀，让同区域不同场景的工单不撞号。"""
    return scene["scene_id"].split("-")[-1]


def _event_dict(ev: Event) -> dict:
    return {
        "event_id": ev.event_id,
        "equip": ev.equip,
        "L": ev.L, "S": ev.S, "R": ev.R,
        "grade": ev.grade,
        "grade_hex": GRADE_HEX.get(ev.grade, "#888"),
        "action": ev.action,
        "due_at": ev.due_at.isoformat(timespec="minutes"),
        "conf": round(ev.conf, 3),
        "evidence": ev.evidence,
        "rule": _rule_text(ev),
    }


def _rule_text(ev: Event) -> dict:
    """把 L/S 的查表过程摊开，回答「为什么是这个等级」。"""
    l_rows = [{"range": f"≤{cap}", "L": val} for cap, val in L_TABLE]
    l_rows.append({"range": f">{L_TABLE[-1][0]}", "L": 4})
    s_why = f"作业类型 {ev.region.scene} → S={ev.S}"
    if ev.region.scene not in S_MAP:
        s_why += "（作业类型未登记，按最低档 S=2）"
    return {
        "L_why": f"{ev.region.name} 近 30 天违规 {ev.region.violation_30d} 次 → L={ev.L}",
        "S_why": s_why,
        "R_why": f"R = L × S = {ev.L} × {ev.S} = {ev.R}",
        "L_table": l_rows,
        "S_table": [{"k": k, "S": v} for k, v in S_MAP.items()],
    }


def _worst(items: list[dict]) -> str:
    order = ["未佩戴", "待复核", "无法判定", "已佩戴", "丢弃"]
    for st in order:
        if any(i["status"] == st for i in items):
            return st
    return "-"


def _summary(final, events, persons) -> dict:
    cnt: dict[str, int] = {}
    for v in final:
        cnt[v.status] = cnt.get(v.status, 0) + 1
    worst = ""
    if events:
        worst = max(events, key=lambda e: GRADE_ORDER.get(e["grade"], 0))["grade"]
    return {
        "persons": len(persons),
        "states": cnt,
        "violations": sum(1 for v in final if v.is_violation),
        "review": cnt.get("待复核", 0),
        "undetermined": cnt.get("无法判定", 0),
        "discarded": cnt.get("丢弃", 0),
        "events": len(events),
        "worst_grade": worst,
    }
