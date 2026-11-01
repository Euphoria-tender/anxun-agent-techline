# -*- coding: utf-8 -*-
"""
安巡 Agent · 价值指标
======================

v1.x 输出的是**告警清单**——一条条「谁违规」。它回答不了运营会问的三个问题：
这个月合规率多少、巡检到底覆盖了多少时间、一条告警能压到多少条。

这个文件从判定账本（`stream.py` 的 `ledger`）算出五个数字：

    巡检覆盖率   系统看得见的时长 / 人工看得见的时长
    判定覆盖率   有定论的轨迹 / 出现的人次        ← v2.0 才有，v1.x 看不见的盲区
    告警降噪率   1 - 事件数 / 逐帧告警数
    合规率       合规人次 / 可判定人次
    违规持续时长 单次违规从定论到消失，P50 / P95   ← 整改闭环的降级口径

为什么合规率要按「轨迹」而不是按「账本条数」算
-----------------------------------------------
账本按状态变化记，一个人站在那儿从「待复核」抖到「已佩戴」再抖回来，
会产生三条记录。按条数算等于让「抖得厉害的人」占更多权重。
所以先把每条轨迹的每种装备收敛成一个**主导状态**（累计观测时长最长的那个），
再按轨迹判合规 / 违规 —— 一个人算一次，不按他抖了几次算。

口径与 v1.2.0 的 `matched` 保持一致
------------------------------------
「待复核」「无法判定」不进合规率分母，和误报率的 `matched` 口径同源：
只统计判得动的人次。分母被排除多少，由「判定覆盖率」单独报，不藏在合规率里。

用法
----
    python src/anxun/stats.py --events data/events_site.json --out data/value_metrics.json
    python src/anxun/stats.py --events a.json b.json --out data/value_metrics.json --hours 8
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# 判定状态。与 rules_engine.Verdict.status 的取值一致，改动要同步。
OK, BAD, REVIEW, UNKNOWN = "已佩戴", "未佩戴", "待复核", "无法判定"

SCHEMA = 1


# ============================================================
# 一、读入与合并

def load(path: str | Path) -> dict:
    """读一份事件 JSON（stream.py / scheduler.py 的 --out 产物）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "events" not in data:
        raise ValueError(f"{path} 不像事件文件：没有 events 字段")
    if "ledger" not in data:
        raise ValueError(
            f"{path} 没有 ledger 字段 —— 它是 v1.x 的产物，只有违规事件、没有全量判定，"
            f"算不出合规率。请用 v2.0 的 stream.py 重跑（或加上 --no-ledger 之外的默认参数）。")
    return data


def merge(bundles: Iterable[dict]) -> dict:
    """把多路 / 多段视频的账本合并成一份。

    合并的是账本与事件，不是指标——先合再算，和「先算再平均」不等价。
    后者会被短视频段稀释（一段 10 秒的视频和一段 10 分钟的视频权重一样）。

    ⚠️ 每个文件的轨迹编号都从 T1 开始，直接拼会把两份素材里的「T1」当成
    同一个人，合规率的分母凭空消失一半。所以按文件序号加 `s0:` 前缀。

    ⚠️ 每份文件的时间轴也都从 0 开始。不加偏移直接拼，趋势图会变成好几段
    视频叠在一起（0–30s 的桶里塞着两条不同素材的人），不是一条连续巡检线。
    这里按累积时长平移，合并后的时间轴 = 「这次巡检开始后第 N 秒」。
    """
    ledger: list[dict] = []
    events: list[dict] = []
    stats: dict[str, Any] = defaultdict(float)
    sources: list[str] = []
    video_s = 0.0
    streams: set[str] = set()

    for i, b in enumerate(bundles):
        tag = f"s{i}"
        s = b.get("stats") or {}
        offset = video_s                                  # 这份素材在总时间轴上的起点
        seg_s = float(s.get("video_s") or 0.0)
        for r in b["ledger"]:
            row = dict(r)                      # 不改调用方传进来的对象
            row["track"] = f"{tag}:{r['track']}"
            row["t_video_s"] = round(float(r["t_video_s"]) + offset, 1)
            row["t_last"] = round(float(r["t_last"]) + offset, 1)
            row["source_index"] = i
            ledger.append(row)
            streams.add(r.get("stream") or "main")
        for e in b["events"]:
            row = dict(e)
            row["track"] = f"{tag}:{e['track']}"
            row["t_video_s"] = round(float(e.get("t_video_s") or 0) + offset, 1)
            row["source_index"] = i
            events.append(row)
        sources.append(str(b.get("source") or b.get("sources") or "?"))
        video_s += seg_s
        for k in ("persons_total", "raw_alarms", "events", "frames_processed",
                  "frames_read", "frames_skipped"):
            stats[k] += float(s.get(k) or 0)

    stats["persons_total"] = int(stats["persons_total"])
    stats["raw_alarms"] = int(stats["raw_alarms"])
    stats["frames_processed"] = int(stats["frames_processed"])
    stats["frames_read"] = int(stats["frames_read"])
    stats["frames_skipped"] = int(stats["frames_skipped"])
    stats["video_s"] = round(video_s, 1)
    stats["events"] = len(events)
    stats["ledger"] = len(ledger)
    stats["streams"] = len(streams)
    return {"sources": sources, "stats": dict(stats),
            "events": events, "ledger": ledger}


# ============================================================
# 二、单个指标

def _pct(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数。q ∈ [0,1]。纯标准库，不引 numpy。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 2)


def _dominant(rows: list[dict]) -> str:
    """一条轨迹上某种装备的主导状态：累计观测时长最长的那个。

    全是瞬时记录（时长都是 0）时退回「观测次数最多」，再退回「最后一条」。
    这样单帧素材也能给出一个稳定答案，不会随机取。
    """
    score: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    order: list[str] = []
    for r in rows:
        st = r.get("status") or UNKNOWN
        if st not in order:
            order.append(st)
        score[st] += max(0.0, float(r.get("t_last", 0)) - float(r.get("t_video_s", 0)))
        count[st] += int(r.get("observations") or 1)

    if sum(score.values()) > 0:
        return max(order, key=lambda s: (score[s], count[s]))
    return max(order, key=lambda s: count[s])


def track_states(ledger: list[dict]) -> dict[str, dict[str, str]]:
    """把账本收敛成 {轨迹: {装备: 主导状态}}。"""
    by_track: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in ledger:
        by_track[r["track"]][r["equip"]].append(r)
    return {t: {eq: _dominant(rows) for eq, rows in eqs.items()}
            for t, eqs in by_track.items()}


def compliance(ledger: list[dict]) -> dict:
    """合规率 = 合规人次 / 可判定人次。

    一个人穿了两样装备，只要有一样判「未佩戴」就算违规；
    两样都「已佩戴」才算合规；含「待复核 / 无法判定」且无违规的，单列不计分。
    """
    ok = bad = review = unknown = 0
    for _t, equips in track_states(ledger).items():
        vals = list(equips.values())
        if any(v == BAD for v in vals):
            bad += 1
        elif vals and all(v == OK for v in vals):
            ok += 1
        elif UNKNOWN in vals:
            unknown += 1
        else:
            review += 1
    n = ok + bad
    return {
        "value": round(ok / n, 4) if n else None,
        "ok": ok, "violation": bad, "review": review, "unknown": unknown,
        "n": n,
        "note": "分母只含判得动的人次；待复核与无法判定单列，不藏进分母",
    }


def denoise(raw_alarms: int, events: int) -> dict:
    """告警降噪率 = 1 - 事件数 / 逐帧告警数。

    分子是冷却去重后的事件，分母是三帧定论后逐帧判违规的次数。
    它衡量的是「同一个人站十分钟只报一次」这件事到底省了多少。
    """
    if raw_alarms <= 0:
        return {"value": None, "raw": raw_alarms, "events": events,
                "note": "没有逐帧告警，无法计算"}
    return {
        "value": round(1 - events / raw_alarms, 4),
        "raw": raw_alarms, "events": events,
        "note": "1 - 去重后事件数 / 去重前逐帧告警次数",
    }


def verdict_coverage(ledger: list[dict], persons_total: int) -> dict:
    """判定覆盖率 = 有定论的轨迹 / 出现的人次。

    v1.x 看不见这个数：只报违规，没人管「多少人根本没被判定」。
    它低说明轨迹碎（分辨率不够 / 人走得太快 / 三帧定论凑不齐），
    和模型准不准是两件事 —— 模型再准，判不到的人也进不了合规率。
    """
    judged = len(track_states(ledger))
    total = max(judged, int(persons_total or 0))
    return {
        "value": round(judged / total, 4) if total else None,
        "judged": judged, "persons_total": total,
        "note": "有定论轨迹 / 进入画面的轨迹数；低说明轨迹碎片化，不是模型不准",
    }


def durations(ledger: list[dict]) -> dict:
    """违规持续时长：单次违规从定论到消失的秒数，取 P50 / P95。

    这是「闭环时长」在没有整改过程时的降级口径 —— 素材里只有人走过，
    没有「戴上帽子再走」的整改动作，所以只能算告警本身的持续时间。
    同一口径在真实现场（有整改）下会自动变成「违规 → 合规」的时长，
    因为账本里会出现一条 `已佩戴` 记录，两者之差就是整改耗时。
    """
    vals = sorted(float(r["t_last"]) - float(r["t_video_s"])
                  for r in ledger
                  if r.get("violation") and float(r.get("t_last", 0)) > float(r.get("t_video_s", 0)))
    return {
        "p50": _pct(vals, 0.50), "p95": _pct(vals, 0.95),
        "max": round(vals[-1], 2) if vals else 0.0,
        "n": len(vals),
        "note": "单次违规的持续时间；素材无整改过程，故未体现「违规→合规」的整改耗时",
    }


def coverage(streams: int, hours: float = 8.0,
             manual_times: float = 2, manual_minutes: float = 15) -> dict:
    """巡检覆盖率 = 系统看得见的时长 / 人工看得见的时长。

    **这是推演值，不是实测值。** 系统侧按「每路每小时都在看」折算；
    人工侧按「每班次 N 次、每次 M 分钟」的默认假设。两个参数都能改，
    报告里必须标注是排班假设 —— 换成真实巡检制度后重算。
    """
    sys_min = max(0.0, streams) * hours * 60.0
    man_min = max(0.0, manual_times) * max(0.0, manual_minutes)
    return {
        "value": round(sys_min / man_min, 1) if man_min else None,
        "system_minutes": round(sys_min, 1),
        "manual_minutes": round(man_min, 1),
        "streams": streams, "hours": hours,
        "assumption": f"人工每班 {manual_times:g} 次 × {manual_minutes:g} 分钟，"
                      f"单路连续 {hours:g} 小时",
        "note": "推演值：人工基线是排班假设，不是实测；换真实制度后重算",
    }


# ============================================================
# 三、聚合视图

def region_ranking(events: list[dict]) -> list[dict]:
    """区域风险排行：按 R = L × S 加权的事件数降序。"""
    agg: dict[str, dict] = {}
    for e in events:
        name = e.get("region") or "未分区"
        a = agg.setdefault(name, {"region": name,
                                  "region_id": e.get("region_id") or "",
                                  "events": 0, "r_sum": 0,
                                  "grades": defaultdict(int)})
        a["events"] += 1
        a["r_sum"] += int(e.get("R") or 0)
        a["grades"][e.get("grade") or "未定级"] += 1
    rows = [{"region": a["region"], "region_id": a["region_id"],
             "events": a["events"], "r_sum": a["r_sum"],
             "r_avg": round(a["r_sum"] / a["events"], 1) if a["events"] else 0,
             "grades": dict(a["grades"])} for a in agg.values()]
    rows.sort(key=lambda x: (-x["r_sum"], -x["events"]))
    return rows


def trend(ledger: list[dict], bucket_s: float = 30.0) -> list[dict]:
    """按时段分桶的合规率曲线，给报告画折线用。"""
    if not ledger or bucket_s <= 0:
        return []
    t_end = max(float(r["t_last"]) for r in ledger)
    nb = max(1, int(t_end // bucket_s) + 1)
    judged: dict[int, set[str]] = defaultdict(set)
    bad: dict[int, set[str]] = defaultdict(set)
    for r in ledger:
        b = min(nb - 1, int(float(r["t_video_s"]) // bucket_s))
        judged[b].add(r["track"])
        if r.get("violation"):
            bad[b].add(r["track"])
    out = []
    for b in range(nb):
        n = len(judged[b])
        out.append({
            "t_start_s": round(b * bucket_s, 1),
            "t_end_s": round(min((b + 1) * bucket_s, t_end), 1),
            "persons": n,
            "violations": len(bad[b]),
            "compliance": round(1 - len(bad[b]) / n, 3) if n else None,
        })
    return out


# ============================================================
# 四、总装与命令行

def compute(bundles: list[dict], *, hours: float = 8.0,
            manual_times: float = 2, manual_minutes: float = 15,
            bucket_s: float = 30.0, streams: int | None = None) -> dict:
    """把 N 份事件文件算成一份价值指标。

    `streams` 不传时按账本里的 stream 名去重计数：单路 stream.py 跑几段都是
    1 路（同一台摄像头的不同时段）；多路 scheduler.py 会带 cam1/cam2 这样的
    名字，能正确数出几路。要按「部署 N 路」推演覆盖率时显式传它。
    """
    m = merge(bundles)
    st = m["stats"]
    n_streams = int(streams) if streams else max(1, int(st.get("streams") or 1))
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sources": m["sources"],
        "basis": {
            "video_s": st.get("video_s", 0.0),
            "streams": n_streams,
            "persons_total": st.get("persons_total", 0),
            "raw_alarms": st.get("raw_alarms", 0),
            "events": len(m["events"]),
            "ledger": len(m["ledger"]),
            "frames_processed": st.get("frames_processed", 0),
        },
        "metrics": {
            "coverage_ratio": coverage(n_streams, hours, manual_times, manual_minutes),
            "verdict_coverage": verdict_coverage(m["ledger"], st.get("persons_total", 0)),
            "denoise_ratio": denoise(st.get("raw_alarms", 0), len(m["events"])),
            "compliance_rate": compliance(m["ledger"]),
            "violation_duration_s": durations(m["ledger"]),
        },
        "region_ranking": region_ranking(m["events"]),
        "trend": trend(m["ledger"], bucket_s),
    }


def format_table(vm: dict) -> str:
    """给人看的一段摘要，CLI 直接打印。"""
    mt = vm["metrics"]
    b = vm["basis"]
    lines = [
        f"样本：{b['video_s']}s · {b['streams']} 路 · 出现 {b['persons_total']} 人次 · "
        f"逐帧告警 {b['raw_alarms']} 次 · 事件 {b['events']} 条 · 账本 {b['ledger']} 条",
        "",
        f"{'指标':<14}{'值':>10}   说明",
        "-" * 62,
    ]
    c = mt["coverage_ratio"]
    lines.append(f"{'巡检覆盖率':<14}{str(c['value']) + '×':>10}   "
                 f"系统 {c['system_minutes']:g} 分钟 vs 人工 {c['manual_minutes']:g} 分钟（推演）")
    v = mt["verdict_coverage"]
    lines.append(f"{'判定覆盖率':<14}{v['value'] * 100:>9.1f}%   "
                 f"{v['judged']}/{v['persons_total']} 人次有定论")
    d = mt["denoise_ratio"]
    lines.append(f"{'告警降噪率':<14}{(d['value'] or 0) * 100:>9.1f}%   "
                 f"{d['raw']} 次 → {d['events']} 条")
    r = mt["compliance_rate"]
    lines.append(f"{'合规率':<14}{(r['value'] or 0) * 100:>9.1f}%   "
                 f"{r['ok']} 合规 / {r['violation']} 违规（n={r['n']}，"
                 f"待复核 {r['review']}、无法判定 {r['unknown']} 不计）")
    u = mt["violation_duration_s"]
    lines.append(f"{'违规持续时长':<14}{u['p50']:>9.1f}s   "
                 f"P95 {u['p95']}s · 最长 {u['max']}s · n={u['n']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="安巡 Agent 价值指标聚合")
    ap.add_argument("--events", nargs="+", required=True,
                    help="一份或多份事件 JSON（stream.py / scheduler.py 的 --out）")
    ap.add_argument("--out", default="", help="指标输出 JSON 路径")
    ap.add_argument("--hours", type=float, default=8.0,
                    help="巡检覆盖率推演用的单路连续时长（小时），默认 8")
    ap.add_argument("--manual-times", type=float, default=2,
                    help="人工每班巡检次数，默认 2")
    ap.add_argument("--manual-minutes", type=float, default=15,
                    help="人工每次巡检分钟数，默认 15")
    ap.add_argument("--bucket", type=float, default=30.0, help="趋势分桶秒数，默认 30")
    ap.add_argument("--streams", type=int, default=0,
                    help="按几路摄像头推演巡检覆盖率；缺省按账本里的 stream 名计数")
    args = ap.parse_args()

    bundles = [load(p) for p in args.events]
    vm = compute(bundles, hours=args.hours, manual_times=args.manual_times,
                 manual_minutes=args.manual_minutes, bucket_s=args.bucket,
                 streams=args.streams or None)
    print(format_table(vm))
    if args.out:
        Path(args.out).write_text(json.dumps(vm, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
