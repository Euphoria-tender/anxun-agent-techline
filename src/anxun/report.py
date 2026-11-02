# -*- coding: utf-8 -*-
"""
安巡 Agent · 巡检价值报告
==========================

把 `stats.py` 算出的指标渲染成一页可截图、可打印的 HTML。

为什么不用 matplotlib 出图
---------------------------
报告要能离线打开、直接截图进 PPT、发给不用装环境的人。matplotlib 存成
PNG 会把数字烧死在像素里，评委问「这个数怎么来的」就得回去翻 JSON。
这里手写内联 SVG：数字仍在 DOM 里，能选中、能改、能用验收脚本 grep。

为什么固定 1280 宽
------------------
PPT 是 16:9。报告按 1280 定宽排版，截图后直接贴满一页，不用缩放对齐。

用法
----
    python src/anxun/report.py --metrics data/value_metrics.json \
                               --events  data/events_site.json \
                               --out     docs/figs/value_report.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# 版式常量。改宽度记得同步 CSS 里的 .page width。
W = 1280
PAD = 44


def _pct(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _num(v, unit: str = "", digits: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer():
        return f"{int(v)}{unit}"
    return f"{v:.{digits}f}{unit}"


def _esc(s) -> str:
    return html.escape(str(s))


# ============================================================
# 图表：内联 SVG

def line_chart(trend: list[dict], w: int = 560, h: int = 190) -> str:
    """合规率时段趋势。人少的桶不画（噪音大），用柱高表示人数。"""
    pts = [(i, t) for i, t in enumerate(trend) if t.get("compliance") is not None]
    if len(pts) < 2:
        return ('<div class="empty">样本时间跨度不足，无法绘制趋势。'
                '换更长的视频或多段素材合并。</div>')

    pad_l, pad_b, pad_t = 38, 26, 14
    iw, ih = w - pad_l - 12, h - pad_b - pad_t

    # x 按**真实时间**映射，不按桶索引等分 —— 最后一个桶常常只有几秒
    # （视频在桶中间结束），等分会把它画得和 30 秒的桶一样宽，视觉上撒谎。
    # 容错：trend 可能来自外部拼装的 JSON，缺字段就补 0，别让整张报告崩掉。
    # v2.1.0 之前这里是 t["t_end_s"] 硬取，缺键直接 KeyError，整页渲染不出来。
    def _g(row: dict, key: str, default: float = 0.0) -> float:
        v = row.get(key, default)
        return default if v is None else float(v)

    trend = [t for t in trend if isinstance(t, dict)]
    t_max = max((_g(t, "t_end_s") for t in trend), default=0) or 1

    def x(sec): return pad_l + (float(sec) / t_max) * iw
    def y(v): return pad_t + (1 - v) * ih

    # 人数柱（背景，浅灰）：按桶的真实时间跨度画，宽度就是那个桶的时长
    max_p = max((int(t.get("persons") or 0) for t in trend), default=1) or 1
    bars = []
    for t in trend:
        if not t.get("persons"):
            continue
        bh = (float(t.get("persons") or 0) / max_p) * ih * 0.30
        x0, x1 = x(_g(t, "t_start_s")), x(_g(t, "t_end_s"))
        bw = max(1.5, x1 - x0 - 1)
        bars.append(f'<rect x="{x0:.1f}" y="{pad_t + ih - bh:.1f}" '
                    f'width="{bw:.1f}" height="{bh:.1f}" fill="#cfd8e3"/>')

    # 折线
    pts_xy = [(x(_g(t, "t_start_s") + (_g(t, "t_end_s") - _g(t, "t_start_s")) / 2), t.get("compliance"))
              for _, t in pts]
    d = " ".join(f"{'M' if k == 0 else 'L'}{px:.1f},{y(v):.1f}"
                 for k, (px, v) in enumerate(pts_xy))
    dots = "".join(f'<circle cx="{px:.1f}" cy="{y(v):.1f}" r="3.2" '
                   f'fill="#0f766e"/>' for px, v in pts_xy)
    # 目标线 60%
    ty = y(0.6)
    grid = (f'<line x1="{pad_l}" y1="{ty:.1f}" x2="{w - 12}" y2="{ty:.1f}" '
            f'stroke="#f59e0b" stroke-width="1" stroke-dasharray="4 3"/>'
            f'<text x="{w - 14}" y="{ty - 4:.1f}" font-size="10" fill="#b45309" '
            f'text-anchor="end">60%</text>')

    ticks = "".join(
        f'<text x="{pad_l - 6}" y="{y(v) + 3.5:.1f}" font-size="10" fill="#64748b" '
        f'text-anchor="end">{int(v * 100)}</text>' for v in (0, 0.5, 1.0))

    first = _g(trend[0], "t_start_s")
    last = _g(trend[-1], "t_end_s")
    axis = (f'<text x="{pad_l}" y="{h - 6}" font-size="10" fill="#64748b">{first:g}s</text>'
            f'<text x="{w - 12}" y="{h - 6}" font-size="10" fill="#64748b" '
            f'text-anchor="end">{last:g}s</text>')

    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="合规率时段趋势">'
            f'<text x="{pad_l}" y="{pad_t - 2}" font-size="10" fill="#94a3b8">'
            f'柱=该时段人次 · 线=合规率</text>'
            f'{grid}{ticks}{"".join(bars)}'
            f'<path d="{d}" fill="none" stroke="#0f766e" stroke-width="2"/>'
            f'{dots}{axis}</svg>')


def bar_chart(rows: list[dict], w: int = 460, h: int = 190,
              label_key: str = "grade", value_key: str = "count",
              color_key: str | None = None) -> str:
    """横向条形图。rows 需已排好序。

    条高与间距都是固定值、整组垂直居中 —— 不能按「可用高度 / 条数」去摊，
    否则只有 2 条时会摊成两条相隔 60px 的横线，看着像坏了。
    """
    if not rows:
        return '<div class="empty">无数据</div>'
    pad_l, pad_r, pad_t = 66, 46, 8
    iw = w - pad_l - pad_r
    n = len(rows)
    bh = 22.0
    gap = 13.0
    total = bh * n + gap * (n - 1)
    top0 = pad_t + max(0.0, (h - pad_t - 10 - total) / 2)
    mx = max((r[value_key] for r in rows), default=1) or 1

    out = []
    for k, r in enumerate(rows):
        top = top0 + k * (bh + gap)
        bl = max(2.0, (r[value_key] / mx) * iw)
        color = r.get(color_key) if color_key else "#0f766e"
        out.append(
            f'<text x="{pad_l - 8}" y="{top + bh / 2 + 3.5:.1f}" font-size="11" '
            f'fill="#334155" text-anchor="end">{_esc(r[label_key])}</text>'
            f'<rect x="{pad_l}" y="{top:.1f}" width="{bl:.1f}" height="{bh:.1f}" '
            f'rx="3" fill="{color}"/>'
            f'<text x="{pad_l + bl + 7:.1f}" y="{top + bh / 2 + 3.5:.1f}" '
            f'font-size="11" fill="#475569">{r[value_key]}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">{"".join(out)}</svg>')


GRADE_COLOR = {"红色": "#dc2626", "橙色": "#ea580c", "黄色": "#ca8a04",
               "蓝色": "#2563eb", "未定级": "#94a3b8"}


# ============================================================
# 页面

def kpi_card(label: str, value: str, sub: str, accent: str = "#0f766e",
             note: str = "") -> str:
    note_html = f'<div class="kpi-note">{_esc(note)}</div>' if note else ""
    return (f'<div class="kpi" style="border-top-color:{accent}">'
            f'<div class="kpi-label">{_esc(label)}</div>'
            f'<div class="kpi-value" style="color:{accent}">{_esc(value)}</div>'
            f'<div class="kpi-sub">{_esc(sub)}</div>{note_html}</div>')


def render(vm: dict, events: list[dict] | None = None) -> str:
    m = vm["metrics"]
    b = vm["basis"]

    cov, ver = m["coverage_ratio"], m["verdict_coverage"]
    den, comp = m["denoise_ratio"], m["compliance_rate"]
    dur = m["violation_duration_s"]

    cards = "".join([
        kpi_card("巡检覆盖率", f"{_num(cov['value'])}×",
                 f"系统 {cov['system_minutes']:g} 分钟 vs 人工 {cov['manual_minutes']:g} 分钟",
                 "#0f766e", note="排班推演，非实测"),
        kpi_card("判定覆盖率", _pct(ver["value"]),
                 f"{ver['judged']}/{ver.get('eligible') or ver['persons_total']} 人次有定论"
                 f"（旧口径 {_pct(ver.get('value_raw'))}）", "#2563eb"),
        kpi_card("告警降噪率", _pct(den["value"]),
                 f"{den['raw']} 次逐帧告警 → {den['events']} 条事件", "#7c3aed"),
        kpi_card("合规率", _pct(comp["value"]),
                 f"{comp['ok']} 合规 / {comp['violation']} 违规（n={comp['n']}）",
                 "#dc2626" if comp["violation"] > comp["ok"] else "#16a34a",
                 note="素材为违规演示样本，非现场水平"),
        kpi_card("违规持续时长", f"{_num(dur['p50'])}s",
                 f"P95 {dur['p95']}s · 最长 {dur['max']}s · n={dur['n']}",
                 "#ea580c", note="素材无整改过程，非整改耗时"),
    ])

    # 等级分布：从事件里数
    grades: dict[str, int] = {}
    for e in (events or []):
        g = e.get("grade") or "未定级"
        grades[g] = grades.get(g, 0) + 1
    order = ["红色", "橙色", "黄色", "蓝色", "未定级"]
    grade_rows = [{"grade": g, "count": grades[g], "color": GRADE_COLOR.get(g, "#94a3b8")}
                  for g in order if grades.get(g)]
    grade_rows.sort(key=lambda r: -r["count"])

    # 区域排行：标签用区域编号（A-03），全名太长会把 SVG 左padding 撑破
    regions = vm.get("region_ranking") or []
    region_rows = [{"region": r.get("region_id") or r["region"][:6],
                    "count": r["events"]} for r in regions[:6]]

    # 明细表
    rows = sorted(events or [], key=lambda e: e.get("t_video_s", 0))[:24]
    detail = "".join(
        f'<tr><td class="mono">{e.get("t_video_s", 0):g}s</td>'
        f'<td class="rg" title="{_esc(e.get("region", ""))}">'
        f'<b>{_esc(e.get("region_id", ""))}</b> {_esc(e.get("region", ""))}</td>'
        f'<td>{_esc(e.get("violation", ""))}</td>'
        f'<td><span class="badge" style="background:{GRADE_COLOR.get(e.get("grade"), "#94a3b8")}">'
        f'{_esc(e.get("grade", ""))}</span></td>'
        f'<td class="mono">R{e.get("R", 0)}</td>'
        f'<td>{_esc(e.get("assignee", ""))}</td>'
        f'<td>{_esc(e.get("action", ""))}</td></tr>' for e in rows)

    more = ("" if len(events or []) <= 24
            else f'<div class="more">仅显示前 24 条，共 {len(events)} 条</div>')

    caveats = [
        f"巡检覆盖率是**推演值**：人工基线 {cov['assumption']}，不是实测。",
        f"判定覆盖率 {_pct(ver['value'])}——没凑满三帧的人不计入合规率分母，"
        f"这个数低说明轨迹碎（分辨率/运动），不是模型不准。",
        f"违规持续时长是**降级口径**：素材里没有「违规→整改→合规」的过程，"
        f"只能算单次告警的持续时间；真实现场有整改时，同一口径自动变成整改耗时。",
        f"样本仅 {b['video_s']:g} 秒、{b['persons_total']} 人次，"
        f"属演示级样本，不代表班次级结论。",
    ]

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>安巡 Agent · 巡检价值报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; background: #eef1f5;
         font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         color: #0f172a; }}
  .page {{ width: {W}px; margin: 0 auto; background: #fff; padding: {PAD}px;
          box-shadow: 0 2px 14px rgba(15,23,42,.10); }}
  h1 {{ margin: 0; font-size: 26px; letter-spacing: .5px; }}
  .meta {{ margin-top: 7px; font-size: 12px; color: #64748b; }}
  .meta b {{ color: #0f766e; font-weight: 600; }}
  .rule {{ height: 3px; background: linear-gradient(90deg,#0f766e,#2563eb 55%,#e2e8f0);
          margin: 16px 0 20px; border-radius: 2px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; }}
  .kpi {{ background: #fbfcfd; border: 1px solid #e6ebf2; border-top: 3px solid;
         border-radius: 6px; padding: 12px 13px 13px; }}
  .kpi-label {{ font-size: 12px; color: #64748b; }}
  .kpi-value {{ font-size: 27px; font-weight: 700; margin: 5px 0 4px;
               font-variant-numeric: tabular-nums; }}
  .kpi-sub {{ font-size: 10.5px; color: #94a3b8; line-height: 1.45; }}
  .kpi-note {{ font-size: 9.5px; color: #b45309; line-height: 1.35; margin-top: 3px; }}
  .charts {{ display: grid; grid-template-columns: 1.22fr 1fr; gap: 16px; margin-top: 20px; }}
  .card {{ border: 1px solid #e6ebf2; border-radius: 6px; padding: 13px 14px 10px; }}
  .card h2 {{ margin: 0 0 8px; font-size: 13px; color: #334155; font-weight: 600; }}
  .empty {{ font-size: 12px; color: #94a3b8; padding: 44px 8px; text-align: center; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 11.5px; }}
  th {{ text-align: left; color: #64748b; font-weight: 500; font-size: 11px;
       border-bottom: 1.5px solid #e2e8f0; padding: 6px 8px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #f1f5f9; color: #334155; }}
  td.rg {{ width: 158px; white-space: normal; line-height: 1.35; font-size: 11px; }}
  td.rg b {{ color: #0f766e; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-variant-numeric: tabular-nums; color: #64748b; }}
  .badge {{ color: #fff; font-size: 10px; padding: 1.5px 7px; border-radius: 9px; }}
  .more {{ font-size: 11px; color: #94a3b8; padding: 7px 8px 0; }}
  .notes {{ margin-top: 20px; background: #fbfcfd; border: 1px solid #e6ebf2;
           border-left: 3px solid #f59e0b; border-radius: 5px; padding: 12px 16px; }}
  .notes h3 {{ margin: 0 0 7px; font-size: 12px; color: #92400e; font-weight: 600; }}
  .notes ul {{ margin: 0; padding-left: 17px; }}
  .notes li {{ font-size: 11px; color: #57534e; line-height: 1.72; }}
  footer {{ margin-top: 16px; font-size: 10.5px; color: #94a3b8;
           display: flex; justify-content: space-between; }}
</style>
</head>
<body>
<div class="page">
  <h1>安巡 Agent · 巡检价值报告</h1>
  <div class="meta">
    样本 <b>{b['video_s']:g}s</b> · {b['streams']} 路 · 出现 <b>{b['persons_total']}</b> 人次 ·
    逐帧告警 <b>{b['raw_alarms']}</b> 次 · 事件 <b>{b['events']}</b> 条 ·
    生成于 {_esc(vm.get('generated_at', ''))}
  </div>
  <div class="rule"></div>

  <div class="kpis">{cards}</div>

  <div class="charts">
    <div class="card"><h2>合规率时段趋势</h2>{line_chart(vm.get('trend') or [])}</div>
    <div class="card"><h2>告警等级分布 / 区域排行</h2>
      {bar_chart(grade_rows, h=88, color_key="color") if grade_rows else '<div class="empty">无告警事件</div>'}
      {bar_chart(region_rows, h=88, label_key="region", value_key="count") if len(region_rows) > 1 else ''}
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>事件明细</h2>
    <table>
      <thead><tr><th>时间</th><th>区域</th><th>违规</th><th>等级</th>
      <th>风险</th><th>责任人</th><th>处置要求</th></tr></thead>
      <tbody>{detail or '<tr><td colspan="7" style="color:#94a3b8">本次样本无告警</td></tr>'}</tbody>
    </table>
    {more}
  </div>

  <div class="notes">
    <h3>口径与局限（引用这些数字时请一并说明）</h3>
    <ul>{''.join(f'<li>{c}</li>' for c in caveats)}</ul>
  </div>

  <footer>
    <span>安巡 Agent · 技术线 · 判定账本 {b['ledger']} 条</span>
    <span>口径定义见 docs/价值指标口径.md</span>
  </footer>
</div>
</body>
</html>"""


def render_slide(vm: dict, events: list[dict] | None = None) -> str:
    """1280×720 单页版，给 PPT 第 15 页直接贴。

    和 `render()` 的区别：去掉事件明细表（投影上看不清，明细在完整版里），
    保留五张 KPI 卡 + 两张图 + 一行口径。高度写死在 720 —— 不是靠
    `overflow:hidden` 硬裁，是靠分配好每块高度让它自然放得下。
    """
    m = vm["metrics"]
    b = vm["basis"]
    cov, ver = m["coverage_ratio"], m["verdict_coverage"]
    den, comp = m["denoise_ratio"], m["compliance_rate"]
    dur = m["violation_duration_s"]

    cards = "".join([
        kpi_card("巡检覆盖率", f"{_num(cov['value'])}×",
                 f"{cov['system_minutes']:g} 分钟 vs 人工 {cov['manual_minutes']:g} 分钟",
                 "#0f766e", note="排班推演"),
        kpi_card("判定覆盖率", _pct(ver["value"]),
                 f"{ver['judged']}/{ver.get('eligible') or ver['persons_total']} 人次有定论"
                 f"（旧口径 {_pct(ver.get('value_raw'))}）", "#2563eb"),
        kpi_card("告警降噪率", _pct(den["value"]),
                 f"{den['raw']} 次 → {den['events']} 条", "#7c3aed"),
        kpi_card("合规率", _pct(comp["value"]),
                 f"{comp['ok']} 合规 / {comp['violation']} 违规",
                 "#dc2626" if comp["violation"] > comp["ok"] else "#16a34a",
                 note="违规演示素材"),
        kpi_card("违规持续时长", f"{_num(dur['p50'])}s",
                 f"P95 {dur['p95']}s · n={dur['n']}",
                 "#ea580c", note="非整改耗时"),
    ])

    grades: dict[str, int] = {}
    for e in (events or []):
        g = e.get("grade") or "未定级"
        grades[g] = grades.get(g, 0) + 1
    order = ["红色", "橙色", "黄色", "蓝色", "未定级"]
    grade_rows = [{"grade": g, "count": grades[g], "color": GRADE_COLOR.get(g, "#94a3b8")}
                  for g in order if grades.get(g)]
    grade_rows.sort(key=lambda r: -r["count"])

    regions = vm.get("region_ranking") or []
    region_rows = [{"region": r.get("region_id") or r["region"][:6],
                    "count": r["events"]} for r in regions[:5]]

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>安巡 Agent · 巡检价值简报 16:9</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #eef1f5; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         color: #0f172a; display: flex; justify-content: center; padding: 16px; }}
  .slide {{ width: 1280px; height: 720px; background: #fff; padding: 28px 36px;
           box-shadow: 0 2px 14px rgba(15,23,42,.10); display: flex; flex-direction: column; }}
  .top {{ display: flex; justify-content: space-between; align-items: baseline; }}
  h1 {{ font-size: 23px; letter-spacing: .5px; }}
  .meta {{ font-size: 11.5px; color: #64748b; }}
  .meta b {{ color: #0f766e; font-weight: 600; }}
  .rule {{ height: 3px; background: linear-gradient(90deg,#0f766e,#2563eb 55%,#e2e8f0);
          margin: 10px 0 14px; border-radius: 2px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(5,1fr); gap: 11px; height: 150px; }}
  .kpi {{ background: #fbfcfd; border: 1px solid #e6ebf2; border-top: 3px solid;
         border-radius: 6px; padding: 12px; display: flex; flex-direction: column;
         justify-content: center; }}
  .kpi-label {{ font-size: 11.5px; color: #64748b; }}
  .kpi-value {{ font-size: 30px; font-weight: 700; margin: 3px 0 2px;
               font-variant-numeric: tabular-nums; line-height: 1.1; }}
  .kpi-sub {{ font-size: 10px; color: #94a3b8; line-height: 1.4; }}
  .kpi-note {{ font-size: 9.5px; color: #b45309; line-height: 1.3; margin-top: 2px; }}
  .charts {{ display: grid; grid-template-columns: 1.18fr 1fr; gap: 14px;
            margin-top: 14px; height: 452px; }}
  .card {{ border: 1px solid #e6ebf2; border-radius: 6px; padding: 10px 12px 6px; }}
  .card h2 {{ font-size: 12.5px; color: #334155; font-weight: 600; margin-bottom: 5px; }}
  .empty {{ font-size: 11px; color: #94a3b8; padding: 40px 8px; text-align: center; }}
  .foot {{ margin-top: 12px; background: #fbfcfd; border: 1px solid #e6ebf2;
          border-left: 3px solid #f59e0b; border-radius: 5px; padding: 9px 14px;
          font-size: 10.5px; color: #57534e; line-height: 1.62; }}
  .foot b {{ color: #92400e; font-weight: 600; }}
</style>
</head>
<body>
<div class="slide">
  <div class="top">
    <h1>安巡 Agent · 巡检价值简报</h1>
    <div class="meta">样本 <b>{b['video_s']:g}s</b> · {b['streams']} 路 ·
      出现 <b>{b['persons_total']}</b> 人次 · 逐帧告警 <b>{b['raw_alarms']}</b> 次 ·
      事件 <b>{b['events']}</b> 条</div>
  </div>
  <div class="rule"></div>

  <div class="kpis">{cards}</div>

  <div class="charts">
    <div class="card"><h2>合规率时段趋势</h2>{line_chart(vm.get('trend') or [], w=640, h=402)}</div>
    <div class="card"><h2>告警等级分布 / 区域排行</h2>
      {bar_chart(grade_rows, w=520, h=192, color_key="color") if grade_rows else '<div class="empty">无告警事件</div>'}
      {bar_chart(region_rows, w=520, h=192, label_key="region", value_key="count") if len(region_rows) > 1 else ''}
    </div>
  </div>

  <div class="foot">
    <b>读图须知</b>：巡检覆盖率是<b>排班推演</b>（人工每班 2 次 × 15 分钟），非实测；
    合规率 {_pct(comp['value'])} 来自<b>违规演示素材</b>，不代表现场水平；
    违规持续时长是<b>单次告警持续</b>，素材无整改过程，故非整改耗时。
  </div>
</div>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="安巡 Agent 巡检价值报告")
    ap.add_argument("--metrics", required=True, help="stats.py 输出的价值指标 JSON")
    ap.add_argument("--events", default="", nargs="*",
                    help="事件 JSON（用于明细表与等级分布），可多份")
    ap.add_argument("--out", required=True, help="输出 HTML 路径")
    ap.add_argument("--slide", action="store_true",
                    help="输出 1280×720 单页版（贴 PPT 用），而不是完整文档版")
    args = ap.parse_args()

    vm = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    events: list[dict] = []
    for p in (args.events or []):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        events.extend(d.get("events") or [])

    out = render_slide(vm, events) if args.slide else render(vm, events)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out, encoding="utf-8")
    print(f"-> {args.out}  ({len(out) / 1024:.1f} KB, {len(events)} 条事件, "
          f"{'16:9 单页' if args.slide else '完整文档'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
