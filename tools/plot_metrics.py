# -*- coding: utf-8 -*-
"""
指标出图 · 给 PPT 第 11 / 13 页直接用
=====================================

JSON 里有数，图上没东西，PPT 就只能贴表格。本脚本把三张图从**同一份 JSON 与
同一份阈值配置**里画出来，改数据重跑即可，不手工描点：

    docs/figs/confusion_matrix.png   混淆矩阵 78/15/10/88（P11，随 metrics_real.json 自动更新）
    docs/figs/pr_curve.png           逐类 PR 曲线（P11）
    docs/figs/risk_matrix.png        L×S 风险矩阵与四色等级（P13 / 视频 3:10）

画布统一 1280×720（与 `docs/技术页_10-13.html` 的 16:9 一致），
贴进 PPT 不用二次缩放。

    python tools/plot_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "docs" / "figs"
DPI, W, H = 100, 12.8, 7.2      # 1280 × 720

INK = "#1A1A1A"
DIM = "#6B6B6B"
ACCENT = "#C62828"       # 红 = 违规/高危（国内安全色习惯）
OK = "#0B6E4F"           # 绿 = 合规
WARN = "#E08A00"
PAPER = "#FFFFFF"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # 中文字体：Windows 优先雅黑，其次黑体；都没有就退回 DejaVu（会缺字，但不报错）
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                            "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": "#D8D8D8",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "axes.grid": True,
        "grid.color": "#ECECEC",
        "grid.linewidth": 0.8,
        "savefig.facecolor": PAPER,
    })
    return plt


def _load(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"{rel} 不存在，先跑 evaluate_real.py")
    return json.loads(p.read_text(encoding="utf-8"))


def _footer(fig, text: str) -> None:
    fig.text(0.012, 0.022, text, fontsize=9, color=DIM, ha="left")


def plot_confusion(plt) -> Path:
    d = _load("data/metrics_real.json")
    r = d["results"]
    c = r.get("confusion") or {}
    tp, fn, fp, tn = (c.get(k, 0) for k in ("tp", "fn", "fp", "tn"))
    m = [[tp, fn], [fp, tn]]

    fig, ax = plt.subplots(figsize=(W, H))
    # 对角线（判对）用绿，非对角线（判错）用红，颜色本身就是结论
    ax.imshow([[1 if i == j else 0 for j in range(2)] for i in range(2)],
              cmap="RdYlGn", vmin=-0.2, vmax=1.2, alpha=0.55)

    labels = [["正确告警\n(TP)", "漏报\n(FN)"], ["误报\n(FP)", "正确放行\n(TN)"]]
    total = max(1, tp + fn + fp + tn)
    for i in range(2):
        for j in range(2):
            v = m[i][j]
            ax.text(j, i - 0.10, f"{v}", ha="center", va="center",
                    fontsize=54, fontweight="bold",
                    color=ACCENT if i != j else OK)
            ax.text(j, i + 0.22, f"{v / total * 100:.1f}%  ·  {labels[i][j]}",
                    ha="center", va="center", fontsize=13, color=INK)
    ax.set_xticks([0, 1], ["判定：违规", "判定：合规"])
    ax.set_yticks([0, 1], ["实际：违规", "实际：合规"])
    ax.tick_params(length=0, labelsize=13)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"混淆矩阵 · {r.get('n_violating')} 违规 / {r.get('n_compliant')} 合规 人次",
                 fontsize=20, fontweight="bold", pad=42, loc="left")
    # 副标题放在标题下方、坐标区上方的留白里，别用数据坐标——那个坐标是反的，会压到标题
    ax.text(0.0, 1.008, f"误报率 {r.get('fp_rate')}%（目标 ≤10%）· "
                        f"漏报率 {r.get('fn_rate')}%（目标 ≤5%）· "
                        f"mAP@0.5 {r.get('mAP50')}",
            transform=ax.transAxes, fontsize=13, color=ACCENT, ha="left", va="bottom")
    _footer(fig, "数据 data/metrics_real.json · 100 张独立测试集 · CC BY 4.0")
    p = OUT / "confusion_matrix.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_pr(plt) -> Path:
    d = _load("data/metrics_real.json")
    r = d["results"]
    fig, ax = plt.subplots(figsize=(W, H))
    colors = {"person": "#2E6DB4", "helmet": ACCENT}
    names = {"person": "person（人体）", "helmet": "helmet（安全帽）"}
    for cls, pts in (r.get("curves") or {}).items():
        if not pts:
            continue
        xy = sorted(((p["recall"], p["precision"]) for p in pts))
        xs = [a for a, _ in xy]
        ys = [b for _, b in xy]
        ap = (r.get("per_class", {}).get(cls) or {}).get("ap")
        ax.plot(xs, ys, lw=2.6, color=colors.get(cls, DIM),
                label=f"{names.get(cls, cls)}  AP@0.5 = {ap:.3f}" if ap is not None
                else names.get(cls, cls))
        ax.fill_between(xs, ys, color=colors.get(cls, DIM), alpha=0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall（召回率）", fontsize=13)
    ax.set_ylabel("Precision（精确率）", fontsize=13)
    ax.set_title(f"PR 曲线 · mAP@0.5 = {r.get('mAP50')}（目标 ≥0.85）",
                 fontsize=20, fontweight="bold", pad=16, loc="left")
    ax.legend(frameon=False, fontsize=13, loc="lower left")
    ax.text(0.99, 0.04, f"测试集 {r.get('n_images')} 张 / {r.get('n_boxes')} 框",
            fontsize=11, color=DIM, ha="right", transform=ax.transAxes)
    _footer(fig, "数据 data/metrics_real.json · 曲线由 evaluate_real.py 逐阈值扫描得到")
    p = OUT / "pr_curve.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_risk(plt) -> Path:
    """风险矩阵图。表格从 rules_engine 直接读，不在这里抄一份。"""
    from anxun import rules_engine as R

    fig, ax = plt.subplots(figsize=(W, H))
    # 四色等级按 R = L × S 切档；配色与前端一致
    bands = [(12, "#C62828", "红色 · 立即停工"), (8, "#F08C00", "橙色 · 当日整改"),
             (4, "#F2C230", "黄色 · 限期整改"), (0, "#3E7BD6", "蓝色 · 加强关注")]

    def color_of(rv: int) -> str:
        for floor, col, _ in bands:
            if rv >= floor:
                return col
        return "#BBBBBB"

    for L in range(1, 5):
        for S in range(1, 5):
            rv = L * S
            ax.add_patch(plt.Rectangle((S - 0.5, L - 0.5), 1, 1,
                                       facecolor=color_of(rv), alpha=0.85,
                                       edgecolor=PAPER, lw=2))
            ax.text(S, L, f"R={rv}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color="#FFFFFF")

    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xticks(range(1, 5), ["1 轻微", "2 一般", "3 较大", "4 严重"])
    ax.set_yticks(range(1, 5), ["1 罕见", "2 偶发", "3 多发", "4 频发"])
    ax.set_xlabel("S · 后果严重度（违规类型 × 场景）", fontsize=13)
    ax.set_ylabel("L · 可能性（近 30 天违规次数）", fontsize=13)
    ax.tick_params(length=0, labelsize=12)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("风险矩阵 · R = L × S 定级", fontsize=20, fontweight="bold",
                 pad=16, loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.85) for _, col, _ in bands]
    ax.legend(handles, [txt for _, _, txt in bands], frameon=False, fontsize=11,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)

    l_txt = " · ".join(f"≤{cap} 次→L={v}" for cap, v in R.L_TABLE) + f" · >{R.L_TABLE[-1][0]} 次→L=4"
    _footer(fig, f"L 查表：{l_txt}｜S 查表见 rules_engine.S_MAP（{len(R.S_MAP)} 条）")
    p = OUT / "risk_matrix.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt = _style()
    for fn in (plot_confusion, plot_pr, plot_risk):
        p = fn(plt)
        print(f"-> {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
