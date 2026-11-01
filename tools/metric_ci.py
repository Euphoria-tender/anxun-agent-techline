# -*- coding: utf-8 -*-
"""
指标置信区间 · 回答「100 张够不够」
===================================

四个指标都是点估计。100 张测试集上，误报率每多/少 1 例就动 1 个百分点，
所以「5.1% 达标」这句话有多硬，取决于区间有多宽。

本脚本做两件事：

1. 对误报率/漏报率算 **Wilson 95% 置信区间**（按人次，不是按图）。
   选 Wilson 而不是正态近似：样本量小、比例接近 0 或 1 时，正态近似会给出负下界，
   Wilson 不会，且小样本下覆盖率更接近名义值。

2. 把「点估计是否达标」和「区间上界是否达标」分开报。
   这两件事经常被混为一谈：点估计达标不等于稳定达标。

mAP 不给区间：自助法需要逐图 AP，当前 JSON 只存了逐类 AP 与曲线。
要么改 evaluate_real 落逐图结果，要么别声称 mAP 有区间——现在选后者。

    python tools/metric_ci.py

输出：
    data/metrics_ci.json          机器可读
    docs/指标置信区间.md           人读（自动生成，勿手改）
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "metrics_ci.json"
OUT_MD = ROOT / "docs" / "指标置信区间.md"

Z = 1.96  # 95%

# 每份指标 JSON 的业务含义，顺带把「这份数是谁跑的」写清楚
SPEC = [
    ("data/metrics_real.json", "helmet · torch-GPU", "handoff"),
    ("data/metrics_onnx.json", "helmet · ONNX-CPU", "handoff"),
    ("data/metrics_vest.json", "vest · torch-GPU", "reference"),
]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval，返回百分比。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, (center - half) * 100), min(100.0, (center + half) * 100))


def block(k: int, n: int, target: float, lower_is_better: bool) -> dict:
    lo, hi = wilson(k, n)
    p = k / n * 100 if n else 0.0
    point_ok = p <= target if lower_is_better else p >= target
    # 区间上界（越低越好的指标）或下界（越高越好的指标）是否也达标
    edge = hi if lower_is_better else lo
    edge_ok = edge <= target if lower_is_better else edge >= target
    return {
        "k": k, "n": n, "rate": round(p, 2),
        "ci95": [round(lo, 1), round(hi, 1)],
        "target": target,
        "point_ok": point_ok,
        "robust_ok": edge_ok,
        "one_case_pp": round(100 / n, 2) if n else None,
    }


def main() -> int:
    payload = {
        "kind": "metric-confidence-interval",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Wilson score interval, 95% (z=1.96)，按人次计",
        "note": "mAP 不给区间：自助法需要逐图 AP，当前 JSON 未落逐图结果",
        "metrics": {},
    }
    for rel, label, role in SPEC:
        p = ROOT / rel
        if not p.exists():
            print(f"跳过（缺文件）：{rel}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        r = d.get("results") or {}
        conf = r.get("confusion") or {}
        fp = block(int(conf.get("fp", 0)), int(r.get("n_compliant", 0)), 10.0, True)
        fn = block(int(conf.get("fn", 0)), int(r.get("n_violating", 0)), 5.0, True)
        payload["metrics"][rel] = {
            "label": label,
            "role": role,
            "rules_profile": (d.get("config") or {}).get("rules_profile", "unknown"),
            "fp_rate": fp,
            "fn_rate": fn,
            "mAP50": {"value": r.get("mAP50"), "target": 0.85, "ci": None,
                      "why": "需逐图 AP 才能自助法；100 张上单图权重 1%，区间会很宽"},
        }

    payload["summary"] = _summarize(payload["metrics"])
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"\n-> {OUT_JSON.relative_to(ROOT)}")
    print(f"-> {OUT_MD.relative_to(ROOT)}")
    return 0


def _summarize(metrics: dict) -> list[str]:
    out = []
    real = metrics.get("data/metrics_real.json")
    if not real:
        return out
    fp, fn = real["fp_rate"], real["fn_rate"]
    out.append(
        f"误报率 {fp['rate']}%（{fp['k']}/{fp['n']}），95% 区间 "
        f"{fp['ci95'][0]}–{fp['ci95'][1]}%，目标 ≤{fp['target']}%。"
        + ("点估计达标，但区间上界越线——只能说「点估计达标」，不能说「稳定达标」。"
           if fp["point_ok"] and not fp["robust_ok"] else
           ("点估计与区间上界都达标。" if fp["robust_ok"] else "未达标。")))
    out.append(
        f"漏报率 {fn['rate']}%（{fn['k']}/{fn['n']}），95% 区间 "
        f"{fn['ci95'][0]}–{fn['ci95'][1]}%，目标 ≤{fn['target']}%。"
        + (f"区间下界 {fn['ci95'][0]}% 仍高于目标 {fn['target']}%——"
           "差距不是 100 张样本量造成的，换更大的测试集也救不回来。"
           if fn["ci95"][0] > fn["target"] else "区间下界已接入目标区间，样本量可能是主因。"))
    out.append(
        f"颗粒度：误报率每 1 例 ≈ {fp['one_case_pp']} 个百分点，漏报率每 1 例 ≈ "
        f"{fn['one_case_pp']} 个百分点。100 张能分辨「差一个量级」，分辨不了「差 1–2 个点」。")
    return out


def _table(metrics: dict) -> list[str]:
    rows = ["| 指标 | 口径 | 点估计 | 95% 区间 | 目标 | 点估计达标 | 区间达标 | 每 1 例影响 |",
            "| :--- | :--- | ---: | :---: | ---: | :---: | :---: | ---: |"]
    for rel, m in metrics.items():
        for name, key in (("误报率", "fp_rate"), ("漏报率", "fn_rate")):
            b = m[key]
            rows.append(
                f"| {name} | {m['label']} | {b['rate']}%（{b['k']}/{b['n']}） | "
                f"{b['ci95'][0]}–{b['ci95'][1]}% | ≤{b['target']}% | "
                f"{'✅' if b['point_ok'] else '❌'} | {'✅' if b['robust_ok'] else '❌'} | "
                f"{b['one_case_pp']} pp |")
    return rows


def _markdown(payload: dict) -> str:
    m = payload["metrics"]
    lines = [
        "# 指标置信区间 · 100 张够不够",
        "",
        "> 本文件由 `python tools/metric_ci.py` 自动生成，**不要手改**。",
        f"> 生成时间 {payload['generated_at']} ｜ 方法：{payload['method']}",
        "",
        "## 一、一句话结论",
        "",
    ]
    lines += [f"- {s}" for s in payload["summary"]] or ["- 指标 JSON 缺失，无法计算。"]
    lines += [
        "",
        "## 二、逐项区间",
        "",
        *_table(m),
        "",
        "读法：**点估计达标** = 跑出来的那个数在目标内；**区间达标** = 95% 置信区间的最不利端也在目标内。",
        "对外只写前者、答辩被追问置信度时能翻出后者，才算把话说圆。",
        "",
        "## 三、mAP 为什么没有区间",
        "",
        "自助法（bootstrap）需要**逐图 AP**，即每张图单独算一次匹配结果。",
        "当前 `data/metrics_*.json` 只存了逐类 AP 和 PR 曲线上的点，没有逐图结果，",
        "用现有数据算出来的「区间」是假区间。要么改 `evaluate_real.py` 落逐图结果，要么不给区间——",
        "这次选后者，并在 PPT 上只报点估计 + 说明测试集规模。",
        "",
        "## 四、答辩口径（可直接念）",
        "",
        "> " + _script(m),
        "",
    ]
    return "\n".join(lines)


def _script(metrics: dict) -> str:
    """把上面算出来的数拼成一段能直接念的答辩话术，避免文档里写死一组旧数字。"""
    real = metrics.get("data/metrics_real.json")
    if not real:
        return "指标 JSON 缺失，先跑 evaluate_real.py。"
    fp, fn = real["fp_rate"], real["fn_rate"]
    return (
        f"四个指标都出自 100 张独立测试集（CC BY 4.0，全程不参与训练与选轮次）。"
        f"误报率 {fp['rate']}%，点估计{'达标' if fp['point_ok'] else '未达标'}，"
        f"95% 置信区间 {fp['ci95'][0]}–{fp['ci95'][1]}%，"
        + ("所以我们只声称「点估计达标」，不声称「稳定达标」。"
           if fp["point_ok"] and not fp["robust_ok"] else "区间同样在目标内。")
        + f"漏报率 {fn['rate']}%，区间下界 {fn['ci95'][0]}%，"
        + (f"仍高于 {fn['target']}% 的目标——这个差距不是样本量小造成的，"
           "我们在逐例漏报上做过归因，大头是权重本身看错。"
           if fn["ci95"][0] > fn["target"] else "样本量可能是主要原因。")
        + f"100 张的颗粒度是每例约 {fp['one_case_pp']} 个百分点，"
        "能分辨「差一个量级」，分辨不了「差一两个点」。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
