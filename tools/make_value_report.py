# -*- coding: utf-8 -*-
"""
安巡 Agent · 一条命令出价值报告
================================

    python tools/make_value_report.py

依次做四件事，任何一步失败就停在那一步并打印原因：

    视频 → 判定账本 → 五个指标 → 两版报告（完整文档 + 16:9 单页）

为什么要有这个脚本：README 第七节写了四条命令，第二条还要把两个 `--events`
参数逐个写对。评委或队友漏一个参数，跑出来的数就和文档对不上，然后怀疑是代码坏了。

默认用 `datasets/site_pan.mp4`（v2.1.0 的评测素材，3.4MB 随仓库分发）。
抽帧率默认取 **5fps**——那是三档表里的推荐档（覆盖率 82.3%，成本 124 次/分）。
换成 2fps 覆盖率会降到 75%，换 10fps 只多 1 条定论却翻倍成本，见 docs/抽帧与触发口径.md。

只用已有账本重算、不跑视频：加 `--no-video`。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun import report, stats                            # noqa: E402

SITE = ROOT / "datasets" / "site_pan.mp4"
EV_SITE = ROOT / "data" / "events_pan.json"
VM = ROOT / "data" / "value_metrics.json"
# v2.0 的旧账本（site_demo / real1080）已移到 data/archive/：
# 那两段素材测不出真实判定能力，留着主目录会让人误当当前口径。
OUT_DOC = ROOT / "docs" / "figs" / "value_report.html"
OUT_SLIDE = ROOT / "docs" / "figs" / "value_report_16x9.html"


def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")


def run_stream(source: Path, region: str, out: Path, fps: str = "5") -> bool:
    """跑一遍视频巡查。失败不抛异常，返回 False 让调用方决定是否继续。"""
    cmd = [sys.executable, str(ROOT / "src" / "anxun" / "stream.py"),
           "--source", str(source), "--fps", fps, "--region", region,
           "--out", str(out)]
    print(f"  $ stream.py --source {source.name} --region {region}")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  ✗ 失败（退出码 {r.returncode}）")
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            print(f"    {line}")
        return False
    for line in (r.stdout or "").strip().splitlines():
        if "告警事件" in line or "->" in line:
            print(f"  {line.strip()}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="一条命令出安巡价值报告")
    ap.add_argument("--no-video", action="store_true",
                    help="跳过视频，只用 data/events_*.json 重算指标与报告")
    ap.add_argument("--keep-events", action="store_true",
                    help="不覆盖已有的 data/events_*.json")
    args = ap.parse_args()

    total = 4 if not args.no_video else 1
    used: list[Path] = []

    # ---- 1. 账本 ----
    if not args.no_video:
        step(1, total, "跑视频，生成判定账本")
        if not SITE.exists():
            print(f"  ✗ 缺 {SITE.relative_to(ROOT)}")
            print("    它随仓库分发；若确实不在，用 python tools/make_site_video.py 重建。")
            return 2
        if not (args.keep_events and EV_SITE.exists()):
            if not run_stream(SITE, "B-01", EV_SITE):
                return 3
        used.append(EV_SITE)
    else:
        step(1, 1, "使用已有账本")
        if EV_SITE.exists():
            used.append(EV_SITE)
            print(f"  · {EV_SITE.relative_to(ROOT)}")

    if not used:
        print("\n✗ 没有任何账本可用。去掉 --no-video 跑一次，"
              "或用 python tools/make_motion_video.py 重建素材。")
        return 4

    # ---- 2. 指标 ----
    if not args.no_video:
        step(2, total, "算五个价值指标")
    bundles = [stats.load(p) for p in used]
    vm = stats.compute(bundles)
    VM.write_text(json.dumps(vm, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {VM.relative_to(ROOT)}（{len(used)} 份账本）")
    print("  " + stats.format_table(vm).replace("\n", "\n  "))

    # ---- 3 / 4. 两版报告 ----
    events: list[dict] = []
    for b in bundles:
        events.extend(b.get("events") or [])

    if not args.no_video:
        step(3, total, "完整文档版（给人看，含事件明细）")
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(report.render(vm, events), encoding="utf-8")
    print(f"  -> {OUT_DOC.relative_to(ROOT)}  ({OUT_DOC.stat().st_size / 1024:.1f} KB)")

    if not args.no_video:
        step(4, total, "16:9 单页版（贴 PPT 第 15 页）")
    OUT_SLIDE.write_text(report.render_slide(vm, events), encoding="utf-8")
    print(f"  -> {OUT_SLIDE.relative_to(ROOT)}  ({OUT_SLIDE.stat().st_size / 1024:.1f} KB)")

    print("\n完成。验收：python src/anxun/verify_all.py --no-serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
