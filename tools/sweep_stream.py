# -*- coding: utf-8 -*-
"""抽帧率 × 帧差阈值的二维扫描，产出三档配置表。

为什么要单独扫这两个参数
------------------------
它们决定的是两件不同的事，v2.0 把它们混在了一个数里：

  · 抽帧率  → 决定「能看到多少人」。2fps 下正常步速的人位移超过框宽，IoU 匹配失败，
              同一个人被切成多条轨迹；提上来才跟得住。
  · 帧差阈值 → 决定「多少帧白跑推理」。阈值太高会把走动帧判成静止，静默漏看。

两者都可调，但**代价不同**：抽帧率线性抬高推理成本，帧差阈值只影响无效帧。
所以表里两列都给，让人按算力选。

用法
----
    python tools/sweep_stream.py                          # 全矩阵，出 JSON + 控制台表
    python tools/sweep_stream.py --fps 2,5 --motion 0,3   # 只跑指定组合
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun import stats as stats_mod                        # noqa: E402


def run_one(source: Path, fps: float, motion: float, tmpdir: Path) -> dict:
    events = tmpdir / f"ev_{fps}_{motion}.json"
    cmd = [sys.executable, str(ROOT / "src" / "anxun" / "stream.py"),
           "--source", str(source), "--fps", str(fps), "--motion", str(motion),
           "--out", str(events)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"stream.py 失败 fps={fps} motion={motion}\n{r.stderr[-800:]}")
    d = json.loads(events.read_text(encoding="utf-8"))
    st = d["stats"]
    vm = stats_mod.compute([d])
    v = vm["metrics"]["verdict_coverage"]

    video_s = float(st.get("video_s") or 0) or 1.0
    # 容量折算用**固定参考耗时**，不用本次实测值：
    # 实测每次受 GPU 状态影响（22–26 ms 都出现过），拿它算容量表会每次都不一样，
    # 文档里的路数就没法复现、也没法被 review。参考值取 README 报的交付口径。
    REF_INFER_MS = 22.0
    # avg_infer_ms 属于「本次运行的性能」，落盘时被 _TIMING_KEYS 剔除了
    # （为了账本可复现），只能从 stdout 抓。
    hit = re.search(r"平均推理\s+([\d.]+)\s*ms", r.stdout)
    avg_ms = float(hit.group(1)) if hit else None
    infer_s = (avg_ms or 0) / 1000.0
    per_hour = st["frames_processed"] / video_s * 3600.0
    return {
        "fps": fps, "motion": motion,
        "video_s": round(video_s, 1),
        "frames_read": st["frames_read"],
        "frames_processed": st["frames_processed"],
        "frames_skipped": st["frames_skipped"],
        "gate_skipped": st.get("gate_skipped", 0),
        "persons_total": st["persons_total"],
        "persons_judgeable": st.get("persons_judgeable", 0),
        "judged": v["judged"],
        "coverage": v["value"],
        "coverage_raw": v["value_raw"],
        "events": st["events"],
        "raw_alarms": st["raw_alarms"],
        "avg_infer_ms": avg_ms,
        # 每路每小时推理次数；GPU 占用率按单次推理耗时折算
        "infer_per_hour": round(per_hour),
        "ref_infer_ms": REF_INFER_MS,
        "gpu_share_1stream": round(per_hour * (REF_INFER_MS / 1000.0) / 3600.0, 4),
        # 分母恒正（per_hour 由上面的 `or 1.0` 保证 video_s > 0），但仍保留判断，
        # 避免以后有人把 video_s 改成可空时踩除零。
        "max_streams_1gpu": (int(1.0 / (per_hour * (REF_INFER_MS / 1000.0) / 3600.0))
                             if per_hour > 0 else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "datasets" / "site_pan.mp4"))
    ap.add_argument("--fps", default="1,2,5,10")
    ap.add_argument("--motion", default="0,3,5,10")
    ap.add_argument("--out", default=str(ROOT / "data" / "sweep_stream.json"))
    ap.add_argument("--tmp", default=str(ROOT / ".sweep_tmp"))
    args = ap.parse_args()

    fps_list = [float(x) for x in args.fps.split(",")]
    motion_list = [float(x) for x in args.motion.split(",")]
    tmpdir = Path(args.tmp)
    tmpdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in fps_list:
        for m in motion_list:
            try:
                row = run_one(Path(args.source), f, m, tmpdir)
            except Exception as exc:                        # noqa: BLE001
                print(f"  [skip] fps={f} motion={m}: {exc}")
                continue
            rows.append(row)
            cov = row["coverage"]
            print(f"  fps={f:>4} motion={m:>4} | 推理 {row['frames_processed']:>4} "
                  f"| 可定论 {row['persons_judgeable']:>3} 有定论 {row['judged']:>3} "
                  f"| 覆盖 {cov if cov is not None else 0:>6.1%} "
                  f"| 旧口径 {row['coverage_raw'] if row['coverage_raw'] is not None else 0:>6.1%} "
                  f"| 事件 {row['events']:>3}")

    payload = {
        "schema": 1,
        "source": Path(args.source).name,
        "note": ("抽帧率决定『能看到多少人』，帧差阈值决定『多少帧白跑推理』。"
                 "容量按固定 22 ms 参考耗时折算（不随本次实测漂移）。"
                 "coverage 是新口径（分母=活到三帧窗口的轨迹），coverage_raw 是 v2.0 旧口径。"
                 "gpu_share_1stream 按单次推理耗时折算，仅用于容量估算，不是压测结果。"),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台三档表
    print("\n" + "=" * 78)
    print(f"{'抽帧':>5} {'帧差':>5} {'推理/分':>8} {'可定论':>7} {'有定论':>7} "
          f"{'覆盖率':>7} {'旧口径':>7} {'单卡路数':>8}")
    print("-" * 78)
    for r in rows:
        cov = r["coverage"] if r["coverage"] is not None else 0
        raw = r["coverage_raw"] if r["coverage_raw"] is not None else 0
        print(f"{r['fps']:>5.0f} {r['motion']:>5.0f} {r['infer_per_hour'] / 60:>8.0f} "
              f"{r['persons_judgeable']:>7} {r['judged']:>7} "
              f"{cov:>6.1%} {raw:>6.1%} {r['max_streams_1gpu'] or 0:>8}")
    print("=" * 78)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
