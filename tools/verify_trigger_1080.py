"""用真实 1080p 素材复验三级触发层（抽帧 / 帧差 / 人员门控）。

为什么需要复验
------------
触发层此前只在 `site_demo.mp4`（640x640 合成动画、15% 占用率）上验证过，
阈值 3.0 也是按它定的。真实部署是 1080p 摄像头，两个关键变量都变了：
  1. 分辨率 -> 编码噪声量级不同，帧差分布会平移，阈值可能失效；
  2. 画面内容 -> 真实照片，无规律运动，占用率也不是 15%。
README 的下一步里明确写了这一条：触发层省得越多，漏看的风险也越大，
必须用现场同规格的素材复核。

复验四个问题
-----------
Q1 1080p 下还能省多少（推理帧数 / 门控跳过 / 前向次数）
Q2 告警和全量推理是否一致（不漏不多 —— 可复现性）
Q3 帧差阈值 3.0 在 1080p + 机位抖动下是否仍成立（阈值要不要改）
Q4 1080p 下单帧耗时多少（真实部署的容量账）

真值对照
-------
本工具用的是 `datasets/ppe_test/test1080`，它带 YOLO 标注（归一化坐标，与分辨率无关）。
所以除了"两种模式互相对照"，还能按 evaluate_real 的口径算每张图应有的违规人次，
检查触发层是不是把真违规跳掉了。这是合成动画做不到的。

用法
----
    python tools/verify_trigger_1080.py [--fps 2] [--motion 3.0] [--jitter-only]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anxun import backend as backend_mod                       # noqa: E402
from anxun.catalog import REGION_MAP                           # noqa: E402
from anxun.evaluate_real import (iou_box, list_images, load_gt,  # noqa: E402
                                 read_names, truth_of)
from anxun.rules_engine import Region                           # noqa: E402
from anxun.stream import VideoInspector                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT = ROOT / "data" / "verify_trigger_1080.json"

# 视频里的第 i 张采样帧对应测试集的第几张图：视频按 sorted(images) 顺序生成，
# 每张停留 hold 帧，抽帧间隔 interval 秒。frame_index -> image_index 的映射在
# motion_diff_scan 里不涉及，这里只用于真值对照：告警时间 t 反查对应图片。
HOLD = 125      # 每张照片停留 5 秒（25fps）
FPS = 25


def frame_diff_scan(video: Path, interval: float) -> list[float]:
    """按抽帧节奏采样，算相邻采样帧的 160x160 灰度均值差。

    这个分布就是帧差阈值要面对的数据：阈值定在分布哪个分位上，
    决定了触发层是"省掉大部分帧"还是"几乎不省"。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开 {video}")
    step = max(1, int(round(FPS * interval)))
    prev, diffs = None, []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(cv2.absdiff(gray, prev).mean()))
            prev = gray
        i += 1
    cap.release()
    return diffs


def run_mode(det, video: Path, motion: float, gate: bool, fps: float,
             region: Region) -> tuple[list[dict], dict, float]:
    """跑一遍视频，返回 (事件, 统计, 墙钟秒)。"""
    if hasattr(det, "gate"):
        det.gate = gate
        det.gate_skipped = 0
    insp = VideoInspector(det, region, fps=fps, motion_threshold=motion,
                          verbose=False)
    t0 = time.perf_counter()
    insp.run(str(video))
    wall = time.perf_counter() - t0
    return insp.events, dict(insp.stats), wall


def event_key(ev: dict) -> tuple:
    """事件的比对键：装备 + 等级 + 风险值（时间戳不纳入，两种模式节奏不同）。"""
    return (ev.get("equip"), ev.get("grade"), ev.get("R"))


def expected_violations() -> dict:
    """按 evaluate_real 的口径算测试集里每张图的违规人次（helmet + vest）。"""
    names = read_names(DATA)
    gt_expect = {}
    for fp in list_images(DATA / "images"):
        gt = load_gt(DATA / "labels" / f"{fp.stem}.txt", names, 1920, 1920)
        n_h = sum(1 for g in gt if g["label"] == "person"
                  and _worn(gt, g, "helmet") is False)
        n_v = sum(1 for g in gt if g["label"] == "person"
                  and _worn(gt, g, "vest") is False)
        if n_h or n_v:
            gt_expect[fp.stem] = {"helmet": n_h, "vest": n_v}
    return gt_expect


def _worn(gt: list[dict], person: dict, equip: str) -> bool | None:
    """该人员是否佩戴了装备；没有对应标注时返回 None。"""
    pos, neg = f"{equip}", f"no{equip}"
    best_pos = best_neg = 0.5
    for g in gt:
        if g["label"] not in (pos, neg):
            continue
        v = iou_box(person, g)
        if g["label"] == pos:
            best_pos = max(best_pos, v)
        else:
            best_neg = max(best_neg, v)
    if best_pos < 0.5 and best_neg < 0.5:
        return None
    return best_pos >= best_neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=2.0, help="抽帧频率")
    ap.add_argument("--motion", type=float, default=3.0, help="帧差阈值")
    ap.add_argument("--hold", type=int, default=HOLD)
    ap.add_argument("--jitter-only", action="store_true", help="只测抖动版")
    ap.add_argument("--sweep", action="store_true",
                    help="额外扫一组帧差阈值，验证默认值是否合适")
    args = ap.parse_args()

    videos = [("real1080_jitter.mp4", "1080p + 机位抖动")]
    if not args.jitter_only:
        videos.insert(0, ("real1080.mp4", "1080p 纯静态切换"))

    det, note, kind = backend_mod.build(prefer="auto")
    print(f"后端：{det.name}（{note}）\n")

    region = REGION_MAP["B-01"]
    results = {}

    for name, label in videos:
        video = ROOT / "datasets" / name
        if not video.exists():
            print(f"缺 {name}，先跑 tools/make_1080_stream.py")
            continue
        print(f"===== {label}（{name}）=====")

        # Q3：帧差分布
        diffs = frame_diff_scan(video, 1.0 / args.fps)
        ds = sorted(diffs)
        pct = lambda q: round(ds[min(len(ds) - 1, int(len(ds) * q))], 3)
        skip_ratio = sum(1 for d in diffs if d < args.motion) / len(diffs)
        print(f"Q3 帧差分布（n={len(diffs)}）：中位 {pct(0.5)}  p90 {pct(0.9)}  "
              f"p99 {pct(0.99)}  最大 {ds[-1]:.1f}")
        print(f"   阈值 {args.motion} 下会被跳过的采样帧：{skip_ratio * 100:.1f}%"
              f"{'  ← 阈值失效，触发层白做' if skip_ratio < 0.1 else ''}")

        # Q1/Q2/Q4：全量 vs 三级触发
        ev_full, st_full, wall_full = run_mode(det, video, 0.0, False, args.fps, region)
        ev_trig, st_trig, wall_trig = run_mode(det, video, args.motion, True, args.fps, region)

        fwd_full = st_full["frames_processed"] * (3 if "vest" in det.name else 2)
        fwd_trig = st_trig["frames_processed"] * (3 if "vest" in det.name else 2) \
            - st_trig.get("gate_skipped", 0)
        print(f"Q1 全量      ：推理 {st_full['frames_processed']} 帧 · 前向 {fwd_full} 次")
        print(f"Q2 三级触发  ：推理 {st_trig['frames_processed']} 帧 · "
              f"静止跳过 {st_trig['frames_skipped']} · 门控跳过 {st_trig.get('gate_skipped', 0)} · "
              f"前向 {fwd_trig} 次（省 {(1 - fwd_trig / max(1, fwd_full)) * 100:.0f}%）")

        kf = [event_key(e) for e in ev_full]
        kt = [event_key(e) for e in ev_trig]
        miss = [k for k in kt if k not in kf]
        extra = [k for k in kf if k not in kt]
        print(f"   告警：全量 {len(kf)} 条 vs 触发 {len(kt)} 条 · "
              f"触发漏 {len(miss)} 条 · 触发多 {len(extra)} 条"
              f"{'  ✅ 一致' if not miss and not extra else '  ⚠️ 不一致'}")

        infer = [e for e in ev_full]
        ms_full = st_full["infer_ms_total"] / max(1, st_full["frames_processed"])
        ms_trig = st_trig["infer_ms_total"] / max(1, st_trig["frames_processed"])
        print(f"Q4 单帧耗时：{ms_full:.1f} ms（全量均值） / {ms_trig:.1f} ms（触发模式）· "
              f"墙钟 {wall_full:.1f}s vs {wall_trig:.1f}s")
        print()

        sweep = {}
        if args.sweep:
            print("阈值扫描（同一全量基准下）：")
            for m in (3.0, 6.0, 10.0, 15.0, 25.0):
                ev_s, st_s, _ = run_mode(det, video, m, True, args.fps, region)
                ks = [event_key(e) for e in ev_s]
                miss = [k for k in ks if k not in kf]
                extra = [k for k in kf if k not in ks]
                fwd = st_s["frames_processed"] * (3 if "vest" in det.name else 2) \
                    - st_s.get("gate_skipped", 0)
                ok = not miss and not extra
                print(f"   {m:>5.1f}：跳过 {st_s['frames_skipped']:>3} · "
                      f"前向 {fwd:>3}（省 {(1 - fwd / max(1, fwd_full)) * 100:>4.0f}%）· "
                      f"告警 {len(ks):>2} 条 {'✅' if ok else '⚠️ 不一致'}")
                sweep[str(m)] = {"frames_skipped": st_s["frames_skipped"],
                                 "forwards": fwd,
                                 "saving": round(1 - fwd / max(1, fwd_full), 4),
                                 "events": len(ks),
                                 "consistent": ok}
            print()

        results[name] = {
            "label": label,
            "threshold_sweep": sweep,
            "resolution": "1920x1920（真实照片 3 倍上采样）",
            "frame_diff": {"n": len(diffs), "p50": pct(0.5), "p90": pct(0.9),
                           "p99": pct(0.99), "max": round(ds[-1], 2),
                           "threshold": args.motion,
                           "skip_ratio": round(skip_ratio, 4)},
            "full": {"events": len(kf), "frames_processed": st_full["frames_processed"],
                     "forwards": fwd_full, "avg_ms": round(ms_full, 1),
                     "wall_s": round(wall_full, 2)},
            "triggered": {"events": len(kt), "frames_processed": st_trig["frames_processed"],
                          "frames_skipped": st_trig["frames_skipped"],
                          "gate_skipped": st_trig.get("gate_skipped", 0),
                          "forwards": fwd_trig,
                          "saving": round(1 - fwd_trig / max(1, fwd_full), 4),
                          "avg_ms": round(ms_trig, 1), "wall_s": round(wall_trig, 2),
                          "event_keys": [list(k) for k in kt]},
            "full_event_keys": [list(k) for k in kf],
            "consistency": {"missing": len(miss), "extra": len(extra),
                            "identical": not miss and not extra},
        }

    # 真值期望（不随模式变，作为背景数据）
    expect = expected_violations()
    results["_gt_expect"] = {
        "images_with_violation": len(expect),
        "helmet_person_frames": sum(v["helmet"] for v in expect.values()),
        "vest_person_frames": sum(v["vest"] for v in expect.values()),
    }
    print(f"真值：{len(expect)} 张图有违规 · helmet {results['_gt_expect']['helmet_person_frames']} "
          f"人次 · vest {results['_gt_expect']['vest_person_frames']} 人次")
    print("（视频链路易出的是「告警条数」而非「违规人次」，两者不直接相等："
          "三帧聚合 + 事件去重会合并同一个人。）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
