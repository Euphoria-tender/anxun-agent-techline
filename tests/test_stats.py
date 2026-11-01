# -*- coding: utf-8 -*-
"""价值指标回归测试（纯账本，不依赖模型权重）。

跑法：在 techline-v2.0.0 目录下

    python -m unittest discover -s tests -v

这几条都是会「不报错但算错数」的地方，不是形式化测试：
  1. 合并两份素材时 T1 串成同一个人 —— 分母凭空少一半
  2. 状态抖动时按条数算合规率 —— 抖得厉害的人权重更大
  3. 「无法判定」混进分母 —— 合规率被模型判不动的人稀释
  4. 一个人两样装备只违规一样 —— 被算成合规
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun import stats                                    # noqa: E402


def rec(track, equip, status, t0, t1=None, violation=False):
    return {"track": track, "equip": equip, "status": status,
            "t_video_s": t0, "t_last": t0 if t1 is None else t1,
            "violation": violation, "observations": 1,
            "region": "B-01", "stream": "main", "conf": 0.9, "reason": ""}


def bundle(ledger, persons=10, raw=20, events=None, video_s=60.0):
    return {"source": "x.mp4", "stats": {"persons_total": persons,
                                        "raw_alarms": raw, "video_s": video_s},
            "events": events if events is not None else [], "ledger": ledger}


class DominantState(unittest.TestCase):
    """主导状态要按累计时长取，不能按最后一条取。"""

    def test_longest_wins_not_last(self):
        rows = [rec("T1", "helmet", "未佩戴", 0, 10, True),
                rec("T1", "helmet", "已佩戴", 10, 11)]
        self.assertEqual(stats._dominant(rows), "未佩戴")

    def test_instant_records_fall_back_to_count(self):
        # 全部是瞬时记录（时长 0）时，谁观测次数多听谁的
        rows = [rec("T1", "helmet", "待复核", 5), rec("T1", "helmet", "已佩戴", 5),
                rec("T1", "helmet", "待复核", 5)]
        self.assertEqual(stats._dominant(rows), "待复核")


class MergeTracks(unittest.TestCase):
    """两份素材的 T1 不是同一个人。"""

    def test_same_name_different_file_not_merged(self):
        a = bundle([rec("T1", "helmet", "已佩戴", 0, 5)])
        b = bundle([rec("T1", "helmet", "未佩戴", 0, 5, True)])
        m = stats.merge([a, b])
        tracks = {r["track"] for r in m["ledger"]}
        self.assertEqual(tracks, {"s0:T1", "s1:T1"})
        self.assertEqual(stats.verdict_coverage(m["ledger"], 2)["judged"], 2)

    def test_merge_does_not_mutate_input(self):
        a = bundle([rec("T1", "helmet", "已佩戴", 0, 5)])
        stats.merge([a])
        self.assertEqual(a["ledger"][0]["track"], "T1")

    def test_timeline_shifted_so_buckets_dont_overlap(self):
        # 两份素材都从 0s 开始，合并后第二份要平移到第一份之后
        a = bundle([rec("T1", "helmet", "已佩戴", 0, 10)], video_s=100.0)
        b = bundle([rec("T1", "helmet", "未佩戴", 0, 5, True)], video_s=60.0)
        m = stats.merge([a, b])
        self.assertEqual(m["ledger"][0]["t_video_s"], 0.0)
        self.assertEqual(m["ledger"][1]["t_video_s"], 100.0)
        self.assertEqual(m["ledger"][1]["t_last"], 105.0)

    def test_duration_unaffected_by_offset(self):
        a = bundle([rec("T1", "helmet", "未佩戴", 0, 4, True)], video_s=100.0)
        b = bundle([rec("T1", "helmet", "未佩戴", 0, 2, True)], video_s=60.0)
        m = stats.merge([a, b])
        self.assertEqual(stats.durations(m["ledger"])["max"], 4.0)


class Compliance(unittest.TestCase):
    def test_any_violation_makes_person_violating(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5),
               rec("T1", "vest", "未佩戴", 0, 5, True)]
        c = stats.compliance(led)
        self.assertEqual((c["ok"], c["violation"]), (0, 1))
        self.assertEqual(c["value"], 0.0)

    def test_all_ok_is_compliant(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5),
               rec("T1", "vest", "已佩戴", 0, 5)]
        self.assertEqual(stats.compliance(led)["value"], 1.0)

    def test_review_and_unknown_excluded_from_denominator(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5),
               rec("T2", "helmet", "未佩戴", 0, 5, True),
               rec("T3", "helmet", "待复核", 0, 5),
               rec("T4", "helmet", "无法判定", 0, 5)]
        c = stats.compliance(led)
        self.assertEqual(c["n"], 2)              # 只有 T1 / T2 进分母
        self.assertEqual(c["value"], 0.5)
        self.assertEqual((c["review"], c["unknown"]), (1, 1))

    def test_no_judgeable_person_returns_none(self):
        led = [rec("T1", "helmet", "待复核", 0, 5)]
        self.assertIsNone(stats.compliance(led)["value"])


class Denoise(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(stats.denoise(100, 10)["value"], 0.9)

    def test_zero_raw_is_none_not_crash(self):
        self.assertIsNone(stats.denoise(0, 0)["value"])


class VerdictCoverage(unittest.TestCase):
    def test_fragmented_tracks_lower_coverage(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5)]
        self.assertEqual(stats.verdict_coverage(led, 10)["value"], 0.1)

    def test_never_exceeds_one(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5)]
        self.assertEqual(stats.verdict_coverage(led, 0)["value"], 1.0)


class Durations(unittest.TestCase):
    def test_only_violation_segments_counted(self):
        led = [rec("T1", "helmet", "未佩戴", 0, 4, True),      # 4s
               rec("T1", "vest", "未佩戴", 10, 12, True),      # 2s
               rec("T1", "helmet", "已佩戴", 20, 30)]          # 不算
        d = stats.durations(led)
        self.assertEqual(d["n"], 2)
        self.assertEqual(d["max"], 4.0)

    def test_zero_length_violation_ignored(self):
        led = [rec("T1", "helmet", "未佩戴", 5, 5, True)]
        self.assertEqual(stats.durations(led)["n"], 0)


class ComputeShape(unittest.TestCase):
    def test_all_five_metrics_present(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5),
               rec("T2", "helmet", "未佩戴", 0, 5, True)]
        vm = stats.compute([bundle(led, persons=2, raw=10)])
        keys = set(vm["metrics"])
        self.assertEqual(keys, {"coverage_ratio", "verdict_coverage", "denoise_ratio",
                                "compliance_rate", "violation_duration_s"})
        self.assertIn("generated_at", vm)
        self.assertNotIn("error", vm)

    def test_missing_ledger_raises_with_hint(self):
        import json
        import tempfile
        p = Path(tempfile.gettempdir()) / "_anxun_v1_events.json"
        p.write_text(json.dumps({"events": [], "stats": {}}), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            stats.load(p)
        self.assertIn("ledger", str(ctx.exception))
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
