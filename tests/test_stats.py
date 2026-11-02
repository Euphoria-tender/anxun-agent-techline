# -*- coding: utf-8 -*-
"""价值指标回归测试（纯账本，不依赖模型权重）。

跑法：在技术线目录下

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
        """旧口径：10 条轨迹里只有 1 条有账本 -> 10%。

        v2.1.0 起 `value` 是新口径（分母只算活到三帧窗口的轨迹），
        旧口径挪到 `value_raw`。
        """
        led = [rec("T1", "helmet", "已佩戴", 0, 5)]
        self.assertEqual(stats.verdict_coverage(led, 10)["value_raw"], 0.1)

    def test_never_exceeds_one(self):
        led = [rec("T1", "helmet", "已佩戴", 0, 5)]
        self.assertEqual(stats.verdict_coverage(led, 0)["value_raw"], 1.0)

    def test_judgeable_denominator(self):
        """新口径：分母用「活到三帧窗口的轨迹数」，短命碎片不计。"""
        led = [rec("T1", "helmet", "已佩戴", 0, 5)]
        # 32 条轨迹里 12 条活到三帧，其中 9 条有定论
        v = stats.verdict_coverage(led, 32, persons_judgeable=12)
        self.assertEqual(v["judged"], 1)
        self.assertEqual(v["eligible"], 12)
        self.assertAlmostEqual(v["value"], 1 / 12, places=3)
        self.assertAlmostEqual(v["value_raw"], 1 / 32, places=3)

    def test_review_is_not_a_verdict(self):
        """待复核/无法判定不算有定论——v2.0 这里用 len(states)，把它们算进去了。"""
        led = [rec("T1", "helmet", "待复核", 0, 5)]
        v = stats.verdict_coverage(led, 4, persons_judgeable=4)
        self.assertEqual(v["judged"], 0)
        self.assertEqual(v["value"], 0.0)

    def test_missing_judgeable_falls_back(self):
        """v2.0 的历史 JSON 没有 persons_judgeable，退回按 observations 估算并打标。"""
        led = [dict(rec("T1", "helmet", "已佩戴", 0, 5), observations=5)]
        v = stats.verdict_coverage(led, 10)
        self.assertTrue(v["eligible_estimated"])
        self.assertEqual(v["eligible"], 1)


class ComplianceScope(unittest.TestCase):
    def test_helmet_only_excludes_vest(self):
        """合规率可以只统计某类装备。

        数据集里 vest 只有 51 个框 / 295 个人——83% 的人本来就没穿反光背心。
        按「两样都得合规」算，分母里几乎全是「没背心所以违规」，跑出来 0% 没有解释力。
        主口径只看 helmet，背心留在 compliance_rate_all 里看链路通不通。
        """
        led = [rec("T1", "helmet", "已佩戴", 0, 5),
               rec("T1", "vest", "未佩戴", 0, 5, True),
               rec("T2", "helmet", "未佩戴", 0, 5, True)]
        full = stats.compliance(led)
        helmet = stats.compliance(led, ("helmet",))
        self.assertEqual(full["ok"], 0)          # 全装备：T1 有背心违规
        self.assertEqual(full["violation"], 2)
        self.assertEqual(helmet["ok"], 1)        # 只看帽子：T1 合规
        self.assertEqual(helmet["violation"], 1)
        self.assertAlmostEqual(helmet["value"], 0.5, places=3)
        self.assertEqual(helmet["scope"], "helmet")

    def test_track_without_that_equip_leaves_denominator(self):
        """一条轨迹完全没评这类装备时，不该占用分母。"""
        led = [rec("T1", "vest", "未佩戴", 0, 5, True)]
        helmet = stats.compliance(led, ("helmet",))
        self.assertEqual(helmet["n"], 0)
        self.assertIsNone(helmet["value"])


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
        # v2.1.0 起多了 compliance_rate_all（背心口径），所以用「至少包含」而不是「恰好等于」：
        # 以后再加派生指标，不该让这条测试无谓地失败。
        self.assertLessEqual({"coverage_ratio", "verdict_coverage", "denoise_ratio",
                              "compliance_rate", "violation_duration_s"}, keys)
        self.assertIn("compliance_rate_all", keys)
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
