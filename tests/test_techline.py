# -*- coding: utf-8 -*-
"""技术线回归测试（不依赖模型权重，纯规则 / 编排 / 指标）。

跑法：在 techline-v1.2.0 目录下

    python -m unittest discover -s tests -v

这几条都是踩过的坑，不是形式化测试：
  1. 择优非单调 —— 多一个废框，把「已佩戴」翻成「未佩戴」
  2. 装备清单不跟模型走 —— 没背心权重时全员假违规
  3. 单图硬凑多帧 —— 「连续 3 帧一致」变成同义反复
  4. 指标口径 —— 漏报率把没检出的人从分母里删掉
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxun import metrics, pipeline                       # noqa: E402
from anxun.detectors import available_equips             # noqa: E402
from anxun.rules_engine import Box, Region, judge_equipment  # noqa: E402


class StubDetector:
    """只有类别的假检测器：验证「判定范围跟模型能力走」。"""

    name = "stub"
    kind = "model"

    def __init__(self, names, boxes=None):
        self.names = names
        self.boxes = boxes or []
        self.calls = 0

    def detect(self, scene, frame_idx=0):
        self.calls += 1
        return {"boxes": list(self.boxes), "infer_ms": 0.0,
                "backend": self.name, "kind": self.kind,
                "width": 1280, "height": 720}

    def detect_image(self, path, frame_idx=0):
        return self.detect({"scene_id": Path(path).stem, "image": str(path)}, frame_idx)


HELMET_ONLY = {0: "person", 1: "Hardhat", 2: "NO-Hardhat"}
HELMET_VEST = {0: "person", 1: "Hardhat", 2: "NO-Hardhat", 3: "vest", 4: "no_vest"}


class TestScoreMonotonic(unittest.TestCase):
    """判定择优必须单调：多一个不达标的框，不能把结论从「已佩戴」拉下来。"""

    def setUp(self):
        self.person = Box("person", 0.9, 100, 100, 300, 600)
        # 包含度 1.0、IoU 0.182 —— 靠包含度达标
        self.good = Box("helmet", 0.90, 170, 120, 230, 160)
        # IoU 0.231（更高）但包含度 0.586 —— 两项都不达标
        self.junk = Box("helmet", 0.88, 176.5, 165.1, 291.8, 219.1)

    def test_only_good_box(self):
        v = judge_equipment(self.person, 0, [self.good], "helmet")
        self.assertEqual(v.status, "已佩戴")

    def test_junk_box_cannot_flip_verdict(self):
        v = judge_equipment(self.person, 0, [self.good, self.junk], "helmet")
        self.assertEqual(v.status, "已佩戴",
                         "多一个不达标的框反而判成未佩戴：择优退化成按 IoU 取最大")

    def test_reason_points_at_the_matched_box(self):
        v = judge_equipment(self.person, 0, [self.good, self.junk], "helmet")
        self.assertAlmostEqual(v.iou, 0.182, places=2)


class TestAvailableEquips(unittest.TestCase):
    """装备清单必须按模型真实类别求交集。"""

    def test_helmet_only_model(self):
        self.assertEqual(available_equips(StubDetector(HELMET_ONLY)), ["helmet"])

    def test_helmet_and_vest_model(self):
        self.assertEqual(available_equips(StubDetector(HELMET_VEST)),
                         ["helmet", "vest"])

    def test_unknown_detector_returns_none(self):
        class NoNames:
            pass
        self.assertIsNone(available_equips(NoNames()))

    def test_no_vest_never_produces_vest_violation(self):
        """场景要判背心，但模型没有背心类 —— 这一项必须整条跳过。"""
        det = StubDetector(HELMET_ONLY, boxes=[Box("person", 0.95, 100, 100, 300, 600)])
        scene = {"scene_id": "T-01", "image": "x.jpg", "frames": 1,
                 "equips": ["helmet", "vest"]}
        can = available_equips(det)
        kept = [e for e in scene["equips"] if e in can]
        out = pipeline.inspect(scene, Region("B-01", "x", "高处", "y", 7),
                               det, equips=kept)
        self.assertEqual(out["equips"], ["helmet"])
        self.assertEqual([v["equip"] for v in out["final"]], ["helmet"],
                         "缺背心权重却仍在判背心：会把「没框」当成「没穿」")


class TestPreprocess(unittest.TestCase):
    """letterbox 的快路径与缩放映射，别为了省拷贝把坐标算错。"""

    def test_noop_when_already_square(self):
        import numpy as np
        from anxun.onnx_backend import letterbox
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        out, r, pad = letterbox(img, 640)
        self.assertIs(out, img)          # 不 resize 不拷贝
        self.assertEqual(r, 1.0)
        self.assertEqual(pad, (0, 0))

    def test_letterbox_maps_back_correctly(self):
        import numpy as np
        from anxun.onnx_backend import letterbox
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out, r, pad = letterbox(img, 640)
        self.assertEqual(out.shape, (640, 640, 3))
        self.assertAlmostEqual(r, 1.0)
        # 上下各留 80 px 灰边
        self.assertEqual(pad[0], 0)
        self.assertIn(pad[1], (79, 80))
        self.assertTrue((out[0, 0] == 114).all())


class TestTemporal(unittest.TestCase):
    """单图不许硬凑多帧；真多图才做时序聚合。"""

    def test_single_image_is_not_aggregated(self):
        det = StubDetector(HELMET_ONLY)
        scene = {"scene_id": "T-02", "image": "x.jpg", "frames": 3,
                 "equips": ["helmet"]}
        out = pipeline.inspect(scene, Region("B-01", "x", "高处", "y", 7), det)
        self.assertEqual(det.calls, 1, "同一张图被推理了多次")
        self.assertFalse(out["temporal"]["aggregated"])
        self.assertEqual(out["temporal"]["used"], 1)

    def test_multi_image_scene_aggregates(self):
        det = StubDetector(HELMET_ONLY)
        scene = {"scene_id": "T-03", "image": "a.jpg",
                 "images": ["a.jpg", "b.jpg", "c.jpg"],
                 "frames": 3, "equips": ["helmet"]}
        out = pipeline.inspect(scene, Region("B-01", "x", "高处", "y", 7), det)
        self.assertEqual(det.calls, 3)
        self.assertTrue(out["temporal"]["aggregated"])
        self.assertEqual(out["temporal"]["used"], 3)

    def test_per_track_window_stays_aligned(self):
        """视频链路是「一条 track 一个窗口」，人数变化不该串人。"""
        from anxun.rules_engine import judge_frame, aggregate_temporal
        f1 = judge_frame([Box("person", .9, 0, 0, 100, 300),
                          Box("helmet", .9, 10, 0, 90, 90)], ["helmet", "vest"])
        f2 = judge_frame([Box("person", .9, 0, 0, 100, 300),
                          Box("helmet", .9, 10, 0, 90, 90)], ["helmet", "vest"])
        final = aggregate_temporal([f1, f2, f2], need=3)
        self.assertEqual([v.equip for v in final], ["helmet", "vest"])
        self.assertEqual(final[0].status, "已佩戴")

    def test_misaligned_frames_abstain_instead_of_guessing(self):
        """帧间人数变化 -> 按下标比会串人，必须转人工而不是硬给结论。"""
        from anxun.rules_engine import judge_frame, aggregate_temporal
        f1 = judge_frame([Box("person", .9, 0, 0, 100, 300),
                          Box("person", .9, 200, 0, 300, 300),
                          Box("helmet", .9, 10, 0, 90, 90)], ["helmet"])
        f2 = judge_frame([Box("person", .9, 400, 0, 500, 300)],
                         ["helmet"])          # 甲离场、丙入画，项数从 2 变 1
        final = aggregate_temporal([f1, f2, f2], need=3)
        self.assertTrue(all(v.status == "待复核" for v in final))
        self.assertFalse(any(v.is_violation for v in final))
        self.assertIn("项数不一致", final[0].reason)


class TestMetricScope(unittest.TestCase):
    """误报/漏报的两个口径都要能算出来，且互不污染。"""

    def _samples(self):
        return [{
            "image_id": "img1",
            "gt_boxes": [], "pred_boxes": [],
            "truth": [
                # 已检出、真违规、却判「已佩戴」-> 判定层漏报
                {"person_id": "G0", "equip": "helmet", "worn": False, "matched": True},
                # 压根没检出、真违规 -> 整条链路漏报
                {"person_id": "G1", "equip": "helmet", "worn": False, "matched": False},
                # 已检出、真合规、判「已佩戴」
                {"person_id": "G2", "equip": "helmet", "worn": True, "matched": True},
            ],
            "judged": [
                {"person_id": "G0", "equip": "helmet", "status": "已佩戴"},
                {"person_id": "G2", "equip": "helmet", "status": "已佩戴"},
            ],
            "latency_ms": 1.0,
        }]

    def test_matched_scope_ignores_undetected(self):
        b = metrics.business_rates(self._samples(), scope="matched")
        self.assertEqual((b["n_violating"], b["n_compliant"]), (1, 1))
        self.assertEqual(b["fn_rate"], 100.0)

    def test_e2e_scope_counts_undetected(self):
        b = metrics.business_rates(self._samples(), scope="e2e")
        self.assertEqual((b["n_violating"], b["n_compliant"]), (2, 1))
        self.assertEqual(b["fn_rate"], 100.0)

    def test_evaluate_reports_both(self):
        res = metrics.evaluate(self._samples(), classes=("person", "helmet"))
        self.assertEqual(res["metric_scope"], "matched")
        self.assertEqual(res["n_violating"], 1)                  # headline 走 matched
        self.assertEqual(res["scope_detail"]["e2e"]["n_violating"], 2)
        self.assertEqual(res["coverage"]["n_truth_known"], 3)
        self.assertEqual(res["coverage"]["n_judged"], 2)
        self.assertEqual(res["coverage"]["rate"], 66.67)


class TestDeliverableGuards(unittest.TestCase):
    """交付文件的写保护与服务的请求体上限 —— 都是「一碰就出事、出事没提示」的地方。"""

    def test_default_out_follows_backend(self):
        """--backend onnx 不能默认写 metrics_real.json（那是 GPU 交付指标）。"""
        from anxun import evaluate_real as E
        self.assertEqual(E._default_out("torch").name, "metrics_real.json")
        self.assertEqual(E._default_out("onnx").name, "metrics_onnx.json")

    def test_body_cap_rejects_oversize(self):
        """声明超大的 Content-Length 必须当场回 413，不能先把内存读满。"""
        import io
        from anxun import app as A

        sent = {}

        class Fake:
            pass

        h = Fake()
        h.headers = {"Content-Length": str(A.MAX_BODY_BYTES + 1)}
        h.rfile = io.BytesIO(b"{}")
        h._json = lambda obj, code=200: sent.update(obj=obj, code=code)

        self.assertIsNone(A.Handler._body(h))
        self.assertEqual(sent["code"], 413)
        self.assertIn("上限", sent["obj"]["error"])

    def test_body_cap_allows_normal(self):
        import io
        from anxun import app as A

        class Fake:
            pass

        h = Fake()
        h.headers = {"Content-Length": "7"}
        h.rfile = io.BytesIO(b'{"a":1}')
        h._json = lambda obj, code=200: None
        self.assertEqual(A.Handler._body(h), {"a": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
