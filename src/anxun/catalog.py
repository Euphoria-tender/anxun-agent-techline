# -*- coding: utf-8 -*-
"""
安巡 Agent · 厂区目录
=====================

技术线只需要两张表：**区域表**（定级用的 L/S 输入）和**场景表**（推理的输入样例）。

场景表来自 `datasets/scenes/scenes.json`，由现场素材生成；没有素材时目录为空，
推理接口仍可用，直接给图片路径即可。

坐标系与检测接口一致：像素坐标，原图尺寸。
"""

from __future__ import annotations

import json
from pathlib import Path

from .rules_engine import Region

# ============================================================
# 一、区域表
#
# violation_30d 决定 L（可能性），scene 决定 S（严重度），两者相乘得风险值 R。
# 这三列是定级唯一的输入，改区域就改这里。

REGIONS = [
    Region("B-01", "2 号车间高处作业平台", "高处", "王克俭", 7, "高处"),
    Region("A-03", "3 号车间东侧通道",     "普通", "李国强", 4, "普通"),
    Region("C-02", "原料库装卸区",         "车辆", "陈志远", 1, "普通"),
    Region("D-05", "喷漆房",               "动火", "刘海涛", 2, "动火"),
    Region("A-01", "一号车间主通道",       "消防通道", "赵敏", 3, "普通"),
]

REGION_MAP = {r.region_id: r for r in REGIONS}


# ============================================================
# 二、场景表（现场素材）

SCENES: list[dict] = []
SCENE_MAP: dict[str, dict] = {}

_SCENE_FILE = Path(__file__).resolve().parents[2] / "datasets" / "scenes" / "scenes.json"

if _SCENE_FILE.exists():
    try:
        for _s in json.loads(_SCENE_FILE.read_text(encoding="utf-8")):
            _p = _SCENE_FILE.parent / _s["file"]
            if not _p.exists():
                continue
            _s = dict(_s)
            _s["image"] = str(_p)
            _s.setdefault("frames", 1)      # 单张现场照片没有时序信息，不做聚合
            # vest 权重已就位。这里显式覆盖而不是 setdefault——场景文件里
            # 历史上写死了 ["helmet"]，setdefault 对它不起作用，改了口径却不生效。
            _s["equips"] = ["helmet", "vest"]
            _s.setdefault("camera", "CAM-" + _s["region_id"] + "-01")
            SCENES.append(_s)
            SCENE_MAP[_s["scene_id"]] = _s
    except Exception as _e:                 # 素材坏了不该拖垮服务
        print(f"[catalog] 场景加载失败：{_e}")


def public_scene(s: dict) -> dict:
    """对外的场景描述，不含图片路径（客户端只需要 id 和说明）。"""
    return {
        "scene_id": s["scene_id"],
        "region_id": s["region_id"],
        "region_name": REGION_MAP[s["region_id"]].name if s["region_id"] in REGION_MAP else "",
        "camera": s["camera"],
        "title": s["title"],
        "hint": s["hint"],
        "equips": s["equips"],
        "frames": s.get("frames", 1),
    }
