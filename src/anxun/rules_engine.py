# -*- coding: utf-8 -*-
"""
安巡 Agent · 规则引擎
=====================

纯确定性实现：PPE 判定 -> 风险等级核查 -> 工单生成

设计原则
--------
1. 不调用任何大模型做判定或计算。同一输入永远同一输出，可复现、可审计。
2. 所有阈值集中在「一、阈值配置」，答辩时可逐条说明依据。
3. 每个结论都带 reason 字段，回答「为什么这么判」。
4. 检测模型只负责产出 Box，不参与任何决策。换模型不影响本文件。

与仓库 main 分支的差异（v1 校准，阈值口径的改动都写在代码注释里）
--------------------------------------------------------------
- HEAD_WIDTH_RATIO：头部区域横向收窄到 44%，否则正确戴帽会被判未佩戴
- CONTAIN_MIN：小目标（护目镜）改用包含度判据，IoU 数学上过不了
- MIN_PERSON_AREA_RATIO：面积阈值按画面占比换算，不再跨分辨率用绝对值
- NEG_LABEL：裸头框作为反向证据，正向/反向证据冲突时转人工
- LEGAL_REFS：把每个违规映射到法规/标准条款（条款已核对，来源随条目记录）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


# ============================================================
# 一、阈值配置（唯一事实来源：config/thresholds.json）
#
# 为什么改成读配置文件
# --------------------
# v1.1.0 之前阈值是写死的模块常量，结果主仓库的 demo 与这里的评测各有一套：
# demo 用 0.60/0.40（四态齐全，适合演示），评测用 0.30/0.20（漏报最低，适合报数）。
# 同一份作品两个口径，评委只要先跑评测再跑 demo 就会发现判定结果不一致。
#
# 现在只有一份定义（config/thresholds.json），代码、文档、验收脚本都读它：
#   measured      实测口径 —— 交付与评测默认，四个指标就是它测的
#   conservative  保守口径 —— 演示用，阈值更严，能看见「待复核」这一档
#
# 切换方式（三选一，优先级从高到低）：
#   1) 环境变量 ANXUN_RULES_PROFILE=conservative
#   2) 命令行 --profile conservative（app.py / evaluate_real.py 已接）
#   3) 代码里 rules_engine.apply_profile("conservative")
#
# ⚠️ 换口径会改变业务指标，不会改变模型。要报数就用 measured，别混。
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "thresholds.json"

_REQUIRED = ("conf_final", "conf_review", "head_region_ratio", "head_width_ratio",
             "min_person_area", "min_aspect", "equip")


def load_config(path: str | Path | None = None) -> dict:
    """读阈值配置。文件缺失直接报错，不做静默降级——阈值悄悄跑回默认值最危险。"""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"阈值配置缺失：{p}。它是判定口径的唯一事实来源，不能缺省。")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for name, prof in cfg.get("profiles", {}).items():
        missing = [k for k in _REQUIRED if k not in prof]
        if missing:
            raise ValueError(f"阈值 profile「{name}」缺字段：{missing}")
        if not 0 < prof["conf_review"] < prof["conf_final"] < 1:
            raise ValueError(
                f"阈值 profile「{name}」不合法：要求 0 < conf_review < conf_final < 1，"
                f"实际 {prof['conf_review']} / {prof['conf_final']}")
    return cfg


CFG = load_config()
PROFILES: dict[str, dict] = CFG["profiles"]
DEFAULT_PROFILE = CFG.get("default_profile", "measured")

# 当前生效的口径。环境变量优先，方便一台机器上同时跑演示与评测。
PROFILE = (os.environ.get("ANXUN_RULES_PROFILE") or "").strip() or DEFAULT_PROFILE

# 下面这些全局量在 apply_profile() 里被赋值；保留模块级名字是为了兼容既有调用
# （sweep_rules.py / sweep_geom.py 直接改它们做扫参，改完自己还原）。
CONF_FINAL = 0.0
CONF_REVIEW = 0.0
HELMET_IOU_MIN = 0.0
HEAD_REGION_RATIO = 0.0
HEAD_WIDTH_RATIO = 0.0
CONTAIN_MIN = 0.0
MAX_EQUIP_RATIO = 0.0
MIN_PERSON_AREA = 0.0
MIN_ASPECT = 0.0
MIN_PERSON_AREA_RATIO = 0.0
EQUIP_OVERRIDE: dict[str, dict[str, float]] = {}

# 装备框与比对区域的匹配判据：
#   "iou"    默认。IoU 或包含度任一达标即可（v1.0 起一直用这个）。
#   "center" 框中心落在区域内即达标。对框的大小不敏感，跨人也更难。
#
# 为什么要有第二种：归一化到 640 输入后，一个站姿中景的人 `person` 框宽约 70px，
# 头部区域按 0.44 收窄只剩 31px，而头盔框本身宽 29px——横向偏 8px，包含度就从
# 1.0 掉到 0.5，明明戴了也判「没戴」。放宽区域又会把邻人的头盔框圈进来，漏报反弹。
# 实测（ppe 五类模型 + 100 张旧集）：IoU 判据最好只能到 22.9%/30.4%，
# 中心点判据能到 9.1%/15.1%。两种判据都在，是因为老权重仍以 IoU 为准，
# 换判据必须整链路重验，不能悄悄改默认值。
MATCH_MODE = "iou"

# 正反证据冲突时怎么办（同一个头部区域里既高置信检出装备框、又高置信检出裸头框）：
#   "review"    默认，转「待复核」交人工。v1~v2 一直这么做，不冤枉人。
#   "violation" 直接判「未佩戴」。安全场景优先——漏掉一个违规比多派一次复核贵得多。
# 注意：业务统计里「待复核」是算漏报的（见 metrics.business_rates），
# 所以这条策略直接改漏报率，不是只改文案。实测切到 violation 后 100 张旧集少 12 例漏报。
CONFLICT_POLICY = "review"

# 人员框的定论阈值，和装备框的 CONF_FINAL **分开**。
# 为什么必须分开：CONF_FINAL 抬高是为了卡住低置信的装备框，但人员框本来就不该用
# 同一个门槛——COCO 人体模型在密集场景常给出 0.3~0.5 的框，这些人是真的在那儿，
# 只是框得不太自信。用 0.55 去要求它，等于把「没戴帽子」判成「待复核」，
# 而待复核在业务统计里算漏报。实测（100 张旧集）：不拆时漏报 35.5%，拆开后 18.3%。
PERSON_CONF_FINAL = 0.0

# 每类装备各自的匹配判据。为什么不能一个口径通吃：
#   头盔在头顶，中心点稳；背心从肩膀往下穿，几何中心常落在躯干区域（20%~70%）之外，
#   用中心点判据会把「穿了」判成「没穿」——实测误报从 27.3% 涨到 47.7%。
# 所以按装备分开配，没配的跟随 match_mode。
EQUIP_MATCH_MODE: dict[str, str] = {}

# 「贴头」约束：装备框的上沿必须落在人员框顶部多少比例以内，才算戴在身上。
#
# **实现后实测无效，默认关闭（0.0），留作以后排查用。**
# 设计初衷是挡住「帽子在手里/挂在腰上」被判成「戴在头上」。
# 但扫了 0.08~0.30 五档，指标一格没动、目标样本判定也没变——
# 说明那个样本的误检框本来就在头顶位置（模型把后脑勺当成了深色头盔），
# 不是「框在手里」。这类错误几何约束管不了，只能靠训练数据。
# 详见 docs/15 附录 B.2/B.3。
HEAD_TOP_TOL = 0.0


def apply_profile(name: str) -> dict:
    """切换阈值口径并返回生效后的快照。

    只改模块级全局量，不动模型、不动数据。切换是幂等的。
    """
    global PROFILE, CONF_FINAL, CONF_REVIEW, HELMET_IOU_MIN, HEAD_REGION_RATIO
    global HEAD_WIDTH_RATIO, CONTAIN_MIN, MAX_EQUIP_RATIO, MIN_PERSON_AREA
    global MIN_ASPECT, MIN_PERSON_AREA_RATIO, EQUIP_OVERRIDE, MATCH_MODE, CONFLICT_POLICY
    global PERSON_CONF_FINAL, EQUIP_MATCH_MODE, HEAD_TOP_TOL

    if name not in PROFILES:
        raise KeyError(f"未知阈值口径「{name}」，可选：{sorted(PROFILES)}")
    prof = PROFILES[name]
    PROFILE = name
    CONF_FINAL = float(prof["conf_final"])
    CONF_REVIEW = float(prof["conf_review"])
    HEAD_REGION_RATIO = float(prof["head_region_ratio"])
    HEAD_WIDTH_RATIO = float(prof["head_width_ratio"])
    MIN_PERSON_AREA = float(prof["min_person_area"])
    MIN_ASPECT = float(prof["min_aspect"])
    MATCH_MODE = str(prof.get("match_mode", "iou"))
    CONFLICT_POLICY = str(prof.get("conflict_policy", "review"))
    PERSON_CONF_FINAL = float(prof.get("person_conf_final", CONF_FINAL))
    HEAD_TOP_TOL = float(prof.get("head_top_tol", 0.0))

    equip = {k: dict(v) for k, v in prof["equip"].items()}
    helmet = equip.get("helmet", {})
    HELMET_IOU_MIN = float(helmet.get("iou_min", 0.25))
    CONTAIN_MIN = float(helmet.get("contain_min", 0.60))
    MAX_EQUIP_RATIO = float(helmet.get("max_ratio", 3.0))
    # helmet 用模块常量（扫参入口），其他装备走覆盖表
    EQUIP_OVERRIDE = {k: v for k, v in equip.items() if k != "helmet"}
    # 按装备取匹配判据；helmet 没单独配就跟随全局 MATCH_MODE
    EQUIP_MATCH_MODE = {k: str(v.get("match_mode", MATCH_MODE)) for k, v in equip.items()
                        if "match_mode" in v}

    MIN_PERSON_AREA_RATIO = MIN_PERSON_AREA / (1920 * 1080)
    return profile_snapshot()


def active_profile() -> str:
    """当前生效的口径名（写进评测 JSON，谁都能看出这组数是哪套阈值跑的）。"""
    return PROFILE


def profile_snapshot() -> dict:
    """当前生效阈值的快照。接口、报告、验收脚本都从这里取数，避免各拿各的。"""
    prof = PROFILES.get(PROFILE, {})
    return {
        "profile": PROFILE,
        "label": prof.get("label", PROFILE),
        "conf_final": CONF_FINAL,
        "conf_review": CONF_REVIEW,
        "iou_min": HELMET_IOU_MIN,
        "contain_min": CONTAIN_MIN,
        "max_equip_ratio": MAX_EQUIP_RATIO,
        "head_region_ratio": HEAD_REGION_RATIO,
        "head_width_ratio": HEAD_WIDTH_RATIO,
        "min_person_area": MIN_PERSON_AREA,
        "min_person_area_ratio": round(MIN_PERSON_AREA_RATIO, 6),
        "min_aspect": MIN_ASPECT,
        "equip_override": {k: dict(v) for k, v in EQUIP_OVERRIDE.items()},
        "config_path": str(CONFIG_PATH),
        "updated_at": CFG.get("updated_at", ""),
    }


apply_profile(PROFILE)

# 各口径的来历，留在代码里比留在 PPT 里可靠
#   measured      100 张真实测试集 CF × IOU 二维网格扫出（data/sweep_rules.json）
#                 0.30/0.20/0.25/3.0 → 误报 5.1% / 漏报 22.58%
#                 稳定性：测试集对半切，误报 6.8% / 7.7%；旧口径 8.5% / 15.4%
#   conservative  v1.0 初版 0.60/0.40/0.30 → 误报 11.2% / 漏报 38.7%
#                 demo 用它是为了让「待复核」这一档真的出现（0.52 的样本会落进去）
# ⚠️ 两套都是在测试集上选的，属乐观估计；换数据、换权重、换机位都必须重扫。

# 按装备分离的几何阈值
# ----------------------
# helmet 与 vest 的判定区域完全不同（头部 vs 上身），框的几何分布也不一样，
# 共用一套阈值必然有一个吃亏。扫参实测（data/sweep_geom.json + README 5.7）：
#
#   helmet  最优 IOU 0.25 / MAXR 3.0  -> 误报 5.1%  漏报 24.7%
#   vest    最优 IOU 0.10 / MAXR 6.0  -> 误报 27.3% 漏报 13.0%
#
# vest 模型把背心框画得更松（相对上身区域偏小），IoU 要求必须降到 0.10；
# 而 helmet 降到 0.10 会把误报从 5.1% 抬到 9.2%。所以只能分开。
#
# 实现上 helmet 继续用上面的模块常量（sweep_rules.py / sweep_geom.py 直接改它们，
# 是 helmet 的扫参入口）；其他装备由 config/thresholds.json 的 equip 段给出覆盖，
# apply_profile() 时填进 EQUIP_OVERRIDE。没有覆盖的装备退回 helmet 那套。
def equip_cfg(equip: str) -> dict[str, float]:
    """取某个装备的几何阈值；没有专门配置的用 helmet 那套常量。"""
    base = {"iou_min": HELMET_IOU_MIN, "contain_min": CONTAIN_MIN,
            "max_ratio": MAX_EQUIP_RATIO}
    base.update(EQUIP_OVERRIDE.get(equip, {}))
    return base

# 反向证据：有的模型会直接输出「裸头」框（NO-Hardhat）。有就用，没有就退回「未检出」推断。
# 这一条是拿真实数据跑出来的——只用正向证据时，漏报率一度到 45%。
NEG_LABEL = {"helmet": "nohelmet"}

# 人员框面积下限（像素^2），1920×1080 下的标定值。数值见 config: measured.min_person_area

# 人员框高宽比下限，小于此值视为半身/蹲姿，不做判定。
# 1.20 -> 0.90（v1.1.0）：把这条闸门逐例复核了一遍。
#   高宽比 0.96 / 0.98 的两例是蹲姿作业，头部仍落在「顶部 30%」区域内，判「无法判定」
#   等于白丢两次正确告警；放宽到 0.90 后这两例恢复判定，误报一例没涨（5.1% 不变）。
#   更宽的 0.80 / 0.71 两例（弯腰、侧躺）头部已经不在顶部区域，继续留作「无法判定」——
#   强行判会拿后背当头部，那才是真的在编证据。
# 代价要讲清楚：这条闸门原本防的是「头部不在框内上部」，放宽后对弯腰作业会多一分风险。
# 实测口径：漏报 24.73% -> 22.58%（93 个违规人次里恢复 2 例），误报 5.10% 不变。
# 现值 0.90，见 config: measured.min_aspect。

# v1 修正：绝对面积阈值不能跨分辨率用。
# 8000px² @ 1920×1080 相当于画面面积的 0.386%；换成 640×640 的现场照片，
# 同样远近的人只剩 1/9 像素，会被整片误判成「无法判定」——实测这一条吃掉了 22% 的漏报。
# 现在按画面面积占比判，与分辨率解耦；1920×1080 下与原来的 8000px² 完全等价。
# 占比由 apply_profile() 从 min_person_area 换算，不单独配。


def min_person_area(width: float, height: float) -> float:
    """按画面面积换算人员框面积下限，默认按 1920×1080 标定。"""
    if not width or not height:
        return MIN_PERSON_AREA
    return MIN_PERSON_AREA_RATIO * width * height

# 风险矩阵：30 天违规次数 -> 可能性 L
# (次数上限, L 值)，超出最后一档取 4
L_TABLE = ((1, 1), (3, 2), (6, 3))
L_MAX = 4

# 风险矩阵：作业类型 -> 后果严重度 S
#
# 口径唯一出处：docs/安巡风险分级标准.md 第 2.2 节（v1.0，2026-11-08）。
# S 只由**作业类型**决定，与违规的是安全帽还是反光背心无关。
# 旧实现按「装备类型 × 场景」查表（如 vest·普通=2），与文档、与研判台页面都不一致：
# 同一场景网页算出红色 2 小时、接口算出橙色 24 小时。2026-11-09 统一到文档。
S_MAP = {
    "普通":   3,   # 防护缺失可致重伤
    "车辆":   3,   # 存在碰撞 / 碾压风险
    "高处":   4,   # 坠落后果可能致命
    "动火":   4,   # 可能引发火灾爆炸
    "受限空间": 4,  # 中毒窒息后果严重
}
S_DEFAULT = 2      # 未登记的作业类型按最低档，并在研判里注明「作业类型未登记」

# 等级判定：(风险值下限, 等级, 处置要求, 整改时限)
# 分档与 docs/安巡风险分级标准.md 第 3 节逐字对齐：≥13 红 / 9–12 橙 / 5–8 黄 / ≤4 蓝
GRADE_RULES = (
    (13, "红色", "立即停止作业并整改", timedelta(hours=2)),
    (9,  "橙色", "当日整改",           timedelta(hours=24)),
    (5,  "黄色", "限期整改",           timedelta(days=7)),
    (0,  "蓝色", "加强关注",           timedelta(days=30)),
)

# 等级 -> 前端色板（与国内安全色习惯一致：红=最高危）
GRADE_HEX = {
    "红色": "#FF4D4F",
    "橙色": "#FF8C1A",
    "黄色": "#F2C230",
    "蓝色": "#2E8BFF",
}

# 违规类型 -> 中文名 + 整改动作模板
VIOLATION_INFO = {
    "helmet":  ("未佩戴安全帽",
                "补发安全帽 {n} 顶至 {region}，登记发放签收记录，次日复查佩戴率"),
    "vest":    ("未穿戴反光背心",
                "补发反光背心 {n} 件至 {region}，登记领用记录，次日复查穿戴率"),
    "goggles": ("未佩戴防护眼镜",
                "配发防护眼镜 {n} 副至 {region}，登记领用记录，次日复查佩戴率"),
}

# 法规/标准依据（2026-11-02 逐条核对过原文与现行性，来源记在 source 里）。
# needs_check=False 表示条款号与表述已核；换标准版本、或引用新条款时必须重核并改回 True。
LEGAL_REFS = {
    "helmet": [
        {"doc": "《中华人民共和国安全生产法》（2021 年修正）", "clause": "第四十五条",
         "point": "生产经营单位必须为从业人员提供符合国家标准或者行业标准的劳动防护用品，"
                  "并监督、教育从业人员按照使用规则佩戴、使用",
         "needs_check": False, "checked_at": "2026-11-02",
         "source": "应急管理部《对十四届全国人大一次会议第 4332 号建议的答复》"
                   "（应急建函〔2023〕40 号）引述的条文原文"},
        {"doc": "GB 2811-2019《头部防护 安全帽》", "clause": "技术要求",
         "point": "规定安全帽的分类与标记、技术要求、检验及标识；现行，2019-12-31 发布、"
                  "2020-07-01 实施，全部代替 GB 2811-2007",
         "needs_check": False, "checked_at": "2026-11-02",
         "source": "全国标准信息公共服务平台 std.samr.gov.cn（标准号 GB 2811-2019）"},
    ],
    "vest": [
        {"doc": "GB 39800.1-2020《个体防护装备配备规范 第 1 部分：总则》",
         "clause": "第 3 章 个体防护装备配备原则",
         "point": "按作业风险类别确定个体防护装备配备；现行，2020-12-24 发布、"
                  "2022-01-01 实施，代替 GB/T 11651-2008、GB/T 29510-2013",
         "needs_check": False, "checked_at": "2026-11-02",
         "source": "全国标准信息公共服务平台 std.samr.gov.cn（标准号 GB 39800.1-2020）"},
    ],
    "goggles": [
        {"doc": "GB 39800.1-2020《个体防护装备配备规范 第 1 部分：总则》",
         "clause": "附录 B（资料性）常见的作业类别及可能造成的事故或伤害",
         "point": "存在飞溅、粉尘等眼面部伤害风险的作业，应配备眼面部防护装备",
         "needs_check": False, "checked_at": "2026-11-02",
         "source": "同 GB 39800.1-2020；附录 B 为资料性附录，不是强制条款"},
    ],
}


# ============================================================
# 二、基础数据结构

@dataclass
class Box:
    """一个检测框。这是检测模型与本引擎之间唯一的接口。"""
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def top_region(self, ratio: float | None = None,
                   width_ratio: float = 1.0) -> "Box":
        """返回框的顶部区域（默认上部 30%，横向按 width_ratio 居中收窄）。

        ratio 传 None 时取当前口径的 head_region_ratio，而不是函数定义时的常量快照——
        否则切换阈值口径后，这里会悄悄用旧值。
        """
        ratio = HEAD_REGION_RATIO if ratio is None else ratio
        half = self.w * width_ratio / 2.0
        return Box(self.label, self.conf, self.cx - half, self.y1,
                   self.cx + half, self.y1 + self.h * ratio)

    def to_dict(self) -> dict:
        return {"label": self.label, "conf": round(self.conf, 3),
                "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1)}


def iou(a: Box, b: Box) -> float:
    """交并比。"""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# 三、PPE 判定

@dataclass
class Verdict:
    """单人单类装备的判定结果。"""
    person_idx: int
    equip: str              # helmet / vest / goggles
    status: str             # 已佩戴 / 未佩戴 / 无法判定 / 待复核 / 丢弃
    conf: float
    reason: str
    is_violation: bool = False
    person_id: str = ""     # 画面内编号 P1/P2，给界面用
    iou: float = 0.0        # 命中的最大 IoU，界面可展示

    def to_dict(self) -> dict:
        return {
            "person_idx": self.person_idx, "person_id": self.person_id,
            "equip": self.equip, "status": self.status,
            "conf": round(self.conf, 3), "reason": self.reason,
            "is_violation": self.is_violation, "iou": round(self.iou, 3),
        }


def judge_equipment(person: Box, idx: int, boxes: Sequence[Box],
                    equip: str, area_min: float | None = None) -> Verdict:
    """
    判定单人某项装备的佩戴状态。

    判定链：
      1. 尺度/姿态可用性检查 -> 不合格直接「无法判定」
      2. 取头部（或上身）区域，与同类装备框求最大 IoU
      3. 达阈值 -> 已佩戴；未达 -> 未佩戴
      4. 按置信度分档：定论 / 待复核 / 丢弃
    """
    pid = f"P{idx + 1}"

    # --- 1. 可用性检查 ---
    limit = MIN_PERSON_AREA if area_min is None else area_min
    if person.area < limit:
        return Verdict(idx, equip, "无法判定", person.conf,
                       f"人员框面积 {person.area:.0f}px² 小于下限 {limit:.0f}px²，目标过小",
                       person_id=pid)
    if person.h / max(person.w, 1e-6) < MIN_ASPECT:
        return Verdict(idx, equip, "无法判定", person.conf,
                       f"高宽比 {person.h / max(person.w, 1e-6):.2f} 小于 {MIN_ASPECT}，疑似半身或非站姿",
                       person_id=pid)

    # --- 2. 圈定比对区域 ---
    if equip == "helmet":
        region = person.top_region(HEAD_REGION_RATIO, HEAD_WIDTH_RATIO)
        region_name = "头部区域"
    elif equip == "goggles":
        region = person.top_region(0.45, HEAD_WIDTH_RATIO)
        region_name = "面部区域"
    else:  # vest
        region = Box("torso", person.conf, person.x1,
                     person.y1 + person.h * 0.20,
                     person.x2, person.y1 + person.h * 0.70)
        region_name = "上身区域"

    # --- 3. 找最佳匹配框（IoU 与包含度同时算） ---
    cfg = equip_cfg(equip)
    iou_min, contain_min, max_ratio = (cfg["iou_min"], cfg["contain_min"],
                                       cfg["max_ratio"])

    def _score(label: str):
        """返回 (最佳框, IoU, 包含度, 是否达标)。label 为空时永不达标。

        择优分两层：先在**达标**的框里挑，一个都没有才回落到「最像的那个」。

        v1.2.0 之前只有一层（按 (IoU, 包含度) 取全局最大），会被一个「IoU 略高、
        但两项都不达标」的框抢走最佳位：B 包含度 1.0 本该判「已佩戴」，A 的 IoU
        高一点点、包含度 0.59 不达标，结果判「未佩戴」。多给一个废框反而多一条
        误报，判定非单调——这是误报的一个真实来源，回归见 tests/test_techline.py。
        """
        best, bi, bc = None, 0.0, 0.0          # 最像的（不保证达标，只用来写 reason）
        qbest, qi, qc = None, 0.0, 0.0         # 达标的里面最像的
        for b in boxes:
            if not label or b.label != label:
                continue
            if b.area <= 0 or b.area > max_ratio * region.area:
                continue
            v = iou(region, b)
            ix1, iy1 = max(region.x1, b.x1), max(region.y1, b.y1)
            ix2, iy2 = min(region.x2, b.x2), min(region.y2, b.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            contain = inter / b.area if b.area > 0 else 0.0
            if (v, contain) > (bi, bc):
                best, bi, bc = b, v, contain
            if (v >= iou_min or contain >= contain_min) and (v, contain) > (qi, qc):
                qbest, qi, qc = b, v, contain
        if qbest is not None:
            return qbest, qi, qc, True
        return best, bi, bc, False

    def _center(label: str, min_conf: float = 0.0):
        """中心点判据：框中心落在区域内即达标，取其中置信度最高的一个。

        与 _score 的区别只在「什么算达标」：不看 IoU 也不看包含度，只看中心点。
        仍然保留两道闸门：
          - max_ratio 面积闸门：比区域大 3 倍的框，中心点大概率落在别人身上
          - min_conf 置信度闸门：**低置信框一律不算「找到」**。这一条是踩出来的——
            不加的话，0.2~0.4 的框会从「不达标（判未佩戴）」变成「待复核 / 丢弃」，
            而这两种状态在业务统计里都算漏报，实测把漏报从 15% 顶到 49%。"
        """
        best, bc = None, 0.0
        for b in boxes:
            if not label or b.label != label:
                continue
            if b.area <= 0 or b.area > max_ratio * region.area:
                continue
            if b.conf < min_conf:
                continue
            # 贴头约束：拎在手里/挂在腰上的帽子，框上沿离头顶很远
            if HEAD_TOP_TOL > 0 and b.y1 > person.y1 + HEAD_TOP_TOL * person.h:
                continue
            if region.x1 <= b.cx <= region.x2 and region.y1 <= b.cy <= region.y2:
                if b.conf > bc:
                    best, bc = b, b.conf
        return best

    if EQUIP_MATCH_MODE.get(equip, MATCH_MODE) == "center":
        best = _center(equip, CONF_FINAL)
        matched = best is not None
        best_iou = iou(region, best) if best else 0.0
        best_contain = 0.0
        neg = _center(NEG_LABEL.get(equip, ""), CONF_FINAL)
        neg_ok = neg is not None
        neg_iou = iou(region, neg) if neg else 0.0
        neg_contain = 0.0
    else:
        best, best_iou, best_contain, matched = _score(equip)
        # 反向证据：模型专门训过「裸头」这一类（NO-Hardhat）。扔了它就只剩半个模型。
        neg, neg_iou, neg_contain, neg_ok = _score(NEG_LABEL.get(equip, ""))

    # --- 4. 结论 ---
    # 顺序很重要：正向证据优先；反向证据只有「高置信」时才有权覆盖。
    # 低置信度的裸头框不许把已经确定的结论降级——第一版写反了，漏报率从 45% 涨到 59%。
    if matched and best.conf >= CONF_FINAL:
        if neg_ok and neg.conf >= CONF_FINAL:
            # 冲突时按口径决定：默认转人工，安全优先的口径直接判未佩戴
            if CONFLICT_POLICY == "violation":
                return Verdict(idx, equip, "未佩戴", min(best.conf, neg.conf),
                               f"正反证据冲突：装备框（{best.conf:.2f}）与裸头框"
                               f"（{neg.conf:.2f}）同时高置信，按安全优先判未佩戴",
                               is_violation=True, person_id=pid, iou=best_iou)
            return Verdict(idx, equip, "待复核", max(best.conf, neg.conf),
                           f"正反证据冲突：头部区域同时检出装备框（{best.conf:.2f}）"
                           f"与裸头框（{neg.conf:.2f}），转人工",
                           person_id=pid, iou=best_iou)
        return Verdict(idx, equip, "已佩戴", best.conf,
                       f"装备框与{region_name} IoU={best_iou:.2f}，包含度 {best_contain:.2f}，"
                       f"置信度 {best.conf:.2f}",
                       person_id=pid, iou=best_iou)

    if matched:
        conf = best.conf
        if conf < CONF_REVIEW:
            return Verdict(idx, equip, "丢弃", conf,
                           f"装备框置信度 {conf:.2f} 低于 {CONF_REVIEW}，不可用",
                           person_id=pid, iou=best_iou)
        return Verdict(idx, equip, "待复核", conf,
                       f"装备框与{region_name} IoU={best_iou:.2f} / 包含度 {best_contain:.2f} 成立，"
                       f"但置信度 {conf:.2f} 落在复核区间",
                       person_id=pid, iou=best_iou)

    if neg_ok and neg.conf >= CONF_FINAL:
        return Verdict(idx, equip, "未佩戴", neg.conf,
                       f"{region_name}内检出裸头框（IoU={neg_iou:.2f}，包含度 {neg_contain:.2f}，"
                       f"置信度 {neg.conf:.2f}），反向证据成立",
                       is_violation=True, person_id=pid, iou=neg_iou)

    # 未匹配到装备框 -> 未佩戴。此时可信度取决于「人」判定得多确定
    conf = person.conf
    if conf < CONF_REVIEW:
        return Verdict(idx, equip, "丢弃", conf,
                       f"人员框置信度 {conf:.2f} 过低，不作为依据",
                       person_id=pid, iou=best_iou)
    if conf < PERSON_CONF_FINAL:
        return Verdict(idx, equip, "待复核", conf,
                       f"{region_name}内未检出装备框，但人员框置信度 {conf:.2f} 落在复核区间",
                       person_id=pid, iou=best_iou)
    return Verdict(idx, equip, "未佩戴", conf,
                   f"{region_name}内未检出装备框（最高 IoU={best_iou:.2f} < {iou_min}，"
                   f"包含度 {best_contain:.2f} < {contain_min}），判定为未佩戴",
                   is_violation=True, person_id=pid, iou=best_iou)


def judge_frame(boxes: Sequence[Box],
                equips: Sequence[str] = ("helmet", "vest"),
                area_min: float | None = None) -> list[Verdict]:
    """对一帧内的所有人做指定装备的判定。area_min 不给则用 1080p 标定值。"""
    persons = [b for b in boxes if b.label == "person"]
    out: list[Verdict] = []
    for i, p in enumerate(persons):
        for e in equips:
            out.append(judge_equipment(p, i, boxes, e, area_min=area_min))
    return out


def aggregate_temporal(frames: Sequence[Sequence[Verdict]],
                       need: int = 3) -> list[Verdict]:
    """
    时序聚合：同一人员连续 need 帧结论一致才定论。
    不足 need 帧的标为「待复核」。用于压制单帧抖动。
    """
    if len(frames) < need:
        res = []
        for v in frames[-1]:
            if v.status in ("已佩戴", "未佩戴"):
                res.append(Verdict(v.person_idx, v.equip, "待复核", v.conf,
                                   f"仅 {len(frames)} 帧，不足 {need} 帧时序确认",
                                   person_id=v.person_id, iou=v.iou))
            else:
                res.append(v)
        return res

    window = frames[-need:]
    # 对齐前提：每帧的判定项数与顺序一致。视频链路是「一条轨迹一个窗口」，
    # 天然满足；多帧图片场景一旦有人进出/漏检，项数就对不上，按下标硬比会
    # 把甲的第 2 项和乙的第 2 项比出个「一致」来。宁可全部转人工，也不编。
    if len({len(w) for w in window}) > 1:
        return [Verdict(v.person_idx, v.equip, "待复核", v.conf,
                        f"帧间判定项数不一致（{[len(w) for w in window]}），无法确认是否同一人",
                        person_id=v.person_id, iou=v.iou) for v in window[-1]]

    final: list[Verdict] = []
    for k, last in enumerate(window[-1]):
        seq = [w[k].status for w in window if k < len(w)]
        if len(seq) == need and len(set(seq)) == 1:
            st = seq[0]
            if st in ("已佩戴", "未佩戴"):
                final.append(Verdict(last.person_idx, last.equip, st, last.conf,
                                     f"连续 {need} 帧一致：" + last.reason,
                                     is_violation=(st == "未佩戴"),
                                     person_id=last.person_id, iou=last.iou))
            else:
                # 无法判定 / 丢弃 属于失效状态，不得被改写为待复核
                final.append(Verdict(last.person_idx, last.equip, st, last.conf,
                                     last.reason,
                                     person_id=last.person_id, iou=last.iou))
        else:
            final.append(Verdict(last.person_idx, last.equip, "待复核", last.conf,
                                 f"连续 {need} 帧结论不一致：{seq}",
                                 person_id=last.person_id, iou=last.iou))
    return final


# ============================================================
# 四、风险矩阵

def likelihood(violation_count_30d: int) -> int:
    """可能性 L：由该区域近 30 天违规次数决定。"""
    for cap, val in L_TABLE:
        if violation_count_30d <= cap:
            return val
    return L_MAX


def severity(scene: str, risk_class: str = "") -> int:
    """后果严重度 S：由作业类型决定（docs/安巡风险分级标准.md §2.2）。

    `scene` 是区域登记的作业类型（普通 / 车辆 / 高处 / 动火 / 受限空间）。
    有些区域登记的是场所属性（如「消防通道」），这时用 `risk_class` 兜底，
    两者都没有就落到 S_DEFAULT=2，并在研判里注明「作业类型未登记」。
    """
    if scene in S_MAP:
        return S_MAP[scene]
    if risk_class in S_MAP:
        return S_MAP[risk_class]
    return S_DEFAULT


def grade_of(risk: int) -> tuple[str, str, timedelta]:
    """风险值 -> (等级, 处置要求, 整改时限)。"""
    for low, name, action, due in GRADE_RULES:
        if risk >= low:
            return name, action, due
    return "蓝色", "加强关注", timedelta(days=30)


# ============================================================
# 五、事件与工单

@dataclass
class Region:
    region_id: str
    name: str
    scene: str                 # 高处 / 起重 / 动火 / 车辆 / 普通 / 消防通道
    owner: str                 # 责任班组长
    violation_30d: int = 0
    risk_class: str = "普通"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    event_id: str
    region: Region
    equip: str
    ts: datetime
    conf: float
    evidence: str
    L: int = 0
    S: int = 0
    R: int = 0
    grade: str = ""
    action: str = ""
    due_at: datetime | None = None


def build_event(region: Region, equip: str, ts: datetime,
                conf: float, evidence: str, event_id: str = "") -> Event:
    """把一条违规判定转成带定级的事件。"""
    L = likelihood(region.violation_30d)
    S = severity(region.scene, getattr(region, "risk_class", ""))
    R = L * S
    g, act, due_delta = grade_of(R)
    return Event(
        event_id=event_id or f"EV-{ts:%Y%m%d%H%M%S}-{region.region_id}",
        region=region, equip=equip, ts=ts, conf=conf, evidence=evidence,
        L=L, S=S, R=R, grade=g, action=act, due_at=ts + due_delta,
    )


def make_ticket(ev: Event, affected: int = 1) -> dict:
    """生成可执行、可追踪的整改工单。"""
    cn, tpl = VIOLATION_INFO.get(ev.equip, ("未知违规", "整改{region}"))
    requirement = tpl.format(n=affected, region=ev.region.name)
    escalate_1 = ev.ts + (ev.due_at - ev.ts) * 0.5
    return {
        "ticket_id": f"TK-{ev.event_id.split('-', 1)[1]}",
        "event_id": ev.event_id,
        "violation": cn,
        "equip": ev.equip,
        "region": ev.region.name,
        "region_id": ev.region.region_id,
        "grade": ev.grade,
        "grade_hex": GRADE_HEX.get(ev.grade, "#888"),
        "risk": ev.R,
        "L": ev.L,
        "S": ev.S,
        "scene": ev.region.scene,
        "requirement": requirement,
        "assignee": ev.region.owner,
        "cc": "部门安全负责人" if ev.grade == "红色" else "",
        "created_at": ev.ts.isoformat(timespec="minutes"),
        "due_at": ev.due_at.isoformat(timespec="minutes"),
        "review_at": (ev.due_at + timedelta(days=1)).isoformat(timespec="minutes"),
        "escalate_at_50pct": escalate_1.isoformat(timespec="minutes"),
        "status": "派单",
        "evidence": ev.evidence,
        "legal": LEGAL_REFS.get(ev.equip, []),
    }


def to_json(obj) -> str:
    import json
    def conv(o):
        if isinstance(o, datetime):
            return o.isoformat(timespec="minutes")
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(str(type(o)))
    return json.dumps(obj, ensure_ascii=False, indent=2, default=conv)
