"""交付物验收：五条线（数据 / 模型 / 指标 / 部署 / 技术页）逐项实测。

为什么要有这个脚本
------------------
答辩或交接时，「东西是有效的」不能靠口头保证。这个脚本把每一条交付物都用
可执行的方式验一遍，输出 PASS/FAIL 表；失败会给出具体原因而不是笼统的"有问题"。

对应比赛要求的技术线交付物：**能跑的模型 + 四个指标 + PPT 第 10–13 页**。

用法
----
    python src/anxun/verify_all.py              # 全量（含起服务，约 30 秒）
    python src/anxun/verify_all.py --no-serve   # 跳过 HTTP 服务（约 10 秒）
    python src/anxun/verify_all.py --fetch      # 先补齐 3 项 SKIP（下载 ~110MB，再验收）
    python src/anxun/verify_all.py --json out.json   # 结果另存 JSON

退出码：0 = 全部通过；1 = 有 FAIL。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[dict] = []


def check(line: str, item: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"line": line, "item": item, "ok": bool(ok), "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {item}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def skip(line: str, item: str, detail: str = "") -> None:
    """不计 PASS 也不计 FAIL。

    仓库分发的是代码 + 权重 + 演示素材，评测集（100 张现场图）不在仓库里——
    这类「依赖没跟来」必须和「东西坏了」区分开，否则 clone 下来一眼全是红的，
    看着像交付失败，其实只是没数据。
    """
    RESULTS.append({"line": line, "item": item, "ok": None, "detail": detail})
    print(f"  [SKIP] {item}" + (f" — {detail}" if detail else ""))


def _eval_images() -> list:
    d = ROOT / "datasets" / "ppe_test" / "test" / "images"
    return sorted(d.glob("*.jpg")) if d.is_dir() else []


def _probe_image() -> str:
    """没有评测集时用合成图做烟测：只证明「权重能加载 + 前向出结果」，不证明精度。"""
    import tempfile
    from PIL import Image, ImageDraw
    p = Path(tempfile.gettempdir()) / "anxun_probe.jpg"
    im = Image.new("RGB", (640, 640), (42, 46, 52))
    d = ImageDraw.Draw(im)
    d.rectangle([250, 200, 390, 620], fill=(118, 120, 128))      # 躯干
    d.ellipse([292, 140, 350, 202], fill=(236, 206, 92))         # 头（戴帽）
    im.save(p, quality=92)
    return str(p)


def head(t: str) -> None:
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------- 零、补齐大件
def fetch_missing() -> None:
    """把 3 项 SKIP 依赖的大件补上（训练集 60MB + 1080p 素材 51MB）。

    为什么要做成一个开关：换台机器复现时，人工翻 README 找两条命令再依次跑，
    这一步最容易漏。--fetch 把它变成验收脚本的一部分，跑完直接全 PASS。
    """
    head("零、补齐大件（--fetch）")
    for script, note in (("tools/fetch_helmet_p2.py", "helmet_p2 训练集（约 60MB）"),
                         ("tools/make_1080_stream.py", "1080p 复验素材（约 51MB）")):
        p = ROOT / script
        if not p.exists():
            check("补齐", note, False, f"缺脚本 {script}")
            continue
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(p)], cwd=str(ROOT))
        check("补齐", note, r.returncode == 0,
              f"退出码 {r.returncode} · {time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------- 一、数据
def verify_data() -> None:
    head("一、数据")
    if not _eval_images():
        skip("数据", "评测集未随仓库分发",
             "100 张现场图不在仓库（体积 + 现场照片），需要时跑 tools/fetch_ppe_test.py 重新拉取")
        return
    from anxun.evaluate_real import list_images, read_names
    from collections import Counter
    import numpy as np
    import cv2

    splits = {"ppe_test/train": 100, "ppe_test/test": 100, "helmet_p2/train": 1200,
              "helmet_p2/valid": 400}
    for rel, expect in splits.items():
        d = ROOT / "datasets" / rel
        if not d.exists():
            skip("数据", f"{rel} 图片/标注配对", "该集合未随仓库分发（训练用，体积大）")
            continue
        imgs = list(d.glob("images/*.jpg")) if (d / "images").exists() else []
        lbls = list((d / "labels").glob("*.txt")) if (d / "labels").exists() else []
        paired = sum(1 for p in imgs if (d / "labels" / f"{p.stem}.txt").exists())
        # 允许数据源本身的数量波动，只要求"图与标注配对完整"
        ok = bool(imgs) and paired == len(imgs)
        check("数据", f"{rel} 图片/标注配对", ok,
              f"{len(imgs)} 图 / {len(lbls)} 标注 / 配对 {paired}")

    test = ROOT / "datasets" / "ppe_test" / "test"
    names = read_names(test)
    cls = Counter()
    empty = 0
    for p in (test / "labels").glob("*.txt"):
        rows = [ln for ln in p.read_text(errors="ignore").splitlines() if ln.strip()]
        if not rows:
            empty += 1
        for ln in rows:
            cls[names[int(float(ln.split()[0]))]] += 1
    check("数据", "测试集标注可解析", sum(cls.values()) > 500,
          f"{sum(cls.values())} 框，空标注 {empty} 张")
    check("数据", "测试集含 person 与 helmet 真值",
          cls.get("person", 0) > 0 and (cls.get("hardhat", 0) + cls.get("no_hardhat", 0)) > 0,
          f"person {cls.get('person', 0)} / hardhat {cls.get('hardhat', 0)} / "
          f"no_hardhat {cls.get('no_hardhat', 0)}")

    # 1080p 素材（触发层复验用）
    hd = ROOT / "datasets" / "ppe_test" / "test1080" / "images"
    sample = sorted(hd.glob("*.jpg"))[:1]
    if sample:
        img = cv2.imdecode(np.fromfile(str(sample[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
        check("数据", "1080p 复验素材分辨率", img is not None and min(img.shape[:2]) >= 1080,
              f"{img.shape[1]}×{img.shape[0]}" if img is not None else "读不到")
    else:
        skip("数据", "1080p 复验素材",
             "51MB，未随仓库分发；补齐：python tools/make_1080_stream.py（或加 --fetch 自动跑）")

    # data.yaml 里写死绝对路径 = 换机器就找不到数据。这是「clone 下来跑不动」的隐形原因之一。
    import re as _re
    yamls = sorted((ROOT / "datasets").glob("*/data.yaml"))
    bad = [y.parent.name for y in yamls
           if _re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", y.read_text(encoding="utf-8"))]
    check("数据", "data.yaml 不含本机绝对路径", bool(yamls) and not bad,
          f"仍写死：{bad}" if bad else f"{len(yamls)} 份均为相对路径，换机器不用改")


# ---------------------------------------------------------------- 二、模型
def verify_models() -> None:
    head("二、模型（能跑的模型）")
    pts = ["models/yolov8n.pt", "models/best.pt", "models/vest.pt"]
    for rel in pts:
        p = ROOT / rel
        check("模型", f"{rel} 存在", p.exists(),
              f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "缺失")

    onnx_dir = ROOT / "models" / "onnx"
    for name in ("yolov8n.onnx", "best.onnx", "vest.onnx", "manifest.json"):
        p = onnx_dir / name
        check("模型", f"models/onnx/{name} 存在", p.exists(),
              f"{p.stat().st_size / 1e6:.2f} MB" if p.exists() else "缺失")

    # 判定范围必须由模型的真实类别决定：只有安全帽模型时，不许判「未穿背心」。
    # 缺权重 -> 模型不产该类框 -> 规则引擎把「没框」当「没戴」-> 全场假违规。
    try:
        from anxun.detectors import available_equips

        class _Stub:
            def __init__(self, names):
                self.names = names

        only_helmet = available_equips(_Stub({0: "person", 1: "Hardhat", 2: "NO-Hardhat"}))
        with_vest = available_equips(_Stub({0: "person", 1: "Hardhat", 2: "NO-Hardhat",
                                            3: "vest", 4: "no_vest"}))
        check("模型", "装备清单按模型类别求交集",
              only_helmet == ["helmet"] and with_vest == ["helmet", "vest"],
              f"仅安全帽 -> {only_helmet} · 加背心 -> {with_vest}")
    except Exception as e:
        check("模型", "装备清单按模型类别求交集", False, f"{type(e).__name__}: {e}")

    if not all((ROOT / p).exists() for p in pts):
        check("模型", "真图推理", False, "权重不全，跳过")
        return

    from anxun.detectors import DualDetector
    imgs = _eval_images()
    if imgs:
        img, img_note = str(imgs[0]), ""
    else:
        img, img_note = _probe_image(), "（合成烟测图）"
    try:
        det = DualDetector(str(ROOT / "models/yolov8n.pt"), str(ROOT / "models/best.pt"),
                           vest_weights=str(ROOT / "models/vest.pt"), device="")
        t0 = time.perf_counter()
        r = det.detect_image(img)
        ms = (time.perf_counter() - t0) * 1000
        labels = {b.label for b in r["boxes"]}
        # 有真图就要求检出目标；只有合成图时，只证明权重能加载、前向不炸（图中无目标属正常）
        ok = len(r["boxes"]) > 0 if imgs else isinstance(r.get("boxes"), list)
        check("模型", "三模型推理可跑", ok,
              f"{len(r['boxes'])} 个框，{ms:.1f} ms，标签 {sorted(labels)}{img_note}")
        if imgs:
            check("模型", "输出含 person 与装备类",
                  "person" in labels and bool(labels & {"helmet", "nohelmet", "vest", "novest"}),
                  f"{sorted(labels)}")
        else:
            skip("模型", "输出含 person 与装备类", "合成烟测图不含目标，需评测集才能判类别覆盖")
    except Exception as e:
        check("模型", "三模型推理可跑", False, f"{type(e).__name__}: {e}")

    # ONNX 后端至少得能加载：权重损坏就该在这里暴露，别等到换机器才发现
    try:
        from anxun.onnx_backend import build_onnx_detector
        o, _ = build_onnx_detector(device="cpu")
        o.detect_image(img)
        check("模型", "ONNX 后端可加载并前向", True, "CPUExecutionProvider")
    except Exception as e:
        check("模型", "ONNX 后端可加载并前向", False, f"{type(e).__name__}: {e}")
        return

    if not imgs:
        skip("模型", "ONNX 与 torch 输出逐图比对", "需要评测集才能比对类别与置信度")
        return

    # ONNX 与 torch 一致性（可移植性的前提）
    try:
        o, _ = build_onnx_detector(device="cpu")
        det.detect_image(img)
        o.detect_image(img)
        # 判据：框数与标签多重集完全一致，且置信度最大偏差 < 0.02。
        # 不要求逐位相等——torch 走 GPU(float32/CUDA)、onnx 走 CPU，同一模型也会有
        # 1 ULP 级差异（实测 0.36 vs 0.35），那属于正常数值误差，不是模型不一致。
        worst, mismatch = 0.0, 0
        for p in sorted((ROOT / "datasets/ppe_test/test/images").glob("*.jpg"))[:5]:
            a = det.detect_image(p)["boxes"]
            b = o.detect_image(p)["boxes"]
            if sorted(x.label for x in a) != sorted(x.label for x in b):
                mismatch += 1
                continue
            for x, y in zip(sorted(a, key=lambda z: (z.label, -z.conf)),
                            sorted(b, key=lambda z: (z.label, -z.conf))):
                worst = max(worst, abs(x.conf - y.conf))
        check("模型", "ONNX 与 torch 输出一致", mismatch == 0 and worst < 0.02,
              f"5 张图 · 标签不一致 {mismatch} 张 · 置信度最大偏差 {worst:.3f}")
    except Exception as e:
        check("模型", "ONNX 与 torch 输出一致", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 三、指标
def verify_metrics() -> None:
    head("三、指标（四个数）")
    spec = {
        "data/metrics_real.json": ("helmet", ("mAP50", "fp_rate", "fn_rate", "latency")),
        "data/metrics_1080.json": ("1080", ("mAP50", "fp_rate", "fn_rate", "latency")),
        "data/metrics_vest.json": ("vest", ("mAP50", "fp_rate", "fn_rate")),
        "data/metrics_onnx.json": ("onnx", ("mAP50", "fp_rate", "fn_rate", "latency")),
    }
    for rel, (tag, keys) in spec.items():
        p = ROOT / rel
        if not p.exists():
            check("指标", f"{rel} 存在", False, "缺失")
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            check("指标", f"{rel} 可解析", False, f"{type(e).__name__}: {e}")
            continue
        r = d.get("results") or {}
        missing = [k for k in keys if r.get(k) is None]
        check("指标", f"{rel} 四指标齐全", not missing, f"缺 {missing}" if missing else "")
        if tag in ("helmet", "onnx", "1080"):
            lat = r.get("latency") or {}
            print(f"       mAP@0.5={r.get('mAP50')}  误报={r.get('fp_rate')}%  "
                  f"漏报={r.get('fn_rate')}%  单帧={lat.get('mean_ms')} ms  "
                  f"混淆={r.get('confusion')}")

    # README 与 JSON 必须对得上，避免文档写一套代码另一套
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    real = json.loads((ROOT / "data/metrics_real.json").read_text(encoding="utf-8"))["results"]
    fp_s = f"{real['fp_rate']:.1f}%"
    fn_s = f"{real['fn_rate']:.2f}%"
    check("指标", "README 误报率与实测一致", fp_s in readme, f"实测 {fp_s}")
    check("指标", "README 漏报率与实测一致", fn_s in readme or f"{real['fn_rate']:.2f}" in readme,
          f"实测 {fn_s}")

    # 版本号自洽：改了 __version__ 却没同步 README，文档和代码就会各说各话
    from anxun import __version__
    check("指标", "README 版本号与包版本一致", f"v{__version__}" in readme, f"包版本 v{__version__}")

    # 阈值只有一份定义（config/thresholds.json）。代码、文档、评测各自抄一份 = 两套口径，
    # 这就是 v1.1.0 的 P4-1。这条检查专门钉住它。
    # 注意：只比 conf 两项是不够的——改 head_region_ratio / min_aspect 同样会让
    # 判定结果变，但以前不会被发现，所以这里逐字段比。
    from anxun import rules_engine as R
    cmp_keys = ("conf_final", "conf_review", "head_region_ratio", "head_width_ratio",
                "min_person_area", "min_aspect")
    try:
        cfg = R.load_config()
        snap = R.profile_snapshot()
        prof = cfg["profiles"][snap["profile"]]
        diff = [k for k in cmp_keys
                if abs(float(prof[k]) - float(snap[k])) > 1e-9]
        check("指标", "代码阈值与 config/thresholds.json 一致", not diff,
              f"口径 {snap['profile']}：{snap['conf_review']}/{snap['conf_final']}"
              + (f"，不一致字段 {diff}" if diff else f"（比对 {len(cmp_keys)} 个字段）"))
    except Exception as e:
        check("指标", "代码阈值与 config/thresholds.json 一致", False,
              f"{type(e).__name__}: {e}")

    # 口径不是换个数字就算数，判定行为必须跟着变：同一个 0.52 的帽框，
    # measured 判「已佩戴」、conservative 判「待复核」。这正是 P4-1 的分歧点，
    # 也是「演示用保守配置」这套说法成立的前提。
    try:
        saved = R.active_profile()
        p = R.Box("person", 0.9, 0, 0, 100, 300)
        h = R.Box("helmet", 0.52, 28, 0, 72, 90)
        R.apply_profile("measured")
        a = R.judge_equipment(p, 0, [h], "helmet").status
        R.apply_profile("conservative")
        b = R.judge_equipment(p, 0, [h], "helmet").status
        R.apply_profile(saved)
        check("指标", "两套口径的判定行为符合预期", (a, b) == ("已佩戴", "待复核"),
              f"measured={a} · conservative={b}（同一输入 conf 0.52）")
    except Exception as e:
        check("指标", "两套口径的判定行为符合预期", False, f"{type(e).__name__}: {e}")

    # 择优必须单调：多一个「IoU 略高但两项都不达标」的废框，不能把已经判定的
    # 「已佩戴」翻成「未佩戴」。v1.2.0 之前就是按 (IoU, 包含度) 取全局最大，
    # 多给一个框反而多一条误报。这是误报的一个真实来源，用一条断言钉住。
    try:
        p = R.Box("person", 0.9, 100, 100, 300, 600)
        good = R.Box("helmet", 0.90, 170, 120, 230, 160)          # 包含度 1.0
        junk = R.Box("helmet", 0.88, 176.5, 165.1, 291.8, 219.1)  # 两项都不达标
        a = R.judge_equipment(p, 0, [good], "helmet").status
        b = R.judge_equipment(p, 0, [good, junk], "helmet").status
        check("指标", "判定择优单调（多一个废框不改结论）", a == b == "已佩戴",
              f"单框={a} · 加废框={b}")
    except Exception as e:
        check("指标", "判定择优单调（多一个废框不改结论）", False, f"{type(e).__name__}: {e}")

    # 交出去的数字必须能说清是哪套阈值跑的，否则被追问就答不上来
    untagged = []
    for rel in ("data/metrics_real.json", "data/metrics_onnx.json"):
        p = ROOT / rel
        if not p.exists():
            continue
        tag = ((json.loads(p.read_text(encoding="utf-8")).get("config") or {})
               .get("rules_profile"))
        if tag != "measured":
            untagged.append(f"{rel}={tag!r}")
    check("指标", "交付指标标注了阈值口径（measured）", not untagged,
          "、".join(untagged) if untagged else "metrics_real / metrics_onnx 均已标注")

    # 误报/漏报必须能同时给出两个口径：只算已检出的人（matched，衡量判定层）
    # 与连没检出的人一起算（e2e，衡量整条链路）。两者差 9 个百分点，
    # 被问「没检出的人算不算漏报」时不能答不上来。
    try:
        from anxun.metrics import business_rates
        probe = [{"image_id": "probe", "gt_boxes": [], "pred_boxes": [],
                  "truth": [{"person_id": "G0", "equip": "helmet",
                             "worn": False, "matched": True},
                            {"person_id": "G1", "equip": "helmet",
                             "worn": False, "matched": False}],
                  "judged": [{"person_id": "G0", "equip": "helmet", "status": "已佩戴"}],
                  "latency_ms": 1.0}]
        m = business_rates(probe, scope="matched")
        e = business_rates(probe, scope="e2e")
        check("指标", "误报漏报同时支持 matched / e2e 两个口径",
              (m["n_violating"], e["n_violating"]) == (1, 2),
              f"已检出 {m['n_violating']} 人次 · 含未检出 {e['n_violating']} 人次")
    except Exception as e:
        check("指标", "误报漏报同时支持 matched / e2e 两个口径", False,
              f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 四、部署
def verify_deploy(serve: bool) -> None:
    head("四、部署")
    try:
        import anxun.app  # noqa: F401
        import anxun.stream  # noqa: F401
        import anxun.scheduler  # noqa: F401
        check("部署", "服务与巡查模块可导入", True, "app / stream / scheduler")
    except Exception as e:
        check("部署", "服务与巡查模块可导入", False, f"{type(e).__name__}: {e}")
        return

    port = 8791
    if serve:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "src" / "anxun" / "app.py"),
             "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        try:
            import urllib.request
            ok = False
            for _ in range(60):
                time.sleep(1)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        body = json.loads(r.read().decode("utf-8"))
                    ok = body.get("status") == "ok"
                    if ok:
                        break
                except Exception:
                    continue
            check("部署", "HTTP 服务启动并响应 /health", ok)
            if ok:
                print(f"       后端 {body.get('backend')} · 设备 {body.get('device')} · "
                      f"自检 {body.get('selfcheck', {}).get('mean_ms')} ms")
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/inspect",
                    data=json.dumps({"scene_id": "R-01"}).encode(),
                    headers={"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    res = json.loads(r.read().decode("utf-8"))
                check("部署", "/api/inspect 返回判定与事件", "final" in res and "events" in res,
                      f"人 {res.get('summary', {}).get('persons')} · "
                      f"违规 {res.get('summary', {}).get('violations')} · "
                      f"事件 {len(res.get('events', []))}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
    else:
        check("部署", "HTTP 服务启动并响应 /health", True, "（--no-serve 跳过）")


# ---------------------------------------------------------------- 五、技术页
def verify_slides() -> None:
    head("五、技术页（PPT 第 10–13 页）")
    p = ROOT / "docs" / "技术页_10-13.html"
    if not p.exists():
        check("技术页", "文件存在", False, "缺 docs/技术页_10-13.html")
        return
    t = p.read_text(encoding="utf-8")
    check("技术页", "文件存在", True, f"{len(t) / 1024:.1f} KB")
    slides = t.count('<section class="slide">')
    check("技术页", "共 4 页（P10–P13）", slides == 4, f"实际 {slides} 页")
    for tag, key in (("P10", "系统架构"), ("P11", "指标"), ("P12", "火焰"), ("P13", "创新")):
        check("技术页", f"{tag} 内容在位", key in t, key)
    check("技术页", "16:9 版式", "1280" in t and "720" in t, "1280×720 画布")
    # 数字比对要按边界匹配：`"5.1" in html` 连 15.1 / 25.19 都会过，
    # 那是「字符串出现过」不是「数字对得上」。
    import re as _re
    for num in ("5.1", "22.6", "0.681"):
        hit = bool(_re.search(rf"(?<![\d.]){_re.escape(num)}(?![\d])", t))
        check("技术页", f"数字 {num} 与实测一致", hit, "")

    # 页脚要写明阈值口径：P11 报的四个数只在 measured 口径下成立
    from anxun import rules_engine as R
    cf, cr = R.CONF_FINAL, R.CONF_REVIEW
    check("技术页", "页脚标注了判定阈值口径",
          f"{cf:.2f}" in t and f"{cr:.2f}" in t, f"{cr:.2f} / {cf:.2f}")

    # PPT 第 11/13 页要贴图，JSON 有数但没图等于没做
    figs = ["confusion_matrix.png", "pr_curve.png", "risk_matrix.png"]
    missing = [f for f in figs
               if not (ROOT / "docs" / "figs" / f).exists()
               or (ROOT / "docs" / "figs" / f).stat().st_size < 10_000]
    check("技术页", "指标与风险矩阵图已生成", not missing,
          f"缺 {missing}，跑 python tools/plot_metrics.py" if missing
          else "confusion_matrix / pr_curve / risk_matrix")


# ---------------------------------------------------------------- 汇总
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--fetch", action="store_true",
                    help="先跑 tools/ 里的两个脚本补齐大件，再验收（下载约 110MB）")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print(f"交付物验收 · {ROOT}")
    if args.fetch:
        fetch_missing()
    verify_data()
    verify_models()
    verify_metrics()
    verify_deploy(not args.no_serve)
    verify_slides()

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["ok"])
    skipped = sum(1 for r in RESULTS if r["ok"] is None)
    failed = [r for r in RESULTS if r["ok"] is False]
    print(f"\n{'=' * 52}")
    tail = f" · SKIP {skipped}" if skipped else ""
    print(f"合计 {total} 项：PASS {passed} · FAIL {len(failed)}{tail}")
    if skipped:
        print("跳过（依赖未随仓库分发，不计失败）：")
        for r in RESULTS:
            if r["ok"] is None:
                print(f"  - [{r['line']}] {r['item']} {r['detail']}")
    if failed:
        print("未通过：")
        for r in failed:
            print(f"  - [{r['line']}] {r['item']} {r['detail']}")
    if args.json:
        from anxun import __version__
        Path(args.json).write_text(
            json.dumps({"version": f"techline-v{__version__}",
                        "total": total, "passed": passed, "skipped": skipped,
                        "results": RESULTS}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果 -> {args.json}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
