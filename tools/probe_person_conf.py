# -*- coding: utf-8 -*-
"""
人体检测召回探针：e2e 漏报的 12 例能不能救回来
==============================================

v3.0.0 的 headline 是 matched 口径（16.13% 漏报），但同一条链路的 e2e 口径是
**25.71%**——差出来的 12 例违规人次，是「人压根没被检出」，不是判定错。

这一层之前没动过：`DualDetector` 的 conf 是 person 与 ppe 共用，调它等于同时改
两个模型，所以扫参时从没单独试过「只降人体检测的 conf」。

本脚本单独改 `det.person.conf`，看 e2e 与 matched 各怎么变。
判据：降 person conf 若能显著降 e2e 漏报、而误报不失控，就是一个真实的方向；
若只是把一堆低质人框放进来、e2e 变差，则否掉。

    python tools/probe_person_conf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anxun.detectors import DualDetector, pick_device
from anxun.evaluate_real import run_eval, warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ppe_test" / "test"
OUT = ROOT / "data" / "person_conf_probe.json"


def main() -> int:
    rows = []
    for pc in (0.25, 0.20, 0.15, 0.10):
        det = DualDetector(str(ROOT / "models" / "yolov8n.pt"),
                           str(ROOT / "models" / "best.pt"),
                           imgsz=640, conf=0.25, device=pick_device(""))
        # 只改人体那一路；ppe 的 conf 保持 0.25 不动
        det.person.conf = pc
        warmup(det)
        out = run_eval(DATA, det, verbose=False)
        r = out["results"]
        m, e = r["scope_detail"]["matched"], r["scope_detail"]["e2e"]
        rows.append({
            "person_conf": pc,
            "persons_pred": out["persons_pred"],
            "coverage": r["coverage"]["rate"],
            "matched": {"fp": m["fp_rate"], "fn": m["fn_rate"],
                        "n_c": m["n_compliant"], "n_v": m["n_violating"]},
            "e2e": {"fp": e["fp_rate"], "fn": e["fn_rate"],
                    "n_c": e["n_compliant"], "n_v": e["n_violating"]},
            "mAP": r["mAP50"],
        })
        print(f"person conf {pc:.2f}: 检出 {out['persons_pred']} 人  "
              f"覆盖 {r['coverage']['rate']}%  "
              f"matched {m['fp_rate']}/{m['fn_rate']}  "
              f"e2e {e['fp_rate']}/{e['fn_rate']}  mAP {r['mAP50']}")

    print()
    print(f"{'conf':>6}{'检出人数':>8}{'覆盖率':>8}{'matched误报':>12}{'matched漏报':>12}"
          f"{'e2e误报':>10}{'e2e漏报':>9}")
    for r in rows:
        print(f"{r['person_conf']:>6.2f}{r['persons_pred']:>8}{r['coverage']:>7.1f}%"
              f"{r['matched']['fp']:>11.2f}%{r['matched']['fn']:>11.2f}%"
              f"{r['e2e']['fp']:>9.2f}%{r['e2e']['fn']:>8.2f}%")

    OUT.write_text(json.dumps({"kind": "person-conf-probe", "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
