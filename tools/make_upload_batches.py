"""生成 push_files 的批量载荷：把待入库文件按体积切成若干批 JSON。

背景
----
GitHub 连接器是 App 安装令牌，不能创建仓库（403 Resource not accessible by
integration），需要用户手动建空仓库，再由本工具准备好的载荷分批推送。

为什么切批：push_files 的 content 是字符串，全部文件约 590KB，一次调用太大；
按 ~110KB 一批切分，单次调用可控，失败了也只需重推那一批。

排除规则（与 .gitignore 一致，此处再显式筛一遍，双保险）：
  models/  datasets/  runs/  __pycache__  *.pyc

用法
----
    python tools/make_upload_batches.py           # 生成 .upload/batch_*.json
    python tools/make_upload_batches.py --list    # 只看会推哪些文件
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".upload"
MAX_BATCH_BYTES = 110_000

# 显式排除（.gitignore 的文字说明，这里用代码兜底）
EXCLUDE_DIRS = {"models", "datasets", "runs", "__pycache__", ".upload", ".git"}
EXCLUDE_SUFFIX = {".pyc", ".pt", ".onnx", ".mp4", ".jpg", ".jpeg", ".png"}

# 已知的过期产物（上游脚本已删，无任何引用）
STALE_FILES = {"data/dataset_stats.json"}

# 根目录要带的文件
ROOT_FILES = ["README.md", ".gitignore", "requirements-cpu.txt",
              "requirements-gpu.txt", "install.bat", "install.sh",
              "run.bat", "run.sh"]


def collect() -> list[Path]:
    files: list[Path] = []
    for d in ("src", "tools", "docs", "data"):
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if p.suffix.lower() in EXCLUDE_SUFFIX:
                continue
            if rel.as_posix() in STALE_FILES:
                continue
            files.append(p)
    for name in ROOT_FILES:
        p = ROOT / name
        if p.exists():
            files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列文件不生成")
    args = ap.parse_args()

    files = collect()
    total = sum(f.stat().st_size for f in files)

    if args.list:
        for f in files:
            print(f"{f.stat().st_size:>8}  {f.relative_to(ROOT).as_posix()}")
        print(f"\n共 {len(files)} 个文件 / {total / 1024:.0f} KB")
        return 0

    if OUT.exists():
        for old in OUT.glob("batch_*.json"):
            old.unlink()
    else:
        OUT.mkdir(parents=True, exist_ok=True)

    batches: list[list[Path]] = []
    cur: list[Path] = []
    cur_size = 0
    for f in files:
        size = f.stat().st_size
        if cur and cur_size + size > MAX_BATCH_BYTES:
            batches.append(cur)
            cur, cur_size = [], 0
        cur.append(f)
        cur_size += size
    if cur:
        batches.append(cur)

    manifest = []
    for i, batch in enumerate(batches, 1):
        payload = {
            "files": [{"path": f.relative_to(ROOT).as_posix(),
                       "content": f.read_text(encoding="utf-8")}
                      for f in batch],
            "message": "",
        }
        size = sum(f.stat().st_size for f in batch)
        (OUT / f"batch_{i:02d}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest.append({"batch": i, "files": len(batch), "kb": round(size / 1024),
                         "paths": [f.relative_to(ROOT).as_posix() for f in batch]})
        print(f"batch_{i:02d}  {len(batch):>2} 个文件  {size / 1024:>6.0f} KB")

    (OUT / "manifest.json").write_text(
        json.dumps({"total_files": len(files), "total_kb": round(total / 1024),
                    "batches": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n共 {len(files)} 个文件 / {total / 1024:.0f} KB，分 {len(batches)} 批")
    print(f"载荷目录：{OUT}（推送完成后可删）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
