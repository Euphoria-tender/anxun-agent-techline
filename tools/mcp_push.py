"""通过本地 MCP 代理直接调用 GitHub 连接器，把文件推送到仓库。

为什么需要这个脚本
----------------
1. `push_files` 的 content 是字符串，57 万字节文本如果经由对话传递，会撑爆上下文；
2. 更严重的是：任何 LLM「读文件再复述进工具参数」的过程都是**重新生成文本**，
   子代理实测在这个环节引入过笔误。上传源码绝不能冒这个风险。

所以改成用脚本走本地 MCP 代理：内容由 Python 原样读取、原样发送，
不经过任何模型转写，也不占用对话上下文。

凭证来源：环境变量 CODEBUDDY_MCP_CONFIG 里 mcpServers.github 的
url / Authorization / X-WorkBuddy-MCP-Context，即当前会话正在使用的同一个连接器。

用法
----
    python tools/mcp_push.py --list-tools                 # 列出代理暴露的工具名
    python tools/mcp_push.py --call get_me                # 单次调用测试
    python tools/mcp_push.py --push .upload/batch_01.json # 推送一批
    python tools/mcp_push.py --push-all                   # 按顺序推完全部批次
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> tuple[str, dict]:
    raw = os.environ.get("CODEBUDDY_MCP_CONFIG", "")
    if not raw:
        raise SystemExit("环境变量里没有 CODEBUDDY_MCP_CONFIG")
    cfg = json.loads(raw)
    gh = cfg.get("mcpServers", {}).get("github")
    if not gh:
        raise SystemExit("MCP 配置里没有 github server")
    return gh["url"], gh.get("headers", {})


URL, HEADERS = _cfg()


class Mcp:
    """最小 MCP 客户端：initialize -> initialized -> tools/call。"""

    def __init__(self):
        self.session = ""
        self._id = 0
        self._initialize()

    def _post(self, payload: dict, notify: bool = False) -> tuple[dict, str]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(URL, data=body, method="POST")
        for k, v in HEADERS.items():
            req.add_header(k, v)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        if self.session:
            req.add_header("mcp-session-id", self.session)
        with urllib.request.urlopen(req, timeout=600) as r:
            sid = r.headers.get("mcp-session-id", "")
            raw = r.read().decode("utf-8", "ignore")
        if notify:
            return {}, sid
        # 流式返回是 SSE：取 data: 开头的行；普通返回直接是 JSON
        text = raw
        for line in raw.splitlines():
            if line.startswith("data:"):
                text = line[5:].strip()
                break
        try:
            return json.loads(text), sid
        except json.JSONDecodeError:
            return {"_raw": raw[:2000]}, sid

    def _initialize(self):
        _, sid = self._post({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05",
                       "capabilities": {},
                       "clientInfo": {"name": "anxun-upload", "version": "1.0"}},
        })
        self.session = sid
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"},
                   notify=True)

    def call(self, name: str, arguments: dict) -> dict:
        self._id += 1
        out, _ = self._post({"jsonrpc": "2.0", "id": self._id,
                             "method": "tools/call",
                             "params": {"name": name, "arguments": arguments}})
        return out

    def list_tools(self) -> list[str]:
        self._id += 1
        out, _ = self._post({"jsonrpc": "2.0", "id": self._id,
                             "method": "tools/list", "params": {}})
        tools = out.get("result", {}).get("tools", [])
        return [t.get("name", "") for t in tools]


def _show(res: dict) -> bool:
    """打印 MCP 返回，判断是否成功。"""
    if "error" in res:
        print(f"  JSON-RPC 错误：{json.dumps(res['error'], ensure_ascii=False)[:500]}")
        return False
    result = res.get("result", {})
    if result.get("isError"):
        content = result.get("content", [])
        msg = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        print(f"  工具报错：{msg[:600]}")
        return False
    content = result.get("content", [])
    for c in content:
        if isinstance(c, dict) and c.get("text"):
            print(f"  {str(c['text'])[:600]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--call", default="", help="工具名，配合 --args")
    ap.add_argument("--args", default="{}", help="JSON 参数")
    ap.add_argument("--push", default="", help="批次 json 路径")
    ap.add_argument("--push-all", action="store_true")
    ap.add_argument("--owner", default="Euphoria-tender")
    ap.add_argument("--repo", default="anxun-agent-techline-v1.0.0")
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    m = Mcp()

    if args.list_tools:
        names = m.list_tools()
        print(f"代理暴露 {len(names)} 个工具，含 github 字样的：")
        for n in names:
            if "git" in n.lower() or "push" in n.lower() or "repo" in n.lower():
                print("  ", n)
        return 0

    if args.call:
        ok = _show(m.call(args.call, json.loads(args.args)))
        return 0 if ok else 1

    def push_one(path: Path) -> bool:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload["files"]
        msg = payload.get("message") or f"chore: {path.name}"
        print(f"推送 {path.name}：{len(files)} 个文件 …")
        ok = _show(m.call("push_files", {
            "owner": args.owner, "repo": args.repo, "branch": args.branch,
            "files": files, "message": msg,
        }))
        print("  ✅ 成功" if ok else "  ❌ 失败")
        return ok

    if args.push:
        return 0 if push_one(Path(args.push)) else 1

    if args.push_all:
        batches = sorted((ROOT / ".upload").glob("batch_*.json"))
        if not batches:
            print("没有批次文件，先跑 tools/make_upload_batches.py")
            return 2
        done, failed = 0, []
        for i, b in enumerate(batches, 1):
            print(f"[{i}/{len(batches)}] ", end="")
            if push_one(b):
                done += 1
            else:
                failed.append(b.name)
                print("  中止：后续批次依赖前一批的分支状态")
                break
        print(f"\n完成 {done}/{len(batches)} 批" + (f"，失败：{failed}" if failed else ""))
        return 0 if not failed else 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
