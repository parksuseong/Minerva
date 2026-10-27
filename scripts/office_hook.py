#!/usr/bin/env python3
"""Claude Code 훅 → 오피스 이벤트 로그(JSONL).

훅 입력(stdin JSON)에서 화면에 필요한 최소 정보만 뽑아 한 줄씩 추가한다.
프롬프트 본문, 명령어 원문, 파일 내용, 메시지 본문은 기록하지 않는다.
어떤 경우에도 0으로 종료해 Claude 작업을 막지 않는다.
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

LOG = Path(os.environ.get("MINERVA_OFFICE_LOG", "~/.minerva/office/events.jsonl")).expanduser()
MAX_BYTES = 2_000_000
LABEL_MAX = 80


def label_for(tool: str, args: dict) -> str:
    if tool == "Bash":
        return args.get("description") or "명령 실행"
    if tool in ("Read", "Edit", "Write", "NotebookEdit"):
        return Path(args.get("file_path") or args.get("notebook_path") or "").name or tool
    if tool in ("Agent", "Task"):
        return args.get("description") or args.get("subagent_type") or "에이전트 호출"
    if tool == "Skill":
        return f"/{args.get('skill', '')}"
    if tool == "WebFetch":
        return urlparse(args.get("url", "")).netloc or "웹 읽기"
    if tool == "WebSearch":
        return "웹 검색"
    if tool in ("Grep", "Glob"):
        return "파일 찾기"
    if tool.endswith("telegram__reply"):
        return "텔레그램 답장"
    if tool == "Artifact":
        return "페이지 게시"
    return ""


def build(payload: dict) -> dict:
    event = payload.get("hook_event_name", "")
    tool = payload.get("tool_name", "")
    args = payload.get("tool_input") or {}
    rec = {
        "ts": time.time(),
        "event": event,
        "session": (payload.get("session_id") or "")[:8],
        "agent_id": payload.get("agent_id") or "",
        "agent_type": payload.get("agent_type") or "",
    }
    if tool:
        rec["tool"] = tool.split("__")[-1] if tool.startswith("mcp__") else tool
        rec["label"] = label_for(tool, args)[:LABEL_MAX]
        if tool in ("Agent", "Task"):
            rec["spawn_type"] = args.get("subagent_type") or "general-purpose"
    if event == "PostToolUse":
        resp = payload.get("tool_response")
        rec["ok"] = not (isinstance(resp, dict) and resp.get("is_error"))
    if event == "PostToolUseFailure":
        rec["ok"] = False
    return rec


def append(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists() and LOG.stat().st_size > MAX_BYTES:
        lines = LOG.read_text().splitlines()[-2000:]
        LOG.write_text("\n".join(lines) + "\n")
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    try:
        append(build(json.load(sys.stdin)))
    except Exception:  # noqa: BLE001 — 훅 실패가 작업을 막으면 안 된다
        pass


if __name__ == "__main__":
    main()
