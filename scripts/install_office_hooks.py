#!/usr/bin/env python3
"""~/.claude/settings.json에 오피스 훅을 등록한다 (이미 있으면 덮어쓰지 않고 추가).

사용: python3 scripts/install_office_hooks.py
"""

import json
from pathlib import Path

SETTINGS = Path("~/.claude/settings.json").expanduser()
HOOK = Path(__file__).resolve().parent / "office_hook.py"
EVENTS = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Notification",
    "Stop",
    "SubagentStart",
    "SubagentStop",
]


def main() -> None:
    settings = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    cmd = {"type": "command", "command": f"python3 {HOOK}", "timeout": 5}
    hooks = settings.setdefault("hooks", {})
    for ev in EVENTS:
        entries = hooks.setdefault(ev, [])
        if any(h.get("command") == cmd["command"] for e in entries for h in e.get("hooks", [])):
            continue
        entry = {"hooks": [cmd]}
        if ev in ("PreToolUse", "PostToolUse"):
            entry["matcher"] = "*"
        entries.append(entry)
    SETTINGS.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    print(f"registered {len(EVENTS)} hook events in {SETTINGS}")


if __name__ == "__main__":
    main()
