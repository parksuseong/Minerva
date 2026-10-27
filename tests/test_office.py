import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minerva.main import app

HOOK = Path(__file__).parent.parent / "scripts" / "office_hook.py"


@pytest.fixture
def office_env(tmp_path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MINERVA_OFFICE_LOG", str(log))
    monkeypatch.delenv("MINERVA_OFFICE_TOKEN", raising=False)
    return log


def run_hook(payload, log):
    subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        text=True,
        check=True,
        env={"MINERVA_OFFICE_LOG": str(log), "PATH": "/usr/bin:/bin"},
    )


def test_hook_records_only_safe_fields(office_env):
    run_hook(
        {
            "hook_event_name": "PreToolUse",
            "session_id": "abcdef1234567890",
            "tool_name": "Bash",
            "tool_input": {"command": "cat ~/.ssh/id_rsa", "description": "키 확인"},
        },
        office_env,
    )
    run_hook({"hook_event_name": "UserPromptSubmit", "prompt": "비밀 지시"}, office_env)
    text = office_env.read_text()
    assert "id_rsa" not in text and "비밀 지시" not in text
    first = json.loads(text.splitlines()[0])
    assert first["label"] == "키 확인" and first["session"] == "abcdef12"


def test_hook_never_fails_on_garbage(office_env):
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json",
        text=True,
        env={"MINERVA_OFFICE_LOG": str(office_env)},
    )
    assert proc.returncode == 0


def test_hook_labels_agent_spawn_and_file_basename(office_env):
    sys.path.insert(0, str(HOOK.parent))
    import office_hook

    rec = office_hook.build(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Agent",
            "tool_input": {"subagent_type": "backtester", "description": "SOXX 검증"},
        }
    )
    assert rec["spawn_type"] == "backtester" and rec["label"] == "SOXX 검증"
    rec = office_hook.build(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/home/x/secret/dir/notes.md"},
        }
    )
    assert rec["label"] == "notes.md"


def test_events_endpoint_pages_by_line(office_env):
    office_env.write_text(
        "\n".join(json.dumps({"event": "Stop", "ts": i}) for i in range(3)) + "\n"
    )
    client = TestClient(app)
    data = client.get("/api/office/events").json()
    assert data["last"] == 3 and [e["seq"] for e in data["events"]] == [1, 2, 3]
    assert client.get("/api/office/events?after=2").json()["events"][0]["seq"] == 3
    assert client.get("/api/office/events?after=99").json()["last"] == 3


def test_events_endpoint_requires_token_when_configured(office_env, monkeypatch):
    monkeypatch.setenv("MINERVA_OFFICE_TOKEN", "s3cret")
    client = TestClient(app)
    assert client.get("/api/office/events").status_code == 403
    assert client.get("/api/office/events?token=s3cret").status_code == 200


def test_office_page_served():
    r = TestClient(app).get("/office")
    assert r.status_code == 200 and "MINERVA OFFICE" in r.text
