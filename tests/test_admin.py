import json

import pytest
from fastapi.testclient import TestClient

from minerva import admin
from minerva import btc_trade as bt
from minerva.main import app

TOKEN = "test-admin-token"
H = {"X-Minerva-Token": TOKEN}


@pytest.fixture
def env(tmp_path, monkeypatch):
    eng = tmp_path / "btc_trade"
    eng.mkdir()
    monkeypatch.setattr(bt, "DIR", eng)
    monkeypatch.setenv("MINERVA_ADMIN_DIR", str(tmp_path / "admin"))
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN", TOKEN)
    sent = []
    monkeypatch.setattr(admin.notify, "send", lambda text: sent.append(text) or True)
    monkeypatch.setattr(admin, "_fetch_geo", lambda ips: {})
    admin.reset_memory()
    yield {"eng": eng, "admin": tmp_path / "admin", "sent": sent}
    admin.reset_memory()


def _access(env) -> list[dict]:
    path = env["admin"] / "access.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/admin/status"),
        ("get", "/api/admin/access"),
        ("post", "/api/admin/halt"),
        ("post", "/api/admin/resume"),
        ("post", "/api/admin/clear-no-entry"),
    ],
)
def test_admin_needs_token(env, method, path):
    client = TestClient(app)
    assert getattr(client, method)(path).status_code == 403
    r = getattr(client, method)(path, headers={"X-Minerva-Token": "wrong"})
    assert r.status_code == 403
    assert not (env["eng"] / "HALT").exists()


def test_admin_locked_when_no_token_configured(env, monkeypatch, tmp_path):
    # 계좌는 토큰이 없으면 공개지만 어드민은 열리면 안 된다
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN")
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "missing.token"))
    client = TestClient(app)
    assert client.get("/api/admin/status").status_code == 403
    assert client.post("/api/admin/halt", headers={"X-Minerva-Token": "x"}).status_code == 403
    assert not (env["eng"] / "HALT").exists()


def test_halt_resume_roundtrip(env):
    client = TestClient(app)
    r = client.post("/api/admin/halt", headers=H)
    assert r.status_code == 200 and r.json()["changed"]
    halt = env["eng"] / "HALT"
    assert halt.exists() and (halt.stat().st_mode & 0o777) == 0o600
    assert bt.halted()
    assert r.json()["engine"]["halt"] is True
    assert "HALT" in env["sent"][0] and "신규 진입만" in env["sent"][0]
    # 이미 정지면 변화 없음, 알림 없음
    assert client.post("/api/admin/halt", headers=H).json()["changed"] is False
    assert len(env["sent"]) == 1
    r = client.post("/api/admin/resume", headers=H)
    assert r.json()["changed"] and not halt.exists() and not bt.halted()
    assert "재개" in env["sent"][1]
    acts = [json.loads(x) for x in (env["admin"] / "actions.jsonl").read_text().splitlines()]
    assert [a["action"] for a in acts] == ["halt", "halt", "resume"]
    assert acts[0]["ip"] == "testclient" or acts[0]["ip"] == "-"
    assert (env["admin"].stat().st_mode & 0o777) == 0o700
    assert ((env["admin"] / "actions.jsonl").stat().st_mode & 0o777) == 0o600


def test_clear_no_entry_shows_and_records_reason(env):
    (env["eng"] / "NO_ENTRY").write_text("2026-11-23 01:00 최근 30일 승률 3/9\n")
    client = TestClient(app)
    st = client.get("/api/admin/status", headers=H).json()
    assert "승률" in st["engine"]["no_entry"]
    r = client.post("/api/admin/clear-no-entry", headers=H).json()
    assert r["changed"] and not (env["eng"] / "NO_ENTRY").exists()
    assert "승률" in env["sent"][0]
    acts = client.get("/api/admin/status", headers=H).json()["actions"]
    assert acts[0]["action"] == "clear_no_entry" and "승률" in acts[0]["detail"]


def test_status_shows_position_and_last_run(env):
    state = {
        "last_run_ms": 1_763_800_000_000,
        "position": {"symbol": "ETHUSDT", "side": "short", "entry": 3000.0, "stop": 3090.0},
    }
    (env["eng"] / "state.json").write_text(json.dumps(state))
    (env["eng"] / "mode").write_text("live\n")
    run = {"event": "run", "decision": "대기", "halts": ["HALT 파일"]}
    (env["eng"] / "events.jsonl").write_text(json.dumps(run, ensure_ascii=False) + "\n")
    e = TestClient(app).get("/api/admin/status", headers=H).json()["engine"]
    assert e["mode"] == "live"
    assert e["position"]["symbol"] == "ETHUSDT" and e["position"]["stop"] == 3090.0
    assert e["last_run_at"].startswith("2025-") or e["last_run_at"].startswith("2026-")
    assert e["last_decision"] == "대기" and e["last_halts"] == ["HALT 파일"]


def test_middleware_logs_pages_not_polls(env):
    client = TestClient(app)
    client.get("/", headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)"})
    client.get("/api/health")
    client.get("/api/daily")
    client.get("/static/common.js")
    rows = _access(env)
    assert [r["path"] for r in rows] == ["/"]
    assert rows[0]["kind"] == "page" and rows[0]["auth"] is None


def test_middleware_logs_token_attempts(env):
    client = TestClient(app)
    client.get("/api/admin/status", headers={"X-Minerva-Token": "bad"})
    client.get("/api/admin/status", headers=H)
    client.get("/api/admin/status", headers=H)  # 같은 기기의 성공 반복은 한 번만
    client.get("/api/admin/status", headers={**H, "X-Minerva-Login": "1"})  # 직접 입력은 늘
    rows = [r for r in _access(env) if r["kind"] == "auth"]
    assert [r["auth"] for r in rows] == ["fail", "ok", "ok"]
    assert "bad" not in (env["admin"] / "access.jsonl").read_text()


def test_brute_force_gets_blocked(env):
    client = TestClient(app)
    for i in range(admin.FAIL_LIMIT):
        client.get("/api/account", headers={"X-Minerva-Token": f"guess{i}"})
    r = client.get("/api/admin/status", headers=H)
    assert r.status_code == 429
    assert _access(env)[-1]["auth"] == "blocked"


def test_access_view_filters_and_escape_free(env):
    client = TestClient(app)
    ua = "<img src=x onerror=alert(1)> Mozilla/5.0 (Windows NT 10.0) Chrome/120 Safari/537"
    client.get("/", headers={"User-Agent": ua, "Referer": "javascript:alert(1)"})
    client.get("/api/admin/status", headers={"X-Minerva-Token": "bad"})
    d = client.get("/api/admin/access", headers=H).json()
    assert d["total"] == len(d["rows"]) >= 2
    assert d["rows"][0]["at"] >= d["rows"][-1]["at"]  # 최신이 위
    page = next(r for r in d["rows"] if r["kind"] == "page")
    assert page["ua"] == ua  # 서버는 그대로 넘기고 화면이 textContent로만 넣는다
    assert page["device"] == "Windows 10/11 · Chrome 120"
    only = client.get("/api/admin/access?auth_only=true", headers=H).json()
    assert {r["kind"] for r in only["rows"]} == {"auth"}
    hidden = client.get("/api/admin/access?hide_me=true", headers=H).json()
    assert hidden["total"] == 0  # 시험 클라이언트는 모두 같은 IP
    assert d["ips"][0]["count"] == d["total"]
    assert client.get("/api/admin/access?limit=5000", headers=H).status_code == 422


def test_static_admin_never_uses_innerhtml():
    from pathlib import Path

    js = (Path(admin.__file__).parent / "static" / "admin.js").read_text()
    assert "innerHTML" not in js.replace("innerHTML을", "")


@pytest.mark.parametrize(
    "ua,label,bot",
    [
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
            "iPhone · iOS 17.1 · Safari 17.1",
            False,
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
            "Windows 10/11 · Chrome 120",
            False,
        ),
        (
            "Mozilla/5.0 (Linux; Android 14; SM-S918N) AppleWebKit/537.36 (KHTML, like Gecko) "
            "SamsungBrowser/23.0 Chrome/115.0.0.0 Mobile Safari/537.36",
            "Android 14 · SM-S918N · Samsung Internet 23",
            False,
        ),
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Mobile/15E148 KAKAOTALK 10.4.5",
            "iPhone · iOS 17.1 · 카카오톡 10.4",
            False,
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like "
            "Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "macOS · Edge 120",
            False,
        ),
        ("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)", None, True),
        ("TelegramBot (like TwitterBot)", None, True),
        (
            "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/142.0.0.0 Mobile Safari/537.36",
            "Android 10 · Chrome 142",
            False,
        ),
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1",
            "iPhone · iOS 26.1 · Safari 26.1",
            False,
        ),
        ("curl/8.4.0", None, True),
        ("python-requests/2.31", None, True),
        ("", "알 수 없음(UA 없음)", True),
    ],
)
def test_parse_ua(ua, label, bot):
    got = admin.parse_ua(ua)
    assert got["bot"] is bot
    if label:
        assert got["label"] == label
    if bot and ua:
        assert got["label"].startswith("봇 · ")


def test_xff_only_trusted_from_loopback():
    assert admin.client_ip("203.0.113.5", "1.2.3.4") == "203.0.113.5"
    assert admin.client_ip("127.0.0.1", "1.2.3.4") == "1.2.3.4"
    # 왼쪽은 접속자가 꾸밀 수 있어 프록시가 붙인 맨 오른쪽
    assert admin.client_ip("127.0.0.1", "6.6.6.6, 198.51.100.7") == "198.51.100.7"
    assert admin.client_ip("127.0.0.1", "not-an-ip") == "127.0.0.1"
    assert admin.client_ip("::1", "2001:db8::1") == "2001:db8::1"
    assert admin.client_ip(None, None) == "-"


def test_xff_spoof_from_remote_peer_ignored(env, monkeypatch):
    # TestClient 상대 주소는 "testclient"(IP 아님) → 원격 취급, XFF를 믿지 않는다
    client = TestClient(app)
    client.get("/", headers={"X-Forwarded-For": "8.8.8.8"})
    assert _access(env)[-1]["ip"] != "8.8.8.8"


def test_locate_caches_and_labels(env):
    calls = []

    def fake(ips):
        calls.append(list(ips))
        return {"8.8.8.8": {"country": "미국", "city": "Mountain View", "isp": "Google"}}

    got = admin.locate(["10.0.0.1", "8.8.8.8", "1.1.1.1", "bogus"], fetch=fake)
    assert got["10.0.0.1"] == "내부" and got["bogus"] == "-"
    assert got["8.8.8.8"] == "미국 Mountain View · Google" and got["1.1.1.1"] == "-"
    again = admin.locate(["8.8.8.8", "1.1.1.1"], fetch=fake)
    assert again["8.8.8.8"].startswith("미국") and len(calls) == 1  # 캐시(실패도 1시간)
    assert ((env["admin"] / "geo.json").stat().st_mode & 0o777) == 0o600


def test_locate_failure_does_not_raise(env):
    def boom(ips):
        raise OSError("timeout")

    assert admin.locate(["8.8.4.4"], fetch=boom) == {"8.8.4.4": "-"}


def test_trim_keeps_recent(env, monkeypatch):
    from datetime import datetime

    d = env["admin"]
    d.mkdir(parents=True)
    path = d / "access.jsonl"
    old = {"at": "2020-01-01 00:00:00", "ip": "1.1.1.1"}
    new = {"at": "2026-11-23 00:00:00", "ip": "2.2.2.2"}
    path.write_text("\n".join(json.dumps(x) for x in [old, new, new, new]) + "\nbroken\n")
    monkeypatch.setattr(admin, "MAX_LINES", 2)
    admin._trim(path, now=datetime(2026, 11, 23, tzinfo=admin.KST))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows == [new, new]


def test_my_ips_includes_token_ok_ips():
    rows = [
        {"ip": "1.1.1.1", "auth": "ok"},
        {"ip": "2.2.2.2", "auth": "fail"},
        {"ip": "3.3.3.3", "kind": "page"},
    ]
    assert admin.my_ips(rows, "9.9.9.9") == {"9.9.9.9", "1.1.1.1"}
    v = admin.access_view(
        rows, me="9.9.9.9", auth_only=False, hide_me=True, hours=None, limit=10, offset=0
    )
    assert {r["ip"] for r in v["rows"]} == {"2.2.2.2", "3.3.3.3"}
