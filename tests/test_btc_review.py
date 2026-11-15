import json
import subprocess
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from minerva import btc_ai, btc_review
from minerva import btc_trade as bt
from minerva.kis import KST
from minerva.main import app

M = 15 * 60_000


def ms(s: str) -> int:
    return btc_review._ms(s)


def bar(t: str, o, h, lo, c):
    return [ms(t), str(o), str(h), str(lo), str(c), "0"]


@pytest.fixture
def files(tmp_path, monkeypatch):
    d = tmp_path / "bt"
    d.mkdir()
    monkeypatch.setattr(bt, "DIR", d)
    acct = tmp_path / "acct"
    acct.mkdir()
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(acct / "latest.json"))
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "none"))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    closed = [
        {"at": "2026-11-15 10:15:20", "mode": "live", "symbol": "BTCUSDT", "side": "long",
         "qty": 0.03, "entry": 100.0, "exit": 97.0, "pnl": -9.0, "pct": -3.0, "why": "stop",
         "opened_at": "2026-11-15 10:00:20"},
        {"at": "2026-11-14 10:15:20", "mode": "live", "symbol": "BTCUSDT", "side": "long",
         "qty": 1, "entry": 1, "exit": 1, "pnl": 0, "pct": 0, "why": "stop",
         "opened_at": "2026-11-14 10:00:20"},
        {"at": "2026-11-15 11:00:00", "mode": "dry", "symbol": "BTCUSDT", "side": "long",
         "qty": 1, "entry": 1, "exit": 1, "pnl": 5, "pct": 0, "why": "tp",
         "opened_at": "2026-11-15 10:00:20"},
    ]
    (d / "closed.jsonl").write_text("\n".join(json.dumps(c) for c in closed) + "\n{broken\n")
    events = [
        {"at": "2026-11-15 10:00:20", "mode": "live", "event": "run", "decision": "BTC 롱 진입",
         "halts": [], "position": None},
        {"at": "2026-11-15 10:15:20", "mode": "live", "event": "signal_passed",
         "symbol": "ETHUSDT", "side": "long", "why": "x"},
        {"at": "2026-11-15 10:15:20", "mode": "live", "event": "signal_skip",
         "symbol": "XAGUSDT", "side": "long", "why": "쉬는 시간"},
        {"at": "2026-11-15 10:30:20", "mode": "live", "event": "ai_choose",
         "options": {"keep": "유지"}, "choice": "keep", "ai_raw": '{"reason": "근거"}'},
        {"at": "2026-11-15 10:45:20", "mode": "live", "event": "run", "decision": "대기",
         "halts": ["HALT"], "position": {"symbol": "BTCUSDT"}},
    ]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (acct / "binance.json").write_text(json.dumps({
        "status": "ok", "updated_at": "2026-11-16 00:10", "equity": 95.0, "net_equity": 94.0,
        "income_daily": {"2026-11-15": {"realized": -9.0, "funding": 0.0, "commission": -0.1}},
        "income_range": ["2026-11-14", None]}))
    (acct / "binance_capital.json").write_text(
        json.dumps({"since": "2026-11-14", "base": 100.0, "deposits": []}))
    (acct / "binance_equity.json").write_text(
        json.dumps({"2026-11-14": 100.0, "2026-11-15": 91.0}))
    return d


def bars_stop():
    # 10:00 진입 100, 10:15 봉에서 97 닿음(그 봉 저가 95는 청산 뒤라 안 셈), 뒤 16봉
    out = [bar("2026-11-15 10:00:00", 100, 101.5, 99, 100),
           bar("2026-11-15 10:15:00", 100, 100.5, 95, 96)]
    t = ms("2026-11-15 10:30:00")
    for i in range(16):
        px = 97 + i * 0.2
        out.append([t + i * M, str(px), str(px + 0.5), str(px - 0.5), str(px), "0"])
    return out


def test_excursion_clamps_at_exit_and_reads_after():
    t = {"side": "long", "entry": 100.0, "exit": 97.0,
         "opened_at": "2026-11-15 10:00:20", "at": "2026-11-15 10:15:20"}
    got = btc_review.excursion(t, bars_stop())
    assert got["mfe_pct"] == 1.5 and got["mae_pct"] == -3.0
    assert got["after"]["high_pct"] == pytest.approx(round((100.5 / 97 - 1) * 100, 2))
    assert got["after"]["low_pct"] == pytest.approx(round((96.5 / 97 - 1) * 100, 2))


def test_excursion_short_tp_and_short_after():
    t = {"side": "short", "entry": 100.0, "exit": 97.0,
         "opened_at": "2026-11-15 10:00:20", "at": "2026-11-15 10:15:20"}
    bars = [bar("2026-11-15 10:00:00", 100, 102, 99, 100),
            bar("2026-11-15 10:15:00", 99, 99, 94, 95)]
    got = btc_review.excursion(t, bars)
    assert got["mfe_pct"] == 3.0 and got["mae_pct"] == -2.0 and got["after"] is None


def test_facts_day_filter_and_fields(files):
    now = datetime(2026, 11, 16, 0, 15, tzinfo=KST)
    calls = []

    def fetch(sym, a, b):
        calls.append(sym)
        return bars_stop()

    f = btc_review.facts("2026-11-15", now, fetch)
    assert [t["why_ko"] for t in f["trades"]] == ["손절"]  # 다른 날·드라이런 제외
    assert f["trades"][0]["hold_min"] == 15 and f["trades"][0]["mae_pct"] == -3.0
    assert calls == ["BTCUSDT"]
    assert f["totals"] == {"n": 1, "wins": 0, "pnl": -9.0}
    assert f["partial"] is False
    assert f["passed"] == {"ETHUSDT": 1} and f["passed_total"] == 1
    assert f["skipped"][0]["symbol"] == "XAGUSDT"
    assert f["choices"][0]["reason"] == "근거"
    assert f["entries"] == [{"at": "2026-11-15 10:00:20", "decision": "BTC 롱 진입"}]
    assert f["halts"] == ["HALT"] and f["end_position"] == {"symbol": "BTCUSDT"}
    assert f["account"]["pnl"] == pytest.approx(-9.0)
    assert f["account"]["commission"] == -0.1


def test_facts_survives_fetch_error(files):
    def boom(*a):
        raise OSError("down")

    f = btc_review.facts("2026-11-15", datetime(2026, 11, 15, 12, tzinfo=KST), boom)
    assert f["trades"][0]["mfe_pct"] is None and f["partial"] is True


def test_parse_marks_ideas_and_rejects_junk():
    raw = 'x {"headline": "h", "good": ["a"], "bad": "no", "ideas": ["줄이자", "y (검증 안 됨)"]} y'
    got = btc_review.parse(raw)
    assert got["headline"] == "h" and got["good"] == ["a"] and got["bad"] == []
    assert got["ideas"] == ["줄이자 (검증 안 됨)", "y (검증 안 됨)"]
    assert btc_review.parse("not json") is None
    assert btc_review.parse('{"good": []}') is None


def test_ask_fallbacks():
    def slow(p):
        raise subprocess.TimeoutExpired("claude", 1)

    assert btc_review.ask({}, slow)[0] is None
    assert btc_review.ask({}, lambda p: "??")[1] == "답을 읽지 못함"
    got, err = btc_review.ask({"day": "d"}, lambda p: '{"headline": "ok"}')
    assert got["headline"] == "ok" and err is None


def test_build_save_load_and_api(files):
    now = datetime(2026, 11, 16, 0, 15, tzinfo=KST)
    rec = btc_review.build("2026-11-15", now, caller=lambda p: '{"headline": "복기"}',
                           fetch=lambda *a: bars_stop())
    btc_review.save(rec)
    btc_review.save({**rec, "day": "2026-11-14"})
    (files / "reviews" / "notes.json").write_text("{}")
    (files / "reviews" / "2026-11-13.json").write_text("{broken")
    got = btc_review.load_all()
    assert [r["day"] for r in got] == ["2026-11-15", "2026-11-14"]
    api = TestClient(app).get("/api/btc/reviews").json()["reviews"]
    assert api[0]["review"]["headline"] == "복기"


def test_api_needs_token(files, monkeypatch):
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN", "secret")
    c = TestClient(app)
    assert c.get("/api/btc/reviews").status_code == 403
    assert c.get("/api/btc/reviews", headers={"X-Minerva-Token": "secret"}).status_code == 200


def test_call_claude_model_override(monkeypatch):
    seen = {}

    class P:
        returncode = 0
        pid = 1

        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd

        def communicate(self, timeout=None):
            return "{}", ""

    monkeypatch.setattr(btc_ai.subprocess, "Popen", P)
    btc_ai.call_claude("hi")
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == btc_ai.MODEL
    btc_ai.call_claude("hi", model="opus", budget="1.00")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--max-budget-usd") + 1] == "1.00"
