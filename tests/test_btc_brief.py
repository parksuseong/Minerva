"""엔진 아침 브리핑: 읽기 전용 상태로 글을 만든다. 시세·텔레그램은 가짜."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from minerva import btc_brief
from minerva import btc_trade as bt
from minerva.kis import KST

NOW = datetime(2026, 11, 9, 8, 30, tzinfo=KST)


@pytest.fixture
def files(tmp_path, monkeypatch):
    d = tmp_path / "btc_trade"
    d.mkdir()
    monkeypatch.setattr(bt, "DIR", d)
    monkeypatch.setattr(bt, "CAPITAL_FILE", tmp_path / "binance_capital.json")
    bt.CAPITAL_FILE.write_text(
        json.dumps({"live": True, "base": 815.53, "since_at": "2026-11-08 23:00", "deposits": []})
    )
    (d / "mode").write_text("live\n")
    pos = {
        "side": "short",
        "qty": 0.031,
        "entry": 76000.0,
        "stop": 78280.0,
        "stop_pct": 3.0,
        "tp": 72200.0,
        "tp_pct": 5.0,
        "opened_at": "2026-11-09 04:15:30",
        "mode": "live",
    }
    (d / "state.json").write_text(json.dumps({"position": pos}))
    rows = [
        {
            "at": "2026-11-09 04:15:20",
            "mode": "live",
            "event": "run",
            "bar": "11-09 04:00",
            "signal": "short",
            "close": 76010.0,
            "decision": "숏 진입",
            "equity": 826.0,
        },
        {
            "at": "2026-11-09 08:15:20",
            "mode": "live",
            "event": "run",
            "bar": "11-09 08:00",
            "signal": None,
            "decision": "대기",
            "equity": 830.2,
            "indicators": {"stoch": 35.2, "cci": -60.1, "rsi": 44.8},
        },
    ]
    (d / "events.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    closed = [
        {
            "at": "2026-11-08 23:40:00",
            "mode": "live",
            "side": "long",
            "pct": -3.0,
            "pnl": -2.4,
            "why": "stop",
        },
        {
            "at": "2026-11-09 04:15:25",
            "mode": "live",
            "side": "long",
            "pct": 1.2,
            "pnl": 0.9,
            "why": "signal",
        },
        {
            "at": "2026-11-08 21:00:00",
            "mode": "dry",
            "side": "short",
            "pct": 2.0,
            "pnl": 1.0,
            "why": "tp",
        },
    ]
    (d / "closed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in closed))
    return d


def test_brief_text(files):
    from minerva import btc_view

    v = btc_view.engine_view(NOW)
    ways = {"1h": "상방", "4h": "하방", "1d": "하방"}
    text = btc_brief.brief_text(v, NOW, 75_500.0, ways, "2026-11-08 23:00:00")
    lines = text.splitlines()
    assert lines[0] == "☀️ BTC 자동매매 아침 브리핑 (11-09 08:30 KST)"
    assert "🔴 실전 · 정지 없음" in text
    assert "숏 0.031 BTC @ 76,000.0 · 지금 75,500.0 (+0.66%, 3배 +1.97%)" in text
    assert "손절 78,280.0 (−3%) · 익절 72,200.0 (+5%)" in text
    assert "밤사이 청산 2건: 롱 -3.00% (손절) · 롱 +1.20% (반대 신호) · 합계 -1.50 USDT" in text
    assert "평가액 830.20 USDT · 시드 815.53 대비 +1.80% (+14.67 USDT)" in text
    assert "−20%선 652.42 (여유 +27.2%)" in text and "30일 승률 1/2 = 50%" in text
    assert "마지막 신호: 11-09 04:00 봉 숏 → 숏 진입" in text
    assert (
        "방향 1h 상방 · 4h 하방 · 1d 하방 · SCR① Stoch 35.2 / CCI -60.1 / RSI 44.8 (11-09 08:00 봉)"
        in text
    )
    for gone in ("오늘의 판", "T6D", "계산기", "5배"):
        assert gone not in text


def test_brief_halt_and_flat(files):
    from minerva import btc_view

    (files / "state.json").write_text(json.dumps({"position": None}))
    (files / "NO_ENTRY").write_text("2026-11-09 03:00:00 평가액 정지\n")
    text = btc_brief.brief_text(btc_view.engine_view(NOW), NOW, None, {}, "2026-11-09 05:00:00")
    assert "⛔ 신규 진입 정지: 2026-11-09 03:00:00 평가액 정지" in text
    assert "포지션: 없음" in text and "밤사이 청산: 없음" in text


def test_main_sends_and_remembers(files, monkeypatch):
    sent = []
    monkeypatch.setattr(btc_brief.btc, "fetch_price", lambda: 75_500.0)
    monkeypatch.setattr(
        btc_brief, "directions", lambda now: {"1h": "중립", "4h": "하방", "1d": "하방"}
    )
    from minerva import notify

    monkeypatch.setattr(notify, "send", lambda t: sent.append(t) or True)
    assert btc_brief.main([]) == 0
    assert "자동매매 아침 브리핑" in sent[0]
    saved = json.loads((files / "brief.json").read_text())["last_at"]
    sent.clear()
    assert btc_brief.main(["--dry"]) == 0
    assert not sent and json.loads((files / "brief.json").read_text())["last_at"] == saved
