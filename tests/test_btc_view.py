"""엔진 상태 API(읽기 전용)와 실전 시작 뒤 계좌 화면."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from minerva import btc_trade as bt
from minerva import btc_view
from minerva.kis import KST
from minerva.main import app

NOW = datetime(2026, 11, 9, 10, 0, tzinfo=KST)


@pytest.fixture
def files(tmp_path, monkeypatch):
    d = tmp_path / "btc_trade"
    d.mkdir()
    monkeypatch.setattr(bt, "DIR", d)
    monkeypatch.setattr(bt, "CAPITAL_FILE", tmp_path / "account" / "binance_capital.json")
    (tmp_path / "account").mkdir()
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(tmp_path / "account" / "latest.json"))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "missing.token"))
    return tmp_path


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def test_pending_before_engine_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "DIR", tmp_path / "none")
    assert btc_view.engine_view()["status"] == "pending"


def test_dry_view(files):
    d = bt.DIR
    pos = {
        "side": "long",
        "qty": 0.03,
        "entry": 76000.0,
        "stop": 73720.0,
        "stop_pct": 3.0,
        "tp": 79800.0,
        "tp_pct": 5.0,
        "opened_at": "2026-11-09 09:30:20",
        "mode": "dry",
    }
    (d / "state.json").write_text(json.dumps({"position": pos, "dry_seed": 815.5}))
    _jsonl(d / "events.jsonl", [
        {"at": "2026-11-09 09:30:20", "mode": "dry", "event": "ai_exits", "side": "long",
         "used": {"stop_pct": 3.0, "tp_pct": 5.0, "reason": "근거"}, "fallback": None, "secs": 4.2,
         "packet": {"big": "x"}},
        {"at": "2026-11-09 09:30:25", "mode": "dry", "event": "run", "bar": "11-09 09:15",
         "signal": "long", "close": 76010.0, "decision": "롱 진입", "equity": 815.5},
        {"at": "2026-11-09 09:45:20", "mode": "dry", "event": "run", "bar": "11-09 09:30",
         "signal": None, "decision": "대기", "equity": 820.0},
    ])  # fmt: skip
    _jsonl(d / "closed.jsonl", [
        {"at": "2026-11-08 23:00:00", "mode": "dry", "side": "short", "pnl": 3.0, "why": "tp"},
        {"at": "2026-11-08 22:00:00", "mode": "live", "side": "long", "pnl": -1.0, "why": "stop"},
    ])  # fmt: skip
    _jsonl(d / "dry.jsonl", [{"at": "2026-11-09 09:30:20", "action": "open", "side": "long"}])
    v = btc_view.engine_view(NOW)
    assert v["mode"] == "dry" and v["position"]["tp"] == 79800.0
    assert v["last_signal"]["decision"] == "롱 진입" and v["last_run"]["equity"] == 820.0
    assert v["last_ai"]["used"]["reason"] == "근거" and "packet" not in v["last_ai"]
    assert [t["mode"] for t in v["trades"]] == ["dry"]  # 지금 모드만
    assert v["gauges"]["seed"] == 815.5 and v["gauges"]["floor"] == pytest.approx(652.4)
    assert v["gauges"]["winrate_n"] == 1 and v["dry_orders"][0]["action"] == "open"
    assert v["params"]["scr"]["stoch_n"] == 9 and v["params"]["leverage"] == 3
    got = TestClient(app).get("/api/btc/engine").json()
    assert got["mode"] == "dry" and got["position"]["side"] == "long"


def test_account_tab_after_start_live(files, monkeypatch):
    """start-live가 쓰는 시드 파일 모양 그대로: 계좌 탭이 그 시각부터 손익을 낸다."""
    acct = files / "account"
    (acct / "binance_capital.json").write_text(
        json.dumps({"live": False, "asset": "USDT", "previous": {"base": 816.4}})
    )
    monkeypatch.setattr(bt.binance, "snapshot", lambda cfg, now, cap: {"net_equity": 815.53})

    class Flat:
        _cfg: dict = {}

        def request(self, method, path, params=None, signed=True):
            assert method == "GET"
            return {
                "/fapi/v1/positionSide/dual": {"dualSidePosition": False},
                "/fapi/v2/positionRisk": [{"symbol": "BTCUSDT", "positionAmt": "0"}],
                "/fapi/v1/openOrders": [],
                "/fapi/v1/openAlgoOrders": [],
            }[path]

    cap = bt.start_live(Flat(), send=lambda m: None)
    since = cap["since_at"]
    (acct / "binance.json").write_text(json.dumps({
        "status": "ok", "updated_at": since, "wallet": 820.0, "equity": 821.0,
        "net_equity": 820.53, "available": 700.0, "positions": [], "orders": [],
        "trades": [{"time": since + ":30", "side": "매수", "qty": 0.03, "price": 76000,
                    "realized": 0, "fee": 1.1}],
    }))  # fmt: skip
    client = TestClient(app)
    got = client.get("/api/account/binance").json()
    assert got.get("live") is not False and got["trades"]
    assert got["returns"]["total"] == pytest.approx(5.0, abs=0.01)
    series = client.get("/api/account/binance/series").json()
    assert series["seed"] == 815.53
    v = TestClient(app).get("/api/btc/engine").json()
    assert v["mode"] == "live" and v["live_since"] == since
