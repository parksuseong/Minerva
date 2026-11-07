import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from minerva import binance
from minerva.main import app

NOW = datetime(2026, 11, 8, 10, 0, tzinfo=binance.KST)
CFG = {"key": "k", "secret": "s"}

ACCOUNT = {
    "totalWalletBalance": "1000.0",
    "totalUnrealizedProfit": "-12.5",
    "totalMarginBalance": "987.5",
    "availableBalance": "700.0",
    "canTrade": True,
    "positions": [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "-0.03",
            "entryPrice": "76000",
            "unrealizedProfit": "-12.5",
            "leverage": "10",
            "isolated": False,
            "initialMargin": "228.0",
        },
        {
            "symbol": "ETHUSDT",
            "positionAmt": "0",
            "entryPrice": "0",
            "unrealizedProfit": "0",
            "leverage": "20",
            "isolated": False,
        },
    ],
}
RISK = [
    {
        "symbol": "BTCUSDT",
        "positionAmt": "-0.03",
        "markPrice": "76416.6",
        "liquidationPrice": "105000",
    }
]


def test_load_config(tmp_path):
    env = tmp_path / "binance.env"
    env.write_text("# c\nBINANCE_API_KEY=abc\nBINANCE_API_SECRET=x=y\n")
    assert binance.load_config(env) == {"key": "abc", "secret": "x=y"}


def test_parse_account_skips_flat_positions():
    out = binance.parse_account(ACCOUNT, RISK)
    assert out["equity"] == 987.5 and out["wallet"] == 1000.0 and out["can_trade"]
    [p] = out["positions"]
    assert p["side"] == "숏" and p["qty"] == 0.03 and p["mark"] == 76416.6
    assert p["margin_type"] == "교차" and p["leverage"] == 10
    assert p["pnl_pct"] == pytest.approx(-12.5 / 228 * 100)


def test_parse_orders_and_trades_newest_first():
    orders = binance.parse_orders(
        [
            {
                "time": 1794000000000,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_MARKET",
                "origQty": "0.03",
                "price": "0",
                "stopPrice": "77000",
                "status": "NEW",
            },
        ]
    )
    assert orders[0]["price"] == 77000 and orders[0]["side"] == "매도"
    trades = binance.parse_trades(
        [
            {
                "time": 1794000000000,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "qty": "0.1",
                "price": "76000",
                "realizedPnl": "0",
                "commission": "3.0",
            },
            {
                "time": 1794000600000,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "qty": "0.1",
                "price": "76100",
                "realizedPnl": "10",
                "commission": "3.0",
            },
        ]
    )
    assert [t["side"] for t in trades] == ["매도", "매수"]


def test_sum_income_by_type():
    rows = [
        {"incomeType": "REALIZED_PNL", "asset": "USDT", "income": "16.5"},
        {"incomeType": "COMMISSION", "asset": "USDT", "income": "-4"},
        {"incomeType": "FUNDING_FEE", "asset": "USDT", "income": "-0.5"},
        {"incomeType": "TRANSFER", "asset": "USDT", "income": "500"},
    ]
    assert binance.sum_income(rows) == {"realized": 16.5, "funding": -0.5, "commission": -4.0}


def test_since_prefers_minute_timestamp():
    day = binance._since_ms({"since": "2026-11-07"})
    at = binance._since_ms({"since": "2026-11-07", "since_at": "2026-11-07 23:13"})
    assert at - day == (23 * 60 + 13) * 60_000
    assert binance._since_ms(None) is None


def test_snapshot_only_sends_get(monkeypatch):
    calls = []

    def fake_get(cfg, path, **params):
        calls.append(path)
        return {
            "/fapi/v2/account": ACCOUNT,
            "/fapi/v2/positionRisk": RISK,
            "/fapi/v1/openOrders": [],
            "/fapi/v1/userTrades": [],
            "/fapi/v1/income": [],
            "/fapi/v1/commissionRate": {
                "makerCommissionRate": "0.0002",
                "takerCommissionRate": "0.0005",
            },
        }[path]

    monkeypatch.setattr(binance, "_get", fake_get)
    data = binance.snapshot(CFG, NOW, {"since": "2026-11-07"})
    assert data["status"] == "ok" and data["updated_at"] == "2026-11-08 10:00"
    assert "/fapi/v1/income" in calls and not any("order" == c.rsplit("/", 1)[-1] for c in calls)


def test_run_keeps_last_snapshot_on_error(tmp_path, monkeypatch):
    out = tmp_path / "binance.json"
    out.write_text(json.dumps({"status": "ok", "equity": 1.0}))
    monkeypatch.setattr(binance, "load_config", lambda: CFG)

    def boom(*a, **k):
        raise binance.BinanceError("HTTP 401: bad key")

    monkeypatch.setattr(binance, "snapshot", boom)
    with pytest.raises(binance.BinanceError):
        binance.run(out)
    data = json.loads(out.read_text())
    assert data["equity"] == 1.0 and "401" in data["error"]


def test_binance_api_returns_and_token(tmp_path, monkeypatch):
    snap = tmp_path / "latest.json"
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(snap))
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "none"))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    client = TestClient(app)
    assert client.get("/api/account/binance").json()["status"] == "not_connected"

    (tmp_path / "binance.json").write_text(
        json.dumps({"status": "ok", "updated_at": "2026-11-08 10:00", "equity": 900.0})
    )
    (tmp_path / "binance_capital.json").write_text(
        json.dumps({"since": "2026-11-07", "base": 1000.0, "deposits": []})
    )
    (tmp_path / "binance_equity.json").write_text(json.dumps({"2026-11-07": 950.0}))
    r = client.get("/api/account/binance").json()["returns"]
    assert r["total"] == -100.0 and r["today"] == -50.0 and "bot" not in r

    token = tmp_path / "view.token"
    token.write_text("t0k")
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(token))
    assert client.get("/api/account/binance").status_code == 403
    assert client.get("/api/account/binance", headers={"X-Minerva-Token": "t0k"}).status_code == 200


def test_snapshot_hides_trades_before_seed(monkeypatch):
    seed = binance._since_ms({"since": "2026-11-07", "since_at": "2026-11-07 23:13"})
    trade = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "qty": "0.1",
        "price": "76000",
        "realizedPnl": "0",
        "commission": "1",
    }
    rows = [{**trade, "time": seed - 60_000}, {**trade, "time": seed + 60_000}]
    pages = {
        "/fapi/v2/account": ACCOUNT,
        "/fapi/v2/positionRisk": RISK,
        "/fapi/v1/openOrders": [],
        "/fapi/v1/userTrades": rows,
        "/fapi/v1/income": [],
        "/fapi/v1/commissionRate": {
            "makerCommissionRate": "0.0002",
            "takerCommissionRate": "0.0005",
        },
    }
    monkeypatch.setattr(binance, "_get", lambda cfg, path, **p: pages[path])
    capital = {"since": "2026-11-07", "since_at": "2026-11-07 23:13"}
    data = binance.snapshot(CFG, NOW, capital)
    assert [t["time"] for t in data["trades"]] == ["2026-11-07 23:14:00"]


def test_net_equity_subtracts_estimated_close_fee():
    positions = [{"qty": 0.1, "mark": 80000.0, "entry": 76000.0}]
    assert binance.close_fee(positions, 0.0005) == pytest.approx(4.0)
    assert binance.close_fee(
        [{"qty": 0.1, "mark": None, "entry": 76000.0}], 0.0005
    ) == pytest.approx(3.8)


def test_snapshot_reports_fee_rate_and_net_equity(monkeypatch):
    pages = {
        "/fapi/v2/account": ACCOUNT,
        "/fapi/v2/positionRisk": RISK,
        "/fapi/v1/openOrders": [],
        "/fapi/v1/userTrades": [],
        "/fapi/v1/income": [],
        "/fapi/v1/commissionRate": {
            "makerCommissionRate": "0.0002",
            "takerCommissionRate": "0.0005",
        },
    }
    monkeypatch.setattr(binance, "_get", lambda cfg, path, **p: pages[path])
    data = binance.snapshot(CFG, NOW, None)
    assert data["fee_rate"] == {"maker": 0.0002, "taker": 0.0005}
    assert data["close_fee_est"] == pytest.approx(0.03 * 76416.6 * 0.0005)
    assert data["net_equity"] == pytest.approx(987.5 - data["close_fee_est"])


def test_record_series_one_point_per_five_minutes(tmp_path):
    path = tmp_path / "series.json"
    ok = {"status": "ok", "equity": 100.0, "net_equity": 99.0}
    binance.record_series(path, ok, datetime(2026, 11, 8, 10, 1, tzinfo=binance.KST))
    binance.record_series(
        path, {**ok, "net_equity": 98.0}, datetime(2026, 11, 8, 10, 4, tzinfo=binance.KST)
    )
    binance.record_series(path, ok, datetime(2026, 11, 8, 10, 5, tzinfo=binance.KST))
    binance.record_series(
        path, {"status": "error"}, datetime(2026, 11, 8, 10, 11, tzinfo=binance.KST)
    )
    pts = json.loads(path.read_text())["points"]
    assert [p["v"] for p in pts] == [98.0, 99.0]
    # 차트용으로 KST 시각을 UTC 자리에 저장한다: 10:00 KST → 10:00로 보인다
    assert datetime.utcfromtimestamp(pts[0]["t"]).strftime("%H:%M") == "10:00"


def test_returns_use_net_equity_and_series_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(tmp_path / "latest.json"))
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "none"))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    (tmp_path / "binance.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "updated_at": "2026-11-08 10:00",
                "equity": 1010.0,
                "net_equity": 1006.0,
            }
        )
    )
    (tmp_path / "binance_capital.json").write_text(
        json.dumps({"since": "2026-11-08", "base": 1000.0, "deposits": []})
    )
    (tmp_path / "binance_series.json").write_text(json.dumps({"points": [{"t": 1, "v": 1006.0}]}))
    client = TestClient(app)
    assert client.get("/api/account/binance").json()["returns"]["total"] == 6.0
    assert client.get("/api/account/binance/series").json() == {
        "seed": 1000.0,
        "points": [{"t": 1, "v": 1006.0}],
    }
