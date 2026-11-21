"""엔진 상태 API(읽기 전용)와 실전 시작 뒤 계좌 화면."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from minerva import btc_trade as bt
from minerva import btc_view, routes
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
    monkeypatch.setattr(btc_view, "WATCH_SYMBOLS", ())  # 관심 종목은 따로 시험
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
    assert v["params"]["scr"]["stoch_n"] == 5 and v["params"]["leverage"] == 3
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
    assert v["live_totals"] == {"n": 0, "wins": 0, "losses": 0, "draws": 0, "winrate": None,
                                "pnl": 0}

    # 합계·승률은 표 상한(MAX_TRADES)과 상관없이 실전 시작 뒤 전체 기록으로
    rows = [{"at": "2000-01-01 00:00:00", "mode": "live", "pnl": 99.0}]
    rows += [{"at": since + f":{i % 60:02d}", "mode": "live", "pnl": 1.0 if i % 2 else -0.5}
             for i in range(btc_view.MAX_TRADES + 10)]
    rows += [{"at": since + ":59", "mode": "dry", "pnl": 7.0}]
    (bt.DIR / "closed.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    v = TestClient(app).get("/api/btc/engine").json()
    n = btc_view.MAX_TRADES + 10
    assert v["live_totals"]["n"] == n and v["live_totals"]["wins"] == n // 2
    assert v["live_totals"]["pnl"] == pytest.approx(n // 2 * 1.0 - (n - n // 2) * 0.5)
    assert len(v["trades"]) == btc_view.MAX_TRADES


def test_engine_view_counts_draws(files):
    d = bt.DIR
    (d / "mode").write_text("live\n")
    bt.CAPITAL_FILE.write_text(json.dumps(
        {"live": True, "since": "2026-11-08", "since_at": "2026-11-08 09:00", "base": 800.0}))
    _jsonl(d / "closed.jsonl", [
        {"at": "2026-11-09 01:00:00", "mode": "live", "pnl": 0.3, "why": "trail", "side": "long",
         "entry": 158.44, "exit": 158.69},  # 진입가 근처 본전 손절 → 무승부
        {"at": "2026-11-09 02:00:00", "mode": "live", "pnl": 9.0, "why": "tp", "pct": 3.0},
        {"at": "2026-11-09 03:00:00", "mode": "live", "pnl": -9.0, "why": "stop", "pct": -3.0},
    ])  # fmt: skip
    v = btc_view.engine_view(NOW)
    t = v["live_totals"]
    assert (t["n"], t["wins"], t["losses"], t["draws"]) == (3, 1, 1, 1)
    assert t["winrate"] == pytest.approx(0.5) and t["pnl"] == pytest.approx(0.3)
    assert [r["outcome"] for r in v["trades"]] == ["loss", "win", "draw"]
    g = v["gauges"]
    assert (g["winrate_n"], g["winrate_wins"], g["winrate_draws"]) == (2, 1, 1)


# ---- 지금 매매하는 종목(신호 가까운 순·바이낸스 종목 버튼·봉 허용 종목) ----


def _universe(d, now: datetime | None = None):
    """BNB·BTC는 들어갈 수 있고, DOGE는 7일 변동폭 2% 미만, NVDA는 최근 7일 승률 60% 미만
    (범위엔 있지만 매매 안 함). 2026-11-18부터 1년 수익 대신 7일 승률로 고른다."""
    now = now or datetime.now(KST)

    def st(wr, ret, rank, kind="COIN"):
        return {"winrate": wr, "ret": ret, "rank": rank, "trades": 40, "days": 182.0,
                "underlying": kind}  # fmt: skip

    stats = {
        "BTCUSDT": st(59.6, 73.8, 1),
        "DOGEUSDT": st(30.0, 10.0, 2),
        "NVDAUSDT": st(55.2, -1.8, 3, "EQUITY"),
        "BNBUSDT": st(66.7, 390.3, 4),
    }
    (d / "universe.json").write_text(
        json.dumps({"updated_at": now.strftime("%Y-%m-%d %H:%M"), "symbols": list(stats),
                    "stats": stats})
    )  # fmt: skip
    ranges = {"BTCUSDT": 4.6, "DOGEUSDT": 1.5, "NVDAUSDT": 8.7, "BNBUSDT": 4.9}
    (d / "ranges.json").write_text(
        json.dumps({"updated_at": now.strftime("%Y-%m-%d %H:%M"), "ranges": ranges})
    )
    good = {"trades": 4, "wins": 3, "winrate": 75.0, "ret": 10.0}
    recent = dict.fromkeys(stats, good)
    recent["NVDAUSDT"] = {"trades": 4, "wins": 2, "winrate": 50.0, "ret": -3.0}
    (d / "recent.json").write_text(
        json.dumps({"updated_at": now.strftime("%Y-%m-%d %H:%M"), "stats": recent})
    )


def _near_rows(syms):
    return [
        {"symbol": s, "side": side, "held": False, "k": 0, "hit": [], "need": [], "bar": 1}
        for s in syms
        for side in ("long", "short")
    ]


def test_traded_view_lists_tradable_by_priority(files):
    _universe(bt.DIR)
    v = btc_view.traded_view()
    assert [r["symbol"] for r in v["symbols"]] == ["BNBUSDT", "BTCUSDT"]
    assert v["window_days"] == 182 and not any(r["held"] for r in v["symbols"])
    assert v["symbols"][0]["name"] == "BNB" and v["symbols"][0]["ret"] == 390.3
    assert v["symbols"][0]["analysis"] is None


def test_traded_view_puts_held_symbol_first_even_if_not_tradable(files):
    _universe(bt.DIR)
    pos = {"symbol": "NVDAUSDT", "side": "long", "entry": 180.0}
    (bt.DIR / "state.json").write_text(json.dumps({"position": pos}))
    got = TestClient(app).get("/api/btc/traded").json()
    rows = got["symbols"]
    assert [r["symbol"] for r in rows] == ["NVDAUSDT", "BNBUSDT", "BTCUSDT"]
    assert rows[0]["held"] and not rows[0]["tradable"] and rows[0]["underlying"] == "EQUITY"
    assert rows[0]["analysis"]["stock"] == "NVDA"  # /api/soxl이 아는 종목


def test_traded_view_old_position_without_symbol_is_btc(files):
    _universe(bt.DIR)
    (bt.DIR / "state.json").write_text(json.dumps({"position": {"side": "short"}}))
    rows = btc_view.traded_view()["symbols"]
    assert rows[0]["symbol"] == "BTCUSDT" and rows[0]["held"] and len(rows) == 2


def test_traded_view_ignores_odd_held_symbol(files):
    _universe(bt.DIR)
    pos = {"symbol": "../etc", "side": "long"}
    (bt.DIR / "state.json").write_text(json.dumps({"position": pos}))
    assert [r["symbol"] for r in btc_view.traded_view()["symbols"]] == ["BNBUSDT", "BTCUSDT"]


def test_near_view_keeps_only_traded_rows(files):
    _universe(bt.DIR)
    pos = {"symbol": "NVDAUSDT", "side": "long"}
    (bt.DIR / "state.json").write_text(json.dumps({"position": pos}))
    rows = _near_rows(["BTCUSDT", "DOGEUSDT", "NVDAUSDT", "BNBUSDT"]) + ["깨짐"]
    near = {"updated_at": "x", "source": "universe", "w": 3, "rows": rows}
    (bt.DIR / "near.json").write_text(json.dumps(near))
    v = TestClient(app).get("/api/btc/near").json()
    assert v["status"] == "ok" and v["window_days"] == 182
    assert v["traded"] == ["NVDAUSDT", "BNBUSDT", "BTCUSDT"]
    assert {r["symbol"] for r in v["rows"]} == {"BTCUSDT", "NVDAUSDT", "BNBUSDT"}
    assert len(v["rows"]) == 6
    assert all(r["held"] == (r["symbol"] == "NVDAUSDT") for r in v["rows"])


def test_near_view_empty_when_nothing_traded(files):
    (bt.DIR / "near.json").write_text(json.dumps({"rows": _near_rows(["SOXLUSDT"])}))
    # universe.json이 없으면 네 종목 기본값 중 수익 +인 BTC·ETH만 매매(SOXL은 손실)
    v = btc_view.near_view()
    assert set(v["traded"]) == {"BTCUSDT", "ETHUSDT"} and v["rows"] == []


def test_kline_whitelist_adds_only_traded_symbols(files):
    from minerva import routes

    _universe(bt.DIR)
    pos = {"symbol": "XAGUSDT", "side": "long"}  # 범위 밖이어도 든 종목은 연다
    (bt.DIR / "state.json").write_text(json.dumps({"position": pos}))
    got = routes.kline_symbols()
    assert got == set(routes.KLINE_SYMBOLS) | {"BNBUSDT", "XAGUSDT"}
    assert "DOGEUSDT" not in got  # 범위에 있어도 매매하지 않는 종목은 닫힌 채
    client = TestClient(app)
    assert client.get("/api/btc/klines?symbol=DOGEUSDT").status_code == 400


def test_kline_whitelist_falls_back_to_four_when_view_fails(files, monkeypatch):
    from minerva import routes

    def boom(now=None):
        raise ValueError("x")

    monkeypatch.setattr(btc_view, "traded_symbols", boom)
    assert routes.kline_symbols() == set(routes.KLINE_SYMBOLS)


def test_analysis_picker_is_stock_and_binance_only():
    page = TestClient(app).get("/").text
    pick = page[page.index('data-pick="an"') : page.index('<div id="an-kr">')]
    assert pick.count("data-show=") == 2
    assert 'data-show="an-kr"' in pick and 'data-show="an-bn"' in pick and "바이낸스" in pick
    for gone in ('id="an-sx"', 'id="an-nv"', 'id="an-et"'):
        assert gone not in page
    assert 'id="bn-sym"' in page and 'id="bn-btc"' in page and 'id="bn-other"' in page


def test_near_and_traded_survive_corrupt_state_and_odd_rows(files):
    # 레드팀 2026-11-13: 깨진 state.json·이상한 종목 값이어도 500이 아니다
    _universe(bt.DIR)
    (bt.DIR / "state.json").write_text("{깨짐")
    rows = _near_rows(["BTCUSDT"]) + [{"symbol": ["x"], "side": "long"}]
    near = {"updated_at": "x", "source": "universe", "w": 3, "rows": rows}
    (bt.DIR / "near.json").write_text(json.dumps(near))
    client = TestClient(app)
    assert client.get("/api/btc/traded").status_code == 200
    v = client.get("/api/btc/near").json()
    assert v["status"] == "ok" and {r["symbol"] for r in v["rows"]} == {"BTCUSDT"}


def test_watch_symbol_always_listed_but_not_traded(files, monkeypatch):
    # 사용자 2026-11-13 "앤트로픽 차트는 꼭 넣어놓자 매매를 하던 안하던"
    monkeypatch.setattr(btc_view, "WATCH_SYMBOLS", ("ANTHROPICUSDT",))
    _universe(bt.DIR)
    v = TestClient(app).get("/api/btc/traded").json()
    last = v["symbols"][-1]
    assert last["symbol"] == "ANTHROPICUSDT" and last["watch"] and not last["tradable"]
    assert not any(r.get("watch") for r in v["symbols"][:-1])
    near = {
        "updated_at": "x",
        "source": "universe",
        "w": 3,
        "rows": _near_rows(["BTCUSDT", "ANTHROPICUSDT"]),
    }
    (bt.DIR / "near.json").write_text(json.dumps(near))
    n = TestClient(app).get("/api/btc/near").json()
    assert "ANTHROPICUSDT" not in n["traded"]
    assert {r["symbol"] for r in n["rows"]} == {"BTCUSDT"}
    assert "ANTHROPICUSDT" in routes.kline_symbols()


def test_account_fills_get_outcome(files):
    """계좌 탭 체결내역: 청산 체결에 엔진 청산 기록의 승/무/패, 진입·못 맞춘 체결은 None."""
    acct = files / "account"
    (acct / "binance_capital.json").write_text(json.dumps(
        {"live": True, "since": "2026-11-17", "since_at": "2026-11-17 00:00", "base": 800.0}))
    _jsonl(bt.DIR / "closed.jsonl", [
        {"at": "2026-11-17 01:15:20", "mode": "live", "symbol": "NVDAUSDT", "side": "long",
         "entry": 158.44, "exit": 158.69, "pnl": 0.3, "pct": 0.16, "why": "trail",
         "opened_at": "2026-11-17 00:30:20"},
        {"at": "2026-11-17 03:00:20", "mode": "live", "symbol": "BTCUSDT", "side": "short",
         "entry": 100.0, "exit": 97.0, "pnl": 9.0, "pct": 3.0, "why": "tp",
         "opened_at": "2026-11-17 02:00:20"},
        {"at": "2026-11-17 05:00:20", "mode": "live", "symbol": "BTCUSDT", "side": "long",
         "entry": 100.0, "exit": 97.0, "pnl": -9.0, "pct": -3.0, "why": "stop",
         "opened_at": "2026-11-17 04:00:20"},
    ])  # fmt: skip
    fill = {"side": "매도", "qty": 1.0, "price": 1.0, "fee": 0.01}
    trades = [
        {**fill, "time": "2026-11-17 04:50:00", "symbol": "BTCUSDT", "realized": -8.9},
        {**fill, "time": "2026-11-17 04:00:21", "symbol": "BTCUSDT", "realized": 0.0},  # 진입
        {**fill, "time": "2026-11-17 02:40:00", "symbol": "BTCUSDT", "realized": 9.1},
        {**fill, "time": "2026-11-17 01:05:00", "symbol": "NVDAUSDT", "realized": 0.4},
        {**fill, "time": "2026-11-17 06:00:00", "symbol": "BTCUSDT", "realized": 1.0},  # 손 매매
    ]
    (acct / "binance.json").write_text(json.dumps({
        "status": "ok", "updated_at": "2026-11-17 06:01", "equity": 800.0, "net_equity": 800.0,
        "positions": [], "orders": [], "trades": trades,
    }))
    got = TestClient(app).get("/api/account/binance").json()["trades"]
    assert [t["outcome"] for t in got] == ["loss", None, "win", "draw", None]


def test_engine_view_rows_carry_their_leverage(files):
    # 2026-11-17 1배 전환: 옛 기록(칸 없음)은 3배, 새 기록은 그 배율 그대로 화면에 넘긴다
    d = bt.DIR
    pos = {"side": "long", "qty": 1.0, "entry": 100.0, "stop": 97.0, "mode": "dry"}
    (d / "state.json").write_text(json.dumps({"position": pos, "dry_seed": 800.0}))
    _jsonl(d / "closed.jsonl", [
        {"at": "2026-11-09 01:00:00", "mode": "dry", "pnl": 1.0, "why": "tp", "pct": 1.0},
        {"at": "2026-11-09 02:00:00", "mode": "dry", "pnl": 1.0, "why": "tp", "pct": 1.0,
         "leverage": 1},
    ])  # fmt: skip
    v = btc_view.engine_view(NOW)
    assert v["position"]["leverage"] == 3
    assert [r["leverage"] for r in v["trades"]] == [1, 3]


def test_engine_trades_need_account_token(tmp_path, monkeypatch):
    # 사용자 2026-11-21: 자동매매 거래 내역·손익·포지션은 비밀번호가 있을 때만
    from fastapi.testclient import TestClient

    from minerva import btc_view, routes
    from minerva.main import app

    full = {
        "status": "ok",
        "live_totals": {"pnl": 5},
        "position": {"symbol": "ANTUSDT"},
        "trades": [{"pnl": 1}],
        "dry_orders": [],
        "last_run": {"at": "x", "position": {"symbol": "ANTUSDT"}, "equity": 900, "signal": None},
        "last_signal": None,
        "gauges": {"seed": 800, "equity": 900, "floor": 640, "winrate_days": 30},
        "params": {"leverage": 3},
    }
    monkeypatch.setattr(btc_view, "engine_view", lambda: dict(full))
    tok = tmp_path / "view.token"
    tok.write_text("secret")
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tok))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    c = TestClient(app)
    v = c.get("/api/btc/engine").json()
    assert v["locked"] and v["trades"] == [] and v["position"] is None
    assert v["live_totals"] is None and v["gauges"]["equity"] is None
    assert "position" not in v["last_run"] and "equity" not in v["last_run"]
    assert v["params"] == {"leverage": 3} and v["gauges"]["winrate_days"] == 30
    v = c.get("/api/btc/engine", headers={"X-Minerva-Token": "secret"}).json()
    assert v["trades"] == [{"pnl": 1}] and "locked" not in v
    assert routes.account_view_token() == "secret"
