"""BTCUSDT 자동매매 엔진 테스트. 바이낸스는 전부 가짜(네트워크 없음)."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

import pytest

from minerva import btc_trade as bt
from minerva.kis import KST

NOW = datetime(2026, 11, 8, 22, 30, 20, tzinfo=KST)
NOW_MS = int(NOW.timestamp() * 1000)
BAR = (NOW_MS // bt.BAR_MS - 1) * bt.BAR_MS  # 마지막 마감 봉 시작
FLT = {
    "tick": "0.10",
    "step": "0.001",
    "min_qty": 0.001,
    "max_qty": 1000.0,
    "min_notional": 100.0,
    "status": "TRADING",
}


class FakeExchange:
    """주문·조건부 주문·포지션을 흉내 내는 바이낸스."""

    def __init__(self, price: float = 100_000.0, available: float = 1000.0):
        self.price = price
        self.available = available
        self.amt = 0.0
        self.orders: dict[str, dict] = {}
        self.algos: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.calls: list[tuple[str, str, dict]] = []
        self.leverage = 10
        self.dual = False
        self.reject_stop = False
        self.lose_response: set[str] = set()  # 실행은 됐는데 응답만 잃는 경로
        self.net_fail: dict[str, int] = {}  # 실행 전 통신 오류 횟수
        self.foreign_open: list[dict] = []
        self.t = NOW_MS
        # 지정가 회차마다 꺼내 쓰는 각본: fill / none / partial:<수량> / gtx / expired / race
        self.limit_script: list[str] = []
        self.limits: dict[str, dict] = {}

    @property
    def writes(self) -> list[tuple[str, str, dict]]:
        return [c for c in self.calls if c[0] in ("POST", "DELETE")]

    def request(self, method, path, params=None, signed=True):
        params = dict(params or {})
        self.calls.append((method, path, params))
        key = f"{method} {path}"
        if self.net_fail.get(key):
            self.net_fail[key] -= 1
            raise bt.NetworkError("timeout")
        out = self._handle(method, path, params)
        if key in self.lose_response:
            self.lose_response.discard(key)
            raise bt.NetworkError("response lost")
        return out

    def _handle(self, method, path, p):
        if path == "/fapi/v2/positionRisk":
            return [{"symbol": "BTCUSDT", "positionAmt": str(self.amt)}]
        if path == "/fapi/v1/openAlgoOrders":
            return [a for a in self.algos.values() if a["algoStatus"] == "NEW"] + [
                o for o in self.foreign_open if "clientAlgoId" in o
            ]
        if path == "/fapi/v1/openOrders":
            mine = [
                {"clientOrderId": cid}
                for cid, o in self.limits.items()
                if o["status"] in ("NEW", "PARTIALLY_FILLED")
            ]
            return mine + [o for o in self.foreign_open if "clientOrderId" in o]
        if path == "/fapi/v2/account":
            eq = self.available
            return {"availableBalance": str(eq), "totalMarginBalance": str(eq), "canTrade": True}
        if path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": self.dual}
        if path == "/fapi/v1/symbolConfig":
            return [{"symbol": "BTCUSDT", "leverage": self.leverage, "marginType": "CROSSED"}]
        if path == "/fapi/v1/leverage":
            self.leverage = int(p["leverage"])
            return {"leverage": self.leverage}
        if path == "/fapi/v1/userTrades":
            end = p.get("endTime", 1 << 62)
            return [t for t in self.trades if p["startTime"] <= t["time"] <= end]
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": []}
        if path == "/fapi/v1/order" and method == "GET":
            cid = p["origClientOrderId"]
            if cid in self.limits:
                return self._limit_status(cid)
            o = self.orders.get(cid)
            if not o:
                raise bt.Rejected(-2013, "Order does not exist.")
            return o
        if path == "/fapi/v1/order" and method == "POST" and p["type"] == "LIMIT":
            assert p["timeInForce"] == "GTX"
            how = self.limit_script.pop(0) if self.limit_script else "fill"
            if how == "gtx":
                raise bt.Rejected(-5022, "Post Only order will be rejected")
            o = {
                "clientOrderId": p["newClientOrderId"],
                "status": "NEW",
                "executedQty": "0",
                "avgPrice": "0",
                "price": p["price"],
                "origQty": p["quantity"],
                "side": p["side"],
                "how": how,
            }
            if how == "expired":
                o["status"] = "EXPIRED"
            self.limits[p["newClientOrderId"]] = o
            return {k: v for k, v in o.items() if k != "how"}
        if path == "/fapi/v1/order" and method == "DELETE":
            o = self.limits.get(p["origClientOrderId"])
            if not o or o["status"] not in ("NEW", "PARTIALLY_FILLED"):
                raise bt.Rejected(-2011, "Unknown order sent.")
            if o["how"] == "race":  # 취소 직전에 다 체결됐다
                self._fill(o, float(o["origQty"]))
                raise bt.Rejected(-2011, "Unknown order sent.")
            o["status"] = "CANCELED"
            return dict(o)
        if path == "/fapi/v1/algoOrder" and method == "GET":
            a = self.algos.get(p["clientAlgoId"])
            if not a:
                raise bt.Rejected(-2013, "Order does not exist.")
            return a
        if path == "/fapi/v1/order" and method == "POST":
            qty = float(p["quantity"])
            sign = 1 if p["side"] == "BUY" else -1
            if p.get("reduceOnly") == "true":
                qty = min(qty, abs(self.amt))
            self.amt = round(self.amt + sign * qty, 6)
            self.t += 1000
            o = {
                "clientOrderId": p["newClientOrderId"],
                "status": "FILLED",
                "avgPrice": str(self.price),
                "executedQty": str(qty),
                "updateTime": self.t,
            }
            self.orders[p["newClientOrderId"]] = o
            self.trades.append(
                {
                    "time": self.t,
                    "side": p["side"],
                    "qty": str(qty),
                    "price": str(self.price),
                    "realizedPnl": "0",
                    "commission": "0.01",
                    "commissionAsset": "USDT",
                }
            )
            return o
        if path == "/fapi/v1/algoOrder" and method == "POST":
            if self.reject_stop:
                raise bt.Rejected(-2021, "Order would immediately trigger.")
            a = {
                "clientAlgoId": p["clientAlgoId"],
                "algoStatus": "NEW",
                "triggerPrice": p["triggerPrice"],
                "side": p["side"],
                "type": p["type"],
            }
            self.algos[p["clientAlgoId"]] = a
            return a
        if path == "/fapi/v1/algoOrder" and method == "DELETE":
            a = self.algos.get(p["clientAlgoId"])
            if not a or a["algoStatus"] != "NEW":
                raise bt.Rejected(-2011, "Unknown order sent.")
            a["algoStatus"] = "CANCELED"
            return a
        raise AssertionError(f"unexpected {method} {path}")

    def _fill(self, o: dict, q: float) -> None:
        done = float(o["executedQty"])
        q = min(q, float(o["origQty"]) - done)
        if q <= 0:
            return
        px = float(o["price"])
        sign = 1 if o["side"] == "BUY" else -1
        self.amt = round(self.amt + sign * q, 6)
        self.t += 1000
        self.trades.append(
            {
                "time": self.t,
                "side": o["side"],
                "qty": str(q),
                "price": str(px),
                "realizedPnl": "0",
                "commission": "0.004",
                "commissionAsset": "USDT",
            }
        )
        o["executedQty"] = str(round(done + q, 6))
        o["avgPrice"] = str(px)
        o["status"] = (
            "FILLED" if float(o["executedQty"]) >= float(o["origQty"]) else "PARTIALLY_FILLED"
        )

    def _limit_status(self, cid: str) -> dict:
        o = self.limits[cid]
        if o["status"] in ("NEW", "PARTIALLY_FILLED"):
            if o["how"] == "fill":
                self._fill(o, float(o["origQty"]))
            elif o["how"].startswith("partial:") and float(o["executedQty"]) == 0:
                self._fill(o, float(o["how"].split(":")[1]))
        return {k: v for k, v in o.items() if k != "how"}

    def trigger_stop(self, cid: str, price: float) -> None:
        """손절이 걸려 포지션이 닫힌 상황."""
        self.algos[cid]["algoStatus"] = "FINISHED"
        side = "SELL" if self.amt > 0 else "BUY"
        self.trades.append(
            {
                "time": self.t + 5000,
                "side": side,
                "qty": str(abs(self.amt)),
                "price": str(price),
                "realizedPnl": "-30",
                "commission": "0.05",
                "commissionAsset": "USDT",
            }
        )
        self.amt = 0.0


class FakeMarket:
    def __init__(self, price: float = 100_000.0):
        self._price = price
        self.rows = rows(300, 100_000.0)
        self.spread = 0.2
        self.mins: list[list] | None = None

    def book(self):
        return self._price, self._price + self.spread

    def minutes(self, n=20):
        if self.mins is not None:
            return self.mins
        t = NOW_MS - NOW_MS % 60000
        px = self._price
        return [
            [t - 120000, px, px, px, px, 1, t - 60001],
            [t - 60000, px, px, px, px, 1, t - 1],
            [t, px, px, px, px, 1, t + 59999],
        ]

    def klines(self, interval):
        return self.rows

    def price(self):
        return self._price

    def funding(self):
        return 0.0001


def rows(n: int, px: float) -> list[list]:
    out = []
    for k in range(n):
        t = BAR - (n - 1 - k) * bt.BAR_MS
        out.append([t, px, px * 1.001, px * 0.999, px, 1.0, t + bt.BAR_MS - 1])
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "DIR", tmp_path / "btc_trade")
    monkeypatch.setattr(bt, "CAPITAL_FILE", tmp_path / "binance_capital.json")
    clock = {"t": 0.0}
    monkeypatch.setattr(bt, "_sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(bt, "_clock", lambda: clock["t"])
    monkeypatch.setattr(bt, "filters", lambda c: dict(FLT))
    monkeypatch.setattr(bt, "chase_gate", lambda m, r, n: {"pct": 1.0, "big": False})
    asked: list[str] = []

    def ai(prompt: str) -> str:
        asked.append(prompt)
        return AI_ANSWER["text"]

    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": null, "reason": "기본과 같게"}'
    monkeypatch.setattr(bt, "AI_CALLER", ai)
    sent: list[str] = []
    bt.DIR.mkdir(parents=True)
    return {"sent": sent, "send": sent.append, "tmp": tmp_path, "asked": asked}


AI_ANSWER = {"text": ""}


def live(env, base: float = 1000.0):
    (bt.DIR / "mode").write_text("live\n")
    bt.CAPITAL_FILE.write_text(json.dumps({"live": True, "base": base, "deposits": []}))


def with_signal(monkeypatch, d, close=100_000.0, bar=BAR, **extra):
    sig = {"dir": d, "bar": bar, "close": close, **extra}
    monkeypatch.setattr(bt, "latest_signal", lambda r, n, last, lookback=2: dict(sig))


def run(env, ex, mk=None):
    return bt.run(now=NOW, client=ex, market_=mk or FakeMarket(ex.price), send=env["send"])


def state() -> dict:
    return json.loads((bt.DIR / "state.json").read_text())


def jsonl(name: str) -> list[dict]:
    path = bt.DIR / name
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def engine_long(ex: FakeExchange, qty=0.029, entry=100_000.0) -> dict:
    """엔진이 연 롱(손절 포함)을 거래소와 상태에 만든다."""
    ex.amt = qty
    sid = f"mnv-{(BAR - bt.BAR_MS * 8) // 60000}-s"
    ex.algos[sid] = {"clientAlgoId": sid, "algoStatus": "NEW", "triggerPrice": "97000.0"}
    pos = {
        "side": "long",
        "qty": qty,
        "entry": entry,
        "stop": 97000.0,
        "stop_id": sid,
        "bar": BAR - bt.BAR_MS * 8,
        "entry_ms": NOW_MS - 3_600_000,
        "opened_at": "2026-11-08 21:30:20",
        "mode": "live",
    }
    bt._save(bt.DIR / "state.json", {"position": pos, "last_bar": BAR - bt.BAR_MS * 8})
    return pos


# ---------------------------------------------------------------- 수량·가격


def test_qty_rounds_down_to_step():
    qty, why = bt.size_qty(1000.0, 76_041.4, 1.0, FLT)
    # 1000 × 0.99 × 3 / 76041.4 = 0.03905… → 0.039
    assert qty == Decimal("0.039") and why == ""
    half, _ = bt.size_qty(1000.0, 76_041.4, 0.5, FLT)
    assert half == Decimal("0.019")


def test_min_notional_and_min_qty_block():
    qty, why = bt.size_qty(30.0, 100_000.0, 1.0, FLT)  # 0.00089 → 0
    assert qty == 0 and "최소" in why
    flt = {**FLT, "min_notional": 5000.0}
    qty, why = bt.size_qty(1000.0, 100_000.0, 1.0, flt)
    assert qty == 0 and "주문 금액" in why


def test_stop_price_never_tighter_than_3pct():
    assert bt.stop_price("long", 76_041.37, "0.10") == Decimal("73760.10")
    assert bt.stop_price("short", 76_041.37, "0.10") == Decimal("78322.70")
    assert float(bt.stop_price("long", 76_041.37, "0.10")) <= 76_041.37 * 0.97
    assert float(bt.stop_price("short", 76_041.37, "0.10")) >= 76_041.37 * 1.03


# ---------------------------------------------------------------- 신호


def _fake_signals(long_at, short_at):
    return lambda h, lo, c, p: {
        "long": long_at,
        "short": short_at,
        "stoch": [50.0] * len(c),
        "cci": [0.0] * len(c),
        "rsi": [50.0] * len(c),
    }


def test_latest_signal_uses_closed_bars_and_engine_params(monkeypatch):
    seen = {}

    def fake(h, lo, c, p):
        seen["n"], seen["p"] = len(c), p
        return _fake_signals([len(c) - 1], [])(h, lo, c, p)

    monkeypatch.setattr(bt.scr_lab, "signals", fake)
    r = rows(300, 100.0)
    r.append([BAR + bt.BAR_MS, 1, 1, 1, 1, 1, BAR + 2 * bt.BAR_MS - 1])  # 진행 중 봉
    sig = bt.latest_signal(r, NOW_MS, None)
    assert seen["n"] == 300 and seen["p"]["stoch_n"] == 9 and seen["p"]["rsi_n"] == 21
    assert seen["p"]["mode"] == "cross" and seen["p"]["need"] == 3 and seen["p"]["w"] == 3
    assert sig["dir"] == "long" and sig["bar"] == BAR


def test_latest_signal_catch_up_and_dedupe(monkeypatch):
    r = rows(300, 100.0)
    monkeypatch.setattr(bt.scr_lab, "signals", _fake_signals([], [298]))
    assert bt.latest_signal(r, NOW_MS, None)["dir"] is None  # 첫 실행은 마지막 봉만
    sig = bt.latest_signal(r, NOW_MS, BAR - 2 * bt.BAR_MS)
    assert sig["dir"] == "short" and sig["age"] == 1  # 한 번 빠진 실행은 따라잡는다
    assert bt.latest_signal(r, NOW_MS, BAR - bt.BAR_MS)["dir"] is None  # 이미 처리
    monkeypatch.setattr(bt.scr_lab, "signals", _fake_signals([299], [299]))
    sig = bt.latest_signal(r, NOW_MS, None)
    assert sig["dir"] is None and sig["conflict"]


def test_latest_signal_flags_stale_data(monkeypatch):
    monkeypatch.setattr(bt.scr_lab, "signals", _fake_signals([], []))
    old = [[r[0] - 3_600_000, *r[1:6], r[6] - 3_600_000] for r in rows(300, 1.0)]
    assert bt.latest_signal(old, NOW_MS, None)["stale"]


def test_real_scr_computation_runs_on_fixture():
    import pathlib

    fx = pathlib.Path(__file__).parent / "fixtures" / "btc_15m_scr.json"
    data = json.loads(fx.read_text())
    bars = data["bars"]
    rs = [[int(b[0]), *map(float, b[1:5]), 0.0, int(b[0]) + bt.BAR_MS - 1] for b in bars]
    end = rs[-1][6] + 1
    sig = bt.latest_signal(rs, end, rs[-3][0])
    assert "dir" in sig and not sig.get("error")


# ---------------------------------------------------------------- 추격 한도


def test_chase_limit_directional():
    assert bt.chase_ok("long", 100.99, 100.0, 1.0)
    assert not bt.chase_ok("long", 101.0, 100.0, 1.0)
    assert bt.chase_ok("long", 95.0, 100.0, 1.0)  # 진입 방향 반대로 벌어진 건 막지 않는다
    assert not bt.chase_ok("short", 99.0, 100.0, 1.0)
    assert bt.chase_ok("short", 99.0, 100.0, 3.0)
    assert bt.chase_ok("short", 104.0, 100.0, 1.0)


def test_run_skips_entry_beyond_chase_limit(env, monkeypatch):
    live(env)
    ex = FakeExchange(price=101_500.0)
    with_signal(monkeypatch, "long", close=100_000.0)
    s = run(env, ex)
    assert not ex.writes and state().get("position") is None
    assert "진입 안 함" in s["decision"] and any("추격 한도" in m for m in env["sent"])
    assert state()["last_bar"] == BAR


def test_big_move_allows_3pct(env, monkeypatch):
    live(env)
    monkeypatch.setattr(bt, "chase_gate", lambda m, r, n: {"pct": 3.0, "big": True})
    ex = FakeExchange(price=102_000.0)
    with_signal(monkeypatch, "long", close=100_000.0)
    run(env, ex)
    assert state()["position"]["side"] == "long"


# ---------------------------------------------------------------- 실전 주문 흐름


def test_flat_long_signal_opens_long_with_stop(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    s = run(env, ex)
    writes = [(m, p) for m, p, _ in ex.writes]
    assert writes == [
        ("POST", "/fapi/v1/leverage"),
        ("POST", "/fapi/v1/order"),
        ("POST", "/fapi/v1/algoOrder"),
    ]
    order = ex.writes[1][2]
    assert order["side"] == "BUY" and order["type"] == "LIMIT" and order["quantity"] == "0.029"
    assert order["timeInForce"] == "GTX" and order["price"] == "100000.0"  # 매수 1호가
    assert order["newClientOrderId"] == f"mnv-{BAR // 60000}-o1"
    stop = ex.writes[2][2]
    assert stop["type"] == "STOP_MARKET" and stop["closePosition"] == "true"
    assert stop["side"] == "SELL" and stop["triggerPrice"] == "97000.0"
    assert stop["workingType"] == "MARK_PRICE" and stop["algoType"] == "CONDITIONAL"
    pos = state()["position"]
    assert pos["side"] == "long" and pos["qty"] == 0.029 and pos["stop"] == 97000.0
    assert s["decision"] == "롱 진입" and ex.leverage == 3


def test_reverse_sequence_long_to_short(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    old = engine_long(ex)
    with_signal(monkeypatch, "short")
    run(env, ex)
    seq = [(m, p, x.get("side"), x.get("reduceOnly")) for m, p, x in ex.writes]
    assert seq == [
        ("DELETE", "/fapi/v1/algoOrder", None, None),  # 1) 엔진 손절 취소
        ("POST", "/fapi/v1/order", "SELL", "true"),  # 2) 시장가 reduceOnly 청산
        ("POST", "/fapi/v1/order", "SELL", None),  # 4) 시장가 숏 진입
        ("POST", "/fapi/v1/algoOrder", "BUY", None),  # 6) 숏 손절
    ]
    assert ex.writes[1][2]["quantity"] == "0.029"
    assert ex.algos[old["stop_id"]]["algoStatus"] == "CANCELED"
    pos = state()["position"]
    assert pos["side"] == "short" and pos["stop"] == pytest.approx(103000.3)  # 매도 1호가 진입
    assert ex.amt == pytest.approx(-0.029)
    closed = jsonl("closed.jsonl")
    assert closed[-1]["why"] == "signal" and closed[-1]["side"] == "long"


def test_same_direction_signal_holds(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    with_signal(monkeypatch, "long")
    s = run(env, ex)
    assert not ex.writes and "유지" in s["decision"]


def test_stop_rejected_closes_position(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.reject_stop = True
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.amt == 0 and state()["position"] is None
    last = ex.writes[-1][2]
    assert last["reduceOnly"] == "true" and last["side"] == "SELL"
    assert any("손절 주문 실패" in m for m in env["sent"])
    assert jsonl("closed.jsonl")[-1]["why"] == "protect"


def test_missing_stop_is_replaced(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    pos = engine_long(ex)
    ex.algos.clear()
    with_signal(monkeypatch, None)
    run(env, ex)
    placed = [x for m, p, x in ex.writes if p == "/fapi/v1/algoOrder"]
    assert len(placed) == 1 and placed[0]["triggerPrice"] == "97000.0"
    assert state()["position"]["stop_id"] == f"mnv-{pos['bar'] // 60000}-s1"
    assert any("다시 걸었어요" in m for m in env["sent"])


def test_missing_stop_and_replace_fails_closes(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.algos.clear()
    ex.reject_stop = True
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0 and state()["position"] is None


def test_stop_fill_is_reconciled(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    pos = engine_long(ex)
    ex.trigger_stop(pos["stop_id"], 97_000.0)
    with_signal(monkeypatch, None)
    run(env, ex)
    c = jsonl("closed.jsonl")[-1]
    assert c["why"] == "stop" and c["exit"] == 97_000.0 and c["pnl"] < 0
    assert state()["position"] is None and not bt.entries_blocked()
    assert not ex.writes
    assert any("손절 체결" in m for m in env["sent"])


def test_after_stop_waits_for_next_signal(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    pos = engine_long(ex)
    ex.trigger_stop(pos["stop_id"], 97_000.0)
    with_signal(monkeypatch, None)
    run(env, ex)
    run(env, ex)
    assert not ex.writes and ex.amt == 0


def test_manual_close_sets_no_entry(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.0  # 사용자가 앱에서 닫음 (손절은 그대로 NEW)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert jsonl("closed.jsonl")[-1]["why"] == "manual"
    assert bt.entries_blocked()


def test_foreign_position_is_left_alone(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.amt = 0.5  # 엔진이 열지 않은 포지션
    with_signal(monkeypatch, "short")
    s = run(env, ex)
    assert not ex.writes and bt.entries_blocked()
    assert any("엔진이 열지 않은" in m for m in env["sent"])
    assert "수동 개입" in s["decision"]
    env["sent"].clear()
    run(env, ex)  # 같은 포지션은 다시 알리지 않는다
    assert not any("엔진이 열지 않은" in m for m in env["sent"])


def test_qty_mismatch_is_left_alone(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.05  # 사용자가 더 샀다
    with_signal(monkeypatch, "short")
    run(env, ex)
    assert not ex.writes and bt.entries_blocked()


def test_foreign_open_order_blocks_entry(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.foreign_open = [{"clientOrderId": "web_abc"}]
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert not [w for w in ex.writes if w[1] == "/fapi/v1/order"]


def test_hedge_mode_blocks_trading(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.dual = True
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert not ex.writes


# ---------------------------------------------------------------- 멱등·재시도


def test_lost_response_is_not_resent(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.lose_response = {"POST /fapi/v1/order"}
    with_signal(monkeypatch, "long")
    run(env, ex)
    sends = [w for w in ex.writes if w[1] == "/fapi/v1/order"]
    assert len(sends) == 1 and ex.amt == pytest.approx(0.029)
    assert state()["position"]["qty"] == 0.029


def test_network_error_retried_at_most_3_times(env, monkeypatch):
    ex = FakeExchange()
    ex.net_fail = {"POST /fapi/v1/order": 10}
    with pytest.raises(bt.NetworkError):
        bt.market(ex, "SELL", Decimal("0.01"), "mnv-1-c", reduce=True)
    assert len([c for c in ex.calls if c[0] == "POST"]) == 4  # 청산: 처음 1 + 재시도 3
    ex.calls.clear()
    with pytest.raises(bt.NetworkError):
        bt.market(ex, "BUY", Decimal("0.01"), "mnv-1-o", reduce=False)
    assert len([c for c in ex.calls if c[0] == "POST"]) == 1  # 진입은 다시 보내지 않는다


def test_rejection_not_retried(env):
    ex = FakeExchange()
    ex.reject_stop = True
    with pytest.raises(bt.Rejected):
        bt.place_stop(ex, "long", Decimal("97000"), "mnv-1-s")
    assert len([c for c in ex.calls if c[0] == "POST"]) == 1


def test_open_response_lost_entirely_is_adopted_next_run(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.lose_response = {"POST /fapi/v1/order"}
    ex.net_fail = {"GET /fapi/v1/order": 0}

    real_query = bt._query
    calls = {"n": 0}

    def flaky(client, path, key, cid):
        if path == "/fapi/v1/order" and cid.endswith("-o"):
            calls["n"] += 1
            if 1 < calls["n"] <= 5:  # 보낸 뒤 확인 조회가 모두 실패
                raise bt.NetworkError("down")
        return real_query(client, path, key, cid)

    monkeypatch.setattr(bt, "_query", flaky)
    with_signal(monkeypatch, "long")
    run(env, ex)
    # 다시 보내지 않고, 같은 실행 안에서 포지션을 보고 이어 받아 손절까지 건다
    assert len([w for w in ex.writes if w[1] == "/fapi/v1/order"]) == 1
    pos = state()["position"]
    assert pos and pos["side"] == "long" and "pending" not in state()
    assert any(a["algoStatus"] == "NEW" for a in ex.algos.values())


def test_open_never_resent_even_if_query_lags(env, monkeypatch):
    """응답을 잃고 조회에도 아직 안 보여도 진입은 두 번 보내지 않는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.lose_response = {"POST /fapi/v1/order"}
    real_query = bt._query
    hide = {"on": True}

    def lagging(client, path, key, cid):
        if path == "/fapi/v1/order" and cid.endswith("-o") and hide["on"]:
            return None
        return real_query(client, path, key, cid)

    monkeypatch.setattr(bt, "_query", lagging)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len([w for w in ex.writes if w[1] == "/fapi/v1/order"]) == 1
    assert ex.amt == pytest.approx(0.029)
    assert state()["position"]["qty"] == 0.029  # 포지션 조회로 이어 받았다


def test_stop_ok_when_open_list_lags(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    real = ex._handle

    def lag(method, path, p):
        if path == "/fapi/v1/openAlgoOrders":
            return []  # 목록 반영이 늦다
        return real(method, path, p)

    ex._handle = lag
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert state()["position"] and ex.amt > 0  # 응답 NEW를 믿고 닫지 않는다


def test_live_position_kept_when_switched_to_dry(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    (bt.DIR / "mode").write_text("dry\n")
    with_signal(monkeypatch, None)
    run(env, ex)
    assert state()["position"] is None and state()["orphan_live"]["side"] == "long"
    assert any("드라이런으로 바뀌었어요" in m for m in env["sent"])
    live(env)
    run(env, ex)
    assert state()["position"]["mode"] == "live" and not ex.writes


def test_mismatch_keeps_a_stop(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.algos.clear()
    ex.amt = 0.05
    with_signal(monkeypatch, None)
    run(env, ex)
    posts = [x for m, p, x in ex.writes]
    assert len(posts) == 1 and posts[0]["type"] == "STOP_MARKET" and ex.amt == 0.05


def test_unknown_pnl_not_counted_in_winrate(env):
    for _ in range(8):
        bt._append("closed.jsonl", {"at": "2026-11-05 10:00:00", "mode": "live", "pnl": None})
    assert bt.winrate("live", NOW) == (0, 0)


def test_ai_fallback_streak_alerts(env, monkeypatch):
    AI_ANSWER["text"] = "모르겠어요"
    ex = FakeExchange()
    for k in range(3):
        with_signal(monkeypatch, "long" if k % 2 == 0 else "short", bar=BAR + k * bt.BAR_MS)
        run(env, ex)
    assert any("3번 연속 기본값" in m for m in env["sent"])


# ---------------------------------------------------------------- 정지 규칙


def test_equity_floor_halts_new_entries(env, monkeypatch):
    live(env, base=1300.0)  # 1000 < 1300 × 0.8
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    s = run(env, ex)
    assert bt.entries_blocked() and not ex.writes
    assert any("평가액" in h for h in s["halts"])
    assert any("신규 진입 정지" in m for m in env["sent"])


def test_winrate_halt(env, monkeypatch):
    live(env)
    for k in range(8):
        bt._append(
            "closed.jsonl",
            {"at": f"2026-11-0{k % 7 + 1} 10:00:00", "mode": "live", "pnl": 1.0 if k < 3 else -1.0},
        )
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    s = run(env, ex)
    assert bt.entries_blocked() and any("승률 3/8" in h for h in s["halts"])
    assert not ex.writes


def test_winrate_needs_8_trades_and_same_mode(env, monkeypatch):
    live(env)
    for _ in range(7):
        bt._append("closed.jsonl", {"at": "2026-11-05 10:00:00", "mode": "live", "pnl": -1.0})
    for _ in range(5):
        bt._append("closed.jsonl", {"at": "2026-11-05 10:00:00", "mode": "dry", "pnl": -1.0})
    assert bt.check_halts("live", {}, 1000.0, NOW, lambda m: None) == []


def test_halt_file_blocks_entry_but_reverse_still_closes(env, monkeypatch):
    live(env)
    (bt.DIR / "HALT").write_text("")
    ex = FakeExchange()
    engine_long(ex)
    with_signal(monkeypatch, "short")
    s = run(env, ex)
    assert ex.amt == 0 and state()["position"] is None  # 롱은 반대 신호로 닫는다
    assert not [w for w in ex.writes if w[2].get("side") == "SELL" and not w[2].get("reduceOnly")]
    assert "진입 정지" in s["decision"]


def test_live_without_seed_blocks_entry(env, monkeypatch):
    (bt.DIR / "mode").write_text("live\n")
    bt.CAPITAL_FILE.write_text(json.dumps({"live": False}))
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    s = run(env, ex)
    assert not ex.writes and any("시드 없음" in h for h in s["halts"])


# ---------------------------------------------------------------- 드라이런


def test_dry_mode_never_writes_to_exchange(env, monkeypatch):
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    run(env, ex)
    with_signal(monkeypatch, "short", bar=BAR + bt.BAR_MS)
    run(env, ex)
    assert not ex.writes
    dry = jsonl("dry.jsonl")
    assert [d["action"] for d in dry] == ["open", "close", "open"]
    assert dry[0]["side"] == "long" and dry[0]["qty"] == 0.029 and dry[0]["path"]
    assert any(m.startswith("[드라이런] 🟢 롱 진입 예정") for m in env["sent"])
    assert any("엔진 시작 — 드라이런" in m for m in env["sent"])
    assert state()["position"]["side"] == "short" and state()["position"]["mode"] == "dry"
    assert jsonl("closed.jsonl")[-1]["mode"] == "dry"


def test_dry_virtual_stop(env, monkeypatch):
    ex = FakeExchange()
    with_signal(monkeypatch, "long", bar=BAR - bt.BAR_MS * 3)
    mk = FakeMarket()
    run(env, ex, mk)
    mk.rows[-1][3] = 96_000.0  # 진입 뒤 봉의 저가가 손절 아래
    with_signal(monkeypatch, None)
    run(env, ex, mk)
    c = jsonl("closed.jsonl")[-1]
    assert c["why"] == "stop" and c["exit"] == 97_000.0 and state()["position"] is None
    assert not ex.writes


def test_dry_position_dropped_when_switching_to_live(env, monkeypatch):
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    run(env, ex)
    live(env)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert state()["position"] is None and not ex.writes


# ---------------------------------------------------------------- 실전 시작


def test_start_live_writes_seed_and_mode(env, monkeypatch):
    bt.CAPITAL_FILE.write_text(
        json.dumps({"live": False, "asset": "USDT", "previous": {"base": 816.4}, "note": "x"})
    )
    bt._save(bt.DIR / "state.json", {"position": {"side": "long", "mode": "dry"}})
    monkeypatch.setattr(bt.binance, "snapshot", lambda cfg, now, cap: {"net_equity": 815.1})
    ex = FakeExchange()
    ex._cfg = {}
    sent = []
    cap = bt.start_live(ex, send=sent.append)
    saved = json.loads(bt.CAPITAL_FILE.read_text())
    assert saved == cap
    assert saved["live"] is True and saved["base"] == 815.1 and saved["asset"] == "USDT"
    assert saved["deposits"] == [] and saved["previous"] == {"base": 816.4}
    assert len(saved["since_at"]) == 16 and "note" not in saved
    assert bt.mode() == "live" and state()["position"] is None
    assert not ex.writes and "실전 시작" in sent[0]


def test_start_live_refuses_with_open_position(env, monkeypatch):
    ex = FakeExchange()
    ex.amt = 0.01
    ex._cfg = {}
    with pytest.raises(SystemExit):
        bt.start_live(ex, send=lambda m: None)
    assert bt.mode() == "dry"


def test_leverage_not_changed_in_dry(env, monkeypatch):
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.leverage == 10 and not ex.writes


# ---------------------------------------------------------------- AI 손절·익절


def test_ai_parse_valid_and_fence():
    v, err = bt.btc_ai.parse('{"stop_pct": 2.5, "tp_pct": 4, "reason": "ATR 기준"}')
    assert err == "" and v == {"stop_pct": 2.5, "tp_pct": 4.0, "reason": "ATR 기준"}
    v, _ = bt.btc_ai.parse('```json\n{"stop_pct": 3, "tp_pct": null, "reason": "x"}\n```')
    assert v["tp_pct"] is None


@pytest.mark.parametrize(
    "text,why",
    [
        ("손절 3%가 좋겠어요", "JSON 아님"),
        ('[{"stop_pct": 3}]', "객체 아님"),
        ('{"stop_pct": 3, "reason": "x"}', "tp_pct 없음"),
        ('{"stop_pct": "3", "tp_pct": null, "reason": "x"}', "stop_pct 숫자 아님"),
        ('{"stop_pct": true, "tp_pct": null, "reason": "x"}', "stop_pct 숫자 아님"),
        ('{"stop_pct": 0.5, "tp_pct": null, "reason": "x"}', "범위 밖"),
        ('{"stop_pct": 6, "tp_pct": null, "reason": "x"}', "범위 밖"),
        ('{"stop_pct": 3, "tp_pct": 12, "reason": "x"}', "범위 밖"),
        ('{"stop_pct": 3, "tp_pct": 0.5, "reason": "x"}', "범위 밖"),
        ('{"stop_pct": 3, "tp_pct": null, "reason": ""}', "reason 없음"),
        ('설명 {"stop_pct": 3, "tp_pct": null, "reason": "x"}', "JSON 아님"),
    ],
)
def test_ai_parse_rejects(text, why):
    v, err = bt.btc_ai.parse(text)
    assert v is None and why in err


def test_ai_decide_fallbacks():
    import subprocess

    def timeout(p):
        raise subprocess.TimeoutExpired("claude", 60)

    def boom(p):
        raise RuntimeError("exit 1")

    for caller, why in ((timeout, "시간 초과"), (boom, "호출 실패"), (lambda p: "", "JSON 아님")):
        out = bt.btc_ai.decide({"side": "long"}, caller)
        assert out["used"]["stop_pct"] == 3.0 and out["used"]["tp_pct"] is None
        assert why in out["fallback"] and out["used"]["reason"].startswith("기본값 사용")


def test_ai_exits_used_for_live_orders(env, monkeypatch):
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 2.0, "tp_pct": 4.0, "reason": "ATR 작아 좁게"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    algos = [x for m, p, x in ex.writes if p == "/fapi/v1/algoOrder"]
    assert [(a["type"], a["triggerPrice"], a["side"]) for a in algos] == [
        ("STOP_MARKET", "98000.0", "SELL"),
        ("TAKE_PROFIT_MARKET", "104000.0", "SELL"),
    ]
    assert all(a["closePosition"] == "true" for a in algos)
    pos = state()["position"]
    assert pos["stop_pct"] == 2.0 and pos["tp"] == 104000.0 and pos["tp_id"].endswith("-t")
    msg = next(m for m in env["sent"] if "롱 진입" in m)
    assert "−2%" in msg and "+4%" in msg and "ATR 작아 좁게" in msg
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "ai_exits"][-1]
    assert ev["parsed"]["stop_pct"] == 2.0 and ev["fallback"] is None and ev["ai_raw"]
    assert "atr14_pct" in env["asked"][0] and "direction_tm" in env["asked"][0]


def test_ai_out_of_range_falls_back_and_is_logged(env, monkeypatch):
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 9, "tp_pct": null, "reason": "넓게"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert state()["position"]["stop"] == 97000.0 and state()["position"]["tp"] is None
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "ai_exits"][-1]
    assert "범위 밖" in ev["fallback"] and ev["used"]["stop_pct"] == 3.0
    assert any("기본값 사용" in m for m in env["sent"])


def test_ai_not_asked_when_halted_or_chasing(env, monkeypatch):
    live(env, base=5000.0)
    with_signal(monkeypatch, "long")
    run(env, FakeExchange())
    assert env["asked"] == []


class SlidingMarket(FakeMarket):
    """AI를 기다리는 사이 가격이 달아난다."""

    def __init__(self, prices):
        super().__init__()
        self.prices = list(prices)

    def price(self):
        return self.prices.pop(0) if len(self.prices) > 1 else self.prices[0]


def test_chase_rechecked_after_ai(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    with_signal(monkeypatch, "short")
    run(env, ex, SlidingMarket([100_000.0, 98_500.0]))
    assert env["asked"]  # 처음엔 한도 안이라 물었다
    assert ex.amt == 0 and state()["position"] is None  # 롱은 닫았고 숏은 건너뜀
    assert any("AI 응답 뒤 다시 봄" in m for m in env["sent"])


def test_tp_fill_cancels_stop(env, monkeypatch):
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 2.0, "reason": "x"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    pos = state()["position"]
    ex.algos[pos["tp_id"]]["algoStatus"] = "FINISHED"
    ex.trades.append(
        {
            "time": ex.t + 9000,
            "side": "SELL",
            "qty": str(ex.amt),
            "price": "102000",
            "realizedPnl": "58",
            "commission": "0.05",
            "commissionAsset": "USDT",
        }
    )
    ex.amt = 0.0
    with_signal(monkeypatch, None)
    run(env, ex)
    c = jsonl("closed.jsonl")[-1]
    assert c["why"] == "tp" and c["exit"] == 102000.0 and c["pnl"] > 0
    assert ex.algos[pos["stop_id"]]["algoStatus"] == "CANCELED"
    assert not bt.entries_blocked() and state()["position"] is None


def test_stop_fill_cancels_tp(env, monkeypatch):
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 2.0, "reason": "x"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    pos = state()["position"]
    ex.trigger_stop(pos["stop_id"], 97_000.0)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert jsonl("closed.jsonl")[-1]["why"] == "stop"
    assert ex.algos[pos["tp_id"]]["algoStatus"] == "CANCELED"


def test_reverse_cancels_stop_and_tp(env, monkeypatch):
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 5.0, "reason": "x"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    old = state()["position"]
    ex.calls.clear()
    with_signal(monkeypatch, "short", bar=BAR + bt.BAR_MS)
    run(env, ex)
    seq = [(m, x.get("clientAlgoId") or x.get("side")) for m, p, x in ex.writes]
    assert seq[:2] == [("DELETE", old["stop_id"]), ("DELETE", old["tp_id"])]
    assert seq[2] == ("POST", "SELL") and ex.writes[2][2]["reduceOnly"] == "true"
    assert ex.algos[old["tp_id"]]["algoStatus"] == "CANCELED"
    assert state()["position"]["side"] == "short"


def test_missing_tp_is_replaced(env, monkeypatch):
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 5.0, "reason": "x"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    tid = state()["position"]["tp_id"]
    del ex.algos[tid]
    with_signal(monkeypatch, None)
    run(env, ex)
    assert state()["position"]["tp_id"].endswith("-t1")
    assert ex.algos[state()["position"]["tp_id"]]["type"] == "TAKE_PROFIT_MARKET"


def test_dry_virtual_tp_and_ai_message(env, monkeypatch):
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 1.5, "reason": "가까운 저항"}'
    ex = FakeExchange()
    with_signal(monkeypatch, "long", bar=BAR - bt.BAR_MS * 3)
    mk = FakeMarket()
    run(env, ex, mk)
    assert any(
        m.startswith("[드라이런] 🟢 롱 진입 예정") and "가까운 저항" in m and "+1.5%" in m
        for m in env["sent"]
    )
    mk.rows[-1][2] = 101_600.0
    with_signal(monkeypatch, None)
    run(env, ex, mk)
    c = jsonl("closed.jsonl")[-1]
    assert c["why"] == "tp" and c["exit"] == pytest.approx(101_500.2) and not ex.writes


# ------------------------------------------ 지정가 사다리 (사용자 2026-11-08 22:48)


def _limits(ex):
    return [x for m, p, x in ex.writes if p == "/fapi/v1/order" and x.get("type") == "LIMIT"]


def _markets(ex):
    return [x for m, p, x in ex.writes if p == "/fapi/v1/order" and x.get("type") == "MARKET"]


def test_fast_market_uses_market(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    mk = FakeMarket()
    t = NOW_MS - NOW_MS % 60000
    mk.mins = [
        [t - 60000, 99_500.0, 100_000.0, 99_500.0, 99_900.0, 1, t - 1],
        [t, 99_900.0, 100_000.0, 99_900.0, 100_000.0, 1, t + 59999],
    ]
    with_signal(monkeypatch, "long")
    run(env, ex, mk)
    assert not _limits(ex) and len(_markets(ex)) == 1
    assert _markets(ex)[0]["newClientOrderId"].endswith("-of")
    assert state()["position"]["fill"] == "fast"
    assert any("시장가(급변동: 1분 움직임" in m for m in env["sent"])


def test_fast_when_already_moved_from_signal(env, monkeypatch):
    live(env)
    ex = FakeExchange(price=100_600.0)
    ex.leverage = 3
    with_signal(monkeypatch, "long", close=100_000.0)
    run(env, ex)
    assert not _limits(ex) and len(_markets(ex)) == 1


def test_limit_filled_first_round(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len(_limits(ex)) == 1 and not _markets(ex)
    pos = state()["position"]
    assert pos["fill"] == "limit" and pos["entry"] == 100_000.0 and pos["qty"] == 0.029
    msg = next(m for m in env["sent"] if "롱 진입" in m)
    assert "지정가 체결(1회차, 메이커)" in msg and "수수료 약 0.58 USDT" in msg


def test_partial_fill_then_reprice_one_tick_higher(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.010", "fill"]
    with_signal(monkeypatch, "long")
    run(env, ex)
    lim = _limits(ex)
    assert [x["quantity"] for x in lim] == ["0.029", "0.019"]
    assert [x["price"] for x in lim] == ["100000.0", "100000.1"]  # 2회차는 한 틱 위
    assert [x["newClientOrderId"][-2:] for x in lim] == ["o1", "o2"]
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.029) and pos["entry"] == pytest.approx(100000.0655)
    assert ex.limits[lim[0]["newClientOrderId"]]["status"] == "CANCELED"


def test_gtx_reject_reprices(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["gtx", "expired", "fill"]
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len(_limits(ex)) == 3 and not _markets(ex)
    assert state()["position"]["fill"] == "limit"
    assert any("지정가 체결(3회차, 메이커)" in m for m in env["sent"])


def test_timeout_sends_remainder_at_market(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.005"] + ["none"] * 10
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert 1 < len(_limits(ex)) <= 6
    m = _markets(ex)
    assert len(m) == 1 and m[0]["quantity"] == "0.024" and m[0]["newClientOrderId"].endswith("-om")
    assert state()["position"]["qty"] == pytest.approx(0.029)
    assert any("지정가 0.005 + 시장가 0.024 BTC(지정가 미체결)" in x for x in env["sent"])
    assert all(o["status"] != "NEW" for o in ex.limits.values())  # 남은 지정가 없음


def test_ladder_respects_time_limit(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"] * 10
    with_signal(monkeypatch, "long")
    run(env, ex)
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "run"][-1]
    assert ev["decision"] == "롱 진입"
    assert len(_limits(ex)) <= 6


class DriftMarket(FakeMarket):
    """사다리 도중 가격이 달아난다."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def book(self):
        self.calls += 1
        px = 100_000.0 if self.calls <= 1 else 101_200.0
        return px, px + 0.2


def test_chase_exceeded_mid_ladder_abandons(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.010", "fill"]
    with_signal(monkeypatch, "long")
    run(env, ex, DriftMarket())
    assert len(_limits(ex)) == 1 and not _markets(ex)
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.010)  # 체결된 만큼만 갖고 손절을 건다
    assert any(a["algoStatus"] == "NEW" for a in ex.algos.values())
    assert any("지정가 일부만 체결" in m and "추격 한도" in m for m in env["sent"])


def test_chase_exceeded_with_nothing_filled_skips(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"]
    with_signal(monkeypatch, "long")
    run(env, ex, DriftMarket())
    assert state().get("position") is None and ex.amt == 0 and not ex.algos
    assert any("진입 안 됨" in m for m in env["sent"])


def test_cancel_fill_race_counts_filled_qty(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["race"]
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len(_limits(ex)) == 1 and not _markets(ex)  # 취소 뒤 다시 세어 더 사지 않는다
    assert ex.amt == pytest.approx(0.029) and state()["position"]["qty"] == pytest.approx(0.029)


def test_tiny_remainder_is_accepted(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.028"] + ["none"] * 10
    monkeypatch.setattr(bt, "filters", lambda c: {**FLT, "min_notional": 150.0})
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len(_limits(ex)) == 1 and not _markets(ex)  # 0.001 BTC(150 USDT 미만)는 버린다
    assert state()["position"]["qty"] == pytest.approx(0.028)


def test_crash_with_resting_limit_is_recovered(env, monkeypatch):
    """지정가를 걸어 둔 채 실행이 죽었다: 다음 실행이 취소하고 체결분을 이어 받아 손절을 건다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.012"]

    def boom(*a, **k):
        raise KeyboardInterrupt  # 프로세스가 죽은 것과 같다(정리 코드도 못 돈다)

    real_cancel = bt.LiveVenue.cancel
    monkeypatch.setattr(bt.LiveVenue, "cancel", boom)
    with_signal(monkeypatch, "long")
    with pytest.raises(KeyboardInterrupt):
        run(env, ex)
    resting = next(iter(ex.limits.values()))
    assert resting["status"] == "PARTIALLY_FILLED" and state()["pending"]
    monkeypatch.setattr(bt.LiveVenue, "cancel", real_cancel)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert resting["status"] == "CANCELED"
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.012) and "pending" not in state()
    assert any(a["algoStatus"] == "NEW" and a["type"] == "STOP_MARKET" for a in ex.algos.values())


def test_dry_ladder_logs_path(env, monkeypatch):
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    run(env, ex)
    d = jsonl("dry.jsonl")[0]
    assert d["path"] in ("limit", "market_fallback") and d["rounds"] >= 1 and d["steps"]
    assert not ex.writes


def test_entry_config_override(env):
    (bt.DIR / "config.json").write_text(
        json.dumps({"entry": {"ladder_rounds": 3, "fast_1m_pct": "x"}})
    )
    cfg = bt.entry_cfg()
    assert cfg["ladder_rounds"] == 3 and cfg["fast_1m_pct"] == 0.3


def test_error_mid_ladder_is_protected_in_same_run(env, monkeypatch):
    """사다리 도중 오류(예: 호가 조회 실패): 같은 실행에서 지정가 취소·포지션 확인·손절."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.012"]

    class Flaky(FakeMarket):
        n = 0

        def book(self):
            self.n += 1
            if self.n > 1:
                raise bt.NetworkError("호가 조회 실패")
            return super().book()

    with_signal(monkeypatch, "long")
    run(env, ex, Flaky())
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.012) and "pending" not in state()
    assert all(
        o["status"] != "NEW" and o["status"] != "PARTIALLY_FILLED" for o in ex.limits.values()
    )
    assert any(a["type"] == "STOP_MARKET" and a["algoStatus"] == "NEW" for a in ex.algos.values())
    assert any("진입 중 오류" in m for m in env["sent"])


def test_one_tick_spread_keeps_resting_order(env, monkeypatch):
    """스프레드가 1틱이면 새 가격이 같다: 취소·재주문 없이 그대로 두어 대기열을 지킨다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"]
    mk = FakeMarket()
    mk.spread = 0.1
    with_signal(monkeypatch, "long")
    run(env, ex, mk)
    assert len(_limits(ex)) == 1  # 한 번만 걸었다
    assert len(_markets(ex)) == 1  # 시간이 다 돼 남은 수량 시장가


def test_uncertain_cancel_stops_ladder(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none", "fill"]
    monkeypatch.setattr(
        bt.LiveVenue, "cancel", lambda self, cid: {"status": "NEW", "executedQty": "0"}
    )
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len(_limits(ex)) == 1 and not _markets(ex)  # 끝 상태를 모르면 더 내지 않는다


def test_position_lag_is_waited_for(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    real = bt.position_row
    calls = {"n": 0}

    def lagging(client):
        calls["n"] += 1
        return {"amt": 0.0, "entry": 0.0} if calls["n"] == 1 else real(client)

    monkeypatch.setattr(bt, "position_row", lagging)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert state()["position"]["qty"] == pytest.approx(0.029) and calls["n"] >= 2


def test_position_unreadable_keeps_pending(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3

    def down(client):
        raise bt.NetworkError("down")

    monkeypatch.setattr(bt, "position_row", down)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert state()["pending"]["open"].endswith("-o") and state().get("position") is None
    assert any("포지션을 확인하지 못했어요" in m for m in env["sent"])
    monkeypatch.undo()


def test_deadline_stops_ladder_without_market(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.010"] + ["none"] * 10
    monkeypatch.setattr(bt, "RUN_DEADLINE_S", 15)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert not _markets(ex) and state()["position"]["qty"] == pytest.approx(0.010)


# ---------------------------------------------------------------- 복구 모드(EC2 재부팅)


def _set_last_run(minutes_ago: float) -> None:
    st = state()
    st["last_run_ms"] = NOW_MS - int(minutes_ago * 60_000)
    bt._save(bt.DIR / "state.json", st)


def test_latest_signal_lookback_counts_missed(monkeypatch):
    r = rows(300, 100.0)
    monkeypatch.setattr(bt.scr_lab, "signals", _fake_signals([280], [290]))
    last = BAR - 40 * bt.BAR_MS
    assert bt.latest_signal(r, NOW_MS, last)["dir"] is None  # 평소 2봉 창 밖
    sig = bt.latest_signal(r, NOW_MS, last, lookback=40)
    assert sig["dir"] == "short" and sig["age"] == 9 and sig["missed"] == 2


def test_run_saves_last_run_and_normal_gap_no_recover(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    _set_last_run(15)
    with_signal(monkeypatch, None)
    s = run(env, ex)
    assert state()["last_run_ms"] == NOW_MS
    assert "recover" not in s and not [m for m in env["sent"] if "복구" in m]


def test_long_gap_runs_recovery_and_reports(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    _set_last_run(180)
    seen = {}

    def fake(r, n, last, lookback=2):
        seen["lookback"] = lookback
        return {
            "dir": "short",
            "bar": BAR - 1 * bt.BAR_MS,
            "close": 100_000.0,
            "age": 1,
            "missed": 3,
            "bar_txt": "11-08 21:15",
        }

    monkeypatch.setattr(bt, "latest_signal", fake)
    s = run(env, ex)
    assert seen["lookback"] == 13  # 180분 / 15 + 1
    assert s["recover"]["gap_min"] == 180 and state()["position"]["side"] == "short"
    msg = next(m for m in env["sent"] if "복구" in m)
    assert "약 180분" in msg and "놓친 신호 3개" in msg and "손절 주문 걸려 있음" in msg


def test_recovery_old_signal_closes_but_does_not_enter(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    _set_last_run(600)
    with_signal(monkeypatch, "short", bar=BAR - 20 * bt.BAR_MS, age=20, missed=1)
    s = run(env, ex)
    assert ex.amt == 0 and state()["position"] is None  # 반대 신호였던 롱은 정리
    assert not [w for w in ex.writes if w[2].get("side") == "SELL" and not w[2].get("reduceOnly")]
    assert "복구" in s["decision"]
    assert any("정리됨" in m for m in env["sent"])


def test_boot_recover_reports_even_short_gap(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    _set_last_run(5)
    with_signal(monkeypatch, None)
    s = bt.run(now=NOW, client=ex, market_=FakeMarket(ex.price), send=env["send"], boot=True)
    assert s["recover"]["gap_min"] == 5
    assert any("복구" in m and "놓친 신호 없음" in m for m in env["sent"])


def test_recover_retries_network_then_succeeds(monkeypatch):
    calls = []

    def fake_run(boot=False, before=None):
        calls.append(boot)
        if len(calls) < 3:
            raise OSError("dns")
        return {"ok": True}

    monkeypatch.setattr(bt, "run", fake_run)
    monkeypatch.setattr(bt, "_uptime", lambda: 0.0)
    monkeypatch.setattr(bt, "_sleep", lambda s: None)
    assert bt.recover() == {"ok": True} and calls == [True, True, True]


def test_recover_does_not_retry_real_rejection(monkeypatch):
    monkeypatch.setattr(bt, "_uptime", lambda: 0.0)
    monkeypatch.setattr(
        bt,
        "run",
        lambda boot=False, before=None: (_ for _ in ()).throw(bt.Rejected(-2019, "margin")),
    )
    monkeypatch.setattr(bt, "_sleep", lambda s: None)
    with pytest.raises(bt.Rejected):
        bt.recover()


def test_recover_skips_if_timer_already_ran_after_boot(env, monkeypatch):
    bt._save(bt.DIR / "state.json", {"last_run_ms": int(bt.time.time() * 1000) - 60_000})
    monkeypatch.setattr(bt, "_uptime", lambda: 600.0)
    monkeypatch.setattr(
        bt, "run", lambda boot=False, before=None: pytest.fail("run 호출 안 돼야 함")
    )
    assert "skipped" in bt.recover()


def test_recovery_three_hour_old_signal_does_not_enter(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    _set_last_run_raw(NOW_MS - 300 * 60_000)
    with_signal(monkeypatch, "long", bar=BAR - 12 * bt.BAR_MS, age=12, missed=1)
    s = run(env, ex)
    assert not ex.writes and ex.amt == 0 and "복구" in s["decision"]


def _set_last_run_raw(ms):
    bt._save(bt.DIR / "state.json", {"last_run_ms": ms, "last_bar": BAR - 30 * bt.BAR_MS})


def test_recovery_report_flags_missing_stop(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    _set_last_run(60)
    with_signal(monkeypatch, None)
    monkeypatch.setattr(bt, "ensure_stop", lambda ctx, st, with_tp=True: True)
    ex.algos[pos["stop_id"]]["algoStatus"] = "CANCELED"  # 거래소에 손절이 없다
    monkeypatch.setattr(bt, "_sleep", lambda s: None)
    s = run(env, ex)
    assert s["recover"]["stop_ok"] is False
    assert any("🆘" in m and "손절" in m for m in env["sent"])


def test_recovery_report_over_24h_warns(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    _set_last_run(30 * 60)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert any("24시간을 넘어" in m for m in env["sent"])


def test_recover_gives_up_at_deadline(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(bt, "_uptime", lambda: 0.0)
    monkeypatch.setattr(bt, "_clock", lambda: t["now"])

    def slp(sec):
        t["now"] += sec

    def boom(boot=False, before=None):
        t["now"] += 60
        raise OSError("down")

    monkeypatch.setattr(bt, "_sleep", slp)
    monkeypatch.setattr(bt, "run", boom)
    with pytest.raises(RuntimeError, match="복구 실행"):
        bt.recover()
    assert t["now"] <= bt.RECOVER_DEADLINE_S + 60


def test_recover_does_not_hammer_rate_limit(monkeypatch):
    import urllib.error

    monkeypatch.setattr(bt, "_uptime", lambda: 0.0)
    err = urllib.error.HTTPError("u", 429, "Too Many", {}, None)
    monkeypatch.setattr(bt, "run", lambda boot=False, before=None: (_ for _ in ()).throw(err))
    monkeypatch.setattr(bt, "_sleep", lambda s: pytest.fail("다시 하면 안 됨"))
    with pytest.raises(urllib.error.HTTPError):
        bt.recover()


# ---------------------------------------------------------------- 리팩터링 버그 수정(2026-11-09)


def test_flip_pnl_excludes_previous_close(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    with_signal(monkeypatch, "short")
    run(env, ex)
    pos = state()["position"]
    assert pos["side"] == "short" and pos.get("after_ms")
    old_close = next(t for t in ex.trades if t["side"] == "SELL")
    old_close["realizedPnl"] = "-30"  # 앞 롱의 손실
    ex.trigger_stop(pos["stop_id"], 103_000.0)
    ex.trades[-1]["realizedPnl"] = "10"
    with_signal(monkeypatch, None)
    run(env, ex)
    last = jsonl("closed.jsonl")[-1]
    assert last["side"] == "short" and last["why"] == "stop"
    assert last["pnl"] == pytest.approx(10 - 0.05 - 0.004, abs=0.02)  # 앞 롱의 −30 안 섞임


def test_protect_close_id_is_per_position(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.reject_stop = True
    with_signal(monkeypatch, "long")
    run(env, ex)
    cids = [w[2]["newClientOrderId"] for w in ex.writes if w[2].get("reduceOnly") == "true"]
    assert cids and all(c.startswith(f"mnv-{BAR // 60000}-x") for c in cids)


def test_protect_close_lookup_failure_still_clears_position(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.reject_stop = True
    monkeypatch.setattr(bt, "_close_result", lambda *a: (_ for _ in ()).throw(bt.NetworkError("x")))
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.amt == 0 and state()["position"] is None
    assert jsonl("closed.jsonl")[-1]["why"] == "protect"
    assert not [m for m in env["sent"] if "닫지 못했어요" in m]


def test_protect_close_retry_uses_new_id(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    pos = engine_long(ex)
    st = state()
    ctx = bt.Ctx(ex, FakeMarket(ex.price), env["send"], NOW, True)
    ex.net_fail["POST /fapi/v1/order"] = 4  # 첫 시도는 통신 오류로 못 냄
    monkeypatch.setattr(bt, "_sleep", lambda s: None)
    bt.protect_close(ctx, st, "protect")
    assert st["position"] is not None and st["position"]["x_n"] == 1
    bt.protect_close(ctx, st, "protect")
    assert st["position"] is None and ex.amt == 0
    sent = [w[2]["newClientOrderId"] for w in ex.writes if w[2].get("reduceOnly") == "true"]
    assert sent[-1] == f"mnv-{pos['bar'] // 60000}-x2"


def test_stale_engine_stop_cancelled_before_new_entry(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    old = "mnv-1-s"
    ex.algos[old] = {"clientAlgoId": old, "algoStatus": "NEW", "triggerPrice": "90000"}
    with_signal(monkeypatch, "long")
    real = {"n": 0}

    def cancel_once(c, keep=None):
        real["n"] += 1
        ex.algos[old]["algoStatus"] = "CANCELED"
        return [old]

    monkeypatch.setattr(bt, "cancel_engine_stops", cancel_once)
    run(env, ex)
    assert ex.algos[old]["algoStatus"] == "CANCELED" and state()["position"]["side"] == "long"


def test_entry_skipped_if_stale_stop_cannot_be_cancelled(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    calls = {"n": 0}

    def flaky(c, keep=None):
        calls["n"] += 1
        if calls["n"] >= 2:  # 대사 때는 되고 진입 직전에는 실패
            raise bt.NetworkError("down")
        return []

    monkeypatch.setattr(bt, "cancel_engine_stops", flaky)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.amt == 0 and state().get("position") is None
    assert any("남은 엔진 손절을 취소하지 못함" in m for m in env["sent"])


# ---------------------------------------------------------------- 엔진 버그 수정(2026-11-11)


def test_stop_fill_seen_in_reconcile_not_counted_in_new_position(env, monkeypatch):
    """실행 시작 직후 체결된 손절이 같은 실행의 새 포지션 손익에 섞였다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    ex.trigger_stop(pos["stop_id"], 97_000.0)  # 체결 시각이 이번 실행 시작(NOW) 뒤
    ex.t += 6000  # 이후 체결은 손절보다 늦다
    with_signal(monkeypatch, "short")
    run(env, ex)
    short = state()["position"]
    assert short["side"] == "short" and short.get("after_ms")
    ex.trigger_stop(short["stop_id"], 103_000.0)
    ex.trades[-1]["realizedPnl"] = "10"
    with_signal(monkeypatch, None)
    run(env, ex)
    last = jsonl("closed.jsonl")[-1]
    assert last["side"] == "short" and last["why"] == "stop"
    assert last["pnl"] == pytest.approx(10 - 0.05 - 0.004, abs=0.02)  # 앞 롱 손절 −30 안 섞임


def test_protect_close_marks_close_time_for_next_entry(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    st = state()
    ctx = bt.Ctx(ex, FakeMarket(ex.price), env["send"], NOW, True)
    bt.protect_close(ctx, st, "protect")
    assert st["position"] is None
    assert st["closed_through_ms"] == max(t["time"] for t in ex.trades if t["side"] == "SELL")


def test_adopted_entry_keeps_entry_fee_in_pnl_window(env, monkeypatch):
    """이어 받은 진입의 손익 창이 이어 받은 시각부터라 진입 체결(수수료)이 빠졌다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.012"]

    def boom(*a, **k):
        raise KeyboardInterrupt

    real_cancel = bt.LiveVenue.cancel
    monkeypatch.setattr(bt.LiveVenue, "cancel", boom)
    with_signal(monkeypatch, "long")
    with pytest.raises(KeyboardInterrupt):
        run(env, ex)
    monkeypatch.setattr(bt.LiveVenue, "cancel", real_cancel)
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    entry_fill = next(t for t in ex.trades if t["side"] == "BUY")
    assert pos["entry_ms"] < entry_fill["time"]
    ex.trigger_stop(pos["stop_id"], 97_000.0)
    ex.trades[-1]["realizedPnl"] = "-10"
    run(env, ex)
    last = jsonl("closed.jsonl")[-1]
    assert last["why"] == "stop"
    assert last["pnl"] == pytest.approx(-10 - 0.05 - 0.004, abs=1e-6)  # 진입 수수료 포함


def test_protect_close_lost_response_then_reduce_only_reject_is_closed(env, monkeypatch):
    """비상 청산 응답을 잃고 다시 보낸 reduceOnly가 '포지션 없음'으로 거부돼도 닫힌 것은 닫힌 것."""
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    st = state()
    ctx = bt.Ctx(ex, FakeMarket(ex.price), env["send"], NOW, True)
    ex.lose_response = {"POST /fapi/v1/order"}
    real_query, real_handle = bt._query, ex._handle

    def lagging(client, path, key, cid):
        if cid.endswith("-x1"):
            return None  # 체결된 주문이 아직 조회에 안 보인다
        return real_query(client, path, key, cid)

    def handle(method, path, p):
        if method == "POST" and p.get("reduceOnly") == "true" and ex.amt == 0:
            raise bt.Rejected(-2022, "ReduceOnly Order is rejected.")
        return real_handle(method, path, p)

    monkeypatch.setattr(bt, "_query", lagging)
    monkeypatch.setattr(ex, "_handle", handle)
    bt.protect_close(ctx, st, "protect")
    assert ex.amt == 0 and st["position"] is None
    assert jsonl("closed.jsonl")[-1]["why"] == "protect"
    assert not [m for m in env["sent"] if "닫지 못했어요" in m]


def test_retried_signal_uses_fresh_entry_ids(env, monkeypatch):
    """pending이 버려진 뒤 같은 신호를 다시 시도하면 옛 진입 주문을 찾아 쓰지 않고 새 ID로 낸다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"]

    def boom(*a, **k):
        raise KeyboardInterrupt

    real_cancel = bt.LiveVenue.cancel
    monkeypatch.setattr(bt.LiveVenue, "cancel", boom)
    with_signal(monkeypatch, "long")
    with pytest.raises(KeyboardInterrupt):
        run(env, ex)
    first = set(ex.limits)
    assert state()["pending"]
    monkeypatch.setattr(bt.LiveVenue, "cancel", real_cancel)
    ex.limit_script = ["fill"]
    run(env, ex)  # 대사가 옛 지정가를 취소하고 pending을 버린 뒤 같은 신호로 다시 들어간다
    second = set(ex.limits) - first
    assert second and all(c.startswith(f"mnv-{BAR // 60000}-o2-") for c in second)
    pos = state()["position"]
    assert pos["side"] == "long" and pos["qty"] == pytest.approx(0.029)
    assert all(ex.limits[c]["status"] == "CANCELED" for c in first)


def test_retried_signal_uses_fresh_stop_and_tp_ids(env, monkeypatch):
    """앞 시도의 손절이 이미 체결된 뒤 같은 신호로 다시 들어가면, 옛 손절 ID를 찾아 '손절 실패'로
    새 포지션을 바로 닫았다(레드팀 2026-11-11)."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 2.0, "reason": "익절 있음"}'
    with_signal(monkeypatch, "long")
    run(env, ex)
    first = state()["position"]
    assert first["stop_id"] == f"mnv-{BAR // 60000}-s"
    st = state()
    st["last_bar"] = BAR - bt.BAR_MS  # last_bar를 적기 전에 실행이 죽은 것과 같다
    bt._save(bt.DIR / "state.json", st)
    ex.trigger_stop(first["stop_id"], 97_000.0)
    ex.t += 6000
    run(env, ex)  # 손절을 대사하고 같은 신호로 다시 들어간다(2번째 시도)
    pos = state()["position"]
    m = BAR // 60000
    assert pos and pos["side"] == "long" and ex.amt > 0
    assert pos["stop_id"] == f"mnv-{m}-s2-" and pos["tp_id"] == f"mnv-{m}-t2-"
    assert ex.algos[pos["stop_id"]]["algoStatus"] == "NEW"
    assert ex.algos[pos["tp_id"]]["algoStatus"] == "NEW"
    assert "protect" not in [c["why"] for c in jsonl("closed.jsonl")]
    # 다시 걸 때도 앞 시도의 s1·t1과 겹치지 않는다
    ex.algos[pos["stop_id"]]["algoStatus"] = "CANCELED"
    ex.algos[pos["tp_id"]]["algoStatus"] = "CANCELED"
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    assert pos["stop_id"] == f"mnv-{m}-s2-1" and pos["tp_id"] == f"mnv-{m}-t2-1"


def test_last_bar_saved_with_new_position(env, monkeypatch):
    """포지션을 저장할 때 신호도 처리한 것으로 적는다(그 뒤에 죽어도 같은 신호로 다시 안 들어감)."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3

    def crash(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(bt, "place_stop", crash)
    with_signal(monkeypatch, "long")
    with pytest.raises(KeyboardInterrupt):
        run(env, ex)
    st = state()
    assert st["position"]["side"] == "long" and st["last_bar"] == BAR
