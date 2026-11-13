"""BTCUSDT 자동매매 엔진 테스트. 바이낸스는 전부 가짜(네트워크 없음)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
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
        self.reject_code = -2021
        self.child_conflict = False  # 걸린 손절·익절이 남긴 주문이 새 reduceOnly를 막게(-2022)
        self.lose_response: set[str] = set()  # 실행은 됐는데 응답만 잃는 경로
        self.net_fail: dict[str, int] = {}  # 실행 전 통신 오류 횟수
        self.foreign_open: list[dict] = []
        self.t = NOW_MS
        # 지정가 회차마다 꺼내 쓰는 각본: fill / none / partial:<수량> / gtx / expired / race
        self.limit_script: list[str] = []
        # reduceOnly(청산) 지정가 각본(진입 각본과 따로): fill / none / partial:<수량> / race
        self.close_script: list[str] = []
        self.limits: dict[str, dict] = {}
        self._oid = 1000

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
            entry = getattr(self, "entry_px", None) or self.__dict__.get("avgs", {}).get(
                "BTCUSDT", self.price
            )
            return [{"symbol": "BTCUSDT", "positionAmt": str(self.amt), "entryPrice": str(entry)}]
        if path == "/fapi/v1/openAlgoOrders":
            return [a for a in self.algos.values() if a["algoStatus"] == "NEW"] + [
                o for o in self.foreign_open if "clientAlgoId" in o
            ]
        if path == "/fapi/v1/openOrders":
            mine = [
                {
                    "clientOrderId": cid,
                    "orderId": o["orderId"],
                    "reduceOnly": o["reduce"],
                    "type": o.get("type", "LIMIT"),
                    "origType": o.get("type", "LIMIT"),
                }
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
        if path == "/fapi/v1/forceOrders":
            return list(getattr(self, "force_orders", []))
        if path == "/fapi/v1/allOrders":
            return [
                {"clientOrderId": o["clientOrderId"], "executedQty": o.get("executedQty", "0")}
                for o in list(self.limits.values()) + list(self.orders.values())
            ]
        if path == "/fapi/v1/order" and method == "GET":
            cid = p["origClientOrderId"]
            if cid in self.limits:
                return self._limit_status(cid)
            o = self.orders.get(cid)
            if not o:
                raise bt.Rejected(-2013, "Order does not exist.")
            return o
        if path == "/fapi/v1/order" and method == "POST" and p["type"] == "MARKET":
            # 시장가는 진입 남은 수량(ID 끝 m)만(사용자 2026-11-13 20:17). 청산·손절·익절은 금지
            assert p.get("reduceOnly") != "true", "청산 시장가 금지"
            assert p["newClientOrderId"].endswith("m"), p["newClientOrderId"]
            qty = float(p["quantity"])
            sign = 1 if p["side"] == "BUY" else -1
            self._book(sign, qty, self.price)
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
        if path == "/fapi/v1/order" and method == "POST" and p["type"] == "LIMIT":
            assert p["timeInForce"] in ("GTX", "GTC")
            reduce = p.get("reduceOnly") == "true"
            if reduce:
                assert p["timeInForce"] == "GTC"
                if self.amt == 0:
                    raise bt.Rejected(-2022, "ReduceOnly Order is rejected.")
                if any(
                    o.get("conflict") and o["status"] in ("NEW", "PARTIALLY_FILLED")
                    for o in self.limits.values()
                ):
                    raise bt.Rejected(-2022, "ReduceOnly Order is rejected.")
                script = self.close_script
            else:
                script = self.limit_script
            how = script.pop(0) if script else "fill"
            if how == "gtx":
                raise bt.Rejected(-5022, "Post Only order will be rejected")
            if how == "margin":
                raise bt.Rejected(-2019, "Margin is insufficient.")
            o = {
                "clientOrderId": p["newClientOrderId"],
                "status": "NEW",
                "executedQty": "0",
                "avgPrice": "0",
                "price": p["price"],
                "origQty": p["quantity"],
                "side": p["side"],
                "how": how,
                "reduce": reduce,
                "tif": p["timeInForce"],
                "orderId": self._next_oid(),
            }
            if how == "expired":
                o["status"] = "EXPIRED"
            self.limits[p["newClientOrderId"]] = o
            if how == "fill" and p["timeInForce"] == "GTC":
                self._fill(o, float(o["origQty"]))  # 닿는 가격의 GTC는 내자마자 체결된다
            return {k: v for k, v in o.items() if k != "how"}
        if path == "/fapi/v1/order" and method == "DELETE":
            if "orderId" in p:
                o = next(
                    (x for x in self.limits.values() if str(x["orderId"]) == str(p["orderId"])),
                    None,
                )
            else:
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
        if path == "/fapi/v1/algoOrder" and method == "POST":
            if self.reject_stop:
                raise bt.Rejected(self.reject_code, "Order would immediately trigger.")
            if p["type"] in ("STOP", "TAKE_PROFIT"):
                # 지정가 조건부: 지정가·수량·reduceOnly·GTC, closePosition 없음
                assert p["price"] and float(p["quantity"]) > 0, p
                assert p["reduceOnly"] == "true" and p["timeInForce"] == "GTC", p
                assert "closePosition" not in p, p
            else:
                assert p["type"] in ("STOP_MARKET", "TAKE_PROFIT_MARKET"), p
                assert p["closePosition"] == "true", p
            a = {
                "clientAlgoId": p["clientAlgoId"],
                "algoStatus": "NEW",
                "triggerPrice": p["triggerPrice"],
                "side": p["side"],
                "type": p["type"],
                **{k: p[k] for k in ("price", "quantity", "reduceOnly", "closePosition") if k in p},
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

    def _next_oid(self) -> int:
        self._oid += 1
        return self._oid

    def _book(self, sign: int, q: float, px: float) -> None:
        """체결을 포지션에 더하고 거래소처럼 평균 진입가를 갱신한다."""
        sym = getattr(self, "_cur", "BTCUSDT")
        avgs = self.__dict__.setdefault("avgs", {})
        old = self.amt
        new = round(old + sign * q, 6)
        if new == 0:
            avgs.pop(sym, None)
        elif old == 0 or (old > 0) != (new > 0):
            avgs[sym] = px  # 새로 열었거나 뒤집혔다: 남은 수량은 이번 가격
        elif abs(new) > abs(old):
            avgs[sym] = (abs(old) * avgs.get(sym, px) + q * px) / abs(new)
        self.amt = new

    def _fill(self, o: dict, q: float) -> None:
        done = float(o["executedQty"])
        q = min(q, float(o["origQty"]) - done)
        if o.get("reduce"):
            q = min(q, abs(self.amt))  # reduceOnly는 포지션보다 많이 체결되지 않는다
        if q <= 0:
            return
        px = float(o["price"])
        sign = 1 if o["side"] == "BUY" else -1
        self._book(sign, q, px)
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

    def trigger_unfilled(self, cid: str, partial: float = 0.0) -> str:
        """지정가 손절·익절이 걸렸는데(갭) 남긴 지정가가 체결 안 됨(또는 partial만). 남긴 주문은
        엔진 밖 ID(거래소가 붙임)이고 algoOrder 조회의 actualOrderId로 찾는다."""
        a = self.algos[cid]
        a["algoStatus"] = "TRIGGERED"
        oid = self._next_oid()
        a["actualOrderId"] = str(oid)
        child = {
            "clientOrderId": f"x-auto-{oid}",
            "status": "NEW",
            "executedQty": "0",
            "avgPrice": "0",
            "price": a.get("price", a["triggerPrice"]),
            "origQty": a.get("quantity", str(abs(self.amt))),
            "side": a["side"],
            "how": "none",
            "reduce": True,
            "tif": "GTC",
            "type": a["type"],
            "orderId": oid,
            "conflict": self.child_conflict,
        }
        self.limits[child["clientOrderId"]] = child
        if partial:
            self._fill(child, partial)
        return child["clientOrderId"]

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
    # run()이 universe.json으로 바꾸는 모듈 값: 시험이 끝나면 되돌린다
    for name in ("SYMBOLS", "SYMBOL_WINRATE", "PRIORITY", "UNIVERSE"):
        monkeypatch.setattr(bt, name, getattr(bt, name))
    monkeypatch.setattr(bt, "_TAGS", {})
    # 다른 흐름 시험은 ETH·SOXL로도 들어간다. 승률 바닥은 따로 시험한다
    monkeypatch.setattr(bt, "MIN_ENTRY_WINRATE", 0.0)
    # 수익 조건도 따로 시험한다
    monkeypatch.setattr(bt, "DEFAULT_RET", dict.fromkeys(bt.DEFAULT_SYMBOLS, 1.0))
    monkeypatch.setattr(bt, "MANUAL_ADOPT", False)  # 옛 "수동 개입이면 멈춤" 시험. 이어받기는 따로
    monkeypatch.setattr(bt, "FEE_BUFFER", 0.01)  # 기존 수량 계산 시험값. 3%는 따로 시험
    monkeypatch.setattr(bt, "AI_EXITS", True)  # AI 손절·익절 흐름 시험. 고정 규칙은 따로 시험
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


def engine_long(ex: FakeExchange, qty=0.029, entry=100_000.0, style: str = "") -> dict:
    """엔진이 연 롱(손절 포함)을 거래소와 상태에 만든다. style을 안 주면 지금 주문 방식."""
    ex.amt = qty
    sid = f"mnv-{(BAR - bt.BAR_MS * 8) // 60000}-s"
    ex.algos[sid] = {
        "clientAlgoId": sid,
        "algoStatus": "NEW",
        "triggerPrice": "97000.0",
        "type": "STOP",
        "side": "SELL",
        "price": "96515.0",
        "quantity": str(qty),
    }
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
        "order_style": style or bt.order_style(),
    }
    bt._save(bt.DIR / "state.json", {"position": pos, "last_bar": BAR - bt.BAR_MS * 8})
    return pos


# ---------------------------------------------------------------- 수량·가격


def test_qty_rounds_down_to_step():
    qty, why = bt.size_qty(1000.0, 76_041.4, 1.0, FLT)
    # 1000 × 0.97 × 3 / 76041.4 = 0.03826… → 0.038
    assert qty == Decimal("0.038") and why == ""
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
    # 지정가 손절: 손절가 97000에 걸리면 0.5% 아래 96515까지 매도 지정가(수량·reduceOnly)
    assert stop["type"] == "STOP" and "closePosition" not in stop
    assert stop["price"] == "96515.0" and stop["quantity"] == "0.029"
    assert stop["reduceOnly"] == "true" and stop["timeInForce"] == "GTC"
    assert stop["side"] == "SELL" and stop["triggerPrice"] == "97000.0"
    assert stop["workingType"] == "MARK_PRICE" and stop["algoType"] == "CONDITIONAL"
    pos = state()["position"]
    assert pos["side"] == "long" and pos["qty"] == 0.029 and pos["stop"] == 97000.0
    assert s["decision"] == "BTC 롱 진입" and ex.leverage == 3


def test_entry_price_is_exchange_average(env, monkeypatch):
    # 사용자 2026-11-14 "3번": 엔진이 센 값이 아니라 거래소 평균가로 손절·익절을 잡는다
    live(env)
    ex = FakeExchange()
    ex.entry_px = 99_900.0  # 거래소가 알려 주는 평균가(엔진이 센 체결가 100,000과 다름)
    with_signal(monkeypatch, "long")
    run(env, ex)
    pos = state()["position"]
    assert pos["entry"] == 99_900.0 and pos["stop"] == 96_903.0  # 99,900 × 0.97
    stop = [w[2] for w in ex.writes if w[1] == "/fapi/v1/algoOrder"][0]
    assert stop["triggerPrice"] == "96903.0"
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "entry_from_exchange"]
    assert ev and ev[0]["exchange"] == 99_900.0


def test_reverse_sequence_long_to_short(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    old = engine_long(ex)
    with_signal(monkeypatch, "short")
    run(env, ex)
    seq = [(m, p, x.get("side"), x.get("reduceOnly")) for m, p, x in ex.writes]
    assert seq == [
        ("POST", "/fapi/v1/order", "SELL", "true"),  # 1) 지정가 reduceOnly 청산(손절은 둔 채)
        ("DELETE", "/fapi/v1/algoOrder", None, None),  # 2) 닫힌 뒤 엔진 손절 취소
        ("POST", "/fapi/v1/order", "SELL", None),  # 3) 지정가 숏 진입
        ("POST", "/fapi/v1/algoOrder", "BUY", "true"),  # 4) 숏 지정가 손절(reduceOnly)
    ]
    close = ex.writes[0][2]
    assert close["quantity"] == "0.029" and close["type"] == "LIMIT"
    assert close["timeInForce"] == "GTC" and close["price"] == "100000.0"  # 매수 1호가에 매도
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
    px = Decimal("100000")
    with pytest.raises(bt.NetworkError):
        bt.limit_order(ex, "SELL", Decimal("0.01"), px, "mnv-1-c", "GTC", reduce=True)
    assert len([c for c in ex.calls if c[0] == "POST"]) == 4  # 청산: 처음 1 + 재시도 3
    ex.calls.clear()
    with pytest.raises(bt.NetworkError):
        bt.limit_order(ex, "BUY", Decimal("0.01"), px, "mnv-1-o", "GTX")
    assert len([c for c in ex.calls if c[0] == "POST"]) == 4  # post-only는 바로 체결 안 됨
    ex.calls.clear()
    with pytest.raises(bt.NetworkError):
        bt.limit_order(ex, "BUY", Decimal("0.01"), px, "mnv-1-ot1", "GTC")
    assert len([c for c in ex.calls if c[0] == "POST"]) == 1  # 바로 체결되는 진입은 다시 안 보냄


def test_rejection_not_retried(env):
    ex = FakeExchange()
    ex.reject_stop = True
    with pytest.raises(bt.Rejected):
        bt.place_stop(ex, "long", Decimal("97000"), "mnv-1-s", qty=0.01)
    assert len([c for c in ex.calls if c[0] == "POST"]) == 1
    with pytest.raises(ValueError):  # 지정가 손절은 수량 없이 내지 않는다
        bt.place_stop(ex, "long", Decimal("97000"), "mnv-1-s2")


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
    assert len(posts) == 1 and posts[0]["type"] == "STOP" and ex.amt == 0.05
    assert posts[0]["quantity"] == "0.05"  # 거래소 수량으로 건다


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
    assert any(m.startswith("[드라이런] 🟢 BTC 롱 진입 예정") for m in env["sent"])
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
    assert [(a["type"], a["triggerPrice"], a["price"], a["side"]) for a in algos] == [
        ("STOP", "98000.0", "97510.0", "SELL"),
        ("TAKE_PROFIT", "104000.0", "104000.0", "SELL"),  # 익절은 익절가 지정가
    ]
    assert all(a["reduceOnly"] == "true" and a["quantity"] == "0.029" for a in algos)
    pos = state()["position"]
    assert pos["stop_pct"] == 2.0 and pos["tp"] == 104000.0 and pos["tp_id"].endswith("-t")
    msg = next(m for m in env["sent"] if "롱 진입" in m)
    assert "−2%" in msg and "+4%" in msg and "ATR 작아 좁게" in msg
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "ai_exits"][-1]
    assert ev["parsed"]["stop_pct"] == 2.0 and ev["fallback"] is None and ev["ai_raw"]
    assert "atr14_pct" in env["asked"][0] and "direction_tm" in env["asked"][0]


def test_fixed_exits_skip_ai_and_use_3pct_tp3(env, monkeypatch):
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    AI_ANSWER["text"] = '{"stop_pct": 1.0, "tp_pct": 1.0, "reason": "좁게"}'
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert env["asked"] == []  # AI에 묻지 않는다
    algos = [x for m, p, x in ex.writes if p == "/fapi/v1/algoOrder"]
    assert [(a["type"], a["triggerPrice"]) for a in algos] == [
        ("STOP", "97000.0"),
        ("TAKE_PROFIT", "103000.0"),
    ]
    pos = state()["position"]
    assert pos["stop"] == 97000.0 and pos["stop_pct"] == 3.0
    assert pos["tp"] == 103000.0 and pos["tp_pct"] == 3.0
    assert not any("기본값 사용" in m for m in env["sent"])
    assert any("⚙️ 고정 규칙(손절 3%·익절 +3%, 증거금 약 +9%)" in m for m in env["sent"])
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "ai_exits"][-1]
    assert ev["fixed"] is True and ev["used"]["tp_pct"] == 3.0


def test_fixed_tp_backfilled_once_on_open_position(env, monkeypatch):
    """고정 익절 전에 연 포지션(익절 없음)에 다음 실행에서 +3% 익절을 한 번 건다."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    assert pos["tp"] == 103000.0 and pos["tp_backfilled"] and pos["tp_id"].endswith("-t1")
    assert ex.algos[pos["tp_id"]]["type"] == "TAKE_PROFIT"
    assert ex.algos[pos["tp_id"]]["quantity"] == "0.029"
    assert any("기본 익절 +3%" in m for m in env["sent"])
    # 익절이 실패해 지워져도 다시 하지 않는다
    ex.algos.pop(pos["tp_id"])
    s = state()
    s["position"].update(tp=None, tp_id=None)
    bt._save(bt.DIR / "state.json", s)
    n = len(ex.algos)
    run(env, ex)
    assert state()["position"]["tp"] is None and len(ex.algos) == n


def _engine_long_tp10(ex: FakeExchange) -> dict:
    """익절 +10%(옛 값)이 걸린 엔진 롱."""
    pos = engine_long(ex)
    tid = "mnv-old-t1"
    ex.algos[tid] = {
        "clientAlgoId": tid,
        "algoStatus": "NEW",
        "triggerPrice": "110000.0",
        "type": "TAKE_PROFIT",
        "price": "110000.0",
        "quantity": "0.029",
    }
    pos.update(tp=110000.0, tp_id=tid, tp_pct=10.0, tp_n=1, tp_backfilled=True)
    bt._save(bt.DIR / "state.json", {"position": pos, "last_bar": BAR - bt.BAR_MS * 8})
    return pos


def test_fixed_tp_retargets_open_position_once(env, monkeypatch):
    """익절 값이 10%→3%로 바뀌면 보유 포지션의 옛 익절을 취소하고 새 값으로 건다(한 번만)."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    _engine_long_tp10(ex)
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    assert ex.algos["mnv-old-t1"]["algoStatus"] == "CANCELED"
    assert pos["tp"] == 103000.0 and pos["tp_pct"] == 3.0
    assert ex.algos[pos["tp_id"]]["triggerPrice"] == "103000.0" and pos["tp_id"] != "mnv-old-t1"
    assert any("익절을 +10%" in m and "+3%" in m for m in env["sent"])
    # 다음 실행은 그대로
    n, cancels = len(ex.algos), sum(1 for m, p, _ in ex.writes if m == "DELETE")
    run(env, ex)
    assert len(ex.algos) == n and sum(1 for m, p, _ in ex.writes if m == "DELETE") == cancels


def test_fixed_tp_retarget_past_price_takes_profit_with_limit(env, monkeypatch):
    """새 익절가를 이미 지났으면(-2021) 지정가 추격으로 익절하고 '익절'로 기록한다."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    _engine_long_tp10(ex)
    ex.reject_stop = True  # 새 익절 주문은 바로 걸릴 값이라 거부
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0
    assert state()["position"] is None
    assert any("지정가로 익절" in m for m in env["sent"])
    closes = [x for m, p, x in ex.writes if x.get("reduceOnly") == "true" and p == "/fapi/v1/order"]
    assert closes and all(x["type"] == "LIMIT" and x["timeInForce"] == "GTC" for x in closes)
    assert closes[0]["price"] == "100000.0"  # 매도는 매수 1호가
    assert jsonl("closed.jsonl")[-1]["why"] == "tp"
    assert not (bt.DIR / "HALT").exists()


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
    # 손절·익절은 지정가 청산이 다 될 때까지 두고(reduceOnly라 겹쳐도 뒤집히지 않음), 닫힌 뒤 취소
    assert seq[0] == ("POST", "SELL") and ex.writes[0][2]["reduceOnly"] == "true"
    assert seq[1:3] == [("DELETE", old["stop_id"]), ("DELETE", old["tp_id"])]
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
    assert ex.algos[state()["position"]["tp_id"]]["type"] == "TAKE_PROFIT"


def test_dry_virtual_tp_and_ai_message(env, monkeypatch):
    AI_ANSWER["text"] = '{"stop_pct": 3.0, "tp_pct": 1.5, "reason": "가까운 저항"}'
    ex = FakeExchange()
    with_signal(monkeypatch, "long", bar=BAR - bt.BAR_MS * 3)
    mk = FakeMarket()
    run(env, ex, mk)
    assert any(
        m.startswith("[드라이런] 🟢 BTC 롱 진입 예정") and "가까운 저항" in m and "+1.5%" in m
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


def _touches(ex):
    """바로 체결되는 진입 지정가(GTC, reduceOnly 아님)."""
    return [
        x for x in _limits(ex) if x.get("timeInForce") == "GTC" and x.get("reduceOnly") != "true"
    ]


def test_fast_move_starts_at_touch_limit_not_market(env, monkeypatch):
    """급변동이면 post-only를 건너뛰고 바로 체결되는 가장 가까운 지정가(롱 = 매도 1호가)부터.
    시장가는 내지 않는다(사용자 2026-11-13 "모든 매매는 지정가로 해")."""
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
    assert not _markets(ex)
    lim = _limits(ex)
    assert len(lim) == 1 and lim[0]["timeInForce"] == "GTC" and lim[0]["price"] == "100000.2"
    assert lim[0]["newClientOrderId"].endswith("-ot1")
    assert state()["position"]["fill"] == "touch"
    assert any("급변동(1분 움직임" in m and "바로 체결되는 지정가(1번)" in m for m in env["sent"])


def test_fast_when_already_moved_from_signal(env, monkeypatch):
    live(env)
    ex = FakeExchange(price=100_600.0)
    ex.leverage = 3
    with_signal(monkeypatch, "long", close=100_000.0)
    run(env, ex)
    assert not _markets(ex) and len(_touches(ex)) == 1 and len(_limits(ex)) == 1


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


def test_timeout_takes_remainder_with_touch_limit(env, monkeypatch):
    """post-only로 다 안 되면 남은 수량을 시장가가 아니라 매도 1호가 지정가(GTC)로."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.005", "none", "fill"]
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert not _markets(ex)
    t = _touches(ex)
    assert len(t) == 1 and t[0]["quantity"] == "0.024" and t[0]["price"] == "100000.2"
    assert t[0]["newClientOrderId"].endswith("-ot1")
    assert state()["position"]["qty"] == pytest.approx(0.029)
    assert any("지정가 0.005(메이커) + 바로 체결되는 지정가 0.024개(1번)" in x for x in env["sent"])
    assert all(o["status"] != "NEW" for o in ex.limits.values())  # 남은 지정가 없음


def test_touch_limit_reprices_until_filled(env, monkeypatch):
    """바로 체결되는 지정가가 안 되면 몇 초마다 취소하고 그때 호가에 다시 건다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = [
        "none",
        "none",
        "none",
        "fill",
    ]  # GTX 1·2회차 → 닿는 가격 1회 안 됨 → 2회 체결
    with_signal(monkeypatch, "long")
    run(env, ex)
    t = _touches(ex)
    assert [x["newClientOrderId"][-3:] for x in t] == ["ot1", "ot2"]
    assert ex.limits[t[0]["newClientOrderId"]]["status"] == "CANCELED"
    assert state()["position"]["qty"] == pytest.approx(0.029) and not _markets(ex)


class JumpMarket(FakeMarket):
    """지정가를 쫓는 사이(호가 조회 jump_after번 뒤) 현재가가 추격 한도 밖으로 달아난다.
    호가는 그대로라 지정가 단계는 계속된다."""

    def __init__(self, jump_after=10, jumped=102_000.0):
        super().__init__()
        self.books, self.jump_after, self.jumped = 0, jump_after, jumped

    def book(self):
        self.books += 1
        return super().book()

    def price(self):
        return self.jumped if self.books > self.jump_after else self._price


def test_nothing_fills_then_market_after_touch_time(env, monkeypatch):
    """사용자 2026-11-13 20:17 "60초 따라가다가 안되면 시장가로 하자": 지정가가 하나도 안 되면
    바로 체결되는 지정가를 60초 쫓은 뒤 남은 수량을 시장가로(ID -om). 지정가는 남기지 않는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"] * 200
    with_signal(monkeypatch, "long")
    run(env, ex)
    m = _markets(ex)
    assert len(m) == 1 and m[0]["quantity"] == "0.029" and m[0]["newClientOrderId"].endswith("-om")
    assert len(_touches(ex)) >= 2
    assert all(o["status"] != "NEW" for o in ex.limits.values())
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.029) and pos["fill"] == "market_fallback"
    assert ex.algos[pos["stop_id"]]["type"] == "STOP"
    assert any("시장가 0.029개" in x for x in env["sent"])


def test_market_fallback_skipped_beyond_chase_limit(env, monkeypatch):
    """시장가 직전 가격이 추격 한도 밖이면 시장가를 내지 않고 진입을 건너뛴다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"] * 200
    with_signal(monkeypatch, "long")
    run(env, ex, JumpMarket())
    assert not _markets(ex) and ex.amt == 0 and not state().get("position")
    assert all(o["status"] != "NEW" for o in ex.limits.values())
    assert any("진입 안 됨" in m and "추격 한도 밖" in m for m in env["sent"])


def test_partial_fill_kept_with_stop_for_filled_qty(env, monkeypatch):
    """일부만 체결되고 시장가는 못 내면(추격 한도 밖) 그 수량만 갖고 손절·익절도 그 수량으로."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["partial:0.010"] + ["none"] * 200
    with_signal(monkeypatch, "long")
    run(env, ex, JumpMarket())
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.010) and ex.amt == pytest.approx(0.010)
    assert ex.algos[pos["stop_id"]]["quantity"] == "0.01"
    assert not _markets(ex) and all(o["status"] != "NEW" for o in ex.limits.values())
    assert any("일부만 체결" in m for m in env["sent"])


def test_lost_market_entry_response_not_resent(env, monkeypatch):
    """진입 시장가 응답을 잃어도 다시 보내지 않고, 포지션으로 이어 받아 손절을 건다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"] * 200
    real = ex._handle

    def lose_market(method, path, p):
        out = real(method, path, p)
        if method == "POST" and p.get("type") == "MARKET":
            raise bt.NetworkError("response lost")
        return out

    ex._handle = lose_market
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert len(_markets(ex)) == 1  # 다시 안 보냄
    pos = state()["position"]
    assert pos["qty"] == pytest.approx(0.029) and ex.amt == pytest.approx(0.029)
    assert any(a["type"] == "STOP" and a["algoStatus"] == "NEW" for a in ex.algos.values())


def test_ladder_respects_time_limit(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["none"] * 10
    with_signal(monkeypatch, "long")
    run(env, ex)
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "run"][-1]
    assert ev["decision"] == "BTC 롱 진입"
    gtx = [x for x in _limits(ex) if x["timeInForce"] == "GTX"]
    assert len(gtx) <= 6 and not _markets(ex)


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
    assert any(
        a["algoStatus"] == "NEW" and a["type"] == "STOP" and a["quantity"] == "0.012"
        for a in ex.algos.values()
    )


def test_dry_ladder_logs_path(env, monkeypatch):
    ex = FakeExchange()
    with_signal(monkeypatch, "long")
    run(env, ex)
    d = jsonl("dry.jsonl")[0]
    assert d["path"] in ("limit", "touch") and d["rounds"] >= 1 and d["steps"]
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
    assert any(a["type"] == "STOP" and a["algoStatus"] == "NEW" for a in ex.algos.values())
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
    gtx = [x for x in _limits(ex) if x["timeInForce"] == "GTX"]
    assert len(gtx) == 1  # 한 번만 걸었다
    assert len(_touches(ex)) == 1 and not _markets(ex)  # 시간이 다 돼 남은 수량은 닿는 지정가


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
    cids = [
        w[2]["newClientOrderId"]
        for w in ex.writes
        if w[2].get("reduceOnly") == "true" and w[1] == "/fapi/v1/order"
    ]
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
    assert st["closing"]["cid"] == f"mnv-{pos['bar'] // 60000}-x1-1"  # 다음에 찾을 수 있게 적음
    bt.protect_close(ctx, st, "protect")
    assert st["position"] is None and ex.amt == 0 and "closing" not in st
    sent = [w[2]["newClientOrderId"] for w in ex.writes if w[2].get("reduceOnly") == "true"]
    # 다시 쫓을 때는 새 ID(같은 ID면 옛 주문을 찾아 아무것도 안 낸다)
    assert sent[-1] == f"mnv-{pos['bar'] // 60000}-x1-2" and sent[0] != sent[-1]


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


def test_lost_touch_entry_keeps_pending_until_position_shows(env, monkeypatch):
    """바로 체결되는 진입 지정가 응답을 잃고 포지션도 아직 0으로 보이면 pending을 남겨, 다음
    대사가 엔진 밖 포지션으로 보지 않고 이어 받아 손절을 건다(다시 내지 않는다)."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    monkeypatch.setattr(bt, "fast_market", lambda *a: "급변동")
    ex.lose_response = {"POST /fapi/v1/order"}
    real_query, real_row = bt._query, bt.position_row
    lag = {"on": True}

    def lagging_query(client, path, key, cid):
        if cid.endswith("-ot1") and lag["on"]:
            return None
        return real_query(client, path, key, cid)

    def lagging_row(client):
        return {"amt": 0.0, "entry": 0.0} if lag["on"] else real_row(client)

    monkeypatch.setattr(bt, "_query", lagging_query)
    monkeypatch.setattr(bt, "position_row", lagging_row)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.amt > 0 and state()["pending"]["open"].endswith("-o")
    assert state().get("last_bar") != BAR
    lag["on"] = False
    run(env, ex)
    pos = state()["position"]
    assert pos and pos["side"] == "long" and not bt.entries_blocked()
    assert any(a["algoStatus"] == "NEW" and a["type"] == "STOP" for a in ex.algos.values())
    assert len([w for w in ex.writes if w[1] == "/fapi/v1/order"]) == 1  # 진입은 한 번만


# ------------------------------------------ 여러 종목 (사용자 2026-11-11, 한 번에 하나)

FLTS = {
    "BTCUSDT": FLT,
    "ETHUSDT": {**FLT, "tick": "0.01", "step": "0.001", "min_notional": 20.0},
    "SOXLUSDT": {**FLT, "tick": "0.01", "step": "0.01", "min_qty": 0.01, "min_notional": 5.0},
    "NVDAUSDT": {**FLT, "tick": "0.01", "step": "0.01", "min_qty": 0.01, "min_notional": 5.0},
}
PRICES = {"BTCUSDT": 100_000.0, "ETHUSDT": 3_000.0, "SOXLUSDT": 30.0, "NVDAUSDT": 180.0}


class MultiExchange(FakeExchange):
    """종목마다 포지션·레버리지·주문을 따로 두는 가짜 바이낸스. 주문 가격은 종목 가격."""

    def __init__(self, **kw):
        self.amts = {s: 0.0 for s in bt.SYMBOLS}
        self._cur = "BTCUSDT"
        super().__init__(**kw)
        self.levs = {"BTCUSDT": 3, "ETHUSDT": 20, "SOXLUSDT": 20, "NVDAUSDT": 20}
        self.margins = {s: "CROSSED" for s in bt.SYMBOLS}
        self.lev_answer: int | None = None  # 레버리지 변경 응답을 다르게(거래소가 다른 값을 줄 때)
        self.tradfi_reject: set[str] = set()

    @property
    def amt(self):
        return self.amts[self._cur]

    @amt.setter
    def amt(self, v):
        self.amts[self._cur] = v

    def _handle(self, method, path, p):
        sym = p.get("symbol")
        self._cur = sym or "BTCUSDT"
        self.price = PRICES[self._cur]
        if path == "/fapi/v2/positionRisk":
            return [
                {
                    "symbol": s,
                    "positionAmt": str(a),
                    "entryPrice": str(self.__dict__.get("avgs", {}).get(s) or PRICES.get(s, 0.0)),
                }
                for s, a in self.amts.items()
                if sym is None or s == sym
            ]
        if path == "/fapi/v1/symbolConfig":
            return [{"symbol": sym, "leverage": self.levs[sym], "marginType": self.margins[sym]}]
        if path == "/fapi/v1/leverage":
            if self.lev_answer is not None:
                return {"leverage": self.lev_answer}
            self.levs[sym] = int(p["leverage"])
            return {"leverage": self.levs[sym]}
        if path == "/fapi/v1/marginType":
            self.margins[sym] = p["marginType"]
            return {"code": 200, "msg": "success"}
        if path == "/fapi/v1/order" and method == "POST" and sym in self.tradfi_reject:
            raise bt.Rejected(-4411, "Please sign TradFi-Perps agreement contract fapi.")
        if path in ("/fapi/v1/openAlgoOrders", "/fapi/v1/openOrders"):
            rows = super()._handle(method, path, p)
            return [r for r in rows if self._sym_of(r) == sym]
        n_trades = len(self.trades)
        out = super()._handle(method, path, p)
        for t in self.trades[n_trades:]:
            t.setdefault("symbol", self._cur)
        if method == "POST" and path == "/fapi/v1/algoOrder":
            out["symbol"] = sym
        if method == "POST" and path == "/fapi/v1/order" and p["type"] == "LIMIT":
            self.limits[p["newClientOrderId"]]["symbol"] = sym
        if path == "/fapi/v1/userTrades":
            out = [t for t in out if t.get("symbol", "BTCUSDT") == sym]
        return out

    def _sym_of(self, r):
        cid = r.get("clientAlgoId") or r.get("clientOrderId")
        if cid in self.algos:
            return self.algos[cid].get("symbol", "BTCUSDT")
        if cid in self.limits:
            return self.limits[cid].get("symbol", "BTCUSDT")
        return r.get("symbol", "BTCUSDT")

    def _limit_status(self, cid):
        self._cur = self.limits[cid].get("symbol", "BTCUSDT")
        return super()._limit_status(cid)


class MultiMarket(FakeMarket):
    """종목마다 따로 봉·가격을 가진 시세. fail에 든 종목은 봉을 못 받는다."""

    def __init__(self, sym="BTCUSDT", pool=None):
        super().__init__(PRICES[sym])
        self.sym = sym
        self.rows = rows(300, PRICES[sym])
        self.spread = 0.01 if PRICES[sym] < 1000 else 0.2
        self.pool = pool if pool is not None else {}
        self.pool[sym] = self
        self.fail: set[str] = set()

    def for_symbol(self, sym):
        if sym not in self.pool:
            MultiMarket(sym, self.pool)
        return self.pool[sym]

    def klines(self, interval):
        if self.sym in self.pool["BTCUSDT"].fail:
            raise bt.NetworkError("feed down")
        return self.rows


@pytest.fixture
def multi(env, monkeypatch):
    monkeypatch.setattr(bt, "filters", lambda c: dict(FLTS[bt._sym(c)]))
    mk = MultiMarket()
    for s in bt.SYMBOLS:
        mk.for_symbol(s)
    return mk


def signals(monkeypatch, mk, dirs: dict, bar=BAR):
    """종목별 신호. 봉 목록(종목마다 다른 리스트)으로 어느 종목인지 안다."""
    by_rows = {id(mk.for_symbol(s).rows): s for s in bt.SYMBOLS}

    def fake(r, n, last, lookback=2):
        sym = by_rows[id(r)]
        d = dirs.get(sym)
        if not d or (last is not None and bar <= last):
            return {"dir": None, "bar": bar, "close": PRICES[sym]}
        return {"dir": d, "bar": bar, "close": PRICES[sym], "bar_txt": bt._bar_txt(bar)}

    monkeypatch.setattr(bt, "latest_signal", fake)


def ai_answers(monkeypatch, choose: str, exits='{"stop_pct": 3.0, "tp_pct": null, "reason": "x"}'):
    asked = {"choose": [], "exits": []}

    def ai(prompt):
        if prompt.startswith(bt.btc_ai.CHOOSE_PROMPT[:30]):
            asked["choose"].append(prompt)
            return choose
        asked["exits"].append(prompt)
        return exits

    monkeypatch.setattr(bt, "AI_CALLER", ai)
    return asked


def mrun(env, ex, mk):
    return bt.run(now=NOW, client=ex, market_=mk, send=env["send"])


def test_old_state_without_symbol_is_btc(env, multi, monkeypatch):
    assert bt.pos_symbol({"side": "long"}) == "BTCUSDT"
    assert bt.held_symbol({"pending": {"close": "x"}}) == "BTCUSDT"
    live(env)
    ex = MultiExchange()
    old = engine_long(ex)  # 예전 모양 상태(symbol 없음)
    assert "symbol" not in old
    signals(monkeypatch, multi, {"BTCUSDT": "short"})
    run_ = mrun(env, ex, multi)
    closes = [x for m, p, x in ex.writes if x.get("reduceOnly") == "true"]
    assert closes and closes[0]["symbol"] == "BTCUSDT"
    assert ex.algos[old["stop_id"]]["algoStatus"] == "CANCELED"
    pos = state()["position"]
    assert pos["symbol"] == "BTCUSDT" and pos["side"] == "short"
    assert run_["decision"].startswith("BTC 숏 진입")
    assert jsonl("closed.jsonl")[-1]["symbol"] == "BTCUSDT"


def test_cid_has_symbol_tag_and_btc_unchanged():
    assert bt._cid(BAR, "o") == f"mnv-{BAR // 60000}-o"
    assert bt._cid(BAR, "o", "ETHUSDT") == f"mnv-E{BAR // 60000}-o"
    ids = {bt._cid(BAR, "s", s) for s in bt.SYMBOLS}
    assert len(ids) == 4 and all(i.startswith(bt.PREFIX) for i in ids)


def test_flat_two_signals_ai_pick_is_honored(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    signals(monkeypatch, multi, {"BTCUSDT": "long", "ETHUSDT": "short"})
    asked = ai_answers(monkeypatch, '{"choice": "enter:ETHUSDT", "reason": "ETH 추세가 강함"}')
    s = mrun(env, ex, multi)
    assert len(asked["choose"]) == 1 and len(asked["exits"]) == 1
    assert '"enter:BTCUSDT"' in asked["choose"][0] and '"none"' in asked["choose"][0]
    assert '"symbol": "ETHUSDT"' in asked["exits"][0]
    syms = {x.get("symbol") for m, p, x in ex.writes}
    assert syms == {"ETHUSDT"}  # BTC에는 아무 주문도 없다
    assert ex.levs["ETHUSDT"] == 3
    order = next(x for m, p, x in ex.writes if p == "/fapi/v1/order")
    assert order["side"] == "SELL" and order["newClientOrderId"].startswith("mnv-E")
    assert float(order["quantity"]) == 0.99  # 1000 × 0.99 × 3 / 3000, ETH 단위 0.001
    pos = state()["position"]
    assert pos["symbol"] == "ETHUSDT" and pos["side"] == "short"
    assert state()["last_bars"]["BTCUSDT"] == BAR  # 고르지 않은 신호도 처리됨
    assert "ETH 숏 진입" in s["decision"] and s["symbol"] == "ETHUSDT"
    assert any("신호 겹침" in m and "ETH 추세가 강함" in m for m in env["sent"])
    assert any(m.startswith("🔴 ETH 숏 진입") for m in env["sent"])


@pytest.mark.parametrize(
    "dirs, want",
    [
        ({"BTCUSDT": "long", "ETHUSDT": "long", "SOXLUSDT": "long"}, "BTCUSDT"),
        ({"ETHUSDT": "short", "NVDAUSDT": "long"}, "NVDAUSDT"),  # 승률 49% > 44%
        ({"SOXLUSDT": "long", "NVDAUSDT": "short"}, "NVDAUSDT"),
        ({"SOXLUSDT": "long", "ETHUSDT": "short"}, "ETHUSDT"),
    ],
)
def test_ai_failure_falls_back_to_priority(env, multi, monkeypatch, dirs, want):
    ex = MultiExchange()  # 드라이런
    signals(monkeypatch, multi, dirs)
    ai_answers(monkeypatch, "모르겠어요")
    mrun(env, ex, multi)
    assert not ex.writes
    pos = state()["position"]
    assert pos["symbol"] == want and pos["side"] == dirs[want] and pos["mode"] == "dry"
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "ai_choose"][-1]
    assert ev["choice"] == f"enter:{want}" and ev["fallback"]


def test_priority_follows_backtest_winrate():
    assert bt.PRIORITY == ("BTCUSDT", "NVDAUSDT", "ETHUSDT", "SOXLUSDT")
    assert set(bt.SYMBOL_WINRATE) == set(bt.SYMBOLS)


def test_choose_packet_has_winrates_and_prompt_prefers_them(env, multi, monkeypatch):
    ex = MultiExchange()
    held_btc_long(ex)
    st = state()
    st["position"]["mode"] = "dry"
    bt._save(bt.DIR / "state.json", st)
    signals(monkeypatch, multi, {"NVDAUSDT": "long", "SOXLUSDT": "short"})
    asked = ai_answers(monkeypatch, '{"choice": "keep", "reason": "x"}')
    mrun(env, ex, multi)
    prompt = asked["choose"][0]
    assert "승률이 높은" in prompt
    data = json.loads(prompt[len(bt.btc_ai.CHOOSE_PROMPT) :])
    rates = {c["symbol"]: c["backtest_winrate_pct"] for c in data["candidates"]}
    assert rates == {"NVDAUSDT": 49, "SOXLUSDT": 46}
    assert data["held"]["backtest_winrate_pct"] == 40


def test_ai_none_enters_nothing(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    signals(monkeypatch, multi, {"BTCUSDT": "long", "ETHUSDT": "long"})
    ai_answers(monkeypatch, '{"choice": "none", "reason": "둘 다 애매"}')
    mrun(env, ex, multi)
    # 고르기 전에 후보 종목을 3배로 맞춰 둔다(ETH 20배 → 3배). 주문은 없다
    assert [(p, x["symbol"]) for m, p, x in ex.writes] == [("/fapi/v1/leverage", "ETHUSDT")]
    assert state().get("position") is None
    assert state()["last_bars"] == {"BTCUSDT": BAR, "ETHUSDT": BAR}


def universe_file(rates: dict, at: datetime = NOW, extra: dict | None = None) -> None:
    """종목 범위 파일(거래량 순 = rates 순서)."""
    stats = {
        s: {
            "rank": i + 1,
            "winrate": r,
            "trades": 50,
            "ret": 10.0,
            "days": 300.0,
            "contract": "PERPETUAL",
        }
        for i, (s, r) in enumerate(rates.items())
    }
    for s, more in (extra or {}).items():
        stats[s].update(more)
    data = {
        "updated_at": at.strftime("%Y-%m-%d %H:%M"),
        "rule": "test",
        "symbols": list(rates),
        "stats": stats,
    }
    (bt.DIR / "universe.json").write_text(json.dumps(data))


# 보유 BTC(40%)보다 SWITCH_EDGE_PP(1%p) 이상 높은 종목이 바꾸기 후보가 되는 승률
HELD_RATES = {"BTCUSDT": 40, "ETHUSDT": 50, "SOXLUSDT": 46, "NVDAUSDT": 49}


def held_btc_long(ex):
    universe_file(HELD_RATES)
    pos = engine_long(ex)
    st = state()
    st["position"]["symbol"] = "BTCUSDT"
    bt._save(bt.DIR / "state.json", st)
    return pos


def test_holding_a_other_signal_ai_keep(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "keep", "reason": "BTC 수익 중"}')
    s = mrun(env, ex, multi)
    assert [p for m, p, x in ex.writes] == ["/fapi/v1/leverage"] and not asked["exits"]
    assert '"held"' in asked["choose"][0] and '"bars_held_15m": 4' in asked["choose"][0]
    assert state()["position"]["symbol"] == "BTCUSDT"
    assert state()["last_bars"]["ETHUSDT"] == BAR
    assert "고르지 않음" in s["decision"]


def test_holding_a_other_signal_ai_switch(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    old = held_btc_long(ex)
    signals(monkeypatch, multi, {"NVDAUSDT": "short"})
    ai_answers(monkeypatch, '{"choice": "switch:NVDAUSDT", "reason": "NVDA가 더 좋아요"}')
    s = mrun(env, ex, multi)
    seq = [(m, p, x.get("symbol"), x.get("reduceOnly")) for m, p, x in ex.writes]
    assert seq[:3] == [
        ("POST", "/fapi/v1/leverage", "NVDAUSDT", None),  # 닫기 전에 갈 종목을 3배로
        ("POST", "/fapi/v1/order", "BTCUSDT", "true"),  # BTC 지정가 청산
        ("DELETE", "/fapi/v1/algoOrder", None, None),  # 닫힌 뒤 BTC 손절 취소
    ]
    assert ex.algos[old["stop_id"]]["algoStatus"] == "CANCELED"
    assert ex.amts["BTCUSDT"] == 0 and ex.amts["NVDAUSDT"] < 0
    entry = next(x for m, p, x in ex.writes if x.get("symbol") == "NVDAUSDT" and "quantity" in x)
    # 1000 × 0.99 × 3 / 180 = 16.5 → NVDA 단위 0.01, 매도 1호가 180.01
    assert float(entry["quantity"]) == 16.5 and entry["newClientOrderId"].startswith("mnv-N")
    assert ex.levs["NVDAUSDT"] == 3
    stop = [x for m, p, x in ex.writes if p == "/fapi/v1/algoOrder" and m == "POST"][-1]
    assert stop["symbol"] == "NVDAUSDT" and stop["triggerPrice"] == "185.42"  # 180.01 × 1.03 올림
    pos = state()["position"]
    assert pos["symbol"] == "NVDAUSDT" and pos["side"] == "short"
    closed = jsonl("closed.jsonl")[-1]
    assert closed["symbol"] == "BTCUSDT" and closed["why"] == "signal"
    assert s["decision"].startswith("NVDA 숏 진입")


def test_holding_a_opposite_plus_b_fallback_flips_a(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    signals(monkeypatch, multi, {"BTCUSDT": "short", "ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, "")  # 빈 답 → 기본 규칙
    mrun(env, ex, multi)
    prompt = asked["choose"][0]
    # 자기 반대 신호가 있으면 유지는 선택지에 없다(사용자 규칙)
    assert '"keep"' not in prompt
    assert '"flip:BTCUSDT"' in prompt and '"switch:ETHUSDT"' in prompt
    assert '"opposite_signal_now": true' in prompt
    orders = [x for m, p, x in ex.writes if p in ("/fapi/v1/order", "/fapi/v1/algoOrder")]
    assert {x.get("symbol") for x in orders if "symbol" in x} == {"BTCUSDT"}
    pos = state()["position"]
    assert pos["symbol"] == "BTCUSDT" and pos["side"] == "short"
    assert state()["last_bars"]["ETHUSDT"] == BAR


def test_holding_a_opposite_plus_b_ai_switch(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    signals(monkeypatch, multi, {"BTCUSDT": "short", "ETHUSDT": "long"})
    ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "ETH"}')
    mrun(env, ex, multi)
    pos = state()["position"]
    assert pos["symbol"] == "ETHUSDT" and pos["side"] == "long"
    assert ex.amts == {"BTCUSDT": 0.0, "ETHUSDT": 0.99, "SOXLUSDT": 0.0, "NVDAUSDT": 0.0}


def test_switch_blocked_by_chase_keeps_a(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    multi.for_symbol("ETHUSDT")._price = 3_100.0  # 신호 종가 3000에서 3.3% → 추격 한도 밖
    asked = ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    mrun(env, ex, multi)
    assert not asked["choose"] and not ex.writes  # 갈 수 없는 종목은 선택지에 없다
    assert state()["position"]["symbol"] == "BTCUSDT"
    assert any("ETH 롱 신호가 났지만" in m and "추격 한도" in m for m in env["sent"])


def test_tradfi_rejection_marks_symbol_unavailable(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    ex.tradfi_reject = {"SOXLUSDT"}
    signals(monkeypatch, multi, {"SOXLUSDT": "long"})
    s = mrun(env, ex, multi)  # 죽지 않는다
    st = state()
    assert st.get("position") is None and "pending" not in st
    rec = st["unavailable"]["SOXLUSDT"]
    assert rec["code"] == -4411 and rec["until_ms"] == NOW_MS + 24 * 3_600_000
    assert sum("TradFi" in m and "SOXL" in m for m in env["sent"]) == 1
    assert ex.amts["SOXLUSDT"] == 0 and "SOXL" in s["decision"]
    # 다음 봉: 같은 종목 신호는 조용히 건너뛰고 다른 종목은 그대로 매매
    n_sent, n_writes = len(env["sent"]), len(ex.writes)
    signals(monkeypatch, multi, {"SOXLUSDT": "long", "ETHUSDT": "long"}, bar=BAR + bt.BAR_MS)
    later = NOW.replace(minute=45)
    bt.run(now=later, client=ex, market_=multi, send=env["send"])
    assert not any("TradFi" in m for m in env["sent"][n_sent:])
    assert not any(x.get("symbol") == "SOXLUSDT" for m, p, x in ex.writes[n_writes:])
    assert state()["position"]["symbol"] == "ETHUSDT"


def test_tradfi_rejected_again_after_24h_not_renotified(env, monkeypatch):
    ctx = bt.Ctx(MultiExchange(), MultiMarket(), env["send"], NOW, True).at("NVDAUSDT")
    st: dict = {}
    err = bt.Rejected(-4411, "Please sign TradFi-Perps agreement contract fapi.")
    bt.mark_unavailable(ctx, st, err)
    assert bt.unavailable(st, "NVDAUSDT", NOW_MS + 1000)
    assert not bt.unavailable(st, "NVDAUSDT", NOW_MS + bt.UNAVAILABLE_MS + 1)
    bt.mark_unavailable(ctx, st, err)
    assert len(env["sent"]) == 1


def test_position_on_unexpected_symbol_blocks_entries(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.amts["ETHUSDT"] = 0.5  # 엔진이 들고 있지 않은 종목
    signals(monkeypatch, multi, {"NVDAUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "switch:NVDAUSDT", "reason": "x"}')
    s = mrun(env, ex, multi)
    assert not asked["choose"] and not ex.writes
    assert (bt.DIR / "NO_ENTRY").exists()
    assert any("ETHUSDT +0.5" in m for m in env["sent"])
    assert "매매 안 함" in s["decision"] and ex.amts["ETHUSDT"] == 0.5


def test_flat_position_on_other_symbol_blocks(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    ex.amts["SOXLUSDT"] = -3.0
    signals(monkeypatch, multi, {"BTCUSDT": "long"})
    mrun(env, ex, multi)
    assert not ex.writes and (bt.DIR / "NO_ENTRY").exists()


def test_one_feed_failure_drops_only_that_symbol(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    multi.fail = {"ETHUSDT"}
    signals(monkeypatch, multi, {"ETHUSDT": "long", "NVDAUSDT": "short"})
    asked = ai_answers(monkeypatch, '{"choice": "none", "reason": "x"}')
    s = mrun(env, ex, multi)
    assert not asked["choose"]  # ETH가 빠져 후보가 하나뿐
    assert state()["position"]["symbol"] == "NVDAUSDT"
    assert "ETHUSDT" in s["feed_bad"]
    assert "last_bars" not in state() or "ETHUSDT" not in state()["last_bars"]


def test_held_feed_failure_still_manages_position(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.algos.clear()  # 손절이 사라졌다 → 대사가 다시 건다(시세와 상관없이)
    multi.fail = {"BTCUSDT"}
    signals(monkeypatch, multi, {})
    s = mrun(env, ex, multi)
    assert any(p == "/fapi/v1/algoOrder" and x["symbol"] == "BTCUSDT" for m, p, x in ex.writes)
    assert "시세 이상" in s["decision"]


def test_dry_multi_close_uses_symbol_price(env, multi, monkeypatch):
    ex = MultiExchange()
    signals(monkeypatch, multi, {"SOXLUSDT": "long"})
    mrun(env, ex, multi)
    assert state()["position"]["symbol"] == "SOXLUSDT"
    multi.for_symbol("SOXLUSDT")._price = 31.0
    signals(monkeypatch, multi, {"SOXLUSDT": "short"}, bar=BAR + bt.BAR_MS)
    bt.run(now=NOW.replace(minute=45), client=ex, market_=multi, send=env["send"])
    c = jsonl("closed.jsonl")[-1]
    assert c["symbol"] == "SOXLUSDT" and c["exit"] == 31.0 and c["pnl"] > 0
    assert not ex.writes
    assert any(m.startswith("[드라이런] 🔴 SOXL 숏 진입 예정") for m in env["sent"])


def test_filters_per_symbol_cached_and_normalized(monkeypatch):
    monkeypatch.setattr(bt, "_FILTERS", {})

    def sym(name, tick, step, notional):
        return {
            "symbol": name,
            "status": "TRADING",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": tick},
                {"filterType": "LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": "1000"},
                {"filterType": "MIN_NOTIONAL", "notional": notional},
            ],
        }

    info = {
        "symbols": [
            sym("BTCUSDT", "0.10", "0.001", "50"),
            sym("ETHUSDT", "0.01", "0.001", "20"),
            sym("SOXLUSDT", "0.01000", "0.01", "5"),
            sym("NVDAUSDT", "0.01000", "0.01", "5"),
        ]
    }
    calls = []

    class C:
        def request(self, method, path, params=None, signed=True):
            calls.append(path)
            return info

    c = C()
    soxl = bt.filters(bt.SymClient(c, "SOXLUSDT"))
    assert soxl["tick"] == "0.01" and soxl["step"] == "0.01" and soxl["min_notional"] == 5.0
    assert bt.filters(bt.SymClient(c, "ETHUSDT"))["min_notional"] == 20.0
    assert bt.filters(c)["tick"] == "0.1"  # 심볼 없는 클라이언트는 BTC
    assert len(calls) == 1  # exchangeInfo 한 번으로 네 종목
    assert str(bt.stop_price("long", 63.12, soxl["tick"])) == "61.22"


# ------------------------------------------ 레드팀 2026-11-11: 닫기 전에 갈 종목 점검


def test_switch_target_with_foreign_order_is_not_offered(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.foreign_open = [{"clientOrderId": "web_eth", "symbol": "ETHUSDT"}]
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    mrun(env, ex, multi)
    assert not asked["choose"] and not ex.writes  # BTC는 닫지 않는다
    assert state()["position"]["symbol"] == "BTCUSDT"
    assert any("ETH 롱 신호가 났지만" in m and "미체결" in m for m in env["sent"])


def test_switch_target_with_position_is_not_offered(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    monkeypatch.setattr(bt, "check_strays", lambda ctx, st, held: True)  # 이 점검만 따로 본다
    ex.amts["NVDAUSDT"] = 1.0
    signals(monkeypatch, multi, {"NVDAUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "switch:NVDAUSDT", "reason": "x"}')
    mrun(env, ex, multi)
    assert not asked["choose"] and not ex.writes
    assert any("이미 포지션" in m for m in env["sent"])


def test_symbol_config_network_error_skips_symbol_not_run(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.net_fail["GET /fapi/v1/symbolConfig"] = bt.RETRIES + 1
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    s = mrun(env, ex, multi)  # 예외가 run() 밖으로 나가지 않는다
    assert not asked["choose"] and not ex.writes
    assert "레버리지 설정 조회 실패" in s["decision"]
    assert state()["position"]["symbol"] == "BTCUSDT"


def test_leverage_answer_must_confirm_3x(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    ex.lev_answer = 20
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    mrun(env, ex, multi)
    assert not any("3배로 바꿨어요" in m for m in env["sent"])
    assert not [x for m, p, x in ex.writes if p == "/fapi/v1/order"]
    assert any("응답이 20배" in m for m in env["sent"])
    assert state().get("position") is None


def test_isolated_margin_switched_to_cross_when_empty(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    ex.margins["ETHUSDT"] = "ISOLATED"
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    mrun(env, ex, multi)
    seq = [(p, x.get("symbol")) for m, p, x in ex.writes]
    assert seq[:2] == [("/fapi/v1/marginType", "ETHUSDT"), ("/fapi/v1/leverage", "ETHUSDT")]
    assert ex.margins["ETHUSDT"] == "CROSSED" and state()["position"]["symbol"] == "ETHUSDT"


def test_isolated_margin_with_open_orders_refuses_entry(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    ex.margins["ETHUSDT"] = "ISOLATED"
    sid = "mnv-E1-s"  # 엔진 접두어라 '엔진 밖 미체결'로는 걸리지 않는다
    ex.algos[sid] = {"clientAlgoId": sid, "algoStatus": "NEW", "symbol": "ETHUSDT"}
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    mrun(env, ex, multi)
    assert not ex.writes and ex.margins["ETHUSDT"] == "ISOLATED"
    assert any("교차로 못 바꿈" in m for m in env["sent"])


def test_switch_entry_failure_after_close_is_reported(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    monkeypatch.setattr(bt, "open_position", lambda *a: None)  # 진입 단계에서 못 들어감
    mrun(env, ex, multi)
    assert ex.amts["BTCUSDT"] == 0 and state().get("position") is None
    assert any("BTC 롱은 닫았지만 ETH 롱 진입은 못 했어요" in m for m in env["sent"])


def test_time_budget_fallback_to_flip_has_reason(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    signals(monkeypatch, multi, {"BTCUSDT": "short", "ETHUSDT": "long"})

    def slow_ai(prompt):
        if prompt.startswith(bt.btc_ai.CHOOSE_PROMPT[:30]):
            bt._sleep(bt.MAX_RUN_SECS + 1)  # 고르는 사이 실행 시간 한도를 넘김
            return '{"choice": "switch:ETHUSDT", "reason": "x"}'
        return '{"stop_pct": 3.0, "tp_pct": null, "reason": "x"}'

    monkeypatch.setattr(bt, "AI_CALLER", slow_ai)
    s = mrun(env, ex, multi)
    assert ex.amts["BTCUSDT"] == 0 and ex.amts["ETHUSDT"] == 0  # 반대 신호라 BTC는 닫힘
    skips = [m for m in env["sent"] if "신호가 났지만" in m]
    assert skips and all(m.split("신호가 났지만", 1)[1].strip() for m in skips)
    assert "초를 넘겨" in s["decision"]


def test_exit_plan_prompt_is_symbol_aware(env, multi, monkeypatch):
    """손절·익절 질문(사용자 2026-11-11 "최대한 손해 보면 안 된다"): 실제 종목 이름, 그 종목의
    15분 폭·ATR·백테스트 승률, 자본 보존 지시. 범위·기본값은 그대로."""
    ex = MultiExchange()
    signals(monkeypatch, multi, {"SOXLUSDT": "long"})
    asked = ai_answers(monkeypatch, "")
    mrun(env, ex, multi)
    prompt = asked["exits"][0]
    assert "종목(SOXLUSDT)" in prompt and "<SYMBOL>" not in prompt
    assert "손해 보면 안 된다" in prompt and "SOXLUSDT는 반도체 3배 ETF" in prompt
    data = json.loads(prompt[prompt.index("자료:\n") + 4 :])
    assert data["symbol"] == "SOXLUSDT" and data["backtest_winrate_pct"] == 17
    assert data["recent_15m_range_pct"]["bars"] == 16
    assert data["recent_15m_range_pct"]["avg"] == pytest.approx(0.2, abs=1e-3)
    assert data["atr14_pct"]["15m"] is not None
    assert data["backtest_ref"] is None  # BTC 통계는 다른 종목에 넘기지 않는다
    pos = state()["position"]  # 답이 없으면 손절 3%·익절 없음 그대로
    assert pos["stop_pct"] == 3.0 and pos["tp"] is None
    assert bt.btc_ai.STOP_RANGE == (1.0, 5.0) and bt.btc_ai.TP_RANGE == (1.0, 10.0)


# ---------------------------------------------------------------- 종목 범위(상위 50)


def grow(monkeypatch, ex, rates: dict, extra: dict | None = None) -> None:
    """범위 파일을 쓰고 가짜 거래소·시세가 새 종목도 알게 한다(가격 10, 단위 0.01)."""
    universe_file(rates, extra=extra)
    for s in rates:
        if s not in PRICES:
            monkeypatch.setitem(PRICES, s, 10.0)
            monkeypatch.setitem(
                FLTS, s, {**FLT, "tick": "0.001", "step": "0.01", "min_notional": 5.0}
            )
        ex.amts.setdefault(s, 0.0)
        ex.levs.setdefault(s, 3)
        ex.margins.setdefault(s, "CROSSED")
    bt.use_universe(NOW_MS)


def sig_all(monkeypatch, mk, dirs: dict, syms, bar=BAR):
    """signals()와 같은데 범위 종목 전체(새 종목 포함)를 안다."""
    by_rows = {id(mk.for_symbol(s).rows): s for s in syms}

    def fake(r, n, last, lookback=2):
        sym = by_rows[id(r)]
        d = dirs.get(sym)
        if not d or (last is not None and bar <= last):
            return {"dir": None, "bar": bar, "close": PRICES[sym]}
        return {"dir": d, "bar": bar, "close": PRICES[sym], "bar_txt": bt._bar_txt(bar)}

    monkeypatch.setattr(bt, "latest_signal", fake)


def test_universe_missing_stale_corrupt_falls_back_to_four(env):
    for setup in (
        lambda: None,
        lambda: universe_file({"DOGEUSDT": 70}, at=NOW - timedelta(days=15)),
        lambda: (bt.DIR / "universe.json").write_text("{깨짐"),
    ):
        (bt.DIR / "universe.json").unlink(missing_ok=True)
        setup()
        u = bt.use_universe(NOW_MS)
        assert u["source"] == "default" and bt.SYMBOLS == bt.DEFAULT_SYMBOLS
        assert bt.PRIORITY == ("BTCUSDT", "NVDAUSDT", "ETHUSDT", "SOXLUSDT")
        assert bt.SYMBOL_WINRATE == bt.DEFAULT_WINRATE
    universe_file(
        {"DOGEUSDT": 70, "BTCUSDT": 55, "PEPEUSDT": 90}, extra={"PEPEUSDT": {"trades": 5}}
    )
    u = bt.use_universe(NOW_MS)
    assert u["source"] == "universe" and bt.SYMBOLS == ("DOGEUSDT", "BTCUSDT", "PEPEUSDT")
    # 승률 순, 표본 작은 PEPE(90%, 5회)는 맨 뒤
    assert bt.PRIORITY == ("DOGEUSDT", "BTCUSDT", "PEPEUSDT")
    assert bt.SYMBOL_WINRATE["DOGEUSDT"] == 70


def test_cid_tags_keep_four_and_are_unique_and_short():
    top50 = (
        "BTCUSDT ETHUSDT SOXLUSDT ZECUSDT SNDKUSDT XAUUSDT SOLUSDT NVDAUSDT SKHYNIXUSDT INITUSDT "
        "MUUSDT SOXSUSDT SPCXUSDT KORUUSDT XRPUSDT CLUSDT 1000PEPEUSDT XAGUSDT SKHYUSDT JCTUSDT "
        "NEARUSDT BZUSDT HYPEUSDT SNXXUSDT AGTUSDT QQQUSDT ARIAUSDT DOGEUSDT NBISUSDT INTCUSDT "
        "BNBUSDT ANTHROPICUSDT AMDUSDT UNIUSDT CRWVUSDT MSTRUSDT ONEUSDT ENAUSDT TQQQUSDT "
        "GOOGLUSDT CRCLUSDT MUUUSDT SAMSUNGUSDT DRAMUSDT SUIUSDT MRVLUSDT AVAXUSDT TSLAUSDT "
        "EWYUSDT INJUSDT PEPEUSDT 1000SHIBUSDT SHIBUSDT"
    ).split()
    tags = bt.cid_tags(top50)
    assert {s: tags[s] for s in bt.DEFAULT_SYMBOLS} == bt.CID_TAG  # 실전 ID 호환
    assert len(set(tags.values())) == len(top50)
    assert all(len(t) <= 5 for t in tags.values())
    assert all(t[:1].isalpha() for s, t in tags.items() if s != "BTCUSDT")
    assert bt.cid_tag("SKHYNIXUSDT") == bt.cid_tag("SKHYNIXUSDT")  # 실행마다 같다
    # 가장 긴 ID도 36자 안(바이낸스 한도)
    worst = max((bt.PREFIX + t + str(BAR // 60000) + "-o12-6m" for t in tags.values()), key=len)
    assert len(worst) <= 36
    # 겹치면 해시 태그로 비킨다
    clash = {"AAAAXUSDT": "AAAAQ", "AAAAYUSDT": "AAAAQ"}
    orig = bt.cid_tag
    try:
        bt.cid_tag = lambda s: clash.get(s) or orig(s)
        t2 = bt.cid_tags(list(clash))
        assert len(set(t2.values())) == 2 and all(len(t) <= 5 for t in t2.values())
    finally:
        bt.cid_tag = orig


def test_flat_many_signals_ai_sees_only_top5(env, multi, monkeypatch):
    ex = MultiExchange()  # 드라이런
    rates = {
        "BTCUSDT": 54, "ETHUSDT": 44, "SOXLUSDT": 17, "NVDAUSDT": 49,
        "AAAUSDT": 60, "BBBUSDT": 58, "CCCUSDT": 30, "DDDUSDT": 99, "EEEUSDT": 41,
    }  # fmt: skip
    grow(monkeypatch, ex, rates, extra={"DDDUSDT": {"trades": 3}})  # DDD 표본 적음 → 맨 뒤
    sig_all(monkeypatch, multi, {s: "long" for s in rates}, rates)
    asked = ai_answers(monkeypatch, "모르겠어요")
    mrun(env, ex, multi)
    assert len(asked["choose"]) == 1
    data = json.loads(asked["choose"][0][len(bt.btc_ai.CHOOSE_PROMPT) :])
    got = [c["symbol"] for c in data["candidates"]]
    assert got == ["AAAUSDT", "BBBUSDT", "BTCUSDT", "NVDAUSDT", "ETHUSDT"]
    assert set(data["options"]) == {f"enter:{s}" for s in got} | {"none"}
    pos = state()["position"]
    assert pos["symbol"] == "AAAUSDT"  # 답이 없으면 1순위
    assert all(state()["last_bars"][s] == BAR for s in rates)  # 나머지 신호도 처리됨
    passed = [e for e in jsonl("events.jsonl") if e.get("event") == "signal_passed"]
    assert {e["symbol"] for e in passed} == {"EEEUSDT", "CCCUSDT", "SOXLUSDT", "DDDUSDT"}


def test_holding_other_signal_below_edge_no_ai_call(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)  # BTC 40%
    grow(monkeypatch, ex, {**HELD_RATES, "ETHUSDT": 40.9})
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    s = mrun(env, ex, multi)
    assert not asked["choose"] and not asked["exits"] and not ex.writes
    assert state()["position"]["symbol"] == "BTCUSDT"
    assert state()["last_bars"]["ETHUSDT"] == BAR
    assert "후보 아님" in s["decision"]
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "signal_passed"]
    assert ev and "1%p" in ev[0]["why"]


def test_holding_opposite_with_no_eligible_switch_flips_without_ai(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    grow(monkeypatch, ex, {**HELD_RATES, "ETHUSDT": 40.9})
    signals(monkeypatch, multi, {"BTCUSDT": "short", "ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    mrun(env, ex, multi)
    assert not asked["choose"]
    pos = state()["position"]
    assert pos["symbol"] == "BTCUSDT" and pos["side"] == "short"


def test_switch_edge_is_one_point():
    # 사용자 2026-11-14: 보유 종목보다 1%p 이상 높으면 바꾸기 후보
    assert bt.SWITCH_EDGE_PP == 1.0


def test_holding_other_signal_exactly_one_point_higher_asks_ai(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)  # BTC 40%
    grow(monkeypatch, ex, {**HELD_RATES, "ETHUSDT": 41})
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "keep", "reason": "x"}')
    mrun(env, ex, multi)
    assert asked["choose"]
    assert state()["position"]["symbol"] == "BTCUSDT"


def test_holding_switch_options_capped_at_three(env, multi, monkeypatch):
    ex = MultiExchange()  # 드라이런
    held_btc_long(ex)
    st = state()
    st["position"]["mode"] = "dry"
    bt._save(bt.DIR / "state.json", st)
    rates = {**HELD_RATES, "AAAUSDT": 70, "BBBUSDT": 65, "CCCUSDT": 44}
    grow(monkeypatch, ex, rates)
    sig_all(monkeypatch, multi, {s: "short" for s in rates if s != "BTCUSDT"}, rates)
    asked = ai_answers(monkeypatch, '{"choice": "keep", "reason": "x"}')
    mrun(env, ex, multi)
    data = json.loads(asked["choose"][0][len(bt.btc_ai.CHOOSE_PROMPT) :])
    assert [c["symbol"] for c in data["candidates"]] == ["AAAUSDT", "BBBUSDT", "ETHUSDT"]
    assert set(data["options"]) == {"keep", "switch:AAAUSDT", "switch:BBBUSDT", "switch:ETHUSDT"}
    assert state()["position"]["symbol"] == "BTCUSDT"


SCHED = {
    "EQUITY": {"sessions": [
        {"startTime": NOW_MS - 3_600_000, "endTime": NOW_MS + 3_600_000, "type": "NO_TRADING"},
    ]},
    "COMMODITY": {"sessions": [
        {"startTime": NOW_MS - 3_600_000, "endTime": NOW_MS + 3_600_000, "type": "REGULAR"},
    ]},
}  # fmt: skip


def test_market_closed_by_schedule(env, monkeypatch):
    calls = []

    def pub(url, timeout=15):
        calls.append(url)
        return {"marketSchedules": SCHED}

    monkeypatch.setattr(bt, "_public", pub)
    ctx = bt.Ctx(MultiExchange(), MultiMarket(), env["send"], NOW, False)
    tf = {"contract": bt.TRADFI}
    assert "NO_TRADING" in bt.market_closed(ctx, {}, {**tf, "underlying": "EQUITY"})
    assert bt.market_closed(ctx, {}, {**tf, "underlying": "COMMODITY"}) == ""
    assert bt.market_closed(ctx, {}, {**tf, "underlying": "PREMARKET"}) == ""  # 모름 → 허용
    assert bt.market_closed(ctx, {}, {**tf, "underlying": "FX"}) == ""  # 시간표 없음 → 허용
    assert bt.market_closed(ctx, {}, {"contract": "PERPETUAL", "underlying": "COIN"}) == ""
    assert len(calls) == 1  # 실행마다 한 번
    st = {"tradfi_until_ms": NOW_MS + 1000}
    assert "약관" in bt.market_closed(ctx, st, {**tf, "underlying": "COMMODITY"})

    def down(url, timeout=15):
        raise OSError("down")

    monkeypatch.setattr(bt, "_public", down)
    ctx2 = bt.Ctx(MultiExchange(), MultiMarket(), env["send"], NOW, False)
    assert bt.market_closed(ctx2, {}, {**tf, "underlying": "EQUITY"}) == ""  # 못 받으면 허용


def test_tradfi_entry_skipped_quietly_when_market_closed(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    monkeypatch.setitem(
        FLTS, "NVDAUSDT", {**FLTS["NVDAUSDT"], "contract": bt.TRADFI, "underlying": "EQUITY"}
    )
    monkeypatch.setattr(bt, "_public", lambda url, timeout=15: {"marketSchedules": SCHED})
    signals(monkeypatch, multi, {"NVDAUSDT": "long", "BTCUSDT": "long"})
    asked = ai_answers(monkeypatch, '{"choice": "enter:NVDAUSDT", "reason": "x"}')
    mrun(env, ex, multi)
    assert not asked["choose"]  # 들어갈 수 있는 건 BTC 하나
    assert state()["position"]["symbol"] == "BTCUSDT"
    assert not any("NVDA" in m and "⏸" in m for m in env["sent"])
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "session_skip"]
    assert ev and ev[0]["symbol"] == "NVDAUSDT"


def test_tradfi_rejection_pauses_all_tradfi(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    for s in ("SOXLUSDT", "NVDAUSDT"):
        monkeypatch.setitem(FLTS, s, {**FLTS[s], "contract": bt.TRADFI, "underlying": "EQUITY"})
    monkeypatch.setattr(bt, "_public", lambda url, timeout=15: {"marketSchedules": {}})
    ex.tradfi_reject = {"SOXLUSDT", "NVDAUSDT"}
    signals(monkeypatch, multi, {"SOXLUSDT": "long"})
    mrun(env, ex, multi)
    assert state()["tradfi_until_ms"] == NOW_MS + bt.UNAVAILABLE_MS
    n_sent, n_writes = len(env["sent"]), len(ex.writes)
    signals(monkeypatch, multi, {"NVDAUSDT": "long"}, bar=BAR + bt.BAR_MS)
    bt.run(now=NOW.replace(minute=45), client=ex, market_=multi, send=env["send"])
    assert not any(x.get("symbol") == "NVDAUSDT" for m, p, x in ex.writes[n_writes:])
    assert not any("TradFi" in m or "⏸" in m for m in env["sent"][n_sent:])  # 한 번만 알림


def test_held_symbol_outside_universe_is_still_managed(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    universe_file({"ETHUSDT": 50, "SOXLUSDT": 46, "NVDAUSDT": 49})  # BTC가 빠졌다
    ex.algos.clear()  # 손절이 사라졌다 → 다시 건다
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    assert "BTCUSDT" not in bt.SYMBOLS
    stops = [x for m, p, x in ex.writes if p == "/fapi/v1/algoOrder" and m == "POST"]
    assert stops and stops[0]["symbol"] == "BTCUSDT"
    # 반대 신호: 닫기만 하고 범위 밖이라 다시 들어가지 않는다
    sig_all(monkeypatch, multi, {"BTCUSDT": "short"}, bt.DEFAULT_SYMBOLS, bar=BAR + bt.BAR_MS)
    s = bt.run(now=NOW.replace(minute=45), client=ex, market_=multi, send=env["send"])
    assert ex.amts["BTCUSDT"] == 0 and state().get("position") is None
    assert jsonl("closed.jsonl")[-1]["symbol"] == "BTCUSDT"
    assert "범위" in s["decision"]


def test_fetch_slow_symbol_dropped_others_kept(env, multi, monkeypatch):
    import time as _t

    monkeypatch.setitem(PRICES, "PEPEUSDT", 0.01)

    slow = multi.for_symbol("ETHUSDT")
    slow.klines = lambda iv: (_t.sleep(1.0), slow.rows)[1]
    ctx = bt.Ctx(MultiExchange(), multi, env["send"], NOW, False)
    t0 = _t.monotonic()
    rows, errs = bt.fetch_rows(ctx, ("PEPEUSDT",), budget_s=0.2)
    assert _t.monotonic() - t0 < 0.9  # 늦은 종목을 기다리지 않는다
    assert errs == {"ETHUSDT": "시간 안에 못 받음"}
    assert set(rows) == {"BTCUSDT", "SOXLUSDT", "NVDAUSDT", "PEPEUSDT"}  # 보유 종목도 받는다


def test_position_on_symbol_outside_universe_blocks_entries(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    ex.amts["PEPEUSDT"] = 1000.0  # 범위 밖 종목 수동 포지션
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    mrun(env, ex, multi)
    assert not ex.writes and state().get("position") is None
    assert (bt.DIR / "NO_ENTRY").exists()
    assert any("PEPEUSDT +1000.0" in m for m in env["sent"])


def test_skip_notices_only_for_top_priority(env, multi, monkeypatch):
    ex = MultiExchange()  # 드라이런
    signals(monkeypatch, multi, {"BTCUSDT": "long", "ETHUSDT": "long", "NVDAUSDT": "long"})
    for s in ("BTCUSDT", "ETHUSDT", "NVDAUSDT"):
        multi.for_symbol(s)._price = PRICES[s] * 1.05  # 모두 추격 한도 밖
    mrun(env, ex, multi)
    skips = [m for m in env["sent"] if "⏸" in m]
    assert len(skips) == 1 and "BTC 롱" in skips[0]
    ev = [e for e in jsonl("events.jsonl") if e.get("event") == "signal_skip"]
    assert {e["symbol"] for e in ev} == {"ETHUSDT", "NVDAUSDT"}


def test_near_json_written_each_run(env, multi, monkeypatch):
    ex = MultiExchange()
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    near = json.loads((bt.DIR / "near.json").read_text())
    assert near["source"] == "default" and near["w"] == 3
    rows = near["rows"]
    assert {(r["symbol"], r["side"]) for r in rows} == {
        (s, d) for s in bt.DEFAULT_SYMBOLS for d in ("long", "short")
    }
    r = next(x for x in rows if x["symbol"] == "BTCUSDT" and x["side"] == "long")
    assert r["winrate"] == 54 and r["priority"] == 1 and r["bar"] == BAR
    assert r["k"] == len(r["hit"]) and r["k"] + len(r["need"]) == 3
    from minerva import btc_view

    v = btc_view.near_view()
    assert v["status"] == "ok" and len(v["rows"]) == 8
    (bt.DIR / "near.json").write_text("{")
    assert btc_view.near_view()["status"] == "pending"


def test_exit_prompt_has_market_kind(env, multi, monkeypatch):
    monkeypatch.setitem(
        FLTS, "NVDAUSDT", {**FLTS["NVDAUSDT"], "contract": bt.TRADFI, "underlying": "EQUITY"}
    )
    monkeypatch.setattr(bt, "_public", lambda url, timeout=15: {"marketSchedules": {}})
    ex = MultiExchange()
    signals(monkeypatch, multi, {"NVDAUSDT": "long"})
    asked = ai_answers(monkeypatch, "")
    mrun(env, ex, multi)
    assert '"market_kind": "TradFi US stock/ETF"' in asked["exits"][0]
    assert "market_kind" in bt.btc_ai.PROMPT and "SOXLUSDT" in bt.btc_ai.PROMPT


def test_prepare_attempts_are_capped(env, multi, monkeypatch):
    monkeypatch.setattr(bt, "PREPARE_MAX", 2)
    ex = MultiExchange()
    signals(monkeypatch, multi, {s: "long" for s in bt.SYMBOLS})
    for s in bt.SYMBOLS:
        multi.for_symbol(s)._price = PRICES[s] * 1.05  # 모두 추격 한도 밖
    mrun(env, ex, multi)
    ev = jsonl("events.jsonl")
    assert len([e for e in ev if e.get("event") == "chase_skip"]) == 2
    assert {e["symbol"] for e in ev if e.get("event") == "signal_passed"} == {
        "ETHUSDT",
        "SOXLUSDT",
    }
    assert all(state()["last_bars"][s] == BAR for s in bt.SYMBOLS)


def test_tradable_set_needs_winrate_45_profit_and_sample():
    stats = {
        "BNBUSDT": {"winrate": 56.9, "ret": 99.4, "trades": 123, "days": 365},
        "XRPUSDT": {"winrate": 43.7, "ret": 82.4, "trades": 151, "days": 365},  # 45% 미만
        "XAUUSDT": {"winrate": 57.9, "ret": -21.0, "trades": 95, "days": 332},  # 손실
        "NEWUSDT": {"winrate": 70.0, "ret": 50.0, "trades": 5, "days": 30},  # 표본 작음
        # 네 종목도 ETH처럼 45% 미만은 안 된다
        "NVDAUSDT": {"winrate": 54.9, "ret": -27.7, "trades": 91, "days": 227},
        "ETHUSDT": {"winrate": 43.3, "ret": 89.6, "trades": 104, "days": 365},
        "SOXLUSDT": {"winrate": 16.0, "ret": -99.3, "trades": 131, "days": 177},
        "BZUSDT": {"winrate": 48.5, "ret": -17.7, "trades": 97, "days": 365},  # 손실
    }
    assert bt.MIN_ENTRY_WINRATE == 45.0
    # 사용자 2026-11-13 "수익률이 +인것만": 네 종목(NVDA)도 손실이면 안 된다
    assert bt.tradable_set(list(stats), stats) == {"BNBUSDT"}
    # universe.json이 없을 때(네 종목): 승률 45% 이상만 → BTC(54)·NVDA(49)
    assert bt._default_tradable() == {"BTCUSDT"}


def test_flat_low_sample_symbol_signal_not_entered(env, multi, monkeypatch):
    ex = MultiExchange()  # 드라이런
    rates = {"BTCUSDT": 54, "ETHUSDT": 44, "SOXLUSDT": 17, "NVDAUSDT": 49, "AAAUSDT": 60}
    grow(monkeypatch, ex, rates, extra={"AAAUSDT": {"trades": 5, "days": 30}})
    sig_all(monkeypatch, multi, {"AAAUSDT": "long"}, rates)
    asked = ai_answers(monkeypatch, "모르겠어요")
    mrun(env, ex, multi)
    assert not asked["choose"]
    assert state().get("position") is None
    ev = [json.loads(x) for x in (bt.DIR / "events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "signal_passed" and e.get("symbol") == "AAAUSDT" for e in ev)


def test_close_qty_uses_exchange_amount_without_trailing_zeros():
    assert str(bt._close_qty(12345.0)) == "12345"
    assert str(bt._close_qty(-40.25)) == "40.25"
    assert str(bt._close_qty(0.003)) == "0.003"
    assert str(bt._close_qty(1e-05)) == "0.00001"


def test_public_stops_after_rate_limit(monkeypatch):
    import io
    import urllib.error

    calls = []

    def fake_open(req, timeout=0):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b""))

    monkeypatch.setattr(bt.urllib.request, "urlopen", fake_open)
    try:
        with pytest.raises(urllib.error.HTTPError):
            bt._public("https://x/a")
        with pytest.raises(OSError):
            bt._public("https://x/b")  # 보내지 않는다
        assert calls == ["https://x/a"]
    finally:
        bt._RATE_HIT.clear()


def test_flat_signal_below_winrate_50_not_entered(env, multi, monkeypatch):
    monkeypatch.setattr(bt, "MIN_ENTRY_WINRATE", 50.0)
    ex = MultiExchange()  # 드라이런
    rates = {"BTCUSDT": 54, "ETHUSDT": 44, "SOXLUSDT": 17, "NVDAUSDT": 49}
    grow(monkeypatch, ex, rates)  # 모두 수익 +10%
    sig_all(monkeypatch, multi, {"ETHUSDT": "long"}, rates)
    asked = ai_answers(monkeypatch, "모르겠어요")
    mrun(env, ex, multi)
    assert not asked["choose"]
    assert state().get("position") is None
    ev = [json.loads(x) for x in (bt.DIR / "events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "signal_passed" and e.get("symbol") == "ETHUSDT" for e in ev)


def test_held_symbol_below_winrate_50_closes_without_flip(env, multi, monkeypatch):
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    monkeypatch.setattr(bt, "MIN_ENTRY_WINRATE", 50.0)  # 보유 BTC 승률 40 → 진입 불가
    sig_all(monkeypatch, multi, {"BTCUSDT": "short"}, bt.DEFAULT_SYMBOLS, bar=BAR + bt.BAR_MS)
    bt.run(now=NOW.replace(minute=45), client=ex, market_=multi, send=env["send"])
    assert ex.amts["BTCUSDT"] == 0 and state().get("position") is None
    assert jsonl("closed.jsonl")[-1]["symbol"] == "BTCUSDT"


def test_fee_buffer_leaves_3pct_margin_headroom():
    assert bt.FEE_BUFFER == 0.03
    flt = {"step": "0.01", "max_qty": 1e9, "min_qty": 0.01, "min_notional": 5.0}
    qty, why = bt.size_qty(767.01, 190.0, 1.0, flt)
    # 767.01 × 0.97 × 3 / 190 = 11.747… → 11.74 (증거금 약 743.6, 가용의 97%)
    assert why == "" and str(qty) == "11.74"


def test_entry_margin_reject_backs_off_and_retries_smaller(env, monkeypatch):
    """사용자 2026-11-12 "거래실패시 백오프": 증거금 부족(-2019)이면 포지션 0·체결 없음을 확인하고
    쉬었다가 새 ID·작은 수량으로 다시 낸다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["margin", "fill"]
    slept = []
    real_sleep = bt._sleep
    monkeypatch.setattr(bt, "_sleep", lambda s: (slept.append(s), real_sleep(s)))
    with_signal(monkeypatch, "long")
    run(env, ex)
    pos = state()["position"]
    assert pos and pos["side"] == "long" and ex.amt > 0
    posts = [p for m, path, p in ex.writes if path == "/fapi/v1/order" and m == "POST"]
    assert len(posts) == 2  # 첫 주문 거부(-2019), 두 번째(새 ID) 체결
    assert posts[1]["newClientOrderId"].endswith("-or1-1")
    # 1000 × 0.99 × 3 / 100000 = 0.029 → 5% 줄여 0.027
    assert posts[0]["quantity"] == "0.029" and posts[1]["quantity"] == "0.027"
    assert bt.ENTRY_BACKOFF_S[0] in slept
    errs = [e for e in jsonl("events.jsonl") if e.get("event") == "enter_error"]
    assert errs and "diag" in errs[0] and errs[0]["try"] == 1
    assert any("다시 시도" in m for m in env["sent"])
    assert any(a["type"] == "STOP" and a["algoStatus"] == "NEW" for a in ex.algos.values())


def test_entry_margin_reject_gives_up_after_backoffs(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["margin"] * 10
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.amt == 0 and state().get("position") is None
    errs = [e for e in jsonl("events.jsonl") if e.get("event") == "enter_error"]
    assert len(errs) == len(bt.ENTRY_BACKOFF_S) + 1


def test_entry_other_reject_not_retried(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["margin"]
    monkeypatch.setattr(bt, "ENTRY_RETRY_CODES", frozenset())
    with_signal(monkeypatch, "long")
    run(env, ex)
    errs = [e for e in jsonl("events.jsonl") if e.get("event") == "enter_error"]
    assert len(errs) == 1 and ex.amt == 0


def test_entry_retry_skipped_when_price_ran_past_chase_limit(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.limit_script = ["margin", "fill"]

    # 첫 거부(-2019) 뒤에는 가격이 달아났다고 본다
    def gone(side, px, close, pct):
        return (
            not (bt.DIR / "events.jsonl").exists()
            or "enter_error" not in (bt.DIR / "events.jsonl").read_text()
        )

    monkeypatch.setattr(bt, "chase_ok", gone)
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert ex.amt == 0 and state().get("position") is None
    assert any("추격 한도 밖" in m for m in env["sent"])


# ------------------------------------------ 수동 포지션 이어받기 (사용자 2026-11-13)


def test_manual_open_is_adopted_with_stop_and_tp(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.amt = 0.02  # 사용자가 앱에서 산 롱
    ex.entry_px = 100_000.0
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    assert pos["manual"] and pos["side"] == "long" and pos["qty"] == 0.02
    assert pos["stop"] == 97000.0 and pos["tp"] == 103000.0
    kinds = {a["type"]: a["triggerPrice"] for a in ex.algos.values() if a["algoStatus"] == "NEW"}
    assert kinds == {"STOP": "97000.0", "TAKE_PROFIT": "103000.0"}
    assert not bt.entries_blocked()
    assert any("이어받아 관리" in m for m in env["sent"])
    n = len(ex.algos)
    run(env, ex)  # 다음 실행: 그대로 둔다(다시 걸지 않음)
    assert len(ex.algos) == n


def test_manual_open_past_stop_is_closed(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.amt = 0.02
    ex.entry_px = 100_000.0
    ex.reject_stop = True  # 이미 손절가 아래라 걸면 바로 체결될 주문 → 거부
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0 and state().get("position") is None


def test_manual_close_of_engine_position_does_not_halt(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.0
    with_signal(monkeypatch, None)
    run(env, ex)
    assert jsonl("closed.jsonl")[-1]["why"] == "manual"
    assert not bt.entries_blocked() and state().get("position") is None


def test_manual_add_to_engine_position_is_readopted(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex, qty=0.029)
    ex.amt = 0.05  # 사용자가 더 샀다
    ex.entry_px = 99_000.0
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    assert pos["qty"] == 0.05 and pos["entry"] == 99000.0 and pos["stop"] == 96030.0
    live_stops = [a for a in ex.algos.values() if a["algoStatus"] == "NEW"]
    assert sorted(a["type"] for a in live_stops) == ["STOP", "TAKE_PROFIT"]
    assert all(a["quantity"] == "0.05" for a in live_stops)  # 새 수량으로 다시 걸었다
    assert not bt.entries_blocked()


def test_manual_other_symbol_guarded_while_engine_holds(env, multi, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.amts["ETHUSDT"] = 0.5  # 사용자가 손으로 산 ETH (엔진은 BTC 보유)
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    g = state()["guards"]["ETHUSDT"]
    assert g["stop"] == 2910.0 and g["tp"] == 3090.0
    eth = [a for a in ex.algos.values() if a["clientAlgoId"] in (g["stop_id"], g["tp_id"])]
    assert sorted(a["type"] for a in eth) == ["STOP", "TAKE_PROFIT"]
    assert all(a["quantity"] == "0.5" for a in eth)
    assert state()["position"]["symbol"] == "BTCUSDT" and not bt.entries_blocked()
    # 손으로 정리하면 지킴 주문을 거둔다
    ex.amts["ETHUSDT"] = 0.0
    mrun(env, ex, multi)
    assert "ETHUSDT" not in state()["guards"]
    assert all(ex.algos[i]["algoStatus"] != "NEW" for i in (g["stop_id"], g["tp_id"]))


def test_manual_other_symbol_adopted_when_flat(env, multi, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = MultiExchange()
    ex.amts["SOXLUSDT"] = -3.0  # 빈손인데 손으로 판 SOXL 숏
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    pos = state()["position"]
    assert pos["symbol"] == "SOXLUSDT" and pos["side"] == "short" and pos["qty"] == 3.0
    assert pos["stop"] == 30.9 and pos["tp"] == 29.1
    assert not bt.entries_blocked()


def test_liquidation_still_halts_with_manual_adopt(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.0
    ex.force_orders = [{"orderId": 1, "status": "FILLED"}]  # 강제 청산
    with_signal(monkeypatch, None)
    run(env, ex)
    assert bt.entries_blocked()


FAIL_ONCE = bt.RETRIES + 1  # 한 번 묻기(재시도 포함) 실패


def test_liquidation_read_failure_pauses_entries_then_rechecks(env, monkeypatch):
    # 사용자 2026-11-14: 강제 청산 기록을 못 읽었다고 바로 멈추지 않는다
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.0
    ex.net_fail["GET /fapi/v1/forceOrders"] = FAIL_ONCE
    with_signal(monkeypatch, "long")
    run(env, ex)
    assert jsonl("closed.jsonl")[-1]["why"] == "manual"
    assert not bt.entries_blocked()
    assert state().get("position") is None and ex.amt == 0.0  # 모르는 동안은 새로 안 들어감
    assert state()["liq_recheck"]["BTCUSDT"]["n"] == 1
    # 다음 실행: 읽히고 강제 청산이 아니다 → 기록을 지우고 평소대로
    with_signal(monkeypatch, None)
    run(env, ex)
    assert "liq_recheck" not in state() and not bt.entries_blocked()


def test_liquidation_recheck_finds_liquidation_halts(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.0
    ex.net_fail["GET /fapi/v1/forceOrders"] = FAIL_ONCE
    with_signal(monkeypatch, None)
    run(env, ex)
    assert not bt.entries_blocked()
    ex.force_orders = [{"orderId": 1, "status": "FILLED"}]
    run(env, ex)
    assert bt.entries_blocked() and "liq_recheck" not in state()


def test_liquidation_unreadable_four_runs_halts(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    engine_long(ex)
    ex.amt = 0.0
    ex.net_fail["GET /fapi/v1/forceOrders"] = 999
    with_signal(monkeypatch, None)
    for _ in range(bt.LIQ_UNKNOWN_MAX - 1):
        run(env, ex)
        assert not bt.entries_blocked()
    run(env, ex)
    assert bt.entries_blocked() and "liq_recheck" not in state()
    assert "못 읽음" in (bt.DIR / "NO_ENTRY").read_text()


def test_qty_shrink_with_unreadable_liquidation_adopts_and_pauses(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex, qty=0.029)
    ex.amt = 0.02  # 줄었다(손으로 일부 정리 또는 ADL 일부)
    ex.net_fail["GET /fapi/v1/forceOrders"] = FAIL_ONCE
    with_signal(monkeypatch, None)
    run(env, ex)
    assert not bt.entries_blocked()
    assert state()["position"]["qty"] == 0.02  # 지금 수량으로 이어받고 손절도 새 수량
    stops = [a for a in ex.algos.values() if a["algoStatus"] == "NEW" and a["type"] == "STOP"]
    assert len(stops) == 1 and stops[0]["quantity"] == "0.02"
    assert "BTCUSDT" in state()["liq_recheck"]
    # 확인 중에도 반대 신호 청산은 한다(새 진입만 쉰다)
    ex.net_fail["GET /fapi/v1/forceOrders"] = FAIL_ONCE
    with_signal(monkeypatch, "short")
    run(env, ex)
    assert ex.amt == 0.0 and state().get("position") is None
    assert not bt.entries_blocked()


def test_manual_flip_or_add_adopts_without_reading_liquidations(env, monkeypatch):
    # 강제 청산은 포지션을 줄이기만 한다: 뒤집기·늘리기는 기록을 안 보고 바로 이어받는다
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex, qty=0.02)
    ex.amt = 0.05
    ex.net_fail["GET /fapi/v1/forceOrders"] = 999
    with_signal(monkeypatch, None)
    run(env, ex)
    pos = state()["position"]
    assert pos["qty"] == 0.05 and pos["side"] == "long"
    stops = [a for a in ex.algos.values() if a["algoStatus"] == "NEW" and a["type"] == "STOP"]
    assert len(stops) == 1 and stops[0]["quantity"] == "0.05"
    ex.amt = -0.02  # 손으로 뒤집었다
    run(env, ex)
    pos = state()["position"]
    assert pos["side"] == "short" and pos["qty"] == 0.02
    stops = [a for a in ex.algos.values() if a["algoStatus"] == "NEW" and a["type"] == "STOP"]
    assert len(stops) == 1 and stops[0]["side"] == "BUY"
    assert "liq_recheck" not in state() and not bt.entries_blocked()
    assert not [c for c in ex.calls if c[1] == "/fapi/v1/forceOrders"]


def test_manual_stop_failure_other_than_trigger_does_not_close(env, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    ex.amt = 0.02
    ex.entry_px = 100_000.0
    real = ex._handle

    def handle(method, path, p):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise bt.Rejected(-4045, "Reach max stop order limit.")
        return real(method, path, p)

    monkeypatch.setattr(ex, "_handle", handle)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0.02 and state()["position"]["manual"]  # 닫지 않는다
    assert any("손절을 걸지 못했어요" in m for m in env["sent"])


def test_guarded_symbol_signal_not_entered(env, multi, monkeypatch):
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.amts["ETHUSDT"] = 0.5
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    assert "ETHUSDT" in state()["guards"]
    signals(monkeypatch, multi, {"ETHUSDT": "long"})
    ai_answers(monkeypatch, '{"choice": "switch:ETHUSDT", "reason": "x"}')
    s = mrun(env, ex, multi)
    assert state()["position"]["symbol"] == "BTCUSDT" and ex.amts["ETHUSDT"] == 0.5
    assert "지키는 중" in s["decision"]


# ------------------------------------------ 모든 매매 지정가 (사용자 2026-11-13 20:00)


def _closes(ex, sym=None):
    """reduceOnly 청산 지정가(일반 주문)."""
    return [
        x
        for m, p, x in ex.writes
        if m == "POST"
        and p == "/fapi/v1/order"
        and x.get("reduceOnly") == "true"
        and (sym is None or x.get("symbol") == sym)
    ]


def test_signal_close_chases_touch_until_filled(env, monkeypatch):
    """반대 신호 청산: 매수 1호가에 reduceOnly 지정가 → 몇 초 안 되면 취소하고 다시
    → 다 될 때까지."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    ex.close_script = ["none", "partial:0.009", "fill"]
    with_signal(monkeypatch, "short")
    run(env, ex)
    c = _closes(ex)
    assert [x["newClientOrderId"] for x in c] == [f"mnv-{BAR // 60000}-c-{k}" for k in (1, 2, 3)]
    assert [x["quantity"] for x in c] == ["0.029", "0.029", "0.02"]  # 남은 수량만 다시
    assert all(x["type"] == "LIMIT" and x["timeInForce"] == "GTC" for x in c)
    assert all(x["price"] == "100000.0" for x in c)  # 매도는 매수 1호가
    assert ex.limits[c[0]["newClientOrderId"]]["status"] == "CANCELED"
    assert not _markets(ex)
    assert jsonl("closed.jsonl")[-1]["why"] == "signal" and "closing" not in state()
    assert state()["position"]["side"] == "short"  # 닫힌 뒤 숏 진입


def test_close_chase_rests_across_runs_then_finishes(env, monkeypatch):
    """실행 안에 다 못 닫으면 마지막 지정가를 걸어 둔 채 끝내고(진입 안 함, 정지 안 함), 다음 실행이
    그 주문을 취소하고 새 호가로 이어서 닫는다. 두 번 닫거나 '수동 청산'으로 보지 않는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    old = engine_long(ex)
    ex.close_script = ["none"] * 500
    with_signal(monkeypatch, "short")
    run(env, ex)
    st = state()
    cl = st["closing"]
    assert st["position"]["side"] == "long" and ex.amt == pytest.approx(0.029)
    assert ex.limits[cl["cid"]]["status"] == "NEW"  # 걸어 둔 채(GTC)
    assert cl["why"] == "signal" and cl["k"] >= 2
    assert ex.algos[old["stop_id"]]["algoStatus"] == "NEW"  # 손절은 그대로 지킨다
    assert not bt.entries_blocked() and not jsonl("closed.jsonl")
    assert any("다음 실행에서 이어서" in m for m in env["sent"])
    assert not [x for x in _limits(ex) if x.get("reduceOnly") != "true"]  # 진입 안 함
    # 다음 실행: 걸어 둔 주문을 취소하고 새 ID로 이어서 → 닫힘
    ex.close_script = []
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.limits[cl["cid"]]["status"] == "CANCELED"
    last = _closes(ex)[-1]["newClientOrderId"]
    assert last == f"{cl['base']}-{cl['k'] + 1}"
    assert ex.amt == 0 and state()["position"] is None and "closing" not in state()
    closed = jsonl("closed.jsonl")
    assert len(closed) == 1 and closed[0]["why"] == "signal"
    assert not bt.entries_blocked()
    assert ex.algos[old["stop_id"]]["algoStatus"] == "CANCELED"


def test_resting_close_filled_between_runs_is_engine_close(env, monkeypatch):
    """걸어 둔 청산 지정가가 실행 사이에 체결되면 엔진 청산(why 그대로)으로 적고 멈추지 않는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    ex.close_script = ["none"] * 500
    with_signal(monkeypatch, "short")
    run(env, ex)
    cid = state()["closing"]["cid"]
    ex._fill(ex.limits[cid], 1.0)  # 실행 사이 체결
    assert ex.amt == 0
    n = len(_closes(ex))
    with_signal(monkeypatch, None)
    run(env, ex)
    assert len(_closes(ex)) == n  # 더 내지 않는다
    closed = jsonl("closed.jsonl")
    assert len(closed) == 1 and closed[0]["why"] == "signal" and closed[0]["pnl"] is not None
    assert not bt.entries_blocked() and "closing" not in state()


def test_stop_triggered_but_unfilled_is_chased(env, monkeypatch):
    """지정가 손절이 걸렸는데(갭) 남긴 지정가가 안 채워졌다: 그 주문을 취소하고 호가로 쫓아
    닫는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    child = ex.trigger_unfilled(pos["stop_id"])
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.limits[child]["status"] == "CANCELED"
    assert ex.amt == 0 and state()["position"] is None
    assert jsonl("closed.jsonl")[-1]["why"] == "stop"
    assert any("손절 지정가가 걸렸는데" in m for m in env["sent"])
    assert not bt.entries_blocked() and not _markets(ex)


def test_stop_triggered_partially_filled_not_treated_as_manual(env, monkeypatch):
    """걸린 손절이 일부만 체결돼 수량이 줄어도 '수동 변경'으로 이어받지 않고 남은 것을 쫓는다."""
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    ex.trigger_unfilled(pos["stop_id"], partial=0.01)
    assert ex.amt == pytest.approx(0.019)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0 and state()["position"] is None
    assert jsonl("closed.jsonl")[-1]["why"] == "stop"
    assert not any("이어받아" in m for m in env["sent"])


def test_stop_would_trigger_closes_with_limit_as_stop(env, monkeypatch):
    """손절을 다시 걸려는데 이미 손절가를 지났다(-2021): 지정가로 닫고 '손절'로 적는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    ex.algos.clear()
    ex.reject_stop = True
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0 and not _markets(ex) and _closes(ex)
    assert jsonl("closed.jsonl")[-1]["why"] == "stop"


def test_migration_places_new_limit_stop_before_cancelling_old(env, monkeypatch):
    """배포 직후: 옛 STOP_MARKET·TAKE_PROFIT_MARKET(closePosition) 포지션은 첫 실행에서 새 지정가
    손절을 먼저 걸고 걸린 것을 본 뒤 옛 손절을 취소, 옛 익절을 취소하고 새 지정가 익절을 건다."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex, style="")
    old_sid = pos["stop_id"]
    ex.algos[old_sid] = {
        "clientAlgoId": old_sid,
        "algoStatus": "NEW",
        "triggerPrice": "97000.0",
        "type": "STOP_MARKET",
        "closePosition": "true",
    }
    tid = f"mnv-{pos['bar'] // 60000}-t"
    ex.algos[tid] = {
        "clientAlgoId": tid,
        "algoStatus": "NEW",
        "triggerPrice": "103000.0",
        "type": "TAKE_PROFIT_MARKET",
        "closePosition": "true",
    }
    st = state()
    st["position"].pop("order_style")
    st["position"].update(tp=103000.0, tp_id=tid, tp_pct=3.0, tp_backfilled=True)
    bt._save(bt.DIR / "state.json", st)
    with_signal(monkeypatch, None)
    run(env, ex)
    seq = [(m, x.get("type") or x.get("clientAlgoId")) for m, p, x in ex.writes]
    assert seq == [
        ("POST", "STOP"),  # 1) 새 지정가 손절 먼저
        ("DELETE", old_sid),  # 2) 옛 손절 취소
        ("DELETE", tid),  # 3) 옛 익절 취소
        ("POST", "TAKE_PROFIT"),  # 4) 새 지정가 익절
    ]
    p = state()["position"]
    assert p["order_style"] == bt.order_style() and p["stop_id"] != old_sid
    assert ex.algos[p["stop_id"]]["quantity"] == "0.029"
    assert ex.algos[p["tp_id"]]["type"] == "TAKE_PROFIT"
    assert ex.algos[old_sid]["algoStatus"] == ex.algos[tid]["algoStatus"] == "CANCELED"
    assert any("새 방식으로 바꿨어요" in m for m in env["sent"])
    n = len(ex.writes)
    run(env, ex)  # 다음 실행은 그대로
    assert len(ex.writes) == n


def test_migration_keeps_old_stop_if_new_one_fails(env, monkeypatch):
    """새 손절이 거부되면 옛 손절을 그대로 두고 다음 실행에서 다시 한다(손절 없는 틈 없음)."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex, style="")
    st = state()
    st["position"].pop("order_style")
    bt._save(bt.DIR / "state.json", st)
    real = ex._handle

    def no_new_stop(method, path, p):
        if method == "POST" and path == "/fapi/v1/algoOrder":
            raise bt.Rejected(-1111, "Precision is over the maximum")
        return real(method, path, p)

    ex._handle = no_new_stop
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.algos[pos["stop_id"]]["algoStatus"] == "NEW"
    assert "order_style" not in state()["position"] and ex.amt == pytest.approx(0.029)
    assert not [w for w in ex.writes if w[0] == "DELETE"]


def test_stop_order_market_switch(env, monkeypatch):
    """STOP_ORDER = "market"이면 손절만 예전 STOP_MARKET(closePosition), 익절은 지정가 그대로."""
    monkeypatch.setattr(bt, "STOP_ORDER", "market")
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    with_signal(monkeypatch, "long")
    run(env, ex)
    algos = [x for m, p, x in ex.writes if p == "/fapi/v1/algoOrder"]
    assert [a["type"] for a in algos] == ["STOP_MARKET", "TAKE_PROFIT"]
    assert algos[0]["closePosition"] == "true" and "quantity" not in algos[0]
    assert state()["position"]["order_style"] == "stop-market/tp-limit"


def test_stop_limit_price_rounds_safe_way():
    assert bt.stop_limit_price("long", Decimal("97000.05"), "0.1") == Decimal("96515.0")
    assert bt.stop_limit_price("short", Decimal("103000"), "0.1") == Decimal("103515.0")
    assert bt.stop_limit_price("short", Decimal("100.01"), "0.01") == Decimal("100.52")


def test_guard_past_stop_is_chased_with_limit(env, multi, monkeypatch):
    """지키는 수동 포지션이 이미 손절가를 지났으면(-2021) 시장가가 아니라 지정가 추격으로 닫는다."""
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.amts["ETHUSDT"] = 0.5
    ex.reject_stop = True  # 새 손절은 바로 걸릴 값(BTC 손절은 이미 걸려 있어 다시 안 건다)
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    c = _closes(ex, "ETHUSDT")
    assert c and c[0]["side"] == "SELL" and c[0]["price"] == "3000.0"
    assert ex.amts["ETHUSDT"] == 0 and "ETHUSDT" not in state()["guards"]
    assert ex.amts["BTCUSDT"] == pytest.approx(0.029) and not _markets(ex)
    assert any("지정가로 닫았어요" in m for m in env["sent"])


def test_guard_close_rests_and_resumes(env, multi, monkeypatch):
    """지키는 포지션 청산이 실행 안에 안 끝나면 걸어 두고 다음 실행이 이어서 닫는다."""
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.amts["ETHUSDT"] = 0.5
    ex.reject_stop = True
    ex.close_script = ["none"] * 500
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    g = state()["guards"]["ETHUSDT"]
    assert g["closing"]["cid"] and ex.limits[g["closing"]["cid"]]["status"] == "NEW"
    ex.close_script = []
    mrun(env, ex, multi)
    assert ex.limits[g["closing"]["cid"]]["status"] == "CANCELED"
    assert ex.amts["ETHUSDT"] == 0 and "ETHUSDT" not in state()["guards"]


def test_guard_old_market_orders_migrated_new_first(env, multi, monkeypatch):
    """옛 방식 지킴 주문은 새 지정가 손절·익절을 건 뒤에 거둔다."""
    monkeypatch.setattr(bt, "MANUAL_ADOPT", True)
    live(env)
    ex = MultiExchange()
    held_btc_long(ex)
    ex.amts["ETHUSDT"] = 0.5
    signals(monkeypatch, multi, {})
    mrun(env, ex, multi)
    st = state()
    g = st["guards"]["ETHUSDT"]
    old = (g["stop_id"], g["tp_id"])
    g.pop("order_style")  # 옛 방식으로 건 지킴
    bt._save(bt.DIR / "state.json", st)
    ex.calls.clear()
    mrun(env, ex, multi)
    g2 = state()["guards"]["ETHUSDT"]
    seq = [(m, x.get("clientAlgoId")) for m, p, x in ex.writes if p == "/fapi/v1/algoOrder"]
    posts = [i for i, (m, _) in enumerate(seq) if m == "POST"]
    dels = [i for i, (m, c) in enumerate(seq) if m == "DELETE" and c in old]
    assert len(dels) == 2 and max(posts) < min(dels)  # 새 주문 먼저, 옛 주문은 그 뒤
    assert g2["order_style"] == bt.order_style() and "stale" not in g2
    assert all(ex.algos[i]["algoStatus"] == "NEW" for i in (g2["stop_id"], g2["tp_id"]))


# ------------------------------------------ 레드팀 2026-11-13 (7ffb3aa 지적)


def test_f1_chase_sleeps_and_caps_when_orders_expire(env, monkeypatch):
    """거래소가 청산 지정가를 바로 끝내도(EXPIRED) 쉬지 않고 다시 내지 않는다: 회차마다 쉬고,
    회차 상한·실행 한도에서 멈춘다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    ex.close_script = ["expired"] * 10_000
    with_signal(monkeypatch, "short")
    run(env, ex)
    c = _closes(ex)
    assert 2 <= len(c) <= bt.CLOSE_MAX_ROUNDS
    assert ex.amt == pytest.approx(0.029) and state()["closing"]["why"] == "signal"
    assert not bt.entries_blocked()


def test_f2_4142_treated_as_would_trigger(env, monkeypatch):
    """-4142(손절·익절이 바로 걸릴 값)도 -2021과 같이: 지정가로 닫고 '손절'로 적는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    engine_long(ex)
    ex.algos.clear()
    ex.reject_stop, ex.reject_code = True, -4142
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.amt == 0 and jsonl("closed.jsonl")[-1]["why"] == "stop"


def test_f3_resume_cancels_triggered_child_first(env, monkeypatch):
    """이어서 쫓는 실행에서도 걸린 손절이 남긴 주문을 먼저 지운다(안 지우면 -2022만 반복)."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    ex.child_conflict = True
    child = ex.trigger_unfilled(pos["stop_id"])
    st = state()
    st["closing"] = {"symbol": "BTCUSDT", "why": "stop", "base": "mnv-x9", "k": 0, "cid": None}
    bt._save(bt.DIR / "state.json", st)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert ex.limits[child]["status"] == "CANCELED"
    assert ex.amt == 0 and jsonl("closed.jsonl")[-1]["why"] == "stop"


def test_f3_reduce_conflict_clears_engine_orders_only(env, monkeypatch):
    """-2022인데 포지션이 남았으면 엔진(mnv-) reduceOnly 주문만 지우고 다시. 사용자 주문은 둔다."""
    live(env)
    ex = FakeExchange()
    pos = engine_long(ex)
    base = {"executedQty": "0", "avgPrice": "0", "price": "105000", "origQty": "0.029"}
    common = {"status": "NEW", "side": "SELL", "how": "none", "reduce": True, "tif": "GTC"}
    ex.limits["mnv-stale-x1-1"] = {
        **base,
        **common,
        "clientOrderId": "mnv-stale-x1-1",
        "orderId": 1,
        "conflict": True,
    }
    ex.limits["user-tp"] = {**base, **common, "clientOrderId": "user-tp", "orderId": 2}
    st = state()
    ctx = bt.Ctx(ex, FakeMarket(ex.price), env["send"], NOW, True)
    bt.protect_close(ctx, st, "protect")
    assert st["position"] is None and ex.amt == 0
    assert ex.limits["mnv-stale-x1-1"]["status"] == "CANCELED"
    assert ex.limits["user-tp"]["status"] == "NEW"
    assert pos


def _engine_long_with_tp(ex):
    pos = engine_long(ex)
    tid = f"mnv-{pos['bar'] // 60000}-t"
    ex.algos[tid] = {
        "clientAlgoId": tid,
        "algoStatus": "NEW",
        "triggerPrice": "103000.0",
        "type": "TAKE_PROFIT",
        "side": "SELL",
        "price": "103000.0",
        "quantity": "0.029",
    }
    st = state()
    st["position"].update(tp=103000.0, tp_id=tid, tp_pct=3.0, tp_backfilled=True)
    bt._save(bt.DIR / "state.json", st)
    return st["position"]


def test_f4_triggered_tp_below_entry_stays_normal_position(env, monkeypatch):
    """익절이 걸렸는데 호가가 진입가보다 나쁘면 닫지 않는다: 남은 익절 주문을 지우고 평소
    포지션으로 둔다(손절 그대로, 익절은 다시 걸고, 알림 한 번, 매매 멈추지 않음)."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = _engine_long_with_tp(ex)
    child = ex.trigger_unfilled(pos["tp_id"])
    with_signal(monkeypatch, None)
    s = run(env, ex, FakeMarket(99_000.0))
    assert ex.limits[child]["status"] == "CANCELED" and not _closes(ex)
    p = state()["position"]
    assert ex.amt == pytest.approx(0.029) and "closing" not in state()
    assert ex.algos[pos["stop_id"]]["algoStatus"] == "NEW"  # 손절은 그대로
    assert p["tp_id"] != pos["tp_id"] and ex.algos[p["tp_id"]]["algoStatus"] == "NEW"
    assert ex.algos[p["tp_id"]]["price"] == "103000.0"
    assert sum("익절가에 닿았지만" in m for m in env["sent"]) == 1
    assert "수동 개입" not in s["decision"]
    # 다시 걸린 익절도 또 걸렸다: 알림은 한 번만
    ex.trigger_unfilled(p["tp_id"])
    run(env, ex, FakeMarket(99_000.0))
    assert sum("익절가에 닿았지만" in m for m in env["sent"]) == 1 and ex.amt > 0


def test_h1_tp_waiting_then_stop_gap_is_chased_as_stop(env, monkeypatch):
    """익절이 진입가 아래라 기다리는 중에 손절이 걸리고 지정가가 안 채워지면 손절로 호가 추격."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = _engine_long_with_tp(ex)
    ex.trigger_unfilled(pos["tp_id"])
    with_signal(monkeypatch, None)
    run(env, ex, FakeMarket(99_000.0))
    assert ex.algos[pos["stop_id"]]["algoStatus"] == "NEW"
    child = ex.trigger_unfilled(pos["stop_id"])
    run(env, ex, FakeMarket(96_000.0))
    assert ex.limits[child]["status"] == "CANCELED"
    assert ex.amt == 0 and _closes(ex)[-1]["price"] == "96000.0"
    assert jsonl("closed.jsonl")[-1]["why"] == "stop"


def test_h1_tp_closing_switches_to_stop_when_stop_hits(env, monkeypatch):
    """익절 청산을 쫓는 도중(이어서) 손절이 걸리면 손절로 바꿔 바닥 없이 쫓고 손절로 적는다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = _engine_long_with_tp(ex)
    child = ex.trigger_unfilled(pos["stop_id"])
    st = state()
    st["closing"] = {"symbol": "BTCUSDT", "why": "tp", "base": "mnv-x7", "k": 0, "cid": None}
    bt._save(bt.DIR / "state.json", st)
    with_signal(monkeypatch, None)
    run(env, ex, FakeMarket(96_000.0))
    assert ex.limits[child]["status"] == "CANCELED"
    assert ex.amt == 0 and _closes(ex)[-1]["price"] == "96000.0"
    assert jsonl("closed.jsonl")[-1]["why"] == "stop"


def test_h2_tp_wait_does_not_freeze_opposite_signal(env, monkeypatch):
    """익절이 진입가 아래로 걸려 기다리는 중에도 반대 신호 청산·진입은 평소대로 한다."""
    monkeypatch.setattr(bt, "AI_EXITS", False)
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = _engine_long_with_tp(ex)
    ex.trigger_unfilled(pos["tp_id"])
    with_signal(monkeypatch, "short", close=99_000.0)
    s = run(env, ex, FakeMarket(99_000.0))
    assert jsonl("closed.jsonl")[-1]["why"] == "signal"
    assert state()["position"]["side"] == "short" and s["decision"].startswith("BTC 숏 진입")


def test_f4_triggered_stop_still_chased_below_entry(env, monkeypatch):
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    ex.trigger_unfilled(pos["stop_id"])
    with_signal(monkeypatch, None)
    run(env, ex, FakeMarket(96_000.0))
    assert ex.amt == 0 and _closes(ex)[-1]["price"] == "96000.0"


def test_f5_children_only_by_actual_order_id(env):
    """actualOrderId가 없으면 짐작해서 엔진 밖 reduceOnly STOP 주문을 지우지 않는다."""
    ex = FakeExchange()
    ex.amt = 0.029
    ex.limits["user-stop"] = {
        "clientOrderId": "user-stop",
        "status": "NEW",
        "executedQty": "0",
        "avgPrice": "0",
        "price": "95000",
        "origQty": "0.029",
        "side": "SELL",
        "how": "none",
        "reduce": True,
        "tif": "GTC",
        "type": "STOP",
        "orderId": 77,
    }
    assert bt.cancel_exit_children(ex, [{"algoStatus": "TRIGGERED"}]) == []
    assert ex.limits["user-stop"]["status"] == "NEW"
    assert bt.cancel_exit_children(ex, [{"actualOrderId": "77"}]) == ["77"]
    assert ex.limits["user-stop"]["status"] == "CANCELED"


def test_f6_unconfirmed_stop_reused_not_stacked(env, monkeypatch):
    """확인 못 해 stop_id로 못 적은 손절이 걸려 있으면 새로 쌓지 않고 그걸 쓴다. 같은 포지션의
    여분 엔진 손절은 취소해 하나만 둔다."""
    live(env)
    ex = FakeExchange()
    ex.leverage = 3
    pos = engine_long(ex)
    ex.algos.clear()
    pre = f"mnv-{pos['bar'] // 60000}-s"
    for n in (2, 3):
        ex.algos[f"{pre}{n}"] = {
            "clientAlgoId": f"{pre}{n}",
            "algoStatus": "NEW",
            "triggerPrice": "97000.0",
            "type": "STOP",
            "quantity": "0.029",
        }
    st = state()
    st["position"].update(stop_id=f"{pre}1", stop_n=3)
    bt._save(bt.DIR / "state.json", st)
    with_signal(monkeypatch, None)
    run(env, ex)
    assert not [x for m, p, x in ex.writes if m == "POST" and p == "/fapi/v1/algoOrder"]
    p = state()["position"]
    live_stops = [a for a in ex.algos.values() if a["algoStatus"] == "NEW"]
    assert len(live_stops) == 1 and p["stop_id"] == live_stops[0]["clientAlgoId"]
