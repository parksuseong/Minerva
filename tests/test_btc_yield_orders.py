"""사용자 대기 지정가가 다른 종목 엔진 신호에 양보하는지."""

import json

import pytest

from minerva import btc_trade as bt


class Client:
    def __init__(self, orders):
        self.orders = orders
        self.calls = []

    def request(self, method, path, params=None, signed=True):
        self.calls.append((method, path, dict(params or {})))
        cid = params["origClientOrderId"]
        if method == "GET":
            if cid not in self.orders:
                raise bt.Rejected(-2013, "Order does not exist.")
            return self.orders[cid]
        if method == "DELETE":
            self.orders[cid]["status"] = "CANCELED"
            return self.orders[cid]
        raise AssertionError(method)


class Ctx:
    def __init__(self, symbol, client):
        self.symbol, self.client, self.events = symbol, client, []

    def at(self, symbol):
        return self

    def event(self, **kw):
        self.events.append(kw)


def order(status="NEW", executed="0"):
    return {
        "status": status,
        "executedQty": executed,
        "side": "SELL",
        "origQty": "0.131",
        "price": "61650",
    }


@pytest.fixture
def ydir(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "DIR", tmp_path)
    return tmp_path


def write(d, rows):
    (d / bt.YIELD_FILE).write_text(json.dumps(rows))


def test_other_symbol_signal_cancels_new_order(ydir):
    write(ydir, [{"symbol": "BTCUSDT", "cid": "usr-1"}])
    c = Client({"usr-1": order()})
    gone = bt.yield_standing_orders(Ctx("ETHUSDT", c))
    assert len(gone) == 1 and "61650" in gone[0]
    assert (
        "DELETE",
        "/fapi/v1/order",
        {"symbol": "BTCUSDT", "origClientOrderId": "usr-1"},
    ) in c.calls
    assert json.loads((ydir / bt.YIELD_FILE).read_text()) == []


def test_same_symbol_signal_keeps_order(ydir):
    write(ydir, [{"symbol": "BTCUSDT", "cid": "usr-1"}])
    c = Client({"usr-1": order()})
    assert bt.yield_standing_orders(Ctx("BTCUSDT", c)) == []
    assert not [x for x in c.calls if x[0] == "DELETE"]
    assert len(json.loads((ydir / bt.YIELD_FILE).read_text())) == 1


def test_partly_filled_or_finished_not_cancelled(ydir):
    write(ydir, [{"symbol": "BTCUSDT", "cid": "a"}, {"symbol": "BTCUSDT", "cid": "b"}])
    c = Client({"a": order("PARTIALLY_FILLED", "0.05"), "b": order("FILLED", "0.131")})
    assert bt.yield_standing_orders(Ctx("ETHUSDT", c)) == []
    assert not [x for x in c.calls if x[0] == "DELETE"]
    assert json.loads((ydir / bt.YIELD_FILE).read_text()) == []


def test_no_file_is_noop(ydir):
    c = Client({})
    assert bt.yield_standing_orders(Ctx("ETHUSDT", c)) == []
    assert c.calls == []
