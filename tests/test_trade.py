import json
from datetime import datetime

import pytest

from minerva import kis, trade

CFG = {"cano": "12345678", "prd": "01", "app_key": "k", "app_secret": "s"}
NOW = datetime(2026, 10, 29, 8, 50, tzinfo=kis.KST)


def _act(action, symbol, name, verified=True, market="KR", ref=10000):
    return {
        "action": action,
        "symbol": symbol,
        "name": name,
        "market": market,
        "verified": verified,
        "date": "2026-10-28",
        "ref_price": ref,
        "label": action,
    }


def _daily(actions, regime="R1"):
    return {"generated_at": "2026-10-28T16:30+09:00", "vix": {"regime": regime}, "actions": actions}


ACCOUNT = {"equity": 1_000_000, "cash_d2": 1_000_000, "positions": [], "orders": []}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "TRADE_DIR", tmp_path)
    monkeypatch.setattr(trade.time, "sleep", lambda s: None)
    monkeypatch.setattr(trade.notify, "send", lambda text: True)


def test_plan_only_domestic_verified_entries():
    daily = _daily(
        [
            _act("LONG_ENTRY", "005930.KS", "삼성전자"),
            _act("LONG_ENTRY", "000660.KS", "하이닉스", verified=False),
            _act("LONG_ENTRY", "NVDA", "엔비디아", market="US"),
            _act("SHORT_ENTRY", "000270.KS", "기아"),
        ]
    )
    orders, _ = trade.plan(daily, ACCOUNT, {})
    assert [(o["code"], o["side"]) for o in orders] == [("005930", "buy")]
    # 20% of 1,000,000 with 3% gap buffer over 10,000원
    assert orders[0]["qty"] == 19


def test_plan_scales_by_regime_and_skips_expensive():
    daily = _daily(
        [_act("LONG_ENTRY", "005930.KS", "A"), _act("LONG_ENTRY", "012450.KS", "B", ref=768000)],
        regime="R2",
    )
    orders, skipped = trade.plan(daily, ACCOUNT, {})
    assert orders[0]["qty"] == 9  # 100,000 / 10,300
    assert any("B" in s and "배정" in s for s in skipped)


def test_plan_caps_positions_and_budget():
    acts = [_act("LONG_ENTRY", f"00000{i}.KS", f"S{i}") for i in range(7)]
    orders, skipped = trade.plan(_daily(acts), ACCOUNT, {})
    assert len(orders) == trade.MAX_POSITIONS
    assert sum(o["qty"] * 10000 * trade.GAP_BUFFER for o in orders) <= 1_000_000
    assert any("한도" in s for s in skipped)


def test_plan_sells_only_bot_owned_positions():
    account = {**ACCOUNT, "positions": [{"code": "005930", "qty": 3}, {"code": "035420", "qty": 1}]}
    daily = _daily(
        [_act("LONG_EXIT", "005930.KS", "삼성전자"), _act("LONG_EXIT", "035420.KS", "네이버")]
    )
    orders, _ = trade.plan(daily, account, {"005930": {"kind": "long"}})
    assert [(o["code"], o["side"], o["qty"]) for o in orders] == [("005930", "sell", 3)]


def test_send_order_dry_does_not_call_api(monkeypatch):
    monkeypatch.setattr(kis, "_request", lambda req: pytest.fail("no API in dry mode"))
    o = trade._order("buy", _act("LONG_ENTRY", "005930.KS", "A"), "005930", 1)
    assert trade.send_order(CFG, o, live=False)["result"] == "dry"


def _live(monkeypatch, posts, sent_check=False):
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    seq = iter(posts)

    def fake_post(cfg, token, path, tr_id, body):
        assert body["ORD_DVSN"] == "01" and body["ORD_UNPR"] == "0"
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(trade, "_post", fake_post)
    monkeypatch.setattr(trade, "_already_sent", lambda cfg, token, o, since: sent_check)
    return trade._order("buy", _act("LONG_ENTRY", "005930.KS", "A"), "005930", 1)


def test_send_order_retries_transport_failure(monkeypatch):
    o = _live(monkeypatch, [kis.KisError("HTTP 500"), {"rt_cd": "0", "output": {"ODNO": "7"}}])
    r = trade.send_order(CFG, o, live=True)
    assert r["result"] == "sent" and r["odno"] == "7" and r["attempts"] == 2


def test_send_order_does_not_duplicate_when_order_landed(monkeypatch):
    o = _live(monkeypatch, [kis.KisError("timeout")], sent_check=True)
    assert trade.send_order(CFG, o, live=True)["result"] == "sent"


def test_send_order_does_not_retry_broker_rejection(monkeypatch):
    o = _live(monkeypatch, [{"rt_cd": "1", "msg_cd": "APBK0952", "msg1": "주문가능금액 초과"}])
    r = trade.send_order(CFG, o, live=True)
    assert r["result"] == "rejected" and "주문가능금액" in r["msg"]


def test_send_order_gives_up_after_retries(monkeypatch):
    o = _live(monkeypatch, [kis.KisError("x")] * (trade.RETRIES + 1))
    assert trade.send_order(CFG, o, live=True)["result"] == "failed"


def test_mode_defaults_to_dry(tmp_path):
    assert trade.mode() == "dry"
    (tmp_path / "mode").write_text("live\n")
    assert trade.mode() == "live"


def test_run_open_respects_halt_and_dedupes(tmp_path, monkeypatch):
    daily_file = tmp_path / "daily.json"
    daily_file.write_text(json.dumps(_daily([_act("LONG_ENTRY", "005930.KS", "A")])))
    monkeypatch.setattr(trade, "DAILY_FILE", daily_file)
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: ACCOUNT)
    (tmp_path / "HALT").write_text("")
    assert trade.run_open(NOW) == []
    (tmp_path / "HALT").unlink()
    assert [r["result"] for r in trade.run_open(NOW)] == ["dry"]
    assert trade.run_open(NOW) == []  # same signal is not sent twice


def test_run_open_skips_stale_analysis(tmp_path, monkeypatch):
    daily = _daily([_act("LONG_ENTRY", "005930.KS", "A")])
    daily["generated_at"] = "2026-10-20T16:30+09:00"
    (tmp_path / "daily.json").write_text(json.dumps(daily))
    monkeypatch.setattr(trade, "DAILY_FILE", tmp_path / "daily.json")
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: pytest.fail("no orders"))
    assert trade.run_open(NOW) == []


def test_watch_sets_stop_then_sells(tmp_path, monkeypatch):
    (tmp_path / "positions.json").write_text(
        json.dumps({"005930": {"name": "A", "kind": "long", "stop": None}})
    )
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    pos = {"code": "005930", "qty": 3, "avg": 10000.0, "price": 9500.0}
    assert trade.run_watch({"positions": [pos]}) == []
    saved = json.loads((tmp_path / "positions.json").read_text())
    assert saved["005930"]["stop"] == 9000
    pos["price"] = 8990.0
    r = trade.run_watch({"positions": [pos]})
    assert r[0]["side"] == "sell" and r[0]["qty"] == 3 and r[0]["result"] == "dry"


def test_inverse_stop_scales_with_leverage():
    assert trade.stop_price("inverse", "252670", 1000) == pytest.approx(860)
    assert trade.stop_price("inverse", "114800", 1000) == pytest.approx(930)


def test_trade_api_reads_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from minerva.main import app

    monkeypatch.setenv("MINERVA_TRADE_DIR", str(tmp_path))
    client = TestClient(app)
    assert client.get("/api/trade").json() == {
        "mode": "dry",
        "halted": False,
        "positions": {},
        "orders": [],
    }
    (tmp_path / "mode").write_text("live")
    (tmp_path / "journal.jsonl").write_text('{"key": "a"}\n{"key": "b"}\n')
    d = client.get("/api/trade").json()
    assert d["mode"] == "live" and [o["key"] for o in d["orders"]] == ["b", "a"]
    assert "/static/trade.js" in client.get("/").text
