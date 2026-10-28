import json
import math
from datetime import datetime

import pytest

from minerva import kis, trade

CFG = {"cano": "12345678", "prd": "01", "app_key": "k", "app_secret": "s"}
NOW = datetime(2026, 10, 29, 8, 50, tzinfo=kis.KST)
REAL_IS_OPEN_DAY = trade.is_open_day  # autouse 픽스처가 가리기 전 원본


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
    monkeypatch.setattr(trade, "is_open_day", lambda cfg, day: True)
    monkeypatch.setattr(kis, "STATE_DIR", tmp_path / "account")
    # 테스트가 실제 증권사 API를 부르면 바로 실패시킨다
    monkeypatch.setattr(kis, "_request", lambda req: pytest.fail("network call in test"))


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
    acts = [_act("LONG_ENTRY", f"{i:06d}.KS", f"S{i}", ref=1000) for i in range(12)]
    rich = {**ACCOUNT, "equity": 10_000_000, "cash_d2": 10_000_000}
    # R2(종목당 10%)면 10종목까지 찬다. R1(20%)이면 예수금이 5종목에서 먼저 바닥난다.
    orders, skipped = trade.plan(_daily(acts, regime="R2"), rich, {})
    assert len(orders) == trade.MAX_POSITIONS == 10
    assert any("자리 부족" in s for s in skipped)
    # 예수금이 모자라면 한도 전에 예산에서 멈춘다
    orders, _ = trade.plan(_daily(acts), {**ACCOUNT, "cash_d2": 500_000}, {})
    assert sum(o["qty"] * 1000 * trade.GAP_BUFFER for o in orders) <= 500_000


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


def test_run_open_skips_holiday(tmp_path, monkeypatch):
    (tmp_path / "daily.json").write_text(json.dumps(_daily([_act("LONG_ENTRY", "005930.KS", "A")])))
    monkeypatch.setattr(trade, "DAILY_FILE", tmp_path / "daily.json")
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(trade, "is_open_day", lambda cfg, day: False)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: pytest.fail("no orders"))
    assert trade.run_open(NOW) == []


def test_drawdown_blocks_only_new_buys(tmp_path, monkeypatch):
    owned = {"035420": {"name": "N", "kind": "long"}}
    (tmp_path / "positions.json").write_text(json.dumps(owned))
    acc = {**ACCOUNT, "positions": [{"code": "035420", "qty": 2, "pnl": 0}]}
    assert trade.check_drawdown(acc) is False  # 첫 고점(봇 손익 0) 기록
    low = {**acc, "positions": [{"code": "035420", "qty": 2, "pnl": -160_000}]}
    assert trade.check_drawdown(low) is True and (tmp_path / "NO_ENTRY").exists()
    daily = _daily([_act("LONG_ENTRY", "005930.KS", "A"), _act("LONG_EXIT", "035420.KS", "N")])
    (tmp_path / "daily.json").write_text(json.dumps(daily))
    monkeypatch.setattr(trade, "DAILY_FILE", tmp_path / "daily.json")
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: low)
    assert [(r["code"], r["side"]) for r in trade.run_open(NOW)] == [("035420", "sell")]


def test_drawdown_ignores_deposits_and_withdrawals():
    trade.check_drawdown({"equity": 1_000_000, "positions": []})
    # 출금으로 평가액이 반토막 나도 봇 손익은 그대로라 막지 않는다
    assert trade.check_drawdown({"equity": 500_000, "positions": []}) is False


def test_drawdown_counts_realized_losses(tmp_path):
    trade.check_drawdown({"equity": 1_000_000, "positions": []})
    (tmp_path / "closed.jsonl").write_text(json.dumps({"key": "k", "pnl": -200_000}) + "\n")
    assert trade.check_drawdown({"equity": 800_000, "positions": []}) is True


def test_reconcile_drops_unfilled_and_rewatches_unsold(tmp_path, monkeypatch):
    (tmp_path / "positions.json").write_text(
        json.dumps({"005930": {"name": "A", "kind": "long", "stop": None}})
    )
    (tmp_path / "journal.jsonl").write_text(
        json.dumps(
            {
                "key": "2026-10-28:035420:sell",
                "side": "sell",
                "result": "sent",
                "code": "035420",
                "name": "N",
                "at": "2026-10-29T08:50:00",
            }
        )
        + "\n"
    )
    acc = {
        **ACCOUNT,
        "positions": [{"code": "035420", "qty": 2}],
        "orders": [{"code": "005930", "status": "미체결"}],
    }
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: acc)
    notes = trade.run_reconcile(NOW)
    owned = json.loads((tmp_path / "positions.json").read_text())
    assert list(owned) == ["035420"] and owned["035420"]["stop"] is None
    assert any("미체결" in n for n in notes) and any("2주 남음" in n for n in notes)


def test_selftest_requires_yes_flag(capsys):
    with pytest.raises(SystemExit):
        trade.main(["selftest"])
    assert "--yes" in capsys.readouterr().err


def test_is_open_day_parses_holiday_api(monkeypatch):
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    out = [{"bass_dt": "20261029", "opnd_yn": "Y"}, {"bass_dt": "20261031", "opnd_yn": "N"}]
    monkeypatch.setattr(kis, "_request", lambda req: ({"rt_cd": "0", "output": out}, {}))
    monkeypatch.setattr(kis.time, "sleep", lambda s: None)
    assert REAL_IS_OPEN_DAY(CFG, NOW) is True
    assert REAL_IS_OPEN_DAY(CFG, datetime(2026, 10, 31, tzinfo=kis.KST)) is False


def test_inverse_signal_sells_bot_longs_first():
    account = {
        **ACCOUNT,
        "cash_d2": 10_000,
        "positions": [
            {"code": "005930", "qty": 3, "price": 50000},
            {"code": "035420", "qty": 1, "price": 9000},
        ],
    }
    owned = {"005930": {"name": "삼성", "kind": "long"}}  # 035420은 사용자가 직접 산 종목
    daily = _daily([_act("BUY", "114800.KS", "KODEX 인버스", ref=1600)])
    orders, _ = trade.plan(daily, account, owned)
    assert [(o["side"], o["code"]) for o in orders] == [("sell", "005930"), ("buy", "114800")]
    assert orders[0]["action"] == "SWITCH"
    # 예산 = 예수금 10,000 + 매도 대금 150,000 (배정 200,000보다 작아 예산이 묶는다)
    assert orders[1]["qty"] == math.floor(160_000 / (1600 * trade.GAP_BUFFER))


def test_buys_wait_for_sells_until_after_open(tmp_path, monkeypatch):
    held = [{"code": "005930", "qty": 3, "price": 50000}]
    account = {**ACCOUNT, "positions": held}
    (tmp_path / "positions.json").write_text(
        json.dumps({"005930": {"name": "삼성", "kind": "long"}})
    )
    daily = _daily([_act("LONG_EXIT", "005930.KS", "삼성"), _act("LONG_ENTRY", "000660.KS", "H")])
    (tmp_path / "daily.json").write_text(json.dumps(daily))
    monkeypatch.setattr(trade, "DAILY_FILE", tmp_path / "daily.json")
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: account)
    first = trade.run_open(NOW)
    assert [(r["side"], r["code"]) for r in first] == [("sell", "005930")]
    account["positions"] = []  # 09:00 시가에 매도 체결
    second = trade.run_open(NOW.replace(hour=9, minute=2))
    assert [(r["side"], r["code"]) for r in second] == [("buy", "000660")]
    assert trade.run_open(NOW.replace(hour=9, minute=2)) == []


def test_rejected_buy_is_not_resent_at_0902(tmp_path, monkeypatch):
    (tmp_path / "journal.jsonl").write_text(
        json.dumps({"key": "2026-10-28:005930:buy", "result": "rejected"}) + "\n"
    )
    (tmp_path / "daily.json").write_text(json.dumps(_daily([_act("LONG_ENTRY", "005930.KS", "A")])))
    monkeypatch.setattr(trade, "DAILY_FILE", tmp_path / "daily.json")
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: ACCOUNT)
    assert trade.run_open(NOW.replace(hour=9, minute=2)) == []


def test_preflight_flags_stale_analysis_and_dry_mode(monkeypatch):
    monkeypatch.setattr(trade, "_timer_active", lambda unit: True)
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: ACCOUNT)
    stale = _daily([])
    stale["generated_at"] = "2026-10-27T16:30+09:00"  # 목요일 08:40 기준 40시간 전
    checks = trade.preflight_checks(NOW.replace(minute=40), stale, lambda: CFG)
    bad = [m for ok, m in checks if not ok]
    assert any("모의" in m for m in bad) and any("일일 분석" in m for m in bad)


def test_preflight_passes_when_ready(tmp_path, monkeypatch):
    (tmp_path / "mode").write_text("live")
    monkeypatch.setattr(trade, "_timer_active", lambda unit: True)
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: ACCOUNT)
    checks = trade.preflight_checks(NOW.replace(minute=40), _daily([]), lambda: CFG)
    assert all(ok for ok, _ in checks)


def test_preflight_monday_accepts_friday_analysis(tmp_path, monkeypatch):
    (tmp_path / "mode").write_text("live")
    monkeypatch.setattr(trade, "_timer_active", lambda unit: False)
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: ACCOUNT)
    friday = _daily([])
    friday["generated_at"] = "2026-10-30T16:31+09:00"
    monday = datetime(2026, 11, 2, 8, 40, tzinfo=kis.KST)
    checks = dict((m, ok) for ok, m in trade.preflight_checks(monday, friday, lambda: CFG))
    assert checks["일일 분석: 2026-10-30T16:31"] is True
    assert checks["타이머 꺼짐: minerva-trade-open.timer"] is False


def test_priority_spends_cash_on_strongest_evidence_first():
    weak = {**_act("LONG_ENTRY", "000270.KS", "약함"), "p_value": 0.2, "net_edge": 0.02}
    strong = {**_act("LONG_ENTRY", "086790.KS", "강함"), "p_value": 0.006, "net_edge": 0.005}
    account = {**ACCOUNT, "cash_d2": 150_000}
    orders, skipped = trade.plan(_daily([weak, strong]), account, {})
    assert [o["name"] for o in orders] == ["강함"]
    assert any("약함" in s for s in skipped)


def test_index_etf_gets_larger_weight_and_inverse_only_one():
    etf = _act("LONG_ENTRY", "069500.KS", "KODEX 200", ref=88905)
    inv1 = {**_act("BUY", "114800.KS", "KODEX 인버스", ref=1600), "p_value": 0.01}
    inv2 = {**_act("BUY", "252670.KS", "곱버스", ref=600), "p_value": 0.01}
    orders, skipped = trade.plan(_daily([inv2, etf, inv1]), ACCOUNT, {})
    by = {o["code"]: o["qty"] for o in orders}
    assert by["069500"] == math.floor(400_000 / (88905 * trade.GAP_BUFFER))  # 40%
    assert "114800" in by and "252670" not in by  # 같은 신호면 1배 하나만
    assert by["114800"] == math.floor(200_000 / (1600 * trade.GAP_BUFFER))
    assert any("곱버스" in s and "한 상품만" in s for s in skipped)


def test_inverse_cap_counts_held_inverse():
    account = {
        **ACCOUNT,
        "positions": [{"code": "252670", "qty": 300, "price": 600, "value": 180_000}],
    }
    inv1 = _act("BUY", "114800.KS", "KODEX 인버스", ref=1600)
    orders, _ = trade.plan(_daily([inv1]), account, {})
    assert orders[0]["qty"] == math.floor(20_000 / (1600 * trade.GAP_BUFFER))


def test_rebalance_trims_overweight_bot_position_only():
    account = {
        **ACCOUNT,
        "positions": [
            {"code": "086790", "qty": 10, "price": 35000, "value": 350_000, "avg": 20000},
            {"code": "035420", "qty": 10, "price": 50000, "value": 500_000, "avg": 20000},
        ],
    }
    owned = {"086790": {"name": "강함", "kind": "long", "opened": "2026-10-01"}}
    orders, _ = trade.plan(_daily([]), account, owned, today="2026-10-29")
    # 35% > 20%×1.5 → 20%(200,000)까지: ceil(150,000/35,000)=5주. 사용자 종목(035420)은 그대로
    assert [(o["code"], o["qty"], o["action"]) for o in orders] == [("086790", 5, "TRIM")]
    assert orders[0]["signal_date"] == "2026-10-29-trim"


def test_trim_keeps_position_tracked():
    owned = {"086790": {"name": "A", "kind": "long"}}
    trade._track(owned, {"side": "sell", "code": "086790", "action": "TRIM"})
    assert "086790" in owned
    trade._track(owned, {"side": "sell", "code": "086790", "action": "LONG_EXIT"})
    assert owned == {}


def _rotation_setup(new_p, opened="2026-10-01"):
    daily = _daily(
        [{**_act("LONG_ENTRY", "086790.KS", "강함", ref=50000), "p_value": new_p, "net_edge": 0.02}]
    )
    daily["assets"] = [
        {"symbol": "000270.KS", "verify": {"long": {"p_value": 0.2, "net_edge": 0.001}}}
    ]
    account = {
        **ACCOUNT,
        "cash_d2": 0,
        "positions": [
            {"code": "000270", "qty": 2, "price": 100000, "value": 200_000, "avg": 95000}
        ],
    }
    owned = {"000270": {"name": "약함", "kind": "long", "opened": opened}}
    return daily, account, owned


def test_rotation_sells_weaker_holding_for_stronger_signal():
    daily, account, owned = _rotation_setup(0.006)
    orders, _ = trade.plan(daily, account, owned, today="2026-10-29")
    assert [(o["side"], o["code"], o["action"]) for o in orders] == [
        ("sell", "000270", "ROTATE"),
        ("buy", "086790", "LONG_ENTRY"),
    ]


def test_rotation_needs_clearly_better_evidence_and_min_hold():
    daily, account, owned = _rotation_setup(0.15)  # 0.15 > 0.2 × 0.5
    orders, skipped = trade.plan(daily, account, owned, today="2026-10-29")
    assert orders == [] and any("교체할 만큼" in s for s in skipped)
    daily, account, owned = _rotation_setup(0.006, opened="2026-10-27")  # 2일 보유
    orders, _ = trade.plan(daily, account, owned, today="2026-10-29")
    assert orders == []


def test_rotation_skips_when_selling_still_cannot_fund():
    daily, account, owned = _rotation_setup(0.006)
    daily["actions"][0]["ref_price"] = 400_000  # 배정 200,000 < 1주 → 교체 대상도 아님
    account["positions"][0].update(value=10_000, qty=1, price=10_000)
    orders, skipped = trade.plan(daily, account, owned, today="2026-10-29")
    assert orders == []


def test_order_line_shows_reason():
    r = {"side": "sell", "result": "sent", "name": "A", "code": "1", "qty": 2, "action": "TRIM"}
    assert "[리밸런싱]" in trade._line(r)
