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
    # 계산 검증용 비중은 고정한다 (운용 비중을 바꿔도 테스트 숫자가 그대로이게)
    monkeypatch.setattr(trade, "WEIGHT", 0.20)
    monkeypatch.setattr(trade, "INDEX_ETF_WEIGHT", 0.40)
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
        # 매수는 전일 종가(10,000) +3% 지정가
        assert body["ORD_DVSN"] == "00" and body["ORD_UNPR"] == "10300"
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
        json.dumps(
            {
                "005930": {"name": "A", "kind": "long", "stop": None, "qty": 3},
                "035420": {"name": "N", "kind": "long", "stop": 9, "qty": 2, "selling": 2},
                "000660": {"name": "H", "kind": "long", "stop": 9, "qty": 1, "selling": 1},
            }
        )
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
    assert list(owned) == ["035420"] and "selling" not in owned["035420"]
    assert any("미체결" in n for n in notes) and any("2주 남음" in n for n in notes)
    assert not any("H" in n for n in notes)  # 체결된 매도는 조용히 정리


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


def test_index_etf_gets_larger_weight():
    etf = _act("LONG_ENTRY", "069500.KS", "KODEX 200", ref=88905)
    stock = _act("LONG_ENTRY", "005930.KS", "삼성전자", ref=10000)
    orders, _ = trade.plan(_daily([etf, stock]), ACCOUNT, {})
    by = {o["code"]: o["qty"] for o in orders}
    assert by["069500"] == math.floor(400_000 / (88905 * trade.GAP_BUFFER))  # 40%
    assert by["005930"] == math.floor(200_000 / (10000 * trade.GAP_BUFFER))  # 20%


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


def test_track_trim_reduces_and_exit_waits_for_fill():
    owned = {"086790": {"name": "A", "kind": "long", "qty": 10}}
    trade._track(owned, {"side": "sell", "code": "086790", "action": "TRIM", "qty": 3})
    assert owned["086790"]["qty"] == 7
    trade._track(
        owned,
        {
            "side": "sell",
            "code": "086790",
            "action": "LONG_EXIT",
            "qty": 7,
            "at": "2026-10-29T08:50:00",
        },
    )
    assert owned["086790"]["selling"] == 7  # 체결 확인 전까지는 지우지 않는다


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


def test_sells_only_bot_quantity_when_user_added_shares():
    account = {
        **ACCOUNT,
        "positions": [{"code": "005930", "qty": 60, "price": 50000, "avg": 48000}],
    }
    owned = {"005930": {"name": "삼성", "kind": "long", "qty": 10}}
    orders, _ = trade.plan(_daily([_act("LONG_EXIT", "005930.KS", "삼성")]), account, owned)
    assert [(o["code"], o["qty"]) for o in orders] == [("005930", 10)]


def test_watch_stop_sells_bot_share_and_waits_for_fill(tmp_path, monkeypatch):
    (tmp_path / "mode").write_text("live")
    (tmp_path / "positions.json").write_text(
        json.dumps({"005930": {"name": "A", "kind": "long", "stop": 9000, "qty": 3}})
    )
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    sent = []
    monkeypatch.setattr(
        trade, "send_order", lambda cfg, o, live: sent.append(o) or {**o, "result": "sent"}
    )
    pos = {"code": "005930", "qty": 8, "avg": 10000.0, "price": 8990.0}
    trade.run_watch({"positions": [pos]})
    assert sent[0]["qty"] == 3
    trade.run_watch({"positions": [pos]})  # 체결 전: 다시 내지 않는다
    assert len(sent) == 1
    trade.run_watch({"positions": [{**pos, "qty": 5}]})  # 봇 몫 3주 체결
    assert json.loads((tmp_path / "positions.json").read_text()) == {}


def test_drawdown_block_keeps_exits_but_drops_buys():
    account = {**ACCOUNT, "positions": [{"code": "005930", "qty": 3, "price": 50000}]}
    owned = {"005930": {"name": "삼성", "kind": "long", "qty": 3}}
    daily = _daily(
        [_act("LONG_EXIT", "005930.KS", "삼성"), _act("LONG_ENTRY", "000660.KS", "하이닉스")]
    )
    orders, skipped = trade.plan(daily, account, owned, allow_buys=False)
    assert [(o["code"], o["side"]) for o in orders] == [("005930", "sell")]
    assert any("손실 한도" in s for s in skipped)


def test_inverse_actions_are_ignored():
    daily = _daily([_act("BUY", "114800.KS", "KODEX 인버스", ref=1600)])
    assert trade.plan(daily, ACCOUNT, {}) == ([], [])


def test_lock_is_exclusive_between_processes(tmp_path):
    import subprocess
    import sys

    held = trade._lock(wait=False)
    assert held is not None
    code = (
        "import sys; from pathlib import Path; from minerva import trade; "
        f"trade.TRADE_DIR = Path({str(tmp_path)!r}); "
        "sys.exit(0 if trade._lock(wait=False) is None else 1)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    held.close()
    assert subprocess.run([sys.executable, "-c", code]).returncode == 1


def test_unpark_funds_buys_and_defers_them():
    account = {
        **ACCOUNT,
        "cash_d2": 20_000,
        "positions": [{"code": trade.PARK_CODE, "qty": 16, "price": 58000, "value": 928_000}],
    }
    daily = _daily([_act("LONG_ENTRY", "069500.KS", "KODEX 200", ref=88905)])
    orders, _ = trade.plan(daily, account, {}, park_qty=16)
    assert orders[0]["code"] == trade.PARK_CODE and orders[0]["action"] == "UNPARK"
    buy = orders[1]
    assert buy["code"] == "069500" and buy["qty"] == math.floor(400_000 / (88905 * 1.03))
    need = buy["qty"] * 88905 * 1.03 - 20_000
    assert orders[0]["qty"] == math.ceil(need * 1.01 / 58000)


def test_no_unpark_when_cash_suffices_or_user_owns_etf():
    daily = _daily([_act("LONG_ENTRY", "069500.KS", "KODEX 200", ref=88905)])
    account = {**ACCOUNT, "positions": [{"code": trade.PARK_CODE, "qty": 16, "price": 58000}]}
    orders, _ = trade.plan(daily, account, {}, park_qty=16)
    assert [o["code"] for o in orders] == ["069500"]
    account["cash_d2"] = 0
    orders, _ = trade.plan(daily, account, {}, park_qty=0)  # 사용자가 산 CD ETF는 안 판다
    assert orders == []


def test_park_idle_cash_once_a_day(tmp_path, monkeypatch):
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: {**ACCOUNT, "cash_d2": 600_000})
    monkeypatch.setattr(trade, "quote", lambda cfg, code: 58000.0)
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    monkeypatch.setattr(trade, "buyable_qty", lambda cfg, t, code, price=None: 100)
    monkeypatch.setattr(trade, "send_order", lambda cfg, o, live: {**o, "result": "sent"})
    now = NOW.replace(hour=9, minute=2)
    r = trade.park_idle_cash(CFG, now, set())
    # (600,000 − 2% × 1,000,000) / (58,000 × 1.002) → 9주
    assert r["qty"] == 9 and json.loads((tmp_path / "park.json").read_text()) == {"qty": 9}
    assert trade.park_idle_cash(CFG, now, {r["key"]}) is None


def test_thin_stock_order_capped_by_trading_value():
    daily = _daily([_act("LONG_ENTRY", "006040.KS", "동원산업", ref=1000)])
    daily["assets"] = [{"symbol": "006040.KS", "name": "동원산업", "adv": 5_000_000}]
    rich = {**ACCOUNT, "equity": 10_000_000, "cash_d2": 10_000_000}
    orders, _ = trade.plan(daily, rich, {})
    # 20% 배정(2,000,000)이 아니라 거래대금 1%(50,000)까지만: 50,000 / 1,030
    assert orders[0]["qty"] == 48


def test_duplicate_signal_is_bought_once():
    a = _act("LONG_ENTRY", "005930.KS", "삼성")
    orders, skipped = trade.plan(_daily([a, dict(a)]), ACCOUNT, {})
    assert [o["code"] for o in orders] == ["005930"]
    assert any("이미 보유" in s for s in skipped)


def test_thin_stock_does_not_trigger_rotation():
    daily = _daily([{**_act("LONG_ENTRY", "006040.KS", "동원", ref=1000), "p_value": 0.001}])
    daily["assets"] = [{"symbol": "006040.KS", "name": "동원", "adv": 5_000_000}]
    account = {
        **ACCOUNT,
        "cash_d2": 0,
        "positions": [{"code": "035420", "qty": 10, "price": 20000, "value": 200_000}],
    }
    owned = {"035420": {"name": "N", "kind": "long", "qty": 10, "opened": "2026-10-01"}}
    orders, skipped = trade.plan(daily, account, owned, today="2026-10-29")
    assert orders == [] and any("거래대금 1% 한도" in s for s in skipped)


def test_limit_price_follows_tick_size():
    assert trade.limit_price(10000 * 1.03) == 10300
    assert trade.limit_price(245000 * 1.03) == 252000  # 252,350 → 500원 단위 내림
    assert trade.limit_price(1122 * 1.03) == 1155  # 5원 미만 단위도 5원 (ETF 호가)
    assert (
        trade.limit_price(58285 * trade.PARK_SLIP, up=True) == 58500
    )  # 지정가지만 매도 호가에 바로 체결
    assert trade.limit_price(600000) == 600000


def test_buys_are_limit_and_sells_are_market(monkeypatch):
    buy = trade._order("buy", _act("LONG_ENTRY", "005930.KS", "A"), "005930", 1)
    sell = trade._order("sell", _act("LONG_EXIT", "005930.KS", "A"), "005930", 1)
    assert buy["limit"] == 10300 and "limit" not in sell
    assert "지정가 10,300원" in trade._line({**buy, "result": "sent"})
    assert "시장가" in trade._line({**sell, "result": "sent"})
    posts = []
    monkeypatch.setattr(kis, "access_token", lambda cfg, now=None: "t")
    monkeypatch.setattr(
        trade, "_post", lambda c, t, p, tr, body: posts.append(body) or {"rt_cd": "0"}
    )
    trade.send_order(CFG, sell, live=True)
    assert posts[0]["ORD_DVSN"] == "01" and posts[0]["ORD_UNPR"] == "0"


def test_partial_limit_fill_is_followed_then_reported(tmp_path, monkeypatch):
    owned = {"005930": {"name": "A", "kind": "long", "stop": None, "qty": 10, "ordered": 10}}
    (tmp_path / "positions.json").write_text(json.dumps(owned))
    acc = {**ACCOUNT, "positions": [{"code": "005930", "qty": 4, "avg": 10000, "price": 10000}]}
    trade.run_watch(acc)
    pos = json.loads((tmp_path / "positions.json").read_text())["005930"]
    assert pos["qty"] == 4 and pos["stop"] == 9000
    acc["positions"][0]["qty"] = 6  # 장중에 더 체결
    trade.run_watch(acc)
    assert json.loads((tmp_path / "positions.json").read_text())["005930"]["qty"] == 6
    acc["positions"][0]["qty"] = 7  # 마감 동시호가 체결
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: acc)
    notes = trade.run_reconcile(NOW)
    pos = json.loads((tmp_path / "positions.json").read_text())["005930"]
    assert pos["qty"] == 7 and "ordered" not in pos
    assert any("10주 중 7주 체결" in n for n in notes)


def test_plan_follows_exit_even_after_verification_dropped():
    account = {**ACCOUNT, "positions": [{"code": "005930", "qty": 3}]}
    daily = _daily([_act("LONG_EXIT", "005930.KS", "삼성전자", verified=False)])
    orders, _ = trade.plan(daily, account, {"005930": {"kind": "long"}})
    assert [(o["code"], o["side"], o["action"]) for o in orders] == [
        ("005930", "sell", "LONG_EXIT")
    ]


def test_plan_never_orders_index_symbols():
    daily = _daily(
        [
            _act("LONG_ENTRY", "^KS11", "코스피", ref=6000),
            _act("LONG_ENTRY", "069500.KS", "KODEX 200"),
        ]
    )
    orders, _ = trade.plan(daily, ACCOUNT, {})
    assert [o["code"] for o in orders] == ["069500"]


def test_watch_saves_sent_stop_even_if_next_order_raises(tmp_path, monkeypatch):
    (tmp_path / "mode").write_text("live")
    (tmp_path / "positions.json").write_text(
        json.dumps(
            {
                "005930": {"name": "A", "kind": "long", "stop": 9000, "qty": 3},
                "000660": {"name": "B", "kind": "long", "stop": 9000, "qty": 2},
            }
        )
    )
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    sent = []

    def send(cfg, o, live):
        if o["code"] == "000660":
            raise kis.KisError("token limit")
        sent.append(o)
        return {**o, "result": "sent"}

    monkeypatch.setattr(trade, "send_order", send)
    held = [
        {"code": "005930", "qty": 3, "avg": 10000.0, "price": 8990.0},
        {"code": "000660", "qty": 2, "avg": 10000.0, "price": 8990.0},
    ]
    with pytest.raises(kis.KisError):
        trade.run_watch({"positions": held})
    saved = json.loads((tmp_path / "positions.json").read_text())
    assert saved["005930"].get("selling") == 3  # A 손절을 냈다는 표시가 남는다
    monkeypatch.setattr(
        trade, "send_order", lambda cfg, o, live: sent.append(o) or {**o, "result": "sent"}
    )
    trade.run_watch({"positions": held})
    assert [o["code"] for o in sent] == ["005930", "000660"]  # A를 또 내지 않는다


def test_stop_key_does_not_collide_with_same_day_exit():
    stop = {"signal_date": "2026-10-30-stop", "code": "005930", "side": "sell"}
    exit_ = {"signal_date": "2026-10-30", "code": "005930", "side": "sell"}
    assert trade.order_key(stop) != trade.order_key(exit_)


def test_preview_plan_counts_parked_cash_and_entry_block(tmp_path):
    (tmp_path / "park.json").write_text(json.dumps({"qty": 16}))
    daily = _daily([_act("LONG_ENTRY", "005930.KS", "A", ref=50000)])
    account = {
        **ACCOUNT,
        "cash_d2": 20_000,
        "positions": [{"code": trade.PARK_CODE, "qty": 16, "price": 58000, "value": 928_000}],
    }
    orders, _ = trade._plan_ahead(daily, account)
    assert any(o["action"] == "UNPARK" for o in orders)  # 파킹 자금을 풀어 산다
    (tmp_path / "NO_ENTRY").write_text("")
    orders, skipped = trade._plan_ahead(daily, account)
    assert not any(o["side"] == "buy" for o in orders) and any("중지" in s for s in skipped)


def test_unfilled_exit_is_retried_next_day(tmp_path, monkeypatch):
    (tmp_path / "positions.json").write_text(
        json.dumps(
            {
                "005930": {
                    "name": "A",
                    "kind": "long",
                    "qty": 3,
                    "stop": 9000,
                    "selling": 3,
                    "sell_action": "LONG_EXIT",
                }
            }
        )
    )
    acc = {**ACCOUNT, "positions": [{"code": "005930", "qty": 3, "price": 10000}]}
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: acc)
    trade.run_reconcile(NOW)
    owned = json.loads((tmp_path / "positions.json").read_text())
    assert owned["005930"]["exit_pending"] == "LONG_EXIT"
    orders, _ = trade.plan(_daily([]), acc, owned, today="2026-10-30")
    assert [(o["code"], o["action"]) for o in orders] == [("005930", "EXIT_RETRY")]
    assert trade.order_key(orders[0]) == "2026-10-30-exit_retry:005930:sell"


def test_rejected_exit_marks_pending(tmp_path, monkeypatch):
    (tmp_path / "positions.json").write_text(
        json.dumps({"005930": {"name": "A", "kind": "long", "qty": 3, "stop": 9000}})
    )
    (tmp_path / "mode").write_text("live")
    monkeypatch.setattr(trade, "DAILY_FILE", tmp_path / "daily.json")
    daily = {**_daily([_act("LONG_EXIT", "005930.KS", "A")]), "generated_at": NOW.isoformat()}
    (tmp_path / "daily.json").write_text(json.dumps(daily))
    acc = {**ACCOUNT, "positions": [{"code": "005930", "qty": 3, "price": 10000}]}
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: acc)
    monkeypatch.setattr(trade, "send_order", lambda cfg, o, live: {**o, "result": "rejected"})
    trade.run_open(NOW)
    owned = json.loads((tmp_path / "positions.json").read_text())
    assert owned["005930"]["exit_pending"] == "LONG_EXIT"


def test_max_hold_sells_without_signal():
    acc = {**ACCOUNT, "positions": [{"code": "005930", "qty": 3, "price": 10000}]}
    owned = {"005930": {"name": "A", "kind": "long", "qty": 3, "opened": "2026-09-01"}}
    orders, _ = trade.plan(_daily([]), acc, owned, today="2026-10-06")  # 평일 25일
    assert [o["action"] for o in orders] == ["MAX_HOLD"]
    owned["005930"]["opened"] = "2026-09-02"
    assert trade.plan(_daily([]), acc, owned, today="2026-10-06")[0] == []
    assert trade.weekdays_between("2026-10-30", "2026-11-02") == 1  # 금 → 월


def test_reconcile_clamps_park_record_to_real_holding(tmp_path, monkeypatch):
    (tmp_path / "park.json").write_text(json.dumps({"qty": 167}))
    acc = {**ACCOUNT, "positions": [{"code": trade.PARK_CODE, "qty": 160, "price": 58000}]}
    monkeypatch.setattr(kis, "load_config", lambda: CFG)
    monkeypatch.setattr(kis, "_snapshot", lambda cfg, now=None: acc)
    notes = trade.run_reconcile(NOW)
    assert json.loads((tmp_path / "park.json").read_text())["qty"] == 160
    assert any("파킹" in n for n in notes)


def test_skipped_order_run_alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(trade.notify, "send", lambda text: sent.append(text) or True)
    monkeypatch.setattr(trade, "_lock", lambda wait: None)
    assert trade.main(["open"]) == 0 and "건너뛰었어요" in sent[0]
    sent.clear()
    trade.main(["watch"])
    assert sent == []
