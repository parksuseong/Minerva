from datetime import date, datetime, timedelta, timezone

from minerva.strategy.daily import completed, fear_status, telegram_summary
from minerva.strategy.indicators import Bar


def bars_until(last: date, n: int = 3):
    return [
        Bar(day=last - timedelta(days=n - 1 - i), open=1, high=1, low=1, close=1, volume=100)
        for i in range(n)
    ]


def test_us_bar_dropped_while_market_open():
    now = datetime(2026, 10, 27, 16, 20, tzinfo=timezone.utc)  # 뉴욕 12:20
    assert completed(bars_until(date(2026, 10, 27)), "US", now)[-1].day == date(2026, 10, 26)
    later = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)  # 뉴욕 17:00
    assert completed(bars_until(date(2026, 10, 27)), "US", later)[-1].day == date(2026, 10, 27)


def test_kr_bar_kept_after_close():
    now = datetime(2026, 10, 27, 7, 0, tzinfo=timezone.utc)  # 서울 16:00
    assert completed(bars_until(date(2026, 10, 27)), "KR", now)[-1].day == date(2026, 10, 27)
    early = datetime(2026, 10, 27, 1, 0, tzinfo=timezone.utc)  # 서울 10:00
    assert completed(bars_until(date(2026, 10, 27)), "KR", early)[-1].day == date(2026, 10, 26)


def test_crypto_drops_today_utc():
    now = datetime(2026, 10, 27, 3, 0, tzinfo=timezone.utc)
    assert completed(bars_until(date(2026, 10, 27)), "CRYPTO", now)[-1].day == date(2026, 10, 26)


def test_fear_signal_detected_on_entry_day():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(3)]
    vix = [Bar(d, 1, 1, 1, c) for d, c in zip(days, [10.0, 10.0, 11.0])]
    vix3m = [Bar(d, 1, 1, 1, 10.0) for d in days]
    out = fear_status(vix, vix3m)
    assert out["last_signal"] == "2026-01-03" and out["state"].startswith("신호 발생")


def _data(**kw):
    base = {
        "generated_at": "2026-10-28T09:10",
        "vix": {
            "close": 18.4,
            "regime": "R2",
            "size_limit": "계획의 1/2",
            "ma50": 16.0,
            "gate": {"w5": 17.2, "w10": 16.1, "open": False},
        },
        "fear": {"state": "없음", "premium": -0.08},
        "assets": [],
        "actions": [],
    }
    base.update(kw)
    return base


def test_telegram_without_actions_says_nothing_to_do():
    text = telegram_summary(_data())
    assert "오늘 할 일: 없음" in text and "17.2/16.1 닫힘" in text


def test_actions_from_entries_and_exits():
    from minerva.strategy.daily import make_actions, telegram_summary

    ver = {"long": {"label": "검증됨"}, "short": {"label": "미검증"}}
    asset = {
        "symbol": "SPY",
        "name": "SPY",
        "market": "US",
        "date": "2026-10-27",
        "verify": ver,
        "tm": {"state": "진입 신호", "ref_price": 100, "stop": 90},
        "short": {
            "state": "청산 신호",
            "reason": "익절 조건",
            "pnl": 0.03,
            "exit": "다음 거래일 시가 청산",
        },
    }
    inv = {
        "symbol": "QID",
        "name": "QID",
        "market": "US",
        "date": "2026-10-27",
        "kind": "inv",
        "verify": {"short": {"label": "검증됨"}},
        "tm": {"state": "진입 신호", "stop_px": 107},
    }
    acts = make_actions([asset], [inv])
    # 종목 자체 숏(SHORT_*)은 실행할 수 없어 주문 목록에 넣지 않는다
    assert [a["action"] for a in acts] == ["BUY", "LONG_ENTRY"]
    assert acts[0]["stop"] == 107 and acts[1]["stop"] == 90
    text = telegram_summary(_data(actions=acts))
    assert "🎯 오늘 할 일 2건" in text and "롱 진입 SPY [✅]" in text


def test_inverse_follows_underlying_short(monkeypatch):
    from minerva.strategy import daily

    bars = bars_until(date(2026, 10, 20), 5)
    monkeypatch.setattr(daily, "fetch_yahoo", lambda s, r: bars)
    monkeypatch.setattr(daily, "short_status", lambda b: {"state": "진입 신호", "stop_px": 1})
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: False)
    monkeypatch.setattr(daily, "verify", lambda s, b, c: {"short": {"label": "검증됨"}})
    inv = daily.Asset("QID", "QID", "US", "inv", "QQQ", -2)
    out = daily.product_status(inv, bars, {"QQQ": "나스닥100"})
    assert out["tm"]["state"] == "진입 신호" and out["underlying"] == "나스닥100"
    # 레버리지·곱버스는 다루지 않는다: 인버스도 지수 ETF도 1배만
    assert all(
        (p.kind == "inv" and p.mult == -1) or (p.kind == "etf" and p.mult == 1)
        for p in daily.PRODUCTS
    )


def test_product_uses_own_price_for_orders(monkeypatch):
    from minerva.strategy import daily

    bars = bars_until(date(2026, 10, 20), 5)
    monkeypatch.setattr(daily, "fetch_yahoo", lambda s, r: bars)
    entry = {"state": "진입 신호", "ref_price": 5638.0, "stop": 5000}
    monkeypatch.setattr(daily, "short_status", lambda b: dict(entry))
    monkeypatch.setattr(daily, "tm_status", lambda b, c: dict(entry))
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: False)
    raw = {
        "label": "검증됨",
        "trades": 90,
        "edge": 0.01,
        "edge_in": 0.01,
        "edge_out": 0.01,
        "p_value": 0.01,
    }
    monkeypatch.setattr(daily, "verify", lambda s, b, c: {"short": raw, "long": raw})
    inv = daily.product_status(daily.Asset("114800.KS", "I", "KR", "inv", "^KS11", -1), bars, {})
    etf = daily.product_status(daily.Asset("069500.KS", "E", "KR", "etf", "^KS11", 1), bars, {})
    close = bars[-1].close
    assert inv["tm"]["ref_price"] == round(close, 2) and etf["tm"]["ref_price"] == round(close, 2)
    assert etf["tm"]["stop"] == round(close * 0.9, 2) and "long" in etf["verify"]
    acts = daily.make_actions([], [inv, etf])
    assert sorted(a["action"] for a in acts) == ["BUY", "LONG_ENTRY"]


def test_verdict_labels():
    from minerva.strategy.daily import _verdict

    def res(n, i, o):
        return {
            "all": {"trades": n, "edge": (i + o) / 2},
            "in_sample": {"edge": i},
            "out_of_sample": {"edge": o},
        }

    assert _verdict(res(40, 0.01, 0.002))["label"] == "검증됨"
    assert _verdict(res(40, -0.01, -0.002))["label"] == "부적합"
    assert _verdict(res(40, 0.01, -0.002))["label"] == "미검증"
    assert _verdict(res(10, 0.01, 0.02))["label"] == "미검증"


def test_wait_levels_give_prices_and_dates():
    from minerva.strategy.daily import _month_end, _week_end, wait_levels

    assert _week_end(date(2026, 10, 27)) == date(2026, 10, 30)
    assert _week_end(date(2026, 10, 30)) == date(2026, 11, 6)
    assert _month_end(date(2026, 12, 3)) == date(2026, 12, 31)
    bars = [
        Bar(date(2025, 1, 1) + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i)
        for i in range(400)
    ]
    lv = wait_levels(bars, "long", {"w5": 18.0, "w10": 16.0})
    assert "16.00" in lv["VIX 주봉 5<10"] and "확정" in lv["주봉 20주선 위"]
    assert "아래로" in wait_levels(bars, "short")["50일선 아래"]


def test_base_asset_gets_short_side_and_fear_blocks_it(monkeypatch):
    from minerva.strategy import daily

    monkeypatch.setattr(daily, "short_status", lambda bars: {"state": "진입 신호", "stop": "x"})
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: False)
    assert daily.base_short([], None)["state"] == "진입 신호"
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: True)
    assert daily.base_short([], None)["missing"] == ["백워데이션 중 신규 숏 금지"]


def test_costs_by_market():
    from minerva.strategy.daily import COST_ETF, COST_KR_STOCK, round_trip_cost

    assert round_trip_cost("005930.KS", "KR") == COST_KR_STOCK
    assert round_trip_cost("^KS11", "KR") == COST_ETF
    assert round_trip_cost("SPY", "US") == COST_ETF


def test_with_cost_demotes_thin_edges():
    from minerva.strategy.daily import with_cost

    v = {"label": "검증됨", "trades": 40, "edge": 0.004, "edge_in": 0.006, "edge_out": 0.002}
    out = with_cost(v, 0.003)
    assert out["label"] == "미검증" and out["gross_label"] == "검증됨"
    assert out["net_edge_out"] == -0.001
    assert with_cost(v, 0.001)["label"] == "검증됨"
    weak = {**v, "edge": 0.001, "edge_in": 0.002, "edge_out": 0.001}
    assert with_cost(weak, 0.003)["label"] == "부적합"


def test_apply_fdr_demotes_lucky_passes():
    from minerva.strategy.daily import apply_fdr

    def asset(sym, p, label="검증됨"):
        return {"symbol": sym, "verify": {"long": {"label": label, "trades": 40, "p_value": p}}}

    assets = [asset("A", 0.001), asset("B", 0.3), asset("C", 0.002, "미검증"), {"error": "x"}]
    passed = apply_fdr(assets, q=0.10)
    assert passed == {"A", "C"}
    assert assets[0]["verify"]["long"]["label"] == "검증됨"
    assert (
        assets[1]["verify"]["long"]["label"] == "미검증"
        and "다중 검정" in assets[1]["verify"]["long"]["why"]
    )
    assert assets[2]["verify"]["long"]["label"] == "미검증"  # 비용 등으로 이미 미검증이면 그대로


def test_regime_uses_only_vix_known_before_korean_close(monkeypatch):
    from datetime import date

    from minerva.strategy import daily
    from minerva.strategy.indicators import Bar

    regs = {date(2026, 10, 26): "R0", date(2026, 10, 27): "R3"}
    monkeypatch.setattr(daily, "vix_regimes", lambda vix: regs)
    monkeypatch.setattr(daily, "weekly_gate_at", lambda weekly, day, keys=None: (None, None, False))
    kr = [Bar(date(2026, 10, 27), 1, 1, 1, 1), Bar(date(2026, 10, 28), 1, 1, 1, 1)]
    out = daily._regime_by_day([], kr)
    # 10/27 미국 마감(R3)은 한국 10/27 장이 끝난 뒤에 나온다 (관문 닫힘이라 R0은 R2로)
    assert out == {date(2026, 10, 27): "R2", date(2026, 10, 28): "R3"}


def test_weekly_gate_uses_last_completed_week_only():
    from datetime import date

    from minerva.strategy.regime import gated_regime, weekly_gate_at

    weekly = [(date(2026, 10, 16), 18.0, 16.0), (date(2026, 10, 23), 15.0, 16.0)]
    weekly.append((date(2026, 10, 28), 20.0, 16.0))  # 진행 중인 이번 주는 보지 않는다
    assert weekly_gate_at(weekly, date(2026, 10, 28)) == (15.0, 16.0, True)
    assert weekly_gate_at(weekly, date(2026, 10, 21)) == (18.0, 16.0, False)
    # 관문이 열리면 매수 허용(R0·R1), 닫히면 일봉 레짐과 상관없이 막는다
    assert gated_regime("R2", True) == "R1" and gated_regime("R0", True) == "R0"
    assert gated_regime("R0", False) == "R2" and gated_regime("R3", False) == "R3"


def test_index_signal_is_not_an_order():
    from minerva.strategy.daily import make_actions

    idx = {
        "symbol": "^KS11",
        "name": "코스피",
        "market": "KR",
        "date": "2026-10-29",
        "verify": {"long": {"label": "검증됨"}},
        "tm": {"state": "진입 신호", "ref_price": 6000, "stop": 5400},
    }
    assert make_actions([idx], []) == []


def test_kr_filler_bars_dropped_but_not_us():
    from minerva.strategy.daily import completed

    d = date(2026, 2, 9)
    real = Bar(d - timedelta(days=1), 100, 110, 95, 105, 1000)
    filler = Bar(d, 105, 105, 105, 105, 0)
    now = datetime(2026, 2, 9, 8, 0, tzinfo=timezone.utc)
    assert completed([real, filler], "KR", now) == [real]
    assert (
        completed([real, filler], "US", datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc))[-1]
        == filler
    )


def test_product_needs_own_significance():
    from minerva.strategy.daily import product_verdict

    raw = {"label": "검증됨", "trades": 90, "edge": 0.01, "edge_in": 0.01, "edge_out": 0.01}
    assert product_verdict({**raw, "p_value": 0.01})["label"] == "검증됨"
    assert product_verdict({**raw, "p_value": 0.2})["label"] == "미검증"
    assert product_verdict(raw)["label"] == "미검증"  # p값이 없으면 통과시키지 않는다


def test_stale_kr_asset_signal_skipped():
    from minerva.strategy.daily import make_actions

    def asset(sym, day):
        return {
            "symbol": sym,
            "name": sym,
            "market": "KR",
            "date": day,
            "verify": {"long": {"label": "검증됨"}},
            "tm": {"state": "진입 신호", "ref_price": 100, "stop": 90},
        }

    acts = make_actions([asset("005930.KS", "2026-10-29"), asset("000660.KS", "2026-10-28")], [])
    assert [a["symbol"] for a in acts] == ["005930.KS"]
