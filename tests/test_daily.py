from datetime import date, datetime, timedelta, timezone

from minerva.strategy import daily
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

    asset = {
        "symbol": "SPY",
        "name": "SPY",
        "market": "US",
        "date": "2026-10-27",
        "verify": {"long": {"label": "검증됨"}},
        "tm": {"state": "진입 신호", "ref_price": 100, "stop": 90},
    }
    acts = make_actions([asset], [])
    assert [a["action"] for a in acts] == ["LONG_ENTRY"] and acts[0]["stop"] == 90
    text = telegram_summary(_data(actions=acts))
    assert "🎯 오늘 할 일 1건" in text and "롱 진입 SPY [✅]" in text


def test_product_uses_own_price_for_orders(monkeypatch):
    from minerva.strategy import daily

    bars = bars_until(date(2026, 10, 20), 5)
    monkeypatch.setattr(daily, "fetch_yahoo", lambda s, r: bars)
    entry = {"state": "진입 신호", "ref_price": 5638.0, "stop": 5000}
    monkeypatch.setattr(daily, "tm_status", lambda b, c: dict(entry))
    raw = {
        "label": "검증됨",
        "trades": 90,
        "edge": 0.01,
        "edge_in": 0.01,
        "edge_out": 0.01,
        "p_value": 0.01,
    }
    monkeypatch.setattr(daily, "verify", lambda s, b, c: {"long": raw})
    etf = daily.product_status(daily.Asset("069500.KS", "E", "KR", "etf", "^KS11", 1), bars, {})
    close = bars[-1].close
    assert etf["tm"]["ref_price"] == round(close, 2) and etf["tm"]["stop"] == round(close * 0.9, 2)
    assert [a["action"] for a in daily.make_actions([], [etf])] == ["LONG_ENTRY"]


def test_no_inverse_products():
    from minerva.strategy import daily

    assert [p.kind for p in daily.PRODUCTS] == ["etf"]


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


def test_verify_ttl_is_spread_over_days():
    from minerva.strategy.daily import VERIFY_TTL, verify_ttl

    syms = [f"{n:06d}.KS" for n in range(100, 150)]
    assert {verify_ttl(s) // 86400 for s in syms} == {4, 5, 6, 7}
    assert verify_ttl("005930.KS") == verify_ttl("005930.KS") <= VERIFY_TTL


def test_make_actions_limits_new_buys_to_top_but_keeps_exits():
    from minerva.strategy import daily

    def asset(i, state):
        tm = {"state": state, "signal": "", "stop": 0, "take": "", "days_held": 0}
        return {
            "symbol": f"{i:06d}.KS",
            "name": str(i),
            "market": "KR",
            "date": "2026-10-30",
            "close": 100,
            "adv": 1,
            "verify": {"long": {"label": "검증됨"}},
            "tm": tm,
        }

    assets = [asset(i, "대기") for i in range(daily.KOSPI_TOP + 2)]
    assets[daily.KOSPI_TOP - 1] = asset(daily.KOSPI_TOP - 1, "진입 신호")
    assets[daily.KOSPI_TOP] = asset(daily.KOSPI_TOP, "진입 신호")
    assets[daily.KOSPI_TOP + 1] = asset(daily.KOSPI_TOP + 1, "청산 신호")
    got = {(a["symbol"], a["action"]) for a in daily.make_actions(assets, [])}
    assert got == {
        (f"{daily.KOSPI_TOP - 1:06d}.KS", "LONG_ENTRY"),
        (f"{daily.KOSPI_TOP + 1:06d}.KS", "LONG_EXIT"),
    }


def _monthly(rows):
    from minerva.strategy.indicators import TMState

    return [TMState(d, 100, 50, c, r, 0, s) for d, c, r, s in rows]


def _month_bars(rows):
    from minerva.strategy.indicators import Bar

    return [Bar(r[0], 1, 1, 1, 1, 1) for r in rows]


def test_monthly_block_starts_on_exit_and_clears_after_dip_and_recovery(monkeypatch):
    from datetime import date

    from minerva.strategy import daily

    rows = [
        (date(2026, 1, 30), 150, 75, 0),
        (date(2026, 2, 27), 90, 65, 3),
        (date(2026, 3, 31), 40, 55, 0),
        (date(2026, 4, 30), -20, 45, 0),
        (date(2026, 5, 29), 30, 55, 0),
        (date(2026, 6, 30), 30, 55, 0),
    ]
    monkeypatch.setattr(daily, "resample", lambda bars, frame: bars)
    monkeypatch.setattr(daily, "MONTHLY_MIN", 1)

    def block(upto, today):
        monkeypatch.setattr(daily, "tm_states", lambda bars: _monthly(rows[:upto]))
        return daily.monthly_block(_month_bars(rows[:upto]), today)

    assert block(1, date(2026, 2, 3)) == {"blocked": False, "since": None}
    got = block(3, date(2026, 4, 1))
    assert (got["blocked"], got["since"], got["dipped"]) == (True, "2026-02-27", False)
    assert block(4, date(2026, 5, 1))["dipped"]
    assert not block(5, date(2026, 6, 1))["blocked"]


def test_monthly_block_ignores_month_in_progress(monkeypatch):
    from datetime import date

    from minerva.strategy import daily

    rows = [(date(2026, 1, 30), 150, 75, 0), (date(2026, 2, 13), 90, 65, 3)]
    monkeypatch.setattr(daily, "resample", lambda bars, frame: bars)
    monkeypatch.setattr(daily, "tm_states", lambda bars: _monthly(rows))
    monkeypatch.setattr(daily, "MONTHLY_MIN", 1)
    assert not daily.monthly_block(_month_bars(rows), date(2026, 2, 13))["blocked"]
    rows[1] = (date(2026, 2, 27), 90, 65, 3)
    assert daily.monthly_block(_month_bars(rows), date(2026, 3, 2))["blocked"]


def test_monthly_block_fails_closed_without_usable_data(monkeypatch):
    from datetime import date

    from minerva.strategy import daily

    assert daily.monthly_block([], date(2026, 11, 2)) == daily.closed_block("자료 없음")
    rows = [(date(2026, 1, 30), 150, 75, 0)]
    monkeypatch.setattr(daily, "resample", lambda bars, frame: bars)
    monkeypatch.setattr(daily, "tm_states", lambda bars: _monthly(rows))
    assert "자료 부족" in daily.monthly_block(_month_bars(rows), date(2026, 2, 2))["error"]


def test_monthly_block_treats_cut_off_month_as_stale():
    from datetime import date

    from minerva.strategy import daily

    rows = [(date(2026, 8, 14), 0, 0, 0)]
    got = daily.monthly_block(_month_bars(rows), date(2026, 9, 1))
    assert got["blocked"] and got["since"] is None and "끊김" in got["error"]
    rows = [(date(2026, 9, 30), 0, 0, 0)]
    got = daily.monthly_block(_month_bars(rows), date(2026, 12, 15))
    assert got["blocked"] and "끊김" in got["error"]


def test_block_spans_keeps_open_span():
    from datetime import date

    from minerva.strategy import daily

    rows = [
        (date(2026, 1, 30), 150, 75, 0),
        (date(2026, 2, 27), 90, 65, 3),
        (date(2026, 3, 31), -1, 45, 0),
        (date(2026, 4, 30), 30, 55, 0),
        (date(2026, 5, 29), 90, 65, 3),
    ]
    assert daily.block_spans(_monthly(rows[:1] + [(rows[0][0], 90, 65, 0)] + rows[1:])) == [
        (date(2026, 2, 27), date(2026, 4, 30)),
        (date(2026, 5, 29), None),
    ]


def test_block_spans_release_on_twenty_month_touch():
    from datetime import date

    from minerva.strategy import daily

    rows = [
        (date(2026, 1, 30), 150, 75, 0),
        (date(2026, 2, 27), 90, 65, 3),
        (date(2026, 3, 31), 40, 55, 0),
    ]
    assert daily.block_spans(_monthly(rows), {date(2026, 3, 31)}) == [
        (date(2026, 2, 27), date(2026, 3, 31))
    ]
    assert daily.block_spans(_monthly(rows)) == [(date(2026, 2, 27), None)]


def test_block_spans_keeps_block_while_new_exit_is_active():
    from datetime import date

    from minerva.strategy import daily

    rows = [
        (date(2026, 1, 30), 150, 75, 0),
        (date(2026, 2, 27), 90, 65, 3),
        (date(2026, 3, 31), 90, 65, 3),
        (date(2026, 4, 30), 40, 55, 0),
    ]
    touches = {date(2026, 3, 31), date(2026, 4, 30)}
    assert daily.block_spans(_monthly(rows), touches) == [(date(2026, 2, 27), date(2026, 4, 30))]


def test_monthly_touches_uses_low_against_twenty_month_average():
    from datetime import date, timedelta

    from minerva.strategy import daily
    from minerva.strategy.indicators import Bar

    days = [date(2024, 1, 15) + timedelta(days=31 * i) for i in range(22)]
    bars = [Bar(d, 100, 110, 105, 100, 1) for d in days]
    bars[-1] = Bar(days[-1], 100, 100, 90, 95, 1)
    bars[-2] = Bar(days[-2], 100, 110, 101, 100, 1)
    assert days[-1] in daily.monthly_touches(bars)
    assert days[-2] not in daily.monthly_touches(bars)


def test_build_keeps_exit_flow_when_index_fetch_fails(monkeypatch):
    from datetime import date

    from minerva.strategy import daily

    bars = _month_bars([(date(2026, 10, 30), 0, 0, 0)])

    def fetch(sym, rng="max"):
        if sym == "^KS11":
            raise OSError("down")
        return bars

    monkeypatch.setattr(daily, "fetch_yahoo", fetch)
    monkeypatch.setattr(daily, "completed", lambda b, m: b)
    monkeypatch.setattr(daily, "vix_regimes", lambda v: {bars[-1].day: "R1"})
    monkeypatch.setattr(daily, "vix_weekly", lambda v: [])
    monkeypatch.setattr(daily, "weekly_gate_at", lambda w, d: (1, 2, True))
    monkeypatch.setattr(daily, "gated_regime", lambda r, c: r)
    monkeypatch.setattr(daily, "watchlist", lambda: [])
    monkeypatch.setattr(daily, "PRODUCTS", [])
    monkeypatch.setattr(daily, "fear_status", lambda v, m: {})
    used = []
    monkeypatch.setattr(daily, "apply_fdr_by_family", lambda a: used.append(a))
    assert daily.build()["market_block"] == daily.closed_block("OSError")
    assert len(used) == 1  # 실전 분석이 묶음별 검정을 쓰는지


def test_make_actions_blocked_keeps_exits_only():
    from minerva.strategy import daily

    def item(sym, state):
        return {
            "symbol": sym,
            "name": sym,
            "market": "KR",
            "date": "2026-10-30",
            "close": 100,
            "adv": 1,
            "verify": {"long": {"label": "검증됨"}},
            "tm": {"state": state, "signal": "", "stop": 0, "take": "", "days_held": 0},
        }

    assets = [item("000001.KS", "진입 신호"), item("000002.KS", "청산 신호")]
    products = [item("069500.KS", "진입 신호")]
    got = daily.make_actions(assets, products, True)
    assert [(a["symbol"], a["action"]) for a in got] == [("000002.KS", "LONG_EXIT")]
    assert len(daily.make_actions(assets, products)) == 3


def _memo_data(regime="R0", **block):
    gate = {"w5": 16.89, "w10": 15.92}
    return {"vix": {"regime": regime, "gate": gate}, "market_block": block}


def test_trade_memo_empty_when_nothing_blocks_buys():
    assert daily.trade_memo(_memo_data(blocked=False, since=None), date(2026, 11, 2)) == []


def test_trade_memo_weekend_after_month_end_points_to_next_close():
    block = dict(blocked=True, since="2026-08-31", dipped=False, ma20=4948.2)
    memo = daily.trade_memo(_memo_data("R2", **block), date(2026, 10, 31))
    assert "11월 저가가 20개월선(4,948)에 닿으면 12월 초부터" in memo[0]
    assert "빨라도 1월 초" in memo[0]
    assert "11월 6일(금)" in memo[1]
    assert memo[2:] == ["두 조건이 모두 풀려야 매수해요.", "청산은 막지 않아요."]


def test_trade_memo_last_business_day_still_waits_for_this_close():
    block = dict(blocked=True, since="2026-08-31", dipped=True, ma20=4948.2)
    memo = daily.trade_memo(_memo_data(**block), date(2026, 11, 30))
    assert "11월 마감에 월봉 CCI 0·RSI 50 위로 회복" in memo[0]
    assert "12월 초부터" in memo[0] and len(memo) == 2


def test_trade_memo_reports_data_failure():
    memo = daily.trade_memo(_memo_data(**daily.closed_block("OSError")), date(2026, 11, 2))
    assert memo[0] == "코스피 자료 문제(OSError)로 신규 매수를 막았어요."


def _plan_data(regime="R0", blocked=False):
    def asset(symbol, name, label):
        return {"symbol": symbol, "name": name, "verify": {"long": {"label": label}}}

    stocks = [asset(f"{i:06d}.KS", f"종목{i}", "미검증") for i in range(120)]
    stocks[3] = asset("000003.KS", "기아", "검증됨")
    stocks[105] = asset("000105.KS", "밖", "검증됨")
    return {
        "vix": {"regime": regime},
        "market_block": {"blocked": blocked},
        "assets": [asset("^KS11", "코스피", "검증됨"), *stocks],
        "products": [asset("069500.KS", "KODEX 200", "검증됨")],
    }


def test_buy_candidates_are_verified_and_inside_top_100():
    assert daily.buy_candidates(_plan_data()) == ["KODEX 200", "기아"]


def test_trade_plan_while_blocked_says_no_buys_and_uses_live_numbers():
    plan = daily.trade_plan(_plan_data("R2", blocked=True), {})
    assert plan[0] == "매수 제한이 풀릴 때까지 매수 주문은 없어요."
    assert plan[1] == "지금 봇 보유 종목이 없어서 청산 주문도 없어요."
    assert "KODEX 200, 기아 2종목" in plan[2]
    assert "+3% 지정가" in plan[3]
    assert "개별주 30%, KODEX 200 60%까지, 최대 10종목" in plan[4]
    assert "−10%" in plan[4]


def test_trade_plan_when_open_names_holdings():
    plan = daily.trade_plan(_plan_data(), {"000270": {"name": "기아"}})
    assert plan[0] == "보유 중인 기아은(는) 청산 신호나 손절(−10%)이 나오면 팔아요."


def test_add_memo_failure_does_not_stop_analysis(monkeypatch):
    monkeypatch.setattr(daily, "read_owned", lambda: 1 / 0)
    data = _plan_data()
    daily.add_memo(data, date(2026, 11, 2))
    assert data["memo"] == ["요약을 만들지 못했어요 (ZeroDivisionError)."] and data["plan"] == []


def test_fdr_by_family_tests_top_100_apart_from_the_rest():
    def asset(sym, p):
        return {"symbol": sym, "verify": {"long": {"label": "검증됨", "trades": 40, "p_value": p}}}

    stocks = [asset(f"{i:03d}.KS", 0.9) for i in range(150)]
    stocks[50] = asset("050.KS", 0.0012)
    us = asset("AAPL", 0.002)
    assets = [*stocks, us]
    # 151개를 한 번에 걸면 둘 다 떨어진다 (1위 기준 0.15/151 ≈ 0.00099, 2위 ≈ 0.00199)
    daily.apply_fdr_by_family(assets, q=0.15)
    label = lambda a: a["verify"]["long"]["label"]  # noqa: E731
    assert label(stocks[50]) == "검증됨"
    assert label(us) == "검증됨"  # 나머지 묶음은 참고용이고 따로 건다
    assert label(stocks[120]) == "미검증"  # 나머지 묶음도 검정을 거친다
    assert label(stocks[0]) == "미검증"
    assert [a["symbol"] for a in daily.trade_family(assets)][:2] == ["000.KS", "001.KS"]
    assert len(daily.trade_family(assets)) == 100


def test_trade_family_keeps_error_rows_in_rank_count():
    rows = [{"symbol": f"{i:03d}.KS", "error": "x"} for i in range(100)]
    rows.append({"symbol": "100.KS"})
    assert "100.KS" not in {a["symbol"] for a in daily.trade_family(rows)}


def test_watchlist_keeps_held_stock_outside_top_for_exits(monkeypatch):
    top = [("005930.KS", "삼성전자")]
    monkeypatch.setattr(daily, "kospi_top", lambda n: top)
    monkeypatch.setattr(daily, "nasdaq_top", lambda n: [("AAPL", "애플")])
    owned = {
        "005930": {"name": "삼성전자"},
        "069500": {"name": "KODEX 200"},
        "000720": {"name": "현대건설"},
    }
    monkeypatch.setattr(daily, "read_owned", lambda: owned)
    symbols = [a.symbol for a in daily.watchlist()]
    assert symbols.count("005930.KS") == 1 and "069500.KS" not in symbols
    assert symbols.index("000720.KS") == symbols.index("005930.KS") + 1  # 상위 뒤에 온다
    assert [a.symbol for a in daily.watchlist() if a.held_only] == ["000720.KS"]


def test_held_stock_outside_top_exits_but_cannot_be_bought():
    def item(i, state):
        return {
            "symbol": f"{i:06d}.KS",
            "name": str(i),
            "market": "KR",
            "date": "2026-10-30",
            "close": 100,
            "adv": 1,
            "verify": {"long": {"label": "검증됨"}},
            "tm": {"state": state, "signal": "", "stop": 0, "take": "", "days_held": 0},
        }

    rows = [item(i, "대기") for i in range(daily.KOSPI_TOP)]
    held = item(daily.KOSPI_TOP, "청산 신호")
    assert [a["action"] for a in daily.make_actions([*rows, held], [])] == ["LONG_EXIT"]
    held = item(daily.KOSPI_TOP, "진입 신호")
    assert daily.make_actions([*rows, held], []) == []
    # 순위를 10개만 받아도 순위 밖 보유 종목은 위치와 무관하게 살 수 없다
    held["held_only"] = True
    assert daily.make_actions([*rows[:10], held], []) == []
    assert held not in daily.trade_family([*rows[:10], held])
    data = {"assets": [*rows[:10], held], "products": []}
    assert held["name"] not in daily.buy_candidates(data)


def test_watchlist_survives_broken_positions_file(monkeypatch):
    def broken():
        raise ValueError("bad json")

    monkeypatch.setattr(daily, "kospi_top", lambda n: [("005930.KS", "삼성전자")])
    monkeypatch.setattr(daily, "nasdaq_top", lambda n: [])
    monkeypatch.setattr(daily, "read_owned", broken)
    sent = []
    monkeypatch.setattr("minerva.notify.send", lambda text: sent.append(text))
    assert "005930.KS" in [a.symbol for a in daily.watchlist()]
    assert "ValueError" in sent[0]


def test_watchlist_survives_positions_file_of_wrong_type(monkeypatch):
    monkeypatch.setattr(daily, "kospi_top", lambda n: [("005930.KS", "삼성전자")])
    monkeypatch.setattr(daily, "nasdaq_top", lambda n: [])
    monkeypatch.setattr(daily, "read_owned", lambda: None)
    monkeypatch.setattr("minerva.notify.send", lambda text: True)
    assert "005930.KS" in [a.symbol for a in daily.watchlist()]
