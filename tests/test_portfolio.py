from datetime import date, timedelta

import pytest

from minerva.strategy import portfolio as pf
from minerva.strategy.indicators import Bar


def _inst(code, prices, entry_days, take_days=(), kind="long", cap=0.2, p=0.01):
    bars = {d: Bar(d, px, px * 1.01, px * 0.99, px, 0) for d, px in prices}
    return pf.Instrument(
        code,
        code,
        kind,
        cap,
        bars,
        {d: True for d in entry_days},
        {d: True for d in take_days},
        p,
        0.01,
    )


def test_buys_next_open_and_exits_on_take():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    prices = [(d, 100 + i) for i, d in enumerate(days)]
    u = _inst("A", prices, [days[0]], [days[4]])
    r = pf.simulate([u], days, {d: "R1" for d in days}, pf.Config(capital=10_000))
    assert r.trades == 2 and r.final > 10_000  # 101에 사서 105에 판다 (비용 차감)
    assert r.fills == [(days[1], "A", "buy"), (days[5], "A", "sell")]


def test_stop_loss_and_cash_limit():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    prices = [
        (days[0], 100),
        (days[1], 100),
        (days[2], 85),
        (days[3], 85),
        (days[4], 85),
        (days[5], 85),
    ]
    u = _inst("A", prices, [days[0]])
    r = pf.simulate([u], days, {d: "R1" for d in days}, pf.Config(capital=10_000))
    # 20% 배정 → 19주, −10% 손절 → 약 −2% 계좌 손실
    assert r.trades == 2 and 9_700 < r.final < 9_850


def _gap(open_, low):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    u = _inst("A", [(d, 100) for d in days], [days[0]])
    u.prices[days[1]] = Bar(days[1], open_, open_ + 1, low, open_, 0)
    return u, days


def _run_limit(open_, low):
    u, days = _gap(open_, low)
    cfg = pf.Config(capital=10_000, limit=1.03, hold=100)
    return pf.simulate([u], days, {d: "R1" for d in days}, cfg)


def test_limit_skips_when_price_stays_above_limit():
    assert _run_limit(110, 105).trades == 0


def test_limit_fills_at_limit_when_gap_comes_back():
    r = _run_limit(110, 101)
    assert r.trades == 1
    assert r.equity[-1][1] < 10_000  # 103에 사서 100으로 평가


def test_limit_fills_at_open_when_below_limit():
    r = _run_limit(101, 100)
    r2 = pf.simulate(
        *(lambda u, d: ([u], d, {x: "R1" for x in d}))(*_gap(101, 100)),
        pf.Config(capital=10_000, hold=100),
    )
    assert r.final == r2.final


def test_park_earns_on_idle_cash_only():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    u = _inst("A", [(d, 100) for d in days], [])
    park = {days[1]: 0.01}
    r = pf.simulate([u], days, {d: "R1" for d in days}, pf.Config(capital=10_000, park=park))
    assert r.final == pytest.approx(10_000 * (1 + 0.01 * 0.98))
    assert r.exposure == 0  # 파킹은 투자 비율에 넣지 않는다


def test_parse_naver_closes():
    xml = '<item data="20200707|49947|49950|49942|49948|11266" />'
    assert pf.parse_naver_closes(xml) == [(date(2020, 7, 7), 49948.0)]


def test_blend_rebalances_each_year():
    d = [date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2), date(2025, 1, 3)]
    up = [(d[0], 1.0), (d[1], 2.0), (d[2], 2.0), (d[3], 4.0)]
    flat = [(x, 1.0) for x in d]
    out = pf.blend([up, flat], [0.5, 0.5])
    # 연말 1.5 → 반반으로 되돌린 뒤 위쪽 슬리브만 2배: 0.75 × 2 + 0.75
    assert out[-1][1] == pytest.approx(2.25)


def test_trend_curve_uses_only_month_end_after_signal():
    days = [date(2024, m, 28) for m in range(1, 13)]
    days += [date(2024, m, 29) for m in range(1, 13)]
    days = sorted(days)
    bars = [Bar(d, 100 + i, 101 + i, 99 + i, 100 + i, 0) for i, d in enumerate(days)]
    start = days[6:]
    curve = pf.trend_curve(bars, start, {d: 0.0 for d in days}, window=3)
    assert curve[0][1] == 1.0  # 신호가 나오기 전엔 파킹
    assert curve[-1][1] > 1.0  # 상승 중이면 들어간다


def test_filter_mode_holds_while_filter_is_on(monkeypatch):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    bars = [Bar(d, 100, 101, 99, 100, 0) for d in days]
    monkeypatch.setattr(pf, "filter_mask", lambda src, rule, ctx: [True, True, False, True])
    monkeypatch.setattr(pf.dm, "_ctx", lambda vix, src: None)
    entry, take = pf._signals_for(bars, "long", [], mode="filter")
    assert [entry[d] for d in days] == [True, True, False, True]
    assert [take[d] for d in days] == [False, False, True, False]


def test_walkforward_rules_refuse_to_fill_cache(tmp_path):
    from minerva.strategy import walkforward as wf

    with pytest.raises(RuntimeError):
        wf.run_verifications(["A"], [date(2024, 1, 2)], [], tmp_path / "c.json", compute=False)
