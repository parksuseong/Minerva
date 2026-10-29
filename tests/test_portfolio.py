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
