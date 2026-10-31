from datetime import date, timedelta

import pytest

from minerva.strategy import btc_v2 as v2
from minerva.strategy.indicators import Bar


def _bars(prices, year=2024):
    d = date(year, 1, 1)
    return [Bar(d + timedelta(days=i), p, p, p, p, 0) for i, p in enumerate(prices)]


LONG = v2.Combo(1, 1, None, "both")


def test_combo_count_is_thirty():
    assert len(v2.COMBOS) == 30 and len(set(v2.COMBOS)) == 30


def test_regime_change_closes_position_on_next_open():
    bars = _bars([100, 100, 100, 200, 300, 300])
    lo = [0, 1, 0, 0, 0, 0]
    reg = [v2.UP, v2.UP, v2.UP, v2.DOWN, v2.DOWN, v2.DOWN]
    r = v2.run(bars, lo, [0] * 6, reg, lambda y: {v2.UP: LONG}, cost=0)
    assert r.final == pytest.approx(3.0)  # 3번 마감에 국면 변화 → 4번 시가 300 청산
    assert r.trades[0][1] == v2.UP


def test_no_plan_means_flat():
    bars = _bars([100, 100, 100, 200])
    r = v2.run(bars, [0, 1, 0, 0], [0] * 4, [v2.SIDE] * 4, lambda y: {v2.UP: LONG}, cost=0)
    assert r.trades == [] and r.final == 1.0


def test_side_limit_blocks_open_but_opposite_signal_still_closes():
    bars = _bars([100, 100, 100, 200, 200, 100, 100])
    lo, sh = [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
    only = v2.Combo(1, 1, None, "long")
    r = v2.run(bars, lo, sh, [v2.UP] * 7, lambda y: {v2.UP: only}, cost=0)
    assert len(r.trades) == 1 and r.final == pytest.approx(2.0)


def test_short_only_ignores_long_signal():
    bars = _bars([100, 100, 100, 200, 200, 200])
    only = v2.Combo(1, 1, None, "short")
    r = v2.run(bars, [0, 1, 0, 0, 0, 0], [0] * 6, [v2.UP] * 6, lambda y: {v2.UP: only}, cost=0)
    assert r.trades == []


def test_stop_uses_combo_of_open_position():
    bars = [
        Bar(date(2024, 1, 1) + timedelta(days=i), o, h, low, c, 0)
        for i, (o, h, low, c) in enumerate(
            [(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 98, 99), (99, 200, 99, 200)]
        )
    ]
    combo = v2.Combo(1, 1, 0.01, "both")
    r = v2.run(bars, [0, 1, 0, 0], [0] * 4, [v2.UP] * 4, lambda y: {v2.UP: combo}, cost=0)
    assert r.final == pytest.approx(0.99)


def test_score_uses_only_trades_before_the_year_and_regime():
    trades = [(2019, v2.UP, 0.1)] * 30 + [(2020, v2.UP, 5.0)] * 30 + [(2019, v2.DOWN, 0.1)] * 30
    assert v2.score(trades, 2020, v2.UP) == pytest.approx(30 * 0.0953, rel=1e-2)
    assert v2.score(trades[:29], 2020, v2.UP) is None


def test_pick_goes_flat_without_positive_history():
    bad = v2.Combo(1, 1, None, "both")
    good = v2.Combo(2, 2, None, "both")
    table = {bad: [(2019, v2.UP, -0.1)] * 40, good: [(2019, v2.UP, 0.1)] * 40}
    assert v2.pick(table, 2020, v2.UP) == good
    assert v2.pick({bad: table[bad]}, 2020, v2.UP) is None
    assert v2.pick(table, 2019, v2.UP) is None


def test_pick_cannot_see_the_test_year():
    late = v2.Combo(3, 3, None, "both")
    early = v2.Combo(2, 2, None, "both")
    table = {
        early: [(2019, v2.UP, 0.01)] * 30,
        late: [(2019, v2.UP, -0.01)] * 30 + [(2020, v2.UP, 9.0)] * 30,
    }
    assert v2.pick(table, 2020, v2.UP) == early


def test_without_top_removes_largest():
    trades = [(2020, 1, 0.5), (2020, 1, -0.1), (2020, 1, 0.1)]
    assert v2.without_top(trades, 1, 365) == pytest.approx(0.9 * 1.1 - 1, rel=1e-2)
