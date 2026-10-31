from datetime import date, timedelta

import pytest

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import Bar


def _bars(prices):
    d = date(2024, 1, 1)
    return [Bar(d + timedelta(days=i), p, p, p, p, 0) for i, p in enumerate(prices)]


def test_edges_mark_first_bar_only():
    assert bt.edges([0, 1, 1, 0, 2, 1], 1) == [False, True, False, False, True, False]
    assert bt.edges([0, 1, 2, 1, 2], 2) == [False, False, True, False, True]


def test_signal_fills_on_next_open_not_same_bar():
    bars = _bars([100, 100, 200, 200, 200])
    lo = [0, 1, 0, 0, 0]
    r = bt.simulate(bars, lo, [0] * 5, 1, 1, cost=0)
    # 1번 봉 신호 → 2번 봉 시가 200 진입 → 수익 없음. 같은 봉 체결이면 두 배가 된다
    assert r.final == pytest.approx(1.0)
    assert r.trades == 1


def test_long_then_short_reverses_and_short_earns_on_fall():
    bars = _bars([100, 100, 100, 200, 200, 100, 100])
    lo = [0, 1, 0, 0, 0, 0, 0]
    sh = [0, 0, 0, 1, 0, 0, 0]
    r = bt.simulate(bars, lo, sh, 1, 1, cost=0)
    assert r.trades == 2
    assert r.long_pnl == pytest.approx(1.0)  # 100 → 200
    assert r.final == pytest.approx(3.0)  # 2.0 숏 200 → 100
    assert r.short_pnl == pytest.approx(1.0)


def test_cost_is_charged_on_both_sides():
    bars = _bars([100, 100, 100, 100])
    r = bt.simulate(bars, [0, 1, 0, 0], [0, 0, 0, 0], 1, 1, cost=0.01)
    assert r.final == pytest.approx(0.99 * 0.99)


def test_close_without_reversal_goes_flat():
    bars = _bars([100, 100, 100, 150, 150, 300, 300])
    lo = [0, 2, 0, 0, 0, 0, 0]
    sh = [0, 0, 0, 1, 0, 0, 0]  # 1단 이탈은 k=2 숏 오픈이 아니라 롱 종료만
    r = bt.simulate(bars, lo, sh, 2, 1, cost=0)
    assert r.trades == 1 and r.final == pytest.approx(1.5)
    assert r.short_pnl == 0


def test_both_opens_on_same_bar_do_not_open():
    bars = _bars([100, 100, 100])
    r = bt.simulate(bars, [0, 1, 0], [0, 1, 0], 1, 1, cost=0)
    assert r.trades == 0


def test_short_is_wiped_out_when_price_doubles():
    bars = _bars([100, 100, 100, 250, 250])
    r = bt.simulate(bars, [0] * 5, [0, 1, 0, 0, 0], 1, 1, cost=0)
    assert r.ruined and r.final == 0.0 and r.mdd == -1.0


def test_range_starts_flat_and_ignores_older_signal():
    bars = _bars([100, 100, 100, 200, 200])
    r = bt.simulate(bars, [0, 1, 0, 0, 0], [0] * 5, 1, 1, cost=0, start=3)
    assert r.trades == 0 and r.final == 1.0


def test_hold_reports_drawdown_from_high():
    bars = _bars([100, 200, 100, 150])
    cagr, mdd, _ = bt.hold(bars, 0, 4)
    assert mdd == pytest.approx(-0.5) and cagr > 0


def test_side_pnl_adds_up_to_total_after_costs():
    bars = _bars([100, 100, 100, 200, 200, 100, 100])
    lo, sh = [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
    r = bt.simulate(bars, lo, sh, 1, 1, cost=0.01)
    assert r.long_pnl + r.short_pnl == pytest.approx(r.final - 1)


def test_ruin_counts_as_a_losing_trade():
    bars = _bars([100, 100, 100, 250, 250])
    r = bt.simulate(bars, [0] * 5, [0, 1, 0, 0, 0], 1, 1, cost=0)
    assert r.trades == 1 and r.win_rate == 0 and r.short_pnl == pytest.approx(-1.0)
