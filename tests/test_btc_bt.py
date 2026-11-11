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


def test_hold_without_drawdown_does_not_divide_by_zero():
    bars = _bars([100, 110, 120, 130])
    cagr, mdd, ratio = bt.hold(bars, 0, 4)
    assert mdd == 0.0 and cagr > 0 and ratio == 0.0


def test_side_pnl_adds_up_to_total_after_costs():
    bars = _bars([100, 100, 100, 200, 200, 100, 100])
    lo, sh = [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
    r = bt.simulate(bars, lo, sh, 1, 1, cost=0.01)
    assert r.long_pnl + r.short_pnl == pytest.approx(r.final - 1)


def test_ruin_counts_as_a_losing_trade():
    bars = _bars([100, 100, 100, 250, 250])
    r = bt.simulate(bars, [0] * 5, [0, 1, 0, 0, 0], 1, 1, cost=0)
    assert r.trades == 1 and r.win_rate == 0 and r.short_pnl == pytest.approx(-1.0)


def _ohlc(rows):
    d = date(2024, 1, 1)
    return [Bar(d + timedelta(days=i), o, h, lo, c, 0) for i, (o, h, lo, c) in enumerate(rows)]


def test_long_stop_fills_at_line_and_waits_for_next_signal():
    bars = _ohlc(
        [(100, 100, 100, 100), (100, 100, 100, 100), (100, 101, 98, 99), (99, 200, 99, 200)]
    )
    r = bt.simulate(bars, [0, 1, 0, 0], [0, 0, 0, 0], 1, 1, cost=0, stop=0.01)
    assert r.trades == 1 and r.final == pytest.approx(0.99)  # 99에 정리, 200 상승은 못 먹는다


def test_short_stop_fills_at_line():
    bars = _ohlc([(100, 100, 100, 100), (100, 100, 100, 100), (100, 102, 99, 99), (99, 99, 50, 50)])
    r = bt.simulate(bars, [0, 0, 0, 0], [0, 1, 0, 0], 1, 1, cost=0, stop=0.01)
    assert r.trades == 1 and r.final == pytest.approx(0.99)


def test_stop_gapped_through_fills_at_open():
    bars = _ohlc(
        [
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 101, 100, 100),
            (90, 91, 89, 90),
        ]
    )
    r = bt.simulate(bars, [0, 1, 0, 0], [0, 0, 0, 0], 1, 1, cost=0, stop=0.01)
    assert r.final == pytest.approx(0.90)


def test_stop_allows_entry_on_a_later_signal():
    bars = _ohlc(
        [
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 98, 99),
            (99, 99, 99, 99),
            (99, 120, 99, 120),
            (120, 120, 120, 120),
        ]
    )
    r = bt.simulate(bars, [0, 1, 0, 1, 0, 0], [0] * 6, 1, 1, cost=0, stop=0.01)
    assert r.trades == 2 and r.final == pytest.approx(0.99 * 120 / 99)


def test_filter_blocks_open_but_not_exit():
    bars = _bars([100, 100, 100, 200, 200, 100, 100])
    lo, sh = [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
    blocked = [False] * 7
    r = bt.simulate(bars, lo, sh, 1, 1, cost=0, allow_long=blocked, allow_short=blocked)
    assert r.trades == 0
    # 롱만 허용: 숏 오픈은 막혀도 숏 1단은 롱 종료로 쓰인다
    r = bt.simulate(bars, lo, sh, 1, 1, cost=0, allow_long=[True] * 7, allow_short=blocked)
    assert r.trades == 1 and r.final == pytest.approx(2.0)


def test_align_uses_only_closed_upper_bars():
    # 상위 봉이 100에 마감하면 100에 마감한 매매 봉부터 보이고 99에는 안 보인다
    assert bt.align([50, 99, 100, 150, 200], [100, 200], [1, -1]) == [0, 0, 1, 1, -1]


def test_state_dir_needs_cci_and_rsi_to_agree():
    from minerva.strategy.indicators import TMState

    day = date(2024, 1, 1)
    make = lambda cci, rsi: TMState(day, 1, 50, cci, rsi, 0, 0)  # noqa: E731
    assert bt.state_dir(make(1, 60)) == 1
    assert bt.state_dir(make(-1, 40)) == -1
    assert bt.state_dir(make(1, 40)) == 0
    assert bt.state_dir(make(None, 40)) == 0


def test_without_top_drops_best_trades():
    assert bt.without_top([0.5, -0.1, 0.1], 1, 365) == pytest.approx(0.9 * 1.1 - 1, rel=1e-2)


def test_log_records_each_trade_return():
    bars = _bars([100, 100, 100, 200, 200])
    rets = []
    bt.simulate(bars, [0, 1, 0, 0, 0], [0] * 5, 1, 1, cost=0, log=rets)
    assert rets == pytest.approx([1.0])


def _state(stoch, cci):
    from minerva.strategy.indicators import TMState

    return TMState(date(2024, 1, 1), 1, stoch, cci, 50, 0, 0)


def test_marked_needs_recent_cci_extreme():
    states = [_state(10, -150), _state(15, -90), _state(30, -80), _state(10, -50), _state(30, -50)]
    lo, sh = bt.marked_levels(states)
    assert lo == [0, 0, 1, 0, 0] and sh == [0] * 5  # 마지막 돌파는 CCI 과열이 없다


def test_marked_short_mirrors_long():
    states = [_state(90, 300), _state(85, 150), _state(70, 120), _state(60, 90)]
    states += [_state(90, 50), _state(70, 50)]
    lo, sh = bt.marked_levels(states)
    assert sh == [0, 0, 1, 0, 0, 0] and lo == [0] * 6


def test_marked_ignores_cci_older_than_window():
    states = [_state(10, -150), _state(10, -50), _state(10, -50), _state(30, -50)]
    assert bt.marked_levels(states)[0] == [0, 0, 0, 0]


def _rsi_state(stoch, rsi):
    from minerva.strategy.indicators import TMState

    return TMState(date(2024, 1, 1), 1, stoch, 0, rsi, 0, 0)


def test_wait_rsi_requires_recent_extreme():
    states = [_rsi_state(10, 25), _rsi_state(10, 35), _rsi_state(30, 40)]
    states += [_rsi_state(10, 45), _rsi_state(10, 45), _rsi_state(10, 45), _rsi_state(30, 45)]
    lo, sh = bt.wait_levels(states, [None] * 7, True, False)
    assert lo == [0, 0, 1, 0, 0, 0, 0] and sh == [0] * 7


def test_wait_hour_cci_needs_matching_side():
    states = [_rsi_state(10, 50), _rsi_state(30, 50), _rsi_state(90, 50), _rsi_state(70, 50)]
    lo, sh = bt.wait_levels(states, [0, 25, 0, -5], False, True)
    assert lo == [0, 0, 0, 0] and sh == [0, 0, 0, 0]  # 롱은 1시간 CCI<0, 숏은 >0이어야 한다
    lo, sh = bt.wait_levels(states, [0, -25, 0, 5], False, True)
    assert lo == [0, 1, 0, 0] and sh == [0, 0, 0, 1]


def test_wait_both_conditions_must_hold():
    states = [_rsi_state(10, 25), _rsi_state(30, 40)]
    assert bt.wait_levels(states, [0, 5], True, True)[0] == [0, 0]
    assert bt.wait_levels(states, [0, -5], True, True)[0] == [0, 1]
    assert bt.wait_levels(states, [None, None], True, True)[0] == [0, 0]
