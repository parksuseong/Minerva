from datetime import date

from minerva.strategy import scr
from minerva.strategy.indicators import TMState


def _state(day, stoch=50.0, cci_v=0.0, rsi_v=50.0):
    return TMState(day, 100.0, stoch, cci_v, rsi_v, 0, 0)


def test_confirmed_uses_only_closed_bars():
    weeks = [_state(date(2024, 1, 5), cci_v=1), _state(date(2024, 1, 12), cci_v=2)]
    assert scr.confirmed(weeks, date(2024, 1, 11)).cci == 1
    assert scr.confirmed(weeks, date(2024, 1, 12)).cci == 2
    assert scr.confirmed(weeks, date(2024, 1, 4)) is None


def test_up_down_use_baselines():
    assert scr.is_up(_state(date(2024, 1, 5), cci_v=10, rsi_v=55))
    assert scr.is_down(_state(date(2024, 1, 5), cci_v=-10, rsi_v=45))
    assert not scr.is_down(_state(date(2024, 1, 5), cci_v=-10, rsi_v=55))


BASE = dict(
    level=1,
    stoch=25.0,
    close=110.0,
    ma5=108.0,
    ma10=107.0,
    ma20=106.0,
    ma50=100.0,
    cci_v=10.0,
    rsi_v=55.0,
    stoch_cross=True,
    htf_down=False,
    weekly_oversold=False,
    pdi=25.0,
    mdi=15.0,
    adx=25.0,
)


def test_stage1_above_50ma():
    assert scr.entry_signal(**BASE)


def test_stage1_blocked_when_higher_timeframe_down():
    assert not scr.entry_signal(**{**BASE, "htf_down": True})


def test_stage1_needs_cci_rsi_baselines():
    assert not scr.entry_signal(**{**BASE, "cci_v": -5.0})


def test_weekly_oversold_blocks_everything():
    assert not scr.entry_signal(**{**BASE, "level": 3, "stoch": 30.0, "weekly_oversold": True})


def test_strong_downtrend_blocks():
    assert not scr.entry_signal(**{**BASE, "adx": 25.0, "pdi": 10.0, "mdi": 30.0})


def test_stage3_needs_stoch_below_50():
    args = {**BASE, "level": 3, "stoch_cross": False, "stoch": 55.0}
    assert not scr.entry_signal(**args)
    assert scr.entry_signal(**{**args, "stoch": 40.0})


def test_stage2_in_short_term_inverted_trend():
    args = {**BASE, "level": 2, "stoch_cross": False, "close": 90.0}
    assert not scr.entry_signal(**args)
    assert scr.entry_signal(**{**args, "ma5": 91.0, "ma10": 92.0, "ma20": 93.0})


def _plan(days, closes, **flags):
    from minerva.strategy.indicators import Bar

    empty = {d: False for d in days}
    fields = dict(
        entry=empty, add2=empty, add3=empty, strong=empty, htf_exit=empty, take=empty, below20=empty
    )
    fields.update(flags)
    bars = {d: Bar(d, c, c, c, c, 0) for d, c in zip(days, closes)}
    return scr.Plan("A", "A", 0.3, bars, **fields)


def _days(n):
    from datetime import timedelta

    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def _run(plan, days):
    return scr.simulate([plan], days, {d: "R1" for d in days}, 10_000, {})


def test_non_strong_position_ignores_weekly_exit():
    days = _days(6)
    p = _plan(days, [100] * 6, entry={days[0]: True}, htf_exit={d: True for d in days})
    assert _run(p, days)[0].trades == 1  # 사기만 했다


def test_strong_position_ignores_stoch_exit_but_sells_on_weekly_exit():
    days = _days(6)
    strong = {d: True for d in days}
    p = _plan(days, [100] * 6, entry={days[0]: True}, strong=strong, take=strong)
    assert _run(p, days)[0].trades == 1
    p = _plan(days, [100] * 6, entry={days[0]: True}, strong=strong, htf_exit={days[3]: True})
    assert _run(p, days)[0].trades == 2


def test_average_stop_closes_whole_position():
    days = _days(5)
    p = _plan(days, [100, 100, 80, 80, 80], entry={days[0]: True})
    res, closed = _run(p, days)
    assert res.trades == 2 and closed[0] < -0.10


def test_log_records_reason_and_won_pnl():
    days = _days(5)
    p = _plan(days, [100, 100, 80, 80, 80], entry={days[0]: True})
    log: list[dict] = []
    scr.simulate([p], days, {d: "R1" for d in days}, 10_000, {}, log)
    assert [t["reason"] for t in log] == ["stop"]
    assert log[0]["opened"] == days[1] and log[0]["pnl"] < 0


def _fail_run(days, closes, fail, **flags):
    p = _plan(days, closes, entry={days[0]: True}, **flags)
    return scr.simulate([p], days, {d: "R1" for d in days}, 10_000, {}, fail=fail)[0].trades


def test_fail_any_sells_on_stoch_return_and_none_keeps_position():
    days = _days(5)
    low = {days[3]: True}
    assert _fail_run(days, [100] * 5, "any", oversold=low) == 2
    assert _fail_run(days, [100] * 5, None, oversold=low) == 1


def test_fail_before80_ignores_return_after_overbought():
    days = _days(6)
    high, low = {days[2]: True}, {days[4]: True}
    assert _fail_run(days, [100] * 6, "before80", oversold=low, overbought=high) == 1
    assert _fail_run(days, [100] * 6, "before80", oversold=low) == 2


def test_fail_ma50_needs_close_above_first():
    days = _days(6)
    up, down = {days[2]: True}, {days[4]: True}
    assert _fail_run(days, [100] * 6, "ma50", below50=down) == 1
    assert _fail_run(days, [100] * 6, "ma50", below50=down, above50=up) == 2


def test_size_factor_scales_or_skips_by_score():
    days = _days(4)
    p = _plan(days, [100] * 4, score={days[0]: 2})
    assert scr.size_factor(p, days[0], "prop") == 2 / 3
    assert scr.size_factor(p, days[0], "all3") == 0
    assert scr.size_factor(p, days[0], None) == 1


def test_all3_tilt_skips_entry_below_full_score():
    days = _days(4)
    p = _plan(days, [100] * 4, entry={days[0]: True}, score={days[0]: 2})
    res = scr.simulate([p], days, {d: "R1" for d in days}, 10_000, {}, tilt="all3")[0]
    assert res.trades == 0


def test_frame_up_accepts_positive_values_or_rising_indicators():
    from minerva.strategy.indicators import TMState

    d0, d1 = _days(2)
    prev = TMState(d0, 100, 10, -50, 40, 0, 0)
    up = TMState(d1, 100, 20, -40, 45, 0, 0)
    flat = TMState(d1, 100, 5, -40, 45, 0, 0)
    assert scr.frame_up([prev, up], d1)
    assert not scr.frame_up([prev, flat], d1)
