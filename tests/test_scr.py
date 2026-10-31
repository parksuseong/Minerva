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
