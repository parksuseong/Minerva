from datetime import date, timedelta

from minerva.strategy.indicators import Bar
from minerva.strategy.short import ShortRule, run_short, short_baseline


def make_bars(closes, start=date(2020, 1, 1)):
    return [
        Bar(day=start + timedelta(days=i), open=c, high=c + 1, low=c - 1, close=c)
        for i, c in enumerate(closes)
    ]


def test_short_profits_when_price_falls():
    bars = make_bars([100 - i for i in range(30)])
    flags = [i == 0 for i in range(30)]
    (t,) = run_short(bars, flags, ShortRule(hold=10, stop=0.5))
    assert t.reason == "time" and t.ret > 0


def test_short_stop_on_rally():
    bars = make_bars([100] * 2 + [100 + i * 5 for i in range(10)])
    flags = [i == 0 for i in range(len(bars))]
    (t,) = run_short(bars, flags, ShortRule(hold=10, stop=0.07))
    assert t.reason == "stop" and t.ret < -0.06


def test_short_baseline_negative_in_uptrend():
    assert short_baseline(make_bars([100 + i for i in range(50)]), 5) < 0
