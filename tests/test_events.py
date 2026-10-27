from datetime import date, timedelta

from minerva.strategy.events import align, dedupe, forward_returns, max_rise, study
from minerva.strategy.indicators import Bar


def make_bars(closes, start=date(2020, 1, 1)):
    return [
        Bar(day=start + timedelta(days=i), open=c, high=c * 1.01, low=c * 0.99, close=c)
        for i, c in enumerate(closes)
    ]


def test_forward_returns_from_next_open():
    bars = make_bars([100, 100, 110, 121])
    assert forward_returns(bars, 0, [1, 2]) == {1: 0.0, 2: 0.10000000000000009}


def test_dedupe_respects_gap():
    assert dedupe([True, True, False, True, True], gap=3) == [0, 3]


def test_max_rise():
    bars = make_bars([100, 100, 120, 90])
    assert abs(max_rise(bars, 0, 3) - 0.212) < 1e-9


def test_study_edge_positive_for_signal_before_rise():
    closes = [100.0] * 50 + [100 + i for i in range(30)] + [130.0] * 50
    flags = [i == 49 for i in range(len(closes))]
    res = study(make_bars(closes), flags, horizons=(20,))
    assert res["events"] == 1 and res["by_h"][20]["all"]["edge"] > 0


def test_align_forward_fills():
    bars = make_bars([1, 2, 3])
    assert align(bars, {bars[0].day: 5.0, bars[2].day: 7.0}) == [5.0, 5.0, 7.0]
