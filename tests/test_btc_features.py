from datetime import date, timedelta

import pytest

from minerva.strategy import btc_features as bf
from minerva.strategy.indicators import Bar


def _bars(rows):
    d = date(2024, 1, 1)
    return [Bar(d + timedelta(days=i), o, h, lo, c, 1) for i, (o, h, lo, c) in enumerate(rows)]


def test_ema_starts_at_first_value_and_follows():
    assert bf.ema([10, 10, 10], 5) == pytest.approx([10, 10, 10])
    assert bf.ema([10, 20], 1) == pytest.approx([10, 20])


def test_rolling_mean_needs_full_window():
    assert bf.rolling_mean([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


def test_rolling_sum_keeps_window():
    assert bf.rolling_sum([1, 2, 3], 2) == [1, 3, 5]


def test_true_range_uses_gap():
    bars = _bars([(10, 11, 9, 10), (15, 16, 14, 15)])
    assert bf.true_range(bars) == [2, 6]


def test_percent_b_centers_at_mean():
    assert bf.percent_b([1, 2, 3], 3)[2] == pytest.approx(1 / (2 * (2 / 3) ** 0.5))
    assert bf.percent_b([5, 5, 5], 3)[2] == 0.0


def test_first_touch_long_and_short():
    bars = _bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 102, 100, 101)])
    assert bf.first_touch(bars, 0, 1) == (True, 2)
    assert bf.first_touch(bars, 0, -1) == (False, 2)


def test_first_touch_ambiguous_bar_is_none():
    bars = _bars([(100, 100, 100, 100), (100, 102, 98, 100)])
    assert bf.first_touch(bars, 0, 1)[0] is None


def test_quintile_bucket():
    cuts = bf.quintiles(list(range(10)))
    assert cuts == [2, 4, 6, 8]
    assert [bf.bucket(v, cuts) for v in (0, 3, 9)] == [0, 1, 4]


def test_summarize_drops_overlapping_signals():
    events = [(1, 0, True, 5), (3, 0, False, 5), (6, 0, False, 8), (2, 1, True, 3)]
    got = bf.summarize(events, 2)
    assert got[0] == {"bin": 0, "n": 2, "rate": 0.5}
    assert got[1]["n"] == 1
