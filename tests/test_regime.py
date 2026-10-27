from datetime import date, timedelta

from minerva.strategy.indicators import Bar
from minerva.strategy.regime import htf_direction, vix_normalizing, vix_regimes


def make_bars(closes, start=date(2024, 1, 1)):
    return [
        Bar(day=start + timedelta(days=i), open=c, high=c, low=c, close=c)
        for i, c in enumerate(closes)
    ]


def test_vix_regimes_flags_spike_then_calm():
    closes = [15.0] * 60 + [30.0] * 3 + [14.0] * 10
    bars = make_bars(closes)
    reg = vix_regimes(bars)
    assert reg[bars[59].day] in ("R0", "R1")
    assert reg[bars[60].day] == "R3"
    norm = vix_normalizing(bars)
    assert not norm[bars[60].day]
    assert norm[bars[66].day]


def test_htf_direction_uses_only_completed_bars():
    closes = [10.0 + i for i in range(400)]
    daily = make_bars(closes)
    up = htf_direction(daily, "M", 10)
    assert up[daily[-1].day]
    first_month_days = [b.day for b in daily if b.day.month == 1 and b.day.year == 2024]
    assert all(d not in up for d in first_month_days)  # 마감된 월봉이 아직 없음
