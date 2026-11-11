import statistics
from datetime import date, timedelta

import pytest

from minerva.strategy import soxl_classic as sc
from minerva.strategy.indicators import Bar


def _bars(closes, vol=100.0):
    d0 = date(2020, 1, 1)
    day = lambda i: d0 + timedelta(days=i)  # noqa: E731
    return [Bar(day(i), c, c * 1.01, c * 0.99, c, vol) for i, c in enumerate(closes)]


def test_bollinger_matches_population_stdev():
    closes = [float(100 + (i % 7)) for i in range(40)]
    mid, up, lo = sc.bollinger(closes)
    sd = statistics.pstdev(closes[-20:])
    assert mid[-1] == pytest.approx(sum(closes[-20:]) / 20)
    assert up[-1] == pytest.approx(mid[-1] + 2 * sd) and lo[-1] == pytest.approx(mid[-1] - 2 * sd)
    assert up[18] is None


def test_profile_puts_heavy_volume_in_top_bins():
    bars = _bars([100.0] * 50 + [120.0] * 10)
    heavy = [Bar(b.day, b.open, b.high, b.low, b.close, 1000.0) for b in bars[-10:]]
    top = sc.profile(bars[:-10] + heavy)
    assert len(top) == 2
    assert any(t["lo"] <= 120 <= t["hi"] for t in top)
    assert sum(t["pct"] for t in top) == pytest.approx(100, abs=0.5)


def test_build_needs_history():
    with pytest.raises(ValueError):
        sc.build(_bars([100.0] * 100), 50.0)


def test_build_long_setup_after_crash_and_rebound():
    # 길게 오르다 급락(볼밴 하단 이탈) 뒤 이틀 반등 → 롱 3단 돌파 + 하단 터치
    closes = [100 + i * 0.1 for i in range(280)] + [120, 110, 100, 92, 88, 86, 92, 99]
    out = sc.build(_bars(closes), 40.0)
    assert out["long_setup"]["bb_touch"] is True
    assert out["ind"]["long_level"] == out["long_setup"]["scr_level"]
    # 환산: SOXX 종가에서 +1%면 SOXL은 +3%
    line = out["lines"][0]
    want = 40.0 * (1 + 3 * (line["soxx"] / out["close"] - 1))
    assert line["soxl"] == pytest.approx(want, abs=0.01)


def test_build_short_setup_turn_near_ma20():
    # 하락 추세 뒤 20일선까지 되오름(스토캐 과매수) → 다음 봉에서 꺾여 마감
    closes = [200 - i * 0.2 for i in range(270)] + [140, 141, 143, 145, 147, 149, 151, 150, 146]
    out = sc.build(_bars(closes), 50.0)
    s = out["short_setup"]
    assert s["near_ma20"] is True and s["stoch_turn"] is True and s["ok"] is True
