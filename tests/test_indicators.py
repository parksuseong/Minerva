from datetime import date, timedelta

from minerva.strategy.indicators import (
    Bar,
    cci,
    ma_snapshot,
    resample,
    rsi,
    sma,
    stoch_fast_k,
    tm_states,
)


def make_bars(closes, start=date(2026, 1, 5)):
    return [
        Bar(day=start + timedelta(days=i), open=c, high=c + 1, low=c - 1, close=c)
        for i, c in enumerate(closes)
    ]


def test_sma():
    assert sma([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


def test_rsi_extremes():
    assert rsi(list(range(1, 30)))[-1] == 100.0
    assert rsi(list(range(30, 1, -1)))[-1] == 0.0
    assert rsi([1.0] * 10)[-1] is None


def test_stoch_fast_k_position_in_range():
    bars = make_bars([10, 11, 12, 13, 14])
    # 최고가 15, 최저가 9, 종가 14 → (14-9)/(15-9)
    assert abs(stoch_fast_k(bars)[-1] - 5 / 6 * 100) < 1e-9


def test_cci_flat_is_zero():
    assert cci(make_bars([5.0] * 25))[-1] == 0.0


def test_tm_long_level_after_selloff_and_rebound():
    closes = [100 - i * 2 for i in range(30)] + [42 + i * 4 for i in range(4)]
    states = tm_states(make_bars(closes), window=4)
    assert max(s.long_level for s in states[-4:]) >= 2
    assert all(s.short_level == 0 for s in states[-4:])


def test_tm_level_requires_fast_indicator_first():
    states = tm_states(make_bars([100.0] * 40))
    assert all(s.long_level == 0 and s.short_level == 0 for s in states)


def test_resample_weekly_and_monthly():
    bars = make_bars(list(range(1, 41)))  # 2026-01-05(월)부터 40일
    weekly = resample(bars, "W")
    assert weekly[0].open == 1 and weekly[0].close == 7
    assert weekly[0].high == 8 and weekly[0].low == 0
    monthly = resample(bars, "M")
    assert len(monthly) == 2 and monthly[0].day == date(2026, 1, 31)


def test_ma_snapshot_alignment():
    up = ma_snapshot([float(i) for i in range(1, 260)])
    assert up["bull_aligned"] and not up["bear_aligned"]
    assert up["above"][200]
    down = ma_snapshot([float(i) for i in range(260, 1, -1)])
    assert down["bear_aligned"]


def test_parse_yahoo_skips_null_rows():
    from minerva.strategy.report import parse_yahoo, summarize

    n = 300
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1767225600 + 86400 * i for i in range(n)],
                    "indicators": {
                        "quote": [
                            {
                                "open": [None] + [100.0 + i for i in range(1, n)],
                                "high": [101.0 + i for i in range(n)],
                                "low": [99.0 + i for i in range(n)],
                                "close": [100.0 + i for i in range(n)],
                                "volume": [None] * n,
                            }
                        ]
                    },
                }
            ]
        }
    }
    bars = parse_yahoo(payload)
    assert len(bars) == n - 1
    daily = summarize(bars)["D"]
    assert daily["bull_aligned"] and "rsi" in daily["overbought"]


def test_dmi_uptrend_has_pdi_above_mdi_and_rising_adx():
    from minerva.strategy.indicators import dmi

    pdi, mdi, adx = dmi(make_bars([100.0 + i * 2 for i in range(60)]))
    assert pdi[-1] > mdi[-1]
    assert adx[-1] is not None and adx[-1] > 50
    assert adx[26] is None and adx[27] is not None
