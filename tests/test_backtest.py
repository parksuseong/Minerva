from datetime import date, timedelta

from minerva.strategy.backtest import Rule, baseline, evaluate, run, signals
from minerva.strategy.indicators import Bar


def make_bars(closes, start=date(2020, 1, 1)):
    return [
        Bar(day=start + timedelta(days=i), open=c, high=c + 1, low=c - 1, close=c)
        for i, c in enumerate(closes)
    ]


def v_shape(depth=30, rise=20):
    return [100 - i * 2 for i in range(depth)] + [100 - depth * 2 + i * 3 for i in range(rise)]


def test_signal_fires_once_per_event():
    bars = make_bars(v_shape())
    sig = signals(bars, Rule(level=1, window=3))
    assert sum(sig) == 1


def test_entry_is_next_bar_open_and_time_exit():
    bars = make_bars(v_shape())
    rule = Rule(level=1, hold=5, stop=None)
    sig_idx = signals(bars, rule).index(True)
    (t,) = run(bars, rule)
    assert t.entry_day == bars[sig_idx + 1].day
    assert t.entry == bars[sig_idx + 1].open
    assert t.bars == 5 and t.reason == "time" and t.ret > 0


def test_stop_exits_at_stop_price():
    closes = v_shape(rise=3) + [30.0] * 5  # 반등 직후 폭락
    bars = make_bars(closes)
    (t,) = run(bars, Rule(level=1, hold=20, stop=0.10))
    assert t.reason == "stop"
    assert abs(t.ret - (-0.10)) < 1e-9 or t.ret < -0.10  # 갭하락이면 시가 체결


def test_above_ma_filter_blocks_downtrend_signal():
    bars = make_bars(v_shape())
    assert not any(signals(bars, Rule(level=1, above_ma=20)))


def test_baseline_flat_market_is_zero():
    assert baseline(make_bars([10.0] * 30), 5) == 0.0


def test_evaluate_reports_split_and_small_sample_warning():
    bars = make_bars(v_shape() * 3)
    res = evaluate(bars, Rule(level=1, hold=5, stop=None))
    assert res["all"]["trades"] >= 1
    assert "in_sample" in res and "out_of_sample" in res
    assert "warning" in res


def test_trend_exit_holds_longer_than_stoch_exit():
    bars = make_bars(v_shape(rise=40))
    stoch_t = run(bars, Rule(level=1, hold=60, stop=None))
    trend_t = run(bars, Rule(level=1, hold=60, stop=None, exit="trend", trail_ma=5))
    assert trend_t[0].bars >= stoch_t[0].bars


def test_vix_and_align_filters_use_context():
    from minerva.strategy.backtest import Context

    bars = make_bars(v_shape())
    rule = Rule(level=1)
    day = bars[signals(bars, rule).index(True)].day
    calm = Context(regime={day: "R0"})
    panic = Context(regime={day: "R3"})
    assert any(signals(bars, Rule(level=1, vix="calm"), calm))
    assert not any(signals(bars, Rule(level=1, vix="calm"), panic))
    assert not any(signals(bars, Rule(level=1, align=True), Context()))


def test_benjamini_hochberg():
    from minerva.strategy.backtest import benjamini_hochberg

    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5, "e": None}
    assert benjamini_hochberg(p, 0.10) == {"a", "b", "c"}  # 0.04 <= 0.1*3/4
    assert benjamini_hochberg(p, 0.01) == {"a"}
    assert benjamini_hochberg({}, 0.1) == set()


def test_random_timing_pvalue_is_deterministic_and_bounded():
    import math
    from datetime import date, timedelta

    from minerva.strategy.backtest import Rule, random_timing_pvalue
    from minerva.strategy.indicators import Bar

    bars = []
    for i in range(400):
        px = 100 + 10 * math.sin(i / 9) + i * 0.05
        bars.append(Bar(date(2020, 1, 1) + timedelta(days=i), px, px + 1, px - 1, px, 0))
    rule = Rule(level=1, hold=10, stop=0.1)
    p1 = random_timing_pvalue(bars, rule, n_perm=50, seed=1)
    assert p1 == random_timing_pvalue(bars, rule, n_perm=50, seed=1)
    assert p1 is None or 1 / 51 <= p1 <= 1
