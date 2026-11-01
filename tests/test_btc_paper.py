import importlib.util
import sys
from datetime import date
from pathlib import Path

from minerva.strategy import btc_paper as bp
from minerva.strategy.indicators import Bar

RAW = Path(__file__).parent.parent / "docs/backtests/raw/btc_swing_label.py"


def bar(t, o, h, l, c):  # noqa: E741
    return bp.PBar(t, o, h, l, c)


def flat(n, px=100.0):
    return [bar(i, px, px, px, px) for i in range(n)]


def test_breakouts_first_new_high_only():
    closes = [10.0] * 5 + [11.0, 12.0, 9.0, 13.0]
    got = bp.breakouts(closes, look=3)
    assert got == [False] * 5 + [True, False, False, True]


def test_net_includes_both_costs():
    assert abs(bp.net(1, 100, 100, 0.0005) + 0.001) < 1e-6
    assert abs(bp.net(-1, 100, 99, 0.0) - 0.01) < 1e-9


def test_stop_exit_fills_line_and_gap_at_open():
    t = bp.Trade("S4", 1, 0, 1, 100.0)
    assert not bp._stop_exit(t, bar(2, 100, 101, 99.5, 100))
    assert bp._stop_exit(t, bar(3, 100, 100, 98, 99))
    assert t.exit == 99.0 and t.why == "stop"
    g = bp.Trade("S4", -1, 0, 1, 100.0)
    assert bp._stop_exit(g, bar(2, 103, 104, 102, 103))
    assert g.exit == 103


def test_replay_l1_enters_next_open_and_exits_on_time(monkeypatch):
    monkeypatch.setattr(bp, "LOOK", 3)
    monkeypatch.setattr(bp, "HOLD", 2)
    monkeypatch.setattr(bp, "breakouts", lambda c: [i == 3 for i in range(len(c))])
    bars = [bar(i, 100, 100.2, 99.8, 100) for i in range(10)]
    got = bp.replay_l1(bars, start=0)
    assert len(got) == 1
    assert (got[0].entry_t, got[0].exit_t, got[0].why) == (4, 6, "time")


def test_replay_l1_ignores_signals_before_start(monkeypatch):
    monkeypatch.setattr(bp, "breakouts", lambda c: [i == 3 for i in range(len(c))])
    bars = [bar(i, 100, 100, 100, 100) for i in range(10)]
    assert bp.replay_l1(bars, start=5) == []


def test_replay_s4_flips_on_opposite(monkeypatch):
    lo = [i == 2 for i in range(8)]
    hi = [i == 4 for i in range(8)]
    monkeypatch.setattr(bp, "s4_pulses", lambda b: (lo, hi))
    bars = [bar(i, 100, 100, 100, 100) for i in range(8)]
    got = bp.replay_s4(bars, start=0)
    assert [t.side for t in got] == [1, -1]
    assert got[0].why == "flip" and got[0].exit_t == 5 and got[1].entry_t == 5


def test_s4_matches_backtest_code():
    spec = importlib.util.spec_from_file_location("btc_swing_label", RAW)
    sl = importlib.util.module_from_spec(spec)
    sys.modules["btc_swing_label"] = sl
    spec.loader.exec_module(sl)
    px, rows = 100.0, []
    for i in range(1500):
        px *= 1 + 0.02 * ((i * 7919) % 11 - 5) / 5 * (1 if (i // 40) % 2 else -1) * 0.3
        px *= 1 + 0.002 * (1 if (i // 90) % 2 else -1)
        rows.append((i, px * 0.999, px * 1.004, px * 0.996, px))
    pb = [bp.PBar(*r) for r in rows]
    raw = [Bar(date(2026, 1, 1), r[1], r[2], r[3], r[4], 1.0) for r in rows]
    lo, hi = sl.rules(raw, sl.confirmed(raw))["S4"]
    a, b = bp.s4_pulses(pb)
    assert [bool(x) for x in lo] == a and [bool(x) for x in hi] == b
    assert sum(a) + sum(b) > 0
