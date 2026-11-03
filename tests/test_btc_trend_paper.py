import json

from minerva.strategy import btc_trend_paper as tp

DAY = tp.DAY
M5 = tp.M5


def bar(t, o, h, l, c):  # noqa: E741
    return tp.Bar(t, o, h, l, c)


def test_rsi_wilder_seed_and_extremes():
    up = [float(i) for i in range(20)]
    r = tp.rsi(up, 14)
    assert r[13] is None and r[14] == 100.0 and r[-1] == 100.0
    down = [float(-i) for i in range(20)]
    assert tp.rsi(down, 14)[-1] == 0.0


def test_stoch_flat_range_is_50():
    h = lo = c = [10.0] * 6
    assert tp.stoch(h, lo, c)[4:] == [50.0, 50.0]


def test_crossed_and_recent():
    x = [None, 10.0, 25.0, 30.0, 15.0]
    assert tp.crossed(x, 20, True) == [False, False, True, False, False]
    assert tp.crossed(x, 20, False) == [False, False, False, False, True]
    assert tp.recent([True, False, False, False]) == [True, True, True, False]


def test_resample_keeps_full_groups_only():
    bars = [bar(i * M5, 1, 2 + i, 0.5, 1.5) for i in range(7)]
    out = tp.resample(bars, 3)
    assert len(out) == 2
    assert out[0] == tp.Bar(0, 1, 4, 0.5, 1.5)
    assert out[1].t == 3 * M5 and out[1].high == 7


def test_structure_breaks_confirmed_swing():
    highs = [5, 6, 7, 10, 7, 6, 5, 5, 5, 5, 5, 5, 11]
    bars = [bar(i, h - 1, h, h - 2, h - 1) for i, h in enumerate(highs)]
    bars[-1] = bar(12, 10, 11, 9, 10.5)
    st = tp.structure(bars, k=3)
    assert st[11] == 0
    assert st[12] == 1


def test_trend_for_uses_previous_day():
    by_day = {0: 1, DAY: -1}
    assert tp.trend_for(DAY + 5 * M5, by_day) == 1
    assert tp.trend_for(2 * DAY, by_day) == -1
    assert tp.trend_for(5 * DAY, by_day) == 0


def test_rets_three_times_with_liquidation():
    r1, r3 = tp.rets(1, 100.0, 110.0, 0.0)
    assert abs(r1 - (0.10 - 0.001)) < 1e-12
    assert abs(r3 - (0.30 - 0.003)) < 1e-12
    assert tp.rets(-1, 100.0, 90.0, 0.40)[1] == -1.0


def _flat5(n, px=100.0, t0=0):
    return [bar(t0 + i * M5, px, px, px, px) for i in range(n)]


def test_replay_trend_exit_and_stop(monkeypatch):
    bars = _flat5(10)
    bars[5] = bar(5 * M5, 100, 100, 96, 99)  # 3% 손절 97에 닿는다
    monkeypatch.setattr(tp, "signal_grid", lambda b, f: ({2 * M5, 7 * M5}, set()))
    by_day = {-DAY: 1}
    out = tp.replay("T8-15m", bars, by_day, start=0)
    assert [t.why for t in out] == ["stop", ""]
    assert out[0].exit == 97.0 and out[0].entry_t == 2 * M5
    assert out[1].exit_t is None
    # 추세가 사라지면 그 봉 시가에 청산
    bars2 = _flat5(2, t0=DAY - 2 * M5) + _flat5(3, 101.0, t0=DAY)
    monkeypatch.setattr(tp, "signal_grid", lambda b, f: ({DAY - 2 * M5}, set()))
    out = tp.replay("T8-15m", bars2, {-DAY: 1, 0: 0}, start=0)
    assert len(out) == 1 and out[0].why == "trend" and out[0].exit == 101.0
    assert out[0].exit_t == DAY


def test_replay_ignores_signals_against_trend(monkeypatch):
    monkeypatch.setattr(tp, "signal_grid", lambda b, f: ({2 * M5}, {3 * M5}))
    out = tp.replay("T8-30m", _flat5(6), {-DAY: -1}, start=0, long_only=False)
    assert len(out) == 1 and out[0].side == -1
    assert tp.replay("T8-30m", _flat5(6), {-DAY: -1}, start=0) == []


def test_run_notifies_once(tmp_path, monkeypatch):
    now = tp.START + 2 * DAY
    sent = []
    monkeypatch.setattr(tp.notify, "send", lambda text: sent.append(text) or True)
    monkeypatch.setattr(tp, "fetch", lambda frame, since, now_: [])
    monkeypatch.setattr(tp, "trend_by_day", lambda d, w: {})
    monkeypatch.setattr(tp, "update5", lambda root, now_: _flat5(3, t0=tp.START))
    monkeypatch.setattr(
        tp,
        "replay",
        lambda rule, b, by: [tp.Trade(rule, 1, tp.START, 100.0)] if rule == "T8-15m" else [],
    )
    res = tp.run(tmp_path, now)
    assert res["trades"] == 1 and len(res["open"]) == 1
    assert len(sent) == 1 and "진입" in sent[0] and "실주문 아님" in sent[0]
    tp.run(tmp_path, now)
    assert len(sent) == 1
    assert json.loads((tmp_path / "seen.json").read_text()) == [f"in:T8-15m:{tp.START}"]
